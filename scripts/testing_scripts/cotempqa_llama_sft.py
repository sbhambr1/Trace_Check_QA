import argparse
import os
import torch
import torch.distributed as dist
from torch.distributed import destroy_process_group
from datasets import load_from_disk, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    logging,
)
from peft import LoraConfig, get_peft_model, PeftModel
from trl import SFTTrainer
from huggingface_hub import login
import wandb # Optional, for tracking

from accelerate import PartialState
from accelerate import Accelerator

# Get the current GPU index for this process
local_rank = PartialState().process_index
    
torch.cuda.empty_cache()

accelerator = Accelerator()

# def find_free_port():
#     with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
#         s.bind(("", 0))  # Bind to any free port
#         return s.getsockname()[1]  # Return the port number

# # # # Set environment variables for single-node multi-GPU training
# os.environ["MASTER_ADDR"] = "127.0.0.1"  # Master node address

# free_port = find_free_port()
# os.environ["MASTER_PORT"] = str(free_port)
# print(f"Using free port: {free_port}")

# init_process_group(
#     backend="nccl",  # Use NCCL for GPU communication
#     world_size=torch.cuda.device_count(),  # Number of GPUs available
#     rank=0,  # Rank of the current process (0 for single-node)
# )

# Suppress verbose logging
# logging.set_verbosity_error()

LLAMA_3_CHAT_TEMPLATE = (
    "{% for message in messages %}"
        "{% if message['role'] == 'system' %}"
            "{{ message['content'] }}"
        "{% elif message['role'] == 'user' %}"
            "{{ '\n\nHuman: ' + message['content'] +  eos_token }}"
        "{% elif message['role'] == 'assistant' %}"
            "{{ '\n\nAssistant: '  + message['content'] +  eos_token  }}"
        "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '\n\nAssistant: ' }}"
    "{% endif %}"
)

def train_sft(
    dataset_path: str,
    base_model_id: str = "meta-llama/Meta-Llama-3.1-8B", # Specify the base Llama 3.1 8B model [2]
    output_dir: str = "./llama3-8b-sft-adapter",
    hf_token: str = None, # Optional: For gated models or pushing to Hub
    wandb_token: str = None, # Optional: For logging to Weights & Biases
    use_qlora: bool = True,
    lora_r: int = 16, # LoRA rank [2]
    lora_alpha: int = 32, # LoRA alpha [2]
    lora_dropout: float = 0.05,
    # Common target modules for Llama models [2]
    lora_target_modules: list = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    batch_size: int = 1, # Keep low for large models [2][3]
    gradient_accumulation_steps: int = 4, # Effective batch size = batch_size * gradient_accumulation_steps [2][3]
    learning_rate: float = 2e-4, # Common learning rate for LoRA [2][3]
    num_train_epochs: int = 1, # Number of training epochs [3]
    max_seq_length: int = 1024, # Adjust based on VRAM and dataset needs [2][3]
    logging_steps: int = 10, # Log metrics every N steps [2][3]
    save_steps: int = 50, # Save checkpoint every N steps
    max_steps: int = -1, # Set to positive value to override epochs
    use_flash_attention_2: bool = True, # Use Flash Attention 2 if available [2]
    gradient_checkpointing: bool = True # Use gradient checkpointing to save memory [2]
):
    """
    Performs Supervised Fine-Tuning (SFT) on a Llama 3.1 model using QLoRA.

    Args:
        dataset_path (str): Path to the pre-processed dataset directory saved by the previous script.
        base_model_id (str): Hugging Face model ID for the base Llama 3.1 model.
        output_dir (str): Directory to save the trained LoRA adapter and checkpoints.
        hf_token (str, optional): Hugging Face API token.
        wandb_token (str, optional): Weights & Biases API token for logging.
        use_qlora (bool): Whether to use QLoRA (4-bit quantization).
        lora_r (int): LoRA rank.
        lora_alpha (int): LoRA alpha scaling factor.
        lora_dropout (float): LoRA dropout rate.
        lora_target_modules (list): List of model module names to apply LoRA to.
        batch_size (int): Training batch size per device.
        gradient_accumulation_steps (int): Number of steps to accumulate gradients over.
        learning_rate (float): Optimizer learning rate.
        num_train_epochs (int): Number of training epochs.
        max_seq_length (int): Maximum sequence length for truncation/padding.
        logging_steps (int): Frequency of logging training metrics.
        save_steps (int): Frequency of saving model checkpoints.
        max_steps (int): Total number of training steps to perform (overrides epochs if > 0).
        use_flash_attention_2 (bool): Whether to enable Flash Attention 2.
        gradient_checkpointing (bool): Whether to use gradient checkpointing.
    """

    # --- Login and Initialization ---
    if hf_token:
        print("Logging into Hugging Face Hub...")
        login(token=hf_token)

    if wandb_token:
        print("Logging into Weights & Biases...")
        try:
            wandb.login(key=wandb_token)
            run = wandb.init(
                project=f"sft-{base_model_id.split('/')[-1]}",
                job_type="training",
                anonymous="allow"
            )
            report_to = "wandb"
        except Exception as e:
            print(f"Wandb login failed: {e}. Training without wandb logging.")
            report_to = "none"
    else:
        report_to = "none" # or "tensorboard"

    # --- Load Dataset ---
    # TODO: testing with sample dataset
    system_message = """You are Llama, an AI assistant created by Philipp to be helpful and honest. Your knowledge spans a wide range of topics, allowing you to engage in substantive conversations and provide analysis on complex subjects."""
    def create_conversation(sample):
        if sample["messages"][0]["role"] == "system":
            return sample
        else:
            sample["messages"] = [{"role": "system", "content": system_message}] + sample["messages"]
            return sample
 
    # Load dataset from the hub
    dataset = load_dataset("HuggingFaceH4/no_robots")
    
    # Add system message to each conversation
    columns_to_remove = list(dataset["train"].features)
    columns_to_remove.remove("messages")
    dataset = dataset.map(create_conversation, remove_columns=columns_to_remove,batched=False)
    
    # Filter out conversations which are corrupted with wrong turns, keep which have even number of turns after adding system message
    dataset["train"] = dataset["train"].filter(lambda x: len(x["messages"][1:]) % 2 == 0)
    dataset["test"] = dataset["test"].filter(lambda x: len(x["messages"][1:]) % 2 == 0)
    
    # save datasets to disk
    dataset["train"].to_json("train_dataset.json", orient="records", force_ascii=False)
    dataset["test"].to_json("test_dataset.json", orient="records", force_ascii=False)
    
    train_dataset = load_dataset(
        "json",
        data_files=os.path.join(".", "train_dataset.json"),
        split="train",
    )
    test_dataset = load_dataset(
        "json",
        data_files=os.path.join(".", "test_dataset.json"),
        split="train",
    )
    
    
    # print(f"Loading dataset from {dataset_path}...")
    # try:
    #     dataset = load_from_disk(dataset_path)
    #     # Optionally split if you don't have a pre-defined split
    #     if 'train' not in dataset:
    #         dataset = dataset.train_test_split(test_size=0.1)
    #     print("Dataset loaded successfully.")
    #     # Ensure the dataset has the expected 'text' field [2][3]
    #     if "text" not in dataset["train"].column_names or "text" not in dataset["test"].column_names:
    #          raise ValueError("Dataset must contain a 'text' column formatted for SFT.")
    # except FileNotFoundError:
    #     print(f"Error: Dataset directory not found at {dataset_path}")
    #     return
    # except Exception as e:
    #     print(f"Error loading dataset: {e}")
    #     return

    # --- Load Tokenizer ---
    print(f"Loading tokenizer for {base_model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    # Llama 3 doesn't have a default pad token, set it to eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # Ensure padding is on the right for causal LMs
    tokenizer.chat_template = LLAMA_3_CHAT_TEMPLATE # Set the chat template for Llama 3.1 [2]
    
    # template dataset
    def template_dataset(examples):
        return{"text":  tokenizer.apply_chat_template(examples["messages"], tokenize=False)}
    
    train_dataset = train_dataset.map(template_dataset, remove_columns=["messages"])
    test_dataset = test_dataset.map(template_dataset, remove_columns=["messages"])
    
    
    # --- Tokenize Dataset ---
    # def tokenize_function(examples):
    #     return tokenizer(
    #         examples["text"],
    #         padding="max_length",  # Pad to max length (or use True for dynamic padding)
    #         truncation=True,       # Truncate to the model's max input length
    #         max_length=512,        # Set max length (adjust based on your model)
    #         return_tensors="pt"    # Return PyTorch tensors
    #     )
    # # Apply tokenization to the dataset
    # tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # # Remove the original "text" column if it's no longer needed
    # tokenized_dataset = tokenized_dataset.remove_columns(["text"])

    # --- Configure Quantization (QLoRA) ---
    if use_qlora:
        print("Setting up QLoRA (4-bit quantization)...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4", # Recommended quantization type
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16, # Use bfloat16 for computation if available [3]
            bnb_4bit_use_double_quant=True, # Use double quantization for more memory savings
        )
        quantization_config = bnb_config
    else:
        print("Using standard float16/bfloat16 precision (no quantization).")
        quantization_config = None
        # Check GPU capability for bfloat16
        if torch.cuda.is_bf16_supported():
             torch_dtype = torch.bfloat16
        else:
             torch_dtype = torch.float16

    # --- Load Base Model ---
    print(f"Loading base model: {base_model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=quantization_config, # Apply QLoRA config here
        # device_map="auto",  # oprint(next(model.parameters()).device)  # Should print "cuda:<rank>"r f"cuda:{accelerator.local_process_index}", # Use the specified device: {"": device} || Automatically distribute model across available GPUs/CPU: "auto"
        # device_map={f"cuda:{i}" for i in range(2)},
        # device_map=device_map,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if use_flash_attention_2 else "eager", # Use Flash Attention 2 if possible [2][3]
    )
    # `setup_chat_format` is usually not needed when using SFTTrainer with pre-formatted text field
    # model, tokenizer = setup_chat_format(model, tokenizer)

    # model = accelerator.prepare(model)

    # --- Configure LoRA ---
    print("Configuring LoRA adapter...")
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    # `get_peft_model` applies the LoRA layers. For QLoRA, the base model is already loaded in 4-bit.
    # If not using QLoRA, prepare_model_for_kbit_training might be needed if loading in 8-bit.
    # model = prepare_model_for_kbit_training(model) # Needed for 8-bit, not typically for 4-bit QLoRA via BitsAndBytesConfig
    # model = get_peft_model(model, peft_config) # SFTTrainer can handle applying PEFT config

    # --- Configure Training Arguments ---
    print("Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        logging_steps=logging_steps,
        num_train_epochs=num_train_epochs,
        max_steps=max_steps, # If > 0, overrides num_train_epochs
        report_to=report_to,
        save_steps=save_steps,
        save_total_limit=2, # Keep only the last 2 checkpoints
        bf16=True if torch.cuda.is_bf16_supported() else False, # SFTTrainer handles mixed precision with QLoRA
        fp16=False if torch.cuda.is_bf16_supported() else True, # Use fp16 if bf16 not available
        optim="paged_adamw_8bit" if use_qlora else "adamw_torch", # Paged optimizer recommended for QLoRA [3]
        lr_scheduler_type="cosine", # A common scheduler
        warmup_ratio=0.03, # Warmup ratio
        # gradient_checkpointing=gradient_checkpointing, # Enable gradient checkpointing [2]
        # group_by_length=True, # Group sequences of similar length to save computation (requires careful dataset handling)
        eval_strategy="steps", # Enable evaluation during training
        eval_steps=0.2, # Evaluate every 20% of the steps within an epoch [3] - Requires eval dataset
        remove_unused_columns=False,
        fsdp="full_shard auto_wrap offload",
        fsdp_config={
            "backward_prefetch": "backward_pre",
            "forward_prefetch": "false",
            "use_orig_params": "false"}
    )
    
    # rank = get_rank()
    # world_size = get_world_size()
    
    # if not dist.is_initialized():
    #     dist.init_process_group(backend="nccl")
    #     print("Process group initialized")

    # --- Initialize SFTTrainer ---
    print("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        # model=torch.nn.parallel.DistributedDataParallel(model),
        model=model,
        args=training_args,
        train_dataset=train_dataset, # Pass the entire dataset, SFTTrainer handles splitting if needed or uses 'train' split by default
        eval_dataset=test_dataset, # Uncomment if you have a 'test' split in your dataset [3]
        peft_config=peft_config, # Pass LoRA config here [3]
    )
    
    # if training_args.gradient_checkpointing:
    #     print("Enabling gradient checkpointing...")
    #     trainer.model.gradient_checkpointing_enable()
    
    # Ensure the dataset tensors are on the same device (if applicable)
    # If using a DataLoader, ensure the `pin_memory` and `non_blocking` options are set correctly
    # def move_to_device(batch, device):
    #     if isinstance(batch, dict):
    #         return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
    #     elif isinstance(batch, list):
    #         return [move_to_device(item, device) for item in batch]
    #     elif torch.is_tensor(batch):
    #         return batch.to(device)
    #     return batch
    
    # # # Example: Move a batch to the correct device during training
    # for batch in trainer.get_train_dataloader():
    #     batch = move_to_device(batch, device)

    # --- Start Training ---
    print("Starting training...")
    train_result = trainer.train()

    # --- Save Metrics and Final Adapter ---
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    print("Saving final LoRA adapter...")
    # SFTTrainer automatically saves the adapter during training based on save_steps
    # You can also save it explicitly after training finishes
    final_adapter_path = os.path.join(output_dir, "final_adapter")
    trainer.model.save_pretrained(final_adapter_path) # Saves only the adapter weights [3]
    tokenizer.save_pretrained(final_adapter_path) # Save tokenizer alongside adapter
    print(f"Training complete. Final LoRA adapter saved to {final_adapter_path}")
    
    destroy_process_group() # Clean up distributed process group

    # --- Clean up Wandb ---
    if report_to == "wandb":
        wandb.finish()

    # --- Optional: Merge Adapter and Save Full Model ---
    # This requires significant memory as it loads the base model in full precision
    # Consider running this as a separate step if memory is constrained
    # print("Merging adapter with base model...")
    # base_model_reload = AutoModelForCausalLM.from_pretrained(
    #     base_model_id,
    #     torch_dtype=torch.float16, # Load in float16 or bfloat16
    #     device_map="auto",
    #     trust_remote_code=True,
    # )
    # merged_model = PeftModel.from_pretrained(base_model_reload, final_adapter_path)
    # merged_model = merged_model.merge_and_unload()
    # print("Saving merged model...")
    # merged_model_path = os.path.join(output_dir, "final_merged_model")
    # merged_model.save_pretrained(merged_model_path)
    # tokenizer.save_pretrained(merged_model_path)
    # print(f"Merged model saved to {merged_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a Llama 3.1 model using SFTTrainer and QLoRA.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the saved dataset directory.")
    parser.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3.1-8B", help="Base model ID from Hugging Face Hub.")
    parser.add_argument("--output_dir", type=str, default="./llama3-8b-sft-adapter", help="Directory to save the adapter.")
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face Hub token (optional).")
    parser.add_argument("--wandb_token", type=str, default=None, help="Weights & Biases token (optional).")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=1, help="Per-device training batch size.")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps.")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate.")
    parser.add_argument("--max_seq_len", type=int, default=1024, help="Maximum sequence length.")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank.")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha.")
    parser.add_argument("--disable_qlora", action='store_true', help="Disable QLoRA (use fp16/bf16 instead).")
    parser.add_argument("--disable_flash_attention", action='store_true', help="Disable Flash Attention 2.")

    args = parser.parse_args()

    train_sft(
        dataset_path=args.dataset_path,
        base_model_id=args.model_id,
        output_dir=args.output_dir,
        hf_token=args.hf_token,
        wandb_token=args.wandb_token,
        num_train_epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_seq_length=args.max_seq_len,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        use_qlora=not args.disable_qlora,
        use_flash_attention_2=not args.disable_flash_attention,
    )
