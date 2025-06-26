import argparse
import os
import torch
import ast
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
    
torch.cuda.empty_cache()

def train_sft(
    expt_name: str = "llama3.1-8b-sft-openthoughts",
    base_model_id: str = "meta-llama/Meta-Llama-3.1-8B", # Specify the base Llama 3.1 8B model
    output_dir: str = "./llama3-8b-sft-adapter-openthoughts",
    dataset_type: str = "r1_trace", # Options: "r1_trace", "explanation", "summary"
    hf_token: str = None, # Optional: For gated models or pushing to Hub
    wandb_token: str = None, # Optional: For logging to Weights & Biases
    use_qlora: bool = False, # Disabled QLoRA as requested
    lora_r: int = 32, # LoRA rank
    lora_alpha: int = 64, # LoRA alpha
    lora_dropout: float = 0.05,
    # Common target modules for Llama models
    lora_target_modules: list = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    batch_size: int = 4, # Keep low for large models
    gradient_accumulation_steps: int = 4, # Effective batch size = batch_size * gradient_accumulation_steps
    learning_rate: float = 1e-5, # Common learning rate for LoRA
    num_train_epochs: int = 3, # Number of training epochs
    max_seq_length: int = 1024, # Adjust based on VRAM and dataset needs
    logging_steps: int = 10, # Log metrics every N steps
    save_steps: int = 50, # Save checkpoint every N steps
    max_steps: int = -1, # Set to positive value to override epochs
    use_flash_attention_2: bool = True, # Use Flash Attention 2 if available
    gradient_checkpointing: bool = True, # Use gradient checkpointing to save memory
    eval_split_ratio: float = 0.1 # Ratio to split train data for evaluation
):
    """
    Performs Supervised Fine-Tuning (SFT) on a Llama 3.1 model using LoRA on OpenThoughts datasets.

    Args:
        expt_name (str): Experiment name for saving logs.
        base_model_id (str): Hugging Face model ID for the base Llama 3.1 model.
        output_dir (str): Directory to save the trained LoRA adapter and checkpoints.
        dataset_type (str): Type of OpenThoughts dataset to use ("r1_trace", "explanation", "summary").
        hf_token (str, optional): Hugging Face API token.
        wandb_token (str, optional): Weights & Biases API token for logging.
        use_qlora (bool): Whether to use QLoRA (4-bit quantization) - disabled by default.
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
        eval_split_ratio (float): Ratio to split training data for evaluation.
    """

    # --- Login and Initialization ---
    if hf_token:
        print("Logging into Hugging Face Hub...")
        login(token=hf_token)

    if wandb_token:
        print("Logging into Weights & Biases...")
        try:
            wandb.login(key=wandb_token)
            # Initialize wandb run
            run = wandb.init(
                entity="wordle",
                project="OpenThoughts-SFT",
                name=f"{expt_name}-{dataset_type}",
            )
            report_to = "wandb"
        except Exception as e:
            print(f"Wandb login failed: {e}. Training without wandb logging.")
            report_to = "none"
    else:
        report_to = "none" # or "tensorboard"

    # --- Load Dataset ---
    if 'llama' in base_model_id.lower():
        system_message = """You are Llama, an AI assistant created to be helpful and honest. Your knowledge spans a wide range of topics, allowing you to engage in substantive conversations and provide analysis on complex subjects."""
    elif 'qwen' in base_model_id.lower():
        system_message = """You are Qwen, an AI assistant created to be helpful and honest. Your knowledge spans a wide range of topics, allowing you to engage in substantive conversations and provide analysis on complex subjects."""
    elif 'gemma' in base_model_id.lower():
        system_message = """You are Gemma, an AI assistant created to be helpful and honest. Your knowledge spans a wide range of topics, allowing you to engage in substantive conversations and provide analysis on complex subjects."""
    elif 'mistral' in base_model_id.lower():
        system_message = """You are Mistral, an AI assistant created to be helpful and honest. Your knowledge spans a wide range of topics, allowing you to engage in substantive conversations and provide analysis on complex subjects."""
    
    def parse_messages_column(sample):
        if isinstance(sample["messages"], str):
            sample["messages"] = ast.literal_eval(sample["messages"])  # Convert string to list
        return sample
    
    def create_conversation(sample):
        if sample["messages"][0]["role"] == "system":
            return sample
        else:
            sample["messages"] = [{"role": "system", "content": system_message}] + sample["messages"]
            return sample

    # Map dataset types to huggingface dataset names
    dataset_mapping = {
        "r1_trace": "sbhambr1/openthoughts_sft_r1_traces",
        "explanation": "sbhambr1/openthoughts_sft_explanations",
        "summary": "sbhambr1/openthoughts_sft_summaries",
        "no_reasoning": "sbhambr1/openthoughts_sft_no_reasoning", 
        "perturbed_reasoning": "sbhambr1/openthoughts_sft_perturbed_reasoning"
    }
    if dataset_type not in dataset_mapping:
        raise ValueError(f"Invalid dataset_type: {dataset_type}. Must be one of {list(dataset_mapping.keys())}")
    dataset_name = dataset_mapping[dataset_type]
    print(f"Loading OpenThoughts {dataset_type} dataset from HuggingFace Hub...")
    # Only load from HuggingFace Hub, raise error if not accessible
    try:
        dataset = load_dataset(dataset_name)
    except Exception as e:
        raise RuntimeError(f"Error loading dataset '{dataset_name}' from HuggingFace Hub: {e}")
    
    # Parse the "messages" column as a list
    dataset = dataset.map(parse_messages_column, batched=False)
    
    # Add system message to each conversation
    columns_to_remove = list(dataset["train"].features)
    columns_to_remove.remove("messages")
    dataset = dataset.map(create_conversation, remove_columns=columns_to_remove, batched=False)
    
    # Split training data for evaluation since no test.csv available
    train_test_split = dataset["train"].train_test_split(test_size=eval_split_ratio, seed=42)
    train_dataset_split = train_test_split["train"]
    eval_dataset_split = train_test_split["test"]
    
    # Save datasets to disk
    train_dataset_split.to_json("train_dataset.json", orient="records", force_ascii=False)
    eval_dataset_split.to_json("eval_dataset.json", orient="records", force_ascii=False)
    
    train_dataset = load_dataset(
        "json",
        data_files=os.path.join(".", "train_dataset.json"),
        split="train",
    )
    eval_dataset = load_dataset(
        "json",
        data_files=os.path.join(".", "eval_dataset.json"),
        split="train",
    )
    
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Evaluation dataset size: {len(eval_dataset)}")
    # --- Load Tokenizer ---
    print(f"Loading tokenizer for {base_model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    # Llama 3 doesn't have a default pad token, set it to eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # Use "right" for Llama models
    
    # Template dataset
    def template_dataset(examples):
        return {"text": tokenizer.apply_chat_template(examples["messages"], tokenize=False)}
    
    train_dataset = train_dataset.map(template_dataset, remove_columns=["messages"])
    eval_dataset = eval_dataset.map(template_dataset, remove_columns=["messages"])

    # --- Configure Quantization (disabled) ---
    if use_qlora:
        print("Setting up QLoRA (4-bit quantization)...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        quantization_config = bnb_config
        torch_dtype = None  # Let quantization handle dtype
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
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if use_flash_attention_2 else "eager",
    )

    model.config.use_cache = False
    
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

    # --- Configure Training Arguments ---
    print("Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir=f"models/{output_dir}-{dataset_type}",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        logging_steps=logging_steps,
        num_train_epochs=num_train_epochs,
        max_steps=max_steps,
        report_to=report_to,
        save_steps=save_steps,
        save_total_limit=2,
        bf16=True if torch.cuda.is_bf16_supported() else False,
        fp16=False if torch.cuda.is_bf16_supported() else True,
        optim="adamw_torch", # Standard optimizer since no QLoRA
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        gradient_checkpointing=gradient_checkpointing,
        eval_strategy="steps",
        eval_steps=0.2, # Evaluate every 20% of the steps
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    # --- Initialize SFTTrainer ---
    print("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        max_seq_length=max_seq_length,
    )
    
    if training_args.gradient_checkpointing:
        print("Enabling gradient checkpointing...")
        trainer.model.gradient_checkpointing_enable()

    # --- Start Training ---
    print(f"Starting training for OpenThoughts {dataset_type} dataset...")
    train_result = trainer.train()

    # --- Save Metrics and Final Adapter ---
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    print("Saving final LoRA adapter...")
    final_adapter_path = os.path.join(f"models/{output_dir}-{dataset_type}", "final_adapter")
    trainer.model.save_pretrained(final_adapter_path)
    tokenizer.save_pretrained(final_adapter_path)
    print(f"Training complete. Final LoRA adapter saved to {final_adapter_path}")

    # --- Clean up Wandb ---
    if report_to == "wandb":
        wandb.finish()

    # Clean up temporary files
    if os.path.exists("train_dataset.json"):
        os.remove("train_dataset.json")
    if os.path.exists("eval_dataset.json"):
        os.remove("eval_dataset.json")

    return final_adapter_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a Llama 3.1 model using SFTTrainer and LoRA on OpenThoughts datasets.")
    parser.add_argument("--expt_name", type=str, default="llama3.1-8b-sft-openthoughts", help="Experiment name for saving logs.")
    parser.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3.1-8B", help="Base model ID from Hugging Face Hub.")
    parser.add_argument("--output_dir", type=str, default="./llama3-8b-sft-adapter-openthoughts", help="Directory to save the adapter.")
    parser.add_argument("--dataset_type", type=str, choices=["r1_trace", "explanation", "summary", "no_reasoning", "perturbed_reasoning"], default="r1_trace", help="Type of OpenThoughts dataset to use.")
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face Hub token (optional).")
    parser.add_argument("--wandb_token", type=str, default=None, help="Weights & Biases token (optional).")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=4, help="Per-device training batch size.")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps.")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate.")
    parser.add_argument("--max_seq_len", type=int, default=1024, help="Maximum sequence length.")
    parser.add_argument("--lora_r", type=int, default=32, help="LoRA rank.")
    parser.add_argument("--lora_alpha", type=int, default=64, help="LoRA alpha.")
    parser.add_argument("--enable_qlora", action='store_true', help="Enable QLoRA (disabled by default).")
    parser.add_argument("--disable_flash_attention", action='store_true', help="Disable Flash Attention 2.")
    parser.add_argument("--eval_split_ratio", type=float, default=0.1, help="Ratio to split training data for evaluation.")

    args = parser.parse_args()

    # Train for the specified dataset type
    print(f"Training on OpenThoughts {args.dataset_type} dataset...")
    
    train_sft(
        expt_name=args.expt_name,
        base_model_id=args.model_id,
        output_dir=args.output_dir,
        dataset_type=args.dataset_type,
        hf_token=args.hf_token,
        wandb_token=args.wandb_token,
        num_train_epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_seq_length=args.max_seq_len,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        use_qlora=args.enable_qlora,  # Disabled by default
        use_flash_attention_2=not args.disable_flash_attention,
        eval_split_ratio=args.eval_split_ratio,
    )
