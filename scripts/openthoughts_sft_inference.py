import os
import sys
import ast
import json
import torch
import argparse
import warnings
import pandas as pd
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

warnings.filterwarnings("ignore")
torch.cuda.empty_cache()

def load_and_format_test_data(dataset_name, split, question_col, answer_col, subset=None):
    """Load and format test data from HuggingFace dataset repo, with optional subset."""
    if subset:
        ds = load_dataset(dataset_name, subset, split=split)
    else:
        ds = load_dataset(dataset_name, split=split)
    data = []
    for item in ds:
        question = item.get(question_col, "")
        answer = item.get(answer_col, "")
        data.append({"question": question, "answer": answer})
    return data

def construct_prompt(sample, system_message):
    # If the sample already has a 'messages' field, use it; otherwise, construct a prompt
    if 'messages' in sample:
        messages = sample['messages']
        if messages[0]['role'] != 'system':
            messages = [{"role": "system", "content": system_message}] + messages
        return messages
    # Otherwise, construct a generic prompt
    prompt = sample.get('question', sample.get('input', ''))
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
    ]
    return messages

def run_inference_on_dataset(
    base_model_id,
    adapter_path,
    test_data,
    output_dir,
    dataset_name,
    max_new_tokens=32768,
    device="cuda"
):
    print(f"\n=== Running inference on {dataset_name} ===")
    print(f"Loaded {len(test_data)} test samples.")

    # Load tokenizer and model
    print(f"Loading tokenizer and base model: {base_model_id}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print(f"Loading LoRA adapter from: {adapter_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    base_model.config.use_cache = False
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.set_adapter("default")
    for param in model.parameters():
        param.requires_grad = False
    model = model.merge_and_unload()
    model.eval()

    # System message
    if 'llama' in base_model_id.lower():
        system_message = "You are Llama, an AI assistant created to be helpful and honest. Your knowledge spans a wide range of topics, allowing you to engage in substantive conversations and provide analysis on complex subjects."
    elif 'qwen' in base_model_id.lower():
        system_message = "You are Qwen, an AI assistant created to be helpful and honest. Your knowledge spans a wide range of topics, allowing you to engage in substantive conversations and provide analysis on complex subjects."
    elif 'gemma' in base_model_id.lower():
        system_message = "You are Gemma, an AI assistant created to be helpful and honest. Your knowledge spans a wide range of topics, allowing you to engage in substantive conversations and provide analysis on complex subjects."
    elif 'mistral' in base_model_id.lower():
        system_message = "You are Mistral, an AI assistant created to be helpful and honest. Your knowledge spans a wide range of topics, allowing you to engage in substantive conversations and provide analysis on complex subjects."
    else:
        system_message = "You are an AI assistant."

    results = []
    for i, sample in enumerate(test_data):
        messages = construct_prompt(sample, system_message)
        prompt = tokenizer.apply_chat_template(messages, tokenize=False)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        prediction = tokenizer.decode(output[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
        results.append({
            'input': prompt,
            'prediction': prediction,
            'gold': sample.get('answer', ''),
            'question': sample.get('question', ''),
            'meta': {k: v for k, v in sample.items() if k not in ['input', 'prediction', 'answer', 'question']}
        })
        if i < 3:
            print(f"Sample {i} prompt: {prompt}")
            print(f"Sample {i} prediction: {prediction}")
            print("-*" * 20)

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{os.path.basename(adapter_path)}_{dataset_name}_inference.jsonl")
    with open(output_path, 'w', encoding='utf-8') as f:
        for res in results:
            f.write(json.dumps(res) + '\n')
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference for OpenThoughts SFT models on AIME/MATH datasets.")
    parser.add_argument("--model_name", type=str, required=True, help="Base model ID.")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to LoRA adapter directory.")
    parser.add_argument("--output_dir", type=str, default="results/OpenThoughts/inference_outputs/", help="Directory to save outputs.")
    parser.add_argument("--max_new_tokens", type=int, default=32768, help="Max output tokens.")
    args = parser.parse_args()

    # Fixed dataset sources and columns
    # AIME2024: split 'train'
    test_data = load_and_format_test_data("Maxwell-Jia/AIME_2024", split="train", question_col="Problem", answer_col="Answer")
    run_inference_on_dataset(
        base_model_id=args.model_name,
        adapter_path=args.adapter_path,
        test_data=test_data,
        output_dir=args.output_dir,
        dataset_name="AIME2024",
        max_new_tokens=args.max_new_tokens,
        device="cuda"
    )
    # AIME2025: two subsets, both with split 'test'
    for subset, label in [("AIME2025-I", "AIME2025-I"), ("AIME2025-II", "AIME2025-II")]:
        test_data = load_and_format_test_data("opencompass/AIME2025", split="test", question_col="question", answer_col="answer", subset=subset)
        run_inference_on_dataset(
            base_model_id=args.model_name,
            adapter_path=args.adapter_path,
            test_data=test_data,
            output_dir=args.output_dir,
            dataset_name=label,
            max_new_tokens=args.max_new_tokens,
            device="cuda"
        )
    # MATH500: split 'test'
    test_data = load_and_format_test_data("HuggingFaceH4/MATH-500", split="test", question_col="problem", answer_col="answer")
    run_inference_on_dataset(
        base_model_id=args.model_name,
        adapter_path=args.adapter_path,
        test_data=test_data,
        output_dir=args.output_dir,
        dataset_name="MATH500",
        max_new_tokens=args.max_new_tokens,
        device="cuda"
    )
