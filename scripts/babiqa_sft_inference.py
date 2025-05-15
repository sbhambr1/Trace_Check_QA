import os
import sys
import ast
import json
import torch
import argparse
import warnings
import pandas as pd
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import PeftModel
import argparse
from babiqa_config import *

warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
torch.cuda.empty_cache()

def evaluate_marco_sft_model(
    base_model_id: str = "meta-llama/Meta-Llama-3.1-8B", # Specify the base Llama 3.1 8B model [2]
    adapter_path: str = "llama3-8b-sft-adapter",
    mode: str = "default",
):  
    output_dir="results/Babiqa/evaluation_outputs/"
    evaluate_result_dir = "results/Babiqa/evaluation_results/"
    
    print(f"Loading tokenizer for {base_model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
        
    final_adapter_path = os.path.join("models/" + adapter_path, "final_adapter")
    print("Merging adapter with base model...")
    base_model_reload = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
        # device_map={"": "cuda:0"},
        trust_remote_code=True,
    )
    merged_model = PeftModel.from_pretrained(base_model_reload, final_adapter_path)
    merged_model.set_adapter("default")
    for param in merged_model.parameters():
        param.requires_grad = False
    merged_model = merged_model.merge_and_unload()
    merged_model = base_model_reload

    # Load test samples from test.csv
    test_csv_path = os.path.join(os.getcwd() + '/data/babiqa/sft_dataset_chat_template/test.csv')
    test_samples = []
    test_df = pd.read_csv(test_csv_path)
    for _, row in test_df.iterrows():
        test_samples.append(row)
            
    all_data = test_samples

    if mode == 'default':
        all_prompts = get_prompts(all_data, default_template)
        
    if 'llama' in base_model_id.lower():
        system_message = """You are Llama, an AI assistant created to be helpful and honest. Your knowledge spans a wide range of topics, allowing you to engage in substantive conversations and provide analysis on complex subjects."""
    elif 'qwen' in base_model_id.lower():
        system_message = """You are Qwen, an AI assistant created to be helpful and honest. Your knowledge spans a wide range of topics, allowing you to engage in substantive conversations and provide analysis on complex subjects."""
    elif 'gemma' in base_model_id.lower():
        system_message = """You are Gemma, an AI assistant created to be helpful and honest. Your knowledge spans a wide range of topics, allowing you to engage in substantive conversations and provide analysis on complex subjects."""
    elif 'mistral' in base_model_id.lower():
        system_message = """You are Mistral, an AI assistant created to be helpful and honest. Your knowledge spans a wide range of topics, allowing you to engage in substantive conversations and provide analysis on complex subjects."""
    else:
        raise ValueError(f"Unknown model type for {base_model_id}. Please specify a valid model.")
        
    system_message_dict = {
        "role": "system",
        "content": system_message
    }
    
    def construct_prompt(sample, system_message_dict):
        return [system_message_dict] + [
            {"role": "user", "content": sample}]
        
    def template_dataset(examples):
        return{"text":  tokenizer.apply_chat_template(examples, tokenize=False)}
            
    all_prompts = [construct_prompt(sample, system_message_dict) for sample in all_prompts]
    all_prompts = [template_dataset(prompt) for prompt in all_prompts]
    
    all_outputs = []
    i = 0
    for prompt in all_prompts:
        inputs = tokenizer(prompt['text'], return_tensors="pt").to("cuda")
        with torch.no_grad():
            output = merged_model.generate(**inputs, max_new_tokens=500)
            if i < 10:
                print(f"Prompt {i}: {prompt}")
                print(f"Output {i}: {tokenizer.decode(output[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)}")
                print("-*-" * 20)
                i += 1
        all_outputs.append(tokenizer.decode(output[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True))
        
    output_data = []
    for prompt, input_data, output in zip(all_prompts, all_data, all_outputs):
        output_data.append({
            'input': prompt['text'],
            'prediction': output,
            'gold': input_data['answer'],
            'question': input_data['question'],
            'query_type': input_data['query_type'],
            'passage': input_data['passages'],
            'reasoning': input_data['reasoning'],
            'answer_passage': input_data['answer_passage'],
        })
        
    output_dir = os.path.join(os.getcwd() + '/', output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    sanitized_model_name = adapter_path.split("/")[-1]
    output_path = os.path.join(output_dir, f"{sanitized_model_name}_{mode}.json")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for data in output_data:
            json_data = json.dumps(data)
            f.write(json_data + '\n')

    result = evaluate_model(output_data, mode)
        
    evaluate_result_dir = os.path.join(os.getcwd() + '/', evaluate_result_dir)
    if not os.path.exists(evaluate_result_dir):
        os.makedirs(evaluate_result_dir)
        
    evaluate_result_path = os.path.join(evaluate_result_dir, f"{sanitized_model_name}_{mode}.json")
        
    with open(evaluate_result_path, 'w', encoding='utf-8') as f:
        json_data = json.dumps(result)
        f.write(json_data + '\n')
        
    print(f"Evaluation results saved to {evaluate_result_path}.")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the BabiQA SFT model.")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="Base model ID.")
    parser.add_argument("--adapter_path", type=str, default="Meta-Llama-3.1-8B-sft-adapter", help="SFT model adapter dir path.")
    parser.add_argument("--mode", type=str, default="default", help="Mode for evaluation.")
    
    args = parser.parse_args()
    
    evaluate_marco_sft_model(args.model_name, args.adapter_path, args.mode)