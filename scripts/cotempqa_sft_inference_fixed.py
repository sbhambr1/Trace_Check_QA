import os
import sys
import ast
import json
import torch
import argparse
import warnings
import pandas as pd
from datasets import load_from_disk, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import PeftModel
import argparse
from cotempqa_config import *
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
torch.cuda.empty_cache()

def evaluate_model_all_data(
        model_name: str,
        data_path: str,
        adapter_path: str,
        output_dir: str,
        evaluate_result_dir: str,
        mode: str,
        tokenizer: AutoTokenizer,
        dataset=None,
        merged_model=None,
    ):
    
    all_data = []
    data_path = os.path.join(os.getcwd() + '/', data_path)
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            all_data.append(data)
            
    # Load test samples from test.csv
    test_csv = dataset["test"].to_pandas()
    test_samples = []
    for _, row in test_csv.iterrows():
        test_samples.append(row['messages'][:-1])
        
    system_message_dict = test_samples[0][0]
    test_data = [data for data in all_data if any(data['question'] in item[1]['content'] for item in test_samples)]
    
    def construct_prompt(sample, system_message_dict=system_message_dict):
        prompt = 'Answer the question based on the context:\n{fact}\nQuestion: {question} Only return the answer.\n'
        prompt = prompt.format(
            fact=sample['facts'],
            question=sample['question']
        )
        prompt = [{"role": "system", "content": system_message_dict}] + [{"role": "user", "content": prompt}]
        return prompt
   
    def template_dataset(examples):
        return{"text":  tokenizer.apply_chat_template(examples, tokenize=False)}
            
    # Construct the prompt for each sample
    all_prompts = [construct_prompt(sample) for sample in test_data]
    all_prompts = [template_dataset(prompt) for prompt in all_prompts]
    
    all_outputs = []
    i = 0
    for prompt in all_prompts:
        inputs = tokenizer(prompt['text'], return_tensors="pt").to("cuda")
        with torch.no_grad():
            output = merged_model.generate(**inputs, max_new_tokens=500)
            if i < 5:
                print(f"Prompt {i}: {prompt}")
                print(f"Output {i}: {tokenizer.decode(output[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)}")
                print("-*-" * 20)
                i += 1
        all_outputs.append(tokenizer.decode(output[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True))
    
    output_data = []
    for prompt, input_data, output in zip(all_prompts, test_data, all_outputs):
        output_data.append({
            'input': prompt,
            'prediction': output,
            'gold': input_data['answer'],
            'triple_element': input_data['triple_element'],
            'question': input_data['question'],
            'facts': input_data['facts']
        })
    
    category_type = data_path.split(".json")[0].split("/")[-1]
    category_dir_result_data = os.path.join(output_dir, category_type)
    if not os.path.exists(category_dir_result_data):
        os.makedirs(category_dir_result_data)
    
    sanitized_model_name = adapter_path
    output_path = os.path.join(category_dir_result_data, f"{sanitized_model_name}_{mode}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for data in output_data:
            json_data = json.dumps(data)
            f.write(json_data + '\n')

    result = evaluate_model(output_data, mode)
        
    category_dir_results = os.path.join(evaluate_result_dir, category_type)
    
    evaluate_result_path = os.path.join(category_dir_results, f"{sanitized_model_name}_{mode}")
    if not os.path.exists(evaluate_result_path):
        os.makedirs(evaluate_result_path)
        
    with open(evaluate_result_path, 'w', encoding='utf-8') as f:
        json_data = json.dumps(result)
        f.write(json_data + '\n')
        
    print(f"Evaluation results saved to {evaluate_result_path}.")

def evaluate_cotemporal_sft_model(
    base_model_id: str = "meta-llama/Llama-3.2-1B-Instruct", # Specify the base Llama 3.1 8B model [2]
    adapter_path: str = "Llama-3",
    mode: str = "few_shot_cot",
):
    dataset = load_dataset("sbhambr1/cotempqa_for_sft_reasoning_facts", data_files={"train": "train.csv", "test": "test.csv"})
    
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
    
    # Parse the "messages" column as a list
    dataset = dataset.map(parse_messages_column, batched=False)
    
    columns_to_remove = list(dataset["train"].features)
    columns_to_remove.remove("messages")
    dataset = dataset.map(create_conversation, remove_columns=columns_to_remove,batched=False)
    
    # save datasets to disk
    dataset["train"].to_json("train_dataset.json", orient="records", force_ascii=False)
    dataset["test"].to_json("test_dataset.json", orient="records", force_ascii=False)
    
    final_adapter_path = os.path.join("models/" + adapter_path, "final_adapter")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
        
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
    
    data_paths = ["./data/cotempqa/mix.json", "./data/cotempqa/equal.json", "./data/cotempqa/overlap.json", "./data/cotempqa/during.json"]
    
    for data_path in data_paths:
        evaluate_model_all_data(
            model_name=base_model_id,
            data_path=data_path,
            adapter_path=adapter_path,
            output_dir="results/Cotempqa/evaluation_outputs/",
            evaluate_result_dir = "results/Cotempqa/evaluation_results/",
            mode=mode,
            tokenizer=tokenizer,
            dataset=dataset,
            merged_model=merged_model
        )
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the Cotemporal SFT model.")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="Base model ID.")
    parser.add_argument("--adapter_path", type=str, default="Llama-3", help="SFT model adapter dir path.")
    parser.add_argument("--mode", type=str, default="default", help="Mode for evaluation.")    
    args = parser.parse_args()
    
    evaluate_cotemporal_sft_model(args.model_name, args.adapter_path, args.mode)
