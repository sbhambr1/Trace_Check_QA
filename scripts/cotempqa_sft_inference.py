import os
import sys
import json
import torch
import argparse
import warnings
import pandas as pd
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import PeftModel
import argparse
from cotempqa_config import *

warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
torch.cuda.empty_cache()

def evaluate_cotemporal_sft_model(
    base_model_id: str = "meta-llama/Meta-Llama-3.1-8B", # Specify the base Llama 3.1 8B model [2]
    adapter_path: str = "llama3-8b-sft-adapter",
    data_path: str = "./data/cotempqa/mix.json",
    mode: str = "few_shot_cot",
    output_dir: str = "results/Cotempqa/evaluation_outputs/mix_few_shot_cot/",
    evaluate_result_dir: str = "results/Cotempqa/evaluation_results/mix_few_shot_cot/",
):
    all_data = []
    data_path = os.path.join(os.getcwd() + '/', data_path)
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            all_data.append(data)
            
    # Load test samples from test.csv
    test_csv_path = os.path.join(os.getcwd() + '/data/cotempqa/sft_dataset_chat_template/test.csv')
    test_samples = []
    test_df = pd.read_csv(test_csv_path)
    for _, row in test_df.iterrows():
        test_samples.append(row['question'])
            
    # Filter out test samples from all_data
    test_data = []
    test_questions = set(test_samples)
    test_data = [data for data in all_data if any(data['question'] in question for question in test_questions)]
    
    all_data = test_data

    if mode == 'default':
        all_prompts = get_prompts(all_data, default_template)
    elif mode == 'few_shot':
        all_prompts = get_prompts(all_data, few_shot_template)
    elif mode == 'few_shot_cot':
        all_prompts = get_prompts(all_data, few_shot_cot_template)
    elif mode == 'few_shot_math_cot':
        all_prompts = get_prompts(all_data, few_shot_math_template)
    elif mode == 'default_with_reasoning':
        all_prompts = get_prompts_with_trace(all_data, default_template_with_trace)
        
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
        trust_remote_code=True,
    )
    merged_model = PeftModel.from_pretrained(base_model_reload, final_adapter_path)
    merged_model.set_adapter("default")
    for param in merged_model.parameters():
        param.requires_grad = False
    merged_model = merged_model.merge_and_unload()
    
    all_outputs = []
    for prompt in all_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            output = merged_model.generate(**inputs, max_new_tokens=500)
        all_outputs.append(tokenizer.decode(output[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True))
        
    output_data = []
    for prompt, input_data, output in zip(all_prompts, all_data, all_outputs):
        prompt = 'Answer the question based on the context:' + prompt.split('Answer the question based on the context:')[-1]
        output_data.append({
            'input': prompt,
            'prediction': output,
            'gold': input_data['answer'],
            'triple_element': input_data['triple_element'],
            'question': input_data['question'],
            'facts': input_data['facts']
        })
        
    filename = os.path.basename(data_path)
    output_dir = os.path.join(os.getcwd() + '/', output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    sanitized_model_name = adapter_path
    output_path = os.path.join(output_dir, f"{sanitized_model_name}_{mode}_{filename}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for data in output_data:
            json_data = json.dumps(data)
            f.write(json_data + '\n')

    result = evaluate_model(output_data, mode)
        
    evaluate_result_path = os.path.join(evaluate_result_dir, f"{sanitized_model_name}_{mode}_{filename}")
    evaluate_result_dir = os.path.join(os.getcwd() + '/', evaluate_result_dir)
    if not os.path.exists(evaluate_result_dir):
        os.makedirs(evaluate_result_dir)
        
    with open(evaluate_result_path, 'w', encoding='utf-8') as f:
        json_data = json.dumps(result)
        f.write(json_data + '\n')
        
    print(f"Evaluation results saved to {evaluate_result_path}.")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the Cotemporal SFT model.")
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3.1-8B", help="Base model ID.")
    parser.add_argument("--adapter_path", type=str, default="Meta-Llama-3.1-8B-sft-adapter", help="SFT model adapter dir path.")
    parser.add_argument("--data_path", type=str, default="data/cotempqa/mix.json", help="Path to the dataset file.")
    parser.add_argument("--mode", type=str, default="few_shot_cot", help="Mode for evaluation.")
    parser.add_argument("--output_dir", type=str, default="results/Cotempqa/evaluation_outputs/mix_few_shot_cot/", help="Output directory.")
    parser.add_argument("--evaluate_result_dir", type=str, default="results/Cotempqa/evaluation_results/mix_few_shot_cot/", help="Path to save the evaluation result.")
    
    args = parser.parse_args()
    
    evaluate_cotemporal_sft_model(args.model_name, args.adapter_path, args.data_path, args.mode, args.output_dir, args.evaluate_result_dir)
