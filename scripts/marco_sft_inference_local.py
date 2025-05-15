import os
import sys
import ast
import json
import torch
import argparse
import warnings
import argparse
from marco_config import *

warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def evaluate_marco_sft_model(
    base_model_id: str = "meta-llama/Meta-Llama-3.1-8B", # Specify the base Llama 3.1 8B model [2]
    adapter_path: str = "llama3-8b-sft-adapter",
    mode: str = "default",
):  
    
    input_path = os.path.join(os.getcwd() + '/results/Marcoqa/evaluation_outputs/')
    result_file = os.path.join(input_path, f"{adapter_path}_{mode}.json")
    
    evaluate_result_dir = "results/Babiqa/evaluation_results/"
    
    output_data = []
    with open(result_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            output_data.append(data)
        

    result = evaluate_model(output_data, mode)
        
    evaluate_result_dir = os.path.join(os.getcwd() + '/', evaluate_result_dir)
    if not os.path.exists(evaluate_result_dir):
        os.makedirs(evaluate_result_dir)
        
    sanitized_model_name = adapter_path.split("/")[-1]    
    evaluate_result_path = os.path.join(evaluate_result_dir, f"{sanitized_model_name}_{mode}.json")
        
    with open(evaluate_result_path, 'w', encoding='utf-8') as f:
        json_data = json.dumps(result)
        f.write(json_data + '\n')
        
    print(f"Evaluation results saved to {evaluate_result_path}.")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the BabiQA SFT model saved results.") 
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-1.7B", help="Base model ID.") # meta-llama/Llama-3.2-1B-Instruct, Qwen/Qwen3-1.7B
    parser.add_argument("--adapter_path", type=str, default="Qwen3-1.7B-sft-adapter-reasoning-facts", help="SFT model adapter dir path.")
    parser.add_argument("--mode", type=str, default="default", help="Mode for evaluation.")
    
    args = parser.parse_args()
    
    evaluate_marco_sft_model(args.model_name, args.adapter_path, args.mode)