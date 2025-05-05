import json
import argparse
from cotempqa_config import *
import os
import sys
import warnings
from rich import print as rich_print
import builtins

builtins.print = rich_print

warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def evaluate_cotemporal(model_name, data_path, mode, output_dir, evaluate_result_dir):
    
    filename = os.path.basename(data_path)
    output_dir = os.path.join(os.getcwd() + '/', output_dir)
    sanitized_model_name = model_name.replace("/", "_")
    output_path = os.path.join(output_dir, f"{sanitized_model_name}_{mode}_{filename}")

    
    
    with open(output_path, 'r', encoding='utf-8') as out_f:
        output_data = []
        for line in out_f:
            output_data.append(json.loads(line))
            
    result = evaluate_model(output_data, mode)
        
    evaluate_result_path = os.path.join(evaluate_result_dir, f"{sanitized_model_name}_{mode}_{filename}")
    evaluate_result_dir = os.path.join(os.getcwd() + '/', evaluate_result_dir)
    if not os.path.exists(evaluate_result_dir):
        os.makedirs(evaluate_result_dir)
        
    with open(evaluate_result_path, 'w', encoding='utf-8') as f:
        json_data = json.dumps(result)
        f.write(json_data + '\n')
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Co-temporal datasets")
    parser.add_argument("--model_name", type=str, help="Path to the model")
    parser.add_argument("--data_path", type=str, help="Path to the dataset file")
    parser.add_argument("--mode", type=str, help="Method to evaluate the co-temporal ability of LLMs")
    parser.add_argument("--output_dir", type=str, help="Path to save the outputs")
    parser.add_argument("--evaluate_result_dir", type=str, help="Path to save the evaluation result")
    
    args = parser.parse_args()

    evaluate_cotemporal(args.model_name, args.data_path, args.mode, args.output_dir, args.evaluate_result_dir)