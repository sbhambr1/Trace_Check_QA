import os
import ast
import json
from tqdm import tqdm
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from marco_config import *

def get_accuracy(trace_eval_path, golden_eval_path):
    answer_accuracy = 0.0
    total = 0.0
        
    golden_csv = pd.read_csv(golden_eval_path)
    
    mode = 'default'
    
    with open(trace_eval_path, 'r') as f:
        for line in tqdm(f):
            total += 1.0
            line = json.loads(line)
            
            result = evaluate_model([line], mode)
            correct_answer = 1 if result['acc'] == 100.0 else 0
            answer_accuracy += correct_answer
            
    answer_accuracy = answer_accuracy / total * 100.0 if total > 0 else 0.0
    return answer_accuracy

def main(adapter_name):
        
    trace_eval_path = f'results/Marcoqa/evaluation_outputs/{adapter_name}'
    if 'reasoning-facts' in adapter_name:
        golden_eval_path = f'data/marcoqa/sft_dataset_reasoning_with_facts_chat_template/test.csv'
    elif 'reasoning' in adapter_name:
        golden_eval_path = f'data/marcoqa/sft_dataset_reasoning_chat_template/test.csv'
    else:
        golden_eval_path = f'data/marcoqa/sft_dataset_chat_template/test.csv'
        
    accuracy = get_accuracy(trace_eval_path, golden_eval_path)
    
    return accuracy


if __name__ == "__main__":
    
    adapter_names = ['Llama-3.2-1B-Instruct-sft-adapter_default.json', 'Qwen3-1.7B-sft-adapter_default.json']
    
    for adapter_name in adapter_names:
        print('\n\n')
        print("="*50)
        print(f"Evaluating adapter: {adapter_name}")
        accuracy = main(adapter_name)
        
        print("Accuracy:", accuracy)
    