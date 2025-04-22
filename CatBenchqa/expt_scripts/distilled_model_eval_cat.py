import sys
import os
import re
import ollama
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
from datasets import load_dataset
from expt_scripts.cat_bench_prompt import PromptGenerator
import subprocess

ds = load_dataset("vanyacohen/CaT-Bench")
cat_bench_prompt = PromptGenerator()

STORAGE_DIR = os.path.join(os.getcwd(), "storage")
MODEL_NAME = "deepseek-r1:8b"
TEMPORAL_RELATION = "before"
EXPT_NAME = MODEL_NAME + "_eval_cat" + "_" + TEMPORAL_RELATION

train_ds, test_ds, eval_ds = ds['train'], ds['test'], ds['validation']

def get_accuracy(dataset, MODEL_NAME, eval_ds = "train", eval_type="answer_only", storage_dir=None, temporal_relation=TEMPORAL_RELATION):
    correct = 0
    total = 0
    prompts = []
    responses = []
    if eval_ds == "train":
        eval_name = "train"
    elif eval_ds == "test":
        eval_name = "test"
    elif eval_ds == "validation":
        eval_name = "validation"
    
    save_dir = os.path.join(storage_dir, EXPT_NAME, eval_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    prompt_save_file = os.path.join(save_dir, "prompts.json")
    response_save_file = os.path.join(save_dir, "responses.json")
    
    for i in tqdm(range(len(dataset))):
        title, prompt_steps_string, i, j, label = PromptGenerator.format_input_for_prompt(dataset[i], temporal_relation)
        if eval_type == "answer_only":
            prompt = PromptGenerator.get_answer_only_prompt(title, prompt_steps_string, i, j, temporal_relation)
        elif eval_type == "answer_explanation":
            prompt = PromptGenerator.get_answer_explanation_prompt(title, prompt_steps_string, i, j, temporal_relation)
        elif eval_type == "explanation_answer":
            prompt = PromptGenerator.get_explanation_answer_prompt(title, prompt_steps_string, i, j, temporal_relation)
        elif eval_type == "nl_answer_explanation":
            prompt = PromptGenerator.get_nl_answer_explanation_prompt(title, prompt_steps_string, i, j, temporal_relation)
        else:
            raise ValueError("Invalid eval_type")
        
        messages = [{"role": "user", "content": prompt}]
        response = ollama.chat(
            model=MODEL_NAME,
            messages=messages
        )
        content = response['message']['content']
        
        prompts.append(prompt)
        responses.append(content)
    
        with open(prompt_save_file, 'w') as f:
            json.dump(prompts, f)
        with open(response_save_file, 'w') as f:
            json.dump(responses, f)
        
    print(f"Prompts and responses saved to {save_dir}")

def main():
    
    command_run = f"ollama start {MODEL_NAME}"
    subprocess.run(command_run, shell=True)

    get_accuracy(test_ds, MODEL_NAME, eval_ds="test", eval_type="answer_only", storage_dir=STORAGE_DIR)
    
    command_stop = f"ollama stop {MODEL_NAME}"
    subprocess.run(command_stop, shell=True)

if __name__ == "__main__":
    main()