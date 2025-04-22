import sys
import os
import re
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
from openai import OpenAI
from datasets import load_dataset
from expt_scripts.cat_bench_prompt import PromptGenerator
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

ds = load_dataset("vanyacohen/CaT-Bench")
cat_bench_prompt = PromptGenerator()

STORAGE_DIR = os.path.join(os.getcwd(), "storage")
TEMPORAL_RELATION = "after"
EXPT_NAME = "r1_eval_cat" + "_" + TEMPORAL_RELATION

api_key=os.environ["DEEPSEEK_API"]
client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

train_ds, test_ds, eval_ds = ds['train'], ds['test'], ds['validation']

def get_accuracy(dataset, client, eval_ds = "train", eval_type="answer_only", storage_dir=None, temporal_relation=TEMPORAL_RELATION):
    prompts = []
    reasoning_contents = []
    responses = []
    error_indices = []
    error_indices_file = os.path.join(storage_dir, EXPT_NAME, "error_indices.json")
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
    reasoning_content_save_file = os.path.join(save_dir, "reasoning_contents.json")
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
        
        reasoning_content = None
        content = None
        
        @retry(wait=wait_random_exponential(min=10, max=60), stop=stop_after_attempt(6))
        def get_valid_response():
            try:
                response = client.chat.completions.create(
                    model="deepseek-reasoner",
                    messages=messages
                )
                reasoning_content = response.choices[0].message.reasoning_content
                content = response.choices[0].message.content

            except Exception as e:
                error_indices.append(i)
                response = "None"
                reasoning_content = "None"
                content = "None"

            return reasoning_content, content
        
        reasoning_content, content = get_valid_response()
        
        prompts.append(prompt)
        reasoning_contents.append(reasoning_content)
        responses.append(content)
    
        try:
            with open(prompt_save_file, 'w') as f:
                json.dump(prompts, f)
        except Exception as e:
            print(e, i)
        try:
            with open(reasoning_content_save_file, 'w') as f:
                json.dump(reasoning_contents, f)
        except Exception as e:
            print(e, i)
        try:
            with open(response_save_file, 'w') as f:
                json.dump(responses, f)
        except Exception as e:
            print(e, i)
        try:
            with open(error_indices_file, 'w') as f:
                json.dump(error_indices, f)
        except Exception as e:
            print(e, i)
        
    print(f"Prompts and responses saved to {save_dir}")

def main():

    get_accuracy(test_ds, client, eval_ds="test", eval_type="answer_only", storage_dir=STORAGE_DIR)

if __name__ == "__main__":
    main()