import sys
import os
import re
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
from openai import OpenAI
from datasets import load_dataset
from expt_scripts.cat_bench_prompt import PromptGenerator

ds = load_dataset("vanyacohen/CaT-Bench")
cat_bench_prompt = PromptGenerator()
STORAGE_DIR = os.path.join(os.getcwd(), "storage")
TEMPORAL_RELATION = "after"
OPENAI_MODEL = "gpt-4o"
EXPT_NAME = OPENAI_MODEL + "_eval_cat" + "_" + TEMPORAL_RELATION

api_key=os.environ["OPENAI_API"]
client = OpenAI(api_key=api_key)

train_ds, test_ds, eval_ds = ds['train'], ds['test'], ds['validation']

def construct_message(prompt, role):
        assert role in ["user", "assistant"]
        new_message = {"role": role, "content": prompt}
        llm_prompt = []
        message = llm_prompt + [new_message]
        return message

def get_accuracy(dataset, client, eval_ds = "train", eval_type="answer_only", storage_dir=None, temporal_relation=TEMPORAL_RELATION):
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
        
        message = construct_message(prompt, role="user") 
        
        try:
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages = message,
                temperature=0,
                max_tokens=2,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop=["\n"]
                )
            content = response.choices[0].message.content
            
        except Exception as e:
            print('[ERROR]: ', e)
            print('[ERROR index]: ', i)
            response = "None"
            content = "None"
        
        prompts.append(prompt)
        responses.append(content)
    
        try:
            with open(prompt_save_file, 'w') as f:
                json.dump(prompts, f)
        except Exception as e:
            print(e, i)
        try:
            with open(response_save_file, 'w') as f:
                json.dump(responses, f)
        except Exception as e:
            print(e, i)
            
    print(f"Prompts and responses saved to {save_dir}")

def main():
    get_accuracy(test_ds, client, eval_ds="test", eval_type="answer_only", storage_dir=STORAGE_DIR)
    
if __name__ == "__main__":
    main()