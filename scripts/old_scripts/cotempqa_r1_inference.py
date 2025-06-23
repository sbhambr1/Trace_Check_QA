import json
import pandas as pd
import argparse
from cotempqa_config import *
import os
import sys
import warnings
from openai import OpenAI
import time
from tqdm import tqdm

warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class DeepSeekR1:
    def __init__(self):
        self.api_key = os.getenv('DEEPSEEK_API')
        self.client = OpenAI(api_key=self.api_key, base_url="https://api.deepseek.com")
        self.none_errors = 0
        
    def query(self, question, retries=3, wait_time=5):
        for attempt in range(retries):
            try:
                response = self.client.chat.completions.create(
                    model="deepseek-reasoner",
                    messages=[
                        {"role": "user", "content": question},
                    ],
                    stream=False
                )
                return response.choices[0].message
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(wait_time)
                else:
                    self.none_errors += 1
                    return None
                    
    def generate(self, all_prompts):
        """
        Generate responses for all prompts using the DeepSeekR1 model.
        
        Parameters:
        all_prompts (list): List of prompts to generate responses for.
        
        Returns:
        list: List of generated traces and answers.
        """
        all_outputs_answers = []
        all_outputs_traces = []
        for prompt in tqdm(all_prompts):
            response = self.query(prompt)
            if response is None:
                all_outputs_answers.append("None")
                all_outputs_traces.append("None")
                continue
            r1_trace = response.reasoning_content
            r1_answer = response.content
            all_outputs_answers.append(r1_answer)
            all_outputs_traces.append(r1_trace)
            
        print(f"Total None Errors: {self.none_errors}")
        return all_outputs_answers, all_outputs_traces

def check_in_test_set(line, test_csv_path):
    """
    Check if the line is in the test set.
    
    Parameters:
    line (str): The line to check.
    test_csv_path (str): Path to the test set CSV file.
    
    Returns:
    int: 1 if in test set, 0 otherwise.
    """
    data = json.loads(line)
    df = pd.read_csv(test_csv_path)
    for index, row in df.iterrows():
        if str(data['answer']) == row['answer']:
            return 1
    return 0

def get_data_from_json(data, data_path):
    """
    Get data from a JSON file.
    
    Parameters:
    data (dict): The data to extract.
    data_path (str): Path to the JSON file.
    
    Returns:
    dict: Extracted data.
    """
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = json.loads(line)
            if str(line['answer']) == data['answer']:
                data = line
                break
            
    return data 

def evaluate_cotemporal(data_path, mode, output_dir, evaluate_result_dir):
    """
    Evaluate the co-temporal reasoning capabilities of a model on a dataset.
    
    Parameters:
    data_path (str): Path to the input dataset.
    mode (str): Evaluation mode (e.g., 'default', 'few_shot', 'few_shot_cot', 'few_shot_math_cot', 'default_with_trace').
    output_dir (str): Directory to save the evaluation outputs.
    evaluate_result_dir (str): Directory to save the evaluation results.
    """
    all_data = []
    data_path = os.path.join(os.getcwd() + '/', data_path)
    # category = data_path.split('/')[-1].split('.json')[0]
    # test_csv_path = os.path.join(os.getcwd(), 'data/cotempqa/sft_dataset_chat_template', f'{category}_test.csv')
    # df = pd.read_csv(test_csv_path)
    # for _, row in df.iterrows():
    #     data = {
    #         'answer': row['answer'],
    #     }
    #     data = get_data_from_json(data, data_path)
    #     all_data.append(data)
        
    # with open(data_path, 'r', encoding='utf-8') as f:
    #     for line in f:
    #         flag = check_in_test_set(line, test_csv_path)
    #         if flag == 0:
    #             continue
    #         data = json.loads(line)
    #         all_data.append(data)
    
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            all_data.append(data)

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
        
    deepseek = DeepSeekR1()
    all_outputs = deepseek.generate(all_prompts)
    all_outputs_answers, all_outputs_traces = all_outputs
    
    output_data = []
    for prompt, input_data, output_answer, output_trace in zip(all_prompts, all_data, all_outputs_answers, all_outputs_traces):
        prompt = 'Answer the question based on the context:' + prompt.split('Answer the question based on the context:')[-1]
        output_data.append({
            'input': prompt,
            'prediction': output_answer,
            'r1_trace': output_trace,
            'gold': input_data['answer'],
            'triple_element': input_data['triple_element'],
            'question': input_data['question'],
            'facts': input_data['facts']
        })
    
    filename = os.path.basename(data_path)
    output_dir = os.path.join(os.getcwd() + '/', output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    sanitized_model_name = 'deepseek_r1'
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Co-temporal datasets")
    parser.add_argument("--data_path", type=str, help="Path to the dataset file")
    parser.add_argument("--mode", type=str, help="Method to evaluate the co-temporal ability of LLMs")
    parser.add_argument("--output_dir", type=str, help="Path to save the outputs")
    parser.add_argument("--evaluate_result_dir", type=str, help="Path to save the evaluation result")
    
    args = parser.parse_args()

    evaluate_cotemporal(args.data_path, args.mode, args.output_dir, args.evaluate_result_dir)
