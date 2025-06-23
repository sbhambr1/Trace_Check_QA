import os
import sys
import json
import time
import warnings
from openai import OpenAI
from tqdm import tqdm
from conversation import Conversation

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
                
    def multi_turn_query(self, problem, r1_trace, r1_answer, question, retries=3, wait_time=5):
        for attempt in range(retries):
            try:
                response = self.client.chat.completions.create(
                    model="deepseek-reasoner",
                    messages=[
                        {"role": "user", "content": problem},
                        {"role": "assistant", "content": r1_trace},
                        {"role": "assistant", "content": r1_answer},
                        {"role": "user", "content": question}
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
                    
    def generate(self, problem, r1_trace, r1_answer, prompt):
        """
        Generate responses for all prompts using the DeepSeekR1 model.
        
        Parameters:
        all_prompts (list): List of prompts to generate responses for.
        
        Returns:
        list: List of generated traces and answers.
        """
        
        response = self.multi_turn_query(problem, r1_trace, r1_answer, prompt)
        if response is None:
            r1_exp_trace = None
            r1_exp_answer = None  
            self.none_errors += 1
            print(f"Total None Errors: {self.none_errors}")
        r1_exp_trace = response.reasoning_content
        r1_exp_answer = response.content
            
        return r1_exp_answer, r1_exp_trace

def create_explanation_prompt():
    """Create a standard community-accepted explanation behind your answer prompt template which can be asked in continuation of a conversation with the R1 model after it has output a trace and a correct answer for the initial problem."""
    prompt = "You have answered the question correctly. Please provide a detailed explanation of the reasoning behind your answer. The explanation should be clear, concise, and easy to understand. \n Explanation: "
    return prompt 

def explain_answer(deepseek, problem, r1_trace, r1_answer):
    """Call API to explain the answer"""
    prompt = create_explanation_prompt()
    r1_exp_ans, r1_exp_trace = deepseek.generate(problem, r1_trace, r1_answer, prompt)
    return r1_exp_ans, r1_exp_trace

def main():
    
    deepseek = DeepSeekR1()
    
    """Main function to process the JSON file"""
    categories = ['during', 'equal', 'mix', 'overlap']
    all_results_with_explanation = []
    for category in categories:
        print(f"Processing category: {category}")
        cat_results_with_explanation = []
        cat_correct_responses_file = 'results/Cotempqa/deepseek_r1_correct_responses_' + category + '.json'
        with open(cat_correct_responses_file, 'r') as f:
            for line in tqdm(f):
                data = json.loads(line)
                problem = data['input']
                r1_trace = data['r1_trace']
                r1_answer = data['prediction']
                r1_exp_ans, r1_exp_trace = explain_answer(deepseek, problem, r1_trace, r1_answer)
                data['r1_exp_ans'] = r1_exp_ans
                data['r1_exp_trace'] = r1_exp_trace
                cat_results_with_explanation.append(data)
                all_results_with_explanation.append(data)
                
        cat_correct_responses_file_with_explanation = 'results/Cotempqa/deepseek_r1_correct_responses_' + category + '_with_explanation.json'
        with open(cat_correct_responses_file_with_explanation, 'w') as f:
            for record in cat_results_with_explanation:
                f.write(json.dumps(record) + '\n')
                
    correct_responses_file_with_explanation = 'results/Cotempqa/deepseek_r1_correct_responses_with_explanation.json'
    with open(correct_responses_file_with_explanation, 'w') as f:
        for record in all_results_with_explanation:
            f.write(json.dumps(record) + '\n')

if __name__ == "__main__":
    
    main()
