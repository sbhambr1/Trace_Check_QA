import os
import sys
import json
import time
import warnings
from tqdm import tqdm
from datasets import load_dataset
from conversation import Conversation

warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def create_explanation_prompt():
    """Create a standard community-accepted explanation behind your answer prompt template which can be asked in continuation of a conversation with the R1 model after it has output a trace and a correct answer for the initial problem."""
    prompt = "You have answered the question correctly. Please provide a concise and very clear explanation of the reasoning behind the answer in less than 100 words. \n Explanation: "
    return prompt 

def explain_answer(llm, problem, r1_trace, r1_answer):
    """Call API to explain the answer"""
    prompt = create_explanation_prompt()
    explanation = llm.get_multi_turn_response(problem, r1_trace, r1_answer, prompt)
    return explanation

def main():
    
    explanation_model = "gpt-4o-mini"
    llm = Conversation(explanation_model, temp=0)
    
    """Main function to process the JSON file"""
    all_results_with_explanation = []
    ds = load_dataset("open-thoughts/OpenThoughts-114k", "metadata", split="train")
    seed = 42
    ds.shuffle(seed=seed)
    data = ds.filter(lambda x: x['domain'] == 'math')[:3000]
    
    for i in tqdm(range(3000)):
        problem = data['problem'][i]
        deepseek_trace = data['deepseek_reasoning'][i]
        deepseek_solution = data['deepseek_solution'][i]
        ground_truth_solution = data['ground_truth_solution'][i]
        domain = data['domain'][i]
        source = data['source'][i]
        explanation = explain_answer(llm, problem, deepseek_trace, deepseek_solution)
        new_data = {
            'input': problem,
            'prediction': deepseek_solution,
            'ground_truth_solution': ground_truth_solution,
            'r1_trace': deepseek_trace,
            'explanation': explanation,
            'domain': domain,
            'source': source
        }
        all_results_with_explanation.append(new_data)
                
    save_path = 'results/OpenThoughts/deepseek_r1_with_explanation_' + explanation_model + '.json'
    with open(save_path, 'w') as f:
        for record in all_results_with_explanation:
            f.write(json.dumps(record) + '\n')

if __name__ == "__main__":
    
    main()
