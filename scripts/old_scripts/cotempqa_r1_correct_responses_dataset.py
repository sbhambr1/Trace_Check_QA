import pandas as pd
import os
import requests
from openai import OpenAI
import time
from cotempqa_config import *
    
def main():
    categories = ['during', 'equal', 'mix', 'overlap']
    all_correct_results = []
    evaluation_outputs_dir = 'results/Cotempqa/evaluation_outputs/'
    for category in categories:
        cat_correct_results = []
        cat_eval_dir = os.path.join(evaluation_outputs_dir, category+'_default')
        r1_output_file = os.path.join(cat_eval_dir, 'deepseek_r1_default_'+category+'.json')
        with open(r1_output_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                res_dict = evaluate_model([data], 'default')
                if res_dict['acc'] == 100.0:
                    all_correct_results.append(data)
                    cat_correct_results.append(data)
                    
        # Save the correct responses for the current category
        cat_correct_responses_file = 'results/Cotempqa/deepseek_r1_correct_responses_' + category + '.json'
        with open(cat_correct_responses_file, 'w') as f:
            for item in cat_correct_results:
                f.write(json.dumps(item) + '\n')
                
    # Save the correct responses to a new JSON file
    correct_responses_file = 'results/Cotempqa/deepseek_r1_correct_responses.json'
    with open(correct_responses_file, 'w') as f:
        for item in all_correct_results:
            f.write(json.dumps(item) + '\n')

        

if __name__ == "__main__":
    main()
