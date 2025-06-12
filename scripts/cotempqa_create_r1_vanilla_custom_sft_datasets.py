import os
import sys
import ast
import json
import time
import warnings
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
from conversation import Conversation

warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def merge_category_csvs(output_dir):
    categories = ['overlap', 'during', 'mix', 'equal']
    train_csvs = []
    test_csvs = []

    for category in categories:
        train_csv_path = os.path.join(output_dir, f"{category}_train.csv")
        test_csv_path = os.path.join(output_dir, f"{category}_test.csv")

        train_csv = pd.read_csv(train_csv_path)
        test_csv = pd.read_csv(test_csv_path)
        print(f"Loaded {category} train and test CSV files.")
        print(f"Train shape: {train_csv.shape}, Test shape: {test_csv.shape}")

        train_csvs.append(train_csv)
        test_csvs.append(test_csv)

    merged_train_csv = pd.concat(train_csvs)
    merged_test_csv = pd.concat(test_csvs)

    merged_train_csv.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    merged_test_csv.to_csv(os.path.join(output_dir, "test.csv"), index=False)

    print("Merged train and test CSV files.")

def main():
    
    vanilla_sft = False
    custom_trace_sft = False
    custom_incorrect_trace_sft = True
    
    categories = ['overlap', 'during', 'mix', 'equal']
    for category in categories:
    
        train_df = pd.DataFrame(columns=['index', 'question', 'reasoning', 'answer', 'messages'])
            
        response_file = 'results/Cotempqa/deepseek_r1_correct_responses_' + category + '.json'
        
        """Main function to process the JSON file"""
            
        with open(response_file, 'r') as f:
            for line in tqdm(f):
                input_text = json.loads(line)['input']
                output_text = json.loads(line)['prediction']
                
                if vanilla_sft:
                    formatted_text = {
                        "content": input_text,
                        "role": "user"
                    }, {
                        "content": output_text,
                        "role": "assistant"
                    }
                    new_row = pd.DataFrame([{
                        'index': None, # Placeholder for index, can be set later
                        'question': input_text,
                        'reasoning': None,
                        'answer': output_text,
                        'messages': list(formatted_text)
                    }])
                    train_df = pd.concat([train_df, new_row], ignore_index=True)
                
                elif custom_trace_sft:
                    og_train_csv = pd.read_csv('data/cotempqa/sft_dataset_reasoning_with_facts_chat_template/train.csv')
                    og_test_csv = pd.read_csv('data/cotempqa/sft_dataset_reasoning_with_facts_chat_template/test.csv')
                    og_combined_csv = pd.concat([og_train_csv, og_test_csv])
                    for index, row in og_combined_csv.iterrows():
                        og_answer = row['answer']
                        if output_text in og_answer:
                            message = row['messages']
                            break
                    
                    new_row = pd.DataFrame([{
                        'index': None, # Placeholder for index, can be set later
                        'question': input_text,
                        'reasoning': None,
                        'answer': output_text,
                        'messages': message
                    }])
                    train_df = pd.concat([train_df, new_row], ignore_index=True)
                
                elif custom_incorrect_trace_sft:
                    og_train_csv = pd.read_csv('data/cotempqa/sft_dataset_reasoning_facts_perturbed_chat_template/train.csv')
                    og_test_csv = pd.read_csv('data/cotempqa/sft_dataset_reasoning_facts_perturbed_chat_template/test.csv')
                    og_combined_csv = pd.concat([og_train_csv, og_test_csv])
                    for index, row in og_combined_csv.iterrows():
                        og_answer = row['answer']
                        if output_text in og_answer:
                            message = row['messages']
                            break
        
            
            if vanilla_sft:
                train_dataset_csv_dir = 'data/cotempqa/sft_dataset_r1_set_vanilla/'
            elif custom_trace_sft:
                train_dataset_csv_dir = 'data/cotempqa/sft_dataset_r1_set_custom_traces/'
            elif custom_incorrect_trace_sft:
                train_dataset_csv_dir = 'data/cotempqa/sft_dataset_r1_set_custom_incorrect_traces/'
                
            if not os.path.exists(train_dataset_csv_dir):
                os.makedirs(train_dataset_csv_dir)
                
            test_df = train_df.sample(frac=0.2, random_state=42)
            train_df = train_df.drop(test_df.index)
            train_df.to_csv(os.path.join(train_dataset_csv_dir, category+'_train.csv'), index=False)
            test_df.to_csv(os.path.join(train_dataset_csv_dir, category+'_test.csv'), index=False)
            
    merge_category_csvs(train_dataset_csv_dir)
    

if __name__ == "__main__":
    
    main()
