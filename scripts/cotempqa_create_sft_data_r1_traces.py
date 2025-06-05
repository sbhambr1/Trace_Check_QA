import pandas as pd
import os
import requests
from openai import OpenAI
import time
from cotempqa_config import *

# TODO

class DeepSeekR1:
    def __init__(self):
        self.api_key = os.gestenv('DEEPSEEK_API')
        self.client = OpenAI(api_key=self.api_key, base_url="https://api.deepseek.com")
        
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
                        raise e
    
def main():
    # Read train and test CSVs
    train_df = pd.read_csv('data/cotempqa/sft_dataset_chat_template/train.csv')
    # test_df = pd.read_csv('data/cotempqa/sft_dataset_chat_template/test.csv')

    deepseek = DeepSeekR1()
    output_data = []

    for index, row in train_df.iterrows():
        index = row['index']
        question = row['question']
        answer = row['answer']
        result = deepseek.query(question)
        r1_trace = result.reasoning_content
        r1_answer = result.content

        # Save reasoning trace and answers
        output_data.append({
                'input': question,
                'prediction': r1_answer,
                'gold': answer,
                'triple_element': input_data['triple_element'],
                'question': input_data['question'],
                'facts': input_data['facts']
            })

    # Evaluate final answers
    correct_answers = [row['correct_answer'] for index, row in train_df.iterrows()]
    final_traces = [trace for trace in traces if trace['answer'] in correct_answers]

    # Create a new DataFrame with traces added to the original train CSV
    final_df = pd.DataFrame(final_traces)
    save_path_dir = 'data/cotempqa/sft_dataset_r1_traces_chat_template/'
    if not os.path.exists(save_path_dir):
        os.makedirs(save_path_dir)
    save_path = os.path.join(save_path_dir, 'train_with_traces.csv')
    final_df.to_csv(save_path, index=False)

if __name__ == "__main__":
    main()
