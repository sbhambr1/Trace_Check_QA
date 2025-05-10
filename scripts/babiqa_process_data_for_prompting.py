import os
import sys
import ast
import json
import warnings
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

query_mapping = {
    "qa1": "single-supporting-fact",
    "qa2": "two-supporting-facts",
    "qa4": "two-arg-relations",
    "qa7": "counting",
    "qa8": "lists-sets",
    "qa12": "conjunction",
    "qa14": "time-reasoning",
    "qa15": "basic-deduction",
    "qa16": "basic-induction",
    "qa17": "positional-reasoning",
    "qa18": "size-reasoning",
}

warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

default_template = 'Answer the question based on the context:\n{facts}\nQuestion: {question} Only return the answer.\n'

reasoning_template = 'The category of the question represents what is being asked in that question. The category of the question asked is: {query_type}.\n'

# Load the dataset from csv
dataset_train = pd.read_csv("data/babiqa/train.csv")
dataset_test = pd.read_csv("data/babiqa/test.csv")

print(len(dataset_train), "rows in train dataset")
print(len(dataset_test), "rows in test dataset")
print('-' * 50)

def process_dataset(df):    
    processed_rows = []
    errors = 0
    for idx, row in df.iterrows():
        # Get all preceding factual statements
        text = ast.literal_eval(row['story'])
        
        # text.keys()
        # dict_keys(['id', 'type', 'text', 'supporting_ids', 'answer'])
        
        # index of first '1' entry in text['type']
        question_idx = text['type'].index(1)
        question = text['text'][question_idx]
        
        # passage_text is all the text before the question
        passage_text = text['text'][:question_idx]
        
        # Get the answer text for the question using the supporting ids
        supporting_ids = next((lst for lst in text['supporting_ids'] if lst), None)
        
        # the first non empty list in the supporting_ids
        answer_passage_indices = [int(x)-1 for x in supporting_ids]
        answer_passage_indices.sort()
        answer_passage = [text['text'][idx] for idx in answer_passage_indices]
        
        answer = next((ans for ans in text['answer'] if ans.strip()), None)
        
        # Create query using template
        query = default_template.format(
            facts="\n".join(passage_text),
            question=question
        )
        
        query_type = query_mapping.get(str(row['qa']))
        if query_type is None:
            errors += 1
            continue
        
        processed_rows.append({
            'passages': passage_text,
            'query': query,
            'question': question,
            'query_type': query_type,
            'reasoning': reasoning_template.format(query_type=query_type),
            'answer_passage': answer_passage,
            'answer': answer,
        })

    print(f"Errors in processing: {errors}")
    return pd.DataFrame(processed_rows)

def filter_test_dataset(df, keep_percentage=0.1):
    filtered_data = pd.DataFrame()
    query_types = df['query_type'].unique()
    for query_type in query_types:
        query_type_data = df[df['query_type'] == query_type]
        num_entries = int(len(query_type_data) * keep_percentage)
        filtered_data = pd.concat([filtered_data, query_type_data.sample(n=num_entries)])
    return filtered_data

def main():
    # Process the training and validation datasets
    train_data = process_dataset(dataset_train)
    test_data = process_dataset(dataset_test)


    filtered_test_data = filter_test_dataset(test_data, keep_percentage=0.1)
    test_data = filtered_test_data

    # Save the processed data to CSV files  
    train_data.to_csv('data/babiqa/train_dataset_for_prompting.csv', index=False)
    test_data.to_csv('data/babiqa/test_dataset_for_prompting.csv', index=False)
    
    print("Train dataset shape:", train_data.shape)
    print("Test dataset shape:", test_data.shape)

    print("Data processing complete. Files saved as train_dataset_for_prompting.csv and test_dataset_for_prompting.csv.")
    
    
if __name__ == "__main__":
    main()