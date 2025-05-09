import os
import sys
import json
import warnings
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import ast
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt_tab')

warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

default_template = 'Answer the question based on the context:\n{facts}\nQuestion: {question}\nOnly return the answer.\n'

reasoning_template = 'The category of the question represents what is being asked in that question. The category of the question asked is: {query_type}.\n'

def get_word_overlaps(passage, answer):
    """
    Get the number of word overlaps between the passage and the answer.
    
    Args:
        passage (str): The passage text.
        answer (str): The answer text.
        
    Returns:
        int: The number of word overlaps.
    """
    passage_words = set(word_tokenize(passage.lower()))
    answer_words = set(word_tokenize(answer.lower()))
    return len(passage_words.intersection(answer_words))

def get_answer_passage(passages, answer):
    """
    Get the passage that contains the answer.
    
    Args:
        passages (list): List of passages.
        answer (str): The answer to find in the passages.
        
    Returns:
        str: The passage that contains the answer.
    """
    max_overlap = 0
    best_passage = None
    for passage in passages:
        overlap = get_word_overlaps(passage, answer)
        if overlap > max_overlap:
            max_overlap = overlap
            best_passage = passage
    return best_passage
    

# Load the dataset from csv
dataset_train = pd.read_csv("data/marcoqa/train.csv")
dataset_val = pd.read_csv("data/marcoqa/validation.csv")

train_rows = []
for index, row in dataset_train.iterrows():
    # Create a new row with the required columns
    passage_text_arr = ast.literal_eval(row['passages'].split("'passage_text': array(")[1].split("dtype=object)")[0][:-8])
    answer = ast.literal_eval(row['answers'])[0]
    # Get the passage that contains the answer
    if 'no answer present' in answer.lower():
        ans_passage = answer
    else:
        ans_passage = get_answer_passage(passage_text_arr, answer)
    
    new_row = {
        'answers': answer,
        'passages': passage_text_arr,
        'query': default_template.format(facts=passage_text_arr, question=row['query']),
        'question': row['query'],
        'query_type': row['query_type'],
        'reasoning': reasoning_template.format(query_type=row['query_type']),
        'answer_passage': ans_passage
    }
    # Append the new row to the df
    train_rows.append(new_row)
    
val_rows = []    
for index, row in dataset_val.iterrows():
    # Create a new row with the required columns
    passage_text_arr = ast.literal_eval(row['passages'].split("'passage_text': array(")[1].split("dtype=object)")[0][:-8])
    answer = ast.literal_eval(row['answers'])[0]
    # Get the passage that contains the answer
    if 'no answer present' in answer.lower():
        ans_passage = answer
    else:
        ans_passage = get_answer_passage(passage_text_arr, answer)
        
    new_row = {
        'answers': row['answers'],
        'passages': passage_text_arr,
        'query': default_template.format(facts=passage_text_arr, question=row['query']),
        'question': row['query'],
        'query_type': row['query_type'],
        'reasoning': reasoning_template.format(query_type=row['query_type']),
        'answer_passage': row['answers']
    }
    # Append the new row to the df
    val_rows.append(new_row)

# Convert the list of rows into a DataFrame
df_train = pd.DataFrame(train_rows)
df_val = pd.DataFrame(val_rows)

# Save the new df to a csv file
df_train.to_csv('data/marcoqa/train_dataset_for_prompting.csv', index=False)
df_val.to_csv('data/marcoqa/val_dataset_for_prompting.csv', index=False)

print("train_dataset_for_prompting.csv and val_dataset_for_prompting.csv created successfully.")
