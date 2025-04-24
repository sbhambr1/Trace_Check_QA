import os
import sys
import json
import warnings
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

default_template = 'Answer the question based on the context:\n{fact}\nQuestion: {question} Only return the answer.\n'

reasoning_template = 'The temporal relation between the event in question and the event in context is: {relation}.\n'

# Load the dataset from csv
dataset = pd.read_csv("data/cotempqa/dataset_with_labels.csv")

# Create a new df with columns: 'index', 'question', 'reasoning', 'answer'
df = pd.DataFrame(columns=['index', 'question', 'reasoning', 'answer'])

rows = []
for index, row in dataset.iterrows():
    # Create a new row with the required columns
    new_row = {
        'index': index,
        'question': default_template.format(fact=row['facts'], question=row['question']),
        'reasoning': reasoning_template.format(relation=row['label']),
        'answer': row['answer']
    }
    # Append the new row to the df
    rows.append(new_row)

# Convert the list of rows into a DataFrame
df = pd.DataFrame(rows)

# Save the new df to a csv file
df.to_csv("data/cotempqa/dataset_for_prompting.csv", index=False)
print("Dataset for prompting saved to data/cotempqa/dataset_for_prompting.csv")
