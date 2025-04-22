import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import json

# Load the dataset
dataset = load_dataset("Warrieryes/CotempQA")


# Load the JSON files
equal_data = []
with open('Cotempqa/data/equal.json', 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():  # skip empty lines
            obj = json.loads(line)
            equal_data.append(obj)
            
mix_data = []
with open('Cotempqa/data/mix.json', 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():  # skip empty lines
            obj = json.loads(line)
            mix_data.append(obj)
            
overlap_data = []
with open('Cotempqa/data/overlap.json', 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():  # skip empty lines
            obj = json.loads(line)
            overlap_data.append(obj)
            
during_data = []
with open('Cotempqa/data/during.json', 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():  # skip empty lines
            obj = json.loads(line)
            during_data.append(obj)
            
# Print dataset statistics
print("Dataset Statistics:")
print(dataset)

# Convert to a Pandas DataFrame if necessary
if not isinstance(dataset, pd.DataFrame):
    dataset = pd.DataFrame(dataset)

# Match data from JSONs in the train dataset and create a new DataFrame with labels
dataset_with_labels = pd.DataFrame()
for i in range(len(dataset['train'])):
    index = dataset['train'][i]['index']
    triple_element = dataset['train'][i]['triple_element']
    question = dataset['train'][i]['question']
    facts = dataset['train'][i]['facts']
    answer = dataset['train'][i]['answer']
    
    # Check for the presence of the question and answer in the JSON data
    if any(q['question'] == question and q['answer'] == answer for q in equal_data):
        label = 'equal'
    elif any(q['question'] == question and q['answer'] == answer for q in mix_data):
        label = 'mix'
    elif any(q['question'] == question and q['answer'] == answer for q in overlap_data):
        label = 'overlap'
    elif any(q['question'] == question and q['answer'] == answer for q in during_data):
        label = 'during'
    else:
        label = 'unknown'  # or handle as needed
    
    # Append to the new DataFrame
    dataset_with_labels = pd.concat(
        [dataset_with_labels, pd.DataFrame([{'index': index, 'triple_element': triple_element, 'question': question, 'facts': facts, 'answer': answer, 'label': label}])],
    ignore_index=True
)
    
# Save the train set with labels to a CSV file
dataset_with_labels.to_csv('Cotempqa/data/dataset_with_labels.csv', index=False)

# Split the dataset into train and test sets
train_set, test_set = train_test_split(dataset_with_labels, test_size=0.2, random_state=42)
test_set, val_set = train_test_split(test_set, test_size=0.5, random_state=42)

# Save the train, test, and validation sets to CSV files
train_set.to_csv('Cotempqa/data/train_set.csv', index=False)
test_set.to_csv('Cotempqa/data/test_set.csv', index=False)
val_set.to_csv('Cotempqa/data/val_set.csv', index=False)

# Print the first few rows of the dataset with labels
print("Dataset with Labels:")
print(dataset_with_labels.head())

# Print the size of the train, test, and validation sets
print("Train Set Size:", len(train_set))
print("Test Set Size:", len(test_set))
print("Validation Set Size:", len(val_set))



