import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import json

# Load the dataset
dataset = load_dataset("Warrieryes/CotempQA")
reasoning_template = 'The temporal relation between the event in question and the event in context is: {relation}.\n'

# Load the JSON files
equal_data = []
with open('Cotempqa/data/equal.json', 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():  # skip empty lines
            obj = json.loads(line)
            obj['reasoning_trace'] = reasoning_template.format(relation='equal')
            equal_data.append(obj)

# Save the new equal.json file
with open('data/cotempqa/equal_with_reasoning.json', 'w', encoding='utf-8') as f:
    for obj in equal_data:
        json.dump(obj, f)
        f.write('\n')

mix_data = []
with open('Cotempqa/data/mix.json', 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():  # skip empty lines
            obj = json.loads(line)
            obj['reasoning_trace'] = reasoning_template.format(relation='mix')
            mix_data.append(obj)

# Save the new mix.json file
with open('data/cotempqa/mix_with_reasoning.json', 'w', encoding='utf-8') as f:
    for obj in mix_data:
        json.dump(obj, f)
        f.write('\n')

overlap_data = []
with open('Cotempqa/data/overlap.json', 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():  # skip empty lines
            obj = json.loads(line)
            obj['reasoning_trace'] = reasoning_template.format(relation='overlap')
            overlap_data.append(obj)

# Save the new overlap.json file
with open('data/cotempqa/overlap_with_reasoning.json', 'w', encoding='utf-8') as f:
    for obj in overlap_data:
        json.dump(obj, f)
        f.write('\n')

during_data = []
with open('Cotempqa/data/during.json', 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():  # skip empty lines
            obj = json.loads(line)
            obj['reasoning_trace'] = reasoning_template.format(relation='during')
            during_data.append(obj)

# Save the new during.json file
with open('data/cotempqa/during_with_reasoning.json', 'w', encoding='utf-8') as f:
    for obj in during_data:
        json.dump(obj, f)
        f.write('\n')
            
print("Data processing complete. New files with reasoning traces have been saved.")

