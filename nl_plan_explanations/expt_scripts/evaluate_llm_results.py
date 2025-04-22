import os
import sys
import re
import json
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datasets import load_dataset
from sklearn.metrics import classification_report

ds = load_dataset("vanyacohen/CaT-Bench")
STORAGE_DIR = os.path.join(os.getcwd(), "storage")

MODEL_NAME = "deepseek-r1"
TEMPORAL_RELATION_B = "before"
TEMPORAL_RELATION_A = "after"
EXPT_NAME_B = MODEL_NAME + "_eval_cat" + "_" + TEMPORAL_RELATION_B
EXPT_NAME_A = MODEL_NAME + "_eval_cat" + "_" + TEMPORAL_RELATION_A

train_ds, test_ds, eval_ds = ds['train'], ds['test'], ds['validation']

def evaluate_response(response, label):
    predicted = response
    
    # DEP question - hence the label will flip based on temporal relation
    if label == True:
        actual = 'yes'
    # NON-DEP question - hence the label will not flip based on temporal relation
    elif label == False:
        actual = 'no'
    
    pattern = r"\b" + re.escape(actual) + r"\b"
    match = re.search(pattern, predicted, re.IGNORECASE)
    
    if match:
        return 1
    else:
        return 0

model_responses_path_b = os.path.join(STORAGE_DIR, EXPT_NAME_B, "test", "responses.json")
model_responses_path_a = os.path.join(STORAGE_DIR, EXPT_NAME_A, "test", "responses.json")

with open(model_responses_path_b, "r") as f:
    model_responses_b = json.load(f)
    
with open(model_responses_path_a, "r") as f:
    model_responses_a = json.load(f) 

# get metrics for before (half are DEP and half are NON-DEP)

labels = []
preds = []
for i in tqdm(range(len(test_ds))):
    label = test_ds[i]['label']
    content = model_responses_b[i]
    
    gold_label = label
    predicted_label = evaluate_response(content, label)
    
    labels.append(gold_label)
    preds.append(predicted_label)
    
# get metrics for after (half are DEP and half are NON-DEP)
for i in tqdm(range(len(test_ds))):
    label = test_ds[i]['label']
    content = model_responses_a[i]
    # for some models, the response is split into multiple lines, so we only consider the last line which has answer yes/no
    if len(content.splitlines()) > 1:
        content = content.splitlines()[-1]
    
    gold_label = label
    predicted_label = evaluate_response(content, label)
    
    labels.append(gold_label)
    preds.append(predicted_label)

print(classification_report(labels, preds, target_names=['nondependent', 'dependent']))
