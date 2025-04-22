import sys
import os
import torch
import re
import json
import warnings
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from expt_scripts.dag_evaluation import EvaluateDAG
from expt_scripts.cat_bench_prompt import PromptGenerator
warnings.filterwarnings("ignore")
torch.cuda.empty_cache()

ds = load_dataset("vanyacohen/CaT-Bench")
cat_bench_prompt = PromptGenerator()

STORAGE_DIR = os.path.join(os.getcwd(), "storage")

#args
MODEL_NAME = "r1"
TEMPORAL_RELATION_B = "before"
EXPT_NAME_B = MODEL_NAME + "_eval_cat" + "_" + TEMPORAL_RELATION_B

train_ds, test_ds, eval_ds = ds['train'], ds['test'], ds['validation']
# Load the dataset
DATASET_PATH = os.path.join(os.getcwd(), "data_storage/cat_bench")
binary_label_path = os.path.join(DATASET_PATH, "binary_label")
# binary_dataset = load_dataset('csv', data_files={'train': os.path.join(binary_label_path, 'train.csv'), 'validation': os.path.join(binary_label_path, 'valid.csv'), 'test': os.path.join(binary_label_path, 'test.csv')})

original_csv = pd.read_csv(os.path.join(binary_label_path, 'eval_test.csv'))
result_csv = original_csv.copy()
result_csv['prediction'] = None

test_accuracy = 0.0

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
with open(model_responses_path_b, "r") as f:
    model_responses_b = json.load(f)

if MODEL_NAME == 'r1':
    model_responses_path_b2 = os.path.join(STORAGE_DIR, "r1_eval_cat_before_second_half", "test", "responses.json")
    with open(model_responses_path_b2, "r") as f:
        model_responses_b2 = json.load(f)
        model_responses_b.extend(model_responses_b2)  # Combine both model_responses jsons together in the given order
    
labels = []
preds = []
for i in tqdm(range(len(test_ds))):
    label = test_ds[i]['label']
    content = model_responses_b[i]
    
    gold_label = label
    predicted_label = evaluate_response(content, label)
    
    labels.append(gold_label)
    preds.append(predicted_label)
    
    result_csv.loc[i, "prediction"] = predicted_label

# Calculate the test accuracy
test_accuracy = sum(preds) / len(preds)

print("[ACC] Test accuracy: ", test_accuracy)

# Save the result csv
CSV_SAVE_PATH = os.path.join(os.getcwd(), "storage/classification_models")
if not os.path.exists(CSV_SAVE_PATH):
    os.makedirs(CSV_SAVE_PATH)
CSV_SAVE_NAME = f"{MODEL_NAME}_result.csv"
result_csv.to_csv(os.path.join(CSV_SAVE_PATH, CSV_SAVE_NAME), index=False)
print(f"[INFO] Result saved at {os.path.join(CSV_SAVE_PATH, CSV_SAVE_NAME)}")

# Evaluate the DAG  
evaluator = EvaluateDAG(result_csv)
evaluator.get_dag_accuracy()