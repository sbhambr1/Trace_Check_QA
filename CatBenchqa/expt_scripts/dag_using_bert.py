import sys
import os
import torch
import warnings
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaForSequenceClassification, DataCollatorWithPadding, BertForSequenceClassification, BertTokenizer, RobertaTokenizer, DistilBertForSequenceClassification, DistilBertTokenizer

warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from expt_scripts.dag_evaluation import EvaluateDAG

#args
cuda_device = 2
num_labels = 2
batch_size = 128
model_name = 'roberta-base' # 'bert-base-uncased', 'roberta-base', 'distilbert-base-uncased'
metric_name = "f1"
max_length = 100
lr = 1e-4
num_epochs = 5
dropout = False
test_batch_size = 1

device = torch.device(f"cuda:{cuda_device}" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

# Load the dataset
DATASET_PATH = os.path.join(os.getcwd(), "data_storage/cat_bench")
binary_label_path = os.path.join(DATASET_PATH, "binary_label")
binary_dataset = load_dataset('csv', data_files={'train': os.path.join(binary_label_path, 'train.csv'), 'validation': os.path.join(binary_label_path, 'valid.csv'), 'test': os.path.join(binary_label_path, 'test.csv')})

class CustomDataset(Dataset):
    def __init__(self, dataset, tokenizer, num_labels=2):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.num_labels = num_labels
        self.sentences = []
        self.labels = []
        self._format_dataset()
        
    def _format_dataset(self):
        for data in self.dataset:
            sentence1 = data['sentence1']
            sentence2 = data['sentence2']
            label = data['label']
            self.sentences.append((sentence1, sentence2))
            self.labels.append(label)            
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        sentence1 = self.sentences[idx][0]
        sentence2 = self.sentences[idx][1]
        label = self.labels[idx]
        inputs = self.tokenizer.encode_plus(
            sentence1, 
            sentence2, 
            add_special_tokens=True, 
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}  # Remove batch dimension
        label_tensor = torch.zeros(self.num_labels, dtype=torch.float)
        label_tensor[label] = 1
        inputs["labels"] = label_tensor
        return inputs

# Load the pre-trained model and tokenizer
if model_name == 'roberta-base':
        tokenizer = RobertaTokenizer.from_pretrained(model_name)
elif model_name == 'distilbert-base-uncased':
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
elif model_name == 'bert-base-uncased':
    tokenizer = BertTokenizer.from_pretrained(model_name)
if dropout:
    if model_name == 'roberta-base':
        model = RobertaForSequenceClassification.from_pretrained(model_name, hidden_dropout_prob=0.3)
    elif model_name == 'distilbert-base-uncased':
        model = DistilBertForSequenceClassification.from_pretrained(model_name, hidden_dropout_prob=0.3)
    elif model_name == 'bert-base-uncased':
        model = BertForSequenceClassification.from_pretrained(model_name, hidden_dropout_prob=0.3)
else:
    if model_name == 'roberta-base':
        model = RobertaForSequenceClassification.from_pretrained(model_name)
    elif model_name == 'distilbert-base-uncased':
        model = DistilBertForSequenceClassification.from_pretrained(model_name)
    elif model_name == 'bert-base-uncased':
        model = BertForSequenceClassification.from_pretrained(model_name)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Create CustomDataset instances
train_dataset = CustomDataset(binary_dataset['train'], tokenizer, num_labels)
valid_dataset = CustomDataset(binary_dataset['validation'], tokenizer, num_labels)
test_dataset = CustomDataset(binary_dataset['test'], tokenizer, num_labels)
    
# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator)
test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, collate_fn=data_collator)

#load the trained model
STORAGE_DIR = os.path.join(os.getcwd(), "models/bert_models")
MODEL_NAME = "cat_bench_" + model_name + f"_batch_{batch_size}_epochs_{num_epochs}_lr_{lr}"
MODEL_PATH = os.path.join(STORAGE_DIR, MODEL_NAME+"/model.safetensors")

state_dict = load_file(MODEL_PATH)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

original_csv = pd.read_csv(os.path.join(binary_label_path, 'eval_test.csv'))
result_csv = original_csv.copy()
result_csv['prediction'] = None

test_accuracy = 0.0

# Get predictions from test loader using the model
with torch.no_grad():
    for i, batch in enumerate(tqdm(test_loader, desc="Predicting labels for test data")):
        inputs = {k: v.to(device) for k, v in batch.items()}['input_ids']
        labels = batch['labels'].to(device)
        attention_mask = batch["attention_mask"].to(device)
        
        outputs = model(input_ids=inputs, attention_mask=attention_mask, labels=labels)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1)
        labels = torch.argmax(labels, dim=1)
        test_accuracy += torch.sum(predictions == labels).item()
        
        # Match the correct row based on sentence1 and sentence2 and copy the prediction to the prediction column
        result_csv.loc[i, "prediction"] = predictions.cpu().numpy()[0]
        
test_accuracy /= len(test_loader.dataset)
print(f"[ACC] Test accuracy: {test_accuracy}")

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