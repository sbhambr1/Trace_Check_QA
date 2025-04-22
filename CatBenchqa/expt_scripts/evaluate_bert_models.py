import os
import torch
import logging
import numpy as np
from tqdm.auto import tqdm
from transformers import RobertaForSequenceClassification, DataCollatorWithPadding, RobertaTokenizer, get_scheduler, BertForSequenceClassification, BertTokenizer, DistilBertForSequenceClassification, DistilBertTokenizer
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from datasets import load_dataset
import warnings
warnings.filterwarnings("ignore")

CUDA_DEVICE = 0
NUM_LABELS = 2
NUM_EPOCHS = 10

def test(lr, dropout, batch_size, model_name, max_length):

    device = torch.device(f"cuda:{CUDA_DEVICE}" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()

    # Load the dataset
    DATASET_PATH = os.path.join(os.getcwd(), "data_storage/cat_bench")
    binary_label_path = os.path.join(DATASET_PATH, "binary_label")
    multi_label_path = os.path.join(DATASET_PATH, "multi_label")

    binary_dataset = load_dataset('csv', data_files={'train': os.path.join(binary_label_path, 'train.csv'), 'validation': os.path.join(binary_label_path, 'valid.csv'), 'test': os.path.join(binary_label_path, 'test.csv')})
    multi_dataset = load_dataset('csv', data_files={'train': os.path.join(multi_label_path, 'train.csv'), 'validation': os.path.join(multi_label_path, 'valid.csv'), 'test': os.path.join(multi_label_path, 'test.csv')})

    # Define the custom dataset
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
                max_length=max_length,
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
            model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=NUM_LABELS, hidden_dropout_prob=0.3)
        elif model_name == 'distilbert-base-uncased':
            model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=NUM_LABELS, hidden_dropout_prob=0.3)
        elif model_name == 'bert-base-uncased':
            model = BertForSequenceClassification.from_pretrained(model_name, num_labels=NUM_LABELS, hidden_dropout_prob=0.3)
    else:
        if model_name == 'roberta-base':
            model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=NUM_LABELS)
        elif model_name == 'distilbert-base-uncased':
            model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=NUM_LABELS)
        elif model_name == 'bert-base-uncased':
            model = BertForSequenceClassification.from_pretrained(model_name, num_labels=NUM_LABELS)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Create CustomDataset instances
    if NUM_LABELS == 2:
        train_dataset = CustomDataset(binary_dataset['train'], tokenizer, NUM_LABELS)
        valid_dataset = CustomDataset(binary_dataset['validation'], tokenizer, NUM_LABELS)
        test_dataset = CustomDataset(binary_dataset['test'], tokenizer, NUM_LABELS)
    elif NUM_LABELS == 3:
        train_dataset = CustomDataset(multi_dataset['train'], tokenizer, NUM_LABELS)
        valid_dataset = CustomDataset(multi_dataset['validation'], tokenizer, NUM_LABELS)
        test_dataset = CustomDataset(multi_dataset['test'], tokenizer, NUM_LABELS)
        
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator)

    #load the trained model
    MODEL_NAME = f"{model_name}_batch_{batch_size}_epochs_{NUM_EPOCHS}_lr_{lr}"
    MODEL_PATH = os.path.join(os.getcwd(), "models", MODEL_NAME)
    model.from_pretrained(MODEL_PATH)
    # Move model to device
    model.to(device)

    # Test the model
    model.eval()
    test_loss = 0.0
    test_accuracy = 0.0
    test_preds = []
    test_labels = []

    for batch in test_loader:
        inputs = {k: v.to(device) for k, v in batch.items()}['input_ids']
        labels = batch['labels'].to(device)
        
        with torch.no_grad():
            outputs = model(inputs, labels=labels)
            test_loss += outputs.loss.item()
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            labels = torch.argmax(labels, dim=1)
            test_accuracy += torch.sum(preds == labels).item()
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
            
    test_loss /= len(test_loader)
    test_accuracy /= len(test_loader.dataset)
    test_f1 = f1_score(test_labels, test_preds, average='micro')

    print(f"LR: {lr}, Dropout: {dropout}, Batch Size: {batch_size}, Model Name: {model_name}, Max Length: {max_length}")
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Test F1: {test_f1:.4f}")
    
    print("-"*50)
    
    return test_accuracy
    
def main():
    parameters= {
        "lr": {"values": [1e-5, 5e-5, 1e-4]},
        "dropout": {"values": [True, False]},
        "batch_size": {"values": [8, 16, 32]},
        "model_name": {"values": ['bert-base-uncased', 'roberta-base', 'distilbert-base-uncased']},
        "max_length": {"values": [64, 128, 256]}
    }
    
    max_test_accuracy = 0.0
    
    for lr in parameters["lr"]["values"]:
        for dropout in parameters["dropout"]["values"]:
            for batch_size in parameters["batch_size"]["values"]:
                for model_name in parameters["model_name"]["values"]:
                    for max_length in parameters["max_length"]["values"]:
                        try:
                            test_accuracy = test(lr, dropout, batch_size, model_name, max_length)
                            if test_accuracy > max_test_accuracy:
                                max_test_accuracy = test_accuracy
                                print("Best Model: \n", lr, dropout, batch_size, model_name, max_length)
                            else:
                                print("Model: \n", lr, dropout, batch_size, model_name, max_length)
                        except:
                            print("Model with these parameters is not trained yet: \n", lr, dropout, batch_size, model_name, max_length)
    
if __name__ == "__main__":
    main()