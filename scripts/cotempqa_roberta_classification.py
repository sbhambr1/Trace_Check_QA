import os
import json
import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
import time
from torch.utils.data import DataLoader, Dataset
from argparse import ArgumentParser
from transformers import RobertaForSequenceClassification, RobertaTokenizer, DataCollatorWithPadding, get_scheduler
    
LABEL_MAPPING = {
    'during': 0,
    'equal': 1,
    'mix': 2,
    'overlap' : 3
}

LABEL_REVERSE_MAPPING = {
    0: 'during',
    1: 'equal',
    2: 'mix',
    3: 'overlap'
}



# Define the custom dataset
class CustomDataset(Dataset):
    def __init__(self, dataset, tokenizer, num_labels=2, max_length=128):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.num_labels = num_labels
        self.max_length = max_length
        self.questions = []
        self.labels = []
        self.inputs = {}
        self._format_dataset()
        
    def _format_dataset(self):
        for data in self.dataset:
            question = data['question']
            label = data['label']
            label = LABEL_MAPPING[label]
            self.questions.append(question)
            self.labels.append(label)            
    
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        question = self.questions[idx]
        self.inputs["input_tokens"] = self.tokenizer(question, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
        return {
            "input_ids": self.inputs["input_tokens"]["input_ids"].squeeze(0),  # to remove batch dim
            "attention_mask": self.inputs["input_tokens"]["attention_mask"].squeeze(0),
            # "token_type_ids": self.inputs["input_tokens"]["token_type_ids"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }
    
def train(dataloader, device, model, optimizer, criterion, lr_scheduler, num_epochs):
    train_metrics = {
        "loss": [],
        "accuracy": []
    }
    valid_metrics = {
        "loss": [],
        "accuracy": []
    }
    test_metrics = {
        "loss": [],
        "accuracy": [] 
    }
    for epoch in range(num_epochs):
        for batch in tqdm(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            # token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            outputs = model(
                input_ids, 
                attention_mask=attention_mask, 
                # token_type_ids=token_type_ids, 
                labels=labels
            )
            criterion = outputs.loss

            # Backward pass
            optimizer.zero_grad()
            criterion.backward()
            optimizer.step()
            lr_scheduler.step()
        
        print("\n")
        print("--- TRAINING ---")
        print(f"Epoch {epoch + 1}, Loss: {criterion.item()}")
        train_metrics["loss"].append(criterion.item())
        # Calculate accuracy
        preds = torch.argmax(outputs.logits, dim=1)
        train_accuracy = torch.sum(preds == labels).item() / len(labels)
        train_metrics["accuracy"].append(train_accuracy)
        
        # Validation
        if (epoch + 1) % 1 == 0:
            valid_loss, valid_accuracy = validate(dataloader, device, model)
            valid_metrics["loss"].append(valid_loss)
            valid_metrics["accuracy"].append(valid_accuracy)
            print("\n")
            
        # Testing
        if (epoch + 1) % 1 == 0:
            test_loss, test_accuracy = validate(dataloader, device, model)
            test_metrics["loss"].append(test_loss)
            test_metrics["accuracy"].append(test_accuracy)
            print("\n")
            
    return train_metrics, valid_metrics, test_metrics
               
def validate(dataloader, device, model):
    model.eval()
    valid_loss = 0.0
    valid_accuracy = 0.0
    valid_preds = []
    valid_labels = []
    
    for batch in tqdm(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        # token_type_ids = batch["token_type_ids"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass
        with torch.no_grad():
            outputs = model(
                input_ids, 
                attention_mask=attention_mask, 
                # token_type_ids=token_type_ids, 
                labels=labels
            )
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            valid_loss += outputs.loss.item()
            valid_accuracy += torch.sum(preds == labels).item()
            valid_preds.extend(preds.cpu().numpy())
            valid_labels.extend(labels.cpu().numpy())
            
    valid_loss /= len(dataloader)
    valid_accuracy /= len(dataloader.dataset)
    print("\n\n")
    print("--- VALIDATION RESULTS ---")
    print(f"Validation Loss: {valid_loss:.4f}")
    print(f"Validation Accuracy: {valid_accuracy:.4f}")
    print("\n\n")
    
    return valid_loss, valid_accuracy
         
def test(dataloader, device, model):
    model.eval()
    test_accuracy = 0.0
    test_preds = []
    test_labels = []

    for batch in tqdm(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        # token_type_ids = batch["token_type_ids"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass
        
        
        with torch.no_grad():
            outputs = model(
                input_ids, 
                attention_mask=attention_mask, 
                # token_type_ids=token_type_ids, 
                labels=labels
            )
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            test_accuracy += torch.sum(preds == labels).item()
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
            
    test_accuracy /= len(dataloader.dataset)
    print("\n\n")
    print("--- TEST RESULTS ---")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print("\n\n")
    
def main(cuda_device, num_labels, max_len, batch_size, num_epochs, model_name, lr, storage_dir, dataset_path):
    
    device = torch.device(f"cuda:{cuda_device}" if torch.cuda.is_available() else "cpu")
    
    torch.manual_seed(0)
    torch.cuda.empty_cache()
    
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Load the dataset
    dataset = load_dataset('csv', data_files={'train': os.path.join(dataset_path, 'train_set.csv'), 'validation': os.path.join(dataset_path, 'val_set.csv'), 'test': os.path.join(dataset_path, 'test_set.csv')})
    
    train_dataset = CustomDataset(dataset['train'], tokenizer, num_labels, max_length=max_len)
    valid_dataset = CustomDataset(dataset['validation'], tokenizer, num_labels, max_length=max_len)
    test_dataset = CustomDataset(dataset['test'], tokenizer, num_labels, max_length=max_len)
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator)
    
    # Move model to device
    model.to(device)

    # Define loss and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss(reduction='none')

    # Define learning rate scheduler
    num_training_steps = num_epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )
    
    train_metrics, valid_metrics, test_metrics = train(train_loader, device, model, optimizer, criterion, lr_scheduler, num_epochs)
    
    training_config = {
        "model_name": model_name,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "learning_rate": lr,
        "max_length": max_len,
        "num_labels": num_labels,
        "optimizer": "Adam",
        "criterion": "CrossEntropyLoss",
    }
    
    model_save_name = "cotempqa/" + model_name.replace("/", "_") + "/" + time.strftime("%Y%m%d-%H%M%S")
    
    path = os.path.join(storage_dir, model_save_name)
    if not os.path.exists(path):
        os.makedirs(path)
    
    metrics_save_path = os.path.join(path, "metrics.json")
    with open(metrics_save_path, 'w') as f:
        json.dump({
            "training_config": training_config,
            "train": train_metrics,
            "valid": valid_metrics,
            "test": test_metrics
        }, f, indent=4)
    
    # Save the model
    model.save_pretrained(path)
    
    print("Model saved to:", path)
    
if __name__ == '__main__':  
    
    ArgumentParser = ArgumentParser()

    # Set the device to GPU if available
    ArgumentParser.add_argument('--device', type=int, default=0, help='GPU device number')
    ArgumentParser.add_argument('--num_labels', type=int, default=4, help='Number of labels for classification')
    ArgumentParser.add_argument('--max_len', type=int, default=128, help='Maximum length of input sequences')
    ArgumentParser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    ArgumentParser.add_argument('--num_epochs', type=int, default=2, help='Number of epochs for training')
    ArgumentParser.add_argument('--model_name', type=str, default='roberta-base', help='Pretrained model name')
    ArgumentParser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for optimizer')
    ArgumentParser.add_argument('--storage_dir', type=str, default=os.path.join(os.getcwd(), 'models/bert_models'), help='Directory to save the model')

    args = ArgumentParser.parse_args()
    
    dataset_path = os.path.join(os.getcwd(), "data/cotempqa")
    if not os.path.exists(args.storage_dir):
        os.makedirs(args.storage_dir)
    
    main(args.device, args.num_labels, args.max_len, args.batch_size, args.num_epochs, args.model_name, args.lr, args.storage_dir, dataset_path)
    
    
    
    