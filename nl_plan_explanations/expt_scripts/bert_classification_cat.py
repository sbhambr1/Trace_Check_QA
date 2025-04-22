import os
import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import BertForSequenceClassification, BertTokenizer, DataCollatorWithPadding, get_scheduler

CUDA_DEVICE = 3
NUM_LABELS = 2
MAX_LEN = 128
BATCH_SIZE = 128
NUM_EPOCHS = 5
MODEL_NAME = 'bert-base-uncased'
LR = 1e-4
STORAGE_DIR = os.path.join(os.getcwd(), "models/bert_models")
DATASET_PATH = os.path.join(os.getcwd(), "data_storage/cat_bench")
if not os.path.exists(STORAGE_DIR):
    os.makedirs(STORAGE_DIR)

# Define the custom dataset
class CustomDataset(Dataset):
    def __init__(self, dataset, tokenizer, num_labels=2, max_length=128):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.num_labels = num_labels
        self.max_length = max_length
        self.sentences = []
        self.labels = []
        self.inputs = {}
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
        self.inputs["input_tokens"] = self.tokenizer(sentence1, sentence2, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
        return {
            "input_ids": self.inputs["input_tokens"]["input_ids"].squeeze(0),  # to remove batch dim
            "attention_mask": self.inputs["input_tokens"]["attention_mask"].squeeze(0),
            "token_type_ids": self.inputs["input_tokens"]["token_type_ids"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }
    
def train(dataloader, device, model, optimizer, criterion, lr_scheduler):
    for epoch in range(NUM_EPOCHS):
        for batch in tqdm(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            outputs = model(
                input_ids, 
                attention_mask=attention_mask, 
                token_type_ids=token_type_ids, 
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
        
def test(dataloader, device, model):
    model.eval()
    test_loss = 0.0
    test_accuracy = 0.0
    test_preds = []
    test_labels = []

    for batch in tqdm(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass
        
        
        with torch.no_grad():
            outputs = model(
                input_ids, 
                attention_mask=attention_mask, 
                token_type_ids=token_type_ids, 
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
    
def main():
    
    device = torch.device(f"cuda:{CUDA_DEVICE}" if torch.cuda.is_available() else "cpu")
    
    torch.manual_seed(0)
    torch.cuda.empty_cache()
    
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Load the dataset
    binary_label_path = os.path.join(DATASET_PATH, "binary_label")

    binary_dataset = load_dataset('csv', data_files={'train': os.path.join(binary_label_path, 'train.csv'), 'validation': os.path.join(binary_label_path, 'valid.csv'), 'test': os.path.join(binary_label_path, 'test.csv')})
    
    train_dataset = CustomDataset(binary_dataset['train'], tokenizer, NUM_LABELS)
    valid_dataset = CustomDataset(binary_dataset['validation'], tokenizer, NUM_LABELS)
    test_dataset = CustomDataset(binary_dataset['test'], tokenizer, NUM_LABELS)
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=data_collator)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=data_collator)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=data_collator)
    
    # Move model to device
    model.to(device)

    # Define loss and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = torch.nn.CrossEntropyLoss(reduction='none')

    # Define learning rate scheduler
    num_training_steps = NUM_EPOCHS * len(train_loader)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )
    
    train(train_loader, device, model, optimizer, criterion, lr_scheduler)
    test(test_loader, device, model)
    model_save_name = "cat_bench_" + MODEL_NAME + f"_batch_{BATCH_SIZE}_epochs_{NUM_EPOCHS}_lr_{LR}"
    path = os.path.join(STORAGE_DIR, model_save_name)
    model.save_pretrained(path)
    
if __name__ == '__main__':  
    main()
    
    
    
    