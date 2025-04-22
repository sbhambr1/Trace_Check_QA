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
import wandb
import warnings
warnings.filterwarnings("ignore")

CUDA_DEVICE = 0
NUM_LABELS = 2
NUM_EPOCHS = 10


def train(lr, dropout, batch_size, model_name, max_length):

    device = torch.device(f"cuda:{CUDA_DEVICE}" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    # Set Wandb to offline mode
    wandb.init(mode="offline")

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

    # Move model to device
    model.to(device)

    # Define loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr)

    # Define learning rate scheduler
    num_training_steps = NUM_EPOCHS * len(train_loader)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    # Testing accuracy before training
    print("Testing accuracy before training...")
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

    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Test F1: {test_f1:.4f}")

    print('--' * 50)
    print("Training the model...")

    # Training loop
    progress_bar = tqdm(range(num_training_steps))

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        train_accuracy = 0.0
        for batch in train_loader:
            inputs = {k: v.to(device) for k, v in batch.items()}['input_ids']
            labels = batch['labels'].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            outputs = model(inputs, attention_mask=attention_mask, labels=labels)
            loss = criterion(outputs.logits, torch.argmax(labels, dim=1))
            loss.backward()
            
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            
            train_loss += loss.item()
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            labels = torch.argmax(labels, dim=1)
            train_accuracy += torch.sum(preds == labels).item()
            train_f1 = f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='micro')
            # train_roc_auc = roc_auc_score(labels.cpu().numpy(), preds.cpu().numpy(), average='micro')
            
        train_loss /= len(train_loader)
        train_accuracy /= len(train_loader.dataset)
        
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        print(f"Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}, Training F1: {train_f1:.4f}")
        print('--' * 50)
            
        # Evaluate the model
        if (epoch + 1) % 1 == 0:
            model.eval()
            eval_loss = 0.0
            eval_accuracy = 0.0
            for batch in valid_loader:
                inputs = {k: v.to(device) for k, v in batch.items()}['input_ids']
                labels = batch['labels'].to(device)
                
                with torch.no_grad():
                    outputs = model(inputs, labels=labels)
                    eval_loss += outputs.loss.item()
                    logits = outputs.logits
                    preds = torch.argmax(logits, dim=1)
                    labels = torch.argmax(labels, dim=1)
                    eval_accuracy += torch.sum(preds == labels).item()
                    eval_f1 = f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='micro')
                    eval_roc_auc = roc_auc_score(labels.cpu().numpy(), preds.cpu().numpy(), average='micro')
                    
            eval_loss /= len(valid_loader)
            eval_accuracy /= len(valid_loader.dataset)
            
            print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
            
            print(f"Validation Loss: {eval_loss:.4f}, Validation Accuracy: {eval_accuracy:.4f}, Validation F1: {eval_f1:.4f}")
            
            print('--' * 50)
            
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

    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Test F1: {test_f1:.4f}")

    # Save the model
    MODEL_SAVE_DIR = os.path.join(os.getcwd(), "storage/classification_models")
    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)
    model_path = os.path.join(os.getcwd(), "models", f"{model_name}_batch_{batch_size}_epochs_{NUM_EPOCHS}_lr_{lr}")
    model.save_pretrained(model_path)

    print(f"Model saved at {model_path}")

# Define the sweep agent
def sweep_agent():
    # Initialize wandb
    wandb.init()
    
    # Configure the hyperparameters
    config = wandb.config
    
    # Set the hyperparameters
    lr = config.lr
    dropout = config.dropout
    batch_size = config.batch_size
    model_name = config.model_name
    max_length = config.max_length
    
    # Call the training function
    train(lr, dropout, batch_size, model_name, max_length)

def main():
    
    # Define the sweep configuration
    sweep_config = {
        "method": "random",
        "parameters": {
            "lr": {"values": [1e-5, 5e-5, 1e-4]},
            "dropout": {"values": [True, False]},
            "batch_size": {"values": [8, 16, 32]},
            "model_name": {"values": ['bert-base-uncased', 'roberta-base', 'distilbert-base-uncased']},
            "max_length": {"values": [64, 128, 256]}
        }
    }

    # Initialize the sweep
    sweep_id = wandb.sweep(sweep_config, project="hyperparam-sweep")
    
    # Run the sweep
    wandb.agent(sweep_id, function=sweep_agent)
    
if __name__ == "__main__":
    main()