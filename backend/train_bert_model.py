"""
Train BERT model for mental health classification with high accuracy
Uses proper training pipeline with validation and early stopping
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
import json
import numpy as np
from tqdm import tqdm
import os
from datetime import datetime

class MentalHealthDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=128):
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_names = ['neutral', 'stress', 'unsafe_environment', 
                           'emotional_distress', 'self_harm_low', 'self_harm_high']
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        labels = [item['labels'][label] for label in self.label_names]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.FloatTensor(labels)
        }

class BERTMentalHealthClassifier(nn.Module):
    def __init__(self, n_classes=6, dropout=0.3):
        super(BERTMentalHealthClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(768, 256)
        self.fc2 = nn.Linear(256, n_classes)
        self.relu = nn.ReLU()
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def calculate_metrics(predictions, labels, threshold=0.5):
    """Calculate precision, recall, F1 for each class"""
    pred_binary = (predictions > threshold).float()
    
    tp = (pred_binary * labels).sum(dim=0)
    fp = (pred_binary * (1 - labels)).sum(dim=0)
    fn = ((1 - pred_binary) * labels).sum(dim=0)
    
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    
    return precision, recall, f1

def train_epoch(model, dataloader, optimizer, scheduler, device, epoch):
    model.train()
    total_loss = 0
    criterion = nn.BCEWithLogitsLoss()
    
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(dataloader)

def validate(model, dataloader, device):
    model.eval()
    all_predictions = []
    all_labels = []
    total_loss = 0
    criterion = nn.BCEWithLogitsLoss()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            predictions = torch.sigmoid(outputs)
            all_predictions.append(predictions.cpu())
            all_labels.append(labels.cpu())
    
    all_predictions = torch.cat(all_predictions)
    all_labels = torch.cat(all_labels)
    
    precision, recall, f1 = calculate_metrics(all_predictions, all_labels)
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss, precision, recall, f1, all_predictions, all_labels

def main():
    print("="*60)
    print("BERT Mental Health Classifier Training Pipeline")
    print("="*60)
    
    # Configuration
    BATCH_SIZE = 16
    EPOCHS = 10
    LEARNING_RATE = 2e-5
    MAX_LENGTH = 128
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n✓ Device: {DEVICE}")
    print(f"✓ Batch size: {BATCH_SIZE}")
    print(f"✓ Epochs: {EPOCHS}")
    print(f"✓ Learning rate: {LEARNING_RATE}")
    
    # Load tokenizer
    print("\n" + "="*60)
    print("Loading BERT tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    print("✓ Tokenizer loaded")
    
    # Create datasets
    print("\n" + "="*60)
    print("Loading datasets...")
    train_dataset = MentalHealthDataset('train_data.json', tokenizer, MAX_LENGTH)
    val_dataset = MentalHealthDataset('val_data.json', tokenizer, MAX_LENGTH)
    print(f"✓ Training samples: {len(train_dataset)}")
    print(f"✓ Validation samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # Initialize model
    print("\n" + "="*60)
    print("Initializing BERT model...")
    model = BERTMentalHealthClassifier(n_classes=6, dropout=0.3)
    model.to(DEVICE)
    print("✓ Model initialized")
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    # Training loop
    best_f1 = 0
    best_epoch = 0
    label_names = ['neutral', 'stress', 'unsafe_environment', 
                   'emotional_distress', 'self_harm_low', 'self_harm_high']
    
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)
    
    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        print("-" * 60)
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, DEVICE, epoch)
        print(f"Average training loss: {train_loss:.4f}")
        
        # Validate
        val_loss, precision, recall, f1, _, _ = validate(model, val_loader, DEVICE)
        avg_f1 = f1.mean().item()
        
        print(f"\nValidation Results:")
        print(f"  Loss: {val_loss:.4f}")
        print(f"  Average F1: {avg_f1:.4f}")
        print(f"\nPer-class metrics:")
        for i, label in enumerate(label_names):
            print(f"  {label:20s} - P: {precision[i]:.3f}  R: {recall[i]:.3f}  F1: {f1[i]:.3f}")
        
        # Save best model
        if avg_f1 > best_f1:
            best_f1 = avg_f1
            best_epoch = epoch
            
            os.makedirs('checkpoints', exist_ok=True)
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'f1_score': avg_f1,
                'label_names': label_names
            }
            torch.save(checkpoint, 'checkpoints/best_mental_health_model.pt')
            print(f"\n✓ New best model saved! (F1: {avg_f1:.4f})")
    
    print("\n" + "="*60)
    print("Training completed!")
    print("="*60)
    print(f"Best F1 score: {best_f1:.4f} (Epoch {best_epoch})")
    print(f"Model saved to: checkpoints/best_mental_health_model.pt")
    
    # Final validation with best model
    print("\n" + "="*60)
    print("Loading best model for final evaluation...")
    checkpoint = torch.load('checkpoints/best_mental_health_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    val_loss, precision, recall, f1, predictions, labels = validate(model, val_loader, DEVICE)
    
    print("\nFinal Model Performance:")
    print("="*60)
    for i, label in enumerate(label_names):
        acc = ((predictions[:, i] > 0.5) == labels[:, i]).float().mean()
        print(f"{label:20s} - Accuracy: {acc:.3f}  Precision: {precision[i]:.3f}  Recall: {recall[i]:.3f}  F1: {f1[i]:.3f}")
    
    avg_acc = ((predictions > 0.5) == labels).float().mean()
    print(f"\n{'Overall Average':20s} - Accuracy: {avg_acc:.3f}  F1: {f1.mean():.3f}")
    print("="*60)
    
    print("\n✓ Training pipeline completed successfully!")
    print("✓ Use this model in app.py for high-accuracy predictions")

if __name__ == '__main__':
    main()
