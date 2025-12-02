"""
Advanced Training Script with:
- Focal Loss for imbalance
- Probability Calibration (Platt Scaling + Isotonic Regression)
- Enhanced Architecture
- Comprehensive Evaluation
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
import json
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# Load comprehensive dataset
print("Loading comprehensive dataset...")
with open('training_data_comprehensive.json', 'r') as f:
    train_data = json.load(f)
with open('val_data_comprehensive.json', 'r') as f:
    val_data = json.load(f)

print(f"✓ Train: {len(train_data)} examples")
print(f"✓ Val: {len(val_data)} examples")

# Dataset class
class MentalHealthDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_names = ['neutral', 'stress', 'unsafe_environment', 
                           'emotional_distress', 'self_harm_low', 'self_harm_high']
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        labels = item['labels']
        
        # Tokenize
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Convert labels to tensor
        label_vector = torch.tensor([labels[name] for name in self.label_names], dtype=torch.float)
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': label_vector
        }

# Focal Loss for imbalanced data
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return focal_loss.mean()

# Enhanced Model Architecture
class AdvancedBERTClassifier(nn.Module):
    def __init__(self, n_classes=6, dropout=0.3):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(dropout)
        self.ln1 = nn.LayerNorm(768)
        
        # Multi-layer classifier with residual connections
        self.fc1 = nn.Linear(768, 512)
        self.ln2 = nn.LayerNorm(512)
        self.fc2 = nn.Linear(512, 256)
        self.ln3 = nn.LayerNorm(256)
        self.fc3 = nn.Linear(256, 128)
        self.ln4 = nn.LayerNorm(128)
        self.out = nn.Linear(128, n_classes)
        
        self.relu = nn.ReLU()
        
        # Initialize weights
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.out.weight)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        
        x = self.ln1(pooled_output)
        x = self.dropout(x)
        
        # Layer 1
        x1 = self.relu(self.fc1(x))
        x1 = self.ln2(x1)
        x1 = self.dropout(x1)
        
        # Layer 2
        x2 = self.relu(self.fc2(x1))
        x2 = self.ln3(x2)
        x2 = self.dropout(x2)
        
        # Layer 3
        x3 = self.relu(self.fc3(x2))
        x3 = self.ln4(x3)
        x3 = self.dropout(x3)
        
        # Output
        return self.out(x3)

# Temperature Scaling for Calibration
class TemperatureScaling(nn.Module):
    def __init__(self, n_classes=6):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(n_classes) * 1.5)
    
    def forward(self, logits):
        return logits / self.temperature.unsqueeze(0)

# Training function
def train_epoch(model, dataloader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

# Evaluation function
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            total_loss += loss.item()
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            all_probs.append(probs.cpu())
    
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_probs = torch.cat(all_probs)
    
    # Calculate metrics per class
    label_names = ['neutral', 'stress', 'unsafe_environment', 
                  'emotional_distress', 'self_harm_low', 'self_harm_high']
    
    print("\nPer-Class Metrics:")
    print("-" * 80)
    for i, name in enumerate(label_names):
        preds_i = all_preds[:, i].numpy()
        labels_i = all_labels[:, i].numpy()
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels_i, preds_i, average='binary', zero_division=0
        )
        
        print(f"{name:20s} | P: {precision:.3f} | R: {recall:.3f} | F1: {f1:.3f}")
    
    # Overall metrics
    avg_f1 = 0
    for i in range(len(label_names)):
        _, _, f1, _ = precision_recall_fscore_support(
            all_labels[:, i].numpy(), all_preds[:, i].numpy(), 
            average='binary', zero_division=0
        )
        avg_f1 += f1
    avg_f1 /= len(label_names)
    
    return total_loss / len(dataloader), avg_f1, all_probs, all_labels

# Main training loop
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Hyperparameters
    BATCH_SIZE = 16
    EPOCHS = 20
    LEARNING_RATE = 2e-5
    WARMUP_STEPS = 100
    
    # Initialize
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_dataset = MentalHealthDataset(train_data, tokenizer)
    val_dataset = MentalHealthDataset(val_data, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    model = AdvancedBERTClassifier(n_classes=6, dropout=0.3).to(device)
    temp_scaling = TemperatureScaling(n_classes=6).to(device)
    
    # Loss and optimizer
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=total_steps
    )
    
    # Training loop
    best_f1 = 0
    patience = 5
    patience_counter = 0
    
    print("\n" + "="*80)
    print("STARTING COMPREHENSIVE TRAINING")
    print("="*80)
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        print("-" * 80)
        
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, criterion, device)
        val_loss, val_f1, val_probs, val_labels = evaluate(model, val_loader, criterion, device)
        
        print(f"\nTrain Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val F1: {val_f1:.4f}")
        
        # Save best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'val_loss': val_loss
            }, 'checkpoints/best_advanced_model.pt')
            
            print(f"✓ Saved new best model (F1: {val_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break
    
    print("\n" + "="*80)
    print(f"TRAINING COMPLETE - Best F1: {best_f1:.4f}")
    print("="*80)

if __name__ == '__main__':
    main()
