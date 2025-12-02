"""
Advanced Deep Learning Training System
=======================================
Multi-architecture training with state-of-the-art techniques
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import (
    BertTokenizer, BertModel,
    RobertaTokenizer, RobertaModel,
    DebertaTokenizer, DebertaModel,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
import json
import numpy as np
from tqdm import tqdm
import os
from datetime import datetime
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import copy
import sys

from advanced_bert_classifier import FocalLoss, LabelSmoothingBCE


class MultiArchitectureDataset(Dataset):
    """Dataset supporting multiple architectures"""
    def __init__(self, data_path, tokenizer, max_length=128, augment=False):
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment = augment
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


class AdvancedBERTModel(nn.Module):
    """Advanced BERT model with attention pooling"""
    def __init__(self, model_name='bert-base-uncased', n_classes=6, dropout=0.3):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.attention = nn.MultiheadAttention(768, num_heads=8, batch_first=True)
        self.classifier = nn.Linear(768, n_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        
        # Attention pooling
        attn_output, _ = self.attention(sequence_output, sequence_output, sequence_output)
        pooled = attn_output.mean(dim=1)
        
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits


class AdvancedRoBERTaModel(nn.Module):
    """Advanced RoBERTa model"""
    def __init__(self, model_name='roberta-base', n_classes=6, dropout=0.3):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(768, n_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits


class AdvancedDeBERTaModel(nn.Module):
    """Advanced DeBERTa model"""
    def __init__(self, model_name='microsoft/deberta-base', n_classes=6, dropout=0.3):
        super().__init__()
        self.deberta = DebertaModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(768, n_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]  # CLS token
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits


def train_architecture(model, train_loader, val_loader, device, config, model_name):
    """Train a single architecture"""
    print(f"\n{'='*80}")
    print(f"Training {model_name}")
    print(f"{'='*80}")
    
    model.to(device)
    
    # Loss function
    if config['loss'] == 'focal':
        class_weights = torch.ones(6).to(device)
        loss_fn = FocalLoss(alpha=class_weights, gamma=config['focal_gamma'])
    elif config['loss'] == 'label_smoothing':
        loss_fn = LabelSmoothingBCE(smoothing=0.1)
    else:
        loss_fn = nn.BCEWithLogitsLoss()
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Scheduler
    total_steps = len(train_loader) * config['epochs']
    warmup_steps = int(total_steps * config['warmup_ratio'])
    
    if config['scheduler'] == 'cosine':
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )
    else:
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )
    
    # Mixed precision training
    scaler = GradScaler() if config.get('mixed_precision', False) else None
    
    # Training loop
    best_f1 = 0
    best_model_state = None
    patience = 0
    max_patience = config.get('patience', 5)
    
    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch + 1}/{config['epochs']}")
        
        # Training
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc="Training"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            if scaler:
                with autocast():
                    logits = model(input_ids, attention_mask)
                    loss = loss_fn(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(input_ids, attention_mask)
                loss = loss_fn(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            scheduler.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                logits = model(input_ids, attention_mask)
                probs = torch.sigmoid(logits).cpu().numpy()
                preds = (probs > 0.5).astype(int)
                
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted', zero_division=0
        )
        accuracy = accuracy_score(all_labels, all_preds)
        
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Accuracy: {accuracy:.4f}")
        print(f"Val F1: {f1:.4f}")
        print(f"Val Precision: {precision:.4f}")
        print(f"Val Recall: {recall:.4f}")
        
        # Save best model
        if f1 > best_f1:
            best_f1 = f1
            best_model_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
        
        # Early stopping
        if patience >= max_patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    return model, best_f1


def main():
    """Train multiple architectures"""
    print("="*80)
    print("ADVANCED DEEP LEARNING TRAINING - MULTI-ARCHITECTURE")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Configuration
    config = {
        'batch_size': 16,
        'epochs': 15,
        'learning_rate': 2e-5,
        'weight_decay': 0.01,
        'dropout': 0.3,
        'focal_gamma': 2.0,
        'warmup_ratio': 0.1,
        'loss': 'focal',
        'scheduler': 'cosine',
        'mixed_precision': True,
        'patience': 5
    }
    
    # Load data (prefer merged data if available)
    train_path = 'merged_train_data.json' if os.path.exists('merged_train_data.json') else 'train_data.json'
    val_path = 'merged_val_data.json' if os.path.exists('merged_val_data.json') else 'val_data.json'
    
    if not os.path.exists(train_path):
        print(f"Error: {train_path} not found")
        print("Please run data generation first or ensure train_data.json exists")
        return
    
    print(f"Using training data: {train_path}")
    print(f"Using validation data: {val_path}")
    
    # Train BERT
    print("\n" + "="*80)
    print("1. Training BERT")
    print("="*80)
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_dataset = MultiArchitectureDataset(train_path, bert_tokenizer, augment=True)
    val_dataset = MultiArchitectureDataset(val_path, bert_tokenizer, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    bert_model = AdvancedBERTModel()
    bert_model, bert_f1 = train_architecture(
        bert_model, train_loader, val_loader, device, config, "BERT"
    )
    
    # Save BERT model
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(bert_model.state_dict(), 'checkpoints/advanced_bert_model.pt')
    print(f"✓ BERT model saved (F1: {bert_f1:.4f})")
    
    # Train RoBERTa (if data available)
    try:
        print("\n" + "="*80)
        print("2. Training RoBERTa")
        print("="*80)
        roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        train_dataset_roberta = MultiArchitectureDataset(train_path, roberta_tokenizer, augment=True)
        val_dataset_roberta = MultiArchitectureDataset(val_path, roberta_tokenizer, augment=False)
        
        train_loader_roberta = DataLoader(train_dataset_roberta, batch_size=config['batch_size'], shuffle=True)
        val_loader_roberta = DataLoader(val_dataset_roberta, batch_size=config['batch_size'], shuffle=False)
        
        roberta_model = AdvancedRoBERTaModel()
        roberta_model, roberta_f1 = train_architecture(
            roberta_model, train_loader_roberta, val_loader_roberta, device, config, "RoBERTa"
        )
        
        torch.save(roberta_model.state_dict(), 'checkpoints/advanced_roberta_model.pt')
        print(f"✓ RoBERTa model saved (F1: {roberta_f1:.4f})")
    except Exception as e:
        print(f"RoBERTa training skipped: {e}")
    
    print("\n" + "="*80)
    print("✓ ADVANCED DL TRAINING COMPLETE!")
    print("="*80)
    print("\nModels saved to checkpoints/")
    print("Next: Use ensemble training to combine models")


if __name__ == '__main__':
    main()

