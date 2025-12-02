"""
Comprehensive Training Script - Fixed for All Edge Cases
========================================================
Trains the BERT model with comprehensive data covering all statement types
including all the edge cases we've been fixing.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
import json
import numpy as np
from tqdm import tqdm
import os
from datetime import datetime
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from scipy.optimize import minimize_scalar

from bert_classifier import BERTMentalHealthClassifier


class MentalHealthDataset(Dataset):
    """Dataset for mental health classification"""
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
        labels = item['labels']
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Convert labels to tensor
        label_tensor = torch.tensor([
            labels.get(label, 0) for label in self.label_names
        ], dtype=torch.float)
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': label_tensor
        }


def calculate_class_weights(dataset):
    """Calculate class weights for imbalanced data"""
    label_counts = {label: 0 for label in dataset.label_names}
    total = len(dataset)
    
    for item in dataset.data:
        labels = item['labels']
        for label in dataset.label_names:
            if labels.get(label, 0) == 1:
                label_counts[label] += 1
    
    # Calculate weights (inverse frequency)
    weights = []
    for label in dataset.label_names:
        count = label_counts[label]
        if count > 0:
            weight = total / (len(dataset.label_names) * count)
        else:
            weight = 0.0
        weights.append(weight)
    
    return torch.tensor(weights, dtype=torch.float)


def train_epoch(model, dataloader, optimizer, scheduler, device, class_weights):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights.to(device))
    
    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        logits = model(input_ids, attention_mask, apply_temperature=False)
        loss = criterion(logits, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def validate(model, dataloader, device, class_weights):
    """Validate the model"""
    model.eval()
    total_loss = 0
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights.to(device))
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(input_ids, attention_mask, apply_temperature=False)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            
            probs = torch.sigmoid(logits).cpu().numpy()
            labels_np = labels.cpu().numpy()
            
            all_predictions.append(probs)
            all_labels.append(labels_np)
    
    avg_loss = total_loss / len(dataloader)
    
    # Calculate metrics
    all_predictions = np.vstack(all_predictions)
    all_labels = np.vstack(all_labels)
    
    # Find optimal thresholds
    optimal_thresholds = find_optimal_thresholds(all_predictions, all_labels, 
                                                  model.label_names if hasattr(model, 'label_names') 
                                                  else ['neutral', 'stress', 'unsafe_environment', 
                                                        'emotional_distress', 'self_harm_low', 'self_harm_high'])
    
    # Apply thresholds
    predictions = (all_predictions >= np.array([optimal_thresholds[label] for label in optimal_thresholds.keys()])).astype(int)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, predictions, average='macro', zero_division=0
    )
    
    return avg_loss, precision, recall, f1, optimal_thresholds


def find_optimal_thresholds(predictions, labels, label_names):
    """Find optimal thresholds for each label"""
    optimal_thresholds = {}
    
    for i, label in enumerate(label_names):
        def f1_score(threshold):
            pred = (predictions[:, i] >= threshold).astype(int)
            if pred.sum() == 0:
                return 0.0
            precision = (labels[:, i] * pred).sum() / pred.sum()
            recall = (labels[:, i] * pred).sum() / (labels[:, i].sum() + 1e-8)
            if precision + recall == 0:
                return 0.0
            return 2 * precision * recall / (precision + recall)
        
        result = minimize_scalar(lambda t: -f1_score(t), bounds=(0, 1), method='bounded')
        optimal_thresholds[label] = result.x
    
    return optimal_thresholds


def main():
    print("="*80)
    print("COMPREHENSIVE BERT TRAINING - FIXED FOR ALL EDGE CASES")
    print("="*80)
    
    # Configuration
    CONFIG = {
        'batch_size': 16,
        'epochs': 15,
        'learning_rate': 2e-5,
        'warmup_ratio': 0.1,
        'max_length': 128,
        'dropout': 0.3,
        'weight_decay': 0.01,
    }
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n✓ Device: {DEVICE}")
    print(f"✓ Configuration:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    
    # Check if training data exists
    if not os.path.exists('train_data.json'):
        print("\n❌ train_data.json not found!")
        print("Please run: python generate_comprehensive_fixed_training_data.py")
        return
    
    # Load tokenizer
    print("\n" + "="*80)
    print("Loading tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    print("✓ Tokenizer loaded")
    
    # Create datasets
    print("\n" + "="*80)
    print("Loading datasets...")
    train_dataset = MentalHealthDataset('train_data.json', tokenizer, CONFIG['max_length'])
    val_dataset = MentalHealthDataset('val_data.json', tokenizer, CONFIG['max_length'])
    print(f"✓ Training samples: {len(train_dataset)}")
    print(f"✓ Validation samples: {len(val_dataset)}")
    
    # Calculate class weights
    class_weights = calculate_class_weights(train_dataset)
    print(f"✓ Class weights calculated: {class_weights.numpy()}")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    
    # Initialize model
    print("\n" + "="*80)
    print("Initializing BERT model...")
    model = BERTMentalHealthClassifier(n_classes=6, dropout=CONFIG['dropout'])
    model.to(DEVICE)
    model.label_names = train_dataset.label_names
    print("✓ Model initialized")
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    total_steps = len(train_loader) * CONFIG['epochs']
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(CONFIG['warmup_ratio'] * total_steps),
        num_training_steps=total_steps
    )
    
    # Training loop
    best_f1 = 0
    best_epoch = 0
    best_thresholds = None
    
    print("\n" + "="*80)
    print("Starting training...")
    print("="*80)
    
    for epoch in range(CONFIG['epochs']):
        print(f"\nEpoch {epoch + 1}/{CONFIG['epochs']}")
        print("-" * 80)
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, DEVICE, class_weights)
        print(f"Train Loss: {train_loss:.4f}")
        
        # Validate
        val_loss, precision, recall, f1, thresholds = validate(model, val_loader, DEVICE, class_weights)
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Optimal Thresholds: {thresholds}")
        
        # Save best model
        if f1 > best_f1:
            best_f1 = f1
            best_epoch = epoch + 1
            best_thresholds = thresholds
            
            os.makedirs('checkpoints', exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch + 1,
                'f1_score': f1,
                'precision': precision,
                'recall': recall,
                'optimal_thresholds': best_thresholds,
            }, 'checkpoints/best_mental_health_model.pt')
            
            print(f"✓ New best model saved! (F1: {f1:.4f})")
    
    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80)
    print(f"Best F1 Score: {best_f1:.4f} (Epoch {best_epoch})")
    print(f"Best Thresholds: {best_thresholds}")
    print(f"Model saved to: checkpoints/best_mental_health_model.pt")


if __name__ == '__main__':
    main()

