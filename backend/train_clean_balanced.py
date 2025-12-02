"""
Train on Clean, Balanced Data with Proper Thresholds
=====================================================
Trains model with:
- Clean, balanced dataset (no label noise)
- Proper thresholds (high for risk categories)
- Correct label distribution
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
import json
import numpy as np
import random
from tqdm import tqdm
import os
from sklearn.metrics import precision_recall_fscore_support
from scipy.optimize import minimize_scalar

from bert_classifier import BERTMentalHealthClassifier


class CleanDataset(Dataset):
    """Dataset for clean balanced training"""
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
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        label_tensor = torch.tensor([
            labels.get(label, 0) for label in self.label_names
        ], dtype=torch.float)
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': label_tensor
        }


def calculate_class_weights(dataset):
    """Calculate class weights for balanced training"""
    label_counts = {label: 0 for label in dataset.label_names}
    total = len(dataset)
    
    for item in dataset.data:
        labels = item['labels']
        for label in dataset.label_names:
            if labels.get(label, 0) == 1:
                label_counts[label] += 1
    
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
    """Validate with proper thresholds"""
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
    
    all_predictions = np.vstack(all_predictions)
    all_labels = np.vstack(all_labels)
    
    # Find optimal thresholds with HIGH thresholds for risk categories
    optimal_thresholds = find_optimal_thresholds_strict(
        all_predictions, all_labels, 
        ['neutral', 'stress', 'unsafe_environment', 
         'emotional_distress', 'self_harm_low', 'self_harm_high']
    )
    
    # Apply thresholds
    predictions = (all_predictions >= np.array([optimal_thresholds[label] for label in optimal_thresholds.keys()])).astype(int)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, predictions, average='macro', zero_division=0
    )
    
    return avg_loss, precision, recall, f1, optimal_thresholds


def find_optimal_thresholds_strict(predictions, labels, label_names):
    """
    Find optimal thresholds with STRICT requirements for risk categories
    - self_harm_high: minimum 0.75 (very high bar)
    - self_harm_low: minimum 0.70 (high bar)
    - unsafe_environment: minimum 0.70 (high bar)
    - Others: optimize normally
    """
    optimal_thresholds = {}
    
    # Strict minimum thresholds for risk categories
    min_thresholds = {
        'self_harm_high': 0.75,
        'self_harm_low': 0.70,
        'unsafe_environment': 0.70,
        'emotional_distress': 0.55,
        'stress': 0.50,
        'neutral': 0.40,
    }
    
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
        
        # Use strict minimum for risk categories
        min_threshold = min_thresholds.get(label, 0.5)
        
        # Optimize but enforce minimum
        result = minimize_scalar(
            lambda t: -f1_score(t), 
            bounds=(min_threshold, 1.0),  # Start from minimum threshold
            method='bounded'
        )
        optimal_thresholds[label] = max(result.x, min_threshold)  # Ensure minimum
    
    return optimal_thresholds


def main():
    print("="*80)
    print("CLEAN BALANCED TRAINING WITH PROPER THRESHOLDS")
    print("="*80)
    
    # Configuration
    CONFIG = {
        'batch_size': 16,
        'epochs': 20,  # More epochs for clean data
        'learning_rate': 2e-5,
        'warmup_ratio': 0.1,
        'max_length': 128,
        'dropout': 0.3,
        'weight_decay': 0.01,
    }
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n✓ Device: {DEVICE}")
    
    # Check for clean data
    train_file = 'clean_balanced_train.json'
    val_file = 'clean_balanced_val.json'
    
    if not os.path.exists(train_file):
        print(f"\n❌ {train_file} not found!")
        print("Please run: python generate_clean_balanced_data.py")
        return
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    print("✓ Tokenizer loaded")
    
    # Create datasets
    print("\nLoading datasets...")
    train_dataset = CleanDataset(train_file, tokenizer, CONFIG['max_length'])
    val_dataset = CleanDataset(val_file, tokenizer, CONFIG['max_length'])
    print(f"✓ Training samples: {len(train_dataset)}")
    print(f"✓ Validation samples: {len(val_dataset)}")
    
    # Calculate class weights
    class_weights = calculate_class_weights(train_dataset)
    print(f"✓ Class weights: {class_weights.numpy()}")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    
    # Initialize model
    print("\nInitializing BERT model...")
    model = BERTMentalHealthClassifier(n_classes=6, dropout=CONFIG['dropout'])
    model.to(DEVICE)
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
            }, 'checkpoints/best_clean_balanced_model.pt')
            
            print(f"✓ New best model saved! (F1: {f1:.4f})")
    
    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80)
    print(f"Best F1 Score: {best_f1:.4f} (Epoch {best_epoch})")
    print(f"Best Thresholds: {best_thresholds}")
    print(f"\n✓ Model saved to: checkpoints/best_clean_balanced_model.pt")
    print("\nKey Improvements:")
    print("  ✓ Clean, balanced dataset (no label noise)")
    print("  ✓ Strict thresholds (self_harm_high: 0.75+, self_harm_low: 0.70+)")
    print("  ✓ Proper label distribution")
    print("  ✓ More training epochs (20)")


if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    main()

