"""
Train Model for Maximum Accuracy on Any Statement
=================================================
Uses comprehensive data, advanced techniques, and rigorous validation
to ensure accurate classification of ANY statement
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import BertTokenizer, get_cosine_schedule_with_warmup
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
import json
import numpy as np
from tqdm import tqdm
import os
from datetime import datetime
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import copy

from advanced_bert_classifier import AdvancedBERTClassifier, FocalLoss


class ComprehensiveDataset(Dataset):
    """Dataset for comprehensive training"""
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


def calculate_class_weights(dataset):
    """Calculate class weights for balanced training"""
    label_counts = np.zeros(6)
    for item in dataset.data:
        labels = [item['labels'][label] for label in dataset.label_names]
        label_counts += np.array(labels)
    
    total = label_counts.sum()
    weights = total / (len(dataset.label_names) * label_counts + 1e-6)
    return torch.FloatTensor(weights)


def train_epoch(model, dataloader, optimizer, scheduler, device, loss_fn, scaler, use_amp):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc='Training'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        with autocast(enabled=use_amp):
            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, labels)
        
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        scheduler.step()
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def validate(model, dataloader, device, loss_fn, thresholds=None):
    """Validate with comprehensive metrics"""
    model.eval()
    all_predictions = []
    all_labels = []
    all_probs = []
    total_loss = 0
    
    if thresholds is None:
        thresholds = {i: 0.5 for i in range(6)}
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()
            
            probs = torch.sigmoid(outputs)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # Apply thresholds
    threshold_array = np.array([thresholds[i] for i in range(6)])
    all_predictions = (all_probs >= threshold_array).astype(int)
    
    # Calculate metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_predictions, average=None, zero_division=0
    )
    
    accuracy = accuracy_score(all_labels, all_predictions)
    f1_macro = f1.mean()
    f1_weighted = np.average(f1, weights=support)
    
    return {
        'loss': total_loss / len(dataloader),
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'precision_per_class': precision.tolist(),
        'recall_per_class': recall.tolist(),
        'f1_per_class': f1.tolist(),
        'support': support.tolist()
    }, all_probs, all_labels


def optimize_thresholds(y_true, y_pred_proba):
    """Optimize thresholds for maximum F1"""
    from scipy.optimize import minimize_scalar
    
    n_classes = y_pred_proba.shape[1]
    optimal_thresholds = {}
    
    for i in range(n_classes):
        def f1_at_threshold(threshold):
            y_pred_binary = (y_pred_proba[:, i] >= threshold).astype(int)
            y_true_binary = y_true[:, i].astype(int)
            
            tp = np.sum((y_pred_binary == 1) & (y_true_binary == 1))
            fp = np.sum((y_pred_binary == 1) & (y_true_binary == 0))
            fn = np.sum((y_pred_binary == 0) & (y_true_binary == 1))
            
            precision = tp / (tp + fp + 1e-10)
            recall = tp / (tp + fn + 1e-10)
            f1 = 2 * precision * recall / (precision + recall + 1e-10)
            
            return -f1
        
        result = minimize_scalar(f1_at_threshold, bounds=(0, 1), method='bounded')
        optimal_thresholds[i] = result.x
    
    return optimal_thresholds


def main():
    print("="*80)
    print("TRAINING FOR MAXIMUM ACCURACY ON ANY STATEMENT")
    print("="*80)
    
    # Configuration
    CONFIG = {
        'model_name': 'bert-base-uncased',
        'batch_size': 16,
        'epochs': 20,
        'learning_rate': 2e-5,
        'dropout': 0.3,
        'pooling_strategy': 'attention',
        'use_residual': True,
        'focal_gamma': 2.0,
        'weight_decay': 0.01,
        'warmup_ratio': 0.1,
        'use_amp': True,
        'patience': 5,
    }
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n✓ Device: {DEVICE}")
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(CONFIG['model_name'])
    
    # Load comprehensive data
    print("\nLoading comprehensive training data...")
    train_dataset = ComprehensiveDataset(
        'train_data_comprehensive.json', tokenizer, max_length=128, augment=False
    )
    val_dataset = ComprehensiveDataset(
        'val_data_comprehensive.json', tokenizer, max_length=128, augment=False
    )
    
    print(f"✓ Training samples: {len(train_dataset)}")
    print(f"✓ Validation samples: {len(val_dataset)}")
    
    # Calculate class weights
    class_weights = calculate_class_weights(train_dataset)
    print(f"✓ Class weights: {class_weights.numpy()}")
    
    # Create weighted sampler
    sample_weights = []
    for item in train_dataset.data:
        labels = np.array([item['labels'][label] for label in train_dataset.label_names])
        weight = np.sum(labels * class_weights.numpy())
        sample_weights.append(max(weight, 0.1))
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=CONFIG['batch_size'], 
        sampler=sampler,
        num_workers=0
    )
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    
    # Initialize model
    print("\nInitializing advanced model...")
    model = AdvancedBERTClassifier(
        model_name=CONFIG['model_name'],
        n_classes=6,
        dropout=CONFIG['dropout'],
        pooling_strategy=CONFIG['pooling_strategy'],
        use_residual=CONFIG['use_residual']
    )
    model.to(DEVICE)
    print("✓ Model initialized")
    
    # Loss function
    alpha = class_weights.to(DEVICE)
    loss_fn = FocalLoss(alpha=alpha, gamma=CONFIG['focal_gamma'])
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay'],
        eps=1e-8
    )
    
    # Scheduler
    total_steps = len(train_loader) * CONFIG['epochs']
    warmup_steps = int(total_steps * CONFIG['warmup_ratio'])
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    
    # Mixed precision
    scaler = GradScaler() if CONFIG['use_amp'] else None
    
    # Training loop
    best_f1 = 0
    best_epoch = 0
    best_thresholds = None
    patience_counter = 0
    label_names = train_dataset.label_names
    
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80)
    
    for epoch in range(1, CONFIG['epochs'] + 1):
        print(f"\nEpoch {epoch}/{CONFIG['epochs']}")
        print("-" * 80)
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, DEVICE, loss_fn, scaler, CONFIG['use_amp']
        )
        print(f"Training loss: {train_loss:.4f}")
        
        # Validate
        val_metrics, val_probs, val_labels = validate(
            model, val_loader, DEVICE, loss_fn, thresholds=best_thresholds
        )
        
        # Optimize thresholds
        optimal_thresholds = optimize_thresholds(val_labels, val_probs)
        optimal_thresholds_dict = {
            label_names[i]: float(optimal_thresholds[i]) 
            for i in range(len(label_names))
        }
        
        # Re-validate with optimal thresholds
        val_metrics_opt, _, _ = validate(
            model, val_loader, DEVICE, loss_fn, thresholds=optimal_thresholds
        )
        
        print(f"\nValidation Results:")
        print(f"  Accuracy: {val_metrics_opt['accuracy']:.4f}")
        print(f"  F1-Macro: {val_metrics_opt['f1_macro']:.4f}")
        print(f"  F1-Weighted: {val_metrics_opt['f1_weighted']:.4f}")
        print(f"\nPer-class F1:")
        for i, label in enumerate(label_names):
            print(f"  {label:20s}: {val_metrics_opt['f1_per_class'][i]:.4f} "
                  f"(threshold: {optimal_thresholds[i]:.3f})")
        
        # Save best model
        current_f1 = val_metrics_opt['f1_weighted']
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_epoch = epoch
            best_thresholds = optimal_thresholds
            patience_counter = 0
            
            os.makedirs('checkpoints', exist_ok=True)
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimal_thresholds': optimal_thresholds_dict,
                'f1_weighted': current_f1,
                'f1_macro': val_metrics_opt['f1_macro'],
                'accuracy': val_metrics_opt['accuracy'],
                'label_names': label_names,
                'config': CONFIG,
                'metrics': val_metrics_opt
            }
            torch.save(checkpoint, 'checkpoints/best_maximum_accuracy_model.pt')
            print(f"\n✓ New best model saved! (F1-Weighted: {current_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG['patience']:
                print(f"\nEarly stopping triggered")
                break
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Best F1-Weighted: {best_f1:.4f} (Epoch {best_epoch})")
    print(f"Model saved to: checkpoints/best_maximum_accuracy_model.pt")
    print(f"Optimal thresholds: {optimal_thresholds_dict}")
    print("\n✓ Model trained for maximum accuracy on any statement!")


if __name__ == '__main__':
    main()

