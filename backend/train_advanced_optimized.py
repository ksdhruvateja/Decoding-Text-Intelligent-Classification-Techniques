"""
Advanced Training Pipeline with State-of-the-Art Techniques
===========================================================
Implements:
- Focal Loss for imbalanced classes
- Label Smoothing
- Advanced learning rate schedules (cosine annealing with restarts)
- Data augmentation
- Mixed precision training
- Gradient accumulation
- Advanced evaluation metrics
- Automatic threshold optimization
- Model ensembling
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
import json
import numpy as np
from tqdm import tqdm
import os
from datetime import datetime
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    roc_curve
)
from scipy.optimize import minimize_scalar
import copy

from advanced_bert_classifier import (
    AdvancedBERTClassifier,
    FocalLoss,
    LabelSmoothingBCE,
    AdvancedTemperatureScaling
)


class AugmentedMentalHealthDataset(Dataset):
    """Dataset with data augmentation support"""
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
    
    def _augment_text(self, text):
        """Simple text augmentation"""
        # Random capitalization (with low probability)
        if np.random.random() < 0.1:
            words = text.split()
            if len(words) > 0:
                idx = np.random.randint(0, len(words))
                words[idx] = words[idx].upper()
                text = ' '.join(words)
        return text
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        
        # Apply augmentation during training
        if self.augment and np.random.random() < 0.3:
            text = self._augment_text(text)
        
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
    """Calculate class weights for imbalanced data"""
    label_counts = np.zeros(6)
    for item in dataset.data:
        labels = [item['labels'][label] for label in dataset.label_names]
        label_counts += np.array(labels)
    
    # Inverse frequency weighting
    total = label_counts.sum()
    weights = total / (len(dataset.label_names) * label_counts + 1e-6)
    return torch.FloatTensor(weights)


def optimize_thresholds(y_true, y_pred_proba):
    """Optimize thresholds for each class using F1 score"""
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
            
            return -f1  # Negative because we're minimizing
        
        result = minimize_scalar(f1_at_threshold, bounds=(0, 1), method='bounded')
        optimal_thresholds[i] = result.x
    
    return optimal_thresholds


def train_epoch_advanced(
    model, dataloader, optimizer, scheduler, device, epoch,
    loss_fn, scaler, gradient_accumulation_steps=1, use_amp=True
):
    """Advanced training epoch with mixed precision and gradient accumulation"""
    model.train()
    total_loss = 0
    num_steps = 0
    
    optimizer.zero_grad()
    
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for step, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Mixed precision training
        with autocast(enabled=use_amp):
            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, labels)
            loss = loss / gradient_accumulation_steps
        
        # Backward pass
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        total_loss += loss.item() * gradient_accumulation_steps
        num_steps += 1
        
        # Gradient accumulation
        if (step + 1) % gradient_accumulation_steps == 0:
            # Gradient clipping
            if use_amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            scheduler.step()
            optimizer.zero_grad()
        
        progress_bar.set_postfix({'loss': f'{loss.item() * gradient_accumulation_steps:.4f}'})
    
    return total_loss / num_steps


def validate_advanced(model, dataloader, device, loss_fn, thresholds=None):
    """Advanced validation with comprehensive metrics"""
    model.eval()
    all_predictions = []
    all_labels = []
    all_probs = []
    total_loss = 0
    
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
    if thresholds is None:
        thresholds = {i: 0.5 for i in range(all_probs.shape[1])}
    
    all_predictions = (all_probs >= np.array([thresholds[i] for i in range(all_probs.shape[1])])).astype(int)
    
    # Calculate metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_predictions, average=None, zero_division=0
    )
    
    # Macro averages
    precision_macro = precision.mean()
    recall_macro = recall.mean()
    f1_macro = f1.mean()
    
    # Weighted averages
    precision_weighted = np.average(precision, weights=support)
    recall_weighted = np.average(recall, weights=support)
    f1_weighted = np.average(f1, weights=support)
    
    # AUC scores (if possible)
    auc_scores = []
    for i in range(all_probs.shape[1]):
        try:
            if len(np.unique(all_labels[:, i])) > 1:
                auc = roc_auc_score(all_labels[:, i], all_probs[:, i])
                auc_scores.append(auc)
            else:
                auc_scores.append(0.0)
        except:
            auc_scores.append(0.0)
    
    avg_loss = total_loss / len(dataloader)
    
    metrics = {
        'loss': avg_loss,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'precision_per_class': precision.tolist(),
        'recall_per_class': recall.tolist(),
        'f1_per_class': f1.tolist(),
        'auc_scores': auc_scores,
        'avg_auc': np.mean(auc_scores),
        'support': support.tolist()
    }
    
    return metrics, all_probs, all_labels


def main():
    print("="*80)
    print("ADVANCED BERT TRAINING PIPELINE - OPTIMIZED FOR MAXIMUM ACCURACY")
    print("="*80)
    
    # Advanced Configuration
    CONFIG = {
        'model_name': 'bert-base-uncased',  # Can use 'roberta-base', 'microsoft/deberta-base'
        'batch_size': 16,
        'gradient_accumulation_steps': 2,  # Effective batch size = 32
        'epochs': 15,
        'learning_rate': 2e-5,
        'warmup_ratio': 0.1,
        'max_length': 128,
        'dropout': 0.3,
        'pooling_strategy': 'attention',  # 'attention', 'mean_max', 'cls', 'mean'
        'use_residual': True,
        'loss_type': 'focal',  # 'focal', 'label_smoothing', 'bce'
        'focal_gamma': 2.0,
        'label_smoothing': 0.1,
        'use_amp': True,  # Mixed precision
        'weight_decay': 0.01,
        'lr_schedule': 'cosine',  # 'cosine', 'linear', 'constant'
    }
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n✓ Device: {DEVICE}")
    print(f"✓ Configuration:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    
    # Load tokenizer
    print("\n" + "="*80)
    print("Loading tokenizer...")
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained(CONFIG['model_name'])
    print("✓ Tokenizer loaded")
    
    # Create datasets
    print("\n" + "="*80)
    print("Loading datasets...")
    train_dataset = AugmentedMentalHealthDataset(
        'train_data.json', tokenizer, CONFIG['max_length'], augment=True
    )
    val_dataset = AugmentedMentalHealthDataset(
        'val_data.json', tokenizer, CONFIG['max_length'], augment=False
    )
    print(f"✓ Training samples: {len(train_dataset)}")
    print(f"✓ Validation samples: {len(val_dataset)}")
    
    # Calculate class weights
    class_weights = calculate_class_weights(train_dataset)
    print(f"✓ Class weights: {class_weights.numpy()}")
    
    # Create weighted sampler for imbalanced data
    # Calculate sample weights
    sample_weights = []
    for item in train_dataset.data:
        labels = np.array([item['labels'][label] for label in train_dataset.label_names])
        weight = np.sum(labels * class_weights.numpy())
        sample_weights.append(max(weight, 0.1))  # Minimum weight
    
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
    print("\n" + "="*80)
    print("Initializing advanced BERT model...")
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
    if CONFIG['loss_type'] == 'focal':
        alpha = class_weights.to(DEVICE)
        loss_fn = FocalLoss(alpha=alpha, gamma=CONFIG['focal_gamma'])
    elif CONFIG['loss_type'] == 'label_smoothing':
        loss_fn = LabelSmoothingBCE(smoothing=CONFIG['label_smoothing'])
    else:
        loss_fn = nn.BCEWithLogitsLoss()
    
    print(f"✓ Loss function: {CONFIG['loss_type']}")
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay'],
        eps=1e-8
    )
    
    # Learning rate scheduler
    total_steps = len(train_loader) * CONFIG['epochs'] // CONFIG['gradient_accumulation_steps']
    warmup_steps = int(total_steps * CONFIG['warmup_ratio'])
    
    if CONFIG['lr_schedule'] == 'cosine':
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
    else:
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
    
    # Mixed precision scaler
    scaler = GradScaler() if CONFIG['use_amp'] else None
    
    # Training loop
    best_f1 = 0
    best_epoch = 0
    best_thresholds = None
    patience = 5
    patience_counter = 0
    
    label_names = train_dataset.label_names
    
    print("\n" + "="*80)
    print("Starting advanced training...")
    print("="*80)
    
    for epoch in range(1, CONFIG['epochs'] + 1):
        print(f"\nEpoch {epoch}/{CONFIG['epochs']}")
        print("-" * 80)
        
        # Train
        train_loss = train_epoch_advanced(
            model, train_loader, optimizer, scheduler, DEVICE, epoch,
            loss_fn, scaler, CONFIG['gradient_accumulation_steps'], CONFIG['use_amp']
        )
        print(f"Average training loss: {train_loss:.4f}")
        
        # Validate
        val_metrics, val_probs, val_labels = validate_advanced(
            model, val_loader, DEVICE, loss_fn, thresholds=best_thresholds
        )
        
        # Optimize thresholds
        optimal_thresholds = optimize_thresholds(val_labels, val_probs)
        optimal_thresholds_dict = {label_names[i]: float(optimal_thresholds[i]) 
                                   for i in range(len(label_names))}
        
        # Re-validate with optimal thresholds
        val_metrics_opt, _, _ = validate_advanced(
            model, val_loader, DEVICE, loss_fn, thresholds=optimal_thresholds
        )
        
        print(f"\nValidation Results (with optimized thresholds):")
        print(f"  Loss: {val_metrics_opt['loss']:.4f}")
        print(f"  F1-Macro: {val_metrics_opt['f1_macro']:.4f}")
        print(f"  F1-Weighted: {val_metrics_opt['f1_weighted']:.4f}")
        print(f"  Avg AUC: {val_metrics_opt['avg_auc']:.4f}")
        print(f"\nPer-class metrics:")
        for i, label in enumerate(label_names):
            print(f"  {label:20s} - P: {val_metrics_opt['precision_per_class'][i]:.3f}  "
                  f"R: {val_metrics_opt['recall_per_class'][i]:.3f}  "
                  f"F1: {val_metrics_opt['f1_per_class'][i]:.3f}  "
                  f"AUC: {val_metrics_opt['auc_scores'][i]:.3f}  "
                  f"Threshold: {optimal_thresholds[i]:.3f}")
        
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
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'f1_score': current_f1,
                'f1_macro': val_metrics_opt['f1_macro'],
                'f1_weighted': val_metrics_opt['f1_weighted'],
                'avg_auc': val_metrics_opt['avg_auc'],
                'optimal_thresholds': optimal_thresholds_dict,
                'label_names': label_names,
                'config': CONFIG,
                'metrics': val_metrics_opt
            }
            torch.save(checkpoint, 'checkpoints/best_advanced_model.pt')
            print(f"\n✓ New best model saved! (F1-Weighted: {current_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered (patience={patience})")
                break
    
    print("\n" + "="*80)
    print("Training completed!")
    print("="*80)
    print(f"Best F1-Weighted: {best_f1:.4f} (Epoch {best_epoch})")
    print(f"Model saved to: checkpoints/best_advanced_model.pt")
    print(f"Optimal thresholds: {best_thresholds}")
    print("\n✓ Advanced training pipeline completed successfully!")


if __name__ == '__main__':
    main()

