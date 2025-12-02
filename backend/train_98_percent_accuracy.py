"""
Train for 98%+ Accuracy - Ultimate Training System
==================================================
Uses all advanced techniques:
- Multiple model architectures
- Ensemble learning
- Advanced data augmentation
- Comprehensive validation
- Iterative improvement
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import BertTokenizer, get_cosine_schedule_with_warmup, AutoModel
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
import json
import numpy as np
from tqdm import tqdm
import os
from datetime import datetime
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
import copy
from collections import defaultdict

from advanced_bert_classifier import AdvancedBERTClassifier, FocalLoss


class UltimateDataset(Dataset):
    """Ultimate dataset with advanced augmentation"""
    def __init__(self, data_path, tokenizer, max_length=128, augment=True):
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment = augment
        self.label_names = ['neutral', 'stress', 'unsafe_environment', 
                           'emotional_distress', 'self_harm_low', 'self_harm_high']
    
    def _augment_text(self, text):
        """Advanced text augmentation"""
        if not self.augment or np.random.random() > 0.3:
            return text
        
        # Synonym replacement (simplified)
        synonyms = {
            'terrible': ['awful', 'horrible', 'bad', 'poor'],
            'great': ['excellent', 'wonderful', 'amazing', 'fantastic'],
            'happy': ['joyful', 'pleased', 'delighted', 'thrilled'],
            'sad': ['unhappy', 'down', 'depressed', 'miserable'],
        }
        
        words = text.split()
        for i, word in enumerate(words):
            word_lower = word.lower().strip('.,!?')
            if word_lower in synonyms:
                words[i] = np.random.choice(synonyms[word_lower])
                break
        
        return ' '.join(words)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        
        # Apply augmentation
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


class MultiModelEnsemble(nn.Module):
    """Ensemble of multiple BERT models"""
    def __init__(self, model_configs, n_classes=6):
        super().__init__()
        self.models = nn.ModuleList()
        
        for config in model_configs:
            model = AdvancedBERTClassifier(
                model_name=config['model_name'],
                n_classes=n_classes,
                dropout=config['dropout'],
                pooling_strategy=config['pooling_strategy'],
                use_residual=config.get('use_residual', True)
            )
            self.models.append(model)
        
        # Weighted combination
        self.weights = nn.Parameter(torch.ones(len(model_configs)) / len(model_configs))
    
    def forward(self, input_ids, attention_mask):
        outputs = []
        for model in self.models:
            outputs.append(model(input_ids, attention_mask))
        
        # Weighted average
        weights = torch.softmax(self.weights, dim=0)
        ensemble_output = sum(w * out for w, out in zip(weights, outputs))
        
        return ensemble_output


def calculate_metrics(y_true, y_pred, label_names):
    """Calculate comprehensive metrics"""
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1.mean()
    f1_weighted = np.average(f1, weights=support)
    
    # Per-class metrics
    per_class = {}
    for i, label in enumerate(label_names):
        per_class[label] = {
            'precision': precision[i],
            'recall': recall[i],
            'f1': f1[i],
            'support': support[i]
        }
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'precision_macro': precision.mean(),
        'recall_macro': recall.mean(),
        'per_class': per_class
    }


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


def validate(model, dataloader, device, loss_fn, label_names, threshold=0.5):
    """Validate with comprehensive metrics"""
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
    
    # Apply threshold
    all_predictions = (all_probs >= threshold).astype(int)
    
    # Calculate metrics
    metrics = calculate_metrics(all_labels, all_predictions, label_names)
    metrics['loss'] = total_loss / len(dataloader)
    
    return metrics, all_probs, all_labels


def optimize_threshold_for_accuracy(y_true, y_pred_proba, target_accuracy=0.98):
    """Find threshold that achieves target accuracy"""
    from scipy.optimize import minimize_scalar
    
    def accuracy_at_threshold(threshold):
        y_pred = (y_pred_proba >= threshold).astype(int)
        acc = accuracy_score(y_true, y_pred)
        return -acc  # Negative because we're minimizing
    
    result = minimize_scalar(accuracy_at_threshold, bounds=(0, 1), method='bounded')
    optimal_threshold = result.x
    optimal_accuracy = -result.fun
    
    # If we can't reach target, try per-class optimization
    if optimal_accuracy < target_accuracy:
        n_classes = y_pred_proba.shape[1]
        thresholds = {}
        for i in range(n_classes):
            def f1_at_threshold(t):
                y_pred_binary = (y_pred_proba[:, i] >= t).astype(int)
                y_true_binary = y_true[:, i].astype(int)
                acc = accuracy_score(y_true_binary, y_pred_binary)
                return -acc
            
            res = minimize_scalar(f1_at_threshold, bounds=(0, 1), method='bounded')
            thresholds[i] = res.x
        
        return thresholds, optimal_accuracy
    
    return optimal_threshold, optimal_accuracy


def main():
    print("="*80)
    print("TRAINING FOR 98%+ ACCURACY - ULTIMATE SYSTEM")
    print("="*80)
    
    # Ultimate Configuration
    CONFIG = {
        'model_configs': [
            {'model_name': 'bert-base-uncased', 'dropout': 0.3, 'pooling_strategy': 'attention'},
            {'model_name': 'bert-base-uncased', 'dropout': 0.25, 'pooling_strategy': 'mean_max'},
            # Can add more: roberta-base, deberta-base, etc.
        ],
        'batch_size': 16,
        'epochs': 30,  # More epochs for 98%+
        'learning_rate': 1e-5,  # Lower LR for fine-tuning
        'focal_gamma': 2.5,  # Higher gamma for hard examples
        'weight_decay': 0.01,
        'warmup_ratio': 0.1,
        'use_amp': True,
        'patience': 7,
        'target_accuracy': 0.98,
    }
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n✓ Device: {DEVICE}")
    
    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Load comprehensive data
    print("\nLoading comprehensive training data...")
    train_dataset = UltimateDataset(
        'train_data_comprehensive.json', tokenizer, max_length=128, augment=True
    )
    val_dataset = UltimateDataset(
        'val_data_comprehensive.json', tokenizer, max_length=128, augment=False
    )
    
    print(f"✓ Training samples: {len(train_dataset)}")
    print(f"✓ Validation samples: {len(val_dataset)}")
    
    # Calculate class weights
    label_counts = np.zeros(6)
    for item in train_dataset.data:
        labels = [item['labels'][label] for label in train_dataset.label_names]
        label_counts += np.array(labels)
    
    total = label_counts.sum()
    class_weights = total / (len(train_dataset.label_names) * label_counts + 1e-6)
    class_weights = torch.FloatTensor(class_weights)
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
    
    # Initialize ensemble model
    print("\nInitializing multi-model ensemble...")
    model = MultiModelEnsemble(CONFIG['model_configs'], n_classes=6)
    model.to(DEVICE)
    print("✓ Ensemble model initialized")
    
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
    best_accuracy = 0
    best_epoch = 0
    best_threshold = 0.5
    patience_counter = 0
    label_names = train_dataset.label_names
    
    print("\n" + "="*80)
    print("STARTING TRAINING FOR 98%+ ACCURACY")
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
            model, val_loader, DEVICE, loss_fn, label_names, threshold=best_threshold
        )
        
        # Optimize threshold for target accuracy
        optimal_threshold, optimal_accuracy = optimize_threshold_for_accuracy(
            val_labels, val_probs, target_accuracy=CONFIG['target_accuracy']
        )
        
        if isinstance(optimal_threshold, dict):
            # Per-class thresholds
            print(f"\nPer-class optimal thresholds:")
            for i, label in enumerate(label_names):
                print(f"  {label}: {optimal_threshold[i]:.3f}")
            threshold_used = optimal_threshold
        else:
            threshold_used = optimal_threshold
            print(f"\nOptimal threshold: {optimal_threshold:.3f}")
        
        # Re-validate with optimal threshold
        if isinstance(threshold_used, dict):
            # Use per-class thresholds
            all_predictions = np.zeros_like(val_probs)
            for i in range(val_probs.shape[1]):
                all_predictions[:, i] = (val_probs[:, i] >= threshold_used[i]).astype(int)
            val_metrics_opt = calculate_metrics(val_labels, all_predictions, label_names)
        else:
            val_metrics_opt, _, _ = validate(
                model, val_loader, DEVICE, loss_fn, label_names, threshold=threshold_used
            )
        
        print(f"\nValidation Results:")
        print(f"  Accuracy: {val_metrics_opt['accuracy']:.4f} ({val_metrics_opt['accuracy']*100:.2f}%)")
        print(f"  F1-Macro: {val_metrics_opt['f1_macro']:.4f}")
        print(f"  F1-Weighted: {val_metrics_opt['f1_weighted']:.4f}")
        print(f"\nPer-class F1:")
        for label in label_names:
            f1 = val_metrics_opt['per_class'][label]['f1']
            print(f"  {label:20s}: {f1:.4f}")
        
        # Check if target accuracy reached
        current_accuracy = val_metrics_opt['accuracy']
        if current_accuracy >= CONFIG['target_accuracy']:
            print(f"\n🎉 TARGET ACCURACY REACHED! {current_accuracy*100:.2f}% >= {CONFIG['target_accuracy']*100}%")
        
        # Save best model
        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            best_epoch = epoch
            best_threshold = threshold_used
            patience_counter = 0
            
            os.makedirs('checkpoints', exist_ok=True)
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimal_threshold': best_threshold,
                'accuracy': current_accuracy,
                'f1_weighted': val_metrics_opt['f1_weighted'],
                'f1_macro': val_metrics_opt['f1_macro'],
                'label_names': label_names,
                'config': CONFIG,
                'metrics': val_metrics_opt
            }
            torch.save(checkpoint, 'checkpoints/best_98_percent_model.pt')
            print(f"\n✓ New best model saved! (Accuracy: {current_accuracy*100:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG['patience']:
                print(f"\nEarly stopping triggered")
                break
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Best Accuracy: {best_accuracy*100:.2f}% (Epoch {best_epoch})")
    print(f"Target: {CONFIG['target_accuracy']*100}%")
    
    if best_accuracy >= CONFIG['target_accuracy']:
        print(f"\n🎉 SUCCESS! Achieved {best_accuracy*100:.2f}% accuracy (target: {CONFIG['target_accuracy']*100}%)")
    else:
        print(f"\n⚠ Accuracy: {best_accuracy*100:.2f}% (target: {CONFIG['target_accuracy']*100}%)")
        print("Consider:")
        print("  - Adding more training data")
        print("  - Training for more epochs")
        print("  - Using larger models")
        print("  - Fine-tuning hyperparameters")
    
    print(f"\nModel saved to: checkpoints/best_98_percent_model.pt")
    print("✓ Training complete!")


if __name__ == '__main__':
    main()

