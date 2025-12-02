"""
Hyperparameter Optimization using Optuna
=========================================
Automatically finds best hyperparameters for maximum accuracy
"""

import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from torch.optim import AdamW
import json
import numpy as np
from sklearn.metrics import f1_score

from advanced_bert_classifier import AdvancedBERTClassifier, FocalLoss
from train_advanced_optimized import AugmentedMentalHealthDataset, validate_advanced


def objective(trial):
    """Optuna objective function"""
    
    # Suggest hyperparameters
    config = {
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-6, 5e-5),
        'batch_size': trial.suggest_categorical('batch_size', [8, 16, 32]),
        'dropout': trial.suggest_uniform('dropout', 0.1, 0.5),
        'weight_decay': trial.suggest_loguniform('weight_decay', 1e-6, 1e-2),
        'focal_gamma': trial.suggest_uniform('focal_gamma', 1.0, 3.0),
        'pooling_strategy': trial.suggest_categorical('pooling_strategy', 
                                                      ['attention', 'mean_max', 'cls']),
        'warmup_ratio': trial.suggest_uniform('warmup_ratio', 0.05, 0.2),
    }
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    train_dataset = AugmentedMentalHealthDataset(
        'train_data.json', tokenizer, max_length=128, augment=True
    )
    val_dataset = AugmentedMentalHealthDataset(
        'val_data.json', tokenizer, max_length=128, augment=False
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Create model
    model = AdvancedBERTClassifier(
        model_name='bert-base-uncased',
        n_classes=6,
        dropout=config['dropout'],
        pooling_strategy=config['pooling_strategy'],
        use_residual=True
    )
    model.to(DEVICE)
    
    # Loss and optimizer
    class_weights = torch.ones(6).to(DEVICE)  # Simplified for optimization
    loss_fn = FocalLoss(alpha=class_weights, gamma=config['focal_gamma'])
    
    optimizer = AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    total_steps = len(train_loader) * 3  # Quick training for optimization
    warmup_steps = int(total_steps * config['warmup_ratio'])
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    
    # Quick training (3 epochs for optimization)
    model.train()
    for epoch in range(3):
        for batch in train_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
    
    # Validate
    metrics, _, _ = validate_advanced(model, val_loader, DEVICE, loss_fn)
    
    return metrics['f1_weighted']


def optimize_hyperparameters(n_trials=50):
    """Run hyperparameter optimization"""
    print("="*80)
    print("HYPERPARAMETER OPTIMIZATION")
    print("="*80)
    
    study = optuna.create_study(
        direction='maximize',
        study_name='bert_optimization',
        pruner=optuna.pruners.MedianPruner()
    )
    
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print("\n" + "="*80)
    print("OPTIMIZATION RESULTS")
    print("="*80)
    print(f"Best F1-Weighted: {study.best_value:.4f}")
    print(f"\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Save results
    with open('best_hyperparameters.json', 'w') as f:
        json.dump(study.best_params, f, indent=2)
    
    print(f"\n✓ Best hyperparameters saved to best_hyperparameters.json")
    
    return study.best_params


if __name__ == '__main__':
    best_params = optimize_hyperparameters(n_trials=30)
    print(f"\nUse these parameters in train_advanced_optimized.py for best results!")

