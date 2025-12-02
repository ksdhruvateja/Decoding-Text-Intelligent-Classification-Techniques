"""
Ensemble Training - Train Multiple Models and Combine
=====================================================
Trains multiple diverse models and creates an ensemble for maximum accuracy
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json
import numpy as np
from tqdm import tqdm
import os
import copy
from datetime import datetime

from advanced_bert_classifier import (
    AdvancedBERTClassifier,
    EnsembleClassifier,
    FocalLoss
)
from train_advanced_optimized import (
    AugmentedMentalHealthDataset,
    validate_advanced,
    calculate_class_weights
)


def train_single_model(
    model_name, pooling_strategy, config, train_loader, val_loader, device, model_id
):
    """Train a single model with specific configuration"""
    print(f"\n{'='*80}")
    print(f"Training Model {model_id}: {model_name} with {pooling_strategy} pooling")
    print(f"{'='*80}")
    
    # Create model
    model = AdvancedBERTClassifier(
        model_name=model_name,
        n_classes=6,
        dropout=config['dropout'],
        pooling_strategy=pooling_strategy,
        use_residual=True
    )
    model.to(device)
    
    # Loss and optimizer
    class_weights = torch.ones(6).to(device)
    loss_fn = FocalLoss(alpha=class_weights, gamma=config['focal_gamma'])
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    from transformers import get_cosine_schedule_with_warmup
    total_steps = len(train_loader) * config['epochs']
    warmup_steps = int(total_steps * config['warmup_ratio'])
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    
    # Training loop
    best_f1 = 0
    best_model_state = None
    
    for epoch in range(1, config['epochs'] + 1):
        model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch}'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
        
        # Validate
        metrics, _, _ = validate_advanced(model, val_loader, device, loss_fn)
        
        if metrics['f1_weighted'] > best_f1:
            best_f1 = metrics['f1_weighted']
            best_model_state = copy.deepcopy(model.state_dict())
        
        print(f"Epoch {epoch}: F1-Weighted = {metrics['f1_weighted']:.4f}")
    
    # Load best model
    model.load_state_dict(best_model_state)
    print(f"✓ Model {model_id} trained. Best F1: {best_f1:.4f}")
    
    return model, best_f1


def main():
    print("="*80)
    print("ENSEMBLE TRAINING - MULTIPLE MODELS FOR MAXIMUM ACCURACY")
    print("="*80)
    
    # Configuration
    config = {
        'batch_size': 16,
        'epochs': 10,
        'learning_rate': 2e-5,
        'dropout': 0.3,
        'focal_gamma': 2.0,
        'weight_decay': 0.01,
        'warmup_ratio': 0.1,
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n✓ Device: {device}")
    
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
    
    # Define ensemble configurations
    ensemble_configs = [
        {'model_name': 'bert-base-uncased', 'pooling': 'attention'},
        {'model_name': 'bert-base-uncased', 'pooling': 'mean_max'},
        {'model_name': 'bert-base-uncased', 'pooling': 'cls'},
        # Can add more: RoBERTa, DeBERTa, etc.
    ]
    
    # Train each model
    models = []
    model_scores = []
    
    for i, model_config in enumerate(ensemble_configs):
        model, score = train_single_model(
            model_config['model_name'],
            model_config['pooling'],
            config,
            train_loader,
            val_loader,
            device,
            i + 1
        )
        models.append(model)
        model_scores.append(score)
    
    # Create ensemble
    print(f"\n{'='*80}")
    print("CREATING ENSEMBLE")
    print(f"{'='*80}")
    
    # Weight models by their performance
    weights = np.array(model_scores)
    weights = weights / weights.sum()  # Normalize
    
    ensemble = EnsembleClassifier(models, weights=weights.tolist())
    ensemble.to(device)
    ensemble.eval()
    
    # Evaluate ensemble
    print("\nEvaluating ensemble...")
    metrics, _, _ = validate_advanced(ensemble, val_loader, device, nn.BCEWithLogitsLoss())
    
    print(f"\nEnsemble Performance:")
    print(f"  F1-Macro: {metrics['f1_macro']:.4f}")
    print(f"  F1-Weighted: {metrics['f1_weighted']:.4f}")
    print(f"  Avg AUC: {metrics['avg_auc']:.4f}")
    
    # Save ensemble
    os.makedirs('checkpoints', exist_ok=True)
    ensemble_checkpoint = {
        'model_states': [model.state_dict() for model in models],
        'weights': weights.tolist(),
        'configs': ensemble_configs,
        'metrics': metrics,
        'timestamp': datetime.now().isoformat()
    }
    torch.save(ensemble_checkpoint, 'checkpoints/ensemble_model.pt')
    
    print(f"\n✓ Ensemble saved to checkpoints/ensemble_model.pt")
    print(f"✓ Ensemble weights: {weights}")
    print("\n✓ Ensemble training completed!")


if __name__ == '__main__':
    main()

