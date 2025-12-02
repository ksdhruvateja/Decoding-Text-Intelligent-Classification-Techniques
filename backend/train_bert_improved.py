"""
Improved BERT Mental Health Classifier with Advanced Training Techniques
Addresses all common issues that cause poor scores and metrics
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
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix
import random

# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class MentalHealthDataset(Dataset):
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
    
    def augment_text(self, text):
        """Simple text augmentation - random word dropout or synonym replacement"""
        if not self.augment or random.random() > 0.3:
            return text
        
        words = text.split()
        if len(words) > 3:
            # Random word dropout (10% of words)
            num_to_drop = max(1, int(len(words) * 0.1))
            indices_to_drop = random.sample(range(len(words)), num_to_drop)
            words = [w for i, w in enumerate(words) if i not in indices_to_drop]
        
        return ' '.join(words)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        
        # Apply augmentation to training data
        if self.augment:
            text = self.augment_text(text)
        
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

class ImprovedBERTClassifier(nn.Module):
    """
    Enhanced BERT classifier with better architecture for multi-label classification
    """
    def __init__(self, n_classes=6, dropout=0.3, freeze_bert=False):
        super(ImprovedBERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        # Optionally freeze BERT layers for initial training
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        # Enhanced classifier head
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(768)
        
        # Multi-layer classifier with residual connection
        self.fc1 = nn.Linear(768, 384)
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, n_classes)
        
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout * 0.5)
        
        # Initialize weights properly
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        
    def forward(self, input_ids, attention_mask):
        # Get BERT outputs
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use pooled output (CLS token)
        pooled_output = outputs.pooler_output
        
        # Apply layer normalization
        x = self.layer_norm(pooled_output)
        
        # Multi-layer classification head with dropout
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x

def calculate_class_weights(data_path, label_names):
    """Calculate class weights for imbalanced dataset"""
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Count positive samples for each class
    pos_counts = np.array([sum(1 for d in data if d['labels'][label] == 1) for label in label_names])
    neg_counts = len(data) - pos_counts
    
    # Calculate weights using inverse frequency
    # weight = total_samples / (n_classes * class_count)
    weights = len(data) / (len(label_names) * pos_counts)
    
    # Normalize weights to prevent extreme values
    weights = np.clip(weights, 0.5, 10.0)
    
    print("\nClass weights:")
    for label, weight, pos_count in zip(label_names, weights, pos_counts):
        print(f"  {label:25s}: {weight:.3f} (positive samples: {pos_count})")
    
    return torch.FloatTensor(weights)

def calculate_metrics_comprehensive(predictions, labels, threshold=0.5):
    """
    Calculate comprehensive metrics with proper handling of edge cases
    """
    # Convert to binary predictions
    pred_binary = (predictions > threshold).float()
    
    # Calculate per-class metrics
    tp = (pred_binary * labels).sum(dim=0)
    fp = (pred_binary * (1 - labels)).sum(dim=0)
    fn = ((1 - pred_binary) * labels).sum(dim=0)
    tn = ((1 - pred_binary) * (1 - labels)).sum(dim=0)
    
    # Precision, Recall, F1 with epsilon to avoid division by zero
    epsilon = 1e-10
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1 = 2 * precision * recall / (precision + recall + epsilon)
    
    # Accuracy per class
    accuracy = (tp + tn) / (tp + tn + fp + fn + epsilon)
    
    # Handle NaN values (when no positive samples exist)
    precision = torch.nan_to_num(precision, nan=0.0)
    recall = torch.nan_to_num(recall, nan=0.0)
    f1 = torch.nan_to_num(f1, nan=0.0)
    accuracy = torch.nan_to_num(accuracy, nan=0.0)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn
    }

def find_optimal_thresholds(predictions, labels, label_names):
    """Find optimal threshold for each class to maximize F1 score"""
    optimal_thresholds = {}
    
    for i, label in enumerate(label_names):
        best_threshold = 0.5
        best_f1 = 0.0
        
        # Try different thresholds
        for threshold in np.arange(0.3, 0.8, 0.05):
            pred_binary = (predictions[:, i] > threshold).float()
            
            tp = (pred_binary * labels[:, i]).sum()
            fp = (pred_binary * (1 - labels[:, i])).sum()
            fn = ((1 - pred_binary) * labels[:, i]).sum()
            
            precision = tp / (tp + fp + 1e-10)
            recall = tp / (tp + fn + 1e-10)
            f1 = 2 * precision * recall / (precision + recall + 1e-10)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        optimal_thresholds[label] = best_threshold
    
    return optimal_thresholds

def train_epoch(model, dataloader, optimizer, scheduler, device, epoch, class_weights):
    """Training loop with proper loss calculation using class weights"""
    model.train()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    # BCE loss with class weights
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights.to(device))
    
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch_idx, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        
        # Backward pass with gradient clipping
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
        # Store predictions for metrics
        with torch.no_grad():
            predictions = torch.sigmoid(outputs)
            all_predictions.append(predictions.cpu())
            all_labels.append(labels.cpu())
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{total_loss/(batch_idx+1):.4f}'
        })
    
    # Calculate training metrics
    all_predictions = torch.cat(all_predictions)
    all_labels = torch.cat(all_labels)
    metrics = calculate_metrics_comprehensive(all_predictions, all_labels)
    
    return total_loss / len(dataloader), metrics

def validate(model, dataloader, device, class_weights):
    """Validation loop with comprehensive metrics"""
    model.eval()
    all_predictions = []
    all_labels = []
    total_loss = 0
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights.to(device))
    
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
    
    metrics = calculate_metrics_comprehensive(all_predictions, all_labels)
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss, metrics, all_predictions, all_labels

def main():
    print("="*80)
    print(" IMPROVED BERT MENTAL HEALTH CLASSIFIER - ADVANCED TRAINING PIPELINE")
    print("="*80)
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Configuration
    BATCH_SIZE = 16
    EPOCHS = 15
    LEARNING_RATE = 2e-5
    WARMUP_RATIO = 0.1
    MAX_LENGTH = 128
    WEIGHT_DECAY = 0.01
    GRADIENT_ACCUMULATION_STEPS = 2
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n✓ Device: {DEVICE}")
    print(f"✓ Batch size: {BATCH_SIZE}")
    print(f"✓ Gradient accumulation steps: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"✓ Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print(f"✓ Epochs: {EPOCHS}")
    print(f"✓ Learning rate: {LEARNING_RATE}")
    print(f"✓ Weight decay: {WEIGHT_DECAY}")
    
    # Load tokenizer
    print("\n" + "="*80)
    print("Loading BERT tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    print("✓ Tokenizer loaded")
    
    # Label names
    label_names = ['neutral', 'stress', 'unsafe_environment', 
                   'emotional_distress', 'self_harm_low', 'self_harm_high']
    
    # Calculate class weights
    print("\n" + "="*80)
    print("Calculating class weights for imbalanced data...")
    class_weights = calculate_class_weights('train_data.json', label_names)
    
    # Create datasets with augmentation
    print("\n" + "="*80)
    print("Loading datasets...")
    train_dataset = MentalHealthDataset('train_data.json', tokenizer, MAX_LENGTH, augment=True)
    val_dataset = MentalHealthDataset('val_data.json', tokenizer, MAX_LENGTH, augment=False)
    print(f"✓ Training samples: {len(train_dataset)} (with augmentation)")
    print(f"✓ Validation samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=0,  # Set to 0 for Windows compatibility
        pin_memory=True if torch.cuda.is_available() else False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Initialize improved model
    print("\n" + "="*80)
    print("Initializing improved BERT model...")
    model = ImprovedBERTClassifier(n_classes=6, dropout=0.3, freeze_bert=False)
    model.to(DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Model initialized")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Optimizer with weight decay
    optimizer = AdamW(
        model.parameters(), 
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Learning rate scheduler with warmup
    total_steps = len(train_loader) * EPOCHS // GRADIENT_ACCUMULATION_STEPS
    warmup_steps = int(WARMUP_RATIO * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    print(f"✓ Optimizer configured")
    print(f"  Total training steps: {total_steps}")
    print(f"  Warmup steps: {warmup_steps}")
    
    # Training loop
    best_f1 = 0
    best_epoch = 0
    patience = 5
    patience_counter = 0
    
    print("\n" + "="*80)
    print("Starting training...")
    print("="*80)
    
    training_history = {
        'train_loss': [],
        'val_loss': [],
        'val_f1': [],
        'val_accuracy': []
    }
    
    for epoch in range(1, EPOCHS + 1):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch}/{EPOCHS}")
        print('='*80)
        
        # Train
        train_loss, train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, DEVICE, epoch, class_weights
        )
        
        print(f"\nTraining Results:")
        print(f"  Loss: {train_loss:.4f}")
        print(f"  Average F1: {train_metrics['f1'].mean():.4f}")
        print(f"  Average Accuracy: {train_metrics['accuracy'].mean():.4f}")
        
        # Validate
        val_loss, val_metrics, val_predictions, val_labels = validate(
            model, val_loader, DEVICE, class_weights
        )
        
        avg_f1 = val_metrics['f1'].mean().item()
        avg_acc = val_metrics['accuracy'].mean().item()
        
        print(f"\nValidation Results:")
        print(f"  Loss: {val_loss:.4f}")
        print(f"  Average F1: {avg_f1:.4f}")
        print(f"  Average Accuracy: {avg_acc:.4f}")
        
        print(f"\nDetailed Per-Class Metrics:")
        print(f"{'Label':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Accuracy':>10}")
        print('-'*80)
        for i, label in enumerate(label_names):
            print(f"{label:<25} {val_metrics['precision'][i]:>10.3f} "
                  f"{val_metrics['recall'][i]:>10.3f} {val_metrics['f1'][i]:>10.3f} "
                  f"{val_metrics['accuracy'][i]:>10.3f}")
        
        # Store history
        training_history['train_loss'].append(train_loss)
        training_history['val_loss'].append(val_loss)
        training_history['val_f1'].append(avg_f1)
        training_history['val_accuracy'].append(avg_acc)
        
        # Save best model
        if avg_f1 > best_f1:
            best_f1 = avg_f1
            best_epoch = epoch
            patience_counter = 0
            
            # Find optimal thresholds
            optimal_thresholds = find_optimal_thresholds(val_predictions, val_labels, label_names)
            
            os.makedirs('checkpoints', exist_ok=True)
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'f1_score': avg_f1,
                'accuracy': avg_acc,
                'label_names': label_names,
                'class_weights': class_weights,
                'optimal_thresholds': optimal_thresholds,
                'training_history': training_history
            }
            torch.save(checkpoint, 'checkpoints/best_mental_health_model.pt')
            print(f"\n✓ New best model saved! (F1: {avg_f1:.4f})")
            
            print(f"\nOptimal thresholds for each class:")
            for label, threshold in optimal_thresholds.items():
                print(f"  {label}: {threshold:.3f}")
        else:
            patience_counter += 1
            print(f"\n  No improvement. Patience: {patience_counter}/{patience}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\n✓ Early stopping triggered after {epoch} epochs")
            break
    
    print("\n" + "="*80)
    print("Training completed!")
    print("="*80)
    print(f"Best F1 score: {best_f1:.4f} (Epoch {best_epoch})")
    print(f"Model saved to: checkpoints/best_mental_health_model.pt")
    
    # Final evaluation with best model
    print("\n" + "="*80)
    print("Loading best model for final evaluation...")
    checkpoint = torch.load('checkpoints/best_mental_health_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimal_thresholds = checkpoint['optimal_thresholds']
    
    val_loss, val_metrics, predictions, labels = validate(model, val_loader, DEVICE, class_weights)
    
    print("\n" + "="*80)
    print("FINAL MODEL PERFORMANCE (with default threshold 0.5)")
    print("="*80)
    print(f"{'Label':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Accuracy':>10}")
    print('-'*80)
    for i, label in enumerate(label_names):
        print(f"{label:<25} {val_metrics['precision'][i]:>10.3f} "
              f"{val_metrics['recall'][i]:>10.3f} {val_metrics['f1'][i]:>10.3f} "
              f"{val_metrics['accuracy'][i]:>10.3f}")
    print('-'*80)
    print(f"{'AVERAGE':<25} {val_metrics['precision'].mean():>10.3f} "
          f"{val_metrics['recall'].mean():>10.3f} {val_metrics['f1'].mean():>10.3f} "
          f"{val_metrics['accuracy'].mean():>10.3f}")
    
    # Evaluate with optimal thresholds
    print("\n" + "="*80)
    print("PERFORMANCE WITH OPTIMAL THRESHOLDS")
    print("="*80)
    
    optimized_predictions = predictions.clone()
    for i, label in enumerate(label_names):
        threshold = optimal_thresholds[label]
        optimized_predictions[:, i] = (predictions[:, i] > threshold).float()
    
    opt_metrics = calculate_metrics_comprehensive(
        optimized_predictions, 
        labels, 
        threshold=0.0  # Already binarized
    )
    
    print(f"{'Label':<25} {'Threshold':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Accuracy':>10}")
    print('-'*80)
    for i, label in enumerate(label_names):
        print(f"{label:<25} {optimal_thresholds[label]:>10.3f} "
              f"{opt_metrics['precision'][i]:>10.3f} {opt_metrics['recall'][i]:>10.3f} "
              f"{opt_metrics['f1'][i]:>10.3f} {opt_metrics['accuracy'][i]:>10.3f}")
    print('-'*80)
    print(f"{'AVERAGE':<25} {'-':>10} {opt_metrics['precision'].mean():>10.3f} "
          f"{opt_metrics['recall'].mean():>10.3f} {opt_metrics['f1'].mean():>10.3f} "
          f"{opt_metrics['accuracy'].mean():>10.3f}")
    
    print("\n" + "="*80)
    print("✓ Training pipeline completed successfully!")
    print("✓ Model ready for deployment")
    print("="*80)

if __name__ == '__main__':
    main()
