"""
Train BERT-based text classifier with proper configuration
Using clean single-label training data
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from transformers.optimization import get_linear_schedule_with_warmup
from torch.optim import AdamW
import json
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import os

# Configuration
CONFIG = {
    'model_name': 'bert-base-uncased',
    'max_length': 128,
    'batch_size': 16,
    'epochs': 10,
    'learning_rate': 2e-5,
    'warmup_steps': 100,
    'weight_decay': 0.01,
    'dropout': 0.3,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# Categories
CATEGORIES = [
    'positive', 'negative', 'neutral', 'stress', 'emotional_distress',
    'self_harm_low', 'self_harm_high', 'unsafe_environment'
]

class TextDataset(Dataset):
    """Dataset for text classification"""
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

class BertClassifier(nn.Module):
    """BERT-based text classifier"""
    def __init__(self, n_classes, dropout=0.3):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(CONFIG['model_name'])
        self.drop = nn.Dropout(p=dropout)
        self.fc = nn.Linear(self.bert.config.hidden_size, n_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs.pooler_output
        output = self.drop(pooled_output)
        return self.fc(output)

def load_data(filename):
    """Load training data"""
    # Resolve path relative to this file for stable execution
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = filename
    if not os.path.exists(path):
        candidate = os.path.join(base_dir, filename)
        if os.path.exists(candidate):
            path = candidate
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    texts = [d['text'] for d in data]
    labels = [CATEGORIES.index(d['category']) for d in data]
    
    return texts, labels

def train_epoch(model, data_loader, optimizer, device, scheduler):
    """Train for one epoch"""
    model.train()
    losses = []
    correct_predictions = 0
    
    for batch in tqdm(data_loader, desc='Training'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        
        _, preds = torch.max(outputs, dim=1)
        correct_predictions += torch.sum(preds == labels)
        
        losses.append(loss.item())
        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    
    return correct_predictions.double() / len(data_loader.dataset), np.mean(losses)

def eval_model(model, data_loader, device):
    """Evaluate model"""
    model.eval()
    losses = []
    correct_predictions = 0
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            
            _, preds = torch.max(outputs, dim=1)
            
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())
    
    return (correct_predictions.double() / len(data_loader.dataset), 
            np.mean(losses), 
            predictions, 
            true_labels)

def main():
    """Main training function"""
    print("="*70)
    print("BERT TEXT CLASSIFIER TRAINING")
    print("="*70)
    print(f"\nConfiguration:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    
    # Load data
    print(f"\nLoading training data...")
    texts, labels = load_data('clean_training_data.json')
    print(f"✅ Loaded {len(texts)} training examples")
    
    # Split data (80% train, 20% validation)
    split_idx = int(0.8 * len(texts))
    train_texts, train_labels = texts[:split_idx], labels[:split_idx]
    val_texts, val_labels = texts[split_idx:], labels[split_idx:]
    
    print(f"  Training: {len(train_texts)} samples")
    print(f"  Validation: {len(val_texts)} samples")
    
    # Initialize tokenizer
    print(f"\nInitializing tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(CONFIG['model_name'])
    
    # Create datasets
    train_dataset = TextDataset(train_texts, train_labels, tokenizer, CONFIG['max_length'])
    val_dataset = TextDataset(val_texts, val_labels, tokenizer, CONFIG['max_length'])
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'])
    
    # Initialize model
    print(f"\nInitializing BERT model...")
    device = torch.device(CONFIG['device'])
    model = BertClassifier(n_classes=len(CATEGORIES), dropout=CONFIG['dropout'])
    model = model.to(device)
    
    # Initialize optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    total_steps = len(train_loader) * CONFIG['epochs']
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=CONFIG['warmup_steps'],
        num_training_steps=total_steps
    )
    
    # Training loop
    print(f"\n{'='*70}")
    print("STARTING TRAINING")
    print(f"{'='*70}\n")
    
    best_accuracy = 0
    best_model = None
    
    for epoch in range(CONFIG['epochs']):
        print(f"\nEpoch {epoch + 1}/{CONFIG['epochs']}")
        print("-" * 70)
        
        train_acc, train_loss = train_epoch(model, train_loader, optimizer, device, scheduler)
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        
        val_acc, val_loss, preds, labels = eval_model(model, val_loader, device)
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            best_model = model.state_dict()
            print(f"✅ New best model! Accuracy: {val_acc:.4f}")
    
    # Save best model
    os.makedirs('checkpoints', exist_ok=True)
    torch.save({
        'model_state_dict': best_model,
        'categories': CATEGORIES,
        'config': CONFIG
    }, 'checkpoints/bert_classifier_best.pt')
    
    # Save tokenizer
    tokenizer.save_pretrained('checkpoints/bert_tokenizer')
    
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"\n✅ Best validation accuracy: {best_accuracy:.4f}")
    print(f"✅ Model saved to: checkpoints/bert_classifier_best.pt")
    print(f"✅ Tokenizer saved to: checkpoints/bert_tokenizer")
    
    # Load best model for final evaluation
    model.load_state_dict(best_model)
    val_acc, val_loss, preds, true_labels = eval_model(model, val_loader, device)
    
    print(f"\n{'='*70}")
    print("FINAL EVALUATION")
    print(f"{'='*70}\n")
    
    # Classification report
    print("Classification Report:")
    print(classification_report(true_labels, preds, target_names=CATEGORIES, zero_division=0))
    
    # Per-class accuracy
    print("\nPer-Class Accuracy:")
    for i, cat in enumerate(CATEGORIES):
        cat_mask = [l == i for l in true_labels]
        if sum(cat_mask) > 0:
            cat_acc = sum([p == l for p, l in zip(preds, true_labels) if l == i]) / sum(cat_mask)
            print(f"  {cat}: {cat_acc:.4f}")

if __name__ == '__main__':
    main()
