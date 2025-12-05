"""
Advanced Text Classifier with LSTM and Attention
Trains on comprehensive dataset with all sentiment and mental health categories
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import os
import re
from collections import Counter
import numpy as np

# Configuration
CONFIG = {
    'vocab_size': 20000,
    'embedding_dim': 256,
    'hidden_dim': 256,
    'num_layers': 2,
    'num_classes': 8,
    'batch_size': 16,
    'epochs': 100,
    'learning_rate': 0.003,
    'dropout': 0.2,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

LABEL_NAMES = ['positive', 'negative', 'neutral', 'stress', 'emotional_distress', 
               'self_harm_low', 'self_harm_high', 'unsafe_environment']

class Tokenizer:
    """Advanced tokenizer with special handling"""
    def __init__(self, vocab_size=20000):
        self.vocab_size = vocab_size
        self.word2idx = {'<PAD>': 0, '<UNK>': 1, '<START>': 2, '<END>': 3}
        self.idx2word = {0: '<PAD>', 1: '<UNK>', 2: '<START>', 3: '<END>'}
    
    def build_vocab(self, texts):
        """Build vocabulary from texts"""
        word_counts = Counter()
        for text in texts:
            words = self.tokenize(text)
            word_counts.update(words)
        
        for word, _ in word_counts.most_common(self.vocab_size - 4):
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        
        print(f"Vocabulary size: {len(self.word2idx)}")
    
    def tokenize(self, text):
        """Tokenize text preserving important words"""
        text = text.lower()
        # Keep apostrophes and hyphens
        text = re.sub(r'[^a-z0-9\s\'-]', ' ', text)
        words = text.split()
        return words
    
    def encode(self, text, max_length=100):
        """Encode text to indices"""
        words = self.tokenize(text)
        indices = [2] + [self.word2idx.get(word, 1) for word in words] + [3]
        
        if len(indices) < max_length:
            indices += [0] * (max_length - len(indices))
        else:
            indices = indices[:max_length-1] + [3]
        
        return indices

class TextDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=100):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        labels = [item['labels'][label] for label in LABEL_NAMES]
        
        indices = self.tokenizer.encode(text, self.max_length)
        
        return {
            'input_ids': torch.LongTensor(indices),
            'labels': torch.FloatTensor(labels)
        }

class AttentionLayer(nn.Module):
    """Attention mechanism for better text understanding"""
    def __init__(self, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_dim, 1)
    
    def forward(self, lstm_output):
        # lstm_output: (batch, seq_len, hidden_dim)
        attention_weights = torch.softmax(self.attention(lstm_output), dim=1)
        # Weighted sum
        context = torch.sum(attention_weights * lstm_output, dim=1)
        return context, attention_weights

class AdvancedTextClassifier(nn.Module):
    """Advanced classifier with bidirectional LSTM and attention"""
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, num_classes, dropout=0.3):
        super(AdvancedTextClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers=num_layers,
            batch_first=True, 
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.attention = AttentionLayer(hidden_dim * 2)
        
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, num_classes)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim // 2)
    
    def forward(self, input_ids):
        # Embedding
        embedded = self.embedding(input_ids)
        
        # LSTM
        lstm_out, _ = self.lstm(embedded)
        
        # Attention
        context, _ = self.attention(lstm_out)
        
        # Fully connected layers with layer norm
        x = self.dropout(context)
        x = self.fc1(x)
        x = self.layer_norm1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.layer_norm2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        
        return x

def train_model():
    print("="*60)
    print("Advanced Text Classifier Training")
    print("="*60)
    print(f"Device: {CONFIG['device']}")
    
    # Load data
    print("\nLoading training data...")
    data_path = 'large_training_data.json'
    with open(data_path, 'r', encoding='utf-8') as f:
        all_data = json.load(f)
    
    print(f"Total samples: {len(all_data)}")
    
    # Split data
    split_idx = int(len(all_data) * 0.8)
    train_data = all_data[:split_idx]
    val_data = all_data[split_idx:]
    
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")
    
    # Build tokenizer
    print("\nBuilding vocabulary...")
    tokenizer = Tokenizer(vocab_size=CONFIG['vocab_size'])
    tokenizer.build_vocab([item['text'] for item in all_data])
    
    # Create datasets
    train_dataset = TextDataset(train_data, tokenizer)
    val_dataset = TextDataset(val_data, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'])
    
    # Initialize model
    print("\nInitializing model...")
    model = AdvancedTextClassifier(
        vocab_size=len(tokenizer.word2idx),
        embedding_dim=CONFIG['embedding_dim'],
        hidden_dim=CONFIG['hidden_dim'],
        num_layers=CONFIG['num_layers'],
        num_classes=CONFIG['num_classes'],
        dropout=CONFIG['dropout']
    )
    model.to(CONFIG['device'])
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Calculate class weights for balanced learning
    print("\nCalculating class weights...")
    class_counts = [0] * CONFIG['num_classes']
    for item in all_data:
        for idx, label in enumerate(LABEL_NAMES):
            class_counts[idx] += item['labels'][label]
    
    # Inverse frequency weighting
    total = len(all_data)
    class_weights = [total / (count + 1) for count in class_counts]
    # Normalize
    max_weight = max(class_weights)
    class_weights = [w / max_weight for w in class_weights]
    class_weights = torch.FloatTensor(class_weights).to(CONFIG['device'])
    
    print("Class weights:", {LABEL_NAMES[i]: f"{class_weights[i]:.2f}" for i in range(len(LABEL_NAMES))})
    
    # Optimizer and loss
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Training loop
    for epoch in range(CONFIG['epochs']):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{CONFIG['epochs']}")
        print(f"{'='*60}")
        
        # Training
        model.train()
        train_loss = 0
        train_batches = 0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(CONFIG['device'])
            labels = batch['labels'].to(CONFIG['device'])
            
            optimizer.zero_grad()
            outputs = model(input_ids)
            loss = criterion(outputs, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(CONFIG['device'])
                labels = batch['labels'].to(CONFIG['device'])
                
                outputs = model(input_ids)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                probs = torch.sigmoid(outputs)
                all_preds.append(probs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        
        # Calculate accuracy
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)
        accuracy = ((all_preds > 0.5) == all_labels).mean()
        
        print(f"\nEpoch {epoch + 1} Results:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Per-class accuracy
        for i, label in enumerate(LABEL_NAMES):
            class_acc = ((all_preds[:, i] > 0.5) == all_labels[:, i]).mean()
            print(f"    {label}: {class_acc:.3f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            os.makedirs('checkpoints', exist_ok=True)
            
            # Save tokenizer separately
            tokenizer_data = {
                'word2idx': tokenizer.word2idx,
                'idx2word': tokenizer.idx2word
            }
            with open('checkpoints/advanced_tokenizer.json', 'w') as f:
                json.dump(tokenizer_data, f)
            
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'accuracy': accuracy,
                'label_names': LABEL_NAMES,
                'config': CONFIG
            }
            
            torch.save(checkpoint, 'checkpoints/advanced_classifier.pt')
            print(f"\n  Model saved! (Val Loss: {avg_val_loss:.4f}, Acc: {accuracy:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= 8:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break
    
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print("Model saved to: checkpoints/advanced_classifier.pt")
    
    # Test predictions
    print(f"\n{'='*60}")
    print("Testing model...")
    print(f"{'='*60}")
    
    test_texts = [
        "This movie was fantastic, I'd watch it again anytime",
        "Shipping took forever and the item was damaged",
        "This book is 300 pages long and written in English",
        "The stock market crashed yesterday after tech earnings disappointed",
        "I feel so alone and nobody understands me",
        "I want to kill myself tonight",
    ]
    
    checkpoint = torch.load('checkpoints/advanced_classifier.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    with torch.no_grad():
        for text in test_texts:
            indices = tokenizer.encode(text)
            input_ids = torch.LongTensor([indices]).to(CONFIG['device'])
            
            outputs = model(input_ids)
            probs = torch.sigmoid(outputs).cpu().numpy()[0]
            
            print(f"\nText: {text}")
            predictions = []
            for idx, label in enumerate(LABEL_NAMES):
                if probs[idx] > 0.5:
                    predictions.append(f"{label}: {probs[idx]:.2f}")
            
            if predictions:
                print(f"  Predictions: {', '.join(predictions)}")
            else:
                max_idx = probs.argmax()
                print(f"  Top: {LABEL_NAMES[max_idx]} ({probs[max_idx]:.2f})")
    
    print(f"\n{'='*60}")
    print("Training pipeline completed!")
    print(f"{'='*60}")

if __name__ == '__main__':
    train_model()
