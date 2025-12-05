"""
Lightweight AI Classifier using PyTorch without transformers dependency
Trains a simple neural network for mental health text classification
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import os
import re
from collections import Counter

# Configuration
CONFIG = {
    'vocab_size': 10000,
    'embedding_dim': 128,
    'hidden_dim': 256,
    'num_classes': 6,
    'batch_size': 32,
    'epochs': 10,
    'learning_rate': 0.001,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

LABEL_NAMES = ['neutral', 'stress', 'unsafe_environment', 
               'emotional_distress', 'self_harm_low', 'self_harm_high']

class Tokenizer:
    """Simple word tokenizer"""
    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
    
    def build_vocab(self, texts):
        """Build vocabulary from texts"""
        # Count word frequencies
        word_counts = Counter()
        for text in texts:
            words = self.tokenize(text)
            word_counts.update(words)
        
        # Add most common words to vocab
        for word, _ in word_counts.most_common(self.vocab_size - 2):
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        
        print(f"Vocabulary size: {len(self.word2idx)}")
    
    def tokenize(self, text):
        """Tokenize text into words"""
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        return text.split()
    
    def encode(self, text, max_length=50):
        """Encode text to indices"""
        words = self.tokenize(text)
        indices = [self.word2idx.get(word, 1) for word in words]
        
        # Pad or truncate
        if len(indices) < max_length:
            indices += [0] * (max_length - len(indices))
        else:
            indices = indices[:max_length]
        
        return indices

class TextDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=50):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        labels = [item['labels'][label] for label in LABEL_NAMES]
        
        # Encode text
        indices = self.tokenizer.encode(text, self.max_length)
        
        return {
            'input_ids': torch.LongTensor(indices),
            'labels': torch.FloatTensor(labels)
        }

class TextClassifier(nn.Module):
    """Simple neural network classifier"""
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, dropout=0.5):
        super(TextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    
    def forward(self, input_ids):
        # Embedding
        embedded = self.embedding(input_ids)
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Use last hidden state
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        
        # Fully connected layers
        x = self.dropout(hidden)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

def train_model():
    print("="*60)
    print("🚀 Lightweight AI Text Classifier Training")
    print("="*60)
    print(f"Device: {CONFIG['device']}")
    
    # Load data
    print("\n📚 Loading training data...")
    data_path = 'training_data.json'
    with open(data_path, 'r', encoding='utf-8') as f:
        all_data = json.load(f)
    
    print(f"Total samples: {len(all_data)}")
    
    # Split data
    split_idx = int(len(all_data) * 0.8)
    train_data = all_data[:split_idx]
    val_data = all_data[split_idx:]
    
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")
    
    # Build tokenizer
    print("\n🔧 Building vocabulary...")
    tokenizer = Tokenizer(vocab_size=CONFIG['vocab_size'])
    tokenizer.build_vocab([item['text'] for item in all_data])
    
    # Create datasets
    train_dataset = TextDataset(train_data, tokenizer)
    val_dataset = TextDataset(val_data, tokenizer)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'])
    
    # Initialize model
    print("\n🤖 Initializing model...")
    model = TextClassifier(
        vocab_size=len(tokenizer.word2idx),
        embedding_dim=CONFIG['embedding_dim'],
        hidden_dim=CONFIG['hidden_dim'],
        num_classes=CONFIG['num_classes']
    )
    model.to(CONFIG['device'])
    
    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    criterion = nn.BCEWithLogitsLoss()
    
    best_val_loss = float('inf')
    
    # Training loop
    for epoch in range(CONFIG['epochs']):
        print(f"\n{'='*60}")
        print(f"📖 Epoch {epoch + 1}/{CONFIG['epochs']}")
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
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
            
            if train_batches % 50 == 0:
                print(f"Batch {train_batches}/{len(train_loader)} - Loss: {loss.item():.4f}")
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(CONFIG['device'])
                labels = batch['labels'].to(CONFIG['device'])
                
                outputs = model(input_ids)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                # Calculate accuracy
                predictions = torch.sigmoid(outputs) > 0.5
                correct += (predictions == labels.bool()).sum().item()
                total += labels.numel()
        
        avg_val_loss = val_loss / len(val_loader)
        accuracy = correct / total
        
        print(f"\n✅ Epoch {epoch + 1} Results:")
        print(f"   Train Loss: {avg_train_loss:.4f}")
        print(f"   Val Loss: {avg_val_loss:.4f}")
        print(f"   Accuracy: {accuracy:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            
            os.makedirs('checkpoints', exist_ok=True)
            
            # Save tokenizer separately as JSON
            tokenizer_data = {
                'word2idx': tokenizer.word2idx,
                'idx2word': tokenizer.idx2word
            }
            
            with open('checkpoints/tokenizer.json', 'w') as f:
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
            
            checkpoint_path = 'checkpoints/lightweight_classifier.pt'
            torch.save(checkpoint, checkpoint_path)
            
            print(f"\n💾 Model saved! Val Loss: {avg_val_loss:.4f}, Acc: {accuracy:.4f}")
    
    print(f"\n{'='*60}")
    print("✅ Training Complete!")
    print(f"{'='*60}")
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print(f"Model saved to: checkpoints/lightweight_classifier.pt")
    
    # Test predictions
    print(f"\n{'='*60}")
    print("🧪 Testing model with sample texts...")
    print(f"{'='*60}")
    
    test_texts = [
        "I'm feeling great today!",
        "I'm so stressed about work",
        "I want to hurt myself",
        "This environment is unsafe",
        "I feel sad and lonely"
    ]
    
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
                    predictions.append(f"{label}: {probs[idx]:.3f}")
            print(f"Predictions: {predictions if predictions else 'No strong predictions'}")
            
            # Show top prediction
            max_idx = probs.argmax()
            print(f"Top prediction: {LABEL_NAMES[max_idx]} ({probs[max_idx]:.3f})")
    
    print(f"\n{'='*60}")
    print("✅ Training pipeline completed successfully!")
    print(f"{'='*60}")

if __name__ == '__main__':
    train_model()
