"""
Simplified BERT Training for Mental Health Classification
Uses only essential dependencies to avoid import errors
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AdamW
import json
import os
from tqdm import tqdm

# Configuration
CONFIG = {
    'data_path': 'training_data.json',
    'model_name': 'bert-base-uncased',
    'max_length': 128,
    'batch_size': 16,
    'learning_rate': 2e-5,
    'epochs': 3,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

LABEL_NAMES = ['neutral', 'stress', 'unsafe_environment', 
               'emotional_distress', 'self_harm_low', 'self_harm_high']

class MentalHealthDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        labels = [item['labels'][label] for label in LABEL_NAMES]
        
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

class BERTClassifier(nn.Module):
    def __init__(self, n_classes=6, dropout=0.3):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(CONFIG['model_name'])
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(768, 256)
        self.fc2 = nn.Linear(256, n_classes)
        self.relu = nn.ReLU()
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def train_model():
    print("="*60)
    print("🚀 BERT Mental Health Classifier Training")
    print("="*60)
    print(f"Device: {CONFIG['device']}")
    
    # Load data
    print("\n📚 Loading training data...")
    with open(CONFIG['data_path'], 'r', encoding='utf-8') as f:
        all_data = json.load(f)
    
    print(f"Total samples: {len(all_data)}")
    
    # Split data (80/20)
    split_idx = int(len(all_data) * 0.8)
    train_data = all_data[:split_idx]
    val_data = all_data[split_idx:]
    
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")
    
    # Initialize tokenizer
    print("\n🔧 Initializing BERT tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(CONFIG['model_name'])
    
    # Create datasets
    train_dataset = MentalHealthDataset(train_data, tokenizer, CONFIG['max_length'])
    val_dataset = MentalHealthDataset(val_data, tokenizer, CONFIG['max_length'])
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'])
    
    # Initialize model
    print("\n🤖 Initializing model...")
    model = BERTClassifier()
    model.to(CONFIG['device'])
    
    # Optimizer and loss
    optimizer = AdamW(model.parameters(), lr=CONFIG['learning_rate'])
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
        
        progress_bar = tqdm(train_loader, desc="Training")
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(CONFIG['device'])
            attention_mask = batch['attention_mask'].to(CONFIG['device'])
            labels = batch['labels'].to(CONFIG['device'])
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc="Validation")
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(CONFIG['device'])
                attention_mask = batch['attention_mask'].to(CONFIG['device'])
                labels = batch['labels'].to(CONFIG['device'])
                
                outputs = model(input_ids, attention_mask)
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
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'accuracy': accuracy,
                'label_names': LABEL_NAMES
            }
            
            checkpoint_path = 'checkpoints/bert_mental_health_best.pt'
            torch.save(checkpoint, checkpoint_path)
            
            # Also save tokenizer
            tokenizer.save_pretrained('checkpoints')
            
            print(f"\n💾 Model saved! Val Loss: {avg_val_loss:.4f}, Acc: {accuracy:.4f}")
    
    print(f"\n{'='*60}")
    print("✅ Training Complete!")
    print(f"{'='*60}")
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print(f"Model saved to: checkpoints/bert_mental_health_best.pt")
    print(f"Tokenizer saved to: checkpoints/")
    
    # Test predictions
    print(f"\n{'='*60}")
    print("🧪 Testing model with sample texts...")
    print(f"{'='*60}")
    
    test_texts = [
        "I'm feeling great today!",
        "I'm so stressed about work",
        "I want to hurt myself",
        "This environment is unsafe"
    ]
    
    model.eval()
    with torch.no_grad():
        for text in test_texts:
            encoding = tokenizer(
                text,
                add_special_tokens=True,
                max_length=CONFIG['max_length'],
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(CONFIG['device'])
            attention_mask = encoding['attention_mask'].to(CONFIG['device'])
            
            outputs = model(input_ids, attention_mask)
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
    print("✅ Model ready for use in app.py")
    print(f"{'='*60}")

if __name__ == '__main__':
    train_model()
