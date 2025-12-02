"""
Quick training script that creates a working trained model
Uses a simpler approach for faster training
"""

import torch
import torch.nn as nn
from transformers import BertTokenizer
import numpy as np
import pandas as pd
import os

# Simplified model for quick training
class SimpleToxicClassifier(nn.Module):
    def __init__(self, vocab_size=30522, embedding_dim=128, hidden_dim=64, num_labels=6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim * 2, num_labels)
        
    def forward(self, input_ids, attention_mask):
        embedded = self.embedding(input_ids)
        lstm_out, _ = self.lstm(embedded)
        # Use last output
        last_hidden = lstm_out[:, -1, :]
        dropped = self.dropout(last_hidden)
        logits = self.fc(dropped)
        return logits

def create_trained_checkpoint():
    """Create a trained model checkpoint with learned patterns"""
    
    print("Creating trained model...")
    
    # Load data
    df = pd.read_csv('data/train.csv')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Initialize model
    model = SimpleToxicClassifier()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    
    # Prepare data
    texts = df['comment_text'].values
    labels = df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].values
    
    # Training
    model.train()
    num_epochs = 10
    
    for epoch in range(num_epochs):
        total_loss = 0
        for i in range(0, len(texts), 8):  # Batch size 8
            batch_texts = texts[i:i+8]
            batch_labels = labels[i:i+8]
            
            # Tokenize
            encoded = tokenizer.batch_encode_plus(
                batch_texts.tolist(),
                add_special_tokens=True,
                max_length=128,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            input_ids = encoded['input_ids']
            attention_mask = encoded['attention_mask']
            target = torch.FloatTensor(batch_labels)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, target)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / (len(texts) // 8)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    # Save checkpoint
    os.makedirs('checkpoints', exist_ok=True)
    checkpoint = {
        'model_state': model.state_dict(),
        'model_type': 'simple',
        'epoch': num_epochs
    }
    
    torch.save(checkpoint, 'checkpoints/best_model_simple.pt')
    print(f"✓ Model saved to checkpoints/best_model_simple.pt")
    
    return model

if __name__ == '__main__':
    create_trained_checkpoint()
