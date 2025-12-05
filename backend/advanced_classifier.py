"""
Advanced classifier loader for production use
"""

import torch
import torch.nn as nn
import json
import re
import os

class Tokenizer:
    """Advanced tokenizer"""
    def __init__(self):
        self.word2idx = {'<PAD>': 0, '<UNK>': 1, '<START>': 2, '<END>': 3}
        self.idx2word = {0: '<PAD>', 1: '<UNK>', 2: '<START>', 3: '<END>'}
    
    def tokenize(self, text):
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s\'-]', ' ', text)
        return text.split()
    
    def encode(self, text, max_length=100):
        words = self.tokenize(text)
        indices = [2] + [self.word2idx.get(word, 1) for word in words] + [3]
        
        if len(indices) < max_length:
            indices += [0] * (max_length - len(indices))
        else:
            indices = indices[:max_length-1] + [3]
        
        return indices

class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_dim, 1)
    
    def forward(self, lstm_output):
        attention_weights = torch.softmax(self.attention(lstm_output), dim=1)
        context = torch.sum(attention_weights * lstm_output, dim=1)
        return context, attention_weights

class AdvancedTextClassifier(nn.Module):
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
        embedded = self.embedding(input_ids)
        lstm_out, _ = self.lstm(embedded)
        context, _ = self.attention(lstm_out)
        
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

class AdvancedClassifier:
    """Production classifier with trained model"""
    
    def __init__(self, model_path='checkpoints/advanced_classifier.pt', tokenizer_path='checkpoints/advanced_tokenizer.json'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.label_names = ['positive', 'negative', 'neutral', 'stress', 'emotional_distress', 
                           'self_harm_low', 'self_harm_high', 'unsafe_environment']
        
        if not os.path.exists(model_path) or not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Model or tokenizer not found")
        
        print(f"Loading advanced classifier from {model_path}...")
        
        # Load tokenizer
        with open(tokenizer_path, 'r') as f:
            tokenizer_data = json.load(f)
        
        self.tokenizer = Tokenizer()
        self.tokenizer.word2idx = tokenizer_data['word2idx']
        self.tokenizer.idx2word = tokenizer_data['idx2word']
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        config = checkpoint['config']
        vocab_size = len(self.tokenizer.word2idx)
        
        # Initialize model
        self.model = AdvancedTextClassifier(
            vocab_size=vocab_size,
            embedding_dim=config['embedding_dim'],
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            num_classes=config['num_classes'],
            dropout=config['dropout']
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        accuracy = checkpoint.get('accuracy', 0)
        print(f"Advanced Classifier loaded! (Accuracy: {accuracy:.2%})")
    
    def classify(self, text):
        """
        Classify text into all categories
        
        Returns:
        {
            'text': str,
            'predictions': [{'label': str, 'score': float}],
            'all_scores': {label: float},
            'primary_category': str,
            'confidence': float,
            'sentiment': str,
            'emotion': str
        }
        """
        # Encode text
        indices = self.tokenizer.encode(text)
        input_ids = torch.LongTensor([indices]).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(input_ids)
            probs = torch.sigmoid(outputs).cpu().numpy()[0]
        
        # Build response
        all_scores = {label: float(probs[idx]) for idx, label in enumerate(self.label_names)}
        
        # Get predictions above threshold
        predictions = []
        for idx, label in enumerate(self.label_names):
            if probs[idx] > 0.5:
                predictions.append({
                    'label': label,
                    'score': float(probs[idx])
                })
        
        # Sort by score
        predictions.sort(key=lambda x: x['score'], reverse=True)
        
        # Determine sentiment
        sentiment = 'neutral'
        if probs[0] > 0.5:  # positive
            sentiment = 'positive'
        elif probs[1] > 0.5:  # negative
            sentiment = 'negative'
        
        # Determine emotion
        emotion = 'neutral'
        if probs[3] > 0.5:  # stress
            emotion = 'stress'
        elif probs[4] > 0.5:  # emotional_distress
            emotion = 'emotional_distress'
        elif probs[5] > 0.5 or probs[6] > 0.5:  # self_harm
            emotion = 'crisis'
        elif probs[7] > 0.5:  # unsafe_environment
            emotion = 'unsafe'
        
        # Primary category
        max_idx = probs.argmax()
        primary_category = self.label_names[max_idx]
        confidence = float(probs[max_idx])
        
        return {
            'text': text,
            'predictions': predictions,
            'all_scores': all_scores,
            'primary_category': primary_category,
            'confidence': confidence,
            'sentiment': sentiment,
            'emotion': emotion
        }

# Test if run directly
if __name__ == '__main__':
    print("="*60)
    print("Testing Advanced Classifier")
    print("="*60)
    
    classifier = AdvancedClassifier()
    
    test_texts = [
        "This movie was fantastic, I'd watch it again anytime",
        "Shipping took forever and the item was damaged",
        "This book is 300 pages long and written in English",
        "The stock market crashed yesterday after tech earnings disappointed",
    ]
    
    for text in test_texts:
        print(f"\nText: {text}")
        result = classifier.classify(text)
        print(f"Sentiment: {result['sentiment']}")
        print(f"Emotion: {result['emotion']}")
        print(f"Primary: {result['primary_category']} ({result['confidence']:.2f})")
        if result['predictions']:
            preds = [f"{p['label']}: {p['score']:.2f}" for p in result['predictions']]
            print(f"Predictions: {', '.join(preds)}")
