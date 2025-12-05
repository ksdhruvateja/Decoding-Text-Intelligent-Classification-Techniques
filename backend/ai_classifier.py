"""
AI-powered classifier using the trained lightweight model
"""

import torch
import torch.nn as nn
import os

class Tokenizer:
    """Simple word tokenizer (must match training)"""
    def __init__(self):
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
    
    def tokenize(self, text):
        """Tokenize text into words"""
        import re
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
        embedded = self.embedding(input_ids)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        x = self.dropout(hidden)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class AITextClassifier:
    """AI-powered text classifier using trained model"""
    
    def __init__(self, checkpoint_path='checkpoints/lightweight_classifier.pt'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.label_names = ['neutral', 'stress', 'unsafe_environment', 
                           'emotional_distress', 'self_harm_low', 'self_harm_high']
        
        # Load checkpoint
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
        
        print(f"Loading AI model from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Load tokenizer from JSON
        import json
        tokenizer_path = 'checkpoints/tokenizer.json'
        with open(tokenizer_path, 'r') as f:
            tokenizer_data = json.load(f)
        
        self.tokenizer = Tokenizer()
        self.tokenizer.word2idx = tokenizer_data['word2idx']
        # Convert keys back to int for idx2word
        self.tokenizer.idx2word = {int(k): v for k, v in tokenizer_data['idx2word'].items()}
        
        # Initialize model
        config = checkpoint['config']
        vocab_size = len(self.tokenizer.word2idx)
        
        self.model = TextClassifier(
            vocab_size=vocab_size,
            embedding_dim=config['embedding_dim'],
            hidden_dim=config['hidden_dim'],
            num_classes=config['num_classes'],
            dropout=0.5
        )
        
        # Load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        accuracy = checkpoint.get('accuracy', 0)
        print(f"✅ AI Model loaded successfully! (Accuracy: {accuracy:.2%})")
    
    def classify(self, text):
        """
        Classify text into mental health categories
        
        Returns:
        {
            'text': str,
            'predictions': [{'label': str, 'score': float}],
            'all_scores': {label: float},
            'primary_category': str,
            'confidence': float
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
        
        # Get primary category
        max_idx = probs.argmax()
        primary_category = self.label_names[max_idx]
        confidence = float(probs[max_idx])
        
        return {
            'text': text,
            'predictions': predictions,
            'all_scores': all_scores,
            'primary_category': primary_category,
            'confidence': confidence
        }

# Test if run directly
if __name__ == '__main__':
    print("="*60)
    print("🧪 Testing AI Text Classifier")
    print("="*60)
    
    classifier = AITextClassifier()
    
    test_texts = [
        "I'm feeling great today!",
        "I'm so stressed about work deadlines",
        "I want to hurt myself",
        "This workplace environment is unsafe",
        "I feel sad and lonely",
        "Everything is fine and normal",
        "I'm struggling with severe depression",
        "Life is wonderful and I'm so happy"
    ]
    
    for text in test_texts:
        print(f"\nText: {text}")
        result = classifier.classify(text)
        print(f"Primary: {result['primary_category']} ({result['confidence']:.3f})")
        if result['predictions']:
            preds = [f"{p['label']}: {p['score']:.3f}" for p in result['predictions']]
            print(f"Predictions: {preds}")
        else:
            print("Predictions: [No strong predictions]")
