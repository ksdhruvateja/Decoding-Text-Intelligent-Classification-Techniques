"""
BERT Mental Health Classifier - Model loader for trained model
"""

import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import numpy as np

class TemperatureScaling(nn.Module):
    """Temperature scaling for calibration"""
    def __init__(self, num_classes=6):
        super(TemperatureScaling, self).__init__()
        self.temperature = nn.Parameter(torch.ones(num_classes) * 1.5)
        
    def forward(self, logits):
        temperature = self.temperature.unsqueeze(0).expand(logits.size(0), -1)
        return logits / temperature

class BERTMentalHealthClassifier(nn.Module):
    def __init__(self, n_classes=6, dropout=0.3):
        super(BERTMentalHealthClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        # Enhanced architecture matching training
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(768)
        
        self.fc1 = nn.Linear(768, 384)
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, n_classes)
        
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout * 0.5)
        
        # Temperature scaling
        self.temperature_scaling = TemperatureScaling(n_classes)
        
    def forward(self, input_ids, attention_mask, apply_temperature=True):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        
        x = self.layer_norm(pooled_output)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        logits = self.fc3(x)
        
        if apply_temperature:
            logits = self.temperature_scaling(logits)
        
        return logits

class MentalHealthClassifierService:
    def __init__(self, model_path='checkpoints/best_mental_health_model.pt', device='cpu'):
        """Initialize the trained BERT classifier"""
        self.device = torch.device(device)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.label_names = ['neutral', 'stress', 'unsafe_environment', 
                           'emotional_distress', 'self_harm_low', 'self_harm_high']
        
        # Load model
        self.model = BERTMentalHealthClassifier(n_classes=6, dropout=0.3)
        # Load with weights_only=False for compatibility
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Load optimal thresholds if available
        self.optimal_thresholds = checkpoint.get('optimal_thresholds', {label: 0.5 for label in self.label_names})
        
        f1_score = checkpoint.get('f1_score', 'N/A')
        print(f"✓ BERT Mental Health Classifier loaded (F1: {f1_score if isinstance(f1_score, str) else f1_score:.3f})")
    
    def classify(self, text: str, threshold: float = None) -> dict:
        """Classify text using trained BERT model with optimal thresholds"""
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Predict with temperature scaling (calibrated probabilities)
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask, apply_temperature=True)
            probabilities = torch.sigmoid(outputs).cpu().numpy()[0]
        
        # Create scores dict
        scores = {label: float(prob) for label, prob in zip(self.label_names, probabilities)}
        
        # Create predictions list using optimal thresholds
        predictions = []
        for label, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            # Use optimal threshold for each class, or default if not available
            label_threshold = self.optimal_thresholds.get(label, threshold if threshold else 0.5)
            if score >= label_threshold:
                predictions.append({
                    'label': label,
                    'confidence': float(score),
                    'threshold': float(label_threshold)
                })
        
        # Determine sentiment and emotion using optimal thresholds
        # Priority order: Check for positive first, then risk categories
        
        # Check for positive content first (higher priority than risk)
        positive_words = ['thank', 'grateful', 'appreciate', 'happy', 'love', 'great', 'wonderful', 'excellent', 'excited', 'joyful', 'amazing', 'blessed', 'proud']
        negative_words = ['hurt', 'pain', 'die', 'kill', 'hate', 'worse', 'can\'t', 'never', 'nothing', 'hopeless']
        text_lower = text.lower()
        has_positive_words = any(word in text_lower for word in positive_words)
        has_negative_words = any(word in text_lower for word in negative_words)
        
        # If clearly positive (positive words WITHOUT negative context), override model
        if has_positive_words and not has_negative_words:
            # Check that it's not actually a high-risk case
            if scores['self_harm_high'] < 0.70 and scores['self_harm_low'] < 0.60:
                sentiment = 'safe'
                emotion = 'positive'
        # Check risk categories with optimal thresholds
        elif scores['self_harm_high'] >= self.optimal_thresholds.get('self_harm_high', 0.70):
            sentiment = 'high_risk'
            emotion = 'self_harm_high'
        elif scores['self_harm_low'] >= self.optimal_thresholds.get('self_harm_low', 0.50):
            sentiment = 'concerning'
            emotion = 'self_harm_low'
        elif scores['unsafe_environment'] >= self.optimal_thresholds.get('unsafe_environment', 0.50):
            sentiment = 'concerning'
            emotion = 'unsafe_environment'
        elif scores['emotional_distress'] >= self.optimal_thresholds.get('emotional_distress', 0.50):
            sentiment = 'concerning'
            emotion = 'emotional_distress'
        elif scores['stress'] >= self.optimal_thresholds.get('stress', 0.50):
            sentiment = 'neutral'
            emotion = 'stress'
        elif scores['neutral'] >= self.optimal_thresholds.get('neutral', 0.45):
            sentiment = 'neutral'
            emotion = 'neutral'
        else:
            sentiment = 'neutral'
            emotion = 'conversational'
        
        return {
            'text': text,
            'predictions': predictions,
            'all_scores': scores,
            'sentiment': sentiment,
            'emotion': emotion
        }

# Global instance
bert_classifier = None

def initialize_bert_classifier():
    """Initialize the BERT classifier"""
    global bert_classifier
    try:
        bert_classifier = MentalHealthClassifierService(
            model_path='checkpoints/best_mental_health_model.pt',
            device='cpu'
        )
        return bert_classifier
    except Exception as e:
        print(f"Error loading BERT model: {e}")
        return None
