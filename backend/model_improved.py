"""
Improved model.py with support for optimal thresholds and better predictions
"""
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import numpy as np
import os

class ImprovedBERTClassifier(nn.Module):
    """
    Enhanced BERT classifier matching the improved training architecture
    """
    def __init__(self, n_classes=6, dropout=0.3):
        super(ImprovedBERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        # Enhanced classifier head (matches training architecture)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(768)
        
        self.fc1 = nn.Linear(768, 384)
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, n_classes)
        
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout * 0.5)
        
    def forward(self, input_ids, attention_mask):
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
        x = self.fc3(x)
        
        return x


class MentalHealthClassifier:
    """
    Production-ready mental health text classifier with optimal thresholds
    """
    def __init__(self, model_path=None, device='cpu'):
        self.device = torch.device(device)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.label_names = ['neutral', 'stress', 'unsafe_environment', 
                           'emotional_distress', 'self_harm_low', 'self_harm_high']
        
        # Initialize model
        self.model = ImprovedBERTClassifier(n_classes=len(self.label_names))
        
        # Default thresholds (can be overridden by checkpoint)
        self.thresholds = {label: 0.5 for label in self.label_names}
        
        # Load checkpoint if provided
        if model_path and os.path.exists(model_path):
            self.load_checkpoint(model_path)
        
        self.model.to(self.device)
        self.model.eval()
        
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint with optimal thresholds"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Load model state
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print(f"✓ Loaded model from {checkpoint_path}")
                
                # Load optimal thresholds if available
                if 'optimal_thresholds' in checkpoint:
                    self.thresholds = checkpoint['optimal_thresholds']
                    print(f"✓ Loaded optimal thresholds")
                    for label, threshold in self.thresholds.items():
                        print(f"  {label}: {threshold:.3f}")
                
                # Print model info
                if 'epoch' in checkpoint:
                    print(f"  Trained for {checkpoint['epoch']} epochs")
                if 'f1_score' in checkpoint:
                    print(f"  Best F1 score: {checkpoint['f1_score']:.4f}")
            else:
                print(f"Warning: Checkpoint format not recognized")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
    
    def preprocess_text(self, text, max_length=128):
        """Clean and tokenize text"""
        # Clean text
        text = str(text).strip()
        text = text.replace('\x00', '').replace('\r', ' ').replace('\n', ' ')
        text = ' '.join(text.split())
        
        if not text:
            text = "[EMPTY]"
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return encoding
    
    def predict(self, text, use_optimal_thresholds=True, return_all_scores=True):
        """
        Predict labels for input text
        
        Args:
            text: Input text to classify
            use_optimal_thresholds: Use optimized thresholds per class (recommended)
            return_all_scores: Return scores for all classes
            
        Returns:
            Dictionary with predictions and scores
        """
        with torch.no_grad():
            # Preprocess
            encoding = self.preprocess_text(text)
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            # Get predictions
            logits = self.model(input_ids, attention_mask)
            probabilities = torch.sigmoid(logits).cpu().numpy()[0]
            
            # Clip to valid range
            probabilities = np.clip(probabilities, 0.0, 1.0)
            
            # Build results
            results = {
                'text': text,
                'predictions': [],
                'all_scores': {}
            }
            
            # Apply thresholds and collect predictions
            for idx, (label, prob) in enumerate(zip(self.label_names, probabilities)):
                # Store all scores
                results['all_scores'][label] = float(round(prob, 4))
                
                # Determine threshold
                threshold = self.thresholds[label] if use_optimal_thresholds else 0.5
                
                # Add to predictions if above threshold
                if prob >= threshold:
                    results['predictions'].append({
                        'label': label,
                        'confidence': float(round(prob, 4)),
                        'threshold_used': float(round(threshold, 3))
                    })
            
            # Sort by confidence
            results['predictions'] = sorted(
                results['predictions'],
                key=lambda x: x['confidence'],
                reverse=True
            )
            
            # Add metadata
            results['num_predictions'] = len(results['predictions'])
            results['used_optimal_thresholds'] = use_optimal_thresholds
            
            return results
    
    def batch_predict(self, texts, use_optimal_thresholds=True):
        """Predict labels for multiple texts"""
        return [self.predict(text, use_optimal_thresholds) for text in texts]
    
    def get_top_prediction(self, text):
        """Get the single highest confidence prediction"""
        result = self.predict(text, use_optimal_thresholds=True)
        
        if result['predictions']:
            return result['predictions'][0]
        else:
            # Return label with highest score even if below threshold
            max_label = max(result['all_scores'].items(), key=lambda x: x[1])
            return {
                'label': max_label[0],
                'confidence': max_label[1],
                'below_threshold': True
            }


# Backward compatibility with old code
class TextClassificationService(MentalHealthClassifier):
    """Alias for backward compatibility"""
    pass


# For legacy code that expects EmotionClassifier
class EmotionClassifier(ImprovedBERTClassifier):
    """Alias for backward compatibility"""
    pass


# Convenience function for quick predictions
def classify_text(text, model_path='checkpoints/best_mental_health_model.pt', device='cpu'):
    """
    Quick classification function
    
    Usage:
        result = classify_text("I'm feeling stressed today")
        print(result['predictions'])
    """
    classifier = MentalHealthClassifier(model_path=model_path, device=device)
    return classifier.predict(text)


if __name__ == '__main__':
    # Test the classifier
    print("="*80)
    print("Testing Mental Health Classifier")
    print("="*80)
    
    # Check for model
    model_path = 'checkpoints/best_mental_health_model.pt'
    if not os.path.exists(model_path):
        print(f"\n❌ Model not found at {model_path}")
        print("   Please train the model first using: python train_bert_improved.py")
    else:
        print(f"\n✓ Loading model from {model_path}")
        classifier = MentalHealthClassifier(model_path=model_path)
        
        # Test predictions
        test_texts = [
            "I'm feeling great today, everything is wonderful!",
            "I'm so stressed with work and deadlines",
            "I feel unsafe in my environment",
            "I'm emotionally struggling and need help",
            "I have thoughts of self-harm",
            "Everything is fine, just a normal day"
        ]
        
        print("\n" + "="*80)
        print("Test Predictions")
        print("="*80)
        
        for text in test_texts:
            result = classifier.predict(text)
            print(f"\nText: {text}")
            print(f"Predictions ({result['num_predictions']} found):")
            if result['predictions']:
                for pred in result['predictions']:
                    print(f"  - {pred['label']}: {pred['confidence']:.3f} "
                          f"(threshold: {pred['threshold_used']:.3f})")
            else:
                print("  (No predictions above threshold)")
            
            # Show all scores
            print(f"All scores:")
            for label, score in result['all_scores'].items():
                print(f"  {label}: {score:.3f}")
        
        print("\n" + "="*80)
        print("✓ Testing completed successfully!")
        print("="*80)
