"""
Calibrated Model for Inference
Uses temperature scaling and optimized thresholds to prevent overprediction
"""

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import numpy as np
import os

class TemperatureScaling(nn.Module):
    """Temperature scaling for probability calibration"""
    def __init__(self, num_classes=6):
        super(TemperatureScaling, self).__init__()
        self.temperature = nn.Parameter(torch.ones(num_classes) * 1.5)
        
    def forward(self, logits):
        temperature = self.temperature.unsqueeze(0).expand(logits.size(0), -1)
        return logits / temperature
    
    def set_temperatures(self, temperatures):
        """Set learned temperatures from checkpoint"""
        self.temperature.data = torch.FloatTensor(temperatures)

class CalibratedBERTClassifier(nn.Module):
    """BERT classifier with temperature scaling"""
    def __init__(self, n_classes=6, dropout=0.3):
        super(CalibratedBERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(768)
        
        self.fc1 = nn.Linear(768, 384)
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, n_classes)
        
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout * 0.5)
        
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

class CalibratedMentalHealthClassifier:
    """
    Production classifier with calibration to prevent overprediction
    
    Key features:
    - Temperature-scaled probabilities (better calibrated)
    - Optimized thresholds per class (reduces false positives)
    - Detailed prediction metadata
    """
    def __init__(self, model_path=None, device='cpu'):
        self.device = torch.device(device)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.label_names = ['neutral', 'stress', 'unsafe_environment', 
                           'emotional_distress', 'self_harm_low', 'self_harm_high']
        
        # Initialize model
        self.model = CalibratedBERTClassifier(n_classes=len(self.label_names))
        
        # Default values (will be overridden by checkpoint)
        self.thresholds = {label: 0.5 for label in self.label_names}
        self.temperatures = [1.5] * len(self.label_names)
        self.is_calibrated = False
        self.threshold_stats = {}
        
        # Load checkpoint
        if model_path and os.path.exists(model_path):
            self.load_checkpoint(model_path)
        
        self.model.to(self.device)
        self.model.eval()
        
    def load_checkpoint(self, checkpoint_path):
        """Load calibrated model checkpoint"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Load model state
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print(f"✓ Loaded model from {checkpoint_path}")
                
                # Check if calibrated
                self.is_calibrated = checkpoint.get('calibrated', False)
                
                if self.is_calibrated:
                    print(f"✓ Model is CALIBRATED (temperature-scaled)")
                    
                    # Load temperatures
                    if 'temperatures' in checkpoint:
                        self.temperatures = checkpoint['temperatures']
                        self.model.temperature_scaling.set_temperatures(self.temperatures)
                        print(f"✓ Loaded temperature scaling parameters")
                    
                    # Load optimal thresholds
                    if 'optimal_thresholds' in checkpoint:
                        self.thresholds = checkpoint['optimal_thresholds']
                        print(f"✓ Loaded optimized thresholds (low FPR)")
                        
                    # Load threshold statistics
                    if 'threshold_stats' in checkpoint:
                        self.threshold_stats = checkpoint['threshold_stats']
                    
                    # Print calibration info
                    if 'ece_before' in checkpoint and 'ece_after' in checkpoint:
                        ece_before = np.mean(checkpoint['ece_before'])
                        ece_after = np.mean(checkpoint['ece_after'])
                        print(f"✓ Calibration improved ECE: {ece_before:.4f} → {ece_after:.4f}")
                    
                    # Print threshold info
                    print(f"\nOptimized thresholds per class:")
                    for label in self.label_names:
                        threshold = self.thresholds[label]
                        stats = self.threshold_stats.get(label, {})
                        specificity = stats.get('specificity', 0)
                        fpr = stats.get('fpr', 0)
                        print(f"  {label:25s}: {threshold:.3f} (FPR: {fpr:.1%}, Spec: {specificity:.1%})")
                else:
                    print(f"⚠️  Model is NOT calibrated (using default thresholds)")
                    
                # Print training info
                if 'epoch' in checkpoint:
                    print(f"  Trained for {checkpoint['epoch']} epochs")
                if 'f1_score' in checkpoint:
                    print(f"  F1 score: {checkpoint['f1_score']:.4f}")
                    
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
    
    def preprocess_text(self, text, max_length=128):
        """Clean and tokenize text"""
        text = str(text).strip()
        text = text.replace('\x00', '').replace('\r', ' ').replace('\n', ' ')
        text = ' '.join(text.split())
        
        if not text:
            text = "[EMPTY]"
        
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
    
    def predict(self, text, use_calibration=True, return_all_scores=True, 
                show_confidence_level=True):
        """
        Predict with calibrated probabilities and optimized thresholds
        
        Args:
            text: Input text
            use_calibration: Use temperature scaling (recommended)
            return_all_scores: Return scores for all classes
            show_confidence_level: Categorize confidence levels
            
        Returns:
            Dictionary with predictions, scores, and metadata
        """
        with torch.no_grad():
            # Preprocess
            encoding = self.preprocess_text(text)
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            # Get predictions with optional temperature scaling
            logits = self.model(input_ids, attention_mask, 
                              apply_temperature=use_calibration and self.is_calibrated)
            probabilities = torch.sigmoid(logits).cpu().numpy()[0]
            
            # Clip to valid range
            probabilities = np.clip(probabilities, 0.0, 1.0)
            
            # Build results
            results = {
                'text': text,
                'predictions': [],
                'all_scores': {},
                'is_calibrated': self.is_calibrated and use_calibration,
                'num_predictions': 0
            }
            
            # Process each class
            for idx, (label, prob) in enumerate(zip(self.label_names, probabilities)):
                # Store all scores
                results['all_scores'][label] = {
                    'probability': float(round(prob, 4)),
                    'threshold': float(round(self.thresholds[label], 3))
                }
                
                # Confidence level categorization
                if show_confidence_level:
                    if prob < 0.2:
                        confidence_level = "very_low"
                    elif prob < 0.4:
                        confidence_level = "low"
                    elif prob < 0.6:
                        confidence_level = "moderate"
                    elif prob < 0.8:
                        confidence_level = "high"
                    else:
                        confidence_level = "very_high"
                    results['all_scores'][label]['confidence_level'] = confidence_level
                
                # Add to predictions if above threshold
                if prob >= self.thresholds[label]:
                    prediction = {
                        'label': label,
                        'probability': float(round(prob, 4)),
                        'threshold': float(round(self.thresholds[label], 3)),
                        'above_threshold': True
                    }
                    
                    if show_confidence_level:
                        prediction['confidence_level'] = confidence_level
                    
                    # Add threshold stats if available
                    if label in self.threshold_stats:
                        stats = self.threshold_stats[label]
                        prediction['stats'] = {
                            'false_positive_rate': float(round(stats['fpr'], 3)),
                            'specificity': float(round(stats['specificity'], 3)),
                            'precision': float(round(stats['precision'], 3))
                        }
                    
                    results['predictions'].append(prediction)
            
            # Sort by probability (highest first)
            results['predictions'] = sorted(
                results['predictions'],
                key=lambda x: x['probability'],
                reverse=True
            )
            
            results['num_predictions'] = len(results['predictions'])
            
            # Add risk assessment
            results['risk_assessment'] = self._assess_risk(results['predictions'])
            
            return results
    
    def _assess_risk(self, predictions):
        """Categorize overall risk level based on predictions"""
        if not predictions:
            return "minimal"
        
        # Check for high-risk categories
        high_risk_labels = ['self_harm_high', 'self_harm_low']
        moderate_risk_labels = ['emotional_distress', 'unsafe_environment']
        
        high_risk_present = any(p['label'] in high_risk_labels for p in predictions)
        moderate_risk_present = any(p['label'] in moderate_risk_labels for p in predictions)
        
        if high_risk_present:
            max_prob = max((p['probability'] for p in predictions if p['label'] in high_risk_labels), default=0)
            if max_prob > 0.8:
                return "critical"
            else:
                return "high"
        elif moderate_risk_present:
            return "moderate"
        else:
            return "low"
    
    def batch_predict(self, texts, use_calibration=True):
        """Predict for multiple texts"""
        return [self.predict(text, use_calibration) for text in texts]
    
    def compare_calibrated_vs_uncalibrated(self, text):
        """
        Compare predictions with and without calibration
        Useful for debugging overprediction issues
        """
        uncalibrated = self.predict(text, use_calibration=False, 
                                    show_confidence_level=False)
        calibrated = self.predict(text, use_calibration=True, 
                                 show_confidence_level=False)
        
        comparison = {
            'text': text,
            'differences': []
        }
        
        for label in self.label_names:
            uncal_prob = uncalibrated['all_scores'][label]['probability']
            cal_prob = calibrated['all_scores'][label]['probability']
            diff = uncal_prob - cal_prob
            
            comparison['differences'].append({
                'label': label,
                'uncalibrated': uncal_prob,
                'calibrated': cal_prob,
                'reduction': diff,
                'reduction_percent': diff / uncal_prob * 100 if uncal_prob > 0 else 0
            })
        
        return comparison

def classify_text(text, model_path='checkpoints/best_mental_health_model.pt', device='cpu'):
    """Quick classification function"""
    classifier = CalibratedMentalHealthClassifier(model_path=model_path, device=device)
    return classifier.predict(text)

if __name__ == '__main__':
    print("="*80)
    print("Testing Calibrated Mental Health Classifier")
    print("="*80)
    
    model_path = 'checkpoints/best_mental_health_model.pt'
    if not os.path.exists(model_path):
        print(f"\n❌ Model not found at {model_path}")
        print("   Please train the model first using: python train_calibrated.py")
    else:
        print(f"\n✓ Loading model from {model_path}")
        classifier = CalibratedMentalHealthClassifier(model_path=model_path)
        
        # Test texts (including neutral ones that might get overpredicted)
        test_texts = [
            "I'm feeling great today!",
            "Just had a normal day at work",
            "I'm a bit stressed about my project deadline",
            "I feel very unsafe in my current situation",
            "I'm struggling emotionally and need support",
            "Having thoughts of self-harm",
            "The weather is nice today"
        ]
        
        print("\n" + "="*80)
        print("Test Predictions (with Calibration)")
        print("="*80)
        
        for text in test_texts:
            print(f"\n{'='*80}")
            print(f"Text: {text}")
            print('-'*80)
            
            result = classifier.predict(text)
            
            print(f"Risk Level: {result['risk_assessment'].upper()}")
            print(f"Calibrated: {'Yes' if result['is_calibrated'] else 'No'}")
            print(f"Predictions: {result['num_predictions']} labels above threshold")
            
            if result['predictions']:
                print("\nTriggered labels:")
                for pred in result['predictions']:
                    print(f"  • {pred['label']:<25s}: {pred['probability']:.3f} "
                          f"(threshold: {pred['threshold']:.3f}, "
                          f"confidence: {pred['confidence_level']})")
                    if 'stats' in pred:
                        print(f"    FPR: {pred['stats']['false_positive_rate']:.1%}, "
                              f"Precision: {pred['stats']['precision']:.1%}")
            else:
                print("\n  (No labels above threshold - low risk)")
            
            print("\nAll scores:")
            for label, data in result['all_scores'].items():
                prob = data['probability']
                threshold = data['threshold']
                level = data['confidence_level']
                marker = "✓" if prob >= threshold else " "
                print(f"  {marker} {label:<25s}: {prob:.3f} (threshold: {threshold:.3f}) [{level}]")
        
        # Demonstrate calibration effect
        print("\n" + "="*80)
        print("Calibration Effect Comparison")
        print("="*80)
        
        test_text = "I'm feeling a bit tired today"
        print(f"\nText: {test_text}")
        print("\nComparing uncalibrated vs calibrated predictions:\n")
        
        comparison = classifier.compare_calibrated_vs_uncalibrated(test_text)
        
        print(f"{'Label':<25} {'Uncalibrated':>12} {'Calibrated':>12} {'Reduction':>12}")
        print('-'*80)
        for diff in comparison['differences']:
            print(f"{diff['label']:<25} {diff['uncalibrated']:>12.4f} {diff['calibrated']:>12.4f} "
                  f"{diff['reduction']:>12.4f} (-{diff['reduction_percent']:.1f}%)")
        
        print("\n" + "="*80)
        print("✓ Testing completed successfully!")
        print("="*80)
