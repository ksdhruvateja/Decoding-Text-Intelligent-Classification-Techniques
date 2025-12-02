import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import numpy as np

class EmotionClassifier(nn.Module):
    """
    Multi-label emotion classification model using BERT
    """
    def __init__(self, num_categories=6, dropout_rate=0.3):
        super(EmotionClassifier, self).__init__()
        self.bert_layer = BertModel.from_pretrained('bert-base-uncased')
        self.dropout_layer = nn.Dropout(dropout_rate)
        self.classifier_layer = nn.Linear(768, num_categories)
        
    def forward(self, input_token_ids, attention_masks):
        bert_outputs = self.bert_layer(
            input_ids=input_token_ids,
            attention_mask=attention_masks
        )
        pooled_representation = bert_outputs.pooler_output
        dropped_output = self.dropout_layer(pooled_representation)
        classification_logits = self.classifier_layer(dropped_output)
        return classification_logits


class TextClassificationService:
    """
    Service for handling text classification operations
    """
    def __init__(self, model_path=None, device='cpu'):
        self.device_name = device
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.emotion_categories = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        self.classification_model = EmotionClassifier(num_categories=len(self.emotion_categories))
        
        if model_path:
            self.load_trained_model(model_path)
        
        self.classification_model.to(self.device_name)
        self.classification_model.eval()
        
    def load_trained_model(self, checkpoint_path):
        """Load pre-trained model weights"""
        checkpoint_data = torch.load(checkpoint_path, map_location=self.device_name)
        self.classification_model.load_state_dict(checkpoint_data['model_state'])
        
    def prepare_text_input(self, text_content, max_sequence_length=128):
        """Tokenize and prepare text for model input"""
        # Clean and normalize text
        text_content = str(text_content).strip()
        # Remove any null bytes or problematic characters
        text_content = text_content.replace('\x00', '').replace('\r', ' ').replace('\n', ' ')
        # Normalize whitespace
        text_content = ' '.join(text_content.split())
        
        encoded_input = self.bert_tokenizer.encode_plus(
            text_content,
            add_special_tokens=True,
            max_length=max_sequence_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return encoded_input
    
    def classify_text(self, input_text, confidence_threshold=0.5):
        """
        Classify text and return predicted labels with confidence scores
        
        Args:
            input_text: Text to classify
            confidence_threshold: Minimum confidence for label prediction
            
        Returns:
            Dictionary with labels and their confidence scores
        """
        with torch.no_grad():
            # Preprocess and encode text
            encoded_data = self.prepare_text_input(input_text)
            token_ids = encoded_data['input_ids'].to(self.device_name)
            attention_mask = encoded_data['attention_mask'].to(self.device_name)
            
            # Get model predictions (logits)
            prediction_logits = self.classification_model(token_ids, attention_mask)
            
            # Apply sigmoid to convert logits to probabilities [0, 1]
            prediction_probabilities = torch.sigmoid(prediction_logits).cpu().numpy()[0]
            
            # Ensure probabilities are properly bounded
            prediction_probabilities = np.clip(prediction_probabilities, 0.0, 1.0)
            
            results = {
                'text': input_text,
                'predictions': [],
                'all_scores': {}
            }
            
            # Map probabilities to correct category labels
            for idx, (category_name, probability_score) in enumerate(zip(self.emotion_categories, prediction_probabilities)):
                # Store all scores with proper precision
                results['all_scores'][category_name] = float(round(probability_score, 4))
                
                # Only include predictions above threshold
                if probability_score >= confidence_threshold:
                    results['predictions'].append({
                        'label': category_name,
                        'confidence': float(round(probability_score, 4))
                    })
            
            # Sort predictions by confidence (highest first)
            results['predictions'] = sorted(
                results['predictions'], 
                key=lambda x: x['confidence'], 
                reverse=True
            )
            
            return results
    
    def batch_classify(self, text_batch, confidence_threshold=0.5):
        """Classify multiple texts at once"""
        return [self.classify_text(text, confidence_threshold) for text in text_batch]