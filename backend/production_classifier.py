"""
Production Hybrid Classifier

Primary flow:
- Keyword rules for explicit safety/intent
- TF-IDF + Logistic regression fallback for nuanced language
- Optional BERT tier when checkpoints are available
"""
import os
from typing import Dict

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

from simple_classifier import SimpleClassifier
from tfidf_classifier import TfidfTextClassifier

class BertClassifierModel(nn.Module):
    """BERT model architecture"""
    def __init__(self, n_classes, dropout=0.3):
        super(BertClassifierModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.drop = nn.Dropout(p=dropout)
        self.fc = nn.Linear(self.bert.config.hidden_size, n_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.drop(pooled_output)
        return self.fc(output)

class ProductionClassifier:
    """
    Production classifier with rule-based + BERT hybrid approach
    """

    HIGH_RISK_CATEGORIES = {"self_harm_high", "self_harm_low", "unsafe_environment"}

    def __init__(self):
        self.simple_classifier = SimpleClassifier()
        self.bert_model = None
        self.tokenizer = None
        self.categories = None
        self.device = torch.device('cpu')
        self.bert_available = self._load_bert_model()
        self.tfidf_classifier = TfidfTextClassifier()
        self.tfidf_available = self.tfidf_classifier.is_available
    
    def _load_bert_model(self):
        """Load BERT model if available"""
        try:
            # Try multiple locations so it works from repo root or backend folder
            base_dir = os.path.dirname(os.path.abspath(__file__))
            candidate_model_paths = [
                os.path.join(base_dir, '..', 'checkpoints', 'bert_classifier_best.pt'),
                os.path.join(base_dir, 'checkpoints', 'bert_classifier_best.pt'),
                'checkpoints/bert_classifier_best.pt',
            ]
            candidate_tokenizer_paths = [
                os.path.join(base_dir, '..', 'checkpoints', 'bert_tokenizer'),
                os.path.join(base_dir, 'checkpoints', 'bert_tokenizer'),
                'checkpoints/bert_tokenizer',
            ]
            model_path = next((p for p in candidate_model_paths if os.path.exists(p)), None)
            tokenizer_path = next((p for p in candidate_tokenizer_paths if os.path.exists(p)), None)
            
            if model_path and tokenizer_path:
                checkpoint = torch.load(model_path, map_location=self.device)
                self.categories = checkpoint['categories']
                
                self.bert_model = BertClassifierModel(
                    n_classes=len(self.categories),
                    dropout=checkpoint['config'].get('dropout', 0.3)
                )
                self.bert_model.load_state_dict(checkpoint['model_state_dict'])
                self.bert_model.to(self.device)
                self.bert_model.eval()
                
                self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
                print(f"BERT model loaded (model: {model_path}, tokenizer: {tokenizer_path})")
                return True
            else:
                print("BERT checkpoints not found. Checked paths:")
                for p in candidate_model_paths:
                    print(f"   - model: {p} exists={os.path.exists(p)}")
                for p in candidate_tokenizer_paths:
                    print(f"   - tokenizer: {p} exists={os.path.exists(p)}")
        except Exception as e:
            print(f"ℹ️  BERT not available: {e}")
        return False
    
    def _bert_classify(self, text: str) -> Dict:
        """Classify using BERT"""
        encoding = self.tokenizer.encode_plus(
            text, add_special_tokens=True, max_length=128,
            return_token_type_ids=False, padding='max_length',
            truncation=True, return_attention_mask=True, return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        
        scores = {cat: float(prob) for cat, prob in zip(self.categories, probabilities)}
        predictions = self._format_predictions(scores)

        primary_idx = probabilities.argmax()
        primary_category = self.categories[primary_idx]
        confidence = float(probabilities[primary_idx])

        sentiment = self._infer_sentiment(scores)
        emotion = self._infer_emotion(scores)

        return {
            'text': text, 'predictions': predictions, 'all_scores': scores,
            'primary_category': primary_category, 'confidence': float(confidence),
            'sentiment': sentiment, 'emotion': emotion, 'model': 'bert'
        }
    
    @staticmethod
    def _format_predictions(scores: Dict[str, float], threshold: float = 0.25):
        # Only include predictions above meaningful threshold
        predictions = [
            {'label': label, 'score': float(score)}
            for label, score in scores.items()
            if score > threshold
        ]
        predictions.sort(key=lambda item: item['score'], reverse=True)
        # Limit to top 3 most confident predictions to avoid noise
        return predictions[:3]

    @classmethod
    def _infer_sentiment(cls, scores: Dict[str, float]) -> str:
        positive_score = scores.get('positive', 0.0)
        negative_score = scores.get('negative', 0.0)
        neutral_score = scores.get('neutral', 0.0)
        stress_score = scores.get('stress', 0.0)
        
        # Clear positive signal
        if positive_score > 0.6 and positive_score > negative_score * 1.5:
            return 'positive'
        
        # Clear negative signal
        if negative_score > 0.5 or stress_score > 0.6:
            return 'negative'
        
        # High-risk categories override to negative
        if any(scores.get(cat, 0.0) > 0.4 for cat in cls.HIGH_RISK_CATEGORIES):
            return 'negative'
        
        # Default to neutral for ambiguous cases
        return 'neutral'

    @classmethod
    def _infer_emotion(cls, scores: Dict[str, float]) -> str:
        # Higher thresholds for crisis categories to avoid false alarms
        if scores.get('self_harm_high', 0.0) > 0.6 or scores.get('self_harm_low', 0.0) > 0.6:
            return 'crisis'
        if scores.get('unsafe_environment', 0.0) > 0.6:
            return 'unsafe'
        if scores.get('emotional_distress', 0.0) > 0.5:
            return 'emotional_distress'
        if scores.get('stress', 0.0) > 0.5:
            return 'stress'
        if scores.get('positive', 0.0) > 0.6:
            return 'positive'
        return 'neutral'

    def _blend_scores(self, primary_scores: Dict[str, float], secondary_scores: Dict[str, float], weight: float = 0.75):
        blended = {}
        for label, score in secondary_scores.items():
            blended[label] = score * weight + primary_scores.get(label, 0.0) * (1 - weight)

        for label, score in primary_scores.items():
            if label not in blended:
                blended[label] = score * (1 - weight)

        for label in self.HIGH_RISK_CATEGORIES:
            blended[label] = max(blended.get(label, 0.0), primary_scores.get(label, 0.0))

        return blended

    def _merge_with_tfidf(self, text: str, simple_result: Dict, tfidf_result: Dict) -> Dict:
        blended_scores = self._blend_scores(simple_result['all_scores'], tfidf_result['all_scores'])
        primary_category = max(blended_scores, key=blended_scores.get)
        confidence = float(blended_scores[primary_category])

        # If TF-IDF is uncertain but rules are confident, keep the rule-based decision
        if confidence < 0.45 and simple_result['confidence'] >= 0.6:
            simple_result['model'] = 'simple'
            return simple_result

        predictions = self._format_predictions(blended_scores)
        sentiment = self._infer_sentiment(blended_scores)
        emotion = self._infer_emotion(blended_scores)

        model_tag = 'tfidf'
        if primary_category == simple_result['primary_category']:
            model_tag = 'hybrid_tfidf'

        return {
            'text': text,
            'predictions': predictions,
            'all_scores': blended_scores,
            'primary_category': primary_category,
            'confidence': confidence,
            'sentiment': sentiment,
            'emotion': emotion,
            'model': model_tag,
        }

    def classify(self, text: str) -> Dict:
        """
        Hybrid classification:
        1. Simple classifier for pattern matching
        2. BERT for complex cases (if available)
        3. TF-IDF fallback when BERT is unavailable
        4. Agreement boosting for consistent signals
        """
        # Handle edge cases
        if text is None:
            text = ""
        text = str(text).strip()
        
        simple_result = self.simple_classifier.classify(text)

        # HIGH-RISK: Only trust if very confident (0.80+) to avoid false alarms
        if simple_result['primary_category'] in self.HIGH_RISK_CATEGORIES and simple_result['confidence'] >= 0.80:
            simple_result['model'] = 'simple'
            return simple_result
        
        # High confidence pattern match (non-crisis categories)
        if simple_result['confidence'] >= 0.90 and simple_result['primary_category'] not in self.HIGH_RISK_CATEGORIES:
            simple_result['model'] = 'simple'
            return simple_result
        
        # NEUTRAL: Trust high-confidence neutral classifications (avoid over-classifying)
        if simple_result['primary_category'] == 'neutral' and simple_result['confidence'] >= 0.75:
            simple_result['model'] = 'simple'
            return simple_result
        
        # Try BERT if available
        if self.bert_available:
            try:
                bert_result = self._bert_classify(text)
                
                # Agreement boost
                if (bert_result['primary_category'] == simple_result['primary_category'] and
                    simple_result['confidence'] > 0.5):
                    bert_result['confidence'] = min(0.98, bert_result['confidence'] * 1.2)
                    bert_result['model'] = 'hybrid'
                
                return bert_result
            except Exception as e:
                print(f"⚠️  BERT error: {e}")

        # Try TF-IDF fallback if BERT is unavailable or failed
        if self.tfidf_available:
            try:
                tfidf_result = self.tfidf_classifier.classify(text)
                return self._merge_with_tfidf(text, simple_result, tfidf_result)
            except Exception as e:
                print(f"⚠️  TF-IDF error: {e}")
                self.tfidf_available = False

        # Fallback
        simple_result['model'] = 'simple'
        return simple_result

if __name__ == '__main__':
    classifier = ProductionClassifier()
    
    tests = [
        "i will kill you",
        "The movie was boring",
        "This book is 300 pages long",
        "I want to kill myself",
        "This movie was fantastic"
    ]
    
    print("\n" + "="*70)
    print("PRODUCTION CLASSIFIER TEST")
    print("="*70)
    
    for text in tests:
        result = classifier.classify(text)
        print(f"\nText: {text}")
        print(f"Category: {result['primary_category']} ({result['confidence']:.2f})")
        print(f"Model: {result['model']}")
