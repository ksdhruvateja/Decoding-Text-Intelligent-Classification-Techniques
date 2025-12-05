"""
Simple but effective rule-based classifier
Handles sentiment and mental health classification with keyword matching
"""

import re
from typing import Dict, List

class SimpleClassifier:
    """Simple keyword-based classifier that actually works"""
    
    def __init__(self):
        # Positive keywords
        self.positive_words = {
            'fantastic', 'amazing', 'excellent', 'great', 'wonderful', 'perfect', 'love', 'loved',
            'best', 'awesome', 'brilliant', 'outstanding', 'superb', 'delicious', 'beautiful',
            'happy', 'excited', 'thrilled', 'proud', 'blessed', 'grateful', 'joyful', 'pleased',
            'like', 'liked', 'enjoy', 'enjoyed', 'good', 'nice', 'cool', 'sweet', 'fun', 'funny'
        }
        
        # Negative keywords
        self.negative_words = {
            'boring', 'terrible', 'awful', 'horrible', 'bad', 'worst', 'poor', 'disappointing',
            'disappointed', 'broke', 'broken', 'damaged', 'waste', 'useless', 'failed', 'hate',
            'disgusting', 'unacceptable', 'frustrating', 'frustrated', 'annoying', 'annoyed',
            'upset', 'angry', 'rude', 'unprofessional', 'slow', 'late', 'never', 'regret'
        }
        
        # Stress keywords
        self.stress_words = {
            'stress', 'stressed', 'overwhelmed', 'pressure', 'deadline', 'worried', 'anxiety',
            'anxious', 'panic', 'busy', 'rushed', 'hectic', 'exhausted', 'tired', 'overworked',
            'crash', 'crashed', 'market', 'financial', 'bills', 'debt', 'money', 'afford'
        }
        
        # Emotional distress
        self.distress_words = {
            'depressed', 'depression', 'sad', 'crying', 'lonely', 'alone', 'empty', 'hopeless',
            'worthless', 'broken', 'lost', 'miserable', 'grief', 'pain', 'hurt', 'numb'
        }
        
        # Self-harm indicators (avoid single words that appear in threats)
        self.self_harm_phrases = [
            'kill myself', 'hurt myself', 'harm myself', 'cut myself', 'end my life', 
            'want to die', 'suicide', 'overdose', 'pills', 'hang myself'
        ]
        
        # Threat/violence keywords (individual words that indicate threats)
        self.threat_words = {
            'murder', 'attack', 'violence', 'weapon',
            'shoot', 'stab', 'beat', 'fight', 'threat', 'threatening'
        }
        
        # Unsafe environment
        self.unsafe_words = {
            'abuse', 'abusive', 'threatening', 'scared', 'fear', 'afraid', 'dangerous',
            'unsafe', 'violence', 'hit', 'hits', 'hitting', 'stalking', 'following'
        }
        
        # Neutral indicators
        self.neutral_words = {
            'is', 'are', 'was', 'were', 'the', 'a', 'an', 'pages', 'long', 'written',
            'located', 'contains', 'includes', 'available', 'made', 'weighs', 'measures'
        }
    
    def classify(self, text: str) -> Dict:
        """
        Classify text into categories
        
        Returns dict with:
        - text: original text
        - predictions: list of {label, score}
        - all_scores: dict of all scores
        - primary_category: main category
        - confidence: confidence score
        - sentiment: positive/negative/neutral
        - emotion: emotion category
        """
        # Handle edge cases
        if text is None:
            text = ""
        text = str(text).strip()
        if not text:
            # Return neutral for empty text
            return {
                'text': '',
                'predictions': [{'label': 'neutral', 'score': 0.5}],
                'all_scores': {'neutral': 0.5},
                'primary_category': 'neutral',
                'confidence': 0.5,
                'sentiment': 'neutral',
                'emotion': 'neutral'
            }
        
        text_lower = text.lower()
        words = set(text_lower.split())
        
        # Initialize scores
        scores = {
            'positive': 0.0,
            'negative': 0.0,
            'neutral': 0.0,
            'stress': 0.0,
            'emotional_distress': 0.0,
            'self_harm_low': 0.0,
            'self_harm_high': 0.0,
            'unsafe_environment': 0.0
        }
        
        # Check for specific threat patterns FIRST (before counting keywords)
        is_threat = False
        is_self_harm = False
        
        # Check for self-harm patterns FIRST (to avoid confusion with threats)
        self_harm_detected = any(phrase in text_lower for phrase in self.self_harm_phrases)
        if self_harm_detected:
            scores['self_harm_high'] = 0.95
            scores['emotional_distress'] = 0.90
            is_self_harm = True
        
        # Threats to others - check for "kill/hurt [person]" patterns (only if NOT self-harm)
        if not self_harm_detected and any(pattern in text_lower for pattern in [
            'kill you', 'kill him', 'kill her', 'kill them', 'kill someone', 'kill everyone',
            'murder you', 'murder him', 'murder her', 'murder them', 'murder someone',
            'hurt you', 'hurt him', 'hurt her', 'hurt them', 'hurt someone', 'hurt everyone',
            'attack you', 'attack him', 'attack her', 'attack them', 'attack the',
            'will kill', 'going to kill', 'gonna kill',
            'will hurt', 'going to hurt', 'gonna hurt',
            'will murder', 'going to murder', 'gonna murder',
            'will attack', 'going to attack', 'planning to attack', 'planning an attack',
            'destroy you', 'destroy him', 'destroy her', 'destroy them',
            'beat you', 'beat him', 'beat her', 'beat them',
            'shoot you', 'shoot him', 'shoot her', 'shoot them', 'shooting up',
            'stab you', 'stab him', 'stab her', 'stab them',
            'bomb', 'blow up', 'explosion', 'explosive',
            'bring gun', 'bringing gun', 'with a gun', 'have a gun',
            'bring knife', 'bringing knife', 'with a knife', 'have a knife',
            'bring weapon', 'bringing weapon', 'with weapon', 'have weapon',
            'death threat', 'threaten', 'threat',
            'violence', 'violent', 'rampage', 'massacre'
        ]):
            scores['unsafe_environment'] = 0.98
            scores['negative'] = 0.90
            is_threat = True
        
        # Count keyword matches (only if not already classified as threat/self-harm)
        positive_count = sum(1 for word in self.positive_words if word in text_lower)
        negative_count = sum(1 for word in self.negative_words if word in text_lower)
        stress_count = sum(1 for word in self.stress_words if word in text_lower)
        distress_count = sum(1 for word in self.distress_words if word in text_lower)
        threat_count = sum(1 for word in self.threat_words if word in text_lower)
        unsafe_count = sum(1 for word in self.unsafe_words if word in text_lower)
        
        # Calculate scores based on counts (skip if already high-confidence classified)
        total_words = len(words)
        if total_words > 0 and not is_threat and not is_self_harm:
            if positive_count > 0:
                scores['positive'] = min(0.95, 0.5 + (positive_count / total_words) * 2)
            
            if negative_count > 0:
                scores['negative'] = max(scores['negative'], min(0.95, 0.5 + (negative_count / total_words) * 2))
            
            if stress_count > 0:
                scores['stress'] = min(0.95, 0.5 + (stress_count / total_words) * 2)
            
            if distress_count > 0:
                scores['emotional_distress'] = max(scores['emotional_distress'], 
                                                   min(0.95, 0.5 + (distress_count / total_words) * 2))
            
            if threat_count > 0:
                scores['unsafe_environment'] = max(scores['unsafe_environment'], 
                                                   min(0.95, 0.5 + (threat_count / total_words) * 2))
            
            if unsafe_count > 0:
                scores['unsafe_environment'] = max(scores['unsafe_environment'], 
                                                   min(0.95, 0.5 + (unsafe_count / total_words) * 2))
        
        # If nothing detected, default to neutral with high confidence
        if all(score < 0.3 for score in scores.values()):
            # Check if it's factual/neutral language
            neutral_count = sum(1 for word in self.neutral_words if word in words)
            has_opinion_words = any(word in text_lower for word in ['love', 'hate', 'amazing', 'terrible', 'great', 'awful'])
            
            if neutral_count > len(words) * 0.25 or not has_opinion_words:
                scores['neutral'] = 0.85  # Strong neutral signal for factual text
            else:
                scores['neutral'] = 0.65  # Moderate neutral for ambiguous text
        
        # Build predictions
        predictions = []
        for label, score in scores.items():
            if score > 0.5:
                predictions.append({
                    'label': label,
                    'score': float(score)
                })
        
        predictions.sort(key=lambda x: x['score'], reverse=True)
        
        # Determine sentiment
        sentiment = 'neutral'
        if scores['positive'] > 0.5:
            sentiment = 'positive'
        elif scores['negative'] > 0.5 or scores['stress'] > 0.5:
            sentiment = 'negative'
        
        # Determine emotion
        emotion = 'neutral'
        if scores['self_harm_high'] > 0.5 or scores['self_harm_low'] > 0.5:
            emotion = 'crisis'
        elif scores['unsafe_environment'] > 0.5:
            emotion = 'unsafe'
        elif scores['emotional_distress'] > 0.5:
            emotion = 'emotional_distress'
        elif scores['stress'] > 0.5:
            emotion = 'stress'
        elif scores['positive'] > 0.6:
            emotion = 'positive'
        elif scores['negative'] > 0.6:
            emotion = 'negative'
        
        # Primary category
        primary_category = max(scores.keys(), key=lambda k: scores[k])
        confidence = scores[primary_category]
        
        return {
            'text': text,
            'predictions': predictions,
            'all_scores': scores,
            'primary_category': primary_category,
            'confidence': float(confidence),
            'sentiment': sentiment,
            'emotion': emotion
        }

# Test
if __name__ == '__main__':
    classifier = SimpleClassifier()
    
    test_texts = [
        "The movie was boring and way too long",
        "The product broke after just two days of use",
        "This book is 300 pages long and written in English",
        "i will kill you",
        "I will kill him",
        "I want to kill myself",
        "The stock market crashed yesterday",
        "This movie was fantastic"
    ]
    
    for text in test_texts:
        result = classifier.classify(text)
        print(f"\nText: {text}")
        print(f"Sentiment: {result['sentiment']}")
        print(f"Emotion: {result['emotion']}")
        print(f"Primary: {result['primary_category']} ({result['confidence']:.2f})")
        if result['predictions']:
            preds = [f"{p['label']}: {p['score']:.2f}" for p in result['predictions']]
            print(f"Predictions: {', '.join(preds)}")
