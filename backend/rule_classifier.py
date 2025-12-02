"""
Rule-based mental health and safety classifier
Uses keyword matching and patterns to detect stress, distress, and safety concerns
"""

import re
from typing import Dict, List

class RuleBasedToxicClassifier:
    def __init__(self):
        """Initialize the rule-based classifier for mental health and safety detection"""
        
        # Define patterns for mental health and safety categories
        self.patterns = {
            'neutral': [
                r'\bhow\b', r'\bwhat\b', r'\bwhen\b', r'\bwhere\b', r'\bwhy\b',
                r'\bexplain\b', r'\bdefine\b', r'\bdescribe\b', r'\binformation\b',
                r'\bfact\b', r'\bdata\b', r'\bresearch\b', r'\bstudy\b',
                r'\bquestion\b', r'\bwondering\b', r'\bcurious\b'
            ],
            'stress': [
                r'\bstress', r'\bpressure\b', r'\bbusy\b', r'\bexhaust',
                r'\btired\b', r'\bworn.*?out\b', r'\bburned.*?out\b', r'\bburnout\b',
                r'\boverwhelm', r'\btoo.*?much\b', r'\bcan\'t.*?keep.*?up\b',
                r'\bstrugg', r'\bdifficult\b', r'\bhard.*?time\b', r'\bchallenging\b',
                r'\bfrustrat', r'\bannoyed\b', r'\bupset\b'
            ],
            'unsafe_environment': [
                r'\bunsafe\b', r'\bdanger', r'\bthreat', r'\bscar', r'\bafraid\b',
                r'\bfear', r'\bviolence\b', r'\babuse', r'\bbully', r'\bharass',
                r'\battack', r'\bhurt.*?me\b', r'\bharm.*?me\b', r'\btrapped\b',
                r'\bescape\b', r'\bhelp.*?me\b', r'\bsomeone.*?hurting\b'
            ],
            'emotional_distress': [
                r'\bdepressed\b', r'\bdepress', r'\banxious\b', r'\banxiety\b',
                r'\bpanic', r'\bsad\b', r'\blonely\b', r'\bloneliness\b',
                r'\bhopeless\b', r'\bhelpless\b', r'\bempty\b', r'\bnumb\b',
                r'\bworthless\b', r'\buseless\b', r'\bbroken\b', r'\blost\b',
                r'\bcan\'t.*?take.*?it\b', r'\bcan\'t.*?handle\b', r'\bgive.*?up\b',
                r'\bno.*?point\b', r'\bwhat\'s.*?the.*?point\b'
            ],
            'self_harm_low': [
                r'\bafraid.*?i.*?might\b', r'\bworried.*?i.*?(might|could)\b',
                r'\bthinking.*?about.*?(harm|hurt)', r'\bcross.*?my.*?mind\b',
                r'\bscared.*?i.*?(might|could)\b', r'\btempted\b',
                r'\burge', r'\bimpuls', r'\bthoughts.*?about.*?(harm|hurt|end)',
                r'\bconcerned.*?i.*?(might|could)', r'\bafraid.*?of.*?what.*?i.*?might\b'
            ],
            'self_harm_high': [
                r'\bsuicid', r'\bkill.*?myself\b', r'\bend.*?my.*?life\b',
                r'\bharm.*?myself\b', r'\bhurt.*?myself\b', r'\bcut.*?myself\b',
                r'\bwant.*?to.*?die\b', r'\bdon\'t.*?want.*?to.*?live\b',
                r'\bend.*?it.*?all\b', r'\bcan\'t.*?go.*?on\b', r'\bno.*?reason.*?to.*?live\b',
                r'\bdo.*?something.*?harmful\b', r'\bmight.*?hurt.*?myself\b',
                r'\bplan.*?to.*?(die|kill|harm|hurt)', r'\bgonna.*?(kill|hurt|harm|end)',
                r'\blife.*?isn\'t.*?worth\b', r'\bbetter.*?off.*?dead\b'
            ]
        }
        
        # Positive/safe patterns that indicate wellbeing
        self.positive_patterns = [
            r'\bthank', r'\bgrateful\b', r'\bappreciat', r'\bhelpful\b',
            r'\bwonderful\b', r'\bamazing\b', r'\bexcellent\b', r'\bgreat\b',
            r'\blove\b', r'\bhappy\b', r'\bjoy', r'\bpleas', r'\bgood\b',
            r'\bbetter\b', r'\bimprove', r'\bprogress\b', r'\bhope', r'\boptimis'
        ]
        
    def classify(self, text: str, threshold: float = 0.5) -> Dict:
        """Classify text using rule-based approach"""
        text_lower = text.lower()
        
        # Check for positive content
        positive_score = sum(1 for pattern in self.positive_patterns 
                           if re.search(pattern, text_lower, re.IGNORECASE))
        
        # Calculate scores for each category
        scores = {}
        category_matches = {}
        
        for category, patterns in self.patterns.items():
            matches = sum(1 for pattern in patterns 
                         if re.search(pattern, text_lower, re.IGNORECASE))
            category_matches[category] = matches
            
            # Base score calculation
            if matches > 0:
                # Progressive scoring: first match = 50%, each additional = +20%
                base_score = min(0.95, 0.50 + (matches - 1) * 0.20)
            else:
                base_score = 0.0
            
            # Reduce scores for positive content
            if positive_score > 0 and category not in ['neutral']:
                multiplier = max(0.3, 1 - (positive_score * 0.20))
                base_score *= multiplier
            
            # Add small variance for realism
            import random
            variance = random.uniform(-0.02, 0.02)
            final_score = max(0.0, min(0.98, base_score + variance))
            
            scores[category] = round(final_score, 4)
        
        # Create predictions list
        predictions = []
        for label, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            if score >= threshold:
                predictions.append({
                    'label': label,
                    'confidence': score
                })
        
        # Determine risk level and sentiment
        high_risk = scores.get('self_harm_high', 0) >= threshold
        low_risk = scores.get('self_harm_low', 0) >= threshold
        distress = scores.get('emotional_distress', 0) >= threshold
        unsafe = scores.get('unsafe_environment', 0) >= threshold
        stress = scores.get('stress', 0) >= threshold
        neutral = scores.get('neutral', 0) >= threshold
        
        # Determine sentiment - prioritize positive content first
        if positive_score >= 1:
            # Positive content detected
            sentiment = 'safe'
            if positive_score >= 3:
                emotion_detected = 'very_positive'
            elif positive_score >= 2:
                emotion_detected = 'positive'
            else:
                emotion_detected = 'friendly'
        elif high_risk:
            sentiment = 'high_risk'
            emotion_detected = 'self_harm_high'
        elif low_risk:
            sentiment = 'concerning'
            emotion_detected = 'self_harm_low'
        elif unsafe:
            sentiment = 'concerning'
            emotion_detected = 'unsafe_environment'
        elif distress:
            sentiment = 'concerning'
            emotion_detected = 'emotional_distress'
        elif stress:
            sentiment = 'neutral'
            emotion_detected = 'stress'
        elif neutral:
            sentiment = 'neutral'
            emotion_detected = 'neutral'
        else:
            sentiment = 'neutral'
            emotion_detected = 'conversational'
        
        return {
            'text': text,
            'predictions': predictions,
            'all_scores': scores,
            'sentiment': sentiment,
            'emotion': emotion_detected
        }

# Create a global instance
rule_based_classifier = RuleBasedToxicClassifier()
