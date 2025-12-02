"""
Hybrid Classifier: LLM + BERT + Rule-Based Overrides
====================================================
Combines the best of all approaches:
1. LLM for semantic understanding
2. BERT for deep learning classification
3. Rule-based overrides for safety guarantees
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json
import re
import os
from typing import Dict, List, Tuple
from multistage_classifier import MultiStageClassifier

class HybridClassifier:
    """
    Hybrid classifier combining LLM, BERT, and rule-based approaches
    """
    
    def __init__(self, llm_model_path="./llm_checkpoints/rlhf_model", 
                 bert_model_path="checkpoints/best_mental_health_model.pt"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load LLM
        self.llm_loaded = False
        if llm_model_path and os.path.exists(llm_model_path):
            try:
                self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_path)
                self.llm_model = AutoModelForCausalLM.from_pretrained(llm_model_path)
                self.llm_model.to(self.device)
                self.llm_model.eval()
                self.llm_loaded = True
                print("✓ LLM loaded")
            except Exception as e:
                print(f"⚠ Could not load LLM: {e}")
                self.llm_loaded = False
        else:
            print("⚠ LLM model not found, using BERT + rules only")
        
        # Load BERT classifier
        self.bert_classifier = None
        try:
            self.bert_classifier = MultiStageClassifier(bert_model_path)
            print("✓ BERT classifier loaded")
        except Exception as e:
            print(f"⚠ Could not load BERT classifier: {e}")
        
        # Rule-based safety overrides
        self.safety_whitelist = {
            'marry', 'marriage', 'wedding', 'propose', 'love', 'cherish', 'adore',
            'confident', 'empowered', 'motivated', 'unstoppable', 'unbreakable',
            'indomitable', 'invincible', 'champion', 'warrior', 'fighter',
            'rise', 'soar', 'overcome', 'conquer', 'thrive', 'flourish'
        }
        
        self.safety_blacklist = {
            'kill myself', 'hurt myself', 'end my life', 'suicide', 'end it all',
            'want to die', 'planning to hurt', 'planning to kill'
        }
    
    def llm_classify(self, text: str) -> Dict:
        """Classify using LLM"""
        if not self.llm_loaded:
            return None
        
        prompt = f"Text: {text}\nLabel:"
        
        try:
            inputs = self.llm_tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.llm_model.generate(
                    **inputs,
                    max_new_tokens=50,
                    temperature=0.3,
                    do_sample=False
                )
            
            response = self.llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
            labels_text = response.split("Label:")[-1].strip()
            
            # Parse labels
            labels = {}
            for item in labels_text.split(','):
                if ':' in item:
                    key, value = item.split(':', 1)
                    labels[key.strip()] = value.strip()
            
            return labels
        except Exception as e:
            print(f"LLM classification error: {e}")
            return None
    
    def rule_based_override(self, text: str) -> Dict:
        """Apply rule-based safety overrides"""
        text_lower = text.lower()
        
        # Whitelist check - force safe classification
        has_whitelist = any(word in text_lower for word in self.safety_whitelist)
        if has_whitelist:
            # Check it's not actually dangerous
            has_blacklist = any(phrase in text_lower for phrase in self.safety_blacklist)
            if not has_blacklist:
                return {
                    'emotion': 'positive',
                    'sentiment': 'positive',
                    'stress': 'low',
                    'risk': 'none',
                    'source': 'rule_whitelist'
                }
        
        # Blacklist check - force risk classification
        has_blacklist = any(phrase in text_lower for phrase in self.safety_blacklist)
        if has_blacklist:
            return {
                'emotion': 'self_harm_high',
                'sentiment': 'negative',
                'stress': 'high',
                'risk': 'high',
                'source': 'rule_blacklist'
            }
        
        return None
    
    def bert_classify(self, text: str) -> Dict:
        """Classify using BERT"""
        if not self.bert_classifier:
            return None
        
        try:
            result = self.bert_classifier.classify(text)
            
            # Convert to standard format
            emotion = result.get('emotion', 'neutral')
            sentiment = result.get('sentiment', 'neutral')
            
            # Determine stress and risk from predictions
            predictions = result.get('predictions', [])
            all_scores = result.get('all_scores', {})
            
            if any(p['label'] == 'self_harm_high' for p in predictions):
                risk = 'high'
                stress = 'high'
            elif any(p['label'] == 'self_harm_low' for p in predictions):
                risk = 'medium'
                stress = 'high'
            elif any(p['label'] == 'stress' for p in predictions):
                stress = 'high'
                risk = 'low'
            elif any(p['label'] == 'emotional_distress' for p in predictions):
                stress = 'medium'
                risk = 'low'
            else:
                stress = 'low'
                risk = 'none'
            
            return {
                'emotion': emotion,
                'sentiment': sentiment,
                'stress': stress,
                'risk': risk,
                'source': 'bert',
                'confidence': predictions[0]['confidence'] if predictions else 0.5
            }
        except Exception as e:
            print(f"BERT classification error: {e}")
            return None
    
    def classify(self, text: str) -> Dict:
        """
        Hybrid classification combining all approaches
        Priority: Rules > LLM > BERT
        """
        # Step 1: Rule-based overrides (highest priority)
        rule_result = self.rule_based_override(text)
        if rule_result:
            return {
                **rule_result,
                'text': text,
                'method': 'hybrid_rule_override'
            }
        
        # Step 2: LLM classification
        llm_result = self.llm_classify(text)
        
        # Step 3: BERT classification
        bert_result = self.bert_classify(text)
        
        # Step 4: Combine results (weighted voting)
        if llm_result and bert_result:
            # Both available - use weighted average
            # Trust LLM more for semantic understanding, BERT for pattern matching
            final_result = {
                'text': text,
                'method': 'hybrid_llm_bert',
                'llm_result': llm_result,
                'bert_result': bert_result
            }
            
            # Combine emotions (prefer LLM)
            final_result['emotion'] = llm_result.get('emotion', bert_result.get('emotion', 'neutral'))
            final_result['sentiment'] = llm_result.get('sentiment', bert_result.get('sentiment', 'neutral'))
            final_result['stress'] = llm_result.get('stress', bert_result.get('stress', 'low'))
            final_result['risk'] = llm_result.get('risk', bert_result.get('risk', 'none'))
            
            return final_result
        elif llm_result:
            return {
                **llm_result,
                'text': text,
                'method': 'hybrid_llm_only'
            }
        elif bert_result:
            return {
                **bert_result,
                'text': text,
                'method': 'hybrid_bert_only'
            }
        else:
            # Fallback
            return {
                'text': text,
                'emotion': 'neutral',
                'sentiment': 'neutral',
                'stress': 'low',
                'risk': 'none',
                'method': 'hybrid_fallback'
            }


def main():
    """Test the hybrid classifier"""
    import os
    
    print("="*80)
    print("HYBRID CLASSIFIER TEST")
    print("="*80)
    
    classifier = HybridClassifier()
    
    test_cases = [
        "I will marry her",
        "I am unstoppable and nothing can hold me back",
        "I'm sick of people wasting my time",
        "I want to kill myself",
        "I went to the store",
        "I'm worried about my exam",
    ]
    
    print("\nTesting hybrid classifier...\n")
    for text in test_cases:
        result = classifier.classify(text)
        print(f"Text: {text}")
        print(f"  Emotion: {result.get('emotion')}")
        print(f"  Sentiment: {result.get('sentiment')}")
        print(f"  Stress: {result.get('stress')}")
        print(f"  Risk: {result.get('risk')}")
        print(f"  Method: {result.get('method')}")
        print()


if __name__ == '__main__':
    import os
    main()

