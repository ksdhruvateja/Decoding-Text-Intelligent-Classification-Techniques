"""
GPT/LLM-Powered Classifier for Maximum Accuracy
===============================================
Uses OpenAI GPT or Hugging Face LLMs for superior classification
"""

import os
import json
from typing import Dict, Optional
import torch

class GPTClassifier:
    """GPT-based classifier using OpenAI API"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.enabled = self.api_key is not None
        
        if self.enabled:
            try:
                import openai
                self.client = openai.OpenAI(api_key=self.api_key)
                print("[GPT] GPT classifier initialized")
            except ImportError:
                print("[GPT] OpenAI library not installed. Install with: pip install openai")
                self.enabled = False
        else:
            print("[GPT] OpenAI API key not found. GPT classifier disabled.")
    
    def classify(self, text: str) -> Dict:
        """Classify text using GPT"""
        if not self.enabled:
            return None
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # or "gpt-4" for better accuracy
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert mental health text classifier. 
                        Classify the text into one of these categories:
                        - neutral: Normal, everyday statements
                        - stress: Complaints, frustrations, negative experiences
                        - emotional_distress: Sadness, anxiety, overwhelming feelings
                        - self_harm_low: Thoughts about self-harm (low risk)
                        - self_harm_high: Plans or intent to self-harm (high risk)
                        - unsafe_environment: Safety concerns, threats
                        
                        Also determine sentiment: safe, concerning, or high_risk
                        
                        Return JSON with: emotion, sentiment, confidence (0-1), reasoning"""
                    },
                    {
                        "role": "user",
                        "content": f"Classify this text: {text}"
                    }
                ],
                temperature=0.1,  # Low temperature for consistent results
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            return {
                'emotion': result.get('emotion', 'neutral'),
                'sentiment': result.get('sentiment', 'safe'),
                'confidence': result.get('confidence', 0.8),
                'reasoning': result.get('reasoning', ''),
                'source': 'gpt'
            }
        except Exception as e:
            print(f"[GPT] Error: {e}")
            return None


class HuggingFaceLLMClassifier:
    """Hugging Face LLM-based classifier"""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        self.model_name = model_name
        self.enabled = os.getenv("ENABLE_HF_LLM", "0").lower() not in ("0", "false", "no")
        
        if self.enabled:
            try:
                from transformers import pipeline
                self.pipeline = pipeline(
                    "text-classification",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    device=0 if torch.cuda.is_available() else -1
                )
                print(f"[HF-LLM] Hugging Face LLM classifier initialized: {model_name}")
            except Exception as e:
                print(f"[HF-LLM] Error initializing: {e}")
                self.enabled = False
        else:
            print("[HF-LLM] Hugging Face LLM classifier disabled")
    
    def classify(self, text: str) -> Dict:
        """Classify text using Hugging Face LLM"""
        if not self.enabled:
            return None
        
        try:
            result = self.pipeline(text)[0]
            # Map sentiment to our categories
            label = result['label'].lower()
            score = result['score']
            
            # Map to our emotion categories
            emotion_mapping = {
                'positive': 'neutral',
                'neutral': 'neutral',
                'negative': 'stress'
            }
            
            emotion = emotion_mapping.get(label, 'neutral')
            sentiment = 'safe' if label == 'positive' else 'concerning'
            
            return {
                'emotion': emotion,
                'sentiment': sentiment,
                'confidence': float(score),
                'source': 'hf_llm'
            }
        except Exception as e:
            print(f"[HF-LLM] Error: {e}")
            return None


class AdvancedLLMEnsemble:
    """Ensemble of LLM classifiers for maximum accuracy"""
    
    def __init__(self):
        self.gpt_classifier = GPTClassifier()
        self.hf_classifier = HuggingFaceLLMClassifier()
        self.enabled = self.gpt_classifier.enabled or self.hf_classifier.enabled
    
    def classify(self, text: str) -> Dict:
        """Get classification from multiple LLMs and ensemble"""
        results = []
        
        # Get GPT classification
        if self.gpt_classifier.enabled:
            gpt_result = self.gpt_classifier.classify(text)
            if gpt_result:
                results.append(gpt_result)
        
        # Get HF LLM classification
        if self.hf_classifier.enabled:
            hf_result = self.hf_classifier.classify(text)
            if hf_result:
                results.append(hf_result)
        
        if not results:
            return None
        
        # Ensemble: Weight GPT more heavily if available
        if len(results) == 2:
            # Both available: Weight GPT 70%, HF 30%
            gpt_weight = 0.7
            hf_weight = 0.3
            
            # Combine emotions (majority vote)
            emotions = [r['emotion'] for r in results]
            emotion = max(set(emotions), key=emotions.count)
            
            # Combine sentiments (majority vote)
            sentiments = [r['sentiment'] for r in results]
            sentiment = max(set(sentiments), key=sentiments.count)
            
            # Weighted confidence
            confidence = (results[0]['confidence'] * gpt_weight + 
                         results[1]['confidence'] * hf_weight)
        else:
            # Single result
            result = results[0]
            emotion = result['emotion']
            sentiment = result['sentiment']
            confidence = result['confidence']
        
        return {
            'emotion': emotion,
            'sentiment': sentiment,
            'confidence': confidence,
            'source': 'llm_ensemble',
            'num_models': len(results)
        }

