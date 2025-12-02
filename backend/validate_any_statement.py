"""
Comprehensive Validation System
================================
Tests the model on diverse statements to ensure accuracy on ANY input
"""

import json
from multistage_classifier import MultiStageClassifier
from typing import List, Dict, Tuple

class StatementValidator:
    """Validates classification on diverse statements"""
    
    def __init__(self):
        self.classifier = MultiStageClassifier()
        self.test_results = []
    
    def test_statement(self, text: str, expected_emotion: str = None, 
                      expected_sentiment: str = None) -> Dict:
        """Test a single statement"""
        result = self.classifier.classify(text)
        
        test_result = {
            'text': text,
            'predicted_emotion': result['emotion'],
            'predicted_sentiment': result['sentiment'],
            'expected_emotion': expected_emotion,
            'expected_sentiment': expected_sentiment,
            'all_scores': result['all_scores'],
            'predictions': result.get('predictions', []),
            'override_applied': result.get('override_applied', False),
            'override_reason': result.get('override_reason'),
            'correct': True
        }
        
        # Check correctness
        if expected_emotion:
            test_result['correct'] = (result['emotion'] == expected_emotion)
        if expected_sentiment:
            test_result['correct'] = test_result['correct'] and (result['sentiment'] == expected_sentiment)
        
        return test_result
    
    def test_comprehensive_suite(self) -> Dict:
        """Test on comprehensive suite of statements"""
        print("="*80)
        print("COMPREHENSIVE VALIDATION - TESTING ANY STATEMENT")
        print("="*80)
        
        test_suite = [
            # Positive statements
            ("I absolutely loved the new restaurant; the food was delicious and the staff were so friendly.", 
             "positive", "safe"),
            ("I'm so happy with my new job! Everything is going great.", "positive", "safe"),
            ("The concert was incredible! Best night ever.", "positive", "safe"),
            ("I'm thrilled with how things are progressing.", "positive", "safe"),
            ("This is the best day I've had in a long time!", "positive", "safe"),
            
            # Negative complaints
            ("The service was terrible, I had to wait an hour and the staff were rude", 
             "stress", "concerning"),
            ("I'm very disappointed with the service. The wait was too long.", 
             "stress", "concerning"),
            ("I'm frustrated with the poor quality of service I received.", 
             "stress", "concerning"),
            ("I'm really disappointed with how things turned out.", 
             "stress", "concerning"),
            ("This situation is really frustrating me.", "stress", "concerning"),
            
            # Neutral activities
            ("I went to the store yesterday to buy some groceries", "neutral", "safe"),
            ("I'm going to work today as usual.", "neutral", "safe"),
            ("I have a meeting scheduled for this afternoon.", "neutral", "safe"),
            ("The meeting starts at 3 PM.", "neutral", "safe"),
            ("I'm planning to visit my family next month.", "neutral", "safe"),
            
            # Emotional distress
            ("I've been feeling really overwhelmed lately.", "emotional_distress", "concerning"),
            ("I'm struggling with my emotions right now.", "emotional_distress", "concerning"),
            ("I feel so sad and I don't know why.", "emotional_distress", "concerning"),
            
            # Self-harm (low)
            ("Sometimes I think about hurting myself.", "self_harm_low", "concerning"),
            ("I've been having thoughts about self-harm.", "self_harm_low", "concerning"),
            
            # Self-harm (high)
            ("I want to hurt myself and end my pain", "self_harm_high", "high_risk"),
            ("I've been thinking about ending it all", "self_harm_high", "high_risk"),
            ("I have a plan to kill myself", "self_harm_high", "high_risk"),
        ]
        
        results = []
        correct_count = 0
        
        for text, expected_emotion, expected_sentiment in test_suite:
            result = self.test_statement(text, expected_emotion, expected_sentiment)
            results.append(result)
            
            if result['correct']:
                correct_count += 1
                status = "✓ CORRECT"
            else:
                status = "✗ INCORRECT"
            
            print(f"\n{status}: {text[:60]}...")
            print(f"  Expected: {expected_emotion}/{expected_sentiment}")
            print(f"  Got: {result['predicted_emotion']}/{result['predicted_sentiment']}")
            if result['override_applied']:
                print(f"  Override: {result['override_reason']}")
        
        accuracy = correct_count / len(test_suite)
        
        print("\n" + "="*80)
        print(f"VALIDATION RESULTS: {correct_count}/{len(test_suite)} correct ({accuracy:.2%})")
        print("="*80)
        
        return {
            'total': len(test_suite),
            'correct': correct_count,
            'accuracy': accuracy,
            'results': results
        }
    
    def test_custom_statements(self, statements: List[str]) -> Dict:
        """Test on custom statements"""
        print("\n" + "="*80)
        print("TESTING CUSTOM STATEMENTS")
        print("="*80)
        
        results = []
        for text in statements:
            result = self.test_statement(text)
            results.append(result)
            
            print(f"\nText: {text}")
            print(f"  Emotion: {result['predicted_emotion']}")
            print(f"  Sentiment: {result['predicted_sentiment']}")
            if result['predictions']:
                print(f"  Predictions:")
                for pred in result['predictions']:
                    print(f"    - {pred['label']}: {pred['confidence']:.2%}")
        
        return {'results': results}


def main():
    """Run comprehensive validation"""
    validator = StatementValidator()
    
    # Test comprehensive suite
    suite_results = validator.test_comprehensive_suite()
    
    # Test your specific examples
    print("\n" + "="*80)
    print("TESTING YOUR SPECIFIC EXAMPLES")
    print("="*80)
    
    your_examples = [
        "I absolutely loved the new restaurant; the food was delicious and the staff were so friendly.",
        "The service was terrible, I had to wait an hour and the staff were rude",
        "I went to the store yesterday to buy some groceries"
    ]
    
    validator.test_custom_statements(your_examples)
    
    # Save results
    with open('validation_results.json', 'w') as f:
        json.dump({
            'comprehensive_suite': suite_results,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    
    print("\n✓ Validation complete! Results saved to validation_results.json")


if __name__ == '__main__':
    from datetime import datetime
    main()

