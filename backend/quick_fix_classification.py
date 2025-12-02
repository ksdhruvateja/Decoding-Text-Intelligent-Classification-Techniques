"""
Quick Fix for Classification Issues
===================================
Applies immediate fixes to classification logic without retraining
Run this to test the improved classification
"""

# This script tests the improved classification on the problematic examples

from multistage_classifier import MultiStageClassifier

def test_examples():
    """Test the problematic examples"""
    print("="*80)
    print("TESTING IMPROVED CLASSIFICATION")
    print("="*80)
    
    classifier = MultiStageClassifier()
    
    test_cases = [
        {
            "text": "I absolutely loved the new restaurant; the food was delicious and the staff were so friendly.",
            "expected": "positive/safe"
        },
        {
            "text": "The service was terrible, I had to wait an hour and the staff were rude",
            "expected": "stress/emotional_distress - concerning"
        },
        {
            "text": "I went to the store yesterday to buy some groceries",
            "expected": "neutral/safe"
        }
    ]
    
    print("\n")
    for i, test in enumerate(test_cases, 1):
        print(f"{'='*80}")
        print(f"Test {i}: {test['text']}")
        print(f"Expected: {test['expected']}")
        print(f"{'-'*80}")
        
        result = classifier.classify(test['text'])
        
        print(f"Result:")
        print(f"  Emotion: {result['emotion']}")
        print(f"  Sentiment: {result['sentiment']}")
        print(f"  Override Applied: {result.get('override_applied', False)}")
        if result.get('override_reason'):
            print(f"  Override Reason: {result['override_reason']}")
        
        if result['predictions']:
            print(f"  Predictions:")
            for pred in result['predictions']:
                print(f"    - {pred['label']}: {pred['confidence']:.2%} (threshold: {pred['threshold']:.2%})")
        
        print(f"\n  All Scores:")
        for label, score in result['all_scores'].items():
            print(f"    - {label}: {score:.2%}")
        
        # Check if correct
        is_correct = False
        if "positive" in test['expected'] and result['sentiment'] == 'safe' and result['emotion'] == 'positive':
            is_correct = True
        elif "concerning" in test['expected'] and result['sentiment'] == 'concerning':
            is_correct = True
        elif "neutral/safe" in test['expected'] and result['sentiment'] == 'safe' and result['emotion'] == 'neutral':
            is_correct = True
        
        print(f"\n  Status: {'✓ CORRECT' if is_correct else '✗ INCORRECT'}")
        print()
    
    print("="*80)
    print("Testing complete!")
    print("\nIf results are still incorrect, run:")
    print("  python retrain_with_corrections.py")
    print("  python train_advanced_optimized.py")


if __name__ == '__main__':
    test_examples()

