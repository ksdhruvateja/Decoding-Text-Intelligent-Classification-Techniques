"""
Test Classification Fix
=======================
Tests the specific issue: "the service was terrible there"
Should classify as stress/emotional_distress - concerning
"""

from multistage_classifier import MultiStageClassifier

def test_specific_case():
    """Test the specific problematic case"""
    print("="*80)
    print("TESTING CLASSIFICATION FIX")
    print("="*80)
    
    classifier = MultiStageClassifier()
    
    test_cases = [
        {
            "text": "the service was terrible there",
            "expected_emotion": "stress",
            "expected_sentiment": "concerning"
        },
        {
            "text": "I absolutely loved the new restaurant; the food was delicious and the staff were so friendly.",
            "expected_emotion": "positive",
            "expected_sentiment": "safe"
        },
        {
            "text": "I went to the store yesterday to buy some groceries",
            "expected_emotion": "neutral",
            "expected_sentiment": "safe"
        },
        {
            "text": "The service was terrible, I had to wait an hour and the staff were rude",
            "expected_emotion": "stress",
            "expected_sentiment": "concerning"
        },
    ]
    
    print("\n")
    all_correct = True
    
    for i, test in enumerate(test_cases, 1):
        print(f"{'='*80}")
        print(f"Test {i}: {test['text']}")
        print(f"Expected: {test['expected_emotion']}/{test['expected_sentiment']}")
        print(f"{'-'*80}")
        
        result = classifier.classify(test['text'])
        
        print(f"Result:")
        print(f"  Emotion: {result['emotion']}")
        print(f"  Sentiment: {result['sentiment']}")
        print(f"  Base Sentiment: {result.get('base_sentiment', 'N/A')}")
        print(f"  Override Applied: {result.get('override_applied', False)}")
        
        if result['predictions']:
            print(f"  Predictions:")
            for pred in result['predictions']:
                print(f"    - {pred['label']}: {pred['confidence']:.2%} (threshold: {pred['threshold']:.2%})")
        
        print(f"\n  All Scores:")
        for label, score in result['all_scores'].items():
            print(f"    - {label}: {score:.2%}")
        
        # Check correctness
        is_correct = (result['emotion'] == test['expected_emotion'] and 
                     result['sentiment'] == test['expected_sentiment'])
        
        if is_correct:
            print(f"\n  Status: ✓ CORRECT")
        else:
            print(f"\n  Status: ✗ INCORRECT")
            print(f"    Expected: {test['expected_emotion']}/{test['expected_sentiment']}")
            print(f"    Got: {result['emotion']}/{result['sentiment']}")
            all_correct = False
        
        print()
    
    print("="*80)
    if all_correct:
        print("✓ ALL TESTS PASSED!")
    else:
        print("✗ SOME TESTS FAILED")
    print("="*80)
    
    return all_correct


if __name__ == '__main__':
    test_specific_case()

