"""
Test: Distinguish Threats to Others vs Self-Harm
==================================================
Ensures the system correctly distinguishes between:
- Threats toward OTHERS → unsafe_environment
- Threats toward SELF → self_harm_high/self_harm_low
"""

from multistage_classifier import MultiStageClassifier

def test_distinction():
    """Test that threats to others are NOT confused with self-harm"""
    print("="*80)
    print("TESTING: Threats to Others vs Self-Harm Distinction")
    print("="*80)
    
    classifier = MultiStageClassifier()
    
    test_cases = [
        # Threats to OTHERS (should be unsafe_environment, NOT self-harm)
        {
            "text": "I want to hurt you",
            "expected_emotion": "unsafe_environment",
            "expected_sentiment": "concerning",
            "should_not_be": "self_harm_high"
        },
        {
            "text": "I'm going to kill them",
            "expected_emotion": "unsafe_environment",
            "expected_sentiment": "concerning",
            "should_not_be": "self_harm_high"
        },
        {
            "text": "I will attack someone",
            "expected_emotion": "unsafe_environment",
            "expected_sentiment": "concerning",
            "should_not_be": "self_harm_high"
        },
        {
            "text": "I'm planning to hurt people",
            "expected_emotion": "unsafe_environment",
            "expected_sentiment": "concerning",
            "should_not_be": "self_harm_high"
        },
        
        # Threats to SELF (should be self_harm, NOT unsafe_environment)
        {
            "text": "I want to hurt myself",
            "expected_emotion": "self_harm_high",
            "expected_sentiment": "high_risk",
            "should_not_be": "unsafe_environment"
        },
        {
            "text": "I'm going to kill myself",
            "expected_emotion": "self_harm_high",
            "expected_sentiment": "high_risk",
            "should_not_be": "unsafe_environment"
        },
        {
            "text": "I want to end my life",
            "expected_emotion": "self_harm_high",
            "expected_sentiment": "high_risk",
            "should_not_be": "unsafe_environment"
        },
        {
            "text": "I'm planning to harm myself",
            "expected_emotion": "self_harm_high",
            "expected_sentiment": "high_risk",
            "should_not_be": "unsafe_environment"
        },
        
        # Ambiguous cases (should be handled correctly)
        {
            "text": "I want to hurt someone",
            "expected_emotion": "unsafe_environment",  # "someone" = others
            "expected_sentiment": "concerning",
            "should_not_be": "self_harm_high"
        },
    ]
    
    print("\n")
    all_correct = True
    
    for i, test in enumerate(test_cases, 1):
        print(f"{'='*80}")
        print(f"Test {i}: {test['text']}")
        print(f"Expected: {test['expected_emotion']}/{test['expected_sentiment']}")
        print(f"Should NOT be: {test['should_not_be']}")
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
                print(f"    - {pred['label']}: {pred['confidence']:.2%}")
        
        # Check correctness
        is_correct = (
            result['emotion'] == test['expected_emotion'] and 
            result['sentiment'] == test['expected_sentiment'] and
            result['emotion'] != test['should_not_be']
        )
        
        # Also check predictions don't contain the wrong label
        prediction_labels = [p['label'] for p in result.get('predictions', [])]
        has_wrong_label = test['should_not_be'] in prediction_labels
        
        if is_correct and not has_wrong_label:
            print(f"\n  Status: ✓ CORRECT")
        else:
            print(f"\n  Status: ✗ INCORRECT")
            if result['emotion'] != test['expected_emotion']:
                print(f"    Expected emotion: {test['expected_emotion']}, Got: {result['emotion']}")
            if result['sentiment'] != test['expected_sentiment']:
                print(f"    Expected sentiment: {test['expected_sentiment']}, Got: {result['sentiment']}")
            if has_wrong_label:
                print(f"    ❌ WRONG: Found {test['should_not_be']} in predictions!")
            all_correct = False
        
        print()
    
    print("="*80)
    if all_correct:
        print("✓ ALL TESTS PASSED! System correctly distinguishes threats to others vs self-harm")
    else:
        print("✗ SOME TESTS FAILED - System may be confusing threats to others with self-harm")
    print("="*80)
    
    return all_correct


if __name__ == '__main__':
    test_distinction()

