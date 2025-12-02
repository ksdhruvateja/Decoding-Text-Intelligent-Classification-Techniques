"""
Test Specific Classification Cases
==================================
Tests the three specific cases mentioned by the user
"""

from multistage_classifier import initialize_multistage_classifier

def test_specific_cases():
    """Test the three specific cases"""
    
    print("="*80)
    print("TESTING SPECIFIC CLASSIFICATION CASES")
    print("="*80)
    
    classifier = initialize_multistage_classifier()
    
    test_cases = [
        {
            "text": "Get lost, you piece of *",
            "expected_emotion": "stress or emotional_distress",
            "expected_sentiment": "concerning",
            "should_not_be": "neutral",
            "description": "Hostile/aggressive language"
        },
        {
            "text": "I love how supportive this community is",
            "expected_emotion": "positive or neutral",
            "expected_sentiment": "safe",
            "should_not_have": ["emotional_distress", "stress", "unsafe_environment"],
            "description": "Purely positive message"
        },
        {
            "text": "This app keeps crashing and it's so frustrating",
            "expected_emotion": "stress or emotional_distress",
            "expected_sentiment": "concerning or neutral",
            "description": "Frustration/distress (should be correct)"
        }
    ]
    
    print(f"\nTesting {len(test_cases)} specific cases...\n")
    
    for i, test in enumerate(test_cases, 1):
        text = test['text']
        expected_emotion = test['expected_emotion']
        expected_sentiment = test.get('expected_sentiment', 'any')
        description = test['description']
        
        print(f"{'='*80}")
        print(f"Test {i}: {description}")
        print(f"{'-'*80}")
        print(f"Text: {text}")
        print(f"Expected: {expected_emotion} ({expected_sentiment})")
        
        # Classify
        result = classifier.classify(text)
        
        emotion = result.get('emotion', 'unknown')
        sentiment = result.get('sentiment', 'unknown')
        predictions = result.get('predictions', [])
        all_scores = result.get('all_scores', {})
        
        print(f"\nResult:")
        print(f"  Emotion: {emotion}")
        print(f"  Sentiment: {sentiment}")
        if predictions:
            print(f"  Top Predictions:")
            for pred in predictions[:3]:
                print(f"    - {pred['label']}: {pred['confidence']*100:.1f}%")
        
        print(f"\n  All Scores:")
        for label, score in sorted(all_scores.items(), key=lambda x: x[1], reverse=True):
            if score > 0.1:  # Only show scores above 10%
                print(f"    {label}: {score*100:.1f}%")
        
        # Check correctness
        expected_emotions = [e.strip() for e in expected_emotion.split('or')]
        is_correct_emotion = emotion in expected_emotions or \
                            any(p['label'] in expected_emotions for p in predictions[:2])
        is_correct_sentiment = expected_sentiment == 'any' or sentiment == expected_sentiment
        
        # Check for should_not_be
        if 'should_not_be' in test:
            should_not_be = test['should_not_be']
            is_wrong = emotion == should_not_be or \
                      any(p['label'] == should_not_be for p in predictions[:2])
            if is_wrong:
                print(f"\n  [X] INCORRECT: Should NOT be {should_not_be}")
                is_correct_emotion = False
        
        # Check for should_not_have
        if 'should_not_have' in test:
            should_not_have = test['should_not_have']
            has_wrong_labels = any(
                p['label'] in should_not_have and p['confidence'] > 0.3 
                for p in predictions
            )
            if has_wrong_labels:
                wrong_labels = [p['label'] for p in predictions if p['label'] in should_not_have and p['confidence'] > 0.3]
                print(f"\n  [X] INCORRECT: Should NOT have high scores for: {', '.join(wrong_labels)}")
                is_correct_emotion = False
        
        if is_correct_emotion and is_correct_sentiment:
            print(f"\n  [OK] CORRECT classification")
        else:
            print(f"\n  [X] INCORRECT classification")
            if not is_correct_emotion:
                print(f"      Expected emotion: {expected_emotion}, Got: {emotion}")
            if not is_correct_sentiment:
                print(f"      Expected sentiment: {expected_sentiment}, Got: {sentiment}")
        
        print()
    
    print("="*80)
    print("TESTING COMPLETE")
    print("="*80)


if __name__ == '__main__':
    test_specific_cases()

