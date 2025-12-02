"""
Comprehensive Model Logic Test
==============================
Tests all aspects of the classification system to ensure accuracy
"""

from multistage_classifier import initialize_multistage_classifier

def test_model_logic():
    """Test all model logic comprehensively"""
    
    print("="*80)
    print("COMPREHENSIVE MODEL LOGIC TEST")
    print("="*80)
    
    classifier = initialize_multistage_classifier()
    
    test_cases = [
        # Test 1: Positive motivational (should be neutral, NOT stress/self-harm)
        {
            "text": "I will ace this test",
            "expected_emotion": ["neutral", "positive"],
            "expected_sentiment": ["safe"],
            "should_not_be": ["stress", "self_harm_high", "self_harm_low", "emotional_distress"]
        },
        {
            "text": "I will crush this presentation",
            "expected_emotion": ["neutral", "positive"],
            "expected_sentiment": ["safe"],
            "should_not_be": ["stress", "self_harm_high", "self_harm_low"]
        },
        
        # Test 2: Self-harm (should be self_harm_high with high threshold)
        {
            "text": "I want to kill myself",
            "expected_emotion": ["self_harm_high"],
            "expected_sentiment": ["high_risk"],
            "should_not_be": ["neutral", "stress"]
        },
        {
            "text": "I'm planning to hurt myself tonight",
            "expected_emotion": ["self_harm_high"],
            "expected_sentiment": ["high_risk"],
            "should_not_be": ["neutral", "stress"]
        },
        
        # Test 3: Threats to others (should be unsafe_environment, NOT self-harm)
        {
            "text": "I want to hurt you",
            "expected_emotion": ["unsafe_environment"],
            "expected_sentiment": ["concerning"],
            "should_not_be": ["self_harm_high", "self_harm_low"]
        },
        {
            "text": "I'm going to get you",
            "expected_emotion": ["unsafe_environment"],
            "expected_sentiment": ["concerning"],
            "should_not_be": ["self_harm_high", "self_harm_low"]
        },
        
        # Test 4: Stress/emotional distress (should NOT trigger on positive)
        {
            "text": "This is so frustrating",
            "expected_emotion": ["stress", "emotional_distress"],
            "expected_sentiment": ["concerning"],
            "should_not_be": ["neutral", "self_harm_high"]
        },
        {
            "text": "I'm really frustrated with this situation",
            "expected_emotion": ["stress", "emotional_distress"],
            "expected_sentiment": ["concerning"],
            "should_not_be": ["neutral", "self_harm_high"]
        },
        
        # Test 5: Positive content (should be safe)
        {
            "text": "I love how supportive this community is",
            "expected_emotion": ["positive", "neutral"],
            "expected_sentiment": ["safe"],
            "should_not_be": ["self_harm_high", "self_harm_low", "stress", "emotional_distress"]
        },
        
        # Test 6: Neutral content
        {
            "text": "I went to the store yesterday",
            "expected_emotion": ["neutral"],
            "expected_sentiment": ["safe"],
            "should_not_be": ["self_harm_high", "self_harm_low", "stress", "emotional_distress"]
        },
        
        # Test 7: Hostile language (should be stress/emotional_distress, NOT self-harm)
        {
            "text": "Get lost, you piece of *",
            "expected_emotion": ["stress", "emotional_distress", "unsafe_environment"],
            "expected_sentiment": ["concerning"],
            "should_not_be": ["self_harm_high", "self_harm_low", "neutral"]
        },
        
        # Test 8: Low confidence self-harm (should NOT trigger if below 0.8)
        {
            "text": "I sometimes feel sad",
            "expected_emotion": ["emotional_distress", "neutral"],
            "expected_sentiment": ["concerning", "safe"],
            "should_not_be": ["self_harm_high"]  # Should not trigger unless very high confidence
        },
    ]
    
    results = {
        "total": len(test_cases),
        "correct": 0,
        "incorrect": 0,
        "errors": []
    }
    
    for i, case in enumerate(test_cases, 1):
        try:
            text = case["text"]
            result = classifier.classify(text)
            
            emotion = result.get('emotion', 'unknown')
            sentiment = result.get('sentiment', 'unknown')
            predictions = result.get('predictions', [])
            
            # Check if emotion matches expected
            emotion_match = emotion in case["expected_emotion"]
            sentiment_match = sentiment in case["expected_sentiment"]
            
            # Check if it's NOT in should_not_be list
            not_forbidden = emotion not in case["should_not_be"]
            
            # Check predictions too
            pred_emotions = [p['label'] for p in predictions[:2]]
            pred_match = any(pe in case["expected_emotion"] for pe in pred_emotions)
            pred_forbidden = not any(pe in case["should_not_be"] for pe in pred_emotions)
            
            is_correct = (
                (emotion_match or pred_match) and
                sentiment_match and
                not_forbidden and
                pred_forbidden
            )
            
            if is_correct:
                results["correct"] += 1
                status = "[OK]"
            else:
                results["incorrect"] += 1
                status = "[X]"
                results["errors"].append({
                    "test": i,
                    "text": text,
                    "got": f"{emotion} ({sentiment})",
                    "expected": f"{case['expected_emotion']} ({case['expected_sentiment']})",
                    "should_not_be": case["should_not_be"]
                })
            
            print(f"\n{status} Test {i}: {text[:50]}{'...' if len(text) > 50 else ''}")
            print(f"    Got: {emotion} ({sentiment})")
            print(f"    Expected: {case['expected_emotion']} ({case['expected_sentiment']})")
            if case["should_not_be"]:
                print(f"    Should NOT be: {case['should_not_be']}")
            
            # Show top predictions
            if predictions:
                pred_strs = [f"{p['label']} ({p['confidence']*100:.1f}%)" for p in predictions[:3]]
                print(f"    Top predictions: {', '.join(pred_strs)}")
            
            # Show override if applied
            if result.get('override_applied'):
                print(f"    Override: {result.get('override_reason', 'N/A')}")
                
        except Exception as e:
            results["incorrect"] += 1
            results["errors"].append({
                "test": i,
                "text": case["text"],
                "error": str(e)
            })
            print(f"\n[ERROR] Test {i}: {case['text'][:50]}")
            print(f"    Error: {e}")
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total Tests: {results['total']}")
    print(f"Correct: {results['correct']}")
    print(f"Incorrect: {results['incorrect']}")
    print(f"Accuracy: {results['correct']/results['total']*100:.1f}%")
    
    if results["errors"]:
        print(f"\n{'='*80}")
        print("ERRORS FOUND:")
        print(f"{'='*80}")
        for error in results["errors"]:
            print(f"\nTest {error['test']}: {error['text']}")
            if 'error' in error:
                print(f"  Error: {error['error']}")
            else:
                print(f"  Got: {error['got']}")
                print(f"  Expected: {error['expected']}")
                print(f"  Should NOT be: {error['should_not_be']}")
    
    print(f"\n{'='*80}")
    
    return results


if __name__ == '__main__':
    test_model_logic()

