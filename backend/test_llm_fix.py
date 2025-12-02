"""
Test LLM Override Fix
=====================
Tests that the LLM override is no longer too aggressive
"""

from multistage_classifier import initialize_multistage_classifier

def test_llm_fix():
    """Test that LLM override doesn't cause false positives"""
    
    print("="*80)
    print("TESTING LLM OVERRIDE FIX")
    print("="*80)
    
    classifier = initialize_multistage_classifier()
    
    test_cases = [
        {
            "text": "I am so angry at you",
            "expected_emotion": ["stress", "emotional_distress", "unsafe_environment"],
            "expected_sentiment": ["concerning"],
            "should_not_be": "self_harm_high"
        },
        {
            "text": "I feel depressed",
            "expected_emotion": ["emotional_distress"],
            "expected_sentiment": ["concerning"],
            "should_not_be": "self_harm_high"
        },
        {
            "text": "I hate you so much",
            "expected_emotion": ["stress", "emotional_distress", "unsafe_environment"],
            "expected_sentiment": ["concerning"],
            "should_not_be": "self_harm_high"
        },
        {
            "text": "I want to kill myself",
            "expected_emotion": ["self_harm_high"],
            "expected_sentiment": ["high_risk"],
            "should_not_be": None  # This SHOULD be self_harm_high
        },
        {
            "text": "I'm going through a tough time",
            "expected_emotion": ["emotional_distress", "stress"],
            "expected_sentiment": ["concerning"],
            "should_not_be": "self_harm_high"
        },
    ]
    
    correct = 0
    total = len(test_cases)
    
    for i, case in enumerate(test_cases, 1):
        text = case["text"]
        result = classifier.classify(text)
        
        emotion = result.get('emotion', 'unknown')
        sentiment = result.get('sentiment', 'unknown')
        
        is_correct = (
            emotion in case["expected_emotion"] and
            sentiment in case["expected_sentiment"] and
            (case["should_not_be"] is None or emotion != case["should_not_be"])
        )
        
        if is_correct:
            correct += 1
            status = "[OK]"
        else:
            status = "[X]"
        
        print(f"\n{status} Test {i}: {text}")
        print(f"    Emotion: {emotion} | Sentiment: {sentiment}")
        print(f"    Expected: {case['expected_emotion']} ({case['expected_sentiment']})")
        if case["should_not_be"]:
            print(f"    Should NOT be: {case['should_not_be']}")
        
        # Show if LLM override was applied
        if result.get('llm_summary') and result['llm_summary'].get('adjustment'):
            print(f"    LLM Adjustment: {result['llm_summary']['adjustment']}")
    
    print(f"\n{'='*80}")
    print(f"RESULTS: {correct}/{total} correct ({correct/total*100:.1f}%)")
    print(f"{'='*80}")


if __name__ == '__main__':
    test_llm_fix()

