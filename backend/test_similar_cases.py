"""
Test Similar Cases
==================
Tests that similar cases are properly detected and analyzed
"""

from multistage_classifier import initialize_multistage_classifier

def test_similar_cases():
    """Test that similar cases are properly handled"""
    
    print("="*80)
    print("TESTING SIMILAR CASES - COMPREHENSIVE ANALYSIS")
    print("="*80)
    
    classifier = initialize_multistage_classifier()
    
    # Test similar hostile cases
    hostile_cases = [
        "Get lost, you piece of trash",
        "Go away, idiot",
        "Shut up, you jerk",
        "Leave me alone, you moron",
        "Fuck off, you loser",
        "You're so stupid",
        "I hate you",
        "You're terrible"
    ]
    
    # Test similar positive cases
    positive_cases = [
        "I love how supportive this community is",
        "This community is amazing",
        "I love this supportive group",
        "Grateful for this community",
        "Thankful for the support",
        "This is so helpful",
        "I appreciate this community",
        "Love the supportive people here"
    ]
    
    print("\n" + "="*80)
    print("HOSTILE/AGGRESSIVE CASES")
    print("="*80)
    
    for text in hostile_cases:
        result = classifier.classify(text)
        emotion = result.get('emotion', 'unknown')
        sentiment = result.get('sentiment', 'unknown')
        analysis = result.get('analysis_details', {})
        patterns = analysis.get('detected_patterns', [])
        
        print(f"\nText: {text}")
        print(f"  Emotion: {emotion} | Sentiment: {sentiment}")
        if patterns:
            print(f"  Patterns: {', '.join([p['type'] for p in patterns])}")
        indicators = analysis.get('key_indicators', [])
        if indicators:
            print(f"  Indicators: {', '.join(indicators[:3])}")
        
        # Check if correctly classified
        is_correct = emotion in ['stress', 'emotional_distress'] and sentiment == 'concerning'
        status = "[OK]" if is_correct else "[X]"
        print(f"  {status} {'Correct' if is_correct else 'Needs review'}")
    
    print("\n" + "="*80)
    print("POSITIVE CASES")
    print("="*80)
    
    for text in positive_cases:
        result = classifier.classify(text)
        emotion = result.get('emotion', 'unknown')
        sentiment = result.get('sentiment', 'unknown')
        analysis = result.get('analysis_details', {})
        patterns = analysis.get('detected_patterns', [])
        
        print(f"\nText: {text}")
        print(f"  Emotion: {emotion} | Sentiment: {sentiment}")
        if patterns:
            print(f"  Patterns: {', '.join([p['type'] for p in patterns])}")
        
        # Check if correctly classified (positive or neutral, safe)
        is_correct = emotion in ['positive', 'neutral'] and sentiment == 'safe'
        status = "[OK]" if is_correct else "[X]"
        print(f"  {status} {'Correct' if is_correct else 'Needs review'}")
    
    print("\n" + "="*80)
    print("ANALYSIS QUALITY CHECK")
    print("="*80)
    
    # Check if analysis details are comprehensive
    test_text = "Get lost, you piece of trash"
    result = classifier.classify(test_text)
    analysis = result.get('analysis_details', {})
    
    checks = {
        'Has sentiment analysis': 'sentiment_analysis' in analysis,
        'Has detected patterns': 'detected_patterns' in analysis and len(analysis.get('detected_patterns', [])) > 0,
        'Has key indicators': 'key_indicators' in analysis and len(analysis.get('key_indicators', [])) > 0,
        'Has classification reasoning': 'classification_reasoning' in analysis and len(analysis.get('classification_reasoning', [])) > 0,
        'Has similar cases': 'similar_cases' in analysis and len(analysis.get('similar_cases', [])) > 0,
    }
    
    for check, passed in checks.items():
        status = "[OK]" if passed else "[X]"
        print(f"  {status} {check}")
    
    all_passed = all(checks.values())
    print(f"\n{'='*80}")
    if all_passed:
        print("[OK] COMPREHENSIVE ANALYSIS WORKING PERFECTLY!")
    else:
        print("[!] Some analysis components missing")
    print("="*80)


if __name__ == '__main__':
    test_similar_cases()

