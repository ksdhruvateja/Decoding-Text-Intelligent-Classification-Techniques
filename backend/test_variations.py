"""
Test Variations of Similar Cases
=================================
Tests various similar cases to ensure comprehensive understanding
"""

from multistage_classifier import initialize_multistage_classifier

def test_variations():
    """Test variations of similar cases"""
    
    print("="*80)
    print("TESTING VARIATIONS OF SIMILAR CASES")
    print("="*80)
    
    classifier = initialize_multistage_classifier()
    
    test_groups = [
        {
            "category": "Hostile/Aggressive Language Variations",
            "tests": [
                "Get lost, you piece of *",
                "Go away, you idiot",
                "Shut up, you're so annoying",
                "Leave me alone, you jerk",
                "You're such a moron",
                "I hate you so much",
                "You're useless and pathetic",
                "Stop bothering me, you fool",
                "You're a complete idiot",
                "Get out of here, you're worthless"
            ],
            "expected": "stress or emotional_distress"
        },
        {
            "category": "Positive Message Variations",
            "tests": [
                "I love how supportive this community is",
                "I really love this community",
                "So grateful for this supportive group",
                "This community is amazing and helpful",
                "I appreciate how kind everyone is here",
                "Thank you for being so supportive",
                "This is the best community ever",
                "I love how helpful everyone is",
                "So thankful for this amazing community",
                "This group is so welcoming and supportive"
            ],
            "expected": "positive or neutral"
        },
        {
            "category": "Frustration/Distress Variations",
            "tests": [
                "This app keeps crashing and it's so frustrating",
                "This is so annoying and frustrating",
                "I'm really frustrated with this situation",
                "This keeps happening and it's irritating",
                "So frustrated with how things are going",
                "This is really annoying me",
                "I can't stand how frustrating this is",
                "This situation is so irritating",
                "I'm getting really frustrated",
                "This is extremely annoying"
            ],
            "expected": "stress or emotional_distress"
        }
    ]
    
    total_tests = 0
    total_correct = 0
    
    for group in test_groups:
        print(f"\n{'='*80}")
        print(f"Category: {group['category']}")
        print(f"{'='*80}")
        
        category_correct = 0
        
        for text in group['tests']:
            total_tests += 1
            result = classifier.classify(text)
            
            emotion = result.get('emotion', 'unknown')
            sentiment = result.get('sentiment', 'unknown')
            predictions = result.get('predictions', [])
            
            expected = group['expected']
            expected_emotions = [e.strip() for e in expected.split('or')]
            
            is_correct = (
                emotion in expected_emotions or
                any(p['label'] in expected_emotions for p in predictions[:2])
            )
            
            if is_correct:
                category_correct += 1
                total_correct += 1
                status = "[OK]"
            else:
                status = "[X]"
            
            print(f"{status} {text[:60]}{'...' if len(text) > 60 else ''}")
            print(f"    -> {emotion} ({sentiment})")
            
            # Show analysis if available
            if result.get('analysis_explanation'):
                analysis = result['analysis_explanation']
                if analysis.get('detected_patterns'):
                    print(f"    Patterns: {', '.join(analysis['detected_patterns'][:3])}")
        
        print(f"\nCategory Accuracy: {category_correct}/{len(group['tests'])} ({category_correct/len(group['tests'])*100:.1f}%)")
    
    print(f"\n{'='*80}")
    print("OVERALL SUMMARY")
    print(f"{'='*80}")
    print(f"Total Tests: {total_tests}")
    print(f"Correct: {total_correct}")
    print(f"Accuracy: {total_correct/total_tests*100:.1f}%")
    print(f"{'='*80}")


if __name__ == '__main__':
    test_variations()

