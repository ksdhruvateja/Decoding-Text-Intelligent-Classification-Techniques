"""
Test Complex Text Recognition
==============================
Tests the system with complex, ambiguous, and challenging texts
"""

from multistage_classifier import initialize_multistage_classifier
import json

def test_complex_texts():
    """Test with complex and challenging texts"""
    
    print("="*80)
    print("TESTING COMPLEX TEXT RECOGNITION")
    print("="*80)
    
    # Initialize classifier
    classifier = initialize_multistage_classifier()
    
    # Complex test cases
    complex_tests = [
        # Ambiguous statements
        {
            "text": "I'm not saying I want to hurt myself, but sometimes I wonder if anyone would notice if I was gone.",
            "expected": "emotional_distress or self_harm_low",
            "category": "Ambiguous self-harm ideation"
        },
        {
            "text": "Everything is fine, really. I'm just tired of pretending everything is fine when it's clearly not.",
            "expected": "emotional_distress or stress",
            "category": "Hidden distress"
        },
        
        # Sarcasm and irony
        {
            "text": "Oh great, another perfect day in paradise. Just what I needed.",
            "expected": "stress or emotional_distress",
            "category": "Sarcasm"
        },
        {
            "text": "I'm so happy I could just die. This is the best day ever.",
            "expected": "stress or emotional_distress",
            "category": "Irony"
        },
        
        # Long complex sentences
        {
            "text": "I've been thinking a lot lately about how life seems to be getting harder and harder, and even though I try to stay positive and tell myself things will get better, there are moments when I feel completely overwhelmed and wonder if I'm strong enough to handle everything that's coming my way.",
            "expected": "emotional_distress or stress",
            "category": "Long complex sentence"
        },
        
        # Mixed emotions
        {
            "text": "I'm excited about the new job but also terrified because I don't know if I'm good enough and I'm worried I'll fail and disappoint everyone.",
            "expected": "stress or emotional_distress",
            "category": "Mixed emotions"
        },
        
        # Context-dependent
        {
            "text": "I can't take this anymore. I really can't.",
            "expected": "emotional_distress or stress",
            "category": "Context-dependent distress"
        },
        
        # Subtle indicators
        {
            "text": "I've been having thoughts that maybe I'm not meant to be here, you know? Like maybe the world would be better off.",
            "expected": "self_harm_low or emotional_distress",
            "category": "Subtle self-harm ideation"
        },
        
        # Positive but complex
        {
            "text": "Despite all the challenges I've faced this year, I'm grateful for the lessons learned and the strength I've discovered within myself.",
            "expected": "neutral or positive",
            "category": "Complex positive"
        },
        
        # Negative but not crisis
        {
            "text": "The project didn't go as planned, and I'm disappointed, but I'll learn from this experience and do better next time.",
            "expected": "stress or neutral",
            "category": "Constructive negative"
        },
        
        # Threats vs self-harm distinction
        {
            "text": "If they don't stop bothering me, I'm going to make them regret it. I know exactly what I'll do.",
            "expected": "unsafe_environment",
            "category": "Threat to others"
        },
        {
            "text": "I've been planning how I would do it. I have everything I need. Tonight might be the night.",
            "expected": "self_harm_high",
            "category": "Self-harm plan"
        },
        
        # Metaphorical language
        {
            "text": "I feel like I'm drowning in responsibilities and there's no one to throw me a lifeline.",
            "expected": "emotional_distress or stress",
            "category": "Metaphorical distress"
        },
        
        # Questions and uncertainty
        {
            "text": "What's the point of trying when nothing ever works out? Why do I even bother?",
            "expected": "emotional_distress or stress",
            "category": "Rhetorical questions showing distress"
        },
        
        # Past tense vs present
        {
            "text": "I used to think about hurting myself, but I'm in a much better place now and have support.",
            "expected": "neutral or positive",
            "category": "Past tense recovery"
        },
        
        # Multiple clauses
        {
            "text": "I know I should be happy because I have everything I need, but I can't shake this feeling of emptiness, and even though I try to focus on the positive, the negative thoughts keep creeping in.",
            "expected": "emotional_distress",
            "category": "Multiple clauses showing internal conflict"
        }
    ]
    
    print(f"\nTesting {len(complex_tests)} complex cases...\n")
    
    results = []
    correct = 0
    
    for i, test in enumerate(complex_tests, 1):
        text = test['text']
        expected = test['expected']
        category = test['category']
        
        print(f"{'='*80}")
        print(f"Test {i}/{len(complex_tests)}: {category}")
        print(f"{'-'*80}")
        print(f"Text: {text[:100]}{'...' if len(text) > 100 else ''}")
        print(f"Expected: {expected}")
        
        # Classify
        result = classifier.classify(text)
        
        emotion = result.get('emotion', 'unknown')
        sentiment = result.get('sentiment', 'unknown')
        predictions = result.get('predictions', [])
        
        print(f"\nResult:")
        print(f"  Emotion: {emotion}")
        print(f"  Sentiment: {sentiment}")
        if predictions:
            print(f"  Top Prediction: {predictions[0]['label']} ({predictions[0]['confidence']*100:.1f}%)")
        
        # Check if result matches expected
        expected_labels = [e.strip() for e in expected.split('or')]
        is_correct = (
            emotion in expected_labels or
            (predictions and any(p['label'] in expected_labels for p in predictions[:2]))
        )
        
        if is_correct:
            correct += 1
            print(f"  [OK] Correct classification")
        else:
            print(f"  [X] May need improvement")
        
        results.append({
            'category': category,
            'text': text,
            'expected': expected,
            'got_emotion': emotion,
            'got_sentiment': sentiment,
            'got_predictions': [p['label'] for p in predictions[:3]],
            'correct': is_correct
        })
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total Tests: {len(complex_tests)}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {correct/len(complex_tests)*100:.1f}%")
    
    # Save results
    with open('complex_text_test_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to complex_text_test_results.json")
    
    # Show cases that need improvement
    incorrect = [r for r in results if not r['correct']]
    if incorrect:
        print(f"\n{'='*80}")
        print(f"CASES THAT MAY NEED IMPROVEMENT ({len(incorrect)}):")
        print(f"{'='*80}")
        for r in incorrect:
            print(f"\nCategory: {r['category']}")
            print(f"Expected: {r['expected']}")
            print(f"Got: {r['got_emotion']} ({r['got_sentiment']})")
            print(f"Text: {r['text'][:80]}...")
    
    return results


if __name__ == '__main__':
    test_complex_texts()

