"""
Test Classification with Custom Text Examples
==============================================
Verify trained models classify custom text accurately
"""

import requests
import json
import sys

API_URL = "http://localhost:5000/api/classify"
API_URL_FORMATTED = "http://localhost:5000/api/classify-formatted"

def test_classification():
    """Test custom text classification"""
    
    test_cases = [
        # Neutral/Safe
        ("The weather is nice today", "neutral"),
        ("I bought a new laptop", "neutral"),
        ("This product works as expected", "neutral"),
        
        # Positive
        ("I'm so happy and excited about this!", "positive"),
        ("Amazing experience, highly recommend", "positive"),
        ("Feeling grateful and blessed today", "positive"),
        
        # Stress
        ("I'm overwhelmed with work deadlines", "stress"),
        ("Can't handle all this pressure anymore", "stress"),
        ("Everything is falling apart", "stress"),
        
        # Emotional Distress
        ("I feel so alone and nobody understands", "emotional_distress"),
        ("The sadness won't go away", "emotional_distress"),
        ("I can't stop crying", "emotional_distress"),
        
        # Self Harm (Low)
        ("Sometimes I wonder if anyone would care", "self_harm_low"),
        ("Life feels meaningless lately", "self_harm_low"),
        
        # Self Harm (High)
        ("I want to end it all", "self_harm_high"),
        ("I have a plan to hurt myself", "self_harm_high"),
        
        # Unsafe Environment
        ("I'll hurt you if you come near me", "unsafe_environment"),
        ("Someone is threatening to attack me", "unsafe_environment"),
        
        # Negative
        ("This is terrible and disappointing", "negative"),
        ("Worst purchase ever, complete waste", "negative"),
    ]
    
    print("="*80)
    print("TESTING CLASSIFICATION ON CUSTOM TEXT")
    print("="*80)
    print(f"\nAPI Endpoint: {API_URL}")
    print(f"Total test cases: {len(test_cases)}\n")
    
    correct = 0
    total = len(test_cases)
    
    for idx, (text, expected_category) in enumerate(test_cases, 1):
        try:
            response = requests.post(
                API_URL,
                json={"text": text},
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                predicted = result.get('primary_category', 'unknown')
                confidence = result.get('confidence', 0.0)
                
                is_correct = predicted == expected_category
                if is_correct:
                    correct += 1
                    status = "✓"
                else:
                    status = "✗"
                
                print(f"{status} Test {idx}/{total}")
                print(f"  Text: {text[:60]}...")
                print(f"  Expected: {expected_category}")
                print(f"  Predicted: {predicted} ({confidence:.2%} confidence)")
                
                if not is_correct:
                    print(f"  ⚠️ MISMATCH")
                print()
                
            else:
                print(f"✗ Test {idx}/{total} - HTTP {response.status_code}")
                print(f"  Text: {text[:60]}...")
                print()
                
        except requests.exceptions.ConnectionError:
            print(f"\n❌ Cannot connect to backend at {API_URL}")
            print("Please start the backend first with:")
            print('  python app.py')
            sys.exit(1)
        except Exception as e:
            print(f"✗ Test {idx}/{total} - Error: {e}")
            print(f"  Text: {text[:60]}...")
            print()
    
    print("="*80)
    print("RESULTS")
    print("="*80)
    print(f"Correct: {correct}/{total} ({correct/total*100:.1f}%)")
    print(f"Accuracy: {correct/total:.2%}")
    
    if correct == total:
        print("\n🎉 Perfect score! All custom text classified correctly.")
    elif correct >= total * 0.8:
        print("\n✅ Good performance! Most custom text classified correctly.")
    elif correct >= total * 0.6:
        print("\n⚠️ Fair performance. Consider adding more training examples.")
    else:
        print("\n❌ Poor performance. Retrain with more diverse examples.")
    
    return correct / total


def test_formatted_output(sample_text="I feel stressed and unsafe"):
    """Test formatted block output"""
    print("\n" + "="*80)
    print("TESTING FORMATTED OUTPUT")
    print("="*80)
    print(f"\nSample text: {sample_text}\n")
    
    try:
        response = requests.post(
            API_URL_FORMATTED,
            json={"text": sample_text},
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        if response.status_code == 200:
            print("Formatted Block Output:")
            print("-" * 80)
            print(response.text)
            print("-" * 80)
            return True
        else:
            print(f"❌ HTTP {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to backend at {API_URL_FORMATTED}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


if __name__ == "__main__":
    print("\n" + "="*80)
    print("BACKEND CLASSIFICATION TEST SUITE")
    print("="*80)
    print("\nThis script tests:")
    print("  1. Classification accuracy on custom text")
    print("  2. Formatted output endpoint")
    print("  3. Model confidence scores")
    print("\nMake sure the backend is running before starting this test.")
    print("="*80)
    
    input("\nPress Enter to start testing...")
    
    # Test classification
    accuracy = test_classification()
    
    # Test formatted output
    test_formatted_output()
    
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)
    print(f"\nOverall Accuracy: {accuracy:.2%}")
    
    if accuracy >= 0.8:
        print("\n✅ Models are performing well on custom text!")
    else:
        print("\n⚠️ Consider retraining with more examples for better accuracy.")
