"""Comprehensive API test for the classifier backend"""
import requests
import json

API_URL = 'http://localhost:5000/api'

print("=" * 70)
print("TESTING FLASK API ENDPOINTS")
print("=" * 70)

# Test 1: Health check
print("\n1. Testing /api/health endpoint...")
try:
    response = requests.get(f'{API_URL}/health')
    print(f"   ✅ Status: {response.status_code}")
    print(f"   Response: {json.dumps(response.json(), indent=2)}")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 2: Classification - Threat
print("\n2. Testing /api/classify with threat text...")
test_cases = [
    {
        "text": "i will kill you",
        "expected": "unsafe_environment"
    },
    {
        "text": "The movie was boring and way too long",
        "expected": "negative"
    },
    {
        "text": "This book is 300 pages long",
        "expected": "neutral"
    },
    {
        "text": "I want to kill myself",
        "expected": "self_harm_high"
    },
    {
        "text": "This movie was fantastic and amazing",
        "expected": "positive"
    },
    {
        "text": "The stock market crashed",
        "expected": "stress"
    }
]

for i, test in enumerate(test_cases, 1):
    print(f"\n   Test {i}: '{test['text']}'")
    try:
        response = requests.post(
            f'{API_URL}/classify',
            json={"text": test['text']},
            headers={'Content-Type': 'application/json'}
        )
        data = response.json()
        primary = data['primary_category']
        confidence = data['confidence']
        sentiment = data['sentiment']
        
        status = "✅" if primary == test['expected'] else "⚠️"
        print(f"   {status} Primary: {primary} ({confidence:.2f}) | Expected: {test['expected']}")
        print(f"      Sentiment: {sentiment}")
        
        if data['predictions']:
            top_3 = data['predictions'][:3]
            preds = [f"{p['label']}: {p['score']:.2f}" for p in top_3]
            print(f"      Top predictions: {', '.join(preds)}")
    except Exception as e:
        print(f"   ❌ Error: {e}")

# Test 3: Categories
print("\n3. Testing /api/categories endpoint...")
try:
    response = requests.get(f'{API_URL}/categories')
    data = response.json()
    print(f"   ✅ Status: {response.status_code}")
    print(f"   Categories: {data['categories']}")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 4: History
print("\n4. Testing /api/history endpoint...")
try:
    response = requests.get(f'{API_URL}/history?limit=3')
    data = response.json()
    print(f"   ✅ Status: {response.status_code}")
    print(f"   Total in history: {data['total']}")
    print(f"   Showing last: {len(data['history'])}")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 5: Statistics
print("\n5. Testing /api/stats endpoint...")
try:
    response = requests.get(f'{API_URL}/stats')
    data = response.json()
    print(f"   ✅ Status: {response.status_code}")
    print(f"   Total classifications: {data['total_classifications']}")
    print(f"   Unique categories: {data['unique_categories_found']}")
    if data['category_counts']:
        print(f"   Category breakdown:")
        for cat, count in data['category_counts'].items():
            print(f"      - {cat}: {count}")
except Exception as e:
    print(f"   ❌ Error: {e}")

print("\n" + "=" * 70)
print("TESTING COMPLETE")
print("=" * 70)
