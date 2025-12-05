"""Test edge cases and error scenarios"""
from app import application

test_client = application.test_client()

print("="*70)
print("EDGE CASE TESTING")
print("="*70)

edge_cases = [
    ("None text", {'text': None}),
    ("Empty string", {'text': ''}),
    ("Whitespace only", {'text': '   '}),
    ("Very long text", {'text': 'a' * 10000}),
    ("Special characters", {'text': '!@#$%^&*()'}),
    ("Unicode", {'text': '测试 🚀 émojis'}),
    ("Numbers only", {'text': '123456'}),
    ("Newlines", {'text': 'line1\nline2\nline3'}),
]

print("\nTesting edge cases for /api/classify:")
for name, payload in edge_cases:
    try:
        response = test_client.post('/api/classify', json=payload, content_type='application/json')
        status = response.status_code
        if status == 200:
            data = response.get_json()
            print(f"  ✅ {name}: {status} - {data.get('primary_category', 'unknown')}")
        else:
            error = response.get_json().get('error', 'Unknown error')
            print(f"  ⚠️  {name}: {status} - {error}")
    except Exception as e:
        print(f"  ❌ {name}: Exception - {e}")

print("\nTesting batch-classify edge cases:")
batch_cases = [
    ("Empty list", {'texts': []}),
    ("None in list", {'texts': [None, 'test']}),
    ("Mixed types", {'texts': ['test', 123, 'another']}),
    ("Very long list", {'texts': ['test'] * 1000}),
]

for name, payload in batch_cases:
    try:
        response = test_client.post('/api/batch-classify', json=payload, content_type='application/json')
        status = response.status_code
        if status == 200:
            data = response.get_json()
            print(f"  ✅ {name}: {status} - {data.get('count', 0)} results")
        else:
            error = response.get_json().get('error', 'Unknown error')
            print(f"  ⚠️  {name}: {status} - {error}")
    except Exception as e:
        print(f"  ❌ {name}: Exception - {e}")

print("\n" + "="*70)


