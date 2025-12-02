import json
import requests

with open('evaluation_dataset.json', 'r') as f:
    eval_data = json.load(f)

API_URL = "http://localhost:5000/api/classify"

results = []
correct = 0
for item in eval_data:
    text = item['text']
    expected = item['expected']
    try:
        resp = requests.post(API_URL, json={"text": text}, timeout=10)
        pred = resp.json().get('emotion', 'unknown')
        passed = (pred == expected)
        results.append({"text": text, "expected": expected, "predicted": pred, "pass": passed})
        if passed:
            correct += 1
        print(f"✓" if passed else "✗", f"Text: {text[:40]}... | Expected: {expected} | Predicted: {pred}")
    except Exception as e:
        print(f"✗ ERROR: {text[:40]}... | {e}")
        results.append({"text": text, "expected": expected, "predicted": "error", "pass": False})

print(f"\nAccuracy: {correct}/{len(eval_data)} = {correct/len(eval_data)*100:.1f}%")
with open('evaluation_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("Results saved to evaluation_results.json")
