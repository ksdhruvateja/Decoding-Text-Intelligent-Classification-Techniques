import json

with open('training_data.json', 'r') as f:
    data = json.load(f)

positive_words = ['happy', 'love', 'grateful', 'excited', 'joyful', 'amazing', 'wonderful', 'great', 'excellent', 'fantastic']
positive_examples = [d for d in data if any(w in d['text'].lower() for w in positive_words)]

print(f"\n✓ Found {len(positive_examples)} examples with positive words")
print(f"Total training examples: {len(data)}")
print(f"Percentage: {len(positive_examples)/len(data)*100:.1f}%\n")

print("First 15 positive examples with their labels:")
for i, d in enumerate(positive_examples[:15], 1):
    labels = d.get('labels', {})
    main_label = max(labels.items(), key=lambda x: x[1])[0] if labels else 'unknown'
    print(f"{i}. [{main_label}] {d['text'][:80]}")
