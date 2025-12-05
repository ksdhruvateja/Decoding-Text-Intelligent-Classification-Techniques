from simple_classifier import SimpleClassifier

classifier = SimpleClassifier()

test_texts = [
    "The movie was boring and way too long",
    "The product broke after just two days of use",
    "This book is 300 pages long and written in English",
    "i will kill you",
    "I want to kill myself",
    "The stock market crashed yesterday",
    "This movie was fantastic"
]

for text in test_texts:
    result = classifier.classify(text)
    print(f"\n{'='*60}")
    print(f"Text: {text}")
    print(f"Sentiment: {result['sentiment']}")
    print(f"Emotion: {result['emotion']}")
    print(f"Primary: {result['primary_category']} ({result['confidence']:.2f})")
    if result['predictions']:
        preds = [f"{p['label']}: {p['score']:.2f}" for p in result['predictions']]
        print(f"Predictions: {', '.join(preds)}")
