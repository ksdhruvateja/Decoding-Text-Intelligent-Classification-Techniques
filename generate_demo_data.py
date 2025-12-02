"""
Demo data generator for testing the application without training
Generates sample predictions for demonstration purposes
"""

import random
import json
from datetime import datetime, timedelta

# Sample texts with varying toxicity levels
SAMPLE_TEXTS = [
    # Clean examples
    "I really enjoyed this movie, it was fantastic!",
    "Thank you for your help, I appreciate it.",
    "The weather is beautiful today.",
    "Great job on the presentation!",
    "Looking forward to our meeting tomorrow.",
    
    # Slightly toxic
    "This is the worst product I've ever bought.",
    "You clearly don't know what you're talking about.",
    "That's a stupid idea.",
    "Are you serious right now?",
    
    # More toxic
    "You're an idiot if you believe that.",
    "This is complete garbage and you should be ashamed.",
    "Shut up and leave me alone.",
    "Nobody cares about your opinion.",
    
    # Highly toxic
    "Go kill yourself.",
    "I hate you and everyone like you.",
    "You deserve to suffer.",
]

CATEGORIES = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

def generate_sample_prediction(text, toxicity_level='low'):
    """
    Generate a sample prediction for demonstration
    
    Args:
        text: The input text
        toxicity_level: 'low', 'medium', 'high'
    """
    
    if toxicity_level == 'low':
        base_scores = {
            'toxic': random.uniform(0.05, 0.15),
            'severe_toxic': random.uniform(0.01, 0.05),
            'obscene': random.uniform(0.02, 0.08),
            'threat': random.uniform(0.01, 0.03),
            'insult': random.uniform(0.02, 0.10),
            'identity_hate': random.uniform(0.01, 0.04),
        }
    elif toxicity_level == 'medium':
        base_scores = {
            'toxic': random.uniform(0.40, 0.65),
            'severe_toxic': random.uniform(0.05, 0.20),
            'obscene': random.uniform(0.15, 0.40),
            'threat': random.uniform(0.05, 0.15),
            'insult': random.uniform(0.30, 0.60),
            'identity_hate': random.uniform(0.05, 0.20),
        }
    else:  # high
        base_scores = {
            'toxic': random.uniform(0.75, 0.95),
            'severe_toxic': random.uniform(0.40, 0.80),
            'obscene': random.uniform(0.50, 0.85),
            'threat': random.uniform(0.30, 0.70),
            'insult': random.uniform(0.60, 0.90),
            'identity_hate': random.uniform(0.30, 0.65),
        }
    
    # Create prediction list for scores above threshold
    predictions = []
    threshold = 0.5
    
    for category, score in base_scores.items():
        if score >= threshold:
            predictions.append({
                'label': category,
                'confidence': score
            })
    
    return {
        'text': text,
        'predictions': predictions,
        'all_scores': base_scores,
        'timestamp': datetime.now().isoformat()
    }


def generate_test_dataset(output_file='demo_predictions.json'):
    """Generate a test dataset with sample predictions"""
    
    dataset = []
    
    # Clean examples
    for i in range(5):
        text = SAMPLE_TEXTS[i]
        prediction = generate_sample_prediction(text, 'low')
        dataset.append(prediction)
    
    # Medium toxicity
    for i in range(5, 9):
        text = SAMPLE_TEXTS[i]
        prediction = generate_sample_prediction(text, 'medium')
        dataset.append(prediction)
    
    # High toxicity
    for i in range(9, len(SAMPLE_TEXTS)):
        text = SAMPLE_TEXTS[i]
        prediction = generate_sample_prediction(text, 'high')
        dataset.append(prediction)
    
    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"Generated {len(dataset)} sample predictions in {output_file}")
    return dataset


def print_demo_stats(predictions):
    """Print statistics about the demo data"""
    
    total = len(predictions)
    category_counts = {}
    total_confidence = 0
    prediction_count = 0
    
    for pred in predictions:
        for p in pred['predictions']:
            label = p['label']
            category_counts[label] = category_counts.get(label, 0) + 1
            total_confidence += p['confidence']
            prediction_count += 1
    
    print("\n" + "="*50)
    print("DEMO DATA STATISTICS")
    print("="*50)
    print(f"Total messages: {total}")
    print(f"Total predictions: {prediction_count}")
    print(f"Average confidence: {(total_confidence/prediction_count)*100:.1f}%")
    print("\nCategory distribution:")
    for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {category}: {count}")
    print("="*50 + "\n")


if __name__ == '__main__':
    print("Generating demo data for Text Classifier...\n")
    predictions = generate_test_dataset()
    print_demo_stats(predictions)
    print("\nYou can now test the application with these sample texts!")