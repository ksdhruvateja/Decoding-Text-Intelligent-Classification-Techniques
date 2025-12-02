"""
Generate Corrective Training Data for Common Misclassifications
===============================================================
Adds specific examples to fix the classification issues:
1. Positive statements (restaurant reviews, compliments) -> positive/safe
2. Negative complaints (service issues) -> stress/emotional_distress
3. Neutral statements (shopping, daily activities) -> neutral/safe
"""

import json
import random

def generate_corrective_data():
    """Generate training data to fix common misclassifications"""
    
    training_data = []
    
    # POSITIVE STATEMENTS - Should be classified as positive/safe
    positive_examples = [
        # Restaurant/Service Positive
        {"text": "I absolutely loved the new restaurant; the food was delicious and the staff were so friendly.", 
         "labels": {"neutral": 1, "stress": 0, "unsafe_environment": 0, "emotional_distress": 0, "self_harm_low": 0, "self_harm_high": 0}},
        {"text": "The restaurant was amazing! Great food and excellent service.", 
         "labels": {"neutral": 1, "stress": 0, "unsafe_environment": 0, "emotional_distress": 0, "self_harm_low": 0, "self_harm_high": 0}},
        {"text": "I had a wonderful experience at the hotel. The staff was very helpful.", 
         "labels": {"neutral": 1, "stress": 0, "unsafe_environment": 0, "emotional_distress": 0, "self_harm_low": 0, "self_harm_high": 0}},
        {"text": "This place is fantastic! I highly recommend it.", 
         "labels": {"neutral": 1, "stress": 0, "unsafe_environment": 0, "emotional_distress": 0, "self_harm_low": 0, "self_harm_high": 0}},
        {"text": "The food was incredible and the atmosphere was perfect.", 
         "labels": {"neutral": 1, "stress": 0, "unsafe_environment": 0, "emotional_distress": 0, "self_harm_low": 0, "self_harm_high": 0}},
        
        # General Positive
        {"text": "I'm so happy with my new job! Everything is going great.", 
         "labels": {"neutral": 1, "stress": 0, "unsafe_environment": 0, "emotional_distress": 0, "self_harm_low": 0, "self_harm_high": 0}},
        {"text": "I feel really grateful for all the support I've received.", 
         "labels": {"neutral": 1, "stress": 0, "unsafe_environment": 0, "emotional_distress": 0, "self_harm_low": 0, "self_harm_high": 0}},
        {"text": "Today was amazing! I accomplished so much.", 
         "labels": {"neutral": 1, "stress": 0, "unsafe_environment": 0, "emotional_distress": 0, "self_harm_low": 0, "self_harm_high": 0}},
        {"text": "I'm excited about the upcoming vacation. It's going to be wonderful!", 
         "labels": {"neutral": 1, "stress": 0, "unsafe_environment": 0, "emotional_distress": 0, "self_harm_low": 0, "self_harm_high": 0}},
        {"text": "The concert was incredible! Best night ever.", 
         "labels": {"neutral": 1, "stress": 0, "unsafe_environment": 0, "emotional_distress": 0, "self_harm_low": 0, "self_harm_high": 0}},
    ]
    
    # NEGATIVE COMPLAINTS - Should be stress/emotional_distress (NOT safe)
    negative_complaint_examples = [
        # Service Complaints
        {"text": "The service was terrible, I had to wait an hour and the staff were rude", 
         "labels": {"neutral": 0, "stress": 1, "unsafe_environment": 0, "emotional_distress": 1, "self_harm_low": 0, "self_harm_high": 0}},
        {"text": "I'm very disappointed with the service. The wait was too long.", 
         "labels": {"neutral": 0, "stress": 1, "unsafe_environment": 0, "emotional_distress": 1, "self_harm_low": 0, "self_harm_high": 0}},
        {"text": "The customer service was awful. They didn't help me at all.", 
         "labels": {"neutral": 0, "stress": 1, "unsafe_environment": 0, "emotional_distress": 1, "self_harm_low": 0, "self_harm_high": 0}},
        {"text": "I'm frustrated with the poor quality of service I received.", 
         "labels": {"neutral": 0, "stress": 1, "unsafe_environment": 0, "emotional_distress": 1, "self_harm_low": 0, "self_harm_high": 0}},
        {"text": "The restaurant was a complete disaster. Terrible food and rude staff.", 
         "labels": {"neutral": 0, "stress": 1, "unsafe_environment": 0, "emotional_distress": 1, "self_harm_low": 0, "self_harm_high": 0}},
        
        # General Negative (but not crisis)
        {"text": "I'm really disappointed with how things turned out.", 
         "labels": {"neutral": 0, "stress": 1, "unsafe_environment": 0, "emotional_distress": 1, "self_harm_low": 0, "self_harm_high": 0}},
        {"text": "This situation is really frustrating me.", 
         "labels": {"neutral": 0, "stress": 1, "unsafe_environment": 0, "emotional_distress": 1, "self_harm_low": 0, "self_harm_high": 0}},
        {"text": "I'm upset about what happened today.", 
         "labels": {"neutral": 0, "stress": 1, "unsafe_environment": 0, "emotional_distress": 1, "self_harm_low": 0, "self_harm_high": 0}},
        {"text": "Things didn't go as planned and I'm feeling down.", 
         "labels": {"neutral": 0, "stress": 1, "unsafe_environment": 0, "emotional_distress": 1, "self_harm_low": 0, "self_harm_high": 0}},
        {"text": "I'm annoyed by the constant delays and poor communication.", 
         "labels": {"neutral": 0, "stress": 1, "unsafe_environment": 0, "emotional_distress": 1, "self_harm_low": 0, "self_harm_high": 0}},
    ]
    
    # NEUTRAL DAILY ACTIVITIES - Should be neutral/safe (NOT risky)
    neutral_daily_examples = [
        # Shopping/Errands
        {"text": "I went to the store yesterday to buy some groceries", 
         "labels": {"neutral": 1, "stress": 0, "unsafe_environment": 0, "emotional_distress": 0, "self_harm_low": 0, "self_harm_high": 0}},
        {"text": "I need to go shopping for groceries this weekend.", 
         "labels": {"neutral": 1, "stress": 0, "unsafe_environment": 0, "emotional_distress": 0, "self_harm_low": 0, "self_harm_high": 0}},
        {"text": "I'm going to the store to pick up some items.", 
         "labels": {"neutral": 1, "stress": 0, "unsafe_environment": 0, "emotional_distress": 0, "self_harm_low": 0, "self_harm_high": 0}},
        {"text": "I bought groceries at the supermarket today.", 
         "labels": {"neutral": 1, "stress": 0, "unsafe_environment": 0, "emotional_distress": 0, "self_harm_low": 0, "self_harm_high": 0}},
        {"text": "I'm planning to go shopping tomorrow morning.", 
         "labels": {"neutral": 1, "stress": 0, "unsafe_environment": 0, "emotional_distress": 0, "self_harm_low": 0, "self_harm_high": 0}},
        
        # Daily Activities
        {"text": "I'm going to work today as usual.", 
         "labels": {"neutral": 1, "stress": 0, "unsafe_environment": 0, "emotional_distress": 0, "self_harm_low": 0, "self_harm_high": 0}},
        {"text": "I have a meeting scheduled for this afternoon.", 
         "labels": {"neutral": 1, "stress": 0, "unsafe_environment": 0, "emotional_distress": 0, "self_harm_low": 0, "self_harm_high": 0}},
        {"text": "I'm cooking dinner tonight.", 
         "labels": {"neutral": 1, "stress": 0, "unsafe_environment": 0, "emotional_distress": 0, "self_harm_low": 0, "self_harm_high": 0}},
        {"text": "I need to finish my homework before class.", 
         "labels": {"neutral": 1, "stress": 0, "unsafe_environment": 0, "emotional_distress": 0, "self_harm_low": 0, "self_harm_high": 0}},
        {"text": "I'm going to the gym after work.", 
         "labels": {"neutral": 1, "stress": 0, "unsafe_environment": 0, "emotional_distress": 0, "self_harm_low": 0, "self_harm_high": 0}},
        
        # Informational
        {"text": "The meeting starts at 3 PM.", 
         "labels": {"neutral": 1, "stress": 0, "unsafe_environment": 0, "emotional_distress": 0, "self_harm_low": 0, "self_harm_high": 0}},
        {"text": "I need to call the doctor's office tomorrow.", 
         "labels": {"neutral": 1, "stress": 0, "unsafe_environment": 0, "emotional_distress": 0, "self_harm_low": 0, "self_harm_high": 0}},
        {"text": "I'm planning to visit my family next month.", 
         "labels": {"neutral": 1, "stress": 0, "unsafe_environment": 0, "emotional_distress": 0, "self_harm_low": 0, "self_harm_high": 0}},
        {"text": "I have an appointment on Friday at 2 PM.", 
         "labels": {"neutral": 1, "stress": 0, "unsafe_environment": 0, "emotional_distress": 0, "self_harm_low": 0, "self_harm_high": 0}},
        {"text": "I'm reading a book about history.", 
         "labels": {"neutral": 1, "stress": 0, "unsafe_environment": 0, "emotional_distress": 0, "self_harm_low": 0, "self_harm_high": 0}},
    ]
    
    # Add more variations
    positive_variations = [
        "I'm thrilled with", "I'm so happy about", "I'm delighted by", 
        "I'm grateful for", "I'm excited about", "I'm pleased with",
        "wonderful", "excellent", "fantastic", "amazing", "great", "awesome"
    ]
    
    negative_variations = [
        "I'm disappointed with", "I'm frustrated by", "I'm upset about",
        "terrible", "awful", "horrible", "bad", "poor", "disappointing"
    ]
    
    # Generate more examples
    for _ in range(20):
        pos_word = random.choice(positive_variations)
        pos_text = f"{pos_word} the service I received today. Everything was perfect!"
        training_data.append({
            "text": pos_text,
            "labels": {"neutral": 1, "stress": 0, "unsafe_environment": 0, "emotional_distress": 0, "self_harm_low": 0, "self_harm_high": 0}
        })
    
    for _ in range(20):
        neg_word = random.choice(negative_variations)
        neg_text = f"{neg_word} the service. I had to wait too long."
        training_data.append({
            "text": neg_text,
            "labels": {"neutral": 0, "stress": 1, "unsafe_environment": 0, "emotional_distress": 1, "self_harm_low": 0, "self_harm_high": 0}
        })
    
    # Combine all
    training_data.extend(positive_examples)
    training_data.extend(negative_complaint_examples)
    training_data.extend(neutral_daily_examples)
    
    return training_data


def merge_with_existing_data():
    """Merge corrective data with existing training data"""
    # Load existing data
    try:
        with open('train_data.json', 'r') as f:
            existing_data = json.load(f)
    except:
        existing_data = []
    
    # Generate corrective data
    corrective_data = generate_corrective_data()
    
    # Merge (add corrective data multiple times for emphasis)
    merged_data = existing_data.copy()
    merged_data.extend(corrective_data)
    merged_data.extend(corrective_data)  # Add twice for emphasis
    
    # Save
    with open('train_data_corrected.json', 'w') as f:
        json.dump(merged_data, f, indent=2)
    
    print(f"Generated {len(corrective_data)} corrective examples")
    print(f"Total training examples: {len(merged_data)}")
    print("Saved to train_data_corrected.json")
    
    return merged_data


if __name__ == '__main__':
    merge_with_existing_data()

