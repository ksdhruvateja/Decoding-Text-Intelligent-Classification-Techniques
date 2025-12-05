"""
Generate comprehensive training data for text classification
Covers: positive, negative, neutral, stress, emotional_distress, self_harm, unsafe_environment
"""

import json

# Comprehensive training data
training_data = [
    # === POSITIVE SENTIMENT ===
    {"text": "This movie was fantastic, I'd watch it again anytime", "labels": {"positive": 1, "negative": 0, "neutral": 0, "stress": 0, "emotional_distress": 0, "self_harm_low": 0, "self_harm_high": 0, "unsafe_environment": 0}},
    {"text": "I'm so excited about my new job!", "labels": {"positive": 1, "negative": 0, "neutral": 0, "stress": 0, "emotional_distress": 0, "self_harm_low": 0, "self_harm_high": 0, "unsafe_environment": 0}},
    {"text": "Had an amazing dinner with friends tonight", "labels": {"positive": 1, "negative": 0, "neutral": 0, "stress": 0, "emotional_distress": 0, "self_harm_low": 0, "self_harm_high": 0, "unsafe_environment": 0}},
    {"text": "I love this product, best purchase ever!", "labels": {"positive": 1, "negative": 0, "neutral": 0, "stress": 0, "emotional_distress": 0, "self_harm_low": 0, "self_harm_high": 0, "unsafe_environment": 0}},
    {"text": "Feeling grateful for my family today", "labels": {"positive": 1, "negative": 0, "neutral": 0, "stress": 0, "emotional_distress": 0, "self_harm_low": 0, "self_harm_high": 0, "unsafe_environment": 0}},
    {"text": "This restaurant has excellent service and delicious food", "labels": {"positive": 1, "negative": 0, "neutral": 0, "stress": 0, "emotional_distress": 0, "self_harm_low": 0, "self_harm_high": 0, "unsafe_environment": 0}},
    {"text": "I'm proud of my accomplishments this year", "labels": {"positive": 1, "negative": 0, "neutral": 0, "stress": 0, "emotional_distress": 0, "self_harm_low": 0, "self_harm_high": 0, "unsafe_environment": 0}},
    {"text": "Beautiful weather today, perfect for a walk", "labels": {"positive": 1, "negative": 0, "neutral": 0, "stress": 0, "emotional_distress": 0, "self_harm_low": 0, "self_harm_high": 0, "unsafe_environment": 0}},
    {"text": "My team won the championship, so happy!", "labels": {"positive": 1, "negative": 0, "neutral": 0, "stress": 0, "emotional_distress": 0, "self_harm_low": 0, "self_harm_high": 0, "unsafe_environment": 0}},
    {"text": "Just got promoted, feeling on top of the world", "labels": {"positive": 1, "negative": 0, "neutral": 0, "stress": 0, "emotional_distress": 0, "self_harm_low": 0, "self_harm_high": 0, "unsafe_environment": 0}},
    
    # === NEGATIVE SENTIMENT (complaints, dissatisfaction) ===
    {"text": "Shipping took forever and the item was damaged", "labels": {"positive": 0, "negative": 1, "neutral": 0, "stress": 1, "emotional_distress": 0, "self_harm_low": 0, "self_harm_high": 0, "unsafe_environment": 0}},
    {"text": "Terrible customer service, very disappointed", "labels": {"positive": 0, "negative": 1, "neutral": 0, "stress": 1, "emotional_distress": 0, "self_harm_low": 0, "self_harm_high": 0, "unsafe_environment": 0}},
    {"text": "This product is a waste of money, doesn't work at all", "labels": {"positive": 0, "negative": 1, "neutral": 0, "stress": 1, "emotional_distress": 0, "self_harm_low": 0, "self_harm_high": 0, "unsafe_environment": 0}},
    {"text": "Food was cold and tasted awful", "labels": {"positive": 0, "negative": 1, "neutral": 0, "stress": 0, "emotional_distress": 0, "self_harm_low": 0, "self_harm_high": 0, "unsafe_environment": 0}},
    {"text": "Worst experience ever, would not recommend", "labels": {"positive": 0, "negative": 1, "neutral": 0, "stress": 1, "emotional_distress": 0, "self_harm_low": 0, "self_harm_high": 0, "unsafe_environment": 0}},
    {"text": "The app keeps crashing, very frustrating", "labels": {"positive": 0, "negative": 1, "neutral": 0, "stress": 1, "emotional_distress": 0, "self_harm_low": 0, "self_harm_high": 0, "unsafe_environment": 0}},
    {"text": "Disappointed with the quality, expected better", "labels": {"positive": 0, "negative": 1, "neutral": 0, "stress": 0, "emotional_distress": 0, "self_harm_low": 0, "self_harm_high": 0, "unsafe_environment": 0}},
    {"text": "Waited 2 hours for delivery, unacceptable", "labels": {"positive": 0, "negative": 1, "neutral": 0, "stress": 1, "emotional_distress": 0, "self_harm_low": 0, "self_harm_high": 0, "unsafe_environment": 0}},
    
    # === NEUTRAL (factual, informational) ===
    {"text": "This book is 300 pages long and written in English", "labels": {"positive": 0, "negative": 0, "neutral": 1, "stress": 0, "emotional_distress": 0, "self_harm_low": 0, "self_harm_high": 0, "unsafe_environment": 0}},
    {"text": "The store opens at 9 AM and closes at 6 PM", "labels": {"positive": 0, "negative": 0, "neutral": 1, "stress": 0, "emotional_distress": 0, "self_harm_low": 0, "self_harm_high": 0, "unsafe_environment": 0}},
    {"text": "It takes about 30 minutes to drive there", "labels": {"positive": 0, "negative": 0, "neutral": 1, "stress": 0, "emotional_distress": 0, "self_harm_low": 0, "self_harm_high": 0, "unsafe_environment": 0}},
    {"text": "The capital of France is Paris", "labels": {"positive": 0, "negative": 0, "neutral": 1, "stress": 0, "emotional_distress": 0, "self_harm_low": 0, "self_harm_high": 0, "unsafe_environment": 0}},
    {"text": "Water boils at 100 degrees Celsius", "labels": {"positive": 0, "negative": 0, "neutral": 1, "stress": 0, "emotional_distress": 0, "self_harm_low": 0, "self_harm_high": 0, "unsafe_environment": 0}},
    {"text": "The meeting is scheduled for Tuesday at 2 PM", "labels": {"positive": 0, "negative": 0, "neutral": 1, "stress": 0, "emotional_distress": 0, "self_harm_low": 0, "self_harm_high": 0, "unsafe_environment": 0}},
    {"text": "There are 12 months in a year", "labels": {"positive": 0, "negative": 0, "neutral": 1, "stress": 0, "emotional_distress": 0, "self_harm_low": 0, "self_harm_high": 0, "unsafe_environment": 0}},
    {"text": "The recipe calls for 2 cups of flour", "labels": {"positive": 0, "negative": 0, "neutral": 1, "stress": 0, "emotional_distress": 0, "self_harm_low": 0, "self_harm_high": 0, "unsafe_environment": 0}},
    
    # === STRESS (work pressure, deadlines, overwhelm) ===
    {"text": "The stock market crashed yesterday after tech earnings disappointed", "labels": {"positive": 0, "negative": 1, "neutral": 0, "stress": 1, "emotional_distress": 0, "self_harm_low": 0, "self_harm_high": 0, "unsafe_environment": 0}},
    {"text": "I have three deadlines tomorrow and I'm not ready", "labels": {"positive": 0, "negative": 0, "neutral": 0, "stress": 1, "emotional_distress": 0, "self_harm_low": 0, "self_harm_high": 0, "unsafe_environment": 0}},
    {"text": "So much work piling up, feeling overwhelmed", "labels": {"positive": 0, "negative": 0, "neutral": 0, "stress": 1, "emotional_distress": 0, "self_harm_low": 0, "self_harm_high": 0, "unsafe_environment": 0}},
    {"text": "Can't sleep worrying about the presentation", "labels": {"positive": 0, "negative": 0, "neutral": 0, "stress": 1, "emotional_distress": 1, "self_harm_low": 0, "self_harm_high": 0, "unsafe_environment": 0}},
    {"text": "Too many responsibilities, don't know where to start", "labels": {"positive": 0, "negative": 0, "neutral": 0, "stress": 1, "emotional_distress": 0, "self_harm_low": 0, "self_harm_high": 0, "unsafe_environment": 0}},
    {"text": "Work is demanding and I'm exhausted", "labels": {"positive": 0, "negative": 0, "neutral": 0, "stress": 1, "emotional_distress": 0, "self_harm_low": 0, "self_harm_high": 0, "unsafe_environment": 0}},
    {"text": "Financial pressure is getting to me", "labels": {"positive": 0, "negative": 0, "neutral": 0, "stress": 1, "emotional_distress": 1, "self_harm_low": 0, "self_harm_high": 0, "unsafe_environment": 0}},
    {"text": "I'm so stressed about passing this exam", "labels": {"positive": 0, "negative": 0, "neutral": 0, "stress": 1, "emotional_distress": 0, "self_harm_low": 0, "self_harm_high": 0, "unsafe_environment": 0}},
    
    # === EMOTIONAL DISTRESS (sadness, loneliness, despair) ===
    {"text": "I feel so alone and nobody understands me", "labels": {"positive": 0, "negative": 0, "neutral": 0, "stress": 0, "emotional_distress": 1, "self_harm_low": 0, "self_harm_high": 0, "unsafe_environment": 0}},
    {"text": "Everything feels hopeless and dark", "labels": {"positive": 0, "negative": 0, "neutral": 0, "stress": 0, "emotional_distress": 1, "self_harm_low": 1, "self_harm_high": 0, "unsafe_environment": 0}},
    {"text": "I'm constantly sad and can't find joy anymore", "labels": {"positive": 0, "negative": 0, "neutral": 0, "stress": 0, "emotional_distress": 1, "self_harm_low": 0, "self_harm_high": 0, "unsafe_environment": 0}},
    {"text": "Lost my best friend and feel empty inside", "labels": {"positive": 0, "negative": 0, "neutral": 0, "stress": 0, "emotional_distress": 1, "self_harm_low": 0, "self_harm_high": 0, "unsafe_environment": 0}},
    {"text": "Depression is overwhelming me every day", "labels": {"positive": 0, "negative": 0, "neutral": 0, "stress": 0, "emotional_distress": 1, "self_harm_low": 1, "self_harm_high": 0, "unsafe_environment": 0}},
    {"text": "I cry every night feeling worthless", "labels": {"positive": 0, "negative": 0, "neutral": 0, "stress": 0, "emotional_distress": 1, "self_harm_low": 1, "self_harm_high": 0, "unsafe_environment": 0}},
    
    # === SELF-HARM LOW (thoughts, mild ideation) ===
    {"text": "Sometimes I think about harming myself", "labels": {"positive": 0, "negative": 0, "neutral": 0, "stress": 0, "emotional_distress": 1, "self_harm_low": 1, "self_harm_high": 0, "unsafe_environment": 0}},
    {"text": "I wonder if anyone would care if I was gone", "labels": {"positive": 0, "negative": 0, "neutral": 0, "stress": 0, "emotional_distress": 1, "self_harm_low": 1, "self_harm_high": 0, "unsafe_environment": 0}},
    {"text": "Life doesn't seem worth living sometimes", "labels": {"positive": 0, "negative": 0, "neutral": 0, "stress": 0, "emotional_distress": 1, "self_harm_low": 1, "self_harm_high": 0, "unsafe_environment": 0}},
    
    # === SELF-HARM HIGH (urgent, immediate danger) ===
    {"text": "I want to kill myself tonight", "labels": {"positive": 0, "negative": 0, "neutral": 0, "stress": 0, "emotional_distress": 1, "self_harm_low": 0, "self_harm_high": 1, "unsafe_environment": 0}},
    {"text": "Going to end my life, have the pills ready", "labels": {"positive": 0, "negative": 0, "neutral": 0, "stress": 0, "emotional_distress": 1, "self_harm_low": 0, "self_harm_high": 1, "unsafe_environment": 0}},
    {"text": "I'm going to hurt myself right now", "labels": {"positive": 0, "negative": 0, "neutral": 0, "stress": 0, "emotional_distress": 1, "self_harm_low": 0, "self_harm_high": 1, "unsafe_environment": 0}},
    
    # === UNSAFE ENVIRONMENT ===
    {"text": "My partner threatens me when they get angry", "labels": {"positive": 0, "negative": 0, "neutral": 0, "stress": 1, "emotional_distress": 1, "self_harm_low": 0, "self_harm_high": 0, "unsafe_environment": 1}},
    {"text": "I'm scared to go home because of violence", "labels": {"positive": 0, "negative": 0, "neutral": 0, "stress": 1, "emotional_distress": 1, "self_harm_low": 0, "self_harm_high": 0, "unsafe_environment": 1}},
    {"text": "Someone is following me and I feel unsafe", "labels": {"positive": 0, "negative": 0, "neutral": 0, "stress": 1, "emotional_distress": 1, "self_harm_low": 0, "self_harm_high": 0, "unsafe_environment": 1}},
    {"text": "My family member is abusive and hurts me", "labels": {"positive": 0, "negative": 0, "neutral": 0, "stress": 1, "emotional_distress": 1, "self_harm_low": 0, "self_harm_high": 0, "unsafe_environment": 1}},
]

# Expand dataset with variations
expanded_data = training_data.copy()

# Add more variations
positive_variations = [
    "Absolutely loved the experience!",
    "Everything was perfect, highly recommend",
    "Best decision I ever made",
    "So happy with my choice",
    "This exceeded all my expectations",
    "Wonderful time, will definitely return",
    "Outstanding quality and service",
    "Could not be happier with the results",
]

negative_variations = [
    "Complete disaster, avoid at all costs",
    "Regret buying this, total waste",
    "Very poor quality, falling apart already",
    "Service was rude and unprofessional",
    "Broken after one use, terrible",
    "Not worth the price at all",
    "Extremely dissatisfied with everything",
]

neutral_variations = [
    "The product weighs 5 pounds",
    "Available in red, blue, and green",
    "Shipping takes 3-5 business days",
    "Located on the second floor",
    "Made of plastic and metal",
    "Dimensions are 10x20x5 inches",
]

stress_variations = [
    "So behind on everything, feeling panicked",
    "Pressure is building up, can't handle it",
    "Deadline stress is killing me",
    "Too much on my plate right now",
    "Anxiety about upcoming evaluation",
    "Worried sick about finances",
]

distress_variations = [
    "Feeling numb and disconnected",
    "Can't stop crying, everything hurts",
    "Lost all motivation for life",
    "Nobody cares about me anymore",
    "Feeling broken and unfixable",
]

for text in positive_variations:
    expanded_data.append({"text": text, "labels": {"positive": 1, "negative": 0, "neutral": 0, "stress": 0, "emotional_distress": 0, "self_harm_low": 0, "self_harm_high": 0, "unsafe_environment": 0}})

for text in negative_variations:
    expanded_data.append({"text": text, "labels": {"positive": 0, "negative": 1, "neutral": 0, "stress": 1, "emotional_distress": 0, "self_harm_low": 0, "self_harm_high": 0, "unsafe_environment": 0}})

for text in neutral_variations:
    expanded_data.append({"text": text, "labels": {"positive": 0, "negative": 0, "neutral": 1, "stress": 0, "emotional_distress": 0, "self_harm_low": 0, "self_harm_high": 0, "unsafe_environment": 0}})

for text in stress_variations:
    expanded_data.append({"text": text, "labels": {"positive": 0, "negative": 0, "neutral": 0, "stress": 1, "emotional_distress": 0, "self_harm_low": 0, "self_harm_high": 0, "unsafe_environment": 0}})

for text in distress_variations:
    expanded_data.append({"text": text, "labels": {"positive": 0, "negative": 0, "neutral": 0, "stress": 0, "emotional_distress": 1, "self_harm_low": 1, "self_harm_high": 0, "unsafe_environment": 0}})

# Save to file
with open('comprehensive_training_data.json', 'w', encoding='utf-8') as f:
    json.dump(expanded_data, f, indent=2, ensure_ascii=False)

print(f"Generated {len(expanded_data)} training examples")
print("Saved to comprehensive_training_data.json")
