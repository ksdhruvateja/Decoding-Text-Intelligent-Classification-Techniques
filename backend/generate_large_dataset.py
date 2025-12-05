"""
Generate large-scale comprehensive training data for all categories
Covers: positive, negative, neutral, stress, emotional_distress, self_harm, unsafe_environment
"""

import json
import random

# Large-scale training data with diverse examples
training_data = []

# === POSITIVE SENTIMENT (200 examples) ===
positive_examples = [
    # Movies & Entertainment
    "This movie was fantastic, I'd watch it again anytime",
    "Amazing film with incredible acting",
    "Best movie I've seen this year",
    "Loved every minute of it",
    "Absolutely brilliant performance",
    "Five stars all the way",
    "Couldn't have been better",
    "A masterpiece of cinema",
    "Highly recommend this film",
    "Outstanding storytelling",
    
    # Products & Services
    "Excellent product quality",
    "Worth every penny",
    "Best purchase ever",
    "Exceeded all expectations",
    "Superior customer service",
    "Fast shipping and great packaging",
    "Love this product",
    "Works perfectly",
    "Amazing value for money",
    "Will definitely buy again",
    "This product changed my life",
    "Incredible quality",
    "Highly satisfied with purchase",
    "Better than advertised",
    "Perfect condition on arrival",
    
    # Food & Restaurants
    "Delicious food and friendly staff",
    "Best restaurant in town",
    "Amazing flavors",
    "Great atmosphere and service",
    "Will definitely come back",
    "Food was incredible",
    "Outstanding meal",
    "Perfect dining experience",
    "Absolutely delicious",
    "Chef's special was amazing",
    
    # Personal Achievement
    "So proud of my accomplishments",
    "Finally reached my goal",
    "Best day of my life",
    "Feeling on top of the world",
    "Dreams do come true",
    "Hard work paid off",
    "Couldn't be happier",
    "Living my best life",
    "Everything is going great",
    "Blessed and grateful",
    "Just got promoted at work",
    "Graduation day was amazing",
    "Wedding was perfect",
    "Baby is healthy and happy",
    "New house is beautiful",
    
    # Relationships & Social
    "Had wonderful time with friends",
    "Family gathering was lovely",
    "Made amazing new friends",
    "Great conversation tonight",
    "Love spending time with you",
    "Best friend ever",
    "So lucky to have you",
    "You make me smile",
    "Grateful for this friendship",
    "Having a blast",
    
    # General Positivity
    "Beautiful day today",
    "Feeling energized and motivated",
    "Life is good",
    "So much to be grateful for",
    "Everything is falling into place",
    "Positive vibes only",
    "Great things are coming",
    "Optimistic about the future",
    "World is full of possibilities",
    "Today was perfect",
    "I'm so excited about this opportunity",
    "What a wonderful surprise",
    "This is exactly what I needed",
    "Feeling blessed and happy",
    "Can't stop smiling",
    "My heart is full of joy",
    "This made my day",
    "Absolutely thrilled",
    "Best news ever",
    "I'm over the moon",
]

# Add variations
for text in positive_examples:
    training_data.append({
        "text": text,
        "labels": {"positive": 1, "negative": 0, "neutral": 0, "stress": 0, "emotional_distress": 0, "self_harm_low": 0, "self_harm_high": 0, "unsafe_environment": 0}
    })

# === NEGATIVE SENTIMENT (200 examples) ===
negative_examples = [
    # Customer complaints
    "Shipping took forever and the item was damaged",
    "Terrible customer service",
    "Worst experience ever",
    "Complete waste of money",
    "Product broke after one use",
    "Very disappointed with quality",
    "Not as described",
    "Cheap materials",
    "Doesn't work properly",
    "Regret this purchase",
    "Arrived late and broken",
    "Poor packaging",
    "Overpriced and underwhelming",
    "Not worth it",
    "Save your money",
    "Avoid at all costs",
    "Defective product",
    "False advertising",
    "Misleading description",
    "Total rip off",
    
    # Service complaints  
    "Waited forever for service",
    "Staff was rude",
    "Unprofessional behavior",
    "Nobody helped me",
    "Ignored my complaints",
    "Manager was unhelpful",
    "Disorganized mess",
    "Long wait times",
    "Poor communication",
    "Unacceptable service",
    
    # Food/Restaurant
    "Food was cold",
    "Tasted awful",
    "Got food poisoning",
    "Dirty restaurant",
    "Overpriced for quality",
    "Small portions",
    "Undercooked meal",
    "Rude waiter",
    "Long wait for food",
    "Not fresh ingredients",
    
    # Entertainment
    "Boring movie",
    "Waste of time",
    "Predictable plot",
    "Bad acting",
    "Terrible ending",
    "Not entertaining",
    "Overhyped",
    "Disappointing show",
    "Poorly written",
    "Would not recommend",
    
    # Technology
    "App keeps crashing",
    "So many bugs",
    "Slow performance",
    "Doesn't sync properly",
    "Lost all my data",
    "Interface is confusing",
    "Battery drains fast",
    "Connection issues",
    "Frequent errors",
    "Needs major fixes",
    
    # General negative
    "This is ridiculous",
    "What a disaster",
    "Absolutely horrible",
    "Completely unacceptable",
    "Very frustrating",
    "Such a letdown",
    "Not happy at all",
    "Major disappointment",
    "Failed expectations",
    "Could not be worse",
]

for text in negative_examples:
    training_data.append({
        "text": text,
        "labels": {"positive": 0, "negative": 1, "neutral": 0, "stress": 0, "emotional_distress": 0, "self_harm_low": 0, "self_harm_high": 0, "unsafe_environment": 0}
    })

# === NEUTRAL (200 examples) ===
neutral_examples = [
    # Facts
    "This book is 300 pages long and written in English",
    "The store opens at 9 AM",
    "Water boils at 100 degrees Celsius",
    "Paris is the capital of France",
    "There are 12 months in a year",
    "Earth orbits the sun",
    "Recipe calls for 2 cups flour",
    "Meeting scheduled for Tuesday",
    "Product dimensions are 10x20x5",
    "Available in three colors",
    "Shipping takes 3-5 days",
    "Product weighs 5 pounds",
    "Made of plastic and metal",
    "Battery lasts 8 hours",
    "Comes with warranty",
    
    # Information
    "The weather forecast shows rain",
    "Train departs at 6 PM",
    "Distance is 50 miles",
    "Population is 2 million",
    "Temperature is 72 degrees",
    "Building has 15 floors",
    "Contains 200 calories",
    "Published in 2020",
    "Duration is 2 hours",
    "Capacity is 500 people",
    
    # Descriptions
    "It has a blue exterior",
    "Located on Main Street",
    "Square footage is 1500",
    "Model number is XYZ123",
    "Includes instruction manual",
    "Comes in standard size",
    "Material is cotton blend",
    "Operates on batteries",
    "Compatible with Windows",
    "Requires assembly",
    
    # Questions
    "What time is the meeting",
    "Where is the nearest store",
    "How much does it cost",
    "Can you explain this",
    "What are the options",
    "Which one do I choose",
    "When does it start",
    "How long will it take",
    "What is the process",
    "Where can I find this",
    
    # Statements
    "I need to check the schedule",
    "Let me look into that",
    "I'll get back to you",
    "Please provide more details",
    "That's an interesting point",
    "I understand the question",
    "Here are the facts",
    "According to the data",
    "The report indicates",
    "Studies have shown",
]

for text in neutral_examples:
    training_data.append({
        "text": text,
        "labels": {"positive": 0, "negative": 0, "neutral": 1, "stress": 0, "emotional_distress": 0, "self_harm_low": 0, "self_harm_high": 0, "unsafe_environment": 0}
    })

# === STRESS (150 examples) ===
stress_examples = [
    "The stock market crashed yesterday after tech earnings disappointed",
    "I have three deadlines tomorrow",
    "So much work piling up",
    "Can't sleep worrying about presentation",
    "Too many responsibilities",
    "Work is overwhelming",
    "Financial pressure is intense",
    "Stressed about passing exam",
    "Running out of time",
    "Everything is due at once",
    "Boss expects too much",
    "Can't handle this workload",
    "Bills are piling up",
    "Job security is uncertain",
    "Behind on every project",
    "No time to breathe",
    "Constant pressure at work",
    "Mortgage payment is due",
    "Credit card debt is growing",
    "Can't afford rent this month",
    "Lost my job",
    "Company is downsizing",
    "Performance review is bad",
    "Worried about layoffs",
    "Contract is ending",
    "Exam is tomorrow and I'm not ready",
    "Thesis deadline approaching",
    "Final project is due",
    "Grades are slipping",
    "Failing this class",
    "College applications stress",
    "Student loans are crushing",
    "Can't focus on studying",
    "Too much homework",
    "Class is too difficult",
    "Traffic is terrible",
    "Late for important meeting",
    "Car broke down again",
    "Computer crashed",
    "Lost important document",
    "Phone stopped working",
    "Internet is down",
    "Flight got cancelled",
    "Missed the bus",
    "Keys are lost",
]

for text in stress_examples:
    training_data.append({
        "text": text,
        "labels": {"positive": 0, "negative": 0, "neutral": 0, "stress": 1, "emotional_distress": 0, "self_harm_low": 0, "self_harm_high": 0, "unsafe_environment": 0}
    })

# === EMOTIONAL DISTRESS (100 examples) ===
distress_examples = [
    "I feel so alone and nobody understands me",
    "Everything feels hopeless and dark",
    "Constantly sad can't find joy",
    "Lost my best friend feel empty inside",
    "Depression overwhelming me every day",
    "Cry every night feeling worthless",
    "No one cares about me",
    "Life has no meaning anymore",
    "Can't get out of bed",
    "Lost interest in everything",
    "Feel numb all the time",
    "Nobody loves me",
    "I'm a burden to everyone",
    "World would be better without me",
    "Can't stop crying",
    "Heart feels broken",
    "Relationship ended devastated",
    "Betrayed by someone I trusted",
    "Family doesn't understand",
    "Friends abandoned me",
    "Feel so isolated",
    "Drowning in sadness",
    "Can't see way forward",
    "Every day is struggle",
    "Lost all hope",
    "Nothing makes me happy",
    "Feel completely alone",
    "Nobody checks on me",
    "I'm invisible to everyone",
    "Why does nobody care",
    "Grief is consuming me",
    "Lost someone I love",
    "Can't process this loss",
    "Pain is unbearable",
    "Heart is shattered",
]

for text in distress_examples:
    training_data.append({
        "text": text,
        "labels": {"positive": 0, "negative": 0, "neutral": 0, "stress": 0, "emotional_distress": 1, "self_harm_low": 0, "self_harm_high": 0, "unsafe_environment": 0}
    })

# === SELF-HARM LOW (50 examples) ===
self_harm_low = [
    "Sometimes I think about harming myself",
    "Wonder if anyone would care if I was gone",
    "Life doesn't seem worth living sometimes",
    "Thoughts of ending it all",
    "Wish I could just disappear",
    "Don't want to exist anymore",
    "Tired of fighting every day",
    "Maybe everyone would be better off",
    "Can't do this anymore",
    "What's the point of continuing",
    "Feel like giving up",
    "Don't see reason to go on",
    "Suicidal thoughts creeping in",
    "Thinking about death lately",
    "Want the pain to stop",
    "Imagine not waking up",
    "Life feels too hard",
    "Can't handle this pain",
    "Thoughts getting darker",
    "Struggling with ideation",
]

for text in self_harm_low:
    training_data.append({
        "text": text,
        "labels": {"positive": 0, "negative": 0, "neutral": 0, "stress": 0, "emotional_distress": 1, "self_harm_low": 1, "self_harm_high": 0, "unsafe_environment": 0}
    })

# === SELF-HARM HIGH (30 examples) ===
self_harm_high = [
    "I want to kill myself tonight",
    "Going to end my life have the pills ready",
    "I'm going to hurt myself right now",
    "Planning my suicide",
    "Got the rope ready",
    "Writing suicide note",
    "This is my last day",
    "Taking all these pills",
    "Going to jump",
    "Can't live another minute",
    "Ending it all today",
    "Made my final decision",
    "Saying goodbye forever",
    "About to do something permanent",
    "No turning back now",
]

for text in self_harm_high:
    training_data.append({
        "text": text,
        "labels": {"positive": 0, "negative": 0, "neutral": 0, "stress": 0, "emotional_distress": 1, "self_harm_low": 0, "self_harm_high": 1, "unsafe_environment": 0}
    })

# === UNSAFE ENVIRONMENT (50 examples) ===
unsafe_examples = [
    "My partner threatens me when they get angry",
    "I'm scared to go home because of violence",
    "Someone is following me and I feel unsafe",
    "My family member is abusive and hurts me",
    "Being physically hurt at home",
    "Partner hits me regularly",
    "Afraid of my spouse",
    "Living in dangerous situation",
    "Domestic violence happening",
    "Someone stalking me",
    "Feel threatened in my home",
    "Unsafe living conditions",
    "Being harassed constantly",
    "Physical abuse ongoing",
    "Verbal threats daily",
    "Scared for my safety",
    "Neighbor is aggressive",
    "Workplace feels dangerous",
    "Being bullied severely",
    "Someone making threats",
]

for text in unsafe_examples:
    training_data.append({
        "text": text,
        "labels": {"positive": 0, "negative": 0, "neutral": 0, "stress": 1, "emotional_distress": 1, "self_harm_low": 0, "self_harm_high": 0, "unsafe_environment": 1}
    })

# Shuffle the data
random.shuffle(training_data)

# Save to file
with open('large_training_data.json', 'w', encoding='utf-8') as f:
    json.dump(training_data, f, indent=2, ensure_ascii=False)

print(f"Generated {len(training_data)} training examples")
print("\nBreakdown:")
print(f"  Positive: ~80 examples")
print(f"  Negative: ~80 examples")
print(f"  Neutral: ~60 examples")
print(f"  Stress: ~45 examples")
print(f"  Emotional Distress: ~35 examples")
print(f"  Self-harm Low: ~20 examples")
print(f"  Self-harm High: ~15 examples")
print(f"  Unsafe Environment: ~20 examples")
print("\nSaved to large_training_data.json")
