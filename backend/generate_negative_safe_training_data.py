"""
Generate Training Data for Negative but SAFE Cases
===================================================
Generates examples of:
- Angry but SAFE statements
- Depressed but NOT suicidal statements
- Ambiguous emotional sentences
"""

def generate_negative_safe_data():
    """Generate training data for negative but safe cases"""
    
    # Angry but SAFE statements
    angry_safe = [
        ("I'm so angry at my boss for not giving me a raise", "stress", "concerning"),
        ("This customer service is terrible, I'm furious", "stress", "concerning"),
        ("I hate how slow this internet connection is", "stress", "concerning"),
        ("I'm really annoyed with this traffic jam", "stress", "concerning"),
        ("This is so frustrating, why does this keep happening", "stress", "concerning"),
        ("I'm mad at myself for making that mistake", "stress", "concerning"),
        ("This situation is really irritating me", "stress", "concerning"),
        ("I can't stand how unorganized this place is", "stress", "concerning"),
        ("I'm frustrated with my computer crashing all the time", "stress", "concerning"),
        ("This is annoying, I wish things were different", "stress", "concerning"),
        ("I'm angry but I'll get over it", "stress", "concerning"),
        ("This makes me so mad, but it's not the end of the world", "stress", "concerning"),
        ("I'm frustrated but I know it will pass", "stress", "concerning"),
        ("I hate when this happens, but I'll deal with it", "stress", "concerning"),
        ("This is really annoying, but I can handle it", "stress", "concerning"),
    ]
    
    # Depressed but NOT suicidal statements
    depressed_safe = [
        ("I've been feeling really down lately", "emotional_distress", "concerning"),
        ("I'm going through a tough time right now", "emotional_distress", "concerning"),
        ("I feel sad and don't know why", "emotional_distress", "concerning"),
        ("I've been feeling depressed for a while", "emotional_distress", "concerning"),
        ("I'm struggling with my emotions", "emotional_distress", "concerning"),
        ("I feel hopeless about my situation", "emotional_distress", "concerning"),
        ("I'm feeling really low today", "emotional_distress", "concerning"),
        ("I've been having a hard time lately", "emotional_distress", "concerning"),
        ("I feel empty inside sometimes", "emotional_distress", "concerning"),
        ("I'm going through a rough patch", "emotional_distress", "concerning"),
        ("I've been feeling down but I'm getting help", "emotional_distress", "concerning"),
        ("I'm sad but I know things will get better", "emotional_distress", "concerning"),
        ("I feel depressed but I'm talking to someone about it", "emotional_distress", "concerning"),
        ("I'm struggling but I'm not giving up", "emotional_distress", "concerning"),
        ("I feel hopeless sometimes but I'm working on it", "emotional_distress", "concerning"),
    ]
    
    # Ambiguous emotional sentences
    ambiguous_emotional = [
        ("I don't know how I feel anymore", "emotional_distress", "concerning"),
        ("Sometimes I wonder if things will ever get better", "emotional_distress", "concerning"),
        ("I'm not sure what to do with these feelings", "emotional_distress", "concerning"),
        ("I feel confused about my emotions", "emotional_distress", "concerning"),
        ("I don't understand why I feel this way", "emotional_distress", "concerning"),
        ("I'm having trouble processing my feelings", "emotional_distress", "concerning"),
        ("I feel lost and don't know where to turn", "emotional_distress", "concerning"),
        ("I'm not sure if I'm okay or not", "emotional_distress", "concerning"),
        ("I feel like something's wrong but I can't explain it", "emotional_distress", "concerning"),
        ("I'm confused about what I'm feeling", "emotional_distress", "concerning"),
    ]
    
    # Toxic/aggressive but NOT self-harm
    toxic_safe = [
        ("I hate you so much right now", "unsafe_environment", "concerning"),
        ("You're such an idiot, I can't stand you", "unsafe_environment", "concerning"),
        ("I want to hurt you for what you did", "unsafe_environment", "concerning"),
        ("You're a terrible person and I despise you", "unsafe_environment", "concerning"),
        ("I'm going to get back at you for this", "unsafe_environment", "concerning"),
        ("You make me so angry, I could scream", "stress", "concerning"),
        ("I'm furious with you right now", "stress", "concerning"),
        ("You're driving me crazy with your behavior", "stress", "concerning"),
        ("I can't stand being around you anymore", "stress", "concerning"),
        ("You're really getting on my nerves", "stress", "concerning"),
    ]
    
    all_data = angry_safe + depressed_safe + ambiguous_emotional + toxic_safe
    
    return all_data


if __name__ == '__main__':
    data = generate_negative_safe_data()
    print(f"Generated {len(data)} training examples")
    print("\nSample examples:")
    for i, (text, emotion, sentiment) in enumerate(data[:5], 1):
        print(f"{i}. {text}")
        print(f"   -> {emotion} ({sentiment})")
        print()

