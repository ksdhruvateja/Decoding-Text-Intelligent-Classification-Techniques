"""
Generate Complex Training Data
==============================
Creates training examples for complex text scenarios
"""

import json

def generate_complex_training_data():
    """Generate training data for complex text scenarios"""
    
    training_data = []
    
    # 1. Ambiguous self-harm ideation (should be self_harm_low, not high)
    ambiguous_self_harm = [
        {
            "text": "I'm not saying I want to hurt myself, but sometimes I wonder if anyone would notice if I was gone.",
            "labels": {"neutral": 0, "stress": 0, "unsafe_environment": 0, "emotional_distress": 1, "self_harm_low": 1, "self_harm_high": 0}
        },
        {
            "text": "Sometimes I think about what it would be like if I wasn't here anymore. Not that I would do anything, just wondering.",
            "labels": {"neutral": 0, "stress": 0, "unsafe_environment": 0, "emotional_distress": 1, "self_harm_low": 1, "self_harm_high": 0}
        },
        {
            "text": "I've been having thoughts that maybe I'm not meant to be here, you know? Like maybe the world would be better off.",
            "labels": {"neutral": 0, "stress": 0, "unsafe_environment": 0, "emotional_distress": 1, "self_harm_low": 1, "self_harm_high": 0}
        },
        {
            "text": "I wonder if anyone would care if I disappeared. Probably not.",
            "labels": {"neutral": 0, "stress": 0, "unsafe_environment": 0, "emotional_distress": 1, "self_harm_low": 1, "self_harm_high": 0}
        }
    ]
    
    # 2. Long complex sentences with distress
    long_complex_distress = [
        {
            "text": "I've been thinking a lot lately about how life seems to be getting harder and harder, and even though I try to stay positive and tell myself things will get better, there are moments when I feel completely overwhelmed and wonder if I'm strong enough to handle everything that's coming my way.",
            "labels": {"neutral": 0, "stress": 1, "unsafe_environment": 0, "emotional_distress": 1, "self_harm_low": 0, "self_harm_high": 0}
        },
        {
            "text": "I know I should be happy because I have everything I need, but I can't shake this feeling of emptiness, and even though I try to focus on the positive, the negative thoughts keep creeping in.",
            "labels": {"neutral": 0, "stress": 0, "unsafe_environment": 0, "emotional_distress": 1, "self_harm_low": 0, "self_harm_high": 0}
        },
        {
            "text": "Everything feels so heavy right now, like I'm carrying the weight of the world on my shoulders, and no matter how hard I try, I can't seem to find a way to make things better or feel like myself again.",
            "labels": {"neutral": 0, "stress": 1, "unsafe_environment": 0, "emotional_distress": 1, "self_harm_low": 0, "self_harm_high": 0}
        }
    ]
    
    # 3. Mixed emotions (should detect stress/distress, not positive)
    mixed_emotions = [
        {
            "text": "I'm excited about the new job but also terrified because I don't know if I'm good enough and I'm worried I'll fail and disappoint everyone.",
            "labels": {"neutral": 0, "stress": 1, "unsafe_environment": 0, "emotional_distress": 1, "self_harm_low": 0, "self_harm_high": 0}
        },
        {
            "text": "I'm happy about the promotion but anxious about the increased responsibility and scared I won't be able to handle it.",
            "labels": {"neutral": 0, "stress": 1, "unsafe_environment": 0, "emotional_distress": 1, "self_harm_low": 0, "self_harm_high": 0}
        },
        {
            "text": "I should be grateful for what I have, but I can't help feeling sad and empty inside, like something is missing.",
            "labels": {"neutral": 0, "stress": 0, "unsafe_environment": 0, "emotional_distress": 1, "self_harm_low": 0, "self_harm_high": 0}
        }
    ]
    
    # 4. Context-dependent distress (should be emotional_distress/stress, not self_harm_high)
    context_distress = [
        {
            "text": "I can't take this anymore. I really can't.",
            "labels": {"neutral": 0, "stress": 1, "unsafe_environment": 0, "emotional_distress": 1, "self_harm_low": 0, "self_harm_high": 0}
        },
        {
            "text": "I'm done. I can't do this anymore.",
            "labels": {"neutral": 0, "stress": 1, "unsafe_environment": 0, "emotional_distress": 1, "self_harm_low": 0, "self_harm_high": 0}
        },
        {
            "text": "This is too much. I can't handle it.",
            "labels": {"neutral": 0, "stress": 1, "unsafe_environment": 0, "emotional_distress": 1, "self_harm_low": 0, "self_harm_high": 0}
        }
    ]
    
    # 5. Constructive negative (should be stress/neutral, not positive)
    constructive_negative = [
        {
            "text": "The project didn't go as planned, and I'm disappointed, but I'll learn from this experience and do better next time.",
            "labels": {"neutral": 1, "stress": 1, "unsafe_environment": 0, "emotional_distress": 0, "self_harm_low": 0, "self_harm_high": 0}
        },
        {
            "text": "I made a mistake, but I'm going to fix it and make sure it doesn't happen again.",
            "labels": {"neutral": 1, "stress": 0, "unsafe_environment": 0, "emotional_distress": 0, "self_harm_low": 0, "self_harm_high": 0}
        },
        {
            "text": "Things didn't work out the way I hoped, but I'm determined to keep trying and improve.",
            "labels": {"neutral": 1, "stress": 1, "unsafe_environment": 0, "emotional_distress": 0, "self_harm_low": 0, "self_harm_high": 0}
        }
    ]
    
    # 6. Threats to others (should be unsafe_environment)
    threats_to_others = [
        {
            "text": "If they don't stop bothering me, I'm going to make them regret it. I know exactly what I'll do.",
            "labels": {"neutral": 0, "stress": 0, "unsafe_environment": 1, "emotional_distress": 0, "self_harm_low": 0, "self_harm_high": 0}
        },
        {
            "text": "I'm going to hurt them if they don't leave me alone.",
            "labels": {"neutral": 0, "stress": 0, "unsafe_environment": 1, "emotional_distress": 0, "self_harm_low": 0, "self_harm_high": 0}
        },
        {
            "text": "They better watch out because I'm not going to take this anymore.",
            "labels": {"neutral": 0, "stress": 1, "unsafe_environment": 1, "emotional_distress": 0, "self_harm_low": 0, "self_harm_high": 0}
        }
    ]
    
    # 7. Metaphorical distress
    metaphorical_distress = [
        {
            "text": "I feel like I'm drowning in responsibilities and there's no one to throw me a lifeline.",
            "labels": {"neutral": 0, "stress": 1, "unsafe_environment": 0, "emotional_distress": 1, "self_harm_low": 0, "self_harm_high": 0}
        },
        {
            "text": "I'm suffocating under all this pressure and can't breathe.",
            "labels": {"neutral": 0, "stress": 1, "unsafe_environment": 0, "emotional_distress": 1, "self_harm_low": 0, "self_harm_high": 0}
        },
        {
            "text": "I'm trapped in this situation with no way out.",
            "labels": {"neutral": 0, "stress": 1, "unsafe_environment": 0, "emotional_distress": 1, "self_harm_low": 0, "self_harm_high": 0}
        }
    ]
    
    # 8. Rhetorical questions showing distress
    rhetorical_distress = [
        {
            "text": "What's the point of trying when nothing ever works out? Why do I even bother?",
            "labels": {"neutral": 0, "stress": 1, "unsafe_environment": 0, "emotional_distress": 1, "self_harm_low": 0, "self_harm_high": 0}
        },
        {
            "text": "Why does everything always go wrong? What did I do to deserve this?",
            "labels": {"neutral": 0, "stress": 1, "unsafe_environment": 0, "emotional_distress": 1, "self_harm_low": 0, "self_harm_high": 0}
        },
        {
            "text": "What's the use? Nothing I do matters anyway.",
            "labels": {"neutral": 0, "stress": 1, "unsafe_environment": 0, "emotional_distress": 1, "self_harm_low": 0, "self_harm_high": 0}
        }
    ]
    
    # 9. Past tense recovery (should be neutral/positive, not self_harm)
    past_tense_recovery = [
        {
            "text": "I used to think about hurting myself, but I'm in a much better place now and have support.",
            "labels": {"neutral": 1, "stress": 0, "unsafe_environment": 0, "emotional_distress": 0, "self_harm_low": 0, "self_harm_high": 0}
        },
        {
            "text": "I had thoughts of suicide in the past, but I've been getting help and things are improving.",
            "labels": {"neutral": 1, "stress": 0, "unsafe_environment": 0, "emotional_distress": 0, "self_harm_low": 0, "self_harm_high": 0}
        },
        {
            "text": "I used to feel hopeless, but now I'm working with a therapist and making progress.",
            "labels": {"neutral": 1, "stress": 0, "unsafe_environment": 0, "emotional_distress": 0, "self_harm_low": 0, "self_harm_high": 0}
        }
    ]
    
    # Combine all
    training_data.extend(ambiguous_self_harm)
    training_data.extend(long_complex_distress)
    training_data.extend(mixed_emotions)
    training_data.extend(context_distress)
    training_data.extend(constructive_negative)
    training_data.extend(threats_to_others)
    training_data.extend(metaphorical_distress)
    training_data.extend(rhetorical_distress)
    training_data.extend(past_tense_recovery)
    
    return training_data


def main():
    """Generate and save complex training data"""
    print("="*80)
    print("GENERATING COMPLEX TRAINING DATA")
    print("="*80)
    
    data = generate_complex_training_data()
    
    print(f"\nGenerated {len(data)} complex training examples")
    
    # Save to file
    output_file = 'complex_training_data.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Saved to {output_file}")
    
    # Show breakdown
    print("\nBreakdown by category:")
    categories = {
        "Ambiguous self-harm": 4,
        "Long complex distress": 3,
        "Mixed emotions": 3,
        "Context-dependent distress": 3,
        "Constructive negative": 3,
        "Threats to others": 3,
        "Metaphorical distress": 3,
        "Rhetorical questions": 3,
        "Past tense recovery": 3
    }
    
    for cat, count in categories.items():
        print(f"  {cat}: {count} examples")
    
    print(f"\nTotal: {len(data)} examples")
    print("\nNext: Merge with existing training data and retrain model")


if __name__ == '__main__':
    main()

