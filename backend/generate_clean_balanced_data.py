"""
Generate Clean, Balanced Training Data
======================================
Creates a properly balanced dataset with correct labels to fix model accuracy issues.
Each category has similar size to prevent imbalance.
"""

import json
import random

def generate_clean_balanced_data():
    """Generate clean, balanced training data with correct labels"""
    
    training_data = []
    
    # ========== POSITIVE / CONFIDENT (500 examples) ==========
    positive_confident = [
        "I know I can achieve anything I put my mind to",
        "I am capable of overcoming any challenge",
        "I believe in myself and my abilities",
        "I have the strength to succeed",
        "I can accomplish my goals",
        "I am determined to reach my dreams",
        "I have confidence in my skills",
        "I know I can do this",
        "I am ready to take on anything",
        "I have what it takes to succeed",
        "I am unstoppable when I set my mind to it",
        "I can handle whatever comes my way",
        "I am strong and capable",
        "I will achieve my objectives",
        "I have the power to make things happen",
        "I am confident in my decisions",
        "I know I can make a difference",
        "I am ready to excel",
        "I have the ability to succeed",
        "I can overcome any obstacle",
    ]
    
    # Generate variations
    positive_templates = [
        "I know I can {action}",
        "I am {confident_word} I can {action}",
        "I have {ability_word} to {action}",
        "I {believe_word} I can {action}",
        "I am {capable_word} of {action}",
    ]
    
    actions = ["achieve anything", "succeed", "overcome challenges", "reach my goals", 
              "accomplish my dreams", "make it happen", "do this", "handle this",
              "get through this", "make a difference"]
    confident_words = ["confident", "sure", "certain", "positive", "convinced"]
    ability_words = ["the ability", "the strength", "the power", "what it takes", "the skills"]
    believe_words = ["believe", "know", "trust"]
    capable_words = ["capable", "able", "ready"]
    
    for _ in range(480):
        template = random.choice(positive_templates)
        text = template.format(
            action=random.choice(actions),
            confident_word=random.choice(confident_words),
            ability_word=random.choice(ability_words),
            believe_word=random.choice(believe_words),
            capable_word=random.choice(capable_words)
        )
        positive_confident.append(text)
    
    for text in positive_confident:
        training_data.append({
            "text": text,
            "labels": {"neutral": 1, "stress": 0, "unsafe_environment": 0, 
                      "emotional_distress": 0, "self_harm_low": 0, "self_harm_high": 0}
        })
    
    # ========== NEUTRAL / EVERYDAY (500 examples) ==========
    neutral_everyday = [
        "Tomorrow is a public holiday",
        "The meeting is scheduled for 3 PM",
        "I need to buy groceries",
        "I have a dentist appointment next week",
        "The weather is nice today",
        "I'm going to the gym later",
        "I finished my work project",
        "I need to call my doctor",
        "I'm planning to visit my family",
        "I have a conference call at 2 PM",
        "I went to the store yesterday",
        "I'm going shopping this weekend",
        "I need to pay my bills",
        "I have a meeting tomorrow morning",
        "I'm going to the library to study",
        "I need to pick up my dry cleaning",
        "I have an appointment next month",
        "I'm going to the park for a walk",
        "I need to finish my assignment",
        "I'm going to work early tomorrow",
    ]
    
    # Generate variations
    neutral_templates = [
        "{timeframe} is {event}",
        "I {action} {timeframe}",
        "I have {activity} {timeframe}",
        "I need to {activity}",
        "I'm going to {activity} {timeframe}",
        "The {subject} is {state}",
    ]
    
    timeframes = ["tomorrow", "today", "next week", "next month", "later", "this weekend"]
    events = ["a public holiday", "a meeting", "an appointment", "a conference call"]
    activities = ["buy groceries", "call my doctor", "visit my family", "go to the gym",
                 "go shopping", "pay my bills", "finish my assignment", "go to work"]
    subjects = ["weather", "meeting", "appointment", "call"]
    states = ["nice", "scheduled", "at 3 PM", "at 2 PM", "tomorrow"]
    
    for _ in range(480):
        template = random.choice(neutral_templates)
        text = template.format(
            timeframe=random.choice(timeframes),
            event=random.choice(events),
            action=random.choice(["have", "need", "going to", "will"]),
            activity=random.choice(activities),
            subject=random.choice(subjects),
            state=random.choice(states)
        )
        neutral_everyday.append(text)
    
    for text in neutral_everyday:
        training_data.append({
            "text": text,
            "labels": {"neutral": 1, "stress": 0, "unsafe_environment": 0, 
                      "emotional_distress": 0, "self_harm_low": 0, "self_harm_high": 0}
        })
    
    # ========== STRESS / FRUSTRATION (500 examples) ==========
    stress_frustration = [
        "I'm worried about my exam tomorrow",
        "I'm stressed about work deadlines",
        "I'm feeling overwhelmed with all these tasks",
        "I'm anxious about the presentation",
        "I'm nervous about the interview",
        "I'm feeling pressured by responsibilities",
        "I'm stressed about meeting my goals",
        "I'm worried about my financial situation",
        "I'm anxious about upcoming changes",
        "I'm feeling overwhelmed",
        "I feel like nothing I do ever works out",
        "I'm frustrated that things aren't going well",
        "I'm annoyed by all these problems",
        "I'm irritated by the constant delays",
        "I'm tired of dealing with this",
        "I'm fed up with all these issues",
        "I'm frustrated with how things are going",
        "I'm stressed about balancing everything",
        "I'm worried about the future",
        "I'm feeling pressured",
    ]
    
    # Generate variations
    stress_templates = [
        "I'm {stress_word} about {stress_source}",
        "I feel {stress_word} {stress_context}",
        "I'm {stress_word} with {stress_source}",
        "I feel like {stress_feeling}",
        "I'm {stress_word} that {stress_situation}",
    ]
    
    stress_words = ["worried", "stressed", "anxious", "nervous", "overwhelmed", 
                   "pressured", "frustrated", "annoyed", "irritated", "tired"]
    stress_sources = ["my exam", "work deadlines", "the presentation", "the interview",
                     "all these tasks", "meeting my goals", "my financial situation",
                     "upcoming changes", "balancing everything", "the future"]
    stress_contexts = ["lately", "recently", "these days", "all the time", "constantly"]
    stress_feelings = ["nothing I do ever works out", "things aren't going well",
                      "I can't catch a break", "everything is going wrong"]
    stress_situations = ["things aren't working", "I can't make progress", 
                       "nothing is going right", "I'm falling behind"]
    
    for _ in range(480):
        template = random.choice(stress_templates)
        text = template.format(
            stress_word=random.choice(stress_words),
            stress_source=random.choice(stress_sources),
            stress_context=random.choice(stress_contexts),
            stress_feeling=random.choice(stress_feelings),
            stress_situation=random.choice(stress_situations)
        )
        stress_frustration.append(text)
    
    for text in stress_frustration:
        training_data.append({
            "text": text,
            "labels": {"neutral": 0, "stress": 1, "unsafe_environment": 0, 
                      "emotional_distress": 1, "self_harm_low": 0, "self_harm_high": 0}
        })
    
    # ========== EMOTIONAL DISTRESS (500 examples) ==========
    emotional_distress = [
        "I feel so overwhelmed and sad all the time",
        "I'm feeling really down and hopeless",
        "I feel empty and meaningless",
        "I'm so depressed and can't find joy",
        "I feel like nothing matters anymore",
        "I'm feeling really low and don't know what to do",
        "I feel so alone and isolated",
        "I'm feeling really depressed and hopeless",
        "I feel like I'm drowning in sadness",
        "I'm so overwhelmed by negative emotions",
        "I feel like I can't cope anymore",
        "I'm feeling so empty inside",
        "I feel like I'm losing myself",
        "I'm feeling so lost and confused",
        "I feel like I'm breaking down",
        "I'm feeling so hopeless about everything",
        "I feel like I'm stuck and can't move forward",
        "I'm feeling so alone in this",
        "I feel like I'm falling apart",
        "I'm feeling so overwhelmed by life",
    ]
    
    # Generate variations
    distress_templates = [
        "I feel {distress_word} {distress_context}",
        "I'm feeling {distress_word} {distress_context}",
        "I feel {distress_word} and {distress_word2}",
        "I'm feeling {distress_word} about {distress_source}",
    ]
    
    distress_words = ["overwhelmed", "sad", "down", "hopeless", "empty", "meaningless",
                     "depressed", "alone", "isolated", "drowning", "lost", "broken",
                     "stuck", "falling apart", "can't cope"]
    distress_contexts = ["all the time", "lately", "recently", "these days", "now", "constantly"]
    distress_sources = ["everything", "life", "the situation", "things", "my future"]
    
    for _ in range(480):
        template = random.choice(distress_templates)
        text = template.format(
            distress_word=random.choice(distress_words),
            distress_word2=random.choice(distress_words),
            distress_context=random.choice(distress_contexts),
            distress_source=random.choice(distress_sources)
        )
        emotional_distress.append(text)
    
    for text in emotional_distress:
        training_data.append({
            "text": text,
            "labels": {"neutral": 0, "stress": 0, "unsafe_environment": 0, 
                      "emotional_distress": 1, "self_harm_low": 0, "self_harm_high": 0}
        })
    
    # ========== SELF-HARM LOW RISK (300 examples) ==========
    self_harm_low = [
        "I sometimes think about ending it all",
        "I wonder if anyone would notice if I was gone",
        "I've been having thoughts about hurting myself",
        "I sometimes think about suicide",
        "I wonder if people would be better off without me",
        "I've been considering ending my life",
        "I sometimes think about not being here anymore",
        "I wonder if anyone would care if I disappeared",
        "I've been having suicidal thoughts",
        "I sometimes think the world would be better without me",
    ]
    
    # Generate variations
    self_harm_low_templates = [
        "I {frequency} think about {self_harm_thought}",
        "I wonder if {self_harm_wonder}",
        "I've been {self_harm_considering}",
        "I {frequency} have thoughts about {self_harm_thought}",
    ]
    
    frequencies = ["sometimes", "occasionally", "often", "frequently", "at times"]
    self_harm_thoughts = ["ending it all", "hurting myself", "suicide", "not being here",
                         "ending my life", "killing myself"]
    self_harm_wonders = ["anyone would notice if I was gone", "people would be better off without me",
                        "anyone would care if I disappeared", "the world would be better without me"]
    self_harm_considering = ["having thoughts about ending it", "considering suicide",
                            "thinking about hurting myself", "having suicidal thoughts"]
    
    for _ in range(290):
        template = random.choice(self_harm_low_templates)
        text = template.format(
            frequency=random.choice(frequencies),
            self_harm_thought=random.choice(self_harm_thoughts),
            self_harm_wonder=random.choice(self_harm_wonders),
            self_harm_considering=random.choice(self_harm_considering)
        )
        self_harm_low.append(text)
    
    for text in self_harm_low:
        training_data.append({
            "text": text,
            "labels": {"neutral": 0, "stress": 0, "unsafe_environment": 0, 
                      "emotional_distress": 0, "self_harm_low": 1, "self_harm_high": 0}
        })
    
    # ========== SELF-HARM HIGH RISK (300 examples) ==========
    self_harm_high = [
        "I want to kill myself and end my pain",
        "I'm planning to hurt myself tonight",
        "I have a plan to end my life",
        "I'm going to kill myself",
        "I want to die and end it all",
        "I'm ready to hurt myself",
        "I have everything I need to end it",
        "I'm going to hurt myself tonight",
        "I want to end my life and I have a plan",
        "I'm planning to commit suicide",
    ]
    
    # Generate variations
    self_harm_high_templates = [
        "I {intent} to {self_harm_action}",
        "I'm {planning} to {self_harm_action} {timeframe}",
        "I have {plan} to {self_harm_action}",
        "I {intent} to {self_harm_action} and {self_harm_reason}",
    ]
    
    intents = ["want", "am going", "will", "plan"]
    self_harm_actions = ["kill myself", "hurt myself", "end my life", "end it all",
                        "commit suicide", "end my pain"]
    planning = ["planning", "ready", "prepared"]
    plans = ["a plan", "everything I need", "a way"]
    self_harm_reasons = ["end my pain", "end it all", "end the suffering"]
    timeframes = ["tonight", "today", "this week", "soon"]
    
    for _ in range(290):
        template = random.choice(self_harm_high_templates)
        text = template.format(
            intent=random.choice(intents),
            self_harm_action=random.choice(self_harm_actions),
            timeframe=random.choice(timeframes),
            planning=random.choice(planning),
            plan=random.choice(plans),
            self_harm_reason=random.choice(self_harm_reasons)
        )
        self_harm_high.append(text)
    
    for text in self_harm_high:
        training_data.append({
            "text": text,
            "labels": {"neutral": 0, "stress": 0, "unsafe_environment": 0, 
                      "emotional_distress": 0, "self_harm_low": 0, "self_harm_high": 1}
        })
    
    # ========== UNSAFE ENVIRONMENT (300 examples) ==========
    unsafe_environment = [
        "I want to hurt you for what you did",
        "I'm going to get you for this",
        "I will attack them for hurting me",
        "I want to harm those people",
        "I'm going to hurt someone who wronged me",
        "I will get revenge on them",
        "I want to hurt the people who hurt me",
        "I'm going to attack those who wronged me",
        "I will harm the people responsible",
        "I want to hurt those who betrayed me",
    ]
    
    # Generate variations
    unsafe_templates = [
        "I {intent} to {harm_action} {target}",
        "I'm going to {harm_action} {target} {reason}",
        "I will {harm_action} {target}",
    ]
    
    harm_actions = ["hurt", "kill", "harm", "attack", "get", "get revenge on"]
    targets = ["you", "them", "him", "her", "those people", "someone",
              "the people who hurt me", "those who wronged me", "the people responsible"]
    reasons = ["for what you did", "for this", "for hurting me", "who wronged me"]
    
    for _ in range(290):
        template = random.choice(unsafe_templates)
        text = template.format(
            intent=random.choice(intents),
            harm_action=random.choice(harm_actions),
            target=random.choice(targets),
            reason=random.choice(reasons)
        )
        unsafe_environment.append(text)
    
    for text in unsafe_environment:
        training_data.append({
            "text": text,
            "labels": {"neutral": 0, "stress": 0, "unsafe_environment": 1, 
                      "emotional_distress": 0, "self_harm_low": 0, "self_harm_high": 0}
        })
    
    # Shuffle
    random.shuffle(training_data)
    
    # Split into train and validation (80/20)
    split_idx = int(len(training_data) * 0.8)
    train_data = training_data[:split_idx]
    val_data = training_data[split_idx:]
    
    # Save to files
    with open('clean_balanced_train.json', 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    
    with open('clean_balanced_val.json', 'w', encoding='utf-8') as f:
        json.dump(val_data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Generated {len(train_data)} training examples")
    print(f"✓ Generated {len(val_data)} validation examples")
    print(f"✓ Total: {len(training_data)} examples")
    print("\nBalanced Categories:")
    print(f"  - Positive/Confident: ~500")
    print(f"  - Neutral/Everyday: ~500")
    print(f"  - Stress/Frustration: ~500")
    print(f"  - Emotional Distress: ~500")
    print(f"  - Self-Harm Low: ~300")
    print(f"  - Self-Harm High: ~300")
    print(f"  - Unsafe Environment: ~300")
    print("\n✓ Files saved: clean_balanced_train.json, clean_balanced_val.json")


if __name__ == '__main__':
    random.seed(42)
    generate_clean_balanced_data()

