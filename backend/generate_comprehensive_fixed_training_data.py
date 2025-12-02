"""
Generate Comprehensive Training Data - Fixed for All Edge Cases
================================================================
Creates a diverse, balanced dataset covering ALL statement types including:
- Positive confident/empowered statements
- Positive relationship/love statements  
- Frustration/annoyance (NOT self-harm)
- Positive statements (restaurant, service, general)
- Neutral statements
- Stress statements
- Emotional distress
- Self-harm (high and low risk)
- Unsafe environment
- All edge cases we've been fixing
"""

import json
import random

def generate_comprehensive_fixed_data():
    """Generate comprehensive training data covering all scenarios"""
    
    training_data = []
    
    # ========== POSITIVE CONFIDENT/EMPOWERED STATEMENTS (100 examples) ==========
    confident_empowered = [
        "I am unstoppable and nothing can hold me back",
        "I will rise above all obstacles and succeed",
        "I'm confident I can overcome any challenge",
        "I am powerful and capable of achieving my goals",
        "I will conquer every obstacle in my path",
        "I'm determined to rise higher and succeed",
        "I am a champion and I will win",
        "I'm fearless and ready to take on anything",
        "I will overcome all challenges and thrive",
        "I am indomitable and nothing can stop me",
        "I will rise up from any setback stronger",
        "Every obstacle makes me stronger and more capable",
        "I fell down but I will rise again even stronger",
        "I'm a warrior and I will fight through anything",
        "I am invincible and ready to dominate",
        "I will soar above all difficulties",
        "I'm unbreakable and nothing can defeat me",
        "I will climb to the top despite all challenges",
        "I am formidable and will succeed",
        "I will prosper despite all obstacles",
    ]
    
    for text in confident_empowered:
        training_data.append({
            "text": text,
            "labels": {"neutral": 1, "stress": 0, "unsafe_environment": 0, 
                      "emotional_distress": 0, "self_harm_low": 0, "self_harm_high": 0}
        })
    
    # Generate variations
    confident_templates = [
        "I am {confident_word} and {action}",
        "I will {rise_action} {obstacle_context}",
        "I'm {confident_word} enough to {action}",
        "I will {overcome_action} {obstacle_word}",
        "{obstacle_word} make me {stronger_word}",
    ]
    
    confident_words = ["confident", "empowered", "motivated", "unstoppable", "unbreakable", 
                      "indomitable", "invincible", "strong", "powerful", "capable", 
                      "determined", "resilient", "fearless", "brave", "courageous"]
    rise_actions = ["rise", "rise up", "rise higher", "soar", "climb", "ascend"]
    overcome_actions = ["overcome", "conquer", "beat", "master", "tackle", "handle"]
    obstacle_words = ["obstacles", "challenges", "difficulties", "problems", "struggles", 
                     "setbacks", "barriers", "hurdles"]
    stronger_words = ["stronger", "better", "smarter", "wiser", "more resilient", "more capable"]
    obstacle_contexts = ["above obstacles", "beyond challenges", "from difficulties", 
                        "despite struggles", "through problems"]
    actions = ["succeed", "win", "excel", "thrive", "flourish", "prosper"]
    
    for _ in range(80):
        template = random.choice(confident_templates)
        text = template.format(
            confident_word=random.choice(confident_words),
            action=random.choice(actions),
            rise_action=random.choice(rise_actions),
            obstacle_context=random.choice(obstacle_contexts),
            overcome_action=random.choice(overcome_actions),
            obstacle_word=random.choice(obstacle_words),
            stronger_word=random.choice(stronger_words)
        )
        training_data.append({
            "text": text,
            "labels": {"neutral": 1, "stress": 0, "unsafe_environment": 0, 
                      "emotional_distress": 0, "self_harm_low": 0, "self_harm_high": 0}
        })
    
    # ========== POSITIVE RELATIONSHIP/LOVE STATEMENTS (100 examples) ==========
    relationship_positive = [
        "I will marry her and spend the rest of my life with her",
        "I will marry him and be his wife forever",
        "I will propose to her next month",
        "I will love her for the rest of my life",
        "I will cherish her always",
        "I will protect her and care for her",
        "I will be her husband and support her",
        "I will make her my wife",
        "I will spend my life with him",
        "I will be devoted to her forever",
        "I will treasure our relationship always",
        "I will commit to her for life",
        "I will adore her forever",
        "I will support her through everything",
        "I will be her partner for life",
        "I will marry my fiancée next year",
        "I will love and protect her always",
        "I will care for her and support her",
        "I will be her soulmate forever",
        "I will dedicate my life to her",
    ]
    
    for text in relationship_positive:
        training_data.append({
            "text": text,
            "labels": {"neutral": 1, "stress": 0, "unsafe_environment": 0, 
                      "emotional_distress": 0, "self_harm_low": 0, "self_harm_high": 0}
        })
    
    # Generate variations
    relationship_templates = [
        "I will {relationship_action} {person}",
        "I will {relationship_action} {person} and {commitment_action}",
        "I will be {person} {relationship_role}",
        "I will {relationship_action} {person} {timeframe}",
    ]
    
    relationship_actions = ["marry", "love", "cherish", "adore", "treasure", "protect", 
                           "care for", "support", "help", "be with", "spend my life with"]
    persons = ["her", "him", "you", "them", "my partner", "my love"]
    commitment_actions = ["forever", "always", "for life", "through everything"]
    relationship_roles = ["husband", "wife", "spouse", "partner", "fiancé", "fiancée"]
    timeframes = ["forever", "always", "for life", "next year", "next month"]
    
    for _ in range(80):
        template = random.choice(relationship_templates)
        text = template.format(
            relationship_action=random.choice(relationship_actions),
            person=random.choice(persons),
            commitment_action=random.choice(commitment_actions),
            relationship_role=random.choice(relationship_roles),
            timeframe=random.choice(timeframes)
        )
        training_data.append({
            "text": text,
            "labels": {"neutral": 1, "stress": 0, "unsafe_environment": 0, 
                      "emotional_distress": 0, "self_harm_low": 0, "self_harm_high": 0}
        })
    
    # ========== FRUSTRATION/ANNOYANCE (NOT SELF-HARM) (150 examples) ==========
    frustration_annoyance = [
        "I'm sick of people wasting my time",
        "I'm fed up with all these delays",
        "I'm frustrated with how slow this is",
        "I'm annoyed by all these interruptions",
        "I'm irritated by the constant problems",
        "I can't stand how inefficient this is",
        "I'm frustrated with people not listening",
        "I'm sick of dealing with this nonsense",
        "I'm fed up with all the mistakes",
        "I'm annoyed by people being late",
        "I'm frustrated with the poor service",
        "I'm irritated by all the noise",
        "I can't take this anymore, it's so frustrating",
        "I'm sick of people making excuses",
        "I'm fed up with all the delays",
        "I'm frustrated with how things are going",
        "I'm annoyed by all these problems",
        "I'm irritated by people not following through",
        "I can't stand all these interruptions",
        "I'm sick of people wasting my time with this",
    ]
    
    for text in frustration_annoyance:
        training_data.append({
            "text": text,
            "labels": {"neutral": 0, "stress": 1, "unsafe_environment": 0, 
                      "emotional_distress": 1, "self_harm_low": 0, "self_harm_high": 0}
        })
    
    # Generate variations
    frustration_templates = [
        "I'm {frustration_word} of {frustration_target}",
        "I'm {frustration_word} with {frustration_target}",
        "I'm {frustration_word} by {frustration_target}",
        "I can't {stand_word} {frustration_target}",
        "I'm {frustration_word} with {frustration_target} {frustration_context}",
    ]
    
    frustration_words = ["sick", "fed up", "frustrated", "annoyed", "irritated", "tired"]
    frustration_targets = ["people", "this", "that", "it", "all these delays", 
                          "all these problems", "all these mistakes", "the constant issues",
                          "people wasting my time", "people not listening", "the poor service"]
    stand_words = ["stand", "take"]
    frustration_contexts = ["anymore", "all the time", "constantly", "every day"]
    
    for _ in range(130):
        template = random.choice(frustration_templates)
        text = template.format(
            frustration_word=random.choice(frustration_words),
            frustration_target=random.choice(frustration_targets),
            stand_word=random.choice(stand_words),
            frustration_context=random.choice(frustration_contexts)
        )
        training_data.append({
            "text": text,
            "labels": {"neutral": 0, "stress": 1, "unsafe_environment": 0, 
                      "emotional_distress": 1, "self_harm_low": 0, "self_harm_high": 0}
        })
    
    # ========== POSITIVE STATEMENTS (200 examples) ==========
    
    # Restaurant/Service Positive
    restaurant_positive = [
        "I absolutely loved the new restaurant; the food was delicious and the staff were so friendly.",
        "The restaurant was amazing! Great food and excellent service.",
        "I had a wonderful experience at the hotel. The staff was very helpful.",
        "This place is fantastic! I highly recommend it.",
        "The food was incredible and the atmosphere was perfect.",
        "Best restaurant I've ever been to! Everything was perfect.",
        "I'm so happy with the service I received. It was outstanding.",
        "The staff went above and beyond. Truly exceptional experience.",
        "I loved every moment of my visit. Highly recommend!",
        "The quality was excellent and the service was top-notch.",
    ]
    
    # General Positive
    general_positive = [
        "I'm so happy with my new job! Everything is going great.",
        "I feel really grateful for all the support I've received.",
        "Today was amazing! I accomplished so much.",
        "I'm excited about the upcoming vacation. It's going to be wonderful!",
        "The concert was incredible! Best night ever.",
        "I'm thrilled with how things are progressing.",
        "I feel blessed to have such wonderful friends.",
        "This is the best day I've had in a long time!",
        "I'm so proud of what I've achieved.",
        "Everything is working out perfectly for me.",
    ]
    
    for text in restaurant_positive + general_positive:
        training_data.append({
            "text": text,
            "labels": {"neutral": 1, "stress": 0, "unsafe_environment": 0, 
                      "emotional_distress": 0, "self_harm_low": 0, "self_harm_high": 0}
        })
    
    # Generate variations
    positive_templates = [
        "I {verb} {noun}. It was {positive_adj}!",
        "The {noun} was {positive_adj}. I {positive_verb} it!",
        "I'm so {positive_emotion} about {noun}. Everything is {positive_adj}!",
    ]
    
    positive_verbs = ["loved", "enjoyed", "appreciated", "adored"]
    positive_nouns = ["restaurant", "service", "experience", "event", "show", "movie"]
    positive_adjs = ["amazing", "wonderful", "fantastic", "excellent", "great", "incredible"]
    positive_emotions = ["happy", "excited", "thrilled", "delighted", "grateful"]
    positive_verbs2 = ["loved", "enjoyed", "recommend", "appreciate"]
    
    for _ in range(180):
        template = random.choice(positive_templates)
        text = template.format(
            verb=random.choice(positive_verbs),
            noun=random.choice(positive_nouns),
            positive_adj=random.choice(positive_adjs),
            positive_verb=random.choice(positive_verbs2),
            positive_emotion=random.choice(positive_emotions)
        )
        training_data.append({
            "text": text,
            "labels": {"neutral": 1, "stress": 0, "unsafe_environment": 0, 
                      "emotional_distress": 0, "self_harm_low": 0, "self_harm_high": 0}
        })
    
    # ========== NEUTRAL STATEMENTS (150 examples) ==========
    
    neutral_statements = [
        "I went to the store to buy groceries",
        "I have a meeting scheduled for 3pm",
        "I need to finish my work project",
        "I'm going to the gym later today",
        "I have an appointment tomorrow morning",
        "I need to call my doctor",
        "I'm planning to visit my family next week",
        "I have to complete my assignment",
        "I'm going to the library to study",
        "I need to pick up my dry cleaning",
        "I have a dentist appointment next month",
        "I'm going shopping this weekend",
        "I need to pay my bills",
        "I have a conference call at 2pm",
        "I'm going to the park for a walk",
    ]
    
    for text in neutral_statements:
        training_data.append({
            "text": text,
            "labels": {"neutral": 1, "stress": 0, "unsafe_environment": 0, 
                      "emotional_distress": 0, "self_harm_low": 0, "self_harm_high": 0}
        })
    
    # Generate variations
    neutral_templates = [
        "I {action} {location}",
        "I have {activity} {timeframe}",
        "I need to {activity}",
        "I'm going to {activity}",
        "I {action} {object}",
    ]
    
    actions = ["went to", "going to", "need to", "have to"]
    locations = ["the store", "work", "school", "the gym", "the library", "the park"]
    activities = ["a meeting", "an appointment", "a call", "a visit", "a conference"]
    timeframes = ["today", "tomorrow", "next week", "next month", "at 3pm", "at 2pm"]
    objects = ["groceries", "my project", "my assignment", "my bills"]
    
    for _ in range(135):
        template = random.choice(neutral_templates)
        text = template.format(
            action=random.choice(actions),
            location=random.choice(locations),
            activity=random.choice(activities),
            timeframe=random.choice(timeframes),
            object=random.choice(objects)
        )
        training_data.append({
            "text": text,
            "labels": {"neutral": 1, "stress": 0, "unsafe_environment": 0, 
                      "emotional_distress": 0, "self_harm_low": 0, "self_harm_high": 0}
        })
    
    # ========== STRESS STATEMENTS (150 examples) ==========
    
    stress_statements = [
        "I'm worried about my exam tomorrow",
        "I'm stressed about work deadlines",
        "I'm anxious about the presentation",
        "I'm nervous about the interview",
        "I'm feeling pressured by all these responsibilities",
        "I'm stressed about meeting my goals",
        "I'm worried about my financial situation",
        "I'm anxious about the upcoming changes",
        "I'm stressed about balancing everything",
        "I'm nervous about the future",
    ]
    
    for text in stress_statements:
        training_data.append({
            "text": text,
            "labels": {"neutral": 0, "stress": 1, "unsafe_environment": 0, 
                      "emotional_distress": 0, "self_harm_low": 0, "self_harm_high": 0}
        })
    
    # Generate variations
    stress_templates = [
        "I'm {stress_word} about {stress_source}",
        "I'm feeling {stress_word} by {stress_source}",
        "I'm {stress_word} about {stress_source} {timeframe}",
    ]
    
    stress_words = ["worried", "stressed", "anxious", "nervous", "pressured", "overwhelmed"]
    stress_sources = ["my exam", "work deadlines", "the presentation", "the interview", 
                     "all these responsibilities", "meeting my goals", "my financial situation",
                     "the upcoming changes", "balancing everything", "the future"]
    
    for _ in range(140):
        template = random.choice(stress_templates)
        text = template.format(
            stress_word=random.choice(stress_words),
            stress_source=random.choice(stress_sources),
            timeframe=random.choice(timeframes)
        )
        training_data.append({
            "text": text,
            "labels": {"neutral": 0, "stress": 1, "unsafe_environment": 0, 
                      "emotional_distress": 0, "self_harm_low": 0, "self_harm_high": 0}
        })
    
    # ========== EMOTIONAL DISTRESS (150 examples) ==========
    
    emotional_distress = [
        "I feel so overwhelmed and sad all the time",
        "I'm feeling really down and hopeless",
        "I feel empty and meaningless",
        "I'm so depressed and can't find joy in anything",
        "I feel like nothing matters anymore",
        "I'm feeling really low and don't know what to do",
        "I feel so alone and isolated",
        "I'm feeling really depressed and hopeless",
        "I feel like I'm drowning in sadness",
        "I'm so overwhelmed by negative emotions",
    ]
    
    for text in emotional_distress:
        training_data.append({
            "text": text,
            "labels": {"neutral": 0, "stress": 0, "unsafe_environment": 0, 
                      "emotional_distress": 1, "self_harm_low": 0, "self_harm_high": 0}
        })
    
    # Generate variations
    distress_templates = [
        "I feel {distress_word} and {distress_word2}",
        "I'm feeling {distress_word} {distress_context}",
        "I feel {distress_word} {distress_context}",
    ]
    
    distress_words = ["overwhelmed", "sad", "down", "hopeless", "empty", "meaningless", 
                     "depressed", "alone", "isolated", "drowning", "lost", "broken"]
    distress_contexts = ["all the time", "lately", "recently", "these days", "now"]
    
    for _ in range(140):
        template = random.choice(distress_templates)
        text = template.format(
            distress_word=random.choice(distress_words),
            distress_word2=random.choice(distress_words),
            distress_context=random.choice(distress_contexts)
        )
        training_data.append({
            "text": text,
            "labels": {"neutral": 0, "stress": 0, "unsafe_environment": 0, 
                      "emotional_distress": 1, "self_harm_low": 0, "self_harm_high": 0}
        })
    
    # ========== SELF-HARM LOW RISK (100 examples) ==========
    
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
    
    for text in self_harm_low:
        training_data.append({
            "text": text,
            "labels": {"neutral": 0, "stress": 0, "unsafe_environment": 0, 
                      "emotional_distress": 0, "self_harm_low": 1, "self_harm_high": 0}
        })
    
    # Generate variations
    self_harm_low_templates = [
        "I {frequency} think about {self_harm_thought}",
        "I wonder if {self_harm_wonder}",
        "I've been {self_harm_considering}",
        "I {frequency} have thoughts about {self_harm_thought}",
    ]
    
    frequencies = ["sometimes", "occasionally", "often", "frequently"]
    self_harm_thoughts = ["ending it all", "hurting myself", "suicide", "not being here", 
                         "ending my life", "killing myself"]
    self_harm_wonders = ["anyone would notice if I was gone", "people would be better off without me",
                        "anyone would care if I disappeared", "the world would be better without me"]
    self_harm_considering = ["having thoughts about ending it", "considering suicide", 
                            "thinking about hurting myself", "having suicidal thoughts"]
    
    for _ in range(90):
        template = random.choice(self_harm_low_templates)
        text = template.format(
            frequency=random.choice(frequencies),
            self_harm_thought=random.choice(self_harm_thoughts),
            self_harm_wonder=random.choice(self_harm_wonders),
            self_harm_considering=random.choice(self_harm_considering)
        )
        training_data.append({
            "text": text,
            "labels": {"neutral": 0, "stress": 0, "unsafe_environment": 0, 
                      "emotional_distress": 0, "self_harm_low": 1, "self_harm_high": 0}
        })
    
    # ========== SELF-HARM HIGH RISK (100 examples) ==========
    
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
    
    for text in self_harm_high:
        training_data.append({
            "text": text,
            "labels": {"neutral": 0, "stress": 0, "unsafe_environment": 0, 
                      "emotional_distress": 0, "self_harm_low": 0, "self_harm_high": 1}
        })
    
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
    
    for _ in range(90):
        template = random.choice(self_harm_high_templates)
        text = template.format(
            intent=random.choice(intents),
            self_harm_action=random.choice(self_harm_actions),
            timeframe=random.choice(["tonight", "today", "this week", "soon"]),
            planning=random.choice(planning),
            plan=random.choice(plans),
            self_harm_reason=random.choice(self_harm_reasons)
        )
        training_data.append({
            "text": text,
            "labels": {"neutral": 0, "stress": 0, "unsafe_environment": 0, 
                      "emotional_distress": 0, "self_harm_low": 0, "self_harm_high": 1}
        })
    
    # ========== UNSAFE ENVIRONMENT (100 examples) ==========
    
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
    
    for text in unsafe_environment:
        training_data.append({
            "text": text,
            "labels": {"neutral": 0, "stress": 0, "unsafe_environment": 1, 
                      "emotional_distress": 0, "self_harm_low": 0, "self_harm_high": 0}
        })
    
    # Generate variations
    unsafe_templates = [
        "I {intent} to {harm_action} {target}",
        "I'm going to {harm_action} {target} {reason}",
        "I will {harm_action} {target}",
    ]
    
    harm_actions = ["hurt", "kill", "harm", "attack", "get", "get revenge on"]
    targets = ["you", "them", "him", "her", "those people", "someone", "the people who hurt me",
              "those who wronged me", "the people responsible", "those who betrayed me"]
    reasons = ["for what you did", "for this", "for hurting me", "who wronged me"]
    
    for _ in range(90):
        template = random.choice(unsafe_templates)
        text = template.format(
            intent=random.choice(intents),
            harm_action=random.choice(harm_actions),
            target=random.choice(targets),
            reason=random.choice(reasons)
        )
        training_data.append({
            "text": text,
            "labels": {"neutral": 0, "stress": 0, "unsafe_environment": 1, 
                      "emotional_distress": 0, "self_harm_low": 0, "self_harm_high": 0}
        })
    
    # Shuffle the data
    random.shuffle(training_data)
    
    # Split into train and validation (80/20)
    split_idx = int(len(training_data) * 0.8)
    train_data = training_data[:split_idx]
    val_data = training_data[split_idx:]
    
    # Save to files
    with open('train_data.json', 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    
    with open('val_data.json', 'w', encoding='utf-8') as f:
        json.dump(val_data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Generated {len(train_data)} training examples")
    print(f"✓ Generated {len(val_data)} validation examples")
    print(f"✓ Total: {len(training_data)} examples")
    print("\nCategories:")
    print(f"  - Positive confident/empowered: ~180")
    print(f"  - Positive relationship: ~180")
    print(f"  - Frustration/annoyance: ~150")
    print(f"  - Positive statements: ~200")
    print(f"  - Neutral statements: ~150")
    print(f"  - Stress: ~150")
    print(f"  - Emotional distress: ~150")
    print(f"  - Self-harm low: ~100")
    print(f"  - Self-harm high: ~100")
    print(f"  - Unsafe environment: ~100")
    print("\n✓ Files saved: train_data.json, val_data.json")

if __name__ == '__main__':
    generate_comprehensive_fixed_data()

