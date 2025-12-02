"""
Generate comprehensive, balanced training dataset with 5000+ examples
Covers all categories with diverse vocabulary and realistic scenarios
"""
import json
import random
from typing import List, Dict

def create_positive_examples(n=1000) -> List[Dict]:
    """Generate diverse positive examples"""
    templates = [
        # Happiness
        "I'm feeling {adj} today!", "Today is an {adj} day!", "I feel so {adj} right now!",
        "I'm really {adj} about {event}!", "Everything is going {adv}!", 
        "I couldn't be {comp} with how things are going!", "Life is {adj} right now!",
        "I'm {adj} to be alive!", "What a {adj} day this has been!",
        
        # Gratitude
        "I'm so grateful for {thing}!", "Thank you for {thing}!", "I really appreciate {thing}!",
        "I feel blessed to have {thing}!", "I'm thankful for {thing} every day!",
        "I love having {thing} in my life!", "I couldn't ask for {comp} {thing}!",
        
        # Achievement
        "I just {achievement} and I'm so proud!", "I finally {achievement}!", 
        "I achieved my goal of {achievement}!", "I'm proud of myself for {achievement}!",
        "I can't believe I {achievement}!", "I successfully {achievement}!",
        
        # Relationships
        "I love spending time with {person}!", "My {person} makes me so happy!",
        "I'm so lucky to have {person} in my life!", "I cherish my relationship with {person}!",
        "I had a wonderful time with {person} today!", "I feel supported by {person}!",
        
        # Excitement
        "I'm so excited for {future_event}!", "I can't wait for {future_event}!",
        "I'm really looking forward to {future_event}!", "I'm thrilled about {future_event}!",
    ]
    
    adjectives = ["happy", "grateful", "amazing", "wonderful", "fantastic", "great", "excellent", 
                  "joyful", "delighted", "thrilled", "blessed", "content", "peaceful", "excited", 
                  "proud", "satisfied", "cheerful", "optimistic", "hopeful", "confident"]
    
    adverbs = ["well", "perfectly", "smoothly", "wonderfully", "amazingly", "fantastically", "great"]
    
    comparatives = ["happier", "better", "more content", "more satisfied", "more grateful"]
    
    events = ["this promotion", "my new job", "graduating", "my wedding", "my new house",
              "this opportunity", "meeting new people", "learning new skills", "my progress",
              "these changes", "my recovery", "my growth", "this project"]
    
    things = ["my family", "my friends", "my health", "this opportunity", "my life", 
              "your support", "everything I have", "my job", "my home", "my pets",
              "good health", "my education", "this experience", "my loved ones"]
    
    achievements = ["got the job", "graduated", "passed my exam", "finished the project",
                   "reached my goal", "completed my degree", "got promoted", "made progress",
                   "overcame my fears", "learned something new", "improved my skills"]
    
    people = ["family", "friends", "partner", "children", "parents", "team", "colleagues",
              "loved ones", "support system"]
    
    future_events = ["the weekend", "my vacation", "tomorrow", "the future", "new opportunities",
                    "what's next", "this new chapter", "upcoming events", "my plans"]
    
    examples = []
    for _ in range(n):
        template = random.choice(templates)
        text = template.format(
            adj=random.choice(adjectives),
            adv=random.choice(adverbs),
            comp=random.choice(comparatives),
            event=random.choice(events),
            thing=random.choice(things),
            achievement=random.choice(achievements),
            person=random.choice(people),
            future_event=random.choice(future_events)
        )
        examples.append({
            "text": text,
            "labels": {"neutral": 1, "stress": 0, "unsafe_environment": 0, 
                      "emotional_distress": 0, "self_harm_low": 0, "self_harm_high": 0}
        })
    
    return examples

def create_neutral_examples(n=1000) -> List[Dict]:
    """Generate neutral everyday conversation examples"""
    templates = [
        # Daily activities
        "I {activity} today.", "I'm going to {activity}.", "I need to {activity}.",
        "I just finished {activity}.", "I'll {activity} later.", "I have to {activity} soon.",
        
        # Information/Questions
        "What time is {event}?", "Where is the {place}?", "How do I {action}?",
        "Can you help me with {task}?", "I'm looking for {item}.",
        "I need information about {topic}.", "What are the {aspect} of {thing}?",
        
        # Statements
        "The {item} is {location}.", "It's {weather} outside.", "I live in {place}.",
        "My name is {name}.", "I work in {field}.", "I study {subject}.",
        "Today is {day}.", "It's {time} right now.",
        
        # Plans
        "I have a meeting at {time}.", "I'm meeting {person} for {activity}.",
        "My appointment is at {time}.", "I'll be there at {time}.",
    ]
    
    activities = ["went to the store", "did laundry", "cooked dinner", "cleaned the house",
                 "walked the dog", "went to work", "attended a meeting", "read a book",
                 "watched TV", "went for a walk", "did groceries", "checked my email",
                 "made breakfast", "took a shower", "went to the gym"]
    
    events = ["the meeting", "the appointment", "the event", "the class", "dinner"]
    places = ["library", "store", "office", "bathroom", "kitchen", "parking lot"]
    actions = ["get there", "do this", "start", "find it", "proceed", "register"]
    tasks = ["this", "the project", "my assignment", "the paperwork"]
    items = ["keys", "phone", "book", "document", "information", "address"]
    topics = ["schedules", "policies", "procedures", "requirements", "options"]
    
    examples = []
    for _ in range(n):
        template = random.choice(templates)
        text = template.format(
            activity=random.choice(activities),
            event=random.choice(events),
            place=random.choice(places),
            action=random.choice(actions),
            task=random.choice(tasks),
            item=random.choice(items),
            topic=random.choice(topics),
            location="here",
            weather=random.choice(["sunny", "cloudy", "rainy", "cold", "warm"]),
            name="Alex",
            field=random.choice(["tech", "education", "healthcare", "retail"]),
            subject=random.choice(["computer science", "business", "engineering"]),
            day=random.choice(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]),
            time=random.choice(["3pm", "morning", "noon", "evening"]),
            person="a friend",
            aspect="advantages and disadvantages",
            thing="this"
        )
        examples.append({
            "text": text,
            "labels": {"neutral": 1, "stress": 0, "unsafe_environment": 0,
                      "emotional_distress": 0, "self_harm_low": 0, "self_harm_high": 0}
        })
    
    return examples

def create_stress_examples(n=800) -> List[Dict]:
    """Generate mild stress/pressure examples"""
    templates = [
        "I'm {adv} worried about {stressor}.", "I feel {adj} about {stressor}.",
        "I'm {adj} with {stressor}.", "{stressor} is really {adj} me.",
        "I can't stop thinking about {stressor}.", "I'm nervous about {stressor}.",
        "I'm having trouble with {stressor}.", "{stressor} has been {adj}.",
        "I feel {adj} because of {stressor}.", "I'm struggling with {stressor}.",
        "I'm finding {stressor} {adj}.", "{stressor} is causing me some {noun}.",
    ]
    
    adverbs = ["a bit", "somewhat", "really", "very", "quite", "pretty", "kinda"]
    adjectives = ["stressed", "overwhelmed", "anxious", "busy", "worried", "concerned",
                 "nervous", "tense", "pressured", "difficult", "challenging", "tough"]
    
    stressors = ["my exam", "work", "my deadline", "this project", "my presentation",
                "my workload", "my schedule", "finances", "my bills", "this assignment",
                "my responsibilities", "time management", "my performance", "my job",
                "this situation", "balancing everything", "meeting expectations"]
    
    nouns = ["stress", "anxiety", "pressure", "worry", "concern", "tension"]
    
    examples = []
    for _ in range(n):
        template = random.choice(templates)
        text = template.format(
            adv=random.choice(adverbs),
            adj=random.choice(adjectives),
            stressor=random.choice(stressors),
            noun=random.choice(nouns)
        )
        examples.append({
            "text": text,
            "labels": {"neutral": 0, "stress": 1, "unsafe_environment": 0,
                      "emotional_distress": 0, "self_harm_low": 0, "self_harm_high": 0}
        })
    
    return examples

def create_emotional_distress_examples(n=600) -> List[Dict]:
    """Generate emotional distress examples"""
    templates = [
        "I feel so {emotion} all the time.", "I'm constantly feeling {emotion}.",
        "Everything feels {adj}.", "I can't seem to feel {positive}.",
        "I feel {emotion} and {emotion2}.", "Nothing makes me feel {positive} anymore.",
        "I'm struggling with {issue}.", "I feel like {negative_thought}.",
        "I'm having a really hard time with {issue}.", "I can't stop feeling {emotion}.",
        "I feel {emotion} every day.", "I'm overwhelmed by {emotion}.",
        "Life feels {adj} right now.", "I don't know how to cope with {issue}.",
    ]
    
    emotions = ["sad", "empty", "numb", "hopeless", "worthless", "alone", "isolated",
               "depressed", "miserable", "broken", "lost", "defeated", "exhausted"]
    
    adjectives = ["meaningless", "pointless", "empty", "overwhelming", "unbearable",
                 "impossible", "hopeless", "dark", "heavy"]
    
    positive = ["happy", "joy", "hope", "motivation", "pleasure", "better"]
    
    issues = ["depression", "loneliness", "my emotions", "these feelings", "sadness",
             "feeling empty", "my mental health", "feeling worthless"]
    
    negative_thoughts = ["nothing matters", "I'm worthless", "I'm a failure",
                        "I can't do anything right", "I'm alone", "nobody cares",
                        "I'm not good enough", "everything is falling apart"]
    
    examples = []
    for _ in range(n):
        template = random.choice(templates)
        text = template.format(
            emotion=random.choice(emotions),
            emotion2=random.choice(emotions),
            adj=random.choice(adjectives),
            positive=random.choice(positive),
            issue=random.choice(issues),
            negative_thought=random.choice(negative_thoughts)
        )
        examples.append({
            "text": text,
            "labels": {"neutral": 0, "stress": 0, "unsafe_environment": 0,
                      "emotional_distress": 1, "self_harm_low": 0, "self_harm_high": 0}
        })
    
    return examples

def create_self_harm_low_examples(n=400) -> List[Dict]:
    """Generate low-level self-harm concern examples"""
    templates = [
        "I sometimes think about {thought}.", "I've been having thoughts about {thought}.",
        "I wonder if {question}.", "Sometimes I wish {wish}.",
        "I've thought about {action} but I wouldn't do it.", "I have {intensity} thoughts about {thought}.",
        "I feel like I want to {mild_action}.", "I'm thinking about {thought} sometimes.",
        "I've wondered what it would be like to {thought}.", "I wish I could just {wish}.",
    ]
    
    thoughts = ["not being here", "disappearing", "giving up", "escaping from everything",
               "not waking up", "being somewhere else", "starting over"]
    
    questions = ["things would be better without me", "anyone would notice if I was gone",
                "life is worth living", "I matter to anyone"]
    
    wishes = ["I could disappear", "I didn't exist", "I could escape", "it would all stop",
             "I didn't have to deal with this", "I could just sleep forever"]
    
    intensity = ["occasional", "passing", "fleeting", "sometimes", "intrusive"]
    
    mild_actions = ["disappear", "run away", "give up", "escape"]
    
    examples = []
    for _ in range(n):
        template = random.choice(templates)
        text = template.format(
            thought=random.choice(thoughts),
            question=random.choice(questions),
            wish=random.choice(wishes),
            intensity=random.choice(intensity),
            mild_action=random.choice(mild_actions),
            action="ending it"
        )
        examples.append({
            "text": text,
            "labels": {"neutral": 0, "stress": 0, "unsafe_environment": 0,
                      "emotional_distress": 0, "self_harm_low": 1, "self_harm_high": 0}
        })
    
    return examples

def create_self_harm_high_examples(n=200) -> List[Dict]:
    """Generate high-risk self-harm examples (CRITICAL DETECTION)"""
    critical_phrases = [
        "I want to hurt myself", "I want to end my life", "I'm going to kill myself",
        "I want to die", "I'm planning to hurt myself", "I can't take this anymore",
        "I want to end it all", "I'm going to end my pain", "I don't want to live anymore",
        "I'm thinking about suicide", "I have a plan to hurt myself", 
        "I want to harm myself", "I'm going to hurt myself", "I need to end this pain",
        "I'm planning to end my life", "I'm considering suicide", "I want to kill myself",
        "I'm going to take my own life", "I have thoughts of ending it",
        "I'm actively planning to hurt myself", "I bought {method} to end my life",
        "I'm going to {method} tonight", "I can't live like this anymore",
        "I'd rather be dead", "Death would be better", "I wish I was dead",
        "I want to end my suffering", "I'm done with life", "Life isn't worth living",
        "I'm going to end this tonight", "I have pills to overdose",
        "I want to cut myself", "I want to jump", "I'm going to hang myself",
        "I plan to {method} soon", "I'm ready to end it", "Tonight is the night",
        "I've decided to end my life", "I'm going to do it", "I can't go on",
        "I want the pain to stop forever", "I'm serious about ending it",
        "I have everything ready to end my life", "I'm going to hurt myself badly",
        "I need to hurt myself", "I deserve to die", "The world is better without me",
        "I'm a burden and should die", "I'm going to end this pain permanently"
    ]
    
    methods = ["pills", "a knife", "rope", "a gun"]
    
    examples = []
    for phrase in critical_phrases[:n]:
        text = phrase.format(method=random.choice(methods)) if "{method}" in phrase else phrase
        examples.append({
            "text": text,
            "labels": {"neutral": 0, "stress": 0, "unsafe_environment": 0,
                      "emotional_distress": 0, "self_harm_low": 0, "self_harm_high": 1}
        })
    
    return examples

def create_unsafe_environment_examples(n=400) -> List[Dict]:
    """Generate unsafe environment examples"""
    templates = [
        "I don't feel safe at {place}.", "I'm scared of {person}.",
        "{person} is {action} me.", "I'm being {action} by {person}.",
        "I feel threatened by {person}.", "I'm afraid to go {place}.",
        "{person} is making me feel unsafe.", "I'm in danger from {person}.",
        "I'm being {action} and I don't know what to do.", "I need help, {person} is {action} me.",
    ]
    
    places = ["home", "work", "school", "here", "my house"]
    people = ["someone", "my partner", "my parent", "someone at work", "a family member"]
    actions = ["threatening", "hurting", "scaring", "harassing", "following", 
              "intimidating", "abusing", "stalking", "attacking"]
    
    examples = []
    for _ in range(n):
        template = random.choice(templates)
        text = template.format(
            place=random.choice(places),
            person=random.choice(people),
            action=random.choice(actions)
        )
        examples.append({
            "text": text,
            "labels": {"neutral": 0, "stress": 0, "unsafe_environment": 1,
                      "emotional_distress": 0, "self_harm_low": 0, "self_harm_high": 0}
        })
    
    return examples

# Generate all examples
print("Generating comprehensive training dataset...")
print("=" * 80)

positive = create_positive_examples(1000)
neutral = create_neutral_examples(1000)
stress = create_stress_examples(800)
distress = create_emotional_distress_examples(600)
self_harm_low = create_self_harm_low_examples(400)
self_harm_high = create_self_harm_high_examples(200)
unsafe = create_unsafe_environment_examples(400)

all_examples = positive + neutral + stress + distress + self_harm_low + self_harm_high + unsafe
random.shuffle(all_examples)

print(f"✓ Generated {len(all_examples)} training examples:")
print(f"  - Positive: {len(positive)}")
print(f"  - Neutral: {len(neutral)}")
print(f"  - Stress: {len(stress)}")
print(f"  - Emotional Distress: {len(distress)}")
print(f"  - Self-Harm Low: {len(self_harm_low)}")
print(f"  - Self-Harm High: {len(self_harm_high)}")
print(f"  - Unsafe Environment: {len(unsafe)}")

# Split into train/val (85/15 split)
split_idx = int(len(all_examples) * 0.85)
train_data = all_examples[:split_idx]
val_data = all_examples[split_idx:]

# Save datasets
with open('training_data_comprehensive.json', 'w') as f:
    json.dump(train_data, f, indent=2)

with open('val_data_comprehensive.json', 'w') as f:
    json.dump(val_data, f, indent=2)

print(f"\n✓ Saved {len(train_data)} training examples to training_data_comprehensive.json")
print(f"✓ Saved {len(val_data)} validation examples to val_data_comprehensive.json")
print("=" * 80)
