"""
Generate Training Data for Positive Motivational Statements
===========================================================
Generates examples of positive motivational statements that should be classified as neutral,
NOT stress or self-harm.
"""

def generate_positive_motivational_data():
    """Generate training data for positive motivational cases"""
    
    # Positive motivational statements (should be neutral, low distress)
    motivational_data = [
        ("I will ace this test", "neutral", "safe"),
        ("I will ace this presentation", "neutral", "safe"),
        ("I will ace this exam", "neutral", "safe"),
        ("I will ace this interview", "neutral", "safe"),
        ("I will crush this project", "neutral", "safe"),
        ("I will nail this assignment", "neutral", "safe"),
        ("I will kill it in this meeting", "neutral", "safe"),
        ("I will rock this presentation", "neutral", "safe"),
        ("I will excel at this job", "neutral", "safe"),
        ("I will succeed in this challenge", "neutral", "safe"),
        ("I will win this competition", "neutral", "safe"),
        ("I will dominate this game", "neutral", "safe"),
        ("I am going to ace this test", "neutral", "safe"),
        ("I am going to crush this project", "neutral", "safe"),
        ("I am going to nail this interview", "neutral", "safe"),
        ("I am going to excel at this", "neutral", "safe"),
        ("I am going to succeed", "neutral", "safe"),
        ("I am going to win", "neutral", "safe"),
        ("I will do great on this test", "neutral", "safe"),
        ("I will be amazing at this", "neutral", "safe"),
        ("I will be excellent in this role", "neutral", "safe"),
        ("I will be fantastic at this job", "neutral", "safe"),
        ("I will be outstanding", "neutral", "safe"),
        ("I will make it happen", "neutral", "safe"),
        ("I will achieve my goal", "neutral", "safe"),
        ("I will get this done", "neutral", "safe"),
        ("I will earn this position", "neutral", "safe"),
        ("I will win this", "neutral", "safe"),
        ("I will show them what I can do", "neutral", "safe"),
        ("I will prove myself", "neutral", "safe"),
        ("I will overcome this challenge", "neutral", "safe"),
        ("I will beat this obstacle", "neutral", "safe"),
        ("I will conquer this task", "neutral", "safe"),
        ("I will master this skill", "neutral", "safe"),
        ("I will handle this situation", "neutral", "safe"),
        ("I will tackle this problem", "neutral", "safe"),
        ("I will thrive in this environment", "neutral", "safe"),
        ("I will flourish in this role", "neutral", "safe"),
        ("I will prosper in this position", "neutral", "safe"),
        ("I will be better tomorrow", "neutral", "safe"),
        ("I will become stronger", "neutral", "safe"),
        ("I will become smarter", "neutral", "safe"),
        ("I will give it my all", "neutral", "safe"),
        ("I will put my best foot forward", "neutral", "safe"),
        ("I will give it 100%", "neutral", "safe"),
        ("I will do my best", "neutral", "safe"),
    ]
    
    return motivational_data


if __name__ == '__main__':
    data = generate_positive_motivational_data()
    print(f"Generated {len(data)} positive motivational training examples")
    print("\nSample examples:")
    for i, (text, emotion, sentiment) in enumerate(data[:5], 1):
        print(f"{i}. {text}")
        print(f"   -> {emotion} ({sentiment})")
        print()

