"""
Generate High-Quality Threat/Violence Training Data
===================================================
Adds explicit violent threat examples to balance the dataset
"""

import json
from pathlib import Path

# Explicit violent threat examples
threat_examples = [
    # Direct threats to kill
    "I will kill him",
    "I'm going to kill you",
    "I'll kill them all",
    "You're dead",
    "I'm gonna murder you",
    "Time to kill someone",
    "Death is coming for you",
    "Prepare to die",
    "I'll end your life",
    "You won't survive this",
    
    # Physical violence threats
    "I'll beat you to death",
    "I'm going to hurt you badly",
    "I will break your bones",
    "You'll be bleeding soon",
    "I'm going to smash your face",
    "I'll destroy you physically",
    "Violence is the only answer",
    "Time to teach you a painful lesson",
    "I'll make you suffer",
    "Pain is coming your way",
    
    # Weapon-related threats
    "I have a knife and I'll use it",
    "My gun is loaded and ready",
    "Bringing a weapon tomorrow",
    "The knife is sharp enough",
    "I know where the guns are",
    "Armed and dangerous",
    "Got my weapon ready",
    "Loaded the gun this morning",
    
    # Attack planning
    "Planning an attack tomorrow",
    "I'm going to attack the school",
    "Time to execute my plan",
    "Attack is scheduled for Monday",
    "Ready to strike",
    "The assault begins tonight",
    "Going on a rampage",
    "Mass shooting planned",
    
    # Specific target threats
    "I'll hurt your family",
    "Your kids are targets",
    "I know where you live and I'm coming",
    "Everyone at that office will pay",
    "Your friends are next",
    "I'm coming for everyone there",
    "No one in that building is safe",
    "They won't see me coming",
    
    # Bomb/terrorism threats
    "I'm going to blow up the building",
    "Bomb is ready to plant",
    "Explosion coming soon",
    "I'll burn it all down",
    "The bomb goes off tomorrow",
    "Fire and destruction ahead",
    
    # Group violence
    "We're going to attack them",
    "My crew will destroy you",
    "Gang violence tonight",
    "Mob is coming for you",
    "Group assault planned",
    
    # Implied immediate danger
    "Right behind you with a knife",
    "In your house right now",
    "Watching you through the window",
    "I'm outside your door",
    "Following you home",
    "Stalking you tonight",
    
    # Revenge threats
    "You'll pay for this with blood",
    "Revenge will be violent",
    "I'll get you back painfully",
    "Payback involves suffering",
    "You hurt me, I'll destroy you",
    
    # Casual violent statements
    "Beat him up after school",
    "Gonna jump that guy",
    "Fight club tonight, someone's dying",
    "Street fight, bring weapons",
    "Brawl is going to be bloody",
    
    # Online threats
    "I'll find you IRL and hurt you",
    "Swatting your address tonight",
    "Doxxing then attacking",
    "Real life consequences coming",
    
    # Specific violent scenarios
    "Bringing a gun to school tomorrow",
    "Shooting up the mall",
    "Running people over with my car",
    "Poisoning the water supply",
    "Arson at the office tonight",
]

# Load existing data
data_path = Path(__file__).parent / "clean_training_data.json"
with open(data_path, 'r', encoding='utf-8') as f:
    existing_data = json.load(f)

print(f"Existing dataset: {len(existing_data)} examples")

# Add new threat examples
new_examples = []
for text in threat_examples:
    new_examples.append({
        "text": text,
        "category": "unsafe_environment"
    })

print(f"Adding {len(new_examples)} violent threat examples")

# Combine and save
updated_data = existing_data + new_examples

output_path = Path(__file__).parent / "balanced_training_data.json"
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(updated_data, f, indent=2, ensure_ascii=False)

print(f"\n✅ Saved {len(updated_data)} examples to: {output_path}")

# Show category distribution
from collections import Counter
distribution = Counter([item['category'] for item in updated_data])
print("\nNew category distribution:")
for category, count in distribution.most_common():
    print(f"  {category}: {count}")
