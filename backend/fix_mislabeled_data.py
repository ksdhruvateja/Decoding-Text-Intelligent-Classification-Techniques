"""
AUTOMATIC DATA FIXER
Fixes all 18 mislabeled samples automatically
"""

import json
import re

def fix_mislabeled_samples():
    """Automatically fix the 18 critical mislabeling issues"""
    
    # Load training data
    with open('train_data.json', 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    # Patterns that indicate HIGH risk (not low)
    high_risk_patterns = [
        r'\bharm myself\b',
        r'\bhurt myself\b',
        r'\bend my pain\b',
        r'\bhaving urges to harm\b',
        r'\btempted to hurt\b',
        r'\bconsidering ways to harm\b',
        r'\bthinking about ways to hurt\b',
        r'\bthinking about ways to end\b',
        r'\bscared I might hurt myself\b',
        r'\bafraid I might hurt myself\b',
        r'\bworried I could hurt myself\b',
        r'\bscared I could hurt myself\b',
        r'\bafraid I could hurt myself\b'
    ]
    
    fixed_count = 0
    
    for sample in train_data:
        text = sample['text'].lower()
        
        # Check if sample has any high-risk patterns
        for pattern in high_risk_patterns:
            if re.search(pattern, text):
                # If it's marked as self_harm_low but should be high
                if sample['labels']['self_harm_low'] == 1:
                    sample['labels']['self_harm_low'] = 0
                    sample['labels']['self_harm_high'] = 1
                    fixed_count += 1
                    print(f"✓ Fixed: \"{sample['text']}\"")
                    print(f"  Changed: self_harm_low → self_harm_high")
                    break
    
    # Save fixed data
    with open('train_data.json', 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Fixed {fixed_count} mislabeled samples")
    print(f"✓ Updated train_data.json")
    
    return fixed_count

if __name__ == '__main__':
    print("="*80)
    print("AUTOMATIC DATA FIXER")
    print("="*80)
    
    fixed = fix_mislabeled_samples()
    
    print("\n" + "="*80)
    print(f"✓ Complete! Fixed {fixed} critical mislabeling issues")
    print("="*80)
