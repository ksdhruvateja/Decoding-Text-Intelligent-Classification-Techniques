"""
COMPREHENSIVE DATA AUDIT AND CLEANING TOOL
Analyzes training data for the issues you identified:
- Label imbalance
- Poor diversity
- Unclear class definitions
- Label overlap
- Mislabeled samples
"""

import json
import numpy as np
from collections import Counter, defaultdict
import re

def load_data():
    """Load training and validation data"""
    with open('train_data.json', 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open('val_data.json', 'r', encoding='utf-8') as f:
        val_data = json.load(f)
    return train_data, val_data

def analyze_label_overlap(data):
    """Analyze how often labels co-occur (identifies overlap issues)"""
    label_names = ['neutral', 'stress', 'unsafe_environment', 
                   'emotional_distress', 'self_harm_low', 'self_harm_high']
    
    # Co-occurrence matrix
    cooccurrence = defaultdict(lambda: defaultdict(int))
    
    for sample in data:
        active_labels = [label for label in label_names if sample['labels'][label] == 1]
        
        # Count co-occurrences
        for i, label1 in enumerate(active_labels):
            for label2 in active_labels[i+1:]:
                cooccurrence[label1][label2] += 1
                cooccurrence[label2][label1] += 1
    
    print("\n" + "="*80)
    print("LABEL CO-OCCURRENCE ANALYSIS (identifying overlaps)")
    print("="*80)
    
    print("\nCo-occurrence matrix:")
    print(f"{'':25s}", end='')
    for label in label_names:
        print(f"{label[:15]:>15s}", end='')
    print()
    print('-'*80)
    
    for label1 in label_names:
        print(f"{label1:25s}", end='')
        for label2 in label_names:
            if label1 == label2:
                print(f"{'--':>15s}", end='')
            else:
                count = cooccurrence[label1][label2]
                print(f"{count:>15d}", end='')
        print()
    
    # Identify problematic overlaps
    print("\n❌ PROBLEMATIC OVERLAPS (co-occur frequently):")
    for label1 in label_names:
        for label2 in label_names:
            if label1 < label2:  # Avoid duplicates
                count = cooccurrence[label1][label2]
                if count > len(data) * 0.1:  # More than 10% co-occurrence
                    print(f"  • {label1} + {label2}: {count} times ({count/len(data)*100:.1f}%)")
                    print(f"    → These categories may be too similar or definitions unclear")

def analyze_text_diversity(data):
    """Analyze text diversity and identify potential issues"""
    print("\n" + "="*80)
    print("TEXT DIVERSITY ANALYSIS")
    print("="*80)
    
    texts = [sample['text'] for sample in data]
    
    # Length analysis
    lengths = [len(text.split()) for text in texts]
    print(f"\nText length statistics:")
    print(f"  Min: {min(lengths)} words")
    print(f"  Max: {max(lengths)} words")
    print(f"  Mean: {np.mean(lengths):.1f} words")
    print(f"  Median: {np.median(lengths):.1f} words")
    
    if np.mean(lengths) < 8:
        print(f"  ❌ ISSUE: Very short texts (avg {np.mean(lengths):.1f} words)")
        print(f"     → Need more context for accurate classification")
    
    # Vocabulary diversity
    all_words = ' '.join(texts).lower().split()
    unique_words = set(all_words)
    vocab_size = len(unique_words)
    vocab_ratio = vocab_size / len(all_words)
    
    print(f"\nVocabulary diversity:")
    print(f"  Total words: {len(all_words)}")
    print(f"  Unique words: {vocab_size}")
    print(f"  Diversity ratio: {vocab_ratio:.3f}")
    
    if vocab_ratio < 0.3:
        print(f"  ❌ ISSUE: Low vocabulary diversity")
        print(f"     → Many repetitive phrases, need more varied examples")
    
    # Check for duplicates
    text_counts = Counter(texts)
    duplicates = [(text, count) for text, count in text_counts.items() if count > 1]
    
    if duplicates:
        print(f"\n❌ DUPLICATE TEXTS FOUND: {len(duplicates)}")
        print(f"  Examples:")
        for text, count in duplicates[:5]:
            print(f"    • \"{text}\" appears {count} times")
        print(f"     → Remove duplicates to improve generalization")

def analyze_per_class_diversity(data):
    """Analyze diversity within each class"""
    label_names = ['neutral', 'stress', 'unsafe_environment', 
                   'emotional_distress', 'self_harm_low', 'self_harm_high']
    
    print("\n" + "="*80)
    print("PER-CLASS DIVERSITY ANALYSIS")
    print("="*80)
    
    for label in label_names:
        positive_samples = [sample for sample in data if sample['labels'][label] == 1]
        
        if not positive_samples:
            continue
        
        texts = [s['text'] for s in positive_samples]
        
        # Word diversity
        all_words = ' '.join(texts).lower().split()
        unique_words = set(all_words)
        
        # Common patterns
        word_freq = Counter(all_words)
        top_words = word_freq.most_common(10)
        
        print(f"\n{label} ({len(positive_samples)} samples):")
        print(f"  Unique words: {len(unique_words)}")
        print(f"  Top words: {', '.join([w for w, c in top_words[:5]])}")
        
        # Check for over-reliance on specific words
        most_common_word_freq = top_words[0][1] / len(all_words) if top_words else 0
        if most_common_word_freq > 0.15:
            print(f"  ⚠️  WARNING: Top word '{top_words[0][0]}' appears {most_common_word_freq*100:.1f}% of the time")
            print(f"     → Class may be over-reliant on specific keywords")

def identify_mislabeled_candidates(data):
    """Find potential mislabeling issues based on text patterns"""
    label_names = ['neutral', 'stress', 'unsafe_environment', 
                   'emotional_distress', 'self_harm_low', 'self_harm_high']
    
    print("\n" + "="*80)
    print("POTENTIAL MISLABELING DETECTION")
    print("="*80)
    
    # Positive indicators that might be labeled as negative
    positive_patterns = [
        r'\b(great|wonderful|amazing|happy|joyful|excellent|fantastic|love|blessed)\b',
        r'\b(excited|thrilled|delighted|grateful|thankful)\b'
    ]
    
    # Severe risk indicators
    severe_patterns = [
        r'\b(kill myself|end (my|it all)|suicide|take my life)\b',
        r'\b(want to die|wish I was dead|better off dead)\b',
        r'\b(harm myself|hurt myself|cut myself)\b'
    ]
    
    # Neutral indicators
    neutral_patterns = [
        r'\b(normal day|regular|ordinary|fine|okay|alright)\b',
        r'\b(weather|coffee|work schedule|meeting)\b'
    ]
    
    issues = []
    
    for idx, sample in enumerate(data):
        text = sample['text'].lower()
        labels = sample['labels']
        
        # Check 1: Positive text marked as distress/harm
        for pattern in positive_patterns:
            if re.search(pattern, text):
                if labels['emotional_distress'] == 1 or labels['self_harm_low'] == 1 or labels['self_harm_high'] == 1:
                    issues.append({
                        'index': idx,
                        'text': sample['text'],
                        'issue': 'Positive text marked as distressed',
                        'severity': 'HIGH'
                    })
                    break
        
        # Check 2: Severe patterns marked as low risk
        for pattern in severe_patterns:
            if re.search(pattern, text):
                if labels['self_harm_high'] == 0:
                    issues.append({
                        'index': idx,
                        'text': sample['text'],
                        'issue': 'Severe risk text not marked as self_harm_high',
                        'severity': 'CRITICAL'
                    })
                    break
        
        # Check 3: Neutral text marked as risk
        for pattern in neutral_patterns:
            if re.search(pattern, text):
                if labels['self_harm_low'] == 1 or labels['self_harm_high'] == 1:
                    issues.append({
                        'index': idx,
                        'text': sample['text'],
                        'issue': 'Neutral text marked as self-harm risk',
                        'severity': 'HIGH'
                    })
                    break
    
    if issues:
        print(f"\n❌ FOUND {len(issues)} POTENTIAL MISLABELING ISSUES:")
        print(f"\nShowing first 15 issues:")
        for i, issue in enumerate(issues[:15]):
            print(f"\n{i+1}. [{issue['severity']}] Sample #{issue['index']}")
            print(f"   Text: \"{issue['text']}\"")
            print(f"   Issue: {issue['issue']}")
    else:
        print("\n✓ No obvious mislabeling issues detected")
    
    return issues

def analyze_class_separability(data):
    """Analyze how well classes can be separated"""
    label_names = ['neutral', 'stress', 'unsafe_environment', 
                   'emotional_distress', 'self_harm_low', 'self_harm_high']
    
    print("\n" + "="*80)
    print("CLASS SEPARABILITY ANALYSIS")
    print("="*80)
    
    # Check for ambiguous samples (multiple risk categories)
    ambiguous_samples = []
    
    for sample in data:
        labels = sample['labels']
        
        # Count how many risk categories are active
        risk_categories = ['stress', 'unsafe_environment', 'emotional_distress', 
                          'self_harm_low', 'self_harm_high']
        active_risks = sum(labels[cat] for cat in risk_categories)
        
        if active_risks > 1:
            ambiguous_samples.append({
                'text': sample['text'],
                'active': [cat for cat in risk_categories if labels[cat] == 1]
            })
    
    if ambiguous_samples:
        ambig_ratio = len(ambiguous_samples) / len(data)
        print(f"\n⚠️  AMBIGUOUS SAMPLES: {len(ambiguous_samples)} ({ambig_ratio*100:.1f}%)")
        print(f"   These samples have multiple risk categories active")
        print(f"\n   Examples:")
        for sample in ambiguous_samples[:5]:
            print(f"   • \"{sample['text']}\"")
            print(f"     Categories: {', '.join(sample['active'])}")
        
        if ambig_ratio > 0.2:
            print(f"\n   ❌ ISSUE: Too many ambiguous samples ({ambig_ratio*100:.1f}%)")
            print(f"      → Classes may be poorly defined or overlapping")

def generate_cleaning_recommendations(data, issues):
    """Generate specific recommendations for data cleaning"""
    print("\n" + "="*80)
    print("DATA CLEANING RECOMMENDATIONS")
    print("="*80)
    
    recommendations = []
    
    # 1. Label balance
    label_names = ['neutral', 'stress', 'unsafe_environment', 
                   'emotional_distress', 'self_harm_low', 'self_harm_high']
    counts = {label: sum(1 for d in data if d['labels'][label] == 1) for label in label_names}
    
    min_count = min(counts.values())
    max_count = max(counts.values())
    
    if max_count / min_count > 1.5:
        recommendations.append({
            'priority': 'HIGH',
            'issue': 'Class imbalance',
            'recommendation': f'Collect more samples for underrepresented classes: {[k for k, v in counts.items() if v < len(data)*0.18]}'
        })
    
    # 2. Mislabeling
    if issues:
        recommendations.append({
            'priority': 'CRITICAL',
            'issue': f'{len(issues)} potential mislabeled samples',
            'recommendation': 'Review and relabel the samples listed in MISLABELING DETECTION section'
        })
    
    # 3. Sample diversity
    if len(data) < 1000:
        recommendations.append({
            'priority': 'HIGH',
            'issue': f'Small dataset ({len(data)} samples)',
            'recommendation': 'Collect at least 1000 samples per class for robust training'
        })
    
    # 4. Text length
    avg_length = np.mean([len(d['text'].split()) for d in data])
    if avg_length < 8:
        recommendations.append({
            'priority': 'MEDIUM',
            'issue': f'Very short texts (avg {avg_length:.1f} words)',
            'recommendation': 'Add more context to samples (target 10-30 words per sample)'
        })
    
    # Print recommendations
    print("\n📋 PRIORITY ACTIONS:")
    
    for i, rec in enumerate(sorted(recommendations, key=lambda x: {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2}[x['priority']])):
        print(f"\n{i+1}. [{rec['priority']}] {rec['issue']}")
        print(f"   → {rec['recommendation']}")

def main():
    print("="*80)
    print(" COMPREHENSIVE DATA AUDIT")
    print(" Identifying: label overlap, poor diversity, mislabeling, class separability")
    print("="*80)
    
    # Load data
    train_data, val_data = load_data()
    print(f"\nLoaded {len(train_data)} training samples, {len(val_data)} validation samples")
    
    # Run all analyses
    analyze_label_overlap(train_data)
    analyze_text_diversity(train_data)
    analyze_per_class_diversity(train_data)
    issues = identify_mislabeled_candidates(train_data)
    analyze_class_separability(train_data)
    generate_cleaning_recommendations(train_data, issues)
    
    # Save issues to file for review
    if issues:
        with open('potential_mislabeling_issues.json', 'w', encoding='utf-8') as f:
            json.dump(issues, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Potential mislabeling issues saved to: potential_mislabeling_issues.json")
    
    print("\n" + "="*80)
    print("✓ Audit complete!")
    print("="*80)

if __name__ == '__main__':
    main()
