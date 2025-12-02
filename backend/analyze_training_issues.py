"""
Quick Training Issues Analysis
Run this to see what problems exist in your current training setup
"""

import json
import numpy as np
import torch
import os

print("="*80)
print("TRAINING ISSUES ANALYSIS")
print("="*80)

# 1. Check data balance
print("\n1. CLASS IMBALANCE CHECK")
print("-"*80)

with open('train_data.json', 'r') as f:
    train_data = json.load(f)

with open('val_data.json', 'r') as f:
    val_data = json.load(f)

labels = ['neutral', 'stress', 'unsafe_environment', 'emotional_distress', 'self_harm_low', 'self_harm_high']

print(f"Training samples: {len(train_data)}")
print(f"Validation samples: {len(val_data)}")
print("\nLabel distribution in training set:")

pos_counts = []
for label in labels:
    count = sum(1 for d in train_data if d['labels'][label] == 1)
    percentage = count / len(train_data) * 100
    pos_counts.append(count)
    print(f"  {label:25s}: {count:4d} ({percentage:5.1f}%)")

# Check if imbalanced
max_count = max(pos_counts)
min_count = min(pos_counts)
imbalance_ratio = max_count / min_count

print(f"\nImbalance ratio: {imbalance_ratio:.2f}x")
if imbalance_ratio > 1.5:
    print("  ❌ ISSUE: Significant class imbalance detected!")
    print("  → FIX: Use class weights in loss function")
else:
    print("  ✓ Classes are reasonably balanced")

# 2. Check dataset size
print("\n2. DATASET SIZE CHECK")
print("-"*80)

if len(train_data) < 500:
    print(f"  ❌ ISSUE: Small training set ({len(train_data)} samples)")
    print("  → FIX: Use data augmentation or collect more data")
elif len(train_data) < 1000:
    print(f"  ⚠️  WARNING: Moderate training set ({len(train_data)} samples)")
    print("  → RECOMMENDATION: Use data augmentation")
else:
    print(f"  ✓ Good training set size ({len(train_data)} samples)")

if len(val_data) < 50:
    print(f"  ❌ ISSUE: Small validation set ({len(val_data)} samples)")
    print("  → FIX: Metrics may be unreliable")
elif len(val_data) < 100:
    print(f"  ⚠️  WARNING: Small validation set ({len(val_data)} samples)")
else:
    print(f"  ✓ Good validation set size ({len(val_data)} samples)")

# 3. Check for data quality issues
print("\n3. DATA QUALITY CHECK")
print("-"*80)

empty_texts = sum(1 for d in train_data if not d['text'].strip())
short_texts = sum(1 for d in train_data if len(d['text'].split()) < 3)
very_long_texts = sum(1 for d in train_data if len(d['text'].split()) > 200)

if empty_texts > 0:
    print(f"  ❌ ISSUE: {empty_texts} empty texts found")
    print("  → FIX: Remove or handle empty texts")
else:
    print(f"  ✓ No empty texts")

if short_texts > 0:
    pct = short_texts / len(train_data) * 100
    print(f"  ⚠️  WARNING: {short_texts} ({pct:.1f}%) very short texts (< 3 words)")
    print("  → RECOMMENDATION: Review short texts for quality")
else:
    print(f"  ✓ No very short texts")

if very_long_texts > 0:
    pct = very_long_texts / len(train_data) * 100
    print(f"  ⚠️  INFO: {very_long_texts} ({pct:.1f}%) long texts (> 200 words)")
    print("  → NOTE: These will be truncated to 128 tokens")

# Check average text length
avg_length = np.mean([len(d['text'].split()) for d in train_data])
print(f"\n  Average text length: {avg_length:.1f} words")

# 4. Check for multi-label issues
print("\n4. MULTI-LABEL DISTRIBUTION")
print("-"*80)

labels_per_sample = []
for d in train_data:
    count = sum(d['labels'][label] for label in labels)
    labels_per_sample.append(count)

avg_labels = np.mean(labels_per_sample)
samples_with_no_labels = sum(1 for c in labels_per_sample if c == 0)
samples_with_multiple = sum(1 for c in labels_per_sample if c > 1)

print(f"  Average labels per sample: {avg_labels:.2f}")
print(f"  Samples with no labels: {samples_with_no_labels}")
print(f"  Samples with multiple labels: {samples_with_multiple} ({samples_with_multiple/len(train_data)*100:.1f}%)")

if samples_with_no_labels > 0:
    print(f"  ⚠️  WARNING: Some samples have no labels!")
    print("  → REVIEW: Check if this is intentional")

if samples_with_multiple / len(train_data) > 0.3:
    print(f"  ✓ Good multi-label distribution")
else:
    print(f"  ⚠️  INFO: Few samples with multiple labels")
    print("  → This is a mostly single-label problem")

# 5. Check model checkpoint
print("\n5. MODEL CHECKPOINT CHECK")
print("-"*80)

checkpoint_path = 'checkpoints/best_mental_health_model.pt'
if os.path.exists(checkpoint_path):
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print(f"  ✓ Checkpoint found: {checkpoint_path}")
        
        if 'epoch' in checkpoint:
            print(f"    Trained epochs: {checkpoint['epoch']}")
        
        if 'f1_score' in checkpoint:
            f1 = checkpoint['f1_score']
            print(f"    Best F1 score: {f1:.4f}")
            
            if f1 < 0.50:
                print(f"    ❌ ISSUE: Very low F1 score!")
                print(f"    → FIX: Retrain with improved script")
            elif f1 < 0.70:
                print(f"    ⚠️  WARNING: Below-average F1 score")
                print(f"    → RECOMMENDATION: Try improved training")
            else:
                print(f"    ✓ Good F1 score")
        
        if 'optimal_thresholds' in checkpoint:
            print(f"    ✓ Has optimal thresholds")
        else:
            print(f"    ⚠️  No optimal thresholds found")
            print(f"    → RECOMMENDATION: Retrain with improved script")
        
        if 'class_weights' in checkpoint:
            print(f"    ✓ Has class weights")
        else:
            print(f"    ❌ ISSUE: No class weights")
            print(f"    → FIX: Retrain with class weighting")
            
    except Exception as e:
        print(f"  ❌ Error loading checkpoint: {e}")
else:
    print(f"  ❌ No checkpoint found at {checkpoint_path}")
    print(f"  → ACTION: Train model first")

# 6. Summary and recommendations
print("\n" + "="*80)
print("SUMMARY AND RECOMMENDATIONS")
print("="*80)

issues_found = []
warnings_found = []

if imbalance_ratio > 1.5:
    issues_found.append("Class imbalance")

if len(train_data) < 500:
    issues_found.append("Small dataset")

if empty_texts > 0 or samples_with_no_labels > 0:
    issues_found.append("Data quality issues")

if not os.path.exists(checkpoint_path):
    issues_found.append("No trained model")
elif 'f1_score' in checkpoint and checkpoint['f1_score'] < 0.50:
    issues_found.append("Poor model performance")

if len(val_data) < 100:
    warnings_found.append("Small validation set")

if short_texts / len(train_data) > 0.1:
    warnings_found.append("Many short texts")

print(f"\n🔴 CRITICAL ISSUES FOUND: {len(issues_found)}")
for issue in issues_found:
    print(f"  • {issue}")

print(f"\n🟡 WARNINGS: {len(warnings_found)}")
for warning in warnings_found:
    print(f"  • {warning}")

print("\n" + "="*80)
print("RECOMMENDED ACTIONS:")
print("="*80)

print("""
1. ✅ Use the improved training script:
   python train_bert_improved.py
   
   This addresses:
   • Class imbalance (with class weights)
   • Small dataset (with augmentation)
   • Poor metrics (with better evaluation)
   • Overfitting (with early stopping)

2. ✅ Monitor training closely:
   • Loss should decrease smoothly
   • F1 scores should be > 0.70
   • Training and validation metrics should be close

3. ✅ After training completes:
   • Check final F1 scores in output
   • Verify optimal thresholds are saved
   • Test with sample predictions

4. ✅ If scores are still low (<0.60):
   • Check data quality (labels correct?)
   • Try different hyperparameters
   • Consider collecting more data
""")

print("="*80)
print("✓ Analysis complete!")
print("="*80)
