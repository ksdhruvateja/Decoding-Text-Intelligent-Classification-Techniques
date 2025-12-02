"""
Fix Confidence Scores - Ensure Accuracy
========================================
Diagnoses and fixes issues with confidence score calculation
"""

import torch
import numpy as np
from multistage_classifier import MultiStageClassifier

def verify_confidence_scores():
    """Verify confidence scores are accurate"""
    print("="*80)
    print("VERIFYING CONFIDENCE SCORES")
    print("="*80)
    
    classifier = MultiStageClassifier()
    
    test_texts = [
        "I absolutely loved the restaurant",
        "The service was terrible",
        "I want to hurt myself",
        "I'm going to kill you",
    ]
    
    print("\nTesting confidence score accuracy...\n")
    
    for text in test_texts:
        print(f"{'='*80}")
        print(f"Text: {text}")
        print(f"{'-'*80}")
        
        result = classifier.classify(text)
        
        # Check if scores are valid
        all_scores = result.get('all_scores', {})
        predictions = result.get('predictions', [])
        
        print(f"All Scores:")
        for label, score in sorted(all_scores.items(), key=lambda x: x[1], reverse=True):
            print(f"  {label:20s}: {score:.4f} ({score*100:.2f}%)")
        
        print(f"\nPredictions (above threshold):")
        if predictions:
            for pred in predictions:
                print(f"  {pred['label']:20s}: {pred['confidence']:.4f} ({pred['confidence']*100:.2f}%) [threshold: {pred['threshold']:.2f}]")
        else:
            print("  None")
        
        # Verify consistency
        print(f"\nVerification:")
        issues = []
        
        # Check 1: Scores should be between 0 and 1
        for label, score in all_scores.items():
            if score < 0 or score > 1:
                issues.append(f"  ❌ {label}: Score {score:.4f} is outside [0, 1]")
        
        # Check 2: Predictions should match scores
        for pred in predictions:
            label = pred['label']
            pred_score = pred['confidence']
            all_score = all_scores.get(label, 0)
            if abs(pred_score - all_score) > 0.001:
                issues.append(f"  ❌ {label}: Prediction score {pred_score:.4f} != all_scores {all_score:.4f}")
        
        # Check 3: Predictions should be above threshold
        for pred in predictions:
            if pred['confidence'] < pred['threshold']:
                issues.append(f"  ❌ {pred['label']}: Confidence {pred['confidence']:.4f} < threshold {pred['threshold']:.4f}")
        
        if issues:
            print("  Issues found:")
            for issue in issues:
                print(issue)
        else:
            print("  ✓ All checks passed")
        
        print()
    
    print("="*80)
    print("Verification complete!")
    print("="*80)


if __name__ == '__main__':
    verify_confidence_scores()

