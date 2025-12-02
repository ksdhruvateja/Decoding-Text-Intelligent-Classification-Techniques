"""
Complete Retraining Pipeline - Comprehensive Fixed Data
=======================================================
Runs the complete pipeline:
1. Generate comprehensive training data (covering all edge cases)
2. Train the model with this data
"""

import os
import sys
import subprocess

def main():
    print("="*80)
    print("COMPREHENSIVE RETRAINING PIPELINE")
    print("="*80)
    print("\nThis will:")
    print("1. Generate comprehensive training data (covering all edge cases)")
    print("2. Train the BERT model with this data")
    print("\nThis may take 30-60 minutes depending on your hardware.")
    print("="*80)
    
    response = input("\nContinue? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled.")
        return
    
    # Step 1: Generate training data
    print("\n" + "="*80)
    print("STEP 1: Generating Comprehensive Training Data")
    print("="*80)
    try:
        result = subprocess.run(
            [sys.executable, 'generate_comprehensive_fixed_training_data.py'],
            check=True,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        print("\n✓ Training data generated successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error generating training data: {e}")
        return
    except FileNotFoundError:
        print("\n❌ generate_comprehensive_fixed_training_data.py not found!")
        print("Please make sure you're running this from the backend directory.")
        return
    
    # Step 2: Train the model
    print("\n" + "="*80)
    print("STEP 2: Training BERT Model")
    print("="*80)
    try:
        result = subprocess.run(
            [sys.executable, 'train_comprehensive_fixed.py'],
            check=True,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        print("\n✓ Model training completed successfully!")
        print("\n" + "="*80)
        print("TRAINING COMPLETE!")
        print("="*80)
        print("\nYour trained model is saved at:")
        print("  checkpoints/best_mental_health_model.pt")
        print("\nThe model should now correctly classify:")
        print("  ✓ Positive confident/empowered statements")
        print("  ✓ Positive relationship/love statements")
        print("  ✓ Frustration/annoyance (NOT self-harm)")
        print("  ✓ All other statement types")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error training model: {e}")
        return
    except FileNotFoundError:
        print("\n❌ train_comprehensive_fixed.py not found!")
        print("Please make sure you're running this from the backend directory.")
        return

if __name__ == '__main__':
    main()

