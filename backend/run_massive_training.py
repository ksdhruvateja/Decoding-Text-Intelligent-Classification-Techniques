"""
Complete Massive Training Pipeline
==================================
Runs the complete pipeline:
1. Collect massive data from all sources
2. Train model on massive dataset
"""

import subprocess
import sys
import os

def main():
    print("="*80)
    print("COMPLETE MASSIVE TRAINING PIPELINE")
    print("="*80)
    print("\nThis will:")
    print("  1. Collect data from multiple sources (LLM, Reddit, datasets)")
    print("  2. Generate comprehensive training data")
    print("  3. Augment data with variations")
    print("  4. Train model on massive dataset")
    print("\nThis may take 2-4 hours depending on your hardware and data sources.")
    print("="*80)
    
    response = input("\nContinue? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled.")
        return
    
    # Step 1: Collect massive data
    print("\n" + "="*80)
    print("STEP 1: COLLECTING MASSIVE DATA")
    print("="*80)
    try:
        result = subprocess.run(
            [sys.executable, 'massive_data_collector.py'],
            check=True,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        print("\n✓ Data collection completed!")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error collecting data: {e}")
        print("Continuing with existing data if available...")
    except FileNotFoundError:
        print("\n❌ massive_data_collector.py not found!")
        print("Please make sure you're running this from the backend directory.")
        return
    
    # Step 2: Train model
    print("\n" + "="*80)
    print("STEP 2: TRAINING MODEL ON MASSIVE DATA")
    print("="*80)
    try:
        result = subprocess.run(
            [sys.executable, 'train_massive.py'],
            check=True,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        print("\n✓ Model training completed!")
        print("\n" + "="*80)
        print("MASSIVE TRAINING COMPLETE!")
        print("="*80)
        print("\nYour trained model is saved at:")
        print("  checkpoints/best_massive_model.pt")
        print("\nThe model has been trained on a massive dataset covering:")
        print("  ✓ LLM-generated examples")
        print("  ✓ Reddit data")
        print("  ✓ Public datasets")
        print("  ✓ Comprehensive fixed data")
        print("  ✓ Augmented variations")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error training model: {e}")
        return
    except FileNotFoundError:
        print("\n❌ train_massive.py not found!")
        print("Please make sure you're running this from the backend directory.")
        return

if __name__ == '__main__':
    main()

