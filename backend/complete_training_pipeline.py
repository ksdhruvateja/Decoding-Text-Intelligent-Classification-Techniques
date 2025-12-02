"""
Complete Training Pipeline for Maximum Accuracy
===============================================
One script to do everything:
1. Generate comprehensive training data
2. Train model with advanced techniques
3. Validate on diverse statements
4. Ensure accuracy on ANY statement
"""

import os
import sys
import subprocess
from datetime import datetime

def run_step(step_name, script_name, description):
    """Run a training step"""
    print("\n" + "="*80)
    print(f"STEP: {step_name}")
    print("="*80)
    print(f"{description}\n")
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            check=True,
            capture_output=False
        )
        print(f"\n✓ {step_name} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {step_name} failed: {e}")
        return False
    except FileNotFoundError:
        print(f"\n✗ Script {script_name} not found")
        return False


def main():
    """Run complete training pipeline"""
    print("="*80)
    print("COMPLETE TRAINING PIPELINE FOR MAXIMUM ACCURACY")
    print("="*80)
    print(f"Started at: {datetime.now().isoformat()}\n")
    
    steps = [
        ("Generate Comprehensive Data", 
         "generate_comprehensive_training_data.py",
         "Creating diverse training dataset covering all statement types"),
        
        ("Train Model for Maximum Accuracy",
         "train_for_maximum_accuracy.py",
         "Training with advanced techniques for best results"),
        
        ("Validate on Diverse Statements",
         "validate_any_statement.py",
         "Testing model on comprehensive test suite"),
    ]
    
    results = []
    for step_name, script, description in steps:
        success = run_step(step_name, script, description)
        results.append((step_name, success))
        
        if not success:
            print(f"\n⚠ Warning: {step_name} failed. Continuing anyway...")
    
    # Summary
    print("\n" + "="*80)
    print("PIPELINE SUMMARY")
    print("="*80)
    
    for step_name, success in results:
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"{status}: {step_name}")
    
    all_success = all(success for _, success in results)
    
    if all_success:
        print("\n" + "="*80)
        print("✓ COMPLETE PIPELINE SUCCESSFUL!")
        print("="*80)
        print("\nYour model is now trained for maximum accuracy on ANY statement!")
        print("\nNext steps:")
        print("1. Test with: python validate_any_statement.py")
        print("2. Use in app: Update multistage_classifier.py to load best_maximum_accuracy_model.pt")
        print("3. Deploy and test with real statements")
    else:
        print("\n" + "="*80)
        print("⚠ SOME STEPS FAILED")
        print("="*80)
        print("Please check the errors above and retry failed steps manually.")
    
    print(f"\nCompleted at: {datetime.now().isoformat()}")


if __name__ == '__main__':
    main()

