"""
Retrain Model for Complex Text Recognition
===========================================
Merges complex training data with existing data and retrains the model
"""

import json
import os
import subprocess
import sys

def merge_training_data():
    """Merge complex training data with existing data"""
    print("="*80)
    print("MERGING TRAINING DATA")
    print("="*80)
    
    all_data = []
    
    # Load existing training data
    existing_files = [
        'train_data.json',
        'comprehensive_training_data.json',
        'complex_training_data.json'
    ]
    
    for file in existing_files:
        if os.path.exists(file):
            print(f"Loading {file}...")
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_data.extend(data)
                print(f"  Added {len(data)} examples")
    
    # Remove duplicates (simple text-based deduplication)
    seen = set()
    unique_data = []
    for item in all_data:
        text_key = item['text'].lower().strip()
        if text_key not in seen:
            seen.add(text_key)
            unique_data.append(item)
    
    print(f"\nTotal unique examples: {len(unique_data)}")
    
    # Split into train/val (80/20)
    import random
    random.seed(42)
    random.shuffle(unique_data)
    
    split_idx = int(len(unique_data) * 0.8)
    train_data = unique_data[:split_idx]
    val_data = unique_data[split_idx:]
    
    # Save merged data
    with open('merged_complex_train_data.json', 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    
    with open('merged_complex_val_data.json', 'w', encoding='utf-8') as f:
        json.dump(val_data, f, indent=2, ensure_ascii=False)
    
    print(f"Train: {len(train_data)} examples")
    print(f"Val: {len(val_data)} examples")
    print("Saved to merged_complex_train_data.json and merged_complex_val_data.json")
    
    return True


def retrain_model():
    """Retrain the model with complex data"""
    print("\n" + "="*80)
    print("RETRAINING MODEL FOR COMPLEX TEXTS")
    print("="*80)
    
    # Update data paths in training script
    training_script = 'train_for_maximum_accuracy.py'
    
    if not os.path.exists(training_script):
        print(f"Error: {training_script} not found")
        return False
    
    # Run training
    print(f"\nRunning {training_script}...")
    print("This may take 2-4 hours...\n")
    
    try:
        # Temporarily rename data files for training
        if os.path.exists('merged_complex_train_data.json'):
            if os.path.exists('train_data.json'):
                os.rename('train_data.json', 'train_data_backup.json')
            os.rename('merged_complex_train_data.json', 'train_data.json')
        
        if os.path.exists('merged_complex_val_data.json'):
            if os.path.exists('val_data.json'):
                os.rename('val_data.json', 'val_data_backup.json')
            os.rename('merged_complex_val_data.json', 'val_data.json')
        
        # Run training
        result = subprocess.run(
            [sys.executable, training_script],
            capture_output=False,
            text=True
        )
        
        # Restore original files
        if os.path.exists('train_data.json'):
            os.rename('train_data.json', 'merged_complex_train_data.json')
        if os.path.exists('train_data_backup.json'):
            os.rename('train_data_backup.json', 'train_data.json')
        
        if os.path.exists('val_data.json'):
            os.rename('val_data.json', 'merged_complex_val_data.json')
        if os.path.exists('val_data_backup.json'):
            os.rename('val_data_backup.json', 'val_data.json')
        
        if result.returncode == 0:
            print("\n[OK] Training completed successfully!")
            return True
        else:
            print("\n[X] Training had errors")
            return False
            
    except Exception as e:
        print(f"\n[X] Error during training: {e}")
        return False


def main():
    """Main function"""
    print("="*80)
    print("RETRAIN MODEL FOR COMPLEX TEXT RECOGNITION")
    print("="*80)
    
    # Step 1: Generate complex training data
    print("\nStep 1: Generating complex training data...")
    from generate_complex_training_data import generate_complex_training_data
    data = generate_complex_training_data()
    
    with open('complex_training_data.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[OK] Generated {len(data)} complex examples")
    
    # Step 2: Merge with existing data
    print("\nStep 2: Merging training data...")
    if merge_training_data():
        print("[OK] Data merged successfully")
    else:
        print("[X] Data merging failed")
        return
    
    # Step 3: Retrain model
    print("\nStep 3: Retraining model...")
    print("[!] This will take 2-4 hours. Continue? (y/n): ", end='')
    response = input().strip().lower()
    
    if response == 'y':
        if retrain_model():
            print("\n" + "="*80)
            print("[OK] MODEL RETRAINED SUCCESSFULLY!")
            print("="*80)
            print("\nNext steps:")
            print("1. Test with: python test_complex_texts.py")
            print("2. Update multistage_classifier.py to use new model")
            print("3. Restart the application")
        else:
            print("\n[X] Training failed. Check errors above.")
    else:
        print("\nTraining skipped. You can run it manually later.")


if __name__ == '__main__':
    main()

