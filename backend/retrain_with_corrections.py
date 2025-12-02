"""
Retrain Model with Corrective Data
==================================
Retrains the model using corrective training data to fix misclassifications
"""

import json
import os
import shutil
from generate_corrective_training_data import merge_with_existing_data

def prepare_training_data():
    """Prepare corrected training data"""
    print("="*80)
    print("PREPARING CORRECTIVE TRAINING DATA")
    print("="*80)
    
    # Generate and merge corrective data
    merged_data = merge_with_existing_data()
    
    # Split into train/val (80/20)
    import random
    random.seed(42)
    random.shuffle(merged_data)
    
    split_idx = int(len(merged_data) * 0.8)
    train_data = merged_data[:split_idx]
    val_data = merged_data[split_idx:]
    
    # Save
    with open('train_data_corrected.json', 'w') as f:
        json.dump(train_data, f, indent=2)
    
    with open('val_data_corrected.json', 'w') as f:
        json.dump(val_data, f, indent=2)
    
    print(f"\n✓ Training data: {len(train_data)} examples")
    print(f"✓ Validation data: {len(val_data)} examples")
    print(f"✓ Saved to train_data_corrected.json and val_data_corrected.json")
    
    return train_data, val_data


def retrain_model():
    """Retrain model with corrected data"""
    print("\n" + "="*80)
    print("RETRAINING MODEL WITH CORRECTIONS")
    print("="*80)
    
    # Prepare data
    train_data, val_data = prepare_training_data()
    
    # Backup original files
    if os.path.exists('train_data.json'):
        shutil.copy('train_data.json', 'train_data_backup.json')
    if os.path.exists('val_data.json'):
        shutil.copy('val_data.json', 'val_data_backup.json')
    
    # Replace with corrected data
    shutil.copy('train_data_corrected.json', 'train_data.json')
    shutil.copy('val_data_corrected.json', 'val_data.json')
    
    print("\n✓ Data files updated")
    print("\nNow run the training script:")
    print("  python train_advanced_optimized.py")
    print("\nOr for faster training:")
    print("  python train_bert_model.py")


if __name__ == '__main__':
    retrain_model()

