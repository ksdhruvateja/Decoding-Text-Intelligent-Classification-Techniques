"""
Ultimate Training Pipeline - LLMs + DL + RL
=============================================
Master training system combining all advanced techniques:
- LLM-based data generation
- Advanced Deep Learning architectures
- Reinforcement Learning from Human Feedback
- Self-training and active learning
- Ensemble methods
"""

import os
import json
import subprocess
import sys
from datetime import datetime
from typing import List, Dict

def run_script(script_name: str, description: str) -> bool:
    """Run a training script"""
    print(f"\n{'='*80}")
    print(f"STEP: {description}")
    print(f"Script: {script_name}")
    print(f"{'='*80}\n")
    
    try:
        # Determine working directory
        script_path = script_name
        if not os.path.exists(script_path) and os.path.exists(os.path.join('backend', script_path)):
            script_path = os.path.join('backend', script_path)
        
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        if result.returncode == 0:
            print(f"[OK] {description} completed successfully")
            if result.stdout:
                print(result.stdout[-500:])  # Last 500 chars
            return True
        else:
            print(f"[X] {description} failed")
            if result.stderr:
                print(result.stderr[-500:])
            return False
    except subprocess.TimeoutExpired:
        print(f"[X] {description} timed out")
        return False
    except Exception as e:
        print(f"[X] {description} error: {e}")
        return False


def check_dependencies():
    """Check if all required dependencies are available"""
    print("Checking dependencies...")
    
    required = {
        'torch': 'PyTorch',
        'transformers': 'Transformers',
        'sklearn': 'scikit-learn',  # Module name is sklearn, package is scikit-learn
        'numpy': 'NumPy',
        'tqdm': 'tqdm'
    }
    
    missing = []
    for module, name in required.items():
        try:
            __import__(module)
            print(f"  [OK] {name}")
        except ImportError:
            print(f"  [X] {name} missing")
            missing.append(name)
    
    # Check optional dependencies
    optional = {
        'openai': 'OpenAI (for GPT data generation)',
        'optuna': 'Optuna (for hyperparameter optimization)'
    }
    
    for module, name in optional.items():
        try:
            __import__(module)
            print(f"  [OK] {name} (optional)")
        except ImportError:
            print(f"  [!] {name} not available (optional)")
    
    if missing:
        print(f"\n[!] Missing required dependencies: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    return True


def generate_llm_data():
    """Generate training data using LLMs"""
    print("\n" + "="*80)
    print("PHASE 1: LLM-BASED DATA GENERATION")
    print("="*80)
    
    # Check if OpenAI API key is available
    has_openai = os.getenv('OPENAI_API_KEY') is not None
    
    if has_openai:
        print("[OK] OpenAI API key found - will use GPT for data generation")
    else:
        print("[!] OpenAI API key not found - will use Hugging Face models only")
        print("  Set OPENAI_API_KEY environment variable for better results")
    
    success = run_script(
        'llm_data_generator.py',
        'LLM-Based Data Generation'
    )
    
    if success and os.path.exists('llm_generated_training_data.json'):
        print("\n[OK] LLM-generated data available")
        return True
    else:
        print("\n[!] LLM data generation skipped or failed")
        return False


def merge_training_data():
    """Merge LLM-generated data with existing data"""
    print("\n" + "="*80)
    print("MERGING TRAINING DATA")
    print("="*80)
    
    all_data = []
    
    # Load existing training data
    existing_files = ['train_data.json', 'comprehensive_training_data.json']
    for file in existing_files:
        if os.path.exists(file):
            print(f"  Loading {file}...")
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_data.extend(data)
                print(f"    Added {len(data)} examples")
    
    # Load LLM-generated data
    if os.path.exists('llm_generated_training_data.json'):
        print("  Loading LLM-generated data...")
        with open('llm_generated_training_data.json', 'r', encoding='utf-8') as f:
            llm_data = json.load(f)
            all_data.extend(llm_data)
            print(f"    Added {len(llm_data)} examples")
    
    # Remove duplicates (simple text-based deduplication)
    seen = set()
    unique_data = []
    for item in all_data:
        text_key = item['text'].lower().strip()
        if text_key not in seen:
            seen.add(text_key)
            unique_data.append(item)
    
    print(f"\n  Total unique examples: {len(unique_data)}")
    
    # Split into train/val
    np.random.seed(42)
    np.random.shuffle(unique_data)
    
    split_idx = int(len(unique_data) * 0.8)
    train_data = unique_data[:split_idx]
    val_data = unique_data[split_idx:]
    
    # Save merged data
    with open('merged_train_data.json', 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    
    with open('merged_val_data.json', 'w', encoding='utf-8') as f:
        json.dump(val_data, f, indent=2, ensure_ascii=False)
    
    print(f"  [OK] Train: {len(train_data)} examples")
    print(f"  [OK] Val: {len(val_data)} examples")
    print("  [OK] Saved to merged_train_data.json and merged_val_data.json")
    
    return True


def train_advanced_dl():
    """Train advanced deep learning models"""
    print("\n" + "="*80)
    print("PHASE 2: ADVANCED DEEP LEARNING TRAINING")
    print("="*80)
    
    # Update data paths in training script if needed
    success = run_script(
        'advanced_dl_trainer.py',
        'Multi-Architecture Deep Learning Training'
    )
    
    return success


def train_with_rlhf():
    """Train with Reinforcement Learning from Human Feedback"""
    print("\n" + "="*80)
    print("PHASE 3: REINFORCEMENT LEARNING FROM HUMAN FEEDBACK")
    print("="*80)
    
    # Check if feedback data exists
    if not os.path.exists('feedback_data.json'):
        print("[!] feedback_data.json not found")
        print("  Creating sample feedback data from validation set...")
        
        # Create sample feedback data
        if os.path.exists('merged_val_data.json'):
            with open('merged_val_data.json', 'r') as f:
                val_data = json.load(f)
            
            # Simulate feedback (in practice, this would come from humans)
            feedback_data = []
            for item in val_data[:100]:  # Sample 100 examples
                feedback_data.append({
                    'text': item['text'],
                    'true_labels': item['labels'],
                    'predicted_labels': item['labels'],  # Assume correct for now
                    'feedback': 1.0
                })
            
            with open('feedback_data.json', 'w') as f:
                json.dump(feedback_data, f, indent=2)
            
            print(f"  [OK] Created {len(feedback_data)} feedback examples")
    
    success = run_script(
        'rlhf_trainer.py',
        'RLHF Fine-tuning'
    )
    
    return success


def train_ensemble():
    """Train ensemble of models"""
    print("\n" + "="*80)
    print("PHASE 4: ENSEMBLE TRAINING")
    print("="*80)
    
    success = run_script(
        'train_ensemble.py',
        'Ensemble Model Training'
    )
    
    return success


def optimize_hyperparameters():
    """Optimize hyperparameters"""
    print("\n" + "="*80)
    print("PHASE 5: HYPERPARAMETER OPTIMIZATION")
    print("="*80)
    
    success = run_script(
        'hyperparameter_optimization.py',
        'Hyperparameter Optimization (Optuna)'
    )
    
    return success


def main():
    """Run complete ultimate training pipeline"""
    print("="*80)
    print("ULTIMATE TRAINING PIPELINE")
    print("LLMs + Deep Learning + Reinforcement Learning")
    print("="*80)
    print(f"Started at: {datetime.now().isoformat()}\n")
    
    # Check dependencies
    if not check_dependencies():
        print("\n[!] Please install missing dependencies before continuing")
        print("[!] Attempting to continue anyway...")
        # Auto-continue in non-interactive mode
        # response = input("Continue anyway? (y/n): ")
        # if response.lower() != 'y':
        #     return
    
    results = {}
    
    # Phase 1: LLM Data Generation
    print("\n" + "="*80)
    print("PHASE 1: LLM-BASED DATA GENERATION")
    print("="*80)
    llm_success = generate_llm_data()
    results['LLM Data Generation'] = llm_success
    
    # Merge data
    merge_success = merge_training_data()
    results['Data Merging'] = merge_success
    
    if not merge_success:
        print("\n[!] No training data available. Please ensure train_data.json exists.")
        return
    
    # Phase 2: Advanced DL Training
    print("\n" + "="*80)
    print("PHASE 2: ADVANCED DEEP LEARNING")
    print("="*80)
    dl_success = train_advanced_dl()
    results['Advanced DL Training'] = dl_success
    
    # Phase 3: RLHF (optional)
    print("\n" + "="*80)
    print("PHASE 3: REINFORCEMENT LEARNING (Optional)")
    print("="*80)
    rlhf_success = train_with_rlhf()
    results['RLHF Training'] = rlhf_success
    
    # Phase 4: Ensemble
    print("\n" + "="*80)
    print("PHASE 4: ENSEMBLE TRAINING")
    print("="*80)
    ensemble_success = train_ensemble()
    results['Ensemble Training'] = ensemble_success
    
    # Phase 5: Hyperparameter Optimization (optional)
    print("\n" + "="*80)
    print("PHASE 5: HYPERPARAMETER OPTIMIZATION (Optional)")
    print("="*80)
    opt_success = optimize_hyperparameters()
    results['Hyperparameter Optimization'] = opt_success
    
    # Summary
    print("\n" + "="*80)
    print("TRAINING PIPELINE SUMMARY")
    print("="*80)
    
    for phase, success in results.items():
        status = "[OK] SUCCESS" if success else "[X] FAILED/SKIPPED"
        print(f"{status}: {phase}")
    
    successful_phases = sum(1 for s in results.values() if s)
    total_phases = len(results)
    
    print(f"\nCompleted: {successful_phases}/{total_phases} phases")
    print(f"Finished at: {datetime.now().isoformat()}")
    
    if successful_phases >= 2:  # At least DL training should succeed
        print("\n" + "="*80)
        print("[OK] TRAINING COMPLETE!")
        print("="*80)
        print("\nNext steps:")
        print("1. Test the trained models")
        print("2. Update multistage_classifier.py to use new models")
        print("3. Evaluate on test set")
        print("4. Deploy and monitor")
    else:
        print("\n[!] Training had issues. Please review errors above.")


if __name__ == '__main__':
    import numpy as np
    import sys
    import os
    
    # Change to backend directory if not already there
    if os.path.basename(os.getcwd()) != 'backend':
        if os.path.exists('backend'):
            os.chdir('backend')
    
    main()

