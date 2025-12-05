"""
Start Backend with Trained Models
=================================
Launches Flask API with hybrid classifier (BERT + TF-IDF + rules)
"""

import subprocess
import sys
import os
from pathlib import Path

def check_checkpoints():
    """Verify trained model checkpoints exist"""
    checkpoints_dir = Path(__file__).parent.parent / "checkpoints"
    
    required_files = [
        "bert_classifier_best.pt",
        "tfidf_classifier.joblib"
    ]
    
    missing = []
    for filename in required_files:
        filepath = checkpoints_dir / filename
        if not filepath.exists():
            missing.append(filename)
    
    if missing:
        print("⚠️ Missing model checkpoints:")
        for f in missing:
            print(f"  - {f}")
        print("\nPlease train models first:")
        print("  python train_tfidf_classifier.py")
        print("  python train_bert_classifier.py")
        return False
    
    print("✅ All model checkpoints found")
    return True

def start_backend():
    """Start Flask backend server"""
    backend_dir = Path(__file__).parent
    app_path = backend_dir / "app.py"
    
    if not app_path.exists():
        print(f"❌ app.py not found at {app_path}")
        return False
    
    print("\n" + "="*80)
    print("STARTING BACKEND SERVER")
    print("="*80)
    print("\nBackend API will be available at:")
    print("  http://localhost:5000")
    print("\nEndpoints:")
    print("  GET  /api/health")
    print("  POST /api/classify")
    print("  POST /api/classify-formatted")
    print("  GET  /api/history")
    print("  GET  /api/stats")
    print("\n" + "="*80)
    print("\nPress Ctrl+C to stop the server")
    print("="*80 + "\n")
    
    try:
        # Set unbuffered output for real-time logs
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'
        
        subprocess.run(
            [sys.executable, str(app_path)],
            cwd=str(backend_dir),
            env=env
        )
    except KeyboardInterrupt:
        print("\n\n✓ Backend server stopped")
    except Exception as e:
        print(f"\n❌ Error starting backend: {e}")
        return False
    
    return True

def main():
    print("="*80)
    print("BACKEND STARTUP SCRIPT")
    print("="*80)
    print("\nThis script will:")
    print("  1. Verify trained model checkpoints")
    print("  2. Start Flask backend server")
    print("  3. Load hybrid classifier (BERT + TF-IDF + rules)")
    print("="*80 + "\n")
    
    # Check checkpoints
    if not check_checkpoints():
        sys.exit(1)
    
    # Start backend
    start_backend()

if __name__ == "__main__":
    main()
