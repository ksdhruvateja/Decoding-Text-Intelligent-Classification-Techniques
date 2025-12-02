# 📂 File Structure Visualization

## Complete Project Structure

```
Bert - text classifier/
│
├── 📄 Readme.md                              # Project overview & setup
├── 📄 MODEL_ARCHITECTURE_DOCUMENTATION.md   # Complete architecture docs (THIS FILE)
│
├── 📁 backend/                              # Backend Python code
│   │
│   ├── 🚀 CORE APPLICATION FILES
│   │   ├── app.py                           # Flask API server (ENTRY POINT)
│   │   ├── multistage_classifier.py         # Main classifier orchestrator
│   │   ├── bert_classifier.py               # BERT model definition & loader
│   │   └── rule_classifier.py               # Rule-based fallback
│   │
│   ├── 🎓 TRAINING FILES
│   │   ├── generate_clean_balanced_data.py  # Generate training dataset
│   │   ├── train_clean_balanced.py          # Train BERT model
│   │   ├── train_comprehensive_fixed.py     # Advanced training
│   │   ├── train_massive.py                 # Large-scale training
│   │   └── retrain_comprehensive.py         # Full retraining pipeline
│   │
│   ├── 🔧 SUPPORTING CLASSIFIERS
│   │   ├── llm_verifier.py                  # LLM-based verification
│   │   ├── gpt_llm_classifier.py           # GPT/OpenAI classifier
│   │   ├── hybrid_classifier.py            # Hybrid Rules + LLM + BERT
│   │   └── model_calibrated.py             # Calibrated model loader
│   │
│   ├── 📊 DATA FILES
│   │   ├── clean_balanced_train.json        # Training dataset (2,320 examples)
│   │   ├── clean_balanced_val.json          # Validation dataset (580 examples)
│   │   ├── training_data.json               # Legacy training data
│   │   └── data/                            # Additional data files
│   │
│   ├── 💾 MODEL CHECKPOINTS
│   │   └── checkpoints/
│   │       ├── best_clean_balanced_model.pt  # Best trained model
│   │       ├── best_mental_health_model.pt   # Legacy model
│   │       └── best_calibrated_model_temp.pt # Calibrated model
│   │
│   ├── 📚 DOCUMENTATION
│   │   ├── QUICK_FIX_TRAINING.md            # Quick training guide
│   │   ├── FIXES_APPLIED.md                 # Recent fixes summary
│   │   ├── HOW_IT_WORKS.md                  # System overview
│   │   ├── LABEL_DEFINITIONS.md             # Category definitions
│   │   └── COMPLETE_LLM_TRAINING_GUIDE.md   # LLM training guide
│   │
│   └── 🧪 TESTING & EVALUATION
│       ├── evaluate_model.py               # Model evaluation
│       ├── test_classification_fix.py       # Test fixes
│       └── comprehensive_evaluation.py      # Full evaluation
│
└── 📁 frontend/                             # React frontend (if exists)
    ├── src/
    ├── public/
    └── package.json
```

---

## 🔄 Data Flow Through Files

### **Classification Request Flow**

```
User Request
    │
    ▼
┌─────────────────────────────────────────┐
│  app.py                                  │
│  ├── initialize_classifier()            │
│  │   └── Loads multistage_classifier.py │
│  │                                        │
│  └── classify_text_endpoint()           │
│      └── Calls classifier.classify()     │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  multistage_classifier.py               │
│  ├── MultiStageClassifier.classify()   │
│  │   ├── Stage 1: Rule-based sentiment │
│  │   │   └── RuleBasedFilter           │
│  │   │       └── analyze_sentiment()  │
│  │   │                                 │
│  │   ├── Stage 2: BERT predictions    │
│  │   │   └── _get_model_predictions() │
│  │   │       └── Uses bert_classifier.py│
│  │   │                                 │
│  │   ├── Stage 3: Rule overrides      │
│  │   │   └── RuleBasedFilter          │
│  │   │       └── check_override()     │
│  │   │                                 │
│  │   ├── Stage 4: Score suppression   │
│  │   │   └── Suppress risk scores     │
│  │   │                                 │
│  │   └── Stage 5: Final classification│
│  │       └── Apply thresholds         │
│  │                                    │
│  └── Returns JSON response            │
└──────────────┬──────────────────────────┘
               │
               ▼
        JSON Response
```

---

### **Training Pipeline Flow**

```
┌─────────────────────────────────────────┐
│  generate_clean_balanced_data.py       │
│  └── Creates training dataset           │
│      ├── clean_balanced_train.json     │
│      └── clean_balanced_val.json      │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  train_clean_balanced.py                │
│  ├── Loads training data                │
│  ├── Initializes BERT model            │
│  │   └── Uses bert_classifier.py       │
│  ├── Training loop (20 epochs)          │
│  │   ├── Forward pass                  │
│  │   ├── Loss calculation              │
│  │   ├── Backward pass                 │
│  │   └── Validation                    │
│  └── Saves checkpoint                   │
│      └── checkpoints/                  │
│          └── best_clean_balanced_model.pt│
└─────────────────────────────────────────┘
```

---

## 📋 File Purpose Summary

### **🚀 Core Application Files**

| File | Purpose | Key Functions |
|------|---------|---------------|
| `app.py` | Flask API server | `initialize_classifier()`, `classify_text_endpoint()` |
| `multistage_classifier.py` | Main orchestrator | `MultiStageClassifier.classify()`, `RuleBasedFilter` |
| `bert_classifier.py` | BERT model | `BERTMentalHealthClassifier`, `MentalHealthClassifierService` |
| `rule_classifier.py` | Rule-based fallback | Pattern matching, keyword detection |

### **🎓 Training Files**

| File | Purpose | Output |
|------|---------|--------|
| `generate_clean_balanced_data.py` | Generate dataset | `clean_balanced_train.json`, `clean_balanced_val.json` |
| `train_clean_balanced.py` | Train model | `checkpoints/best_clean_balanced_model.pt` |
| `train_comprehensive_fixed.py` | Advanced training | Model with Focal Loss, Label Smoothing |
| `train_massive.py` | Large-scale training | Model trained on massive dataset |

### **🔧 Supporting Files**

| File | Purpose |
|------|---------|
| `llm_verifier.py` | LLM-based verification |
| `gpt_llm_classifier.py` | GPT/OpenAI integration |
| `hybrid_classifier.py` | Hybrid Rules + LLM + BERT |
| `model_calibrated.py` | Temperature-scaled model |

---

## 🎯 Key File Relationships

```
app.py
  └── imports → multistage_classifier.py
      ├── uses → bert_classifier.py (for BERT model)
      ├── uses → RuleBasedFilter (for rules)
      └── uses → LLMVerifier (for LLM validation)

multistage_classifier.py
  ├── imports → bert_classifier.py
  │   └── BERTMentalHealthClassifier
  │
  ├── RuleBasedFilter class
  │   ├── analyze_sentiment()
  │   └── check_override()
  │
  └── MultiStageClassifier class
      ├── __init__() → Loads model from checkpoint
      └── classify() → Main classification pipeline

bert_classifier.py
  ├── BERTMentalHealthClassifier (PyTorch model)
  └── MentalHealthClassifierService (Service wrapper)

train_clean_balanced.py
  ├── imports → bert_classifier.py
  │   └── BERTMentalHealthClassifier
  │
  ├── CleanDataset (PyTorch Dataset)
  └── Training loop
      └── Saves → checkpoints/best_clean_balanced_model.pt
```

---

## 🔍 File Dependencies Graph

```
┌─────────────────┐
│    app.py       │
│  (Entry Point)  │
└────────┬────────┘
         │
         ▼
┌─────────────────────────┐
│ multistage_classifier.py│
│  (Main Orchestrator)    │
└─────┬───────────┬───────┘
      │           │
      ▼           ▼
┌──────────┐  ┌──────────────┐
│bert_class│  │rule_classifier│
│ifier.py  │  │.py (fallback) │
└────┬─────┘  └──────────────┘
     │
     ▼
┌─────────────────────────┐
│ checkpoints/            │
│ best_clean_balanced_    │
│ model.pt                │
└─────────────────────────┘
```

---

## 📊 Data Files Structure

```
backend/
├── clean_balanced_train.json
│   └── Array of training examples
│       [
│         {
│           "text": "I know I can achieve...",
│           "labels": {
│             "neutral": 1,
│             "stress": 0,
│             ...
│           }
│         },
│         ...
│       ]
│
├── clean_balanced_val.json
│   └── Array of validation examples (same structure)
│
└── checkpoints/
    └── best_clean_balanced_model.pt
        └── PyTorch checkpoint
            {
              "model_state_dict": {...},
              "optimal_thresholds": {...},
              "f1_score": 0.95,
              "epoch": 20
            }
```

---

## 🎓 Training vs Inference Files

### **Training Mode**
```
generate_clean_balanced_data.py
    → Creates JSON data files
        ↓
train_clean_balanced.py
    → Loads JSON data
    → Trains BERT model
    → Saves checkpoint
```

### **Inference Mode**
```
app.py
    → Loads checkpoint
        ↓
multistage_classifier.py
    → Loads BERT model from checkpoint
    → Classifies text
```

---

## 🔑 Key Concepts by File

### **`app.py`**
- **Flask application**
- **HTTP endpoints**
- **Request/Response handling**

### **`multistage_classifier.py`**
- **Multi-stage pipeline**
- **Rule-based overrides**
- **Score suppression**
- **Final classification**

### **`bert_classifier.py`**
- **Neural network architecture**
- **Model loading**
- **Inference**

### **`train_clean_balanced.py`**
- **Training loop**
- **Loss calculation**
- **Threshold optimization**
- **Model saving**

---

## 📝 Quick Reference

### **To Start the API**:
```bash
cd backend
python app.py
```

### **To Train a Model**:
```bash
cd backend
python generate_clean_balanced_data.py
python train_clean_balanced.py
```

### **To Test Classification**:
```bash
curl -X POST http://localhost:5000/api/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "Your text here"}'
```

---

**This visualization helps you understand:**
- ✅ Which files do what
- ✅ How data flows through the system
- ✅ File dependencies
- ✅ Training vs inference mode
- ✅ Where to find specific functionality

