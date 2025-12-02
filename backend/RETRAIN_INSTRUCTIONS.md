# Comprehensive Retraining Instructions

## 🎯 Overview

This retraining pipeline fixes all the misclassification issues by training the model with comprehensive data covering all edge cases:

- ✅ Positive confident/empowered statements ("I am unstoppable")
- ✅ Positive relationship statements ("I will marry her")
- ✅ Frustration/annoyance ("I'm sick of people wasting my time")
- ✅ All other statement types

## 🚀 Quick Start

### Option 1: Run Everything at Once (Recommended)

```bash
cd backend
python retrain_comprehensive.py
```

This will:
1. Generate comprehensive training data (~1,360 examples)
2. Train the BERT model (15 epochs, ~30-60 minutes)

### Option 2: Run Steps Separately

#### Step 1: Generate Training Data

```bash
cd backend
python generate_comprehensive_fixed_training_data.py
```

This creates:
- `train_data.json` - Training examples (~1,088 examples)
- `val_data.json` - Validation examples (~272 examples)

#### Step 2: Train the Model

```bash
cd backend
python train_comprehensive_fixed.py
```

This trains the model and saves it to:
- `checkpoints/best_mental_health_model.pt`

## 📊 Training Data Breakdown

The comprehensive training data includes:

| Category | Examples | Labels |
|----------|----------|--------|
| Positive confident/empowered | ~180 | neutral=1, all others=0 |
| Positive relationship/love | ~180 | neutral=1, all others=0 |
| Frustration/annoyance | ~150 | stress=1, emotional_distress=1, others=0 |
| Positive statements | ~200 | neutral=1, all others=0 |
| Neutral statements | ~150 | neutral=1, all others=0 |
| Stress | ~150 | stress=1, others=0 |
| Emotional distress | ~150 | emotional_distress=1, others=0 |
| Self-harm low | ~100 | self_harm_low=1, others=0 |
| Self-harm high | ~100 | self_harm_high=1, others=0 |
| Unsafe environment | ~100 | unsafe_environment=1, others=0 |
| **Total** | **~1,360** | |

## ⚙️ Training Configuration

- **Model**: BERT-base-uncased
- **Epochs**: 15
- **Batch Size**: 16
- **Learning Rate**: 2e-5
- **Max Length**: 128 tokens
- **Optimizer**: AdamW with weight decay
- **Scheduler**: Linear warmup + decay
- **Loss**: BCEWithLogitsLoss with class weights

## 📈 Expected Results

After training, the model should correctly classify:

✅ **"I am unstoppable"** → positive/neutral (NOT self-harm)
✅ **"I will marry her"** → positive/neutral (NOT self-harm or unsafe_environment)
✅ **"I'm sick of people wasting my time"** → stress/emotional_distress (NOT self-harm or unsafe_environment)
✅ **"I want to kill myself"** → self_harm_high (high_risk)
✅ **"I'm worried about my exam"** → stress (concerning)
✅ **"I went to the store"** → neutral (safe)

## 🔧 Requirements

```bash
pip install torch transformers scikit-learn numpy tqdm scipy
```

## 📝 Notes

- Training takes 30-60 minutes on CPU, 10-20 minutes on GPU
- The model will automatically find optimal thresholds for each label
- Best model is saved based on F1 score
- The trained model will be used automatically by `multistage_classifier.py`

## 🐛 Troubleshooting

### Error: "train_data.json not found"
- Run `generate_comprehensive_fixed_training_data.py` first

### Error: "CUDA out of memory"
- Reduce batch size in `train_comprehensive_fixed.py` (change `batch_size` from 16 to 8)

### Low accuracy after training
- Increase epochs (change from 15 to 20-25)
- Check that training data was generated correctly
- Verify class weights are balanced

## ✅ Verification

After training, test the model with:

```python
from multistage_classifier import initialize_multistage_classifier

classifier = initialize_multistage_classifier('checkpoints/best_mental_health_model.pt')

# Test cases
test_cases = [
    "I am unstoppable and nothing can hold me back",
    "I will marry her and spend the rest of my life with her",
    "I'm sick of people wasting my time",
    "I want to kill myself",
    "I went to the store",
]

for text in test_cases:
    result = classifier.classify(text)
    print(f"\nText: {text}")
    print(f"Emotion: {result['emotion']}")
    print(f"Sentiment: {result['sentiment']}")
    print(f"Predictions: {result['predictions']}")
```

