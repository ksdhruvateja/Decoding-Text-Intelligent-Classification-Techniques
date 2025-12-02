# 🧠 Mental Health Text Classifier - Complete Architecture Documentation

## 📋 Table of Contents
1. [System Overview](#system-overview)
2. [File Structure & Purpose](#file-structure--purpose)
3. [System Architecture Flow](#system-architecture-flow)
4. [Classification Pipeline](#classification-pipeline)
5. [Training Pipeline](#training-pipeline)
6. [API Endpoints](#api-endpoints)
7. [Data Flow Diagrams](#data-flow-diagrams)

---

## 🎯 System Overview

This is a **Multi-Stage Mental Health Text Classifier** that combines:
- **BERT Deep Learning Model** (semantic understanding)
- **Rule-Based Filters** (pattern matching & safety overrides)
- **LLM Verification** (GPT/LLM ensemble for validation)
- **Strict Thresholds** (prevents false positives)

### Classification Categories
1. **neutral** - Everyday, informational statements
2. **stress** - Worry, pressure, anxiety
3. **emotional_distress** - Sadness, depression, hopelessness
4. **self_harm_low** - Suicidal ideation (thoughts)
5. **self_harm_high** - Active suicidal intent (plans)
6. **unsafe_environment** - Threats to others

---

## 📁 File Structure & Purpose

### 🚀 **Core Application Files**

```
backend/
├── app.py                          # Flask API server (entry point)
├── multistage_classifier.py        # Main classification orchestrator
├── bert_classifier.py              # BERT model definition & loader
└── rule_classifier.py              # Rule-based fallback classifier
```

#### **`app.py`** - Flask API Server
- **Purpose**: HTTP API endpoints for classification
- **Key Functions**:
  - `initialize_classifier()` - Loads MultiStageClassifier on startup
  - `classify_text_endpoint()` - POST `/api/classify` - Classify single text
  - `batch_classify_endpoint()` - POST `/api/batch-classify` - Classify multiple texts
  - `get_conversation_history()` - GET `/api/history` - Get classification logs
- **Flow**: `Request → app.py → classifier_service.classify() → Response`

#### **`multistage_classifier.py`** - Main Classifier
- **Purpose**: Orchestrates all classification stages
- **Key Classes**:
  - `RuleBasedFilter` - Pattern matching, sentiment analysis, overrides
  - `MultiStageClassifier` - Main orchestrator (combines all stages)
- **Key Methods**:
  - `classify(text)` - Complete classification pipeline
  - `_get_model_predictions(text)` - BERT model inference
  - `_generate_analysis_details()` - Creates detailed output

#### **`bert_classifier.py`** - BERT Model
- **Purpose**: BERT neural network definition and loading
- **Key Classes**:
  - `BERTMentalHealthClassifier` - PyTorch BERT model (6 output classes)
  - `TemperatureScaling` - Calibration for confidence scores
  - `MentalHealthClassifierService` - Service wrapper for BERT
- **Model Architecture**:
  ```
  Input Text → BERT Encoder → Pooled Output (768 dim)
    → FC1 (384 dim) → FC2 (192 dim) → FC3 (6 classes)
    → Sigmoid → Probabilities
  ```

---

### 🎓 **Training Files**

```
backend/
├── generate_clean_balanced_data.py    # Generate training dataset
├── train_clean_balanced.py            # Train BERT model
├── train_comprehensive_fixed.py       # Advanced training (Focal Loss, etc.)
├── train_massive.py                   # Large-scale training
└── retrain_comprehensive.py           # Full retraining pipeline
```

#### **`generate_clean_balanced_data.py`**
- **Purpose**: Creates balanced training dataset
- **Output**: `clean_balanced_train.json`, `clean_balanced_val.json`
- **Categories**: 500 positive, 500 neutral, 500 stress, 500 distress, 300 self-harm low, 300 self-harm high, 300 unsafe
- **Total**: 2,900 examples

#### **`train_clean_balanced.py`**
- **Purpose**: Trains BERT with strict thresholds
- **Features**:
  - Strict minimum thresholds (self_harm_high: 0.75+)
  - 20 epochs
  - Class weights for balance
  - Optimal threshold finding
- **Output**: `checkpoints/best_clean_balanced_model.pt`

---

### 🔧 **Supporting Files**

```
backend/
├── llm_verifier.py                   # LLM-based verification
├── gpt_llm_classifier.py             # GPT/OpenAI classifier
├── hybrid_classifier.py              # Hybrid Rules + LLM + BERT
└── model_calibrated.py               # Calibrated model loader
```

---

## 🔄 System Architecture Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER REQUEST                              │
│              POST /api/classify {"text": "..."}                  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                         app.py                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  initialize_classifier()                                 │  │
│  │    → Loads MultiStageClassifier                          │  │
│  │    → Loads BERT model from checkpoint                    │  │
│  │    → Initializes RuleBasedFilter                          │  │
│  │    → Initializes LLM Verifier                             │  │
│  └──────────────────────────────────────────────────────────┘  │
│                           │                                      │
│                           ▼                                      │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  classify_text_endpoint()                                 │  │
│  │    → classifier_service.classify(text)                    │  │
│  └──────────────────────────────────────────────────────────┘  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              multistage_classifier.py                            │
│                    MultiStageClassifier                          │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  STAGE 1: Rule-Based Sentiment Analysis                 │  │
│  │    → RuleBasedFilter.analyze_sentiment(text)            │  │
│  │    → Returns: ('positive'|'neutral'|'negative', conf)   │  │
│  └──────────────────────────────────────────────────────────┘  │
│                           │                                      │
│                           ▼                                      │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  STAGE 2: BERT Model Predictions                         │  │
│  │    → _get_model_predictions(text)                        │  │
│  │    → Tokenize text                                       │  │
│  │    → BERT forward pass                                   │  │
│  │    → Sigmoid → Probabilities for 6 classes               │  │
│  └──────────────────────────────────────────────────────────┘  │
│                           │                                      │
│                           ▼                                      │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  STAGE 3: Rule-Based Overrides                           │  │
│  │    → RuleBasedFilter.check_override(text, predictions)   │  │
│  │    → Checks for:                                         │  │
│  │      • Achievement patterns ("I know I can achieve")    │  │
│  │      • Relationship patterns ("I will marry her")        │  │
│  │      • Confident patterns ("I am unstoppable")          │  │
│  │      • Frustration patterns ("I'm sick of...")          │  │
│  │      • Self-harm patterns                                │  │
│  │      • Threats to others                                 │  │
│  │    → If override found: Force classification             │  │
│  └──────────────────────────────────────────────────────────┘  │
│                           │                                      │
│                           ▼                                      │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  STAGE 4: Score Suppression & Adjustment                 │  │
│  │    → Suppress risk scores for positive/neutral           │  │
│  │    → Boost appropriate categories                         │  │
│  │    → Apply strict thresholds                             │  │
│  └──────────────────────────────────────────────────────────┘  │
│                           │                                      │
│                           ▼                                      │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  STAGE 5: Final Classification                            │  │
│  │    → Apply thresholds to probabilities                   │  │
│  │    → Select highest confidence label                     │  │
│  │    → Generate detailed analysis                          │  │
│  └──────────────────────────────────────────────────────────┘  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                      JSON RESPONSE                                │
│  {                                                               │
│    "emotion": "positive",                                        │
│    "sentiment": "safe",                                          │
│    "all_scores": {...},                                         │
│    "predictions": [...],                                         │
│    "analysis": {...}                                            │
│  }                                                               │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔍 Classification Pipeline (Detailed)

### **Stage 1: Rule-Based Sentiment Analysis**

**File**: `multistage_classifier.py` → `RuleBasedFilter.analyze_sentiment()`

**Process**:
1. **Check Achievement Patterns** (Highest Priority)
   - Patterns: "I know I can achieve", "I believe I can", "I have the ability to"
   - If found → Return `('positive', 0.98)`

2. **Check Relationship Patterns**
   - Patterns: "I will marry her", "I love you", "commitment"
   - If found → Return `('positive', 0.98)`

3. **Check Confident Patterns**
   - Patterns: "I am unstoppable", "I can overcome obstacles"
   - If found → Return `('positive', 0.95)`

4. **Check Hostile/Threat Patterns**
   - Patterns: "I will hurt you", "I want to kill them"
   - If found → Return `('negative', 0.9)`

5. **Check Self-Harm Patterns**
   - Patterns: "I want to kill myself", "I'm planning suicide"
   - If found → Return `('negative', 0.95)`

6. **Count Keywords**
   - Positive keywords: "love", "happy", "great", "wonderful", etc.
   - Negative keywords: "hate", "sad", "angry", "terrible", etc.
   - Calculate sentiment based on counts

7. **Default**
   - If no clear signal → Return `('neutral', 0.5)`

---

### **Stage 2: BERT Model Predictions**

**File**: `multistage_classifier.py` → `MultiStageClassifier._get_model_predictions()`
**Model File**: `bert_classifier.py` → `BERTMentalHealthClassifier`

**Process**:
1. **Tokenization**
   ```python
   tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
   encoding = tokenizer(text, max_length=128, padding='max_length', truncation=True)
   ```

2. **BERT Forward Pass**
   ```
   Input IDs → BERT Encoder → Pooled Output (768 dim)
   ```

3. **Neural Network Layers**
   ```
   Pooled Output (768)
     → Dropout (0.3)
     → LayerNorm
     → FC1 (768 → 384) + ReLU + Dropout
     → FC2 (384 → 192) + ReLU + Dropout
     → FC3 (192 → 6) → Logits
   ```

4. **Temperature Scaling** (if calibrated)
   ```
   Logits → Temperature Scaling → Calibrated Logits
   ```

5. **Sigmoid Activation**
   ```
   Logits → Sigmoid → Probabilities [0.0 - 1.0] for each class
   ```

6. **Output**
   ```python
   {
     'neutral': 0.85,
     'stress': 0.20,
     'emotional_distress': 0.15,
     'self_harm_low': 0.05,
     'self_harm_high': 0.02,
     'unsafe_environment': 0.03
   }
   ```

---

### **Stage 3: Rule-Based Overrides**

**File**: `multistage_classifier.py` → `RuleBasedFilter.check_override()`

**Override Types**:

#### **Override -2: Achievement/Confidence Statements**
- **Pattern**: "I know I can achieve", "I believe I can"
- **Action**: Force `emotion='positive'`, suppress ALL risk scores to 0%

#### **Override -1: Positive Relationship Statements**
- **Pattern**: "I will marry her", "I love you"
- **Action**: Force `emotion='positive'`, suppress risk scores to 0.01-0.05%

#### **Override 0: Confident/Empowered Statements**
- **Pattern**: "I am unstoppable", "I can overcome obstacles"
- **Action**: Force `emotion='positive'`, suppress risk scores to 0%

#### **Override 1: Frustration/Annoyance**
- **Pattern**: "I'm sick of...", "I can't stand..."
- **Action**: Force `emotion='stress'`, route to `emotional_distress`, suppress self-harm

#### **Override 2: Pure Neutral Activities**
- **Pattern**: "The meeting is moved", "Tomorrow is a holiday"
- **Action**: Force `emotion='neutral'`, suppress ALL risk scores to 0%

#### **Override 3: Threats to Others**
- **Pattern**: "I will hurt you", "I want to kill them"
- **Action**: Force `unsafe_environment=1.0`, suppress self-harm

#### **Override 4: Self-Harm (High Risk)**
- **Pattern**: "I want to kill myself", "I have a plan to end my life"
- **Action**: Force `self_harm_high=1.0`

#### **Override 5: Self-Harm (Low Risk)**
- **Pattern**: "I sometimes think about suicide", "I wonder if anyone would notice"
- **Action**: Force `self_harm_low=1.0`

---

### **Stage 4: Score Suppression & Adjustment**

**File**: `multistage_classifier.py` → `MultiStageClassifier.classify()`

**Suppression Logic**:

1. **Achievement Statements** (Stage 4)
   ```python
   if has_achievement_pattern:
       model_scores['stress'] = 0.0
       model_scores['emotional_distress'] = 0.0
       model_scores['self_harm_high'] = 0.0
       model_scores['self_harm_low'] = 0.0
       model_scores['unsafe_environment'] = 0.0
       model_scores['neutral'] = 0.95
   ```

2. **Relationship Statements** (Stage 4a)
   ```python
   if has_relationship_pattern:
       model_scores['stress'] = min(model_scores['stress'], 0.05)
       model_scores['self_harm_high'] = min(model_scores['self_harm_high'], 0.01)
       model_scores['neutral'] = 0.9
   ```

3. **Confident Statements** (Stage 4b)
   ```python
   if has_confident_pattern:
       model_scores['stress'] = 0.0
       model_scores['self_harm_high'] = 0.0
       model_scores['neutral'] = 0.9
   ```

4. **Neutral Activities** (Stage 4c)
   ```python
   if has_neutral_activity and base_sentiment == 'neutral':
       model_scores['stress'] = 0.0
       model_scores['self_harm_high'] = 0.0
       model_scores['neutral'] = 0.9
   ```

5. **Sentiment-Based Adjustment**
   ```python
   if base_sentiment == 'positive' and confidence >= 0.7:
       # Suppress low-confidence risk predictions
       for risk_label in ['self_harm_high', 'self_harm_low', 'unsafe_environment']:
           if model_scores[risk_label] < 0.75:
               model_scores[risk_label] = 0.0
   ```

---

### **Stage 5: Final Classification**

**File**: `multistage_classifier.py` → `MultiStageClassifier.classify()`

**Process**:

1. **Apply Thresholds**
   ```python
   thresholds = {
       'self_harm_high': 0.85,      # Very strict
       'self_harm_low': 0.75,       # Strict
       'unsafe_environment': 0.75,  # Strict
       'emotional_distress': 0.60,  # Medium-high
       'stress': 0.55,              # Medium
       'neutral': 0.45               # Lower
   }
   
   # Only predict label if probability >= threshold
   predictions = []
   for label, score in model_scores.items():
       if score >= thresholds[label]:
           predictions.append({
               'label': label,
               'confidence': score,
               'threshold': thresholds[label]
           })
   ```

2. **Select Primary Emotion**
   ```python
   if predictions:
       primary_emotion = max(predictions, key=lambda x: x['confidence'])['label']
   else:
       primary_emotion = 'neutral'  # Default
   ```

3. **Determine Sentiment**
   ```python
   if primary_emotion in ['self_harm_high', 'self_harm_low', 'unsafe_environment']:
       sentiment = 'unsafe'
   elif primary_emotion in ['stress', 'emotional_distress']:
       sentiment = 'concerning'
   else:
       sentiment = 'safe'
   ```

4. **Generate Analysis**
   ```python
   analysis = {
       'base_sentiment': base_sentiment,
       'sentiment_confidence': sent_confidence,
       'model_predictions': predictions,
       'override_applied': override_applied,
       'llm_insight': llm_insight,
       'risk_level': calculate_risk_level(predictions)
   }
   ```

5. **Return Result**
   ```python
   return {
       'text': text,
       'emotion': primary_emotion,
       'sentiment': sentiment,
       'all_scores': model_scores,
       'predictions': predictions,
       'analysis': analysis
   }
   ```

---

## 🎓 Training Pipeline

### **Step 1: Generate Training Data**

**File**: `generate_clean_balanced_data.py`

```python
# Generate balanced dataset
generate_clean_balanced_data()
  → Creates clean_balanced_train.json (2,320 examples)
  → Creates clean_balanced_val.json (580 examples)
```

**Data Structure**:
```json
{
  "text": "I know I can achieve anything I put my mind to",
  "labels": {
    "neutral": 1,
    "stress": 0,
    "unsafe_environment": 0,
    "emotional_distress": 0,
    "self_harm_low": 0,
    "self_harm_high": 0
  }
}
```

---

### **Step 2: Train Model**

**File**: `train_clean_balanced.py`

**Training Process**:
1. **Load Data**
   ```python
   train_dataset = CleanDataset('clean_balanced_train.json', tokenizer)
   val_dataset = CleanDataset('clean_balanced_val.json', tokenizer)
   ```

2. **Initialize Model**
   ```python
   model = BERTMentalHealthClassifier(n_classes=6, dropout=0.3)
   ```

3. **Training Loop** (20 epochs)
   ```python
   for epoch in range(20):
       # Train
       for batch in train_loader:
           logits = model(input_ids, attention_mask)
           loss = BCEWithLogitsLoss(logits, labels)
           loss.backward()
           optimizer.step()
       
       # Validate
       predictions = model(validation_data)
       f1_score = calculate_f1(predictions, true_labels)
       
       # Find optimal thresholds
       optimal_thresholds = find_optimal_thresholds_strict(
           predictions, true_labels,
           min_thresholds={
               'self_harm_high': 0.75,
               'self_harm_low': 0.70,
               ...
           }
       )
       
       # Save best model
       if f1_score > best_f1:
           torch.save({
               'model_state_dict': model.state_dict(),
               'optimal_thresholds': optimal_thresholds,
               'f1_score': f1_score
           }, 'checkpoints/best_clean_balanced_model.pt')
   ```

4. **Output**
   - Model: `checkpoints/best_clean_balanced_model.pt`
   - Contains: model weights, optimal thresholds, F1 score

---

## 🌐 API Endpoints

### **Base URL**: `http://localhost:5000`

### **1. Health Check**
```
GET /api/health
```
**Response**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2024-01-01T12:00:00"
}
```

---

### **2. Classify Text**
```
POST /api/classify
Content-Type: application/json

{
  "text": "I know I can achieve anything I put my mind to"
}
```

**Response**:
```json
{
  "text": "I know I can achieve anything I put my mind to",
  "emotion": "positive",
  "sentiment": "safe",
  "all_scores": {
    "neutral": 0.95,
    "stress": 0.0,
    "emotional_distress": 0.0,
    "self_harm_low": 0.0,
    "self_harm_high": 0.0,
    "unsafe_environment": 0.0
  },
  "predictions": [
    {
      "label": "neutral",
      "confidence": 0.95,
      "threshold": 0.45
    }
  ],
  "analysis": {
    "base_sentiment": "positive",
    "sentiment_confidence": 0.98,
    "override_applied": true,
    "override_reason": "Rule: Achievement/Confidence statement"
  },
  "timestamp": "2024-01-01T12:00:00"
}
```

---

### **3. Batch Classify**
```
POST /api/batch-classify
Content-Type: application/json

{
  "texts": [
    "I know I can achieve anything",
    "Tomorrow is a holiday",
    "I feel sad"
  ]
}
```

**Response**:
```json
{
  "results": [
    { "emotion": "positive", ... },
    { "emotion": "neutral", ... },
    { "emotion": "emotional_distress", ... }
  ],
  "count": 3,
  "timestamp": "2024-01-01T12:00:00"
}
```

---

### **4. Get History**
```
GET /api/history?limit=50
```

**Response**:
```json
{
  "history": [
    { "text": "...", "emotion": "...", ... },
    ...
  ],
  "total": 100
}
```

---

## 📊 Data Flow Diagrams

### **Complete Request Flow**

```
User Input: "I know I can achieve anything I put my mind to"
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│ app.py: classify_text_endpoint()                       │
│   → Validates input                                    │
│   → Calls classifier_service.classify(text)             │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│ multistage_classifier.py: MultiStageClassifier.classify()│
│                                                          │
│   STAGE 1: Rule-Based Sentiment                        │
│   ┌──────────────────────────────────────────────┐     │
│   │ RuleBasedFilter.analyze_sentiment()          │     │
│   │   → Detects: "I know I can achieve"         │     │
│   │   → Returns: ('positive', 0.98)             │     │
│   └──────────────────────────────────────────────┘     │
│                     │                                    │
│                     ▼                                    │
│   STAGE 2: BERT Model Predictions                      │
│   ┌──────────────────────────────────────────────┐     │
│   │ _get_model_predictions()                      │     │
│   │   → Tokenize text                            │     │
│   │   → BERT forward pass                        │     │
│   │   → Returns: {                                │     │
│   │       'neutral': 0.85,                        │     │
│   │       'stress': 0.20,                         │     │
│   │       'self_harm_high': 0.15  ← WRONG!      │     │
│   │     }                                         │     │
│   └──────────────────────────────────────────────┘     │
│                     │                                    │
│                     ▼                                    │
│   STAGE 3: Rule-Based Override                         │
│   ┌──────────────────────────────────────────────┐     │
│   │ RuleBasedFilter.check_override()            │     │
│   │   → Detects: Achievement pattern            │     │
│   │   → Returns: {                               │     │
│   │       'emotion': 'positive',                │     │
│   │       'sentiment': 'safe',                  │     │
│   │       'override_reason': 'Achievement...'   │     │
│   │     }                                        │     │
│   └──────────────────────────────────────────────┘     │
│                     │                                    │
│                     ▼                                    │
│   STAGE 4: Score Suppression                           │
│   ┌──────────────────────────────────────────────┐     │
│   │ Suppress ALL risk scores                      │     │
│   │   → self_harm_high: 0.15 → 0.0               │     │
│   │   → stress: 0.20 → 0.0                        │     │
│   │   → neutral: 0.85 → 0.95                     │     │
│   └──────────────────────────────────────────────┘     │
│                     │                                    │
│                     ▼                                    │
│   STAGE 5: Final Classification                        │
│   ┌──────────────────────────────────────────────┐     │
│   │ Apply thresholds                             │     │
│   │   → neutral: 0.95 >= 0.45 ✓                 │     │
│   │   → self_harm_high: 0.0 < 0.85 ✗            │     │
│   │   → Primary: 'neutral'                       │     │
│   │   → Sentiment: 'safe'                         │     │
│   └──────────────────────────────────────────────┘     │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│ Return JSON Response                                     │
│ {                                                        │
│   "emotion": "neutral",                                 │
│   "sentiment": "safe",                                   │
│   "all_scores": {                                        │
│     "neutral": 0.95,                                    │
│     "self_harm_high": 0.0,  ← FIXED!                   │
│     ...                                                  │
│   }                                                      │
│ }                                                        │
└─────────────────────────────────────────────────────────┘
```

---

### **Model Architecture Diagram**

```
┌─────────────────────────────────────────────────────────────┐
│                    Input Text                               │
│         "I know I can achieve anything"                     │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              BERT Tokenizer                                  │
│  [CLS] i know i can achieve anything [SEP] [PAD] ...        │
│  → Token IDs: [101, 1045, 2338, ...]                       │
│  → Attention Mask: [1, 1, 1, ..., 0, 0, 0]                 │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              BERT Encoder (bert-base-uncased)                │
│  ┌────────────────────────────────────────────────────┐     │
│  │ 12 Transformer Layers                              │     │
│  │   → Self-Attention                                 │     │
│  │   → Feed-Forward                                   │     │
│  │   → Layer Normalization                            │     │
│  └────────────────────────────────────────────────────┘     │
│  → Pooled Output: [768-dimensional vector]                  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Neural Network Head                             │
│  ┌────────────────────────────────────────────────────┐     │
│  │  FC1: 768 → 384                                     │     │
│  │    → ReLU + Dropout(0.3)                           │     │
│  └────────────────────────────────────────────────────┘     │
│                       │                                      │
│                       ▼                                      │
│  ┌────────────────────────────────────────────────────┐     │
│  │  FC2: 384 → 192                                     │     │
│  │    → ReLU + Dropout(0.15)                           │     │
│  └────────────────────────────────────────────────────┘     │
│                       │                                      │
│                       ▼                                      │
│  ┌────────────────────────────────────────────────────┐     │
│  │  FC3: 192 → 6                                      │     │
│  │    → Logits: [-0.5, 0.3, -0.2, 0.1, -0.8, -0.3]   │     │
│  └────────────────────────────────────────────────────┘     │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Temperature Scaling (if calibrated)             │
│  Logits / Temperature → Calibrated Logits                    │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Sigmoid Activation                              │
│  [0.38, 0.57, 0.45, 0.52, 0.31, 0.43]                      │
│  → Probabilities for each class                             │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Output Probabilities                            │
│  {                                                           │
│    'neutral': 0.38,                                         │
│    'stress': 0.57,                                          │
│    'unsafe_environment': 0.45,                              │
│    'emotional_distress': 0.52,                              │
│    'self_harm_low': 0.31,                                   │
│    'self_harm_high': 0.43                                   │
│  }                                                           │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔑 Key Concepts

### **1. Multi-Stage Classification**
- **Why**: Combines rule-based safety with ML accuracy
- **How**: Rules catch edge cases, BERT provides semantic understanding

### **2. Strict Thresholds**
- **Why**: Prevents false positives (e.g., positive statements flagged as self-harm)
- **How**: Risk categories require high confidence (0.75+) to trigger

### **3. Score Suppression**
- **Why**: BERT can misclassify positive statements as risky
- **How**: Rule-based patterns detect positive/neutral content and force risk scores to 0%

### **4. Pattern Detection Priority**
1. Achievement/Confidence (highest)
2. Relationship/Love
3. Confident/Empowered
4. Frustration/Annoyance
5. Neutral Activities
6. Threats to Others
7. Self-Harm

---

## 📝 Summary

This system uses a **hybrid approach** combining:
- **BERT** for semantic understanding
- **Rules** for safety and edge cases
- **Strict thresholds** to prevent false positives
- **Score suppression** to correct misclassifications

The result is a robust classifier that:
- ✅ Correctly identifies positive statements
- ✅ Correctly identifies neutral statements
- ✅ Accurately detects real risk
- ✅ Prevents false alarms

---

## 🚀 Quick Start

1. **Start API Server**:
   ```bash
   cd backend
   python app.py
   ```

2. **Test Classification**:
   ```bash
   curl -X POST http://localhost:5000/api/classify \
     -H "Content-Type: application/json" \
     -d '{"text": "I know I can achieve anything"}'
   ```

3. **Train New Model**:
   ```bash
   python generate_clean_balanced_data.py
   python train_clean_balanced.py
   ```

---

**Documentation Version**: 1.0  
**Last Updated**: 2024  
**Model Version**: Clean Balanced v1.0

