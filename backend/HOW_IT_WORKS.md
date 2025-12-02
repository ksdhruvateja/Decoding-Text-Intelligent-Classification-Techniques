# How the Current Model Works

## 🏗️ System Architecture

```
User Input (Text)
       ↓
Flask API (app.py)
       ↓
MultiStageClassifier (multistage_classifier.py)
       ↓
┌─────────────────────────────────────────────┐
│  Multi-Stage Classification Pipeline        │
└─────────────────────────────────────────────┘
       ↓
   Result (JSON)
```

## 📊 Current Classification Flow

### **Stage 1: Rule-Based Sentiment Analysis** 
**File**: `multistage_classifier.py` → `RuleBasedFilter.analyze_sentiment()`

**What it does:**
- Analyzes text to determine base sentiment: `positive`, `neutral`, or `negative`
- Uses keyword matching and pattern recognition
- Checks for:
  - ✅ Positive keywords: "love", "amazing", "grateful", "confident", "empowered", "marry"
  - ✅ Negative keywords: "hate", "terrible", "frustrated", "sick of"
  - ✅ Hostile patterns: "get lost", "shut up", insults
  - ✅ Self-harm patterns: "kill myself", "hurt myself"
  - ✅ Relationship patterns: "I will marry her", "I will love her"
  - ✅ Confident patterns: "I am unstoppable", "I will rise above"

**Output**: `(sentiment, confidence)` e.g., `('positive', 0.95)`

---

### **Stage 2: BERT Model Predictions**
**File**: `multistage_classifier.py` → `MultiStageClassifier._get_model_predictions()`

**What it does:**
- Loads trained BERT model from `checkpoints/best_calibrated_model_temp.pt` (or fallback)
- Tokenizes text using BERT tokenizer
- Runs through BERT neural network
- Gets probability scores for 6 categories:
  - `neutral`
  - `stress`
  - `unsafe_environment`
  - `emotional_distress`
  - `self_harm_low`
  - `self_harm_high`

**Output**: Dictionary of scores like:
```python
{
  'neutral': 0.85,
  'stress': 0.12,
  'unsafe_environment': 0.05,
  'emotional_distress': 0.08,
  'self_harm_low': 0.02,
  'self_harm_high': 0.01
}
```

---

### **Stage 3: Rule-Based Overrides** (HIGHEST PRIORITY)
**File**: `multistage_classifier.py` → `RuleBasedFilter.check_override()`

**What it does:**
Checks for specific patterns and **overrides** model predictions if needed:

1. **Positive Relationship Override** (Priority -1)
   - Detects: "I will marry her", "I will love her", etc.
   - **Action**: Force `emotion: positive`, `sentiment: safe`, suppress ALL risk scores
   - **Why**: Prevents false positives on relationship statements

2. **Confident/Empowered Override** (Priority 0)
   - Detects: "I am unstoppable", "I will rise above obstacles", etc.
   - **Action**: Force `emotion: positive`, suppress risk scores
   - **Why**: Prevents false positives on empowering statements

3. **General Positive Override** (Priority 1)
   - Detects: Strong positive sentiment with high confidence
   - **Action**: Suppress crisis labels, classify as safe
   - **Why**: Prevents false positives on positive content

4. **Frustration Override** (Priority 2)
   - Detects: "I'm sick of...", "I'm frustrated with...", etc.
   - **Action**: Classify as `stress`/`emotional_distress`, NOT self-harm
   - **Why**: Prevents misclassification of frustration as self-harm

5. **Self-Harm Override** (Priority 2b)
   - Detects: Direct self-harm patterns
   - **Action**: Force high-risk classification
   - **Why**: Ensures dangerous statements are caught

**Output**: Override dict or `None` (if no override needed)

---

### **Stage 4: Score Suppression & Adjustment**
**File**: `multistage_classifier.py` → `MultiStageClassifier.classify()`

**What it does:**
- Suppresses risk scores for confident/empowered statements
- Suppresses risk scores for positive relationship statements
- Boosts stress/emotional_distress for frustration statements
- Adjusts scores based on sentiment analysis

**Example**:
```python
# If "I am unstoppable" detected:
model_scores['self_harm_high'] = min(model_scores['self_harm_high'], 0.05)  # Suppress
model_scores['stress'] = min(model_scores['stress'], 0.1)  # Suppress
model_scores['neutral'] = max(model_scores['neutral'], 0.8)  # Boost
```

---

### **Stage 5: Threshold Application**
**File**: `multistage_classifier.py` → `MultiStageClassifier.classify()`

**What it does:**
- Applies category-specific thresholds:
  - `self_harm_high`: 0.80 (very high bar)
  - `self_harm_low`: 0.70 (high bar)
  - `unsafe_environment`: 0.70 (high bar)
  - `emotional_distress`: 0.55 (medium-high)
  - `stress`: 0.50 (medium)
  - `neutral`: 0.40 (lower bar)

- Only includes predictions above threshold in final result

**Output**: List of predictions like:
```python
[
  {'label': 'neutral', 'confidence': 0.85, 'threshold': 0.40, 'source': 'model'},
  {'label': 'stress', 'confidence': 0.56, 'threshold': 0.50, 'source': 'model'}
]
```

---

### **Stage 6: Final Classification Decision**
**File**: `multistage_classifier.py` → `MultiStageClassifier.classify()`

**What it does:**
- Determines final `emotion` and `sentiment` based on:
  1. Override (if applied)
  2. Top prediction (if any)
  3. Base sentiment + scores (if no predictions)

**Logic**:
```python
if override_applied:
    emotion = override['emotion']
    sentiment = override['sentiment']
elif predictions:
    top_label = predictions[0]['label']
    emotion = top_label
    # Map to sentiment based on label
else:
    # Use base sentiment
    if base_sentiment == 'positive':
        emotion = 'positive'
        sentiment = 'safe'
    elif base_sentiment == 'negative':
        emotion = 'stress'  # Default for negative
        sentiment = 'concerning'
    else:
        emotion = 'neutral'
        sentiment = 'safe'
```

---

## 🔄 Complete Example Flow

### Example 1: "I will marry her"

```
Input: "I will marry her"
       ↓
Stage 1: Sentiment Analysis
  → Detects "marry" → positive relationship keyword
  → Returns: ('positive', 0.98)
       ↓
Stage 2: BERT Predictions
  → Model might incorrectly predict: self_harm_high: 0.79
       ↓
Stage 3: Rule-Based Override
  → Detects relationship pattern: "I will marry her"
  → OVERRIDE APPLIED!
  → Returns: {
      'emotion': 'positive',
      'sentiment': 'safe',
      'confidence': 0.98,
      'override_reason': 'Positive relationship statement'
    }
       ↓
Stage 4-6: SKIPPED (override takes priority)
       ↓
Final Result:
{
  'emotion': 'positive',
  'sentiment': 'safe',
  'all_scores': {...},  # Suppressed risk scores
  'predictions': [{'label': 'positive', ...}],
  'override_applied': True
}
```

### Example 2: "I'm sick of people wasting my time"

```
Input: "I'm sick of people wasting my time"
       ↓
Stage 1: Sentiment Analysis
  → Detects "sick of" → negative sentiment
  → Returns: ('negative', 0.75)
       ↓
Stage 2: BERT Predictions
  → Model might incorrectly predict: self_harm_high: 0.14, unsafe_environment: 0.27
       ↓
Stage 3: Rule-Based Override
  → Checks for frustration patterns
  → Detects: "sick of" → frustration pattern
  → OVERRIDE APPLIED!
  → Returns: {
      'emotion': 'stress',
      'sentiment': 'concerning',
      'confidence': 0.75,
      'override_reason': 'Frustration/annoyance detected'
    }
       ↓
Stage 4: Score Suppression
  → Suppresses self_harm scores
  → Suppresses unsafe_environment scores
  → Boosts stress/emotional_distress
       ↓
Final Result:
{
  'emotion': 'stress',
  'sentiment': 'concerning',
  'all_scores': {
    'stress': 0.57,
    'emotional_distress': 0.16,
    'self_harm_high': 0.01,  # Suppressed
    'unsafe_environment': 0.05  # Suppressed
  },
  'predictions': [{'label': 'stress', ...}]
}
```

### Example 3: "I want to kill myself"

```
Input: "I want to kill myself"
       ↓
Stage 1: Sentiment Analysis
  → Detects self-harm pattern: "kill myself"
  → Returns: ('negative', 1.0)
       ↓
Stage 2: BERT Predictions
  → Model correctly predicts: self_harm_high: 0.95
       ↓
Stage 3: Rule-Based Override
  → Detects self-harm pattern
  → OVERRIDE APPLIED!
  → Returns: {
      'emotion': 'self_harm_high',
      'sentiment': 'high_risk',
      'confidence': 0.95
    }
       ↓
Final Result:
{
  'emotion': 'self_harm_high',
  'sentiment': 'high_risk',
  'all_scores': {...},
  'predictions': [{'label': 'self_harm_high', 'confidence': 0.95}]
}
```

---

## 🎯 Key Features

### 1. **Multi-Layer Protection**
- Rule-based overrides catch edge cases BEFORE model predictions
- BERT model provides deep learning pattern recognition
- Sentiment analysis provides context

### 2. **False Positive Prevention**
- Whitelist for safe statements (marry, love, confident, etc.)
- Suppression logic reduces risk scores for positive content
- Frustration detection prevents misclassification as self-harm

### 3. **Safety Guarantees**
- Blacklist for dangerous statements (kill myself, etc.)
- High thresholds for risk categories (80% for self_harm_high)
- Override system ensures critical cases are caught

### 4. **Fallback System**
- If BERT model not loaded → uses rule-based only
- If override doesn't apply → uses model predictions
- If no predictions → uses sentiment analysis

---

## 📁 File Structure

```
backend/
├── app.py                          # Flask API (entry point)
├── multistage_classifier.py        # Main classifier (CURRENTLY USED)
│   ├── RuleBasedFilter             # Rule-based sentiment & overrides
│   └── MultiStageClassifier        # Orchestrates all stages
├── bert_classifier.py              # BERT model loader
├── rule_classifier.py              # Fallback rule-based classifier
├── hybrid_classifier.py            # Future: LLM + BERT + Rules
└── checkpoints/
    └── best_calibrated_model_temp.pt  # Trained BERT model
```

---

## 🔧 Current Limitations

1. **BERT Model**: May not be fully trained on all edge cases
   - **Solution**: Run `python retrain_comprehensive.py`

2. **No LLM**: Currently only uses BERT + rules
   - **Solution**: Implement `hybrid_classifier.py` for LLM support

3. **Rule-Based**: Some edge cases might not be covered
   - **Solution**: Add more patterns as needed

---

## ✅ What Works Well

- ✅ Rule-based overrides catch most false positives
- ✅ Sentiment analysis provides good context
- ✅ Multi-stage approach provides multiple safety layers
- ✅ Override system ensures critical cases are handled

---

## 🚀 To Improve Accuracy

1. **Retrain BERT model** with comprehensive data:
   ```bash
   python retrain_comprehensive.py
   ```

2. **Add LLM fine-tuning** (optional, advanced):
   ```bash
   python prepare_llm_finetuning_data.py
   python llm_finetune_classifier.py
   python rlhf_tune_classifier.py
   ```

3. **Use hybrid classifier** (combines all approaches):
   ```python
   from hybrid_classifier import HybridClassifier
   classifier = HybridClassifier()
   ```

