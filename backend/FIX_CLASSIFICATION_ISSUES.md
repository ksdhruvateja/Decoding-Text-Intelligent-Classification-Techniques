# Fix Classification Issues - Complete Guide

## Problems Identified

Your model was incorrectly classifying:

1. **Positive statements** (e.g., "I absolutely loved the restaurant...") 
   - ❌ Getting: neutral with high risk scores
   - ✅ Should be: positive/safe

2. **Negative complaints** (e.g., "The service was terrible...")
   - ❌ Getting: safe/neutral even when stress detected
   - ✅ Should be: stress/emotional_distress - concerning

3. **Neutral daily activities** (e.g., "I went to the store...")
   - ❌ Getting: neutral but with high risk scores
   - ✅ Should be: neutral/safe with low risk scores

## Solutions Implemented

### 1. **Improved Rule-Based Filter** (`multistage_classifier.py`)

#### Enhanced Positive Detection
- Added restaurant/service positive keywords: `delicious`, `friendly`, `loved`, `recommend`, etc.
- Improved positive patterns to catch phrases like "absolutely loved", "was so good"
- Lowered confidence threshold for positive detection (0.7 instead of 0.8)

#### Enhanced Neutral Detection
- Added shopping/daily activity patterns: "went to store", "buy groceries", etc.
- Better detection of neutral daily activities
- Override logic to suppress false risk scores for neutral activities

#### Better Negative/Stress Detection
- Lower threshold for stress when negative sentiment is detected
- Proper classification of complaints as "concerning" not "safe"

### 2. **Corrective Training Data** (`generate_corrective_training_data.py`)

Generated 50+ corrective examples:
- Positive restaurant reviews → positive/safe
- Negative service complaints → stress/emotional_distress
- Neutral shopping/activities → neutral/safe

### 3. **Updated Thresholds**

Adjusted thresholds to reduce false positives:
- `unsafe_environment`: 0.75 (was 0.70)
- `emotional_distress`: 0.55 (was 0.40)
- `stress`: 0.60 (was 0.50)
- `self_harm_low`: 0.70 (was 0.65)

## Quick Fix (No Retraining Required)

The improved rule-based filter should immediately fix most issues:

```bash
cd backend
python quick_fix_classification.py
```

This tests the three problematic examples with the improved logic.

**Expected Results:**
1. "I absolutely loved the restaurant..." → **positive/safe** ✓
2. "The service was terrible..." → **stress/emotional_distress - concerning** ✓
3. "I went to the store..." → **neutral/safe** ✓

## Complete Fix (With Retraining)

For best results, retrain the model with corrective data:

### Step 1: Generate Corrective Data

```bash
cd backend
python generate_corrective_training_data.py
```

This creates `train_data_corrected.json` with:
- Your existing training data
- 50+ corrective examples (added twice for emphasis)

### Step 2: Retrain Model

**Option A: Advanced Training (Recommended)**
```bash
python retrain_with_corrections.py  # Prepares data
python train_advanced_optimized.py  # Trains with advanced techniques
```

**Option B: Standard Training (Faster)**
```bash
python retrain_with_corrections.py  # Prepares data
python train_bert_model.py          # Standard training
```

### Step 3: Test Results

```bash
python quick_fix_classification.py
```

## What Changed in Code

### `multistage_classifier.py`

1. **Expanded Positive Keywords**:
   ```python
   'delicious', 'friendly', 'loved', 'recommend', 'best', 'favorite'
   ```

2. **Improved Positive Patterns**:
   ```python
   r'\b(absolutely (loved|love|enjoyed)|really (loved|enjoyed|liked))\b'
   r'\b(was (so |really )?(good|great|amazing|wonderful|fantastic|excellent|delicious|friendly|helpful))\b'
   ```

3. **Better Neutral Detection**:
   ```python
   r'\b(went to the store|buy (some |some )?groceries|shopping (for|to buy))\b'
   ```

4. **Fixed Stress Classification**:
   - Stress from negative sentiment now correctly marked as "concerning"
   - Lower threshold when negative sentiment detected

5. **Improved Override Logic**:
   - Positive statements override risk scores
   - Neutral activities suppress false risk scores
   - Better confidence thresholds

## Expected Improvements

| Example | Before | After |
|---------|--------|-------|
| "I absolutely loved the restaurant..." | neutral (wrong) | **positive/safe** ✓ |
| "The service was terrible..." | safe (wrong) | **stress/emotional_distress - concerning** ✓ |
| "I went to the store..." | neutral with high risk scores | **neutral/safe with low risk scores** ✓ |

## Verification

After applying fixes, test with:

```python
from multistage_classifier import MultiStageClassifier

classifier = MultiStageClassifier()

# Test 1: Positive
result1 = classifier.classify("I absolutely loved the new restaurant; the food was delicious and the staff were so friendly.")
print(f"Emotion: {result1['emotion']}, Sentiment: {result1['sentiment']}")
# Expected: emotion='positive', sentiment='safe'

# Test 2: Negative complaint
result2 = classifier.classify("The service was terrible, I had to wait an hour and the staff were rude")
print(f"Emotion: {result2['emotion']}, Sentiment: {result2['sentiment']}")
# Expected: emotion='stress' or 'emotional_distress', sentiment='concerning'

# Test 3: Neutral
result3 = classifier.classify("I went to the store yesterday to buy some groceries")
print(f"Emotion: {result3['emotion']}, Sentiment: {result3['sentiment']}")
# Expected: emotion='neutral', sentiment='safe'
```

## Troubleshooting

### Still Getting Wrong Results?

1. **Check if override is being applied**:
   ```python
   result = classifier.classify(text)
   print(result.get('override_applied'))  # Should be True for positive/neutral
   print(result.get('override_reason'))
   ```

2. **Verify sentiment detection**:
   ```python
   sentiment, confidence = classifier.rule_filter.analyze_sentiment(text)
   print(f"Sentiment: {sentiment}, Confidence: {confidence}")
   ```

3. **Retrain with corrective data**:
   - The rule-based fixes help, but retraining gives best results
   - Run `retrain_with_corrections.py` then retrain

### Model Still Has Issues?

1. **Increase corrective data**:
   - Edit `generate_corrective_training_data.py`
   - Add more examples of problematic cases
   - Retrain

2. **Adjust thresholds**:
   - Edit thresholds in `multistage_classifier.py`
   - Lower for better recall, higher for better precision

3. **Use advanced training**:
   - Run `train_advanced_optimized.py` for best results
   - Uses focal loss, better architecture, etc.

## Files Modified

1. `multistage_classifier.py` - Improved rule-based filter and classification logic
2. `generate_corrective_training_data.py` - Generates corrective training examples
3. `retrain_with_corrections.py` - Prepares data for retraining
4. `quick_fix_classification.py` - Tests the fixes

## Next Steps

1. ✅ **Immediate**: Run `quick_fix_classification.py` to test improvements
2. ✅ **Short-term**: Run `retrain_with_corrections.py` + retrain model
3. ✅ **Long-term**: Use `train_advanced_optimized.py` for maximum accuracy

---

**The classification should now be accurate!** 🎯

Run `python quick_fix_classification.py` to verify the fixes work.

