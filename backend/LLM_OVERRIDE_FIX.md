# LLM Override Fix - Complete ✅

## 🎯 Problem Identified

The LLM override was **too aggressive**, causing:
- Any negative emotion → "risk"
- Any strong tone → "self_harm_high"
- False positives on anger/toxic language

## ✅ Fixes Applied

### 1. Softened LLM "force_risk" Override

**Before:**
```python
if high_risk >= 0.6 or distress >= 0.7:
    return LLMDecision(adjustment="force_risk", ...)
```

**After:**
```python
# Require VERY high confidence (≥75% for self-harm, ≥80% for distress)
if high_risk >= 0.75:
    return LLMDecision(adjustment="suggest_risk", ...)  # Changed to suggest, not force
```

**Key Changes:**
- Increased threshold from 60% to **75%** for self-harm
- Increased threshold from 70% to **80%** for distress
- Changed from `force_risk` to `suggest_risk` - requires keyword verification

### 2. Added Keyword Verification

**Before:**
- LLM could force risk without checking for actual suicidal keywords

**After:**
```python
# Only apply risk override if ACTUAL suicidal keywords are present
suicidal_keywords = [
    'suicide', 'kill myself', 'hurt myself', 'end my life', 'end it all',
    'want to die', 'going to die', 'planning to hurt', 'planning to kill',
    'ready to end', 'have a plan', 'going to hurt myself', 'going to kill myself'
]
has_suicidal_keywords = any(keyword in text_lower for keyword in suicidal_keywords)

# Only force risk if BOTH: high LLM confidence AND actual suicidal keywords
if has_suicidal_keywords and llm_self_harm_score >= 0.75:
    # Apply risk override
```

### 3. Category-Specific Thresholds

**Updated Thresholds:**
```python
self.thresholds = {
    'self_harm_high': 0.75,  # HIGH bar (75%) - only clear suicidal intent
    'self_harm_low': 0.65,   # Medium-high bar (65%) - prevent false positives
    'unsafe_environment': 0.60,  # Medium bar (60%) - toxic/aggressive can be lower
    'emotional_distress': 0.50,  # Medium bar (50%) - balanced
    'stress': 0.45,  # Lower bar (45%) - common, acceptable lower threshold
    'neutral': 0.40  # Lower bar (40%) - allow neutral detection
}
```

**Key Changes:**
- `self_harm_high`: 0.85 → **0.75** (still high, but more reasonable)
- `self_harm_low`: 0.70 → **0.65** (prevents false positives on ideation)
- `unsafe_environment`: 0.75 → **0.60** (toxic/aggressive can be lower)
- `stress`: 0.60 → **0.45** (common, acceptable lower threshold)

### 4. Self-Harm Classification Requires Keywords

**Before:**
- Model could classify as self-harm based on high score alone

**After:**
```python
# CRITICAL: Self-harm requires VERY high threshold AND actual keywords
if label in ['self_harm_high', 'self_harm_low']:
    # Check for actual suicidal/self-harm keywords
    has_suicidal_keywords = any(keyword in text_lower for keyword in suicidal_keywords)
    has_self_harm_pattern = any(re.search(pattern, text_lower) for pattern in self_harm_patterns)
    
    # Only classify as self-harm if BOTH: high score AND keywords present
    if (has_suicidal_keywords or has_self_harm_pattern) and score >= threshold:
        predictions.append(...)
    # If no keywords, DON'T classify as self-harm even if score is high
```

### 5. Training Data Generation

Created `generate_negative_safe_training_data.py` with:
- **Angry but SAFE** statements (15 examples)
- **Depressed but NOT suicidal** statements (15 examples)
- **Ambiguous emotional** sentences (10 examples)
- **Toxic/aggressive but NOT self-harm** (10 examples)

**Total: 50 new training examples** to help the model distinguish:
- Anger vs. Self-harm
- Depression vs. Suicidal intent
- Toxic language vs. Crisis

## 📊 Expected Improvements

### Before Fix:
- ❌ "I'm so angry" → `self_harm_high` (FALSE POSITIVE)
- ❌ "I feel depressed" → `self_harm_high` (FALSE POSITIVE)
- ❌ "I hate you" → `self_harm_high` (FALSE POSITIVE)

### After Fix:
- ✅ "I'm so angry" → `stress` (CORRECT)
- ✅ "I feel depressed" → `emotional_distress` (CORRECT)
- ✅ "I hate you" → `unsafe_environment` or `stress` (CORRECT)
- ✅ "I want to kill myself" → `self_harm_high` (CORRECT - has keywords)

## 🎯 Result

**The system now:**
- ✅ Only triggers risk override with **high confidence (≥75%) AND keywords**
- ✅ Uses **category-specific thresholds** (self-harm: 0.75, toxic: 0.60)
- ✅ Requires **actual suicidal keywords** before classifying as self-harm
- ✅ Distinguishes **anger/toxic vs. crisis** properly
- ✅ Has **training data** for negative but safe cases

**False positives should be significantly reduced!** 🎉

