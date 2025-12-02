# Threats to Others vs Self-Harm - Classification Guide

## 🎯 Critical Distinction

**DO NOT confuse threats toward others with self-harm.**

The system must clearly distinguish between:
- **Threats to OTHERS** → `unsafe_environment` (concerning)
- **Threats to SELF** → `self_harm_high` or `self_harm_low` (high_risk/concerning)

## 📊 Classification Rules

### Threats to OTHERS → `unsafe_environment`

**Patterns:**
- "I want to hurt **you/them/him/her/someone/people**"
- "I'm going to kill **you/them/him/her/someone**"
- "I will attack **you/them/him/her/someone**"
- "I'm planning to hurt **people/others**"
- "I hate **you/them/him/her/someone**"

**Key Words:**
- Targets: you, them, him, her, someone, people, others
- Actions: hurt, kill, attack, get (revenge), threaten

**Classification:**
- Emotion: `unsafe_environment`
- Sentiment: `concerning`
- Risk Level: Medium-High (external threat)

### Threats to SELF → `self_harm_high` or `self_harm_low`

**Patterns:**
- "I want to hurt **myself**"
- "I'm going to kill **myself**"
- "I want to end **my life**"
- "I'm planning to harm **myself**"
- "I want to die"

**Key Words:**
- Targets: myself, my life, myself
- Actions: hurt myself, kill myself, harm myself, suicide, end my life

**Classification:**
- Emotion: `self_harm_high` (if active intent) or `self_harm_low` (if passive ideation)
- Sentiment: `high_risk` (high) or `concerning` (low)
- Risk Level: High (self-harm)

## 🔍 Detection Logic

### Rule-Based Detection

The system uses regex patterns to detect:

1. **Threats to Others Patterns:**
   ```python
   r'\b(want to|going to|plan to|will)\s+(hurt|kill|harm|attack|get)\s+(you|them|him|her|someone|people|others)\b'
   r'\b(i (want|will|am going) to (hurt|kill|harm|attack|get) (you|them|him|her|someone|people))\b'
   ```

2. **Self-Harm Patterns:**
   ```python
   r'\b(want to|going to|plan to)\s+(die|kill myself|hurt myself|end (my )?life)\b'
   r'\b(suicide|kill myself|end it all|harm myself)\b'
   ```

### Priority Order

1. **Check for threats to others FIRST**
   - If detected → `unsafe_environment`
   - Do NOT classify as self-harm

2. **Then check for self-harm**
   - If detected → `self_harm_high` or `self_harm_low`
   - Only if no threats to others

3. **Prevent confusion**
   - Remove self-harm predictions if threats to others detected
   - Ensure correct classification

## ✅ Examples

### Correct Classification

| Statement | Classification | Reason |
|-----------|---------------|--------|
| "I want to hurt you" | `unsafe_environment` / `concerning` | Threat to OTHERS |
| "I'm going to kill them" | `unsafe_environment` / `concerning` | Threat to OTHERS |
| "I will attack someone" | `unsafe_environment` / `concerning` | Threat to OTHERS |
| "I want to hurt myself" | `self_harm_high` / `high_risk` | Threat to SELF |
| "I'm going to kill myself" | `self_harm_high` / `high_risk` | Threat to SELF |
| "I want to end my life" | `self_harm_high` / `high_risk` | Threat to SELF |

### Incorrect Classification (Should NOT Happen)

| Statement | Wrong Classification | Correct Classification |
|-----------|----------------------|----------------------|
| "I want to hurt you" | ❌ `self_harm_high` | ✅ `unsafe_environment` |
| "I'm going to kill them" | ❌ `self_harm_high` | ✅ `unsafe_environment` |
| "I want to hurt myself" | ❌ `unsafe_environment` | ✅ `self_harm_high` |

## 🧪 Testing

Run the test script to verify the distinction:

```bash
cd backend
python test_threats_vs_selfharm.py
```

This tests:
- Threats to others → `unsafe_environment`
- Threats to self → `self_harm_high`
- No confusion between the two

## 🔧 Implementation Details

### Rule-Based Override

The system has explicit overrides:

1. **Threats to Others Override:**
   ```python
   if threats_to_others_detected:
       return {
           'emotion': 'unsafe_environment',
           'sentiment': 'concerning',
           'override_reason': 'Rule: Threats toward others detected (NOT self-harm)'
       }
   ```

2. **Self-Harm Override:**
   ```python
   if self_harm_detected:
       return {
           'emotion': 'self_harm_high',
           'sentiment': 'high_risk',
           'override_reason': 'Rule: Critical self-harm language detected (threats to self)'
       }
   ```

### Model Prediction Correction

Before final classification, the system:
1. Checks for threats to others patterns
2. Checks for self-harm patterns
3. Removes incorrect predictions (e.g., self-harm if threats to others detected)
4. Ensures correct classification

## 📝 Training Data Guidelines

When creating training data:

### Threats to Others Examples:
```json
{
  "text": "I want to hurt you",
  "labels": {
    "neutral": 0,
    "stress": 0,
    "unsafe_environment": 1,  // ✓ Correct
    "emotional_distress": 0,
    "self_harm_low": 0,       // ✗ NOT self-harm
    "self_harm_high": 0       // ✗ NOT self-harm
  }
}
```

### Self-Harm Examples:
```json
{
  "text": "I want to hurt myself",
  "labels": {
    "neutral": 0,
    "stress": 0,
    "unsafe_environment": 0,   // ✗ NOT unsafe_environment
    "emotional_distress": 0,
    "self_harm_low": 0,
    "self_harm_high": 1        // ✓ Correct
  }
}
```

## ⚠️ Common Mistakes to Avoid

1. **Don't classify "I want to hurt you" as self-harm**
   - Target is "you" (others), not "myself"
   - Should be `unsafe_environment`

2. **Don't classify "I want to hurt myself" as unsafe_environment**
   - Target is "myself" (self), not others
   - Should be `self_harm_high`

3. **Don't confuse ambiguous statements**
   - "I want to hurt someone" → `unsafe_environment` (someone = others)
   - "I want to hurt myself" → `self_harm_high` (myself = self)

## 🎯 Summary

**Key Principle:**
- **Target = Others** → `unsafe_environment`
- **Target = Self** → `self_harm_high` or `self_harm_low`

**Detection Order:**
1. Check for threats to others FIRST
2. Then check for self-harm
3. Prevent confusion

**Result:**
- Accurate classification
- No confusion between threats to others and self-harm
- Proper risk assessment

---

**The system now correctly distinguishes threats to others from self-harm!** ✅

