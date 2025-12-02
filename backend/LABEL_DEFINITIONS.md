# 🎯 CLEAR LABEL DEFINITIONS & GUIDELINES

## Critical Issues Found

The audit revealed:
- ❌ **18 CRITICAL mislabeling issues** (self-harm texts marked as low risk)
- ❌ **Very short texts** (avg 5.9 words - not enough context)
- ❌ **Low vocabulary diversity** (23.9% - repetitive phrases)
- ❌ **No label overlap** (single-label problem, not multi-label)

---

## ✅ NEW CLEAR LABEL DEFINITIONS

### 1. **NEUTRAL** (Baseline/No Risk)
**Definition:** Everyday statements with no emotional distress or risk indicators.

**Examples:**
- ✅ "What is the weather like today?"
- ✅ "I need to finish my work project"
- ✅ "Can you help me with this task?"
- ✅ "I'm going to the store later"
- ✅ "The meeting is scheduled for 3pm"

**Key Indicators:**
- Factual statements
- Questions about logistics
- Routine activities
- No emotional language

**NOT Neutral:**
- ❌ "I'm feeling tired" (could be stress)
- ❌ "Everything is overwhelming" (emotional distress)

---

### 2. **STRESS** (Manageable Pressure)
**Definition:** Experiencing temporary pressure or worry about specific situations, but managing.

**Examples:**
- ✅ "I'm stressed about my upcoming exam"
- ✅ "Work deadlines are making me anxious"
- ✅ "I'm worried about meeting my goals"
- ✅ "Feeling pressured by all these responsibilities"
- ✅ "I'm nervous about the presentation tomorrow"

**Key Indicators:**
- Specific stressor mentioned (work, school, deadlines)
- Temporary/situational
- Still functional
- Uses words: stressed, anxious, worried, nervous, pressured

**NOT Stress:**
- ❌ "I can't cope anymore" → emotional_distress
- ❌ "I hate my life" → emotional_distress

---

### 3. **UNSAFE_ENVIRONMENT** (External Threat)
**Definition:** Feeling physically or emotionally unsafe due to external circumstances or people.

**Examples:**
- ✅ "Someone is threatening me at work"
- ✅ "I don't feel safe in my relationship"
- ✅ "I'm being bullied and harassed"
- ✅ "My living situation feels dangerous"
- ✅ "Someone is stalking me online"

**Key Indicators:**
- External threat or danger
- Another person/situation causing harm
- Words: unsafe, threatened, bullied, harassed, stalked, abused
- Focus on ENVIRONMENT, not internal feelings

**NOT Unsafe Environment:**
- ❌ "I feel emotionally unsafe" → emotional_distress
- ❌ "I might hurt myself" → self_harm

---

### 4. **EMOTIONAL_DISTRESS** (Internal Suffering)
**Definition:** Significant internal emotional pain, struggling to cope, but no self-harm ideation.

**Examples:**
- ✅ "I feel completely overwhelmed and can't cope"
- ✅ "I'm emotionally exhausted and breaking down"
- ✅ "Everything feels hopeless and pointless"
- ✅ "I can't stop crying and feel so alone"
- ✅ "I'm drowning in sadness and despair"

**Key Indicators:**
- Intense negative emotions
- Difficulty coping
- Hopelessness, despair, overwhelm
- Internal suffering (not external threat)
- No mention of self-harm

**NOT Emotional Distress:**
- ❌ "I'm sad today" → stress (mild)
- ❌ "I want to hurt myself" → self_harm

---

### 5. **SELF_HARM_LOW** (Passive/Ideation)
**Definition:** Thoughts about self-harm but not active plans. Considering possibility.

**Examples:**
- ✅ "I sometimes think about hurting myself"
- ✅ "I've been having intrusive thoughts of self-harm"
- ✅ "I wonder what it would be like to hurt myself"
- ✅ "Part of me wants to hurt myself but I haven't"
- ✅ "I'm scared I might hurt myself if things get worse"

**Key Indicators:**
- Passive voice: "might", "sometimes", "wondering", "scared I could"
- No immediate intent
- Ideation without plan
- Seeking help/expressing concern

**NOT Low Risk:**
- ❌ "I'm going to hurt myself" → self_harm_high
- ❌ "I have a plan to harm myself" → self_harm_high

---

### 6. **SELF_HARM_HIGH** (Active/Intent)
**Definition:** Active self-harm intent, concrete plans, or past self-harm behavior.

**Examples:**
- ✅ "I'm going to hurt myself tonight"
- ✅ "I have a plan to end my life"
- ✅ "I want to kill myself"
- ✅ "I'm actively planning to harm myself"
- ✅ "I've been cutting myself regularly"
- ✅ "I'm ready to end it all"

**Key Indicators:**
- Active language: "going to", "will", "planning", "ready"
- Specific methods mentioned
- Past self-harm behavior
- Immediate danger
- Suicidal ideation with intent

**NOT High Risk:**
- ❌ "I sometimes think about it" → self_harm_low
- ❌ "I'm worried I might" → self_harm_low

---

## 🎯 CLASSIFICATION DECISION TREE

Use this to decide labels:

```
1. Does text mention self-harm, suicide, or hurting self?
   YES → Go to step 2
   NO → Go to step 3

2. Is there ACTIVE INTENT or PLAN?
   YES → SELF_HARM_HIGH
   NO → Is it passive thoughts/ideation? → SELF_HARM_LOW

3. Does text mention external threat/danger from others?
   YES → UNSAFE_ENVIRONMENT
   NO → Go to step 4

4. Is there significant internal emotional suffering?
   YES → EMOTIONAL_DISTRESS
   NO → Go to step 5

5. Is there temporary/situational pressure or worry?
   YES → STRESS
   NO → NEUTRAL
```

---

## 📊 CORRECTED THRESHOLDS (Based on Severity)

| Category | Threshold | Reasoning |
|----------|-----------|-----------|
| **neutral** | 0.45 | Lower threshold (more permissive) |
| **stress** | 0.40 | Common, lower threshold acceptable |
| **unsafe_environment** | 0.55 | Higher - external threat is serious |
| **emotional_distress** | 0.50 | Medium - significant but not life-threatening |
| **self_harm_low** | 0.60 | Higher - avoid false positives on ideation |
| **self_harm_high** | 0.70 | **HIGHEST - critical category, must be confident** |

---

## 🔧 RELABELING INSTRUCTIONS

### The 18 Critical Mislabeled Samples:

**Current:** Marked as `self_harm_low`  
**Should be:** `self_harm_high`

**Texts to relabel:**
All samples containing:
- "harm myself"
- "hurt myself"  
- "end my pain"
- "having urges to harm"

These show **ACTIVE INTENT**, not passive ideation!

**Action Required:**
1. Load `potential_mislabeling_issues.json`
2. Change all 18 samples from `self_harm_low: 1` to `self_harm_high: 1`
3. Set `self_harm_low: 0` for these samples

---

## 📝 DATA COLLECTION GUIDELINES

### What to Collect:

**1. More Neutral Examples (need 200+ more):**
- Factual questions
- Routine activities
- Logistics inquiries
- Weather/news questions
- No emotional content

**2. More Diverse Stress Examples (need 150+ more):**
- Different stressors: work, school, family, money, health
- Various intensity levels
- Different phrasing styles
- Include context (10-30 words)

**3. Clear Positive Examples (need 100+):**
- "I'm feeling great today!"
- "Everything is going wonderfully"
- "I'm so happy and blessed"
- "Life is amazing right now"
- "I'm excited about the future"

**4. Severe Self-Harm Examples (need 100+ more):**
- Active suicidal ideation
- Concrete plans
- Past self-harm behavior
- Immediate danger signals
- Different phrasings of same concept

**5. Contextual Variations:**
- Same concept, different words
- Different intensity levels
- Slang/informal language
- Regional variations

### Length Guidelines:
- **Minimum:** 8 words
- **Target:** 10-30 words
- **Maximum:** 50 words

---

## ⚖️ LABEL BALANCE TARGET

| Category | Current | Target | Need |
|----------|---------|--------|------|
| neutral | 127 | 300 | +173 |
| stress | 104 | 250 | +146 |
| unsafe_environment | 86 | 200 | +114 |
| emotional_distress | 107 | 250 | +143 |
| self_harm_low | 77 | 200 | +123 |
| self_harm_high | 82 | 200 | +118 |
| **TOTAL** | **583** | **1400** | **+817** |

---

## 🚫 COMMON LABELING MISTAKES TO AVOID

| ❌ Wrong | ✅ Correct | Reason |
|----------|-----------|---------|
| "I'm tired" → emotional_distress | "I'm tired" → stress | Not severe enough |
| "I might hurt myself" → self_harm_low | "I might hurt myself" → self_harm_high | Contains "hurt myself" = active |
| "Someone yelled at me" → unsafe_environment | "Someone yelled at me" → stress | Not ongoing threat |
| "I'm sad" → emotional_distress | "I'm sad" → stress | Too mild |
| "I'm happy" → neutral | Create separate "positive" class | Needs distinction |
| "Normal day" → stress | "Normal day" → neutral | No stressor |

---

## 🎯 VALIDATION CHECKLIST

Before adding a sample to training data:

- [ ] Text is at least 8 words long
- [ ] Label matches the decision tree above
- [ ] No ambiguity about which category
- [ ] Text is natural/realistic
- [ ] Not duplicate or very similar to existing
- [ ] Severity matches threshold guidance
- [ ] Self-harm samples are correctly classified as low/high

---

## 🔄 NEXT STEPS

1. **Fix the 18 mislabeled samples** (CRITICAL)
2. **Collect 800+ new samples** following guidelines above
3. **Run data augmentation** (paraphrases, variations)
4. **Retrain with calibrated model**
5. **Test on edge cases**

---

This will fix your:
✅ Label confusion  
✅ Classification errors  
✅ Overprediction issues  
✅ Poor separability
