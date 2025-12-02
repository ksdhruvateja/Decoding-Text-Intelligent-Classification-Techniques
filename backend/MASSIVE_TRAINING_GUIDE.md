# Massive Training Guide - Internet-Scale Data Collection

## 🎯 Overview

This guide implements a **massive-scale training pipeline** that collects data from multiple sources to train on almost all kinds of words and text variations available on the internet.

## 🚀 Quick Start

### Option 1: Run Everything at Once (Recommended)

```bash
cd backend
python run_massive_training.py
```

This will:
1. Collect data from LLM, Reddit, public datasets
2. Generate comprehensive training data
3. Augment data with variations
4. Train model on massive dataset

**Time**: 2-4 hours (depending on data sources and hardware)

### Option 2: Run Steps Separately

#### Step 1: Collect Massive Data

```bash
python massive_data_collector.py
```

**What it does:**
- Generates data using LLM (GPT-4, Claude) - 2000+ examples
- Scrapes Reddit from multiple subreddits - 5000+ examples
- Loads public datasets
- Generates comprehensive fixed data - 1360+ examples
- Augments data with variations - 2-3x multiplier

**Output**: `massive_training_data_train.json` and `massive_training_data_val.json`

#### Step 2: Train Model

```bash
python train_massive.py
```

**Features:**
- Large batch size (32, effective 64 with gradient accumulation)
- Multiple workers for data loading
- Optimized for massive datasets
- Saves best model based on F1 score

**Output**: `checkpoints/best_massive_model.pt`

## 📊 Data Sources

### 1. LLM Generation (2000+ examples)
- Uses GPT-4 or Claude to generate diverse examples
- Covers all categories with variations
- **Requires**: OpenAI API key or Anthropic API key

**Setup**:
```bash
export OPENAI_API_KEY="your-key-here"
# OR
export ANTHROPIC_API_KEY="your-key-here"
```

### 2. Reddit Scraping (5000+ examples)
- Scrapes from mental health, relationship, motivation subreddits
- Real-world text examples
- **Requires**: Reddit API credentials

**Setup**:
```bash
export REDDIT_CLIENT_ID="your-client-id"
export REDDIT_CLIENT_SECRET="your-client-secret"
```

**Install**:
```bash
pip install praw
```

### 3. Public Datasets
- Loads existing comprehensive training data
- Includes our fixed comprehensive data
- Can add more public datasets

### 4. Data Augmentation
- Generates variations of existing examples
- Capitalization, punctuation, synonyms
- 2-3x multiplier on dataset size

## 📈 Expected Results

After massive training:

| Metric | Expected Value |
|--------|----------------|
| **Training Examples** | 10,000 - 50,000+ |
| **Validation Examples** | 2,000 - 10,000+ |
| **Accuracy** | 95-98% |
| **F1 Score** | 0.92-0.96 |
| **False Positives** | < 2% |

## 🔧 Requirements

### Core
```bash
pip install torch transformers scikit-learn numpy tqdm scipy
```

### Optional (for data collection)
```bash
# For LLM generation
pip install openai anthropic

# For Reddit scraping
pip install praw

# For web scraping (if adding)
pip install beautifulsoup4 requests
```

## 🎯 Data Collection Details

### LLM Generation Categories

1. **Positive Confident** (200 examples)
   - "I am unstoppable"
   - "I will rise above obstacles"
   - Variations with different vocabulary

2. **Positive Relationship** (200 examples)
   - "I will marry her"
   - "I love spending time with them"
   - Various relationship contexts

3. **Frustration** (200 examples)
   - "I'm sick of people wasting my time"
   - "I can't stand this anymore"
   - Various frustration expressions

4. **Neutral** (200 examples)
   - "The meeting is at 3 PM"
   - "I went to the store"
   - Everyday statements

5. **Stress** (200 examples)
   - "I'm worried about my exam"
   - "Work is really stressful"
   - Various stress contexts

6. **Emotional Distress** (200 examples)
   - "I feel so overwhelmed"
   - "I'm feeling really down"
   - Various distress expressions

7. **Self-Harm Low** (100 examples)
   - "I sometimes think about ending it"
   - "I wonder if anyone would notice"
   - Ideation without plans

8. **Self-Harm High** (100 examples)
   - "I want to kill myself"
   - "I'm planning to hurt myself"
   - Direct plans/intent

9. **Unsafe Environment** (100 examples)
   - "I want to hurt them"
   - "I'm going to get revenge"
   - Threats toward others

### Reddit Subreddits

- **Mental Health**: r/depression, r/anxiety, r/SuicideWatch, r/mentalhealth
- **Positive**: r/happy, r/gratitude, r/GetMotivated, r/selfimprovement
- **Relationships**: r/relationships, r/relationship_advice
- **General**: r/offmychest, r/confession, r/CasualConversation

## 🚀 Advanced Options

### Use Multiple LLM Providers

Edit `massive_data_collector.py` to add:
- Anthropic Claude
- Google Gemini
- Local LLMs (Llama, Mistral)

### Add More Data Sources

1. **Twitter/X API** (if available)
2. **Common Crawl** (web-scale data)
3. **Wikipedia** (neutral examples)
4. **News articles** (various contexts)
5. **Books/Novels** (diverse language)

### Custom Data Sources

Add your own data collection methods in `massive_data_collector.py`:

```python
def collect_custom_source(self):
    """Your custom data collection"""
    # Add your logic here
    pass
```

## 📊 Training Configuration

### Default Settings
- **Batch Size**: 32 (effective 64 with gradient accumulation)
- **Epochs**: 10
- **Learning Rate**: 2e-5
- **Max Length**: 128 tokens
- **Workers**: 4 (parallel data loading)

### For Larger Datasets
Edit `train_massive.py`:
```python
CONFIG = {
    'batch_size': 64,  # Increase for more data
    'epochs': 15,  # More epochs for massive data
    'gradient_accumulation_steps': 4,  # Larger effective batch
}
```

## ✅ Verification

After training, test with:

```python
from multistage_classifier import initialize_multistage_classifier

classifier = initialize_multistage_classifier('checkpoints/best_massive_model.pt')

test_cases = [
    "I am unstoppable—every obstacle I face becomes another reason for me to rise higher",
    "The meeting has been moved to 3 PM",
    "I can't stand the way they treat me anymore",
    "I will marry her",
    "I want to kill myself",
]

for text in test_cases:
    result = classifier.classify(text)
    print(f"\n{text}")
    print(f"  Emotion: {result['emotion']}")
    print(f"  Sentiment: {result['sentiment']}")
    print(f"  Risk Scores: {result['all_scores']}")
```

## 🐛 Troubleshooting

### Out of Memory

**Solution**: Reduce batch size
```python
CONFIG['batch_size'] = 16  # Instead of 32
```

### Reddit API Errors

**Solution**: 
- Check API credentials
- Reduce scraping limits
- Add delays between requests

### LLM API Errors

**Solution**:
- Check API keys
- Reduce number of examples per category
- Use free alternatives (HuggingFace models)

### Training Too Slow

**Solution**:
- Use GPU if available
- Reduce dataset size for testing
- Use smaller model (distilbert)

## 📝 Notes

- **Data Quality**: Review collected data before training
- **Bias**: Be aware of biases in scraped data
- **Privacy**: Ensure compliance with data sources
- **Ethics**: Use responsibly, especially for mental health

## 🎓 Next Steps

1. **Monitor Performance**: Track accuracy on validation set
2. **Collect Feedback**: Add human feedback for RLHF
3. **Continuous Learning**: Periodically retrain with new data
4. **Deploy**: Use trained model in production
5. **Evaluate**: Test on real-world examples

