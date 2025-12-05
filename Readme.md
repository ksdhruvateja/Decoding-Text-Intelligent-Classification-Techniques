# Decoding Text: Intelligent Classification Techniques

A sophisticated **Hybrid Text Classification System** that combines BERT deep learning, rule-based pattern matching, and TF-IDF for accurate sentiment and emotion detection with safety-focused threat detection.

## 🎯 Features

- **Hybrid Classification Pipeline**: Combines BERT + Rule-based + TF-IDF classifiers
- **9 Classification Categories**: positive, negative, neutral, stress, emotional_distress, self_harm_low, self_harm_high, unsafe_environment, threat_of_violence
- **Emotion Vector Detection**: Detects positive, negative, stress, unsafe, crisis, emotional_distress, neutral emotions
- **Smart Alert System**: Only triggers alerts for genuine high-risk content (self-harm, threats, unsafe environments)
- **Threat Detection**: Specialized patterns for violent threats, weapons, attack planning
- **React Frontend**: Modern, responsive UI with real-time classification
- **Flask Backend**: RESTful API with production-ready classifier
- **95.6% Validation Accuracy**: Trained BERT model on balanced dataset

## 🏗️ Architecture

This project uses a **Production Hybrid Classifier** approach:

1. **Simple Classifier (Rule-Based)** - Fast pattern matching for explicit threats, self-harm, sentiment keywords
2. **BERT Classifier** - Deep learning (bert-base-uncased) for semantic understanding of complex text
3. **TF-IDF Fallback** - Traditional ML for robust classification when BERT is uncertain
4. **Smart Blending** - Combines predictions with confidence-based weighting
5. **Safety Overrides** - Always prioritizes high-risk detection (self-harm, threats)

## 📁 Project Structure

```
.
├── backend/                          # Backend Python code
│   ├── app.py                       # Flask application with API endpoints
│   ├── production_classifier.py     # Hybrid classifier (BERT + Rules + TF-IDF)
│   ├── simple_classifier.py         # Rule-based pattern matching
│   ├── bert_classifier.py           # BERT model wrapper
│   ├── tfidf_classifier.py          # TF-IDF classifier
│   ├── requirements.txt             # Python dependencies
│   ├── venv/                        # Python virtual environment
│   ├── checkpoints/                 # Trained models
│   │   ├── bert_classifier_best.pt  # BERT model (95.6% accuracy)
│   │   ├── bert_tokenizer/          # BERT tokenizer files
│   │   └── tfidf_classifier.joblib  # TF-IDF model (76.4% accuracy)
│   ├── balanced_training_data.json  # 526 balanced training examples
│   └── clean_training_data.json     # Original training dataset
├── frontend/                        # React frontend
│   ├── src/
│   │   ├── App.js                   # Main React component with classification UI
│   │   ├── App.css                  # Modern styling with gradients
│   │   └── index.js                 # React entry point
│   ├── package.json                 # Node.js dependencies
│   └── public/                      # Static assets
├── checkpoints/                     # Root-level model checkpoints
│   ├── bert_classifier_best.pt      # Trained BERT model
│   ├── bert_tokenizer/              # Tokenizer config
│   └── tfidf_classifier.joblib      # Trained TF-IDF model
└── README.md                        # This file
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+** (Python 3.13 recommended)
- **Node.js 14+** and npm
- **Git** (for cloning the repository)
- **8GB+ RAM** (for running BERT model)

### Installation

#### 1. Clone the Repository

```bash
git clone https://github.com/ksdhruvateja/Decoding-Text-Intelligent-Classification-Techniques.git
cd Decoding-Text-Intelligent-Classification-Techniques
```

#### 2. Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create virtual environment (Windows)
python -m venv venv
.\venv\Scripts\activate

# Create virtual environment (Linux/Mac)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download/verify model checkpoints (should already be in checkpoints/)
# Models are stored via Git LFS: bert_classifier_best.pt, tfidf_classifier.joblib
```

#### 3. Frontend Setup

```bash
# Navigate to frontend directory (from root)
cd frontend

# Install Node.js dependencies
npm install
```

### Running the Application

#### Option 1: Run Both Services Manually

**Terminal 1 - Backend:**
```bash
cd backend
.\venv\Scripts\activate  # Windows
# OR
source venv/bin/activate  # Linux/Mac

python app.py
```
Backend will start at `http://localhost:5000`

**Terminal 2 - Frontend:**
```bash
cd frontend
npm start
```
Frontend will start at `http://localhost:3000` and automatically open in your browser.

#### Option 2: Run as Background Jobs (PowerShell/Windows)

```powershell
# Start backend
cd backend
Start-Job -ScriptBlock { 
    cd 'C:\path\to\project\backend'
    & '.\venv\Scripts\python.exe' app.py 
} -Name "BackendServer"

# Start frontend (opens in minimized window)
cd frontend
Start-Process powershell -ArgumentList "-NoExit", "-Command", "npm start" -WindowStyle Minimized
```

#### Option 3: Use Batch Scripts (Windows)

Create `start-all.bat`:
```batch
@echo off
start "Backend Server" cmd /k "cd backend && venv\Scripts\activate && python app.py"
start "Frontend Server" cmd /k "cd frontend && npm start"
```

### Verify Services are Running

```powershell
# Check backend health
curl http://localhost:5000/api/health

# Check frontend
curl http://localhost:3000
```

### Usage

1. Open your browser to `http://localhost:3000`
2. Enter text in the input field
3. Click "Classify Text" or press Enter
4. View the classification results:
   - **Primary Category**: Main classification (positive, negative, neutral, stress, etc.)
   - **Sentiment**: Overall sentiment (positive, negative, neutral, high_risk)
   - **Emotion Vector**: Detected emotion (positive, negative, stress, unsafe, crisis, neutral)
   - **Alert Status**: "Safe channel" or "Alert raised" (only for self-harm/threats)
   - **Confidence Scores**: Percentage breakdown by category
   - **Pipeline Stages**: See how the text was processed through each stage

## 📡 API Endpoints

### `POST /api/classify`
Classify a single text input.

**Request**:
```json
{
  "text": "This phone is amazing and works perfectly!",
  "threshold": 0.5
}
```

**Response**:
```json
{
  "text": "This phone is amazing and works perfectly!",
  "primary_category": "positive",
  "confidence": 0.95,
  "sentiment": "positive",
  "emotion": "positive",
  "model": "simple",
  "predictions": [
    {"label": "positive", "score": 0.95}
  ],
  "all_scores": {
    "positive": 0.95,
    "negative": 0.0,
    "neutral": 0.0,
    "stress": 0.0,
    "emotional_distress": 0.0,
    "self_harm_low": 0.0,
    "self_harm_high": 0.0,
    "unsafe_environment": 0.0
  },
  "timestamp": "2025-12-04T23:45:12.123456"
}
```

### `POST /api/classify-formatted`
Returns classification in formatted text block.

**Request**: Same as `/api/classify`

**Response**: Plain text formatted block

### `GET /api/health`
Health check endpoint.

**Response**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2025-12-04T23:45:12.123456"
}
```

### `POST /api/batch-classify`
Classify multiple texts at once.

**Request**:
```json
{
  "texts": [
    "I am feeling great today!",
    "This is stressful",
    "I will kill him"
  ],
  "threshold": 0.5
}
```

**Response**:
```json
{
  "results": [...],
  "count": 3,
  "timestamp": "2025-12-04T23:45:12.123456"
}
```

### `GET /api/history?limit=50`
Get classification history.

**Response**:
```json
{
  "history": [...],
  "total": 50
}
```

### `DELETE /api/history/clear`
Clear classification history.

### `GET /api/categories`
Get available classification categories.

### `GET /api/stats`
Get classification statistics.

## 🎓 Training

### Training Data

The models are trained on a balanced dataset with 526 examples:
- **positive**: 98 examples (compliments, achievements, joy)
- **negative**: 98 examples (criticism, disappointment, boredom)
- **neutral**: 80 examples (factual statements, observations)
- **stress**: 60 examples (pressure, deadlines, overwhelm)
- **emotional_distress**: 40 examples (sadness, hopelessness, despair)
- **self_harm**: 40 examples (suicidal thoughts and intent)
- **unsafe_environment**: 110 examples (threats, violence, weapons, attack planning)

### Generate Training Data

```bash
cd backend

# Generate balanced dataset
python generate_clean_training_data.py

# Add threat examples
python add_threat_data.py
```

### Train BERT Model

```bash
cd backend
python train_bert_classifier.py
```

**Training Results:**
- Validation Accuracy: **95.6%**
- Model: `bert-base-uncased`
- Epochs: 10 with early stopping
- Optimizer: AdamW with learning rate 2e-5
- Saved to: `checkpoints/bert_classifier_best.pt`

### Train TF-IDF Classifier

```bash
cd backend
python train_tfidf_classifier.py
```

**Training Results:**
- Test Accuracy: **76.4%**
- Saved to: `checkpoints/tfidf_classifier.joblib`

## 📊 Classification Categories

### Primary Categories

1. **positive** - Positive sentiment, compliments, achievements, joy, satisfaction
   - Examples: "This phone is amazing!", "I love this movie!", "Great job!"

2. **negative** - Negative sentiment, criticism, disappointment, boredom
   - Examples: "The movie was boring", "This phone is terrible", "I hate this"

3. **neutral** - Factual statements, observations, informational content
   - Examples: "The package arrived yesterday", "This book is 300 pages long"

4. **stress** - Worry, pressure, anxiety, overwhelm, time pressure
   - Examples: "I am so overwhelmed with work", "Too many deadlines", "I can't handle this"

5. **emotional_distress** - Sadness, depression, hopelessness, despair
   - Examples: "I feel so hopeless", "Nothing matters anymore", "I'm so sad"

6. **self_harm_low** - Suicidal ideation (passive thoughts)
   - Examples: "I wish I didn't exist", "Life isn't worth living"

7. **self_harm_high** - Active suicidal intent (plans, methods)
   - Examples: "I want to kill myself", "I have a plan to end it"

8. **unsafe_environment** - Threats to others, violence, weapons, attack planning
   - Examples: "I will kill him", "Bringing a gun to school", "Planning an attack"

9. **threat_of_violence** - Derived from unsafe_environment score
   - Computed category for explicit threats

### Emotion Categories

- **positive** - Joyful, happy, satisfied (from positive category)
- **negative** - Disappointed, bored, critical (from negative category)
- **neutral** - Calm, informational (default for factual text)
- **stress** - Anxious, overwhelmed, pressured
- **emotional_distress** - Sad, hopeless, depressed
- **unsafe** - Threatening, dangerous (from unsafe_environment)
- **crisis** - Suicidal, immediate danger (from self_harm_high/low)

### High-Risk Categories

The system triggers alerts **only** for these categories:
- `self_harm_high`
- `self_harm_low`
- `unsafe_environment`

All other categories (positive, negative, neutral, stress) show "Safe channel".

## 🔧 Configuration

### Backend Configuration

**Environment Variables:**
- `PORT`: Backend port (default: 5000)

**app.py Settings:**
- `debug=False`: Production mode
- `use_reloader=False`: Prevents double loading of models
- `host='0.0.0.0'`: Allows external connections

### Frontend Configuration

**Environment Variables** (`frontend/.env.development`):
```env
REACT_APP_API_URL=http://localhost:5000/api
```

**Frontend API Base URL** (hardcoded in `App.js`):
```javascript
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';
```

### Model Thresholds

Configured in `production_classifier.py`:
- **High-risk categories**: 0.80 (self_harm, unsafe_environment)
- **Emotion/crisis**: 0.60 (emotional_distress, stress)
- **Neutral trust**: 0.75+ (high confidence for neutral classification)
- **Prediction limit**: Top 3 categories only
- **Minimum threshold**: 0.25 for any category to appear

## 🧪 Testing

### Test Classification

```bash
cd backend
python -c "from simple_classifier import SimpleClassifier; c = SimpleClassifier(); print(c.classify('This phone is amazing!'))"
```

### Test API

```bash
# Health check
curl http://localhost:5000/api/health

# Classify text
curl -X POST http://localhost:5000/api/classify \
  -H "Content-Type: application/json" \
  -d '{"text":"This phone is amazing!"}'
```

### Test Cases

The system correctly handles:
- ✅ **"This phone is amazing!"** → positive, emotion: positive, "Safe channel"
- ✅ **"The movie was boring"** → negative, emotion: negative, "Safe channel"
- ✅ **"The package arrived yesterday"** → neutral, emotion: neutral, "Safe channel"
- ✅ **"I am overwhelmed with work"** → stress, emotion: stress, "Safe channel"
- ✅ **"I will kill him"** → unsafe_environment (98%), emotion: unsafe, "Alert raised"
- ✅ **"I want to kill myself"** → self_harm_high (95%), emotion: crisis, "Critical alert"

## 🛠️ Technologies

### Frontend
- **React 18** - Modern UI framework
- **Lucide Icons** - Beautiful icon library
- **CSS3** - Responsive design with gradients and animations
- **Fetch API** - HTTP requests to backend

### Backend
- **Flask 2.3+** - Python web framework
- **Flask-CORS** - Cross-origin resource sharing
- **Python 3.13** - Latest Python features

### Machine Learning
- **PyTorch 2.1+** - Deep learning framework
- **Transformers (Hugging Face)** - BERT model implementation
- **bert-base-uncased** - Pretrained BERT model
- **scikit-learn** - TF-IDF and metrics
- **joblib** - Model serialization

### Development
- **Node.js & npm** - Frontend package management
- **pip & venv** - Python package management
- **Git & Git LFS** - Version control with large file support

### Deployment
- **Local Development** - Flask + React dev servers
- **Production Ready** - Can deploy to Heroku, Render, Railway, AWS
- **Git LFS** - For storing large model files (924 MB total)

## 📚 Documentation

- **[LABEL_DEFINITIONS.md](./backend/LABEL_DEFINITIONS.md)** - Detailed category definitions and examples
- **[HOW_IT_WORKS.md](./backend/HOW_IT_WORKS.md)** - System architecture and workflow
- **[ADVANCED_TRAINING_GUIDE.md](./backend/ADVANCED_TRAINING_GUIDE.md)** - Model training instructions
- **[QUICK_START.md](./QUICK_START.md)** - Quick setup guide

## 🐛 Troubleshooting

### Backend Won't Start

**Issue:** UnicodeEncodeError with emoji characters
**Solution:** Emojis removed from print statements in v1.2.0+

**Issue:** Port 5000 already in use
**Solution:** 
```bash
# Windows
netstat -ano | findstr :5000
taskkill /PID <process_id> /F

# Linux/Mac
lsof -ti:5000 | xargs kill -9
```

### BERT Model Not Loading

**Issue:** `FileNotFoundError: checkpoints/bert_classifier_best.pt`
**Solution:** Ensure Git LFS is installed and models are pulled:
```bash
git lfs install
git lfs pull
```

**Issue:** Out of memory when loading BERT
**Solution:** Requires 8GB+ RAM. Use TF-IDF fallback if memory limited.

### Frontend Connection Issues

**Issue:** `CORS error` or `Network request failed`
**Solution:** 
1. Ensure backend is running on port 5000
2. Check `REACT_APP_API_URL` in frontend `.env`
3. Verify Flask-CORS is installed: `pip install flask-cors`

### Classification Issues

**Issue:** Wrong emotion or sentiment
**Solution:** System now correctly detects emotions after v1.2.0 update

**Issue:** False alerts on negative text
**Solution:** Alert logic fixed in v1.2.0 - only shows for high-risk categories

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Test thoroughly (backend + frontend)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## 📝 License

This project is part of an academic research project for **Group 5 - Decoding Text: Intelligent Classification Techniques**.

## 👥 Authors

- **Group 5** - Initial work and research
- **ksdhruvateja** - Repository maintainer

## 📧 Support

For issues, questions, or feature requests:
- Open an issue on [GitHub Issues](https://github.com/ksdhruvateja/Decoding-Text-Intelligent-Classification-Techniques/issues)
- Contact the development team

## 🙏 Acknowledgments

- **Hugging Face** - For BERT models and Transformers library
- **PyTorch Team** - For deep learning framework
- **Flask Team** - For lightweight Python web framework
- **React Team** - For modern frontend framework

---

**Built with ❤️ for intelligent text classification and mental health safety**

**Version:** 1.2.0 | **Last Updated:** December 2025
