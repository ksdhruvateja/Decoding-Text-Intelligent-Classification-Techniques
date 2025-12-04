# Decoding-Text-Intelligent-Classification-Techniques

A sophisticated **Multi-Stage Mental Health Text Classifier** that combines BERT deep learning, rule-based filters, and LLM verification for accurate and safe text classification.

## 🎯 Features

- **Multi-Stage Classification Pipeline**: Combines BERT, rules, and LLM verification
- **6 Classification Categories**: neutral, stress, emotional_distress, self_harm_low, self_harm_high, unsafe_environment
- **Strict Thresholds**: Prevents false positives on positive/neutral statements
- **Rule-Based Overrides**: Catches edge cases and ensures safety
- **React Frontend**: Beautiful, modern UI for text classification
- **Flask Backend**: RESTful API with comprehensive endpoints
- **Local Development**: Run locally with Flask backend and React frontend

## 🏗️ Architecture

This project uses a hybrid approach:

1. **BERT Model** - Deep learning for semantic understanding
2. **Rule-Based Filters** - Pattern matching and safety overrides
3. **LLM Verification** - GPT/LLM ensemble for validation
4. **Strict Thresholds** - High confidence requirements for risk categories

See [MODEL_ARCHITECTURE_DOCUMENTATION.md](./MODEL_ARCHITECTURE_DOCUMENTATION.md) for complete details.

## 📁 Project Structure

```
.
├── backend/               # Backend Python code
│   ├── app.py            # Flask application
│   ├── multistage_classifier.py  # Main classifier
│   ├── bert_classifier.py        # BERT model
│   ├── rule_classifier.py        # Rule-based fallback
│   ├── requirements.txt   # Python dependencies
│   └── checkpoints/       # Trained models
├── frontend/             # React frontend
│   ├── src/
│   │   ├── App.js        # Main React component
│   │   └── App.css       # Styling
│   └── package.json
├── api/                  # API serverless functions
│   └── index.py          # Python API handler
└── README.md
```

See [FILE_STRUCTURE_VISUAL.md](./FILE_STRUCTURE_VISUAL.md) for detailed structure.

## 🚀 Quick Start

### Local Development

#### Backend

```bash
cd backend
pip install -r requirements.txt
python app.py
```

Backend runs on `http://localhost:5000`

#### Frontend

```bash
cd frontend
npm install
npm start
Frontend runs on `http://localhost:3000`

### Full Stack Local Setup

```bash
# Terminal 1 - Backend
cd backend
pip install -r requirements.txt
python app.py

# Terminal 2 - Frontend
cd frontend
npm install
npm start
```

Visit `http://localhost:3000` to use the app!project in Netlify
4. Set `BACKEND_URL` environment variable
5. Deploy!

## 📡 API Endpoints

### `POST /api/classify`
Classify a single text input.

**Request**:
```json
{
  "text": "I know I can achieve anything I put my mind to"
}
```

**Response**:
```json
{
  "emotion": "neutral",
  "sentiment": "safe",
  "all_scores": {
    "neutral": 0.95,
    "stress": 0.0,
    "self_harm_high": 0.0,
    ...
  },
  "predictions": [...]
}
```

### `GET /api/health`
Health check endpoint.

### `POST /api/batch-classify`
Classify multiple texts at once.

### `GET /api/history`
Get classification history.

See [MODEL_ARCHITECTURE_DOCUMENTATION.md](./MODEL_ARCHITECTURE_DOCUMENTATION.md) for all endpoints.

## 🎓 Training

### Generate Training Data

```bash
cd backend
python generate_clean_balanced_data.py
```

### Train Model

```bash
python train_clean_balanced.py
```

Model saved to `checkpoints/best_clean_balanced_model.pt`

See [QUICK_FIX_TRAINING.md](./backend/QUICK_FIX_TRAINING.md) for details.

## 📊 Classification Categories

1. **neutral** - Everyday, informational statements
2. **stress** - Worry, pressure, anxiety
3. **emotional_distress** - Sadness, depression, hopelessness
4. **self_harm_low** - Suicidal ideation (thoughts)
5. **self_harm_high** - Active suicidal intent (plans)
6. **unsafe_environment** - Threats to others

## 🔧 Configuration

### Environment Variables

**Development** (`frontend/.env.development`):
```
REACT_APP_API_URL=http://localhost:5000/api
```

## 📚 Documentation

- **[MODEL_ARCHITECTURE_DOCUMENTATION.md](./MODEL_ARCHITECTURE_DOCUMENTATION.md)** - Complete architecture guide
- **[FILE_STRUCTURE_VISUAL.md](./FILE_STRUCTURE_VISUAL.md)** - File structure visualization
- **[LABEL_DEFINITIONS.md](./backend/LABEL_DEFINITIONS.md)** - Classification category definitions
- **[HOW_IT_WORKS.md](./backend/HOW_IT_WORKS.md)** - System workflow explanation

## 🛠️ Technologies

- **Frontend**: React 18, Lucide Icons, CSS3
- **Backend**: Flask, Flask-CORS, Python 3.9+
- **ML/AI**: PyTorch, Transformers (BERT), scikit-learn
- **Development**: Node.js, npm, pip
- **Version Control**: Git, GitHub
- **Frontend**: React, CSS3
- **Backend**: Flask, Python
- **ML**: PyTorch, Transformers (BERT)
- **Deployment**: Netlify (Frontend), Render/Railway (Backend)
- **Version Control**: Git, GitHub

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📧 Support

For issues and questions, please open an issue on GitHub.

---

**Built with ❤️ for intelligent text classification**
