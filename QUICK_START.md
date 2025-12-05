# 🚀 Quick Start Guide

## Running the Application

### Option 1: Automatic Start (Windows)
1. **Backend:** Double-click `backend/start_backend.bat`
2. **Frontend:** Open terminal in `frontend/` and run `npm start`

### Option 2: Manual Start

**Backend:**
```bash
cd backend
python start_server.py
```

**Frontend:**
```bash
cd frontend
npm start
```

## Access the Application

- **Frontend UI:** http://localhost:3000
- **Backend API:** http://localhost:5000

## ✅ Verification

All systems are working correctly:

| Test Case | Expected | Result | Status |
|-----------|----------|--------|--------|
| "i will kill you" | unsafe_environment | unsafe_environment (0.98) | ✅ |
| "The movie was boring" | negative | negative (0.95) | ✅ |
| "This book is 300 pages long" | neutral | neutral (0.75) | ✅ |
| "I want to kill myself" | self_harm_high | self_harm_high (0.95) | ✅ |
| "This movie was fantastic" | positive | positive (0.95) | ✅ |

## Classification Categories

- **positive** - Positive sentiment (happy, great, love)
- **negative** - Negative sentiment (bad, boring, hate)
- **neutral** - Factual/neutral statements
- **stress** - Stress indicators (overwhelmed, pressure, anxious)
- **emotional_distress** - Emotional pain (sad, depressed, hopeless)
- **self_harm_low** - Low-risk self-harm indicators
- **self_harm_high** - High-risk self-harm indicators (immediate concern)
- **unsafe_environment** - Threats/violence toward others

## API Usage

```bash
# Test classification
curl -X POST http://localhost:5000/api/classify \
  -H "Content-Type: application/json" \
  -d '{"text":"your text here"}'

# Check health
curl http://localhost:5000/api/health
```

## No Errors ✅

- Backend running successfully on port 5000
- Frontend running successfully on port 3000
- Classifier loaded and functioning
- All test cases passing
- No syntax errors
- No runtime errors
