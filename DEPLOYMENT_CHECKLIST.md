# ✅ Deployment Checklist

## Before Pushing to GitHub

### 1. Environment Files
- [ ] Run `setup-env.sh` (Linux/Mac) or `setup-env.bat` (Windows)
- [ ] Verify `frontend/.env.development` exists
- [ ] Verify `frontend/.env.production` exists

### 2. Model Files
- [ ] Ensure model checkpoints are in `backend/checkpoints/`
- [ ] Check file sizes (Vercel has 50MB per file limit)
- [ ] If models are too large, consider:
  - Using quantized models
  - Storing in external storage (S3, etc.)
  - Using smaller models

### 3. Git Configuration
- [ ] Verify `.gitignore` is correct
- [ ] Check that sensitive files are not committed
- [ ] Ensure `node_modules/` and `venv/` are ignored

### 4. Code Review
- [ ] Test locally (frontend + backend)
- [ ] Verify API endpoints work
- [ ] Check that all imports are correct

## Git Commands

```bash
# Initialize git (if not already done)
git init

# Add all files
git add .

# Check what will be committed
git status

# Commit
git commit -m "Initial commit - Ready for Vercel deployment"

# Add remote (if not already added)
git remote add origin https://github.com/ksdhruvateja/Decoding-Text-Intelligent-Classification-Techniques.git

# Push to GitHub
git branch -M main
git push -u origin main
```

## After Pushing to GitHub

### 1. Vercel Deployment
- [ ] Go to vercel.com and sign in
- [ ] Click "New Project"
- [ ] Import GitHub repository
- [ ] Configure build settings:
  - Framework: Other
  - Build Command: `cd frontend && npm install && npm run build`
  - Output Directory: `frontend/build`
- [ ] Add environment variable: `REACT_APP_API_URL=/api`
- [ ] Deploy!

### 2. Post-Deployment Testing
- [ ] Test health endpoint: `https://your-project.vercel.app/api/health`
- [ ] Test classification: `https://your-project.vercel.app/api/classify`
- [ ] Test frontend UI
- [ ] Check function logs for errors

### 3. Monitoring
- [ ] Monitor Vercel dashboard for errors
- [ ] Check function execution times
- [ ] Monitor API response times

## Troubleshooting

### Build Fails
- Check Vercel build logs
- Verify all dependencies in `package.json`
- Check Python version compatibility

### API Returns 500
- Check Vercel function logs
- Verify model files are accessible
- Check Python dependencies in `api/requirements.txt`

### CORS Errors
- Verify CORS is configured in `api/index.py`
- Check frontend API URL configuration

### Model Loading Issues
- Check model file paths
- Verify model files are included in deployment
- Consider lazy loading (already implemented)

## File Structure Verification

Ensure your project has:
```
.
├── api/
│   ├── index.py
│   └── requirements.txt
├── backend/
│   ├── app.py
│   ├── multistage_classifier.py
│   ├── bert_classifier.py
│   └── checkpoints/
├── frontend/
│   ├── src/
│   ├── package.json
│   ├── .env.development
│   └── .env.production
├── vercel.json
├── .gitignore
└── README.md
```

## Quick Commands Reference

```bash
# Setup environment files
bash setup-env.sh        # Linux/Mac
setup-env.bat           # Windows

# Git commands
git add .
git commit -m "Your message"
git push origin main

# Local testing
cd backend && python app.py
cd frontend && npm start
```

---

**Ready to deploy? Follow this checklist step by step! ✅**

