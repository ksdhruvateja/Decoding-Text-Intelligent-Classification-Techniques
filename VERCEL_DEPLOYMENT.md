# 🚀 Vercel Deployment Guide

## Prerequisites

1. **GitHub Account** - Your code should be on GitHub
2. **Vercel Account** - Sign up at [vercel.com](https://vercel.com)
3. **Model Files** - Ensure model checkpoints are in `backend/checkpoints/`

## Step 1: Prepare Your Repository

### Initialize Git (if not already done)

```bash
# Initialize git
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit - Ready for Vercel deployment"

# Add remote
git remote add origin https://github.com/ksdhruvateja/Decoding-Text-Intelligent-Classification-Techniques.git

# Push to GitHub
git branch -M main
git push -u origin main
```

## Step 2: Deploy to Vercel

### Option A: Deploy via Vercel Dashboard (Recommended)

1. **Go to [vercel.com](https://vercel.com)** and sign in
2. **Click "New Project"**
3. **Import your GitHub repository**
   - Select: `ksdhruvateja/Decoding-Text-Intelligent-Classification-Techniques`
4. **Configure Project**:
   - **Framework Preset**: Other
   - **Root Directory**: `./` (root)
   - **Build Command**: `cd frontend && npm install && npm run build`
   - **Output Directory**: `frontend/build`
5. **Environment Variables** (if needed):
   - `REACT_APP_API_URL` = `/api` (for production)
6. **Click "Deploy"**

### Option B: Deploy via Vercel CLI

```bash
# Install Vercel CLI
npm i -g vercel

# Login to Vercel
vercel login

# Deploy
vercel

# Follow prompts:
# - Set up and deploy? Yes
# - Which scope? (select your account)
# - Link to existing project? No
# - Project name? decoding-text-classifier
# - Directory? ./
# - Override settings? No
```

## Step 3: Configure Vercel Settings

After deployment, go to **Project Settings** → **General**:

1. **Build & Development Settings**:
   - Framework Preset: Other
   - Build Command: `cd frontend && npm install && npm run build`
   - Output Directory: `frontend/build`
   - Install Command: `cd frontend && npm install`

2. **Environment Variables**:
   - `REACT_APP_API_URL` = `/api` (for production)

## Step 4: Important Notes

### Model Files Size

⚠️ **Vercel has a 50MB limit per file and 100MB total for serverless functions**

If your model files are too large:

1. **Option 1**: Use Vercel's file system (models are included in deployment)
2. **Option 2**: Store models in external storage (S3, Google Cloud Storage)
3. **Option 3**: Use a smaller model or quantized version

### Cold Start Performance

- First request may take 10-30 seconds (model loading)
- Subsequent requests are fast
- Consider using Vercel Pro for better performance

### API Routes

All API routes are automatically available at:
- `https://your-project.vercel.app/api/classify`
- `https://your-project.vercel.app/api/health`
- etc.

## Step 5: Test Deployment

After deployment, test your endpoints:

```bash
# Health check
curl https://your-project.vercel.app/api/health

# Classify text
curl -X POST https://your-project.vercel.app/api/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "I know I can achieve anything"}'
```

## Troubleshooting

### Issue: Build Fails

**Solution**: Check build logs in Vercel dashboard
- Ensure all dependencies are in `package.json`
- Check Python version compatibility
- Verify model files are included

### Issue: API Returns 500 Error

**Solution**: 
- Check Vercel function logs
- Ensure model files are in correct location
- Verify all Python dependencies are in `api/requirements.txt`

### Issue: Model Loading Takes Too Long

**Solution**:
- Use lazy loading (already implemented)
- Consider model quantization
- Use Vercel Pro for better performance

### Issue: CORS Errors

**Solution**: 
- CORS is already configured in `api/index.py`
- Ensure frontend uses correct API URL

## Project Structure for Vercel

```
.
├── api/                    # Serverless functions
│   ├── index.py           # Main API handler
│   └── requirements.txt    # Python dependencies
├── backend/               # Backend code (imported by api/)
│   ├── multistage_classifier.py
│   ├── bert_classifier.py
│   └── checkpoints/       # Model files
├── frontend/              # React frontend
│   ├── src/
│   ├── package.json
│   └── build/             # Build output (generated)
├── vercel.json           # Vercel configuration
└── package.json          # Root package.json
```

## Environment Variables

### Production (Vercel)
- `REACT_APP_API_URL` = `/api`

### Development (Local)
- `REACT_APP_API_URL` = `http://localhost:5000/api`

## Monitoring

- **Vercel Dashboard**: View logs, analytics, and performance
- **Function Logs**: Check API function execution logs
- **Analytics**: Monitor request counts and response times

## Next Steps

1. ✅ Deploy to Vercel
2. ✅ Test all endpoints
3. ✅ Monitor performance
4. ✅ Set up custom domain (optional)
5. ✅ Configure CI/CD (automatic on Git push)

## Support

If you encounter issues:
1. Check Vercel deployment logs
2. Review function logs
3. Test locally first
4. Check Vercel documentation: https://vercel.com/docs

---

**Happy Deploying! 🚀**

