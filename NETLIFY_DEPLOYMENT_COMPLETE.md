# 🚀 Complete Netlify Deployment Guide

## Overview

This application uses a **two-tier deployment architecture**:
- **Frontend**: Deployed on Netlify (React app + serverless functions)
- **Backend**: Deployed on Render, Railway, or similar Python hosting service

## 📋 Prerequisites

- GitHub account
- Netlify account (free tier works)
- Render/Railway account (for backend)

---

## Part 1: Deploy Backend (Python Flask API)

### Option A: Deploy to Render (Recommended - Free Tier)

1. **Go to Render**: https://render.com
2. **Sign in** with GitHub
3. Click **"New +"** → **"Web Service"**
4. **Connect your repository**
5. **Configure the service**:
   ```
   Name: text-classifier-backend
   Region: Choose closest to you
   Branch: main
   Root Directory: backend
   Runtime: Python 3
   Build Command: pip install -r requirements.txt
   Start Command: gunicorn app:application --bind 0.0.0.0:$PORT
   ```
6. **Set Environment Variables** (if needed):
   - Add any API keys or secrets your model needs
7. Click **"Create Web Service"**
8. **Wait for deployment** (5-10 minutes first time)
9. **Copy the URL**: It will look like `https://text-classifier-backend-xxxx.onrender.com`
10. **Test it**: Visit `https://your-backend-url.onrender.com/api/health`

### Option B: Deploy to Railway

1. **Go to Railway**: https://railway.app
2. **Sign in** with GitHub
3. Click **"New Project"** → **"Deploy from GitHub repo"**
4. **Select your repository**
5. Railway will auto-detect Python
6. **Add environment variables** if needed
7. Railway will provide a URL like `https://your-app.up.railway.app`
8. **Test it**: Visit `https://your-app.up.railway.app/api/health`

### Backend Requirements

Make sure your `backend/requirements.txt` includes:
```
flask==3.0.0
flask-cors==4.0.0
gunicorn==21.2.0
torch>=2.6.0
transformers>=4.35.0
pandas>=2.1.3
numpy>=1.24.3
scikit-learn>=1.3.2
```

**Important**: Add `gunicorn` if not present!

---

## Part 2: Deploy Frontend to Netlify

### Step 1: Connect to GitHub

1. **Go to Netlify**: https://app.netlify.com
2. **Sign in** with GitHub
3. Click **"Add new site"** → **"Import an existing project"**
4. **Connect to GitHub** and authorize Netlify
5. **Select your repository**

### Step 2: Configure Build Settings

Netlify should auto-detect these settings, but verify:

```
Base directory: frontend
Build command: npm install && npm run build
Publish directory: frontend/build
```

### Step 3: Set Environment Variables

**CRITICAL STEP** - This is what makes the backend work!

1. In Netlify dashboard, go to: **Site settings** → **Environment variables**
2. Click **"Add a variable"** → **"Add a single variable"**
3. Add:
   ```
   Key: BACKEND_URL
   Value: https://your-backend-url.onrender.com
   ```
   ⚠️ **Replace with your actual backend URL from Part 1**
   ⚠️ **Do NOT include trailing slash**
   ⚠️ **Do NOT include /api path**

4. Click **"Create variable"**

### Step 4: Deploy

1. Click **"Deploy site"**
2. Wait for build to complete (2-5 minutes)
3. Netlify will provide a URL like `https://random-name-12345.netlify.app`

### Step 5: Test Your Deployment

1. **Visit your Netlify URL**
2. **Open browser console** (F12)
3. **Type a message** and submit
4. Check if classification works

**Check health endpoint**: Visit `https://your-site.netlify.app/api/health`

---

## 🔧 Troubleshooting

### Issue: "Backend not configured" error

**Solution**: 
1. Go to Netlify → Site settings → Environment variables
2. Verify `BACKEND_URL` is set correctly
3. Redeploy: Deploys → Trigger deploy → Deploy site

### Issue: CORS errors in browser console

**Solution**: 
- Check that your backend has CORS enabled
- Verify backend is running: visit `/api/health`
- Check backend logs on Render/Railway

### Issue: 503 or timeout errors

**Solution**:
- Render free tier spins down after inactivity
- First request may take 30-60 seconds (cold start)
- Try refreshing after a minute

### Issue: Backend works locally but not on Render

**Solution**:
1. Check Render logs for errors
2. Verify all dependencies in `requirements.txt`
3. Make sure `gunicorn` is installed
4. Check start command: `gunicorn app:application --bind 0.0.0.0:$PORT`

### Issue: Changes not showing on Netlify

**Solution**:
1. Make sure changes are pushed to GitHub
2. Go to Netlify → Deploys → Trigger deploy
3. Clear browser cache (Ctrl+Shift+R)

---

## 🔄 Making Updates

### Update Backend

1. **Push changes to GitHub**:
   ```bash
   git add backend/
   git commit -m "Update backend"
   git push
   ```
2. **Render/Railway auto-deploys** from GitHub (check settings)
3. **Test**: Visit `https://your-backend-url/api/health`

### Update Frontend

1. **Push changes to GitHub**:
   ```bash
   git add frontend/
   git commit -m "Update frontend"
   git push
   ```
2. **Netlify auto-deploys** from GitHub
3. **Verify**: Check Netlify dashboard for build status

---

## 📊 Monitoring

### Check Backend Health
```
https://your-backend-url.onrender.com/api/health
```

Should return:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2025-12-04T..."
}
```

### Check Frontend + Backend Connection
```
https://your-site.netlify.app/api/health
```

Should return:
```json
{
  "frontend_status": "healthy",
  "backend_status": { ... },
  "backend_url": "https://your-backend-url.onrender.com"
}
```

---

## 💰 Cost

- **Netlify**: Free tier includes 100GB bandwidth/month
- **Render**: Free tier with limitations (spins down after 15 min inactivity)
- **Railway**: Free $5 credit/month

---

## 🎯 Quick Checklist

- [ ] Backend deployed to Render/Railway
- [ ] Backend URL copied
- [ ] Backend health check working
- [ ] Netlify connected to GitHub
- [ ] `BACKEND_URL` environment variable set in Netlify
- [ ] Frontend deployed on Netlify
- [ ] Test classification working end-to-end
- [ ] No CORS errors in browser console

---

## 📞 Need Help?

Common URLs to check:
1. **Backend Health**: `https://your-backend.onrender.com/api/health`
2. **Frontend**: `https://your-site.netlify.app`
3. **Frontend Health**: `https://your-site.netlify.app/api/health`

If all three work, your deployment is successful! 🎉
