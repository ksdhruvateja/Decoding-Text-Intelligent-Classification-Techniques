# Netlify Deployment Guide

## Overview
This guide explains how to deploy the Text Classifier frontend to Netlify. Since Netlify doesn't natively support Python backends, you'll need to deploy the backend separately.

## Architecture

### Frontend (Netlify)
- React application
- Serverless functions (Node.js) as API proxy

### Backend (Separate Service)
Deploy the Python backend to one of these platforms:
- **Render** (Recommended - Free tier available)
- **Railway**
- **Heroku**
- **PythonAnywhere**
- **AWS Lambda with API Gateway**

## Step 1: Deploy Backend

### Option A: Deploy to Render (Recommended)

1. Go to https://render.com and sign up
2. Click "New +" → "Web Service"
3. Connect your GitHub repository
4. Configure:
   - **Name**: `text-classifier-backend`
   - **Root Directory**: `backend`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
5. Add environment variables if needed
6. Click "Create Web Service"
7. Copy the deployment URL (e.g., `https://text-classifier-backend.onrender.com`)

### Option B: Deploy to Railway

1. Go to https://railway.app
2. Click "Start a New Project" → "Deploy from GitHub repo"
3. Select your repository
4. Railway will auto-detect Python and deploy
5. Copy the deployment URL

## Step 2: Deploy Frontend to Netlify

### Prerequisites
- GitHub account
- Netlify account (free)

### Deployment Steps

1. **Push to GitHub** (if not already done):
   ```bash
   git add .
   git commit -m "Add Netlify deployment configuration"
   git push origin main
   ```

2. **Connect to Netlify**:
   - Go to https://app.netlify.com
   - Click "Add new site" → "Import an existing project"
   - Choose "GitHub" and authorize
   - Select your repository

3. **Configure Build Settings**:
   - **Base directory**: `frontend`
   - **Build command**: `npm install && npm run build`
   - **Publish directory**: `frontend/build`
   - Click "Show advanced" → "New variable"
     - Key: `BACKEND_URL`
     - Value: Your backend URL from Step 1 (e.g., `https://text-classifier-backend.onrender.com`)

4. **Deploy**:
   - Click "Deploy site"
   - Wait for deployment to complete
   - Your site will be available at `https://[random-name].netlify.app`

## Step 3: Update Frontend API URL

Update `frontend/src/App.js` to use the Netlify function:

```javascript
// Change from:
const response = await fetch('http://localhost:5000/classify', {
  
// To:
const response = await fetch('/api/classify', {
```

## Step 4: Configure Environment Variables

In Netlify dashboard:
1. Go to "Site settings" → "Environment variables"
2. Add:
   - `NODE_VERSION`: `18`
   - `BACKEND_URL`: Your Python backend URL

## File Structure

```
├── netlify.toml              # Netlify configuration
├── _redirects                # URL redirects
├── netlify/
│   └── functions/
│       ├── classify.js       # Serverless function (API proxy)
│       └── package.json      # Function dependencies
└── frontend/
    └── ...                   # React app
```

## Custom Domain (Optional)

1. In Netlify dashboard: "Domain settings" → "Add custom domain"
2. Follow instructions to configure DNS
3. Netlify provides free HTTPS

## Troubleshooting

### Build Fails
- Check Node.js version (should be 18+)
- Verify build command in `netlify.toml`
- Check build logs in Netlify dashboard

### API Not Working
- Verify `BACKEND_URL` environment variable is set
- Check backend is deployed and accessible
- Verify CORS is enabled on backend
- Check Netlify function logs

### Backend Not Responding
- Ensure backend service is running
- Check backend logs on your hosting platform
- Verify API endpoints are correct
- Test backend URL directly in browser

## Alternative: Netlify + Docker

If you want to deploy everything on Netlify, consider:
1. Using Netlify's Docker support (Beta)
2. Converting Python backend to Node.js
3. Using Netlify Edge Functions with WASM

## Cost

- **Netlify**: Free tier (100GB bandwidth, 300 build minutes/month)
- **Render**: Free tier (with spin-down after inactivity)
- **Railway**: Free trial ($5 credit), then pay-as-you-go

## Monitoring

- **Netlify Analytics**: Built-in (paid)
- **Frontend Logs**: Netlify dashboard → Functions
- **Backend Logs**: Your backend hosting platform

## Continuous Deployment

Both Netlify and Render support automatic deployments:
- Push to `main` branch → Auto-deploy frontend to Netlify
- Push to `main` branch → Auto-deploy backend to Render

## Support

- Netlify Docs: https://docs.netlify.com
- Render Docs: https://render.com/docs
- Railway Docs: https://docs.railway.app
