# 🚀 Deploy to Railway (EASIEST!)

Railway is perfect for this project - it auto-detects everything and just works!

## Quick Deploy (3 Minutes)

### Step 1: Deploy Backend to Railway

1. **Go to Railway**: https://railway.app
2. **Sign up/Login** with GitHub
3. Click **"New Project"**
4. Select **"Deploy from GitHub repo"**
5. Choose: `Decoding-Text-Intelligent-Classification-Techniques`
6. Railway will automatically:
   - Detect it's a Python app ✅
   - Install dependencies from `backend/requirements.txt` ✅
   - Run with gunicorn ✅
   - Give you a public URL ✅

7. **Wait 2-3 minutes** for deployment
8. **Get your URL**: Click on your deployment → Settings → Get the public URL
   - Will look like: `https://your-app.up.railway.app`

### Step 2: Test Your Backend

Visit: `https://your-app.up.railway.app/api/health`

Should see:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "..."
}
```

### Step 3: Deploy Frontend to Netlify (or Vercel)

**Option A: Netlify**
1. Go to https://netlify.com
2. Import from GitHub
3. Set build settings:
   - Base directory: `frontend`
   - Build command: `npm install && npm run build`
   - Publish directory: `frontend/build`
4. **Environment variable**:
   - `BACKEND_URL` = `https://your-app.up.railway.app` (from Railway)
5. Deploy!

**Option B: Vercel**
1. Go to https://vercel.com
2. Import from GitHub
3. Vercel auto-detects React
4. **Environment variable**:
   - `REACT_APP_API_URL` = `https://your-app.up.railway.app/api`
5. Deploy!

## ✅ That's It!

Your app is now live with:
- Backend on Railway (handles ML/AI)
- Frontend on Netlify/Vercel (handles UI)

## 🔧 Railway Configuration

Railway automatically detects:
- `backend/requirements.txt` → Installs Python packages
- `Procfile` → Runs gunicorn
- `runtime.txt` → Uses Python 3.9

No manual configuration needed!

## 💰 Cost

**100% FREE** on Railway's free tier:
- $5 credit per month
- More than enough for this project
- Automatic SSL
- Custom domain support

## 🎯 Quick Commands

### View Railway Logs
```bash
# Install Railway CLI (optional)
npm i -g @railway/cli

# Login
railway login

# View logs
railway logs
```

### Update Deployment
Just push to GitHub - Railway auto-deploys! 🚀

```bash
git add .
git commit -m "Update backend"
git push
```

Railway will automatically redeploy in ~2 minutes.

## 🆘 Troubleshooting

### Build fails?
- Check Railway logs in dashboard
- Verify `backend/requirements.txt` has all dependencies
- Make sure `gunicorn` is in requirements.txt ✅ (already added)

### App crashes?
- Check Railway logs
- Verify `app:application` exists in `backend/app.py` ✅
- Port is automatically set by Railway via `$PORT` ✅

### Can't access API?
- Make sure Railway gave you a public URL (Settings → Generate Domain)
- Test `/api/health` endpoint first

---

## 📋 Full Setup Checklist

- [x] Code pushed to GitHub
- [x] `Procfile` created for Railway
- [x] `runtime.txt` specifies Python version
- [x] `gunicorn` in requirements.txt
- [ ] Deploy to Railway (you do this)
- [ ] Copy Railway URL
- [ ] Deploy frontend with Railway URL as env var
- [ ] Test the app!

**Ready to deploy! Just go to railway.app and connect your GitHub repo!** 🎉
