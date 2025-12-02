# 🚀 Git Setup Commands for Vercel Deployment

## Complete Git Setup & Push Commands

Run these commands in order to set up your repository and push to GitHub:

### Step 1: Initialize Git (if not already done)

```bash
git init
```

### Step 2: Setup Environment Files

**Windows (PowerShell)**:
```powershell
cd frontend
echo REACT_APP_API_URL=http://localhost:5000/api | Out-File -FilePath .env.development -Encoding utf8
echo REACT_APP_API_URL=/api | Out-File -FilePath .env.production -Encoding utf8
cd ..
```

**Linux/Mac**:
```bash
bash setup-env.sh
```

### Step 3: Add All Files to Git

```bash
# Add all files
git add .

# Check what will be committed
git status
```

### Step 4: Create Initial Commit

```bash
git commit -m "Initial commit - Ready for Vercel deployment"
```

### Step 5: Set Main Branch

```bash
git branch -M main
```

### Step 6: Add Remote Repository

```bash
git remote add origin https://github.com/ksdhruvateja/Decoding-Text-Intelligent-Classification-Techniques.git
```

**Note**: If remote already exists, use:
```bash
git remote set-url origin https://github.com/ksdhurateja/Decoding-Text-Intelligent-Classification-Techniques.git
```

### Step 7: Push to GitHub

```bash
git push -u origin main
```

## Complete One-Liner (After Initial Setup)

If you've already initialized git and added the remote:

```bash
git add . && git commit -m "Update: Ready for Vercel deployment" && git push origin main
```

## Verify Your Setup

### Check Git Status
```bash
git status
```

### Check Remote
```bash
git remote -v
```

### Check Branches
```bash
git branch
```

## Important Files to Verify

Before pushing, ensure these files exist:

- ✅ `.gitignore` - Properly configured
- ✅ `vercel.json` - Vercel configuration
- ✅ `api/index.py` - Serverless function
- ✅ `api/requirements.txt` - Python dependencies
- ✅ `frontend/.env.development` - Development API URL
- ✅ `frontend/.env.production` - Production API URL
- ✅ `package.json` - Root package.json
- ✅ `README.md` - Project documentation

## What Gets Committed

### ✅ Included:
- All source code (`backend/`, `frontend/src/`, `api/`)
- Configuration files (`vercel.json`, `package.json`)
- Documentation (`README.md`, `*.md`)
- Model checkpoints (if under size limit)

### ❌ Excluded (via .gitignore):
- `node_modules/`
- `venv/` and virtual environments
- `__pycache__/`
- `.env` files (but `.env.development` and `.env.production` are included)
- Build outputs (`build/`, `dist/`)
- Log files

## Troubleshooting

### Issue: "fatal: remote origin already exists"
```bash
git remote remove origin
git remote add origin https://github.com/ksdhurateja/Decoding-Text-Intelligent-Classification-Techniques.git
```

### Issue: "error: failed to push some refs"
```bash
# Pull first, then push
git pull origin main --allow-unrelated-histories
git push -u origin main
```

### Issue: Large files warning
If model files are too large:
```bash
# Check file sizes
git ls-files | xargs ls -lh | sort -k5 -hr | head -20

# Consider Git LFS for large files
git lfs install
git lfs track "*.pt"
git add .gitattributes
```

## Next Steps After Pushing

1. ✅ Go to [vercel.com](https://vercel.com)
2. ✅ Click "New Project"
3. ✅ Import your GitHub repository
4. ✅ Configure and deploy!

See [VERCEL_DEPLOYMENT.md](./VERCEL_DEPLOYMENT.md) for complete deployment guide.

---

**Ready to push? Run the commands above! 🚀**

