@echo off
REM Setup environment files for frontend (Windows)

echo Setting up environment files...

REM Create .env.development
echo REACT_APP_API_URL=http://localhost:5000/api > frontend\.env.development

REM Create .env.production
echo REACT_APP_API_URL=/api > frontend\.env.production

echo ✓ Environment files created!
echo   - frontend/.env.development (for local development)
echo   - frontend/.env.production (for Vercel deployment)

