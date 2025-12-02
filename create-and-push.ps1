# PowerShell script to help push to GitHub
# Make sure the repository exists on GitHub first!

Write-Host "=== GitHub Push Helper ===" -ForegroundColor Cyan
Write-Host ""

# Check git status
Write-Host "Checking git status..." -ForegroundColor Yellow
git status

Write-Host ""
Write-Host "Current remote URL:" -ForegroundColor Yellow
git remote get-url origin

Write-Host ""
Write-Host "=== IMPORTANT ===" -ForegroundColor Red
Write-Host "The repository doesn't exist on GitHub yet!" -ForegroundColor Red
Write-Host ""
Write-Host "Please follow these steps:" -ForegroundColor Yellow
Write-Host "1. Go to: https://github.com/new" -ForegroundColor White
Write-Host "2. Repository name: Decoding-Text-Intelligent-Classification-Techniques" -ForegroundColor White
Write-Host "3. Choose Public or Private" -ForegroundColor White
Write-Host "4. DO NOT check any boxes (no README, .gitignore, or license)" -ForegroundColor White
Write-Host "5. Click 'Create repository'" -ForegroundColor White
Write-Host ""
Write-Host "After creating the repository, press any key to push..." -ForegroundColor Green
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

Write-Host ""
Write-Host "Pushing to GitHub..." -ForegroundColor Yellow
git push -u origin main

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "✓ Successfully pushed to GitHub!" -ForegroundColor Green
    Write-Host "Repository URL: https://github.com/ksdhurateja/Decoding-Text-Intelligent-Classification-Techniques" -ForegroundColor Cyan
} else {
    Write-Host ""
    Write-Host "✗ Push failed. Make sure:" -ForegroundColor Red
    Write-Host "  - Repository exists on GitHub" -ForegroundColor Red
    Write-Host "  - You have access to the repository" -ForegroundColor Red
    Write-Host "  - Your GitHub username is correct: ksdhurateja" -ForegroundColor Red
}

