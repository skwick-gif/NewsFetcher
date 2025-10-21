# MarketPulse - Start Both Servers
# הפעלת FastAPI Backend + Flask Frontend במקביל

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "🚀 MarketPulse - Starting Services" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Set working directory
Set-Location D:\Projects\NewsFetcher

# Start FastAPI Backend (port 8000)
Write-Host "📊 Starting FastAPI Backend on port 8000..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "py app\main_simple_backend.py"
Start-Sleep -Seconds 3

# Start Flask Frontend (port 5000)
Write-Host "🎨 Starting Flask Frontend on port 5000..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "py app\server.py"
Start-Sleep -Seconds 2

Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host "✅ Both servers are starting!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host ""
Write-Host "📊 FastAPI Backend: http://localhost:8000" -ForegroundColor White
Write-Host "   📖 API Docs: http://localhost:8000/docs" -ForegroundColor Gray
Write-Host ""
Write-Host "🎨 Flask Dashboard: http://localhost:5000" -ForegroundColor White
Write-Host "   🌐 Open this in your browser!" -ForegroundColor Gray
Write-Host ""
Write-Host "Press Ctrl+C in each window to stop the servers" -ForegroundColor DarkGray
