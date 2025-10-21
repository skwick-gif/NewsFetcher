# MarketPulse - Start Integrated Server
# הפעלת המערכת המשולבת עם כל הפיצ'רים

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "🚀 MarketPulse - Integrated System v2.1.0" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Set working directory
Set-Location D:\Projects\NewsFetcher

# Start Enhanced FastAPI Backend (port 8000) - INTEGRATED VERSION
Write-Host "🎯 Starting INTEGRATED FastAPI Backend on port 8000..." -ForegroundColor Yellow
Write-Host "   ✅ Real-time alerts + WebSocket support" -ForegroundColor Green
Write-Host "   ✅ ML/AI predictions" -ForegroundColor Green  
Write-Host "   ✅ Financial data + market streaming" -ForegroundColor Green
Write-Host "   ✅ Enhanced endpoints (40+ APIs)" -ForegroundColor Green
Write-Host ""
Start-Process powershell -ArgumentList "-NoExit", "-Command", "py app\main_realtime.py"
Start-Sleep -Seconds 3

# Start Flask Frontend (port 5000)
Write-Host "🎨 Starting Flask Frontend on port 5000..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "py app\server.py"
Start-Sleep -Seconds 2

Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host "✅ INTEGRATED MarketPulse System Online!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host ""
Write-Host "🎯 MAIN API: http://localhost:8000" -ForegroundColor White
Write-Host "   📖 API Docs: http://localhost:8000/docs" -ForegroundColor Gray
Write-Host "   🔌 WebSocket: ws://localhost:8000/ws/alerts" -ForegroundColor Gray
Write-Host "   📊 Dashboard: http://localhost:8000/dashboard" -ForegroundColor Gray
Write-Host ""
Write-Host "🎨 Flask UI: http://localhost:5000" -ForegroundColor White
Write-Host "   🌐 Alternative interface" -ForegroundColor Gray
Write-Host ""
Write-Host "🎉 Features Active:" -ForegroundColor Cyan
Write-Host "   ✅ Real-time market monitoring" -ForegroundColor Green
Write-Host "   ✅ AI-powered predictions" -ForegroundColor Green
Write-Host "   ✅ WebSocket alerts" -ForegroundColor Green
Write-Host "   ✅ Financial data streaming" -ForegroundColor Green
Write-Host "   ✅ ML model training" -ForegroundColor Green
Write-Host ""
Write-Host "Press Ctrl+C in each window to stop the servers" -ForegroundColor DarkGray
