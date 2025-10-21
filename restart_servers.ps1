# Restart Both Servers
Write-Host "Stopping all Python processes..." -ForegroundColor Yellow
Get-Process | Where-Object {$_.ProcessName -like "*python*"} | Stop-Process -Force
Start-Sleep -Seconds 2

Write-Host "Starting Backend..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd D:\Projects\NewsFetcher\MarketPulse; py app\main_production.py"
Start-Sleep -Seconds 5

Write-Host "Starting Frontend..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd D:\Projects\NewsFetcher\MarketPulse; py app\server.py"
Start-Sleep -Seconds 3

Write-Host ""
Write-Host "Servers restarted!" -ForegroundColor Green
Write-Host "Backend: http://localhost:8000" -ForegroundColor White
Write-Host "Frontend: http://localhost:5000" -ForegroundColor White
