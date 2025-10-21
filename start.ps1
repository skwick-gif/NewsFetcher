# MarketPulse Dashboard - Startup Script# MarketPulse Dashboard - Startup Script

# This script starts both Backend and Frontend servers# This script starts both Backend and Frontend servers



Write-Host "`n========================================" -ForegroundColor CyanWrite-Host "`n========================================" -ForegroundColor Cyan

Write-Host "  MarketPulse Dashboard - Startup" -ForegroundColor CyanWrite-Host "  MarketPulse Dashboard - Startup" -ForegroundColor Cyan

Write-Host "========================================`n" -ForegroundColor CyanWrite-Host "========================================`n" -ForegroundColor Cyan



# Kill existing Python processes on ports 8000 and 5000# Kill existing Python processes on ports 8000 and 5000

Write-Host "Checking for existing processes..." -ForegroundColor YellowWrite-Host "🔍 Checking for existing processes..." -ForegroundColor Yellow

$backend = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue | Select-Object -First 1$backend = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue | Select-Object -First 1

$frontend = Get-NetTCPConnection -LocalPort 5000 -ErrorAction SilentlyContinue | Select-Object -First 1$frontend = Get-NetTCPConnection -LocalPort 5000 -ErrorAction SilentlyContinue | Select-Object -First 1



if ($backend) {if ($backend) {

    Write-Host "Port 8000 in use - killing process $($backend.OwningProcess)" -ForegroundColor Red    Write-Host "⚠️  Port 8000 in use - killing process $($backend.OwningProcess)" -ForegroundColor Red

    Stop-Process -Id $backend.OwningProcess -Force -ErrorAction SilentlyContinue    Stop-Process -Id $backend.OwningProcess -Force -ErrorAction SilentlyContinue

    Start-Sleep -Seconds 2    Start-Sleep -Seconds 2

}}



if ($frontend) {if ($frontend) {

    Write-Host "Port 5000 in use - killing process $($frontend.OwningProcess)" -ForegroundColor Red    Write-Host "⚠️  Port 5000 in use - killing process $($frontend.OwningProcess)" -ForegroundColor Red

    Stop-Process -Id $frontend.OwningProcess -Force -ErrorAction SilentlyContinue    Stop-Process -Id $frontend.OwningProcess -Force -ErrorAction SilentlyContinue

    Start-Sleep -Seconds 2    Start-Sleep -Seconds 2

}}



# Start Backend (FastAPI on port 8000)# Start Backend (FastAPI on port 8000)

Write-Host "`nStarting Backend (Port 8000)..." -ForegroundColor GreenWrite-Host "`n� Starting Backend (Port 8000)..." -ForegroundColor Green

Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd D:\Projects\NewsFetcher\MarketPulse; Write-Host '=== BACKEND SERVER (Port 8000) ===' -ForegroundColor Cyan; py app\main_production.py"Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd D:\Projects\NewsFetcher\MarketPulse; Write-Host '=== BACKEND SERVER (Port 8000) ===' -ForegroundColor Cyan; py app\main_production.py"

Start-Sleep -Seconds 5Start-Sleep -Seconds 5



# Verify Backend started# Verify Backend started

Write-Host "Verifying Backend..." -ForegroundColor YellowWrite-Host "⏳ Verifying Backend..." -ForegroundColor Yellow

try {try {

    $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -TimeoutSec 3 -ErrorAction Stop    $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -TimeoutSec 3 -ErrorAction Stop

    Write-Host "Backend running on http://localhost:8000" -ForegroundColor Green    Write-Host "✅ Backend running on http://localhost:8000" -ForegroundColor Green

} catch {} catch {

    Write-Host "Backend failed to start! Check the Backend window for errors." -ForegroundColor Red    Write-Host "❌ Backend failed to start! Check the Backend window for errors." -ForegroundColor Red

    Write-Host "Press any key to exit..."    Write-Host "Press any key to exit..."

    $null = $Host.UI.RawUI.ReadKey('NoEcho,IncludeKeyDown')    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

    exit 1    exit 1

}}



# Start Frontend (Flask on port 5000)# Start Frontend (Flask on port 5000)

Write-Host "`nStarting Frontend (Port 5000)..." -ForegroundColor GreenWrite-Host "`n🚀 Starting Frontend (Port 5000)..." -ForegroundColor Green

Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd D:\Projects\NewsFetcher\MarketPulse; Write-Host '=== FRONTEND SERVER (Port 5000) ===' -ForegroundColor Green; py app\server.py"Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd D:\Projects\NewsFetcher\MarketPulse; Write-Host '=== FRONTEND SERVER (Port 5000) ===' -ForegroundColor Green; py app\server.py"

Start-Sleep -Seconds 5Start-Sleep -Seconds 5



# Verify Frontend started# Verify Frontend started

Write-Host "Verifying Frontend..." -ForegroundColor YellowWrite-Host "⏳ Verifying Frontend..." -ForegroundColor Yellow

try {try {

    $response = Invoke-WebRequest -Uri "http://localhost:5000/health" -TimeoutSec 3 -ErrorAction Stop    $response = Invoke-WebRequest -Uri "http://localhost:5000/health" -TimeoutSec 3 -ErrorAction Stop

    Write-Host "Frontend running on http://localhost:5000" -ForegroundColor Green    Write-Host "✅ Frontend running on http://localhost:5000" -ForegroundColor Green

} catch {} catch {

    Write-Host "Frontend failed to start! Check the Frontend window for errors." -ForegroundColor Red    Write-Host "❌ Frontend failed to start! Check the Frontend window for errors." -ForegroundColor Red

    Write-Host "Press any key to exit..."    Write-Host "Press any key to exit..."

    $null = $Host.UI.RawUI.ReadKey('NoEcho,IncludeKeyDown')    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

    exit 1    exit 1

}}



# Success!# Success!

Write-Host "`n========================================" -ForegroundColor CyanWrite-Host "`n========================================" -ForegroundColor Cyan

Write-Host "  MarketPulse is ready!" -ForegroundColor GreenWrite-Host "  ✅ MarketPulse is ready!" -ForegroundColor Green

Write-Host "========================================" -ForegroundColor CyanWrite-Host "========================================" -ForegroundColor Cyan

Write-Host "`nDashboard: http://localhost:5000" -ForegroundColor YellowWrite-Host "`n📊 Dashboard: http://localhost:5000" -ForegroundColor Yellow

Write-Host "Backend API: http://localhost:8000/docs" -ForegroundColor YellowWrite-Host "🔧 Backend API: http://localhost:8000/docs" -ForegroundColor Yellow

Write-Host "`nIMPORTANT: Do NOT close the Backend/Frontend windows!" -ForegroundColor RedWrite-Host "`n⚠️  IMPORTANT: Do NOT close the Backend/Frontend windows!" -ForegroundColor Red

Write-Host "Closing them will stop the servers.`n" -ForegroundColor RedWrite-Host "    Closing them will stop the servers.`n" -ForegroundColor Red



# Open browser# Open browser

$openBrowser = Read-Host "Open dashboard in browser? (Y/N)"$openBrowser = Read-Host "Open dashboard in browser? (Y/N)"

if ($openBrowser -eq "Y" -or $openBrowser -eq "y") {if ($openBrowser -eq "Y" -or $openBrowser -eq "y") {

    Start-Process "http://localhost:5000"    Start-Process "http://localhost:5000"

    Write-Host "Browser opened!" -ForegroundColor Green    Write-Host "✅ Browser opened!" -ForegroundColor Green

}}



Write-Host "`nPress any key to exit this launcher..." -ForegroundColor GrayWrite-Host "`nPress any key to exit this launcher..." -ForegroundColor Gray

Write-Host "(The servers will keep running in their windows)" -ForegroundColor GrayWrite-Host "(The servers will keep running in their windows)" -ForegroundColor Gray

$null = $Host.UI.RawUI.ReadKey('NoEcho,IncludeKeyDown')$null = $Host.UI.RawUI.ReadKey('NoEcho,IncludeKeyDown')

