# Minimal starter script for MarketPulse
# 1) Start backend
# 2) Wait (configurable)
# 3) Start frontend
# 4) Open browser to dashboard

param(
    [int]$waitSeconds = 5
)

$project = "D:\\Projects\\NewsFetcher"

Write-Host "Starting Backend..."
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd $project; py app\\main_production.py"

Write-Host "Waiting $waitSeconds seconds for backend to initialize..."
Start-Sleep -Seconds $waitSeconds

Write-Host "Starting Frontend..."
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd $project; py app\\server.py"

Write-Host "Opening browser to http://localhost:5000"
Start-Process "http://localhost:5000"

Write-Host "Done. Watch the two new windows for logs."
