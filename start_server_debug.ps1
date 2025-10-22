# Start server and keep it running
$ErrorActionPreference = "Continue"

Write-Host "🚀 Starting MarketPulse Server..." -ForegroundColor Cyan
Write-Host "📍 Working directory: $(Get-Location)" -ForegroundColor Yellow

# Change to app directory
Set-Location "D:\Projects\NewsFetcher\app"

Write-Host "✅ Changed to: $(Get-Location)" -ForegroundColor Green

# Run server
Write-Host "🔄 Launching Python server..." -ForegroundColor Cyan
py main_realtime.py

Write-Host "Server exited with code: $LASTEXITCODE" -ForegroundColor Red
Read-Host "Press Enter to exit"
