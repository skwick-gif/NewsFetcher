# Quick Start Script for Tariff Radar
# Run this to start the system quickly

Write-Host "🚀 Tariff Radar - Quick Start" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""

# Check if Docker is running
Write-Host "⏳ Checking Docker..." -ForegroundColor Yellow
try {
    docker --version | Out-Null
    Write-Host "✅ Docker is installed" -ForegroundColor Green
} catch {
    Write-Host "❌ Docker is not running! Please start Docker Desktop first." -ForegroundColor Red
    exit 1
}

# Navigate to project directory
Set-Location -Path "D:\Projects\NewsFetcher\tariff-radar"
Write-Host "📁 Working directory: $(Get-Location)" -ForegroundColor Cyan
Write-Host ""

# Check if .env exists
if (-not (Test-Path ".env")) {
    Write-Host "❌ .env file not found!" -ForegroundColor Red
    Write-Host "   Please copy .env.example to .env and configure API keys" -ForegroundColor Yellow
    exit 1
}

# Check Perplexity API key
$perplexityKey = (Get-Content .env | Select-String "PERPLEXITY_API_KEY=").ToString().Split("=")[1].Trim()
if ($perplexityKey -match "your-actual|here") {
    Write-Host "⚠️  Warning: Perplexity API key looks invalid" -ForegroundColor Yellow
    Write-Host "   LLM analysis may not work. Update PERPLEXITY_API_KEY in .env" -ForegroundColor Yellow
} else {
    Write-Host "✅ Perplexity API key found: $($perplexityKey.Substring(0,15))..." -ForegroundColor Green
}
Write-Host ""

# Stop existing containers
Write-Host "🛑 Stopping existing containers..." -ForegroundColor Yellow
docker-compose down 2>&1 | Out-Null
Write-Host "✅ Stopped" -ForegroundColor Green
Write-Host ""

# Build images
Write-Host "🔨 Building Docker images (this may take 5-10 minutes)..." -ForegroundColor Yellow
$buildResult = docker-compose build 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Build successful!" -ForegroundColor Green
} else {
    Write-Host "❌ Build failed!" -ForegroundColor Red
    Write-Host $buildResult
    exit 1
}
Write-Host ""

# Start services
Write-Host "🚀 Starting services..." -ForegroundColor Yellow
docker-compose up -d
Start-Sleep -Seconds 5
Write-Host ""

# Check status
Write-Host "📊 Service Status:" -ForegroundColor Cyan
docker-compose ps
Write-Host ""

# Wait for API to be ready
Write-Host "⏳ Waiting for API to be ready..." -ForegroundColor Yellow
$maxAttempts = 30
$attempt = 0
$apiReady = $false

while ($attempt -lt $maxAttempts -and -not $apiReady) {
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -TimeoutSec 2 -ErrorAction SilentlyContinue
        if ($response.StatusCode -eq 200) {
            $apiReady = $true
            Write-Host "✅ API is ready!" -ForegroundColor Green
        }
    } catch {
        Write-Host "." -NoNewline -ForegroundColor Yellow
        Start-Sleep -Seconds 2
        $attempt++
    }
}

Write-Host ""

if (-not $apiReady) {
    Write-Host "⚠️  API didn't respond in time. Check logs:" -ForegroundColor Yellow
    Write-Host "   docker-compose logs api" -ForegroundColor Cyan
    Write-Host ""
}

# Display access URLs
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host "✅ Tariff Radar is Running!" -ForegroundColor Green
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host ""
Write-Host "🌐 Access URLs:" -ForegroundColor Cyan
Write-Host "   Dashboard:  http://localhost:8000" -ForegroundColor White
Write-Host "   API Docs:   http://localhost:8000/docs" -ForegroundColor White
Write-Host "   Health:     http://localhost:8000/health" -ForegroundColor White
Write-Host ""
Write-Host "📊 Useful Commands:" -ForegroundColor Cyan
Write-Host "   View worker logs:  docker-compose logs -f worker" -ForegroundColor White
Write-Host "   View all logs:     docker-compose logs -f" -ForegroundColor White
Write-Host "   Check DB:          docker-compose exec postgres psql -U postgres -d tariff_radar" -ForegroundColor White
Write-Host "   Stop system:       docker-compose down" -ForegroundColor White
Write-Host ""
Write-Host "⏰ The system runs background tasks every 30 minutes" -ForegroundColor Yellow
Write-Host "   Check 'docker-compose logs -f worker' to see article processing" -ForegroundColor Yellow
Write-Host ""
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan

# Offer to open browser
$openBrowser = Read-Host "Open dashboard in browser? (Y/N)"
if ($openBrowser -eq "Y" -or $openBrowser -eq "y") {
    Start-Process "http://localhost:8000/health"
    Start-Process "http://localhost:8000/docs"
    Write-Host "✅ Browser opened!" -ForegroundColor Green
}

Write-Host ""
Write-Host "💡 Tip: Run 'docker-compose logs -f worker' to see live processing" -ForegroundColor Cyan
Write-Host ""
