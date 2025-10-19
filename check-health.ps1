# System Health Check Script
# Verifies that Tariff Radar is working correctly

Write-Host "🏥 Tariff Radar - System Health Check" -ForegroundColor Cyan
Write-Host "=======================================" -ForegroundColor Cyan
Write-Host ""

$allGood = $true

# Test 1: Docker containers
Write-Host "1️⃣  Checking Docker containers..." -ForegroundColor Yellow
$containers = docker-compose ps --format json 2>&1 | ConvertFrom-Json
$runningCount = ($containers | Where-Object { $_.State -eq "running" }).Count

if ($runningCount -ge 5) {
    Write-Host "   ✅ $runningCount containers running" -ForegroundColor Green
} else {
    Write-Host "   ❌ Only $runningCount containers running (expected 6+)" -ForegroundColor Red
    $allGood = $false
}
Write-Host ""

# Test 2: API Health
Write-Host "2️⃣  Checking API health..." -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Uri "http://localhost:8000/health" -ErrorAction Stop
    if ($response.status -eq "healthy") {
        Write-Host "   ✅ API is healthy" -ForegroundColor Green
    } else {
        Write-Host "   ⚠️  API status: $($response.status)" -ForegroundColor Yellow
    }
} catch {
    Write-Host "   ❌ API is not responding" -ForegroundColor Red
    $allGood = $false
}
Write-Host ""

# Test 3: Database connection
Write-Host "3️⃣  Checking database..." -ForegroundColor Yellow
try {
    $dbCheck = docker-compose exec -T postgres psql -U postgres -d tariff_radar -c "\dt" 2>&1
    if ($dbCheck -match "articles") {
        Write-Host "   ✅ Database tables exist" -ForegroundColor Green
        
        # Count articles
        $articleCount = docker-compose exec -T postgres psql -U postgres -d tariff_radar -t -c "SELECT COUNT(*) FROM articles;" 2>&1
        $count = [int]($articleCount -replace '\D','')
        Write-Host "   📊 Articles in database: $count" -ForegroundColor Cyan
        
        if ($count -gt 0) {
            Write-Host "   ✅ Articles are being collected!" -ForegroundColor Green
        } else {
            Write-Host "   ⚠️  No articles yet (tasks may not have run)" -ForegroundColor Yellow
        }
    }
} catch {
    Write-Host "   ❌ Database connection failed" -ForegroundColor Red
    $allGood = $false
}
Write-Host ""

# Test 4: Celery Worker
Write-Host "4️⃣  Checking Celery worker..." -ForegroundColor Yellow
$workerLogs = docker-compose logs worker --tail 50 2>&1
if ($workerLogs -match "ready" -or $workerLogs -match "registered") {
    Write-Host "   ✅ Worker is running" -ForegroundColor Green
} else {
    Write-Host "   ⚠️  Worker status unclear" -ForegroundColor Yellow
}

# Check for ML components
if ($workerLogs -match "Embeddings disabled") {
    Write-Host "   ℹ️  Embeddings: DISABLED (expected)" -ForegroundColor Cyan
} else {
    Write-Host "   ✅ Embeddings: ENABLED" -ForegroundColor Green
}

if ($workerLogs -match "Classifier disabled") {
    Write-Host "   ℹ️  Classifier: DISABLED (expected)" -ForegroundColor Cyan
} else {
    Write-Host "   ✅ Classifier: ENABLED" -ForegroundColor Green
}
Write-Host ""

# Test 5: Celery Beat Scheduler
Write-Host "5️⃣  Checking task scheduler..." -ForegroundColor Yellow
$schedulerLogs = docker-compose logs scheduler --tail 20 2>&1
if ($schedulerLogs -match "beat.*starting" -or $schedulerLogs -match "Scheduler") {
    Write-Host "   ✅ Scheduler is running" -ForegroundColor Green
} else {
    Write-Host "   ⚠️  Scheduler status unclear" -ForegroundColor Yellow
}
Write-Host ""

# Test 6: Perplexity configuration
Write-Host "6️⃣  Checking Perplexity API..." -ForegroundColor Yellow
$envCheck = docker-compose exec -T api env 2>&1 | Select-String "PERPLEXITY_API_KEY"
if ($envCheck -match "pplx-") {
    Write-Host "   ✅ Perplexity API key configured" -ForegroundColor Green
} else {
    Write-Host "   ❌ Perplexity API key not found" -ForegroundColor Red
    $allGood = $false
}
Write-Host ""

# Test 7: Recent task execution
Write-Host "7️⃣  Checking task execution..." -ForegroundColor Yellow
$recentLogs = docker-compose logs worker --since 30m 2>&1
if ($recentLogs -match "Processed.*articles") {
    $processedLine = ($recentLogs | Select-String "Processed.*articles" | Select-Object -Last 1).ToString()
    Write-Host "   ✅ Tasks are executing" -ForegroundColor Green
    Write-Host "   📝 $processedLine" -ForegroundColor Cyan
} else {
    Write-Host "   ⚠️  No recent task execution (may need to wait)" -ForegroundColor Yellow
    Write-Host "   💡 Tasks run every 30 minutes" -ForegroundColor Cyan
}
Write-Host ""

# Summary
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
if ($allGood) {
    Write-Host "✅ System Status: HEALTHY" -ForegroundColor Green
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "🎉 Everything looks good!" -ForegroundColor Green
    Write-Host "   The system is collecting and analyzing articles." -ForegroundColor White
    Write-Host ""
    Write-Host "📊 Next steps:" -ForegroundColor Cyan
    Write-Host "   • View articles: http://localhost:8000/api/articles" -ForegroundColor White
    Write-Host "   • Check API docs: http://localhost:8000/docs" -ForegroundColor White
    Write-Host "   • Monitor logs: docker-compose logs -f worker" -ForegroundColor White
} else {
    Write-Host "⚠️  System Status: NEEDS ATTENTION" -ForegroundColor Yellow
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "🔧 Troubleshooting steps:" -ForegroundColor Yellow
    Write-Host "   1. Check full logs: docker-compose logs" -ForegroundColor White
    Write-Host "   2. Restart services: docker-compose restart" -ForegroundColor White
    Write-Host "   3. Rebuild: docker-compose build --no-cache" -ForegroundColor White
    Write-Host "   4. Check SETUP_PLAN.md for detailed instructions" -ForegroundColor White
}
Write-Host ""

# Offer detailed logs
$showLogs = Read-Host "Show detailed worker logs? (Y/N)"
if ($showLogs -eq "Y" -or $showLogs -eq "y") {
    Write-Host ""
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
    Write-Host "Worker Logs (last 50 lines):" -ForegroundColor Cyan
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
    docker-compose logs worker --tail 50
}

Write-Host ""
