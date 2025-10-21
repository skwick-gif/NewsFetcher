# Test Dashboard Integration
# בדיקה מהירה שהכל עובד

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "🧪 MarketPulse Integration Test" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Test 1: Backend Health
Write-Host "[Test 1] Backend Health Check..." -ForegroundColor Yellow
try {
    $backend = Invoke-RestMethod -Uri "http://localhost:8000/health"
    if ($backend.status -eq "healthy") {
        Write-Host "  ✅ Backend is healthy" -ForegroundColor Green
    } else {
        Write-Host "  ❌ Backend unhealthy" -ForegroundColor Red
    }
} catch {
    Write-Host "  ❌ Backend not responding" -ForegroundColor Red
    exit 1
}

# Test 2: Frontend Health
Write-Host "[Test 2] Frontend Health Check..." -ForegroundColor Yellow
try {
    $frontend = Invoke-RestMethod -Uri "http://localhost:5000/health"
    if ($frontend.status -eq "healthy") {
        Write-Host "  ✅ Frontend is healthy" -ForegroundColor Green
    } else {
        Write-Host "  ❌ Frontend unhealthy" -ForegroundColor Red
    }
} catch {
    Write-Host "  ❌ Frontend not responding" -ForegroundColor Red
    exit 1
}

# Test 3: Market Indices API
Write-Host "[Test 3] Market Indices API..." -ForegroundColor Yellow
try {
    $indices = Invoke-RestMethod -Uri "http://localhost:8000/api/financial/market-indices"
    if ($indices.status -eq "success") {
        $count = ($indices.data | Get-Member -MemberType NoteProperty).Count
        Write-Host "  ✅ Received $count market indices" -ForegroundColor Green
    }
} catch {
    Write-Host "  ❌ Market indices API failed" -ForegroundColor Red
}

# Test 4: Market Sentiment API
Write-Host "[Test 4] Market Sentiment API..." -ForegroundColor Yellow
try {
    $sentiment = Invoke-RestMethod -Uri "http://localhost:8000/api/financial/market-sentiment"
    if ($sentiment.status -eq "success") {
        Write-Host "  ✅ Sentiment: $($sentiment.data.label) ($($sentiment.data.score)%)" -ForegroundColor Green
    }
} catch {
    Write-Host "  ❌ Market sentiment API failed" -ForegroundColor Red
}

# Test 5: Frontend Proxy (Market Indices)
Write-Host "[Test 5] Frontend Proxy Test..." -ForegroundColor Yellow
try {
    $proxyIndices = Invoke-RestMethod -Uri "http://localhost:5000/api/financial/market-indices"
    if ($proxyIndices.status -eq "success") {
        Write-Host "  ✅ Frontend proxy works correctly" -ForegroundColor Green
    }
} catch {
    Write-Host "  ❌ Frontend proxy failed" -ForegroundColor Red
}

# Test 6: AI Status
Write-Host "[Test 6] AI Models Status..." -ForegroundColor Yellow
try {
    $aiStatus = Invoke-RestMethod -Uri "http://localhost:8000/api/ai/status"
    if ($aiStatus.status -eq "success") {
        Write-Host "  ✅ AI models available" -ForegroundColor Green
    }
} catch {
    Write-Host "  ❌ AI status failed" -ForegroundColor Red
}

# Test 7: Stock Analysis
Write-Host "[Test 7] Stock Analysis (AAPL)..." -ForegroundColor Yellow
try {
    $stock = Invoke-RestMethod -Uri "http://localhost:8000/api/ai/comprehensive-analysis/AAPL"
    if ($stock.status -eq "success") {
        Write-Host "  ✅ Stock analysis works: $($stock.data.recommendation) (Confidence: $($stock.data.confidence))" -ForegroundColor Green
    }
} catch {
    Write-Host "  ❌ Stock analysis failed: $_" -ForegroundColor Red
}

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "✅ Integration Tests Complete!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Open Dashboard: http://localhost:5000" -ForegroundColor White
Write-Host "API Docs: http://localhost:8000/docs" -ForegroundColor Gray
Write-Host ""
