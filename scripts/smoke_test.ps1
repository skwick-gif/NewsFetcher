param(
    [string]$BaseUrl = "http://localhost:8000",
    [string]$Symbol = "AAPL",
    [switch]$StartServer = $true,
    [int]$WaitSeconds = 8
)

Write-Host "Running API smoke tests against $BaseUrl" -ForegroundColor Cyan

function Test-Endpoint {
    param([string]$Path)
    try {
        $res = Invoke-RestMethod -Uri "$BaseUrl$Path" -TimeoutSec 30
        Write-Host "OK  $Path" -ForegroundColor Green
        return $res
    } catch {
        Write-Host "FAIL $Path : $($_.Exception.Message)" -ForegroundColor Red
        return $null
    }
}

${serverJob} = $null
if ($StartServer) {
    try {
        $repoRoot = (Split-Path $PSScriptRoot -Parent)
        Write-Host "Starting backend server in background..." -ForegroundColor DarkCyan
        $serverJob = Start-Job -ScriptBlock {
            param($root)
            Set-Location $root
            py run.py
        } -ArgumentList $repoRoot | Out-Null
        Start-Sleep -Seconds $WaitSeconds
    } catch {
        Write-Host "Failed to start server job: $($_.Exception.Message)" -ForegroundColor Red
    }
}

# Health
$prov = Test-Endpoint "/api/sentiment/providers"
if ($prov -ne $null) { $prov | ConvertTo-Json -Depth 5 | Out-Null }

# Market data (optional)
Test-Endpoint "/api/financial/market-indices" | Out-Null

# News sentiment
$newsSent = Test-Endpoint "/api/news/sentiment/$Symbol"
if ($newsSent -ne $null) {
    if ($newsSent.data -and $newsSent.data.Count -gt 0) {
        Write-Host "News sentiment rows: $($newsSent.data.Count)" -ForegroundColor Yellow
    } elseif ($newsSent.message) {
        Write-Host $newsSent.message -ForegroundColor DarkYellow
    }
}

# Recent news
$news = Test-Endpoint "/api/news/$Symbol?limit=5"
if ($news -ne $null) {
    Write-Host "Articles: $($news.articles.Count)" -ForegroundColor Yellow
}

Write-Host "Smoke tests completed." -ForegroundColor Cyan

if ($serverJob) {
    try {
        Stop-Job $serverJob.Id -ErrorAction SilentlyContinue
        Receive-Job $serverJob.Id | Out-Null
        Remove-Job $serverJob.Id -Force -ErrorAction SilentlyContinue
        Write-Host "Background server stopped." -ForegroundColor DarkCyan
    } catch {
        Write-Host "Failed to stop background server: $($_.Exception.Message)" -ForegroundColor Yellow
    }
}
