# Train All Progressive Models - Silent Mode (for Task Scheduler)
# Runs without confirmation
# Logs everything to file

$ErrorActionPreference = "Stop"
$logFile = "logs\progressive_training_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"

# Create logs directory
New-Item -ItemType Directory -Force -Path "logs" | Out-Null

# Start transcript
Start-Transcript -Path $logFile -Append

Write-Host "=========================================================================="
Write-Host "🚀 PROGRESSIVE ML TRAINING - SILENT MODE"
Write-Host "=========================================================================="
Write-Host "Started: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host ""

$startTime = Get-Date

try {
    # Change to script directory
    Set-Location $PSScriptRoot
    
    # Run the training
    Write-Host "🏃 Running training script..."
    py train_all_models_progressive.py
    
    if ($LASTEXITCODE -eq 0) {
        $endTime = Get-Date
        $duration = $endTime - $startTime
        
        Write-Host ""
        Write-Host "=========================================================================="
        Write-Host "✅ TRAINING COMPLETED"
        Write-Host "=========================================================================="
        Write-Host "Duration: $($duration.Hours)h $($duration.Minutes)m"
        Write-Host "Log: $logFile"
        Write-Host "=========================================================================="
        
        Stop-Transcript
        exit 0
    } else {
        throw "Training script failed with exit code $LASTEXITCODE"
    }
}
catch {
    Write-Host ""
    Write-Host "❌ ERROR: $_"
    Write-Host ""
    Stop-Transcript
    exit 1
}
