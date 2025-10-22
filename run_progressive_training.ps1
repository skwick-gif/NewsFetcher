# Train All Progressive Models (Transformer & CNN)
# Skips stocks that already have LSTM trained
# Run this overnight for full training

Write-Host "=" -NoNewline
Write-Host ("=" * 69)
Write-Host "🚀 PROGRESSIVE ML TRAINING - ALL STOCKS" -ForegroundColor Cyan
Write-Host "=" -NoNewline
Write-Host ("=" * 69)
Write-Host ""

$startTime = Get-Date
Write-Host "📅 Started: $($startTime.ToString('yyyy-MM-dd HH:mm:ss'))" -ForegroundColor Green
Write-Host ""

Write-Host "🎯 Training Plan:" -ForegroundColor Yellow
Write-Host "   • Models: Transformer, CNN" -ForegroundColor White
Write-Host "   • Mode: Progressive (1d → 7d → 30d)" -ForegroundColor White
Write-Host "   • Skips stocks with existing models" -ForegroundColor White
Write-Host "   • Estimated time: 6-12 hours" -ForegroundColor White
Write-Host ""

# Confirm
Write-Host "⚠️  This will run for several hours. Continue? (Y/N): " -ForegroundColor Yellow -NoNewline
$confirm = Read-Host

if ($confirm -ne 'Y' -and $confirm -ne 'y') {
    Write-Host "❌ Cancelled" -ForegroundColor Red
    exit
}

Write-Host ""
Write-Host "🏃 Starting training..." -ForegroundColor Green
Write-Host ""

# Run the training script
try {
    py train_all_models_progressive.py
    
    if ($LASTEXITCODE -eq 0) {
        $endTime = Get-Date
        $duration = $endTime - $startTime
        
        Write-Host ""
        Write-Host "=" -NoNewline
        Write-Host ("=" * 69)
        Write-Host "✅ TRAINING COMPLETED SUCCESSFULLY" -ForegroundColor Green
        Write-Host "=" -NoNewline
        Write-Host ("=" * 69)
        Write-Host ""
        Write-Host "⏱️  Duration: $($duration.Hours)h $($duration.Minutes)m $($duration.Seconds)s" -ForegroundColor Cyan
        Write-Host "📝 Check logs/ folder for detailed results" -ForegroundColor Cyan
        Write-Host "💾 Models saved in: app/ml/models/" -ForegroundColor Cyan
        Write-Host ""
    } else {
        Write-Host ""
        Write-Host "❌ TRAINING FAILED - Check logs for details" -ForegroundColor Red
        Write-Host ""
        exit 1
    }
}
catch {
    Write-Host ""
    Write-Host "❌ ERROR: $_" -ForegroundColor Red
    Write-Host ""
    exit 1
}

Write-Host "Press any key to exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
