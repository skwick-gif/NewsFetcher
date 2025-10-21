# Setup Weekly Training and Daily Scanning Automation
# Run this in PowerShell as Administrator

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "Setting up ML Training & Scanning Automation" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

# ==================== TASK 1: WEEKLY DATA PREPARATION ====================
Write-Host "`n📅 Task 1: Weekly Data Preparation (Sunday 03:00 AM)" -ForegroundColor Yellow

$weeklyDataPrepAction = New-ScheduledTaskAction `
    -Execute "py" `
    -Argument "prepare_1000_stocks.py" `
    -WorkingDirectory "D:\Projects\NewsFetcher\MarketPulse"

$weeklyDataPrepTrigger = New-ScheduledTaskTrigger `
    -Weekly `
    -DaysOfWeek Sunday `
    -At "3:00AM"

$weeklyDataPrepSettings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RunOnlyIfNetworkAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 30)

Register-ScheduledTask `
    -TaskName "ML_WeeklyDataPreparation" `
    -Description "Prepare updated features for 933 stocks every Sunday at 03:00 AM (before training)" `
    -Action $weeklyDataPrepAction `
    -Trigger $weeklyDataPrepTrigger `
    -Settings $weeklyDataPrepSettings `
    -User $env:USERNAME `
    -RunLevel Limited `
    -Force

Write-Host "✅ Weekly data preparation task created!" -ForegroundColor Green

# ==================== TASK 2: WEEKLY MODEL TRAINING ====================
Write-Host "`n📅 Task 2: Weekly Model Training (Sunday 04:00 AM)" -ForegroundColor Yellow

$weeklyTrainingAction = New-ScheduledTaskAction `
    -Execute "py" `
    -Argument "train_on_1000_stocks.py" `
    -WorkingDirectory "D:\Projects\NewsFetcher\MarketPulse"

$weeklyTrainingTrigger = New-ScheduledTaskTrigger `
    -Weekly `
    -DaysOfWeek Sunday `
    -At "4:00AM"

$weeklyTrainingSettings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RunOnlyIfNetworkAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Hours 4)

Register-ScheduledTask `
    -TaskName "ML_WeeklyTraining" `
    -Description "Train ML models on 933 stocks every Sunday at 04:00 AM (after data preparation)" `
    -Action $weeklyTrainingAction `
    -Trigger $weeklyTrainingTrigger `
    -Settings $weeklyTrainingSettings `
    -User $env:USERNAME `
    -RunLevel Limited `
    -Force

Write-Host "✅ Weekly training task created!" -ForegroundColor Green

# ==================== TASK 3: DAILY SCANNING ====================
Write-Host "`n📅 Task 3: Daily Stock Scanning (Every day 17:10)" -ForegroundColor Yellow

$dailyScanAction = New-ScheduledTaskAction `
    -Execute "py" `
    -Argument "daily_scan.py" `
    -WorkingDirectory "D:\Projects\NewsFetcher\MarketPulse"

$dailyScanTrigger = New-ScheduledTaskTrigger `
    -Daily `
    -At "5:10PM"

$dailyScanSettings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RunOnlyIfNetworkAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Hours 1)

Register-ScheduledTask `
    -TaskName "ML_DailyScanning" `
    -Description "Scan all 10,825 stocks daily at 17:10 (after price updates) using trained ML models" `
    -Action $dailyScanAction `
    -Trigger $dailyScanTrigger `
    -Settings $dailyScanSettings `
    -User $env:USERNAME `
    -RunLevel Limited `
    -Force

Write-Host "✅ Daily scanning task created!" -ForegroundColor Green

# ==================== DISPLAY ALL TASKS ====================
Write-Host "`n📋 All ML Automation Tasks:" -ForegroundColor Cyan
Get-ScheduledTask -TaskName "ML_*", "StockData_*" | 
    Select-Object TaskName, State, @{L="Next Run";E={$_.Triggers[0].StartBoundary}} |
    Format-Table -AutoSize

Write-Host "`n📌 Management Commands:" -ForegroundColor Yellow
Write-Host "  View all tasks:           Get-ScheduledTask -TaskName 'ML_*'" -ForegroundColor White
Write-Host "  Run data prep now:        Start-ScheduledTask -TaskName 'ML_WeeklyDataPreparation'" -ForegroundColor White
Write-Host "  Run training now:         Start-ScheduledTask -TaskName 'ML_WeeklyTraining'" -ForegroundColor White
Write-Host "  Run scanning now:         Start-ScheduledTask -TaskName 'ML_DailyScanning'" -ForegroundColor White
Write-Host "  Disable data prep:        Disable-ScheduledTask -TaskName 'ML_WeeklyDataPreparation'" -ForegroundColor White
Write-Host "  Disable training:         Disable-ScheduledTask -TaskName 'ML_WeeklyTraining'" -ForegroundColor White
Write-Host "  Enable training:          Enable-ScheduledTask -TaskName 'ML_WeeklyTraining'" -ForegroundColor White
Write-Host "  Remove task:              Unregister-ScheduledTask -TaskName 'ML_WeeklyTraining' -Confirm:`$false" -ForegroundColor White
Write-Host "  Open Task Scheduler GUI:  taskschd.msc" -ForegroundColor White

Write-Host "`n📊 Complete Automation Schedule:" -ForegroundColor Cyan
Write-Host "  ┌─────────────────────────────────────────────────────────────┐" -ForegroundColor White
Write-Host "  │  📅 SUNDAY:                                                 │" -ForegroundColor White
Write-Host "  │     02:00 - Download fundamentals (existing)                │" -ForegroundColor Green
Write-Host "  │     03:00 - Prepare features for 933 stocks (NEW) 🆕       │" -ForegroundColor Yellow
Write-Host "  │     04:00 - Train ML models on updated data (NEW) 🆕        │" -ForegroundColor Yellow
Write-Host "  │                                                              │" -ForegroundColor White
Write-Host "  │  📅 DAILY (Mon-Sun):                                        │" -ForegroundColor White
Write-Host "  │     17:00 - Download latest prices (existing)               │" -ForegroundColor Green
Write-Host "  │     17:10 - Scan all 10,825 stocks (NEW) 🆕                 │" -ForegroundColor Yellow
Write-Host "  └─────────────────────────────────────────────────────────────┘" -ForegroundColor White

Write-Host "`n✅ Automation setup complete!" -ForegroundColor Green
Write-Host "`n💡 Key Points:" -ForegroundColor Cyan
Write-Host "   • Data prep runs BEFORE training to ensure fresh data" -ForegroundColor White
Write-Host "   • Training uses the most recent week of price data" -ForegroundColor White
Write-Host "   • Daily scans use updated prices from 17:00" -ForegroundColor White
Write-Host "   • All logs saved to logs/ directory" -ForegroundColor White

Write-Host "`n⚠️  Next steps:" -ForegroundColor Yellow
Write-Host "  1. Run initial training: py train_on_1000_stocks.py" -ForegroundColor White
Write-Host "  2. Test daily scan:      py daily_scan.py" -ForegroundColor White
Write-Host "  3. Check logs in:        logs/" -ForegroundColor White
