# Windows Task Scheduler Setup Script
# Run this in PowerShell as Administrator

# Task 1: Daily Price Updates (5:00 PM every day)
$dailyAction = New-ScheduledTaskAction `
    -Execute "py" `
    -Argument "run_daily_update.py" `
    -WorkingDirectory "D:\Projects\NewsFetcher\MarketPulse\ToUseForData"

$dailyTrigger = New-ScheduledTaskTrigger `
    -Daily `
    -At "5:00PM"

$dailySettings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RunOnlyIfNetworkAvailable

Register-ScheduledTask `
    -TaskName "StockData_DailyPriceUpdate" `
    -Description "Download daily stock prices after market close (5:00 PM)" `
    -Action $dailyAction `
    -Trigger $dailyTrigger `
    -Settings $dailySettings `
    -User $env:USERNAME `
    -RunLevel Limited `
    -Force

Write-Host "✅ Daily price update task created successfully!" -ForegroundColor Green

# Task 2: Weekly Fundamentals (Sunday 2:00 AM)
$weeklyAction = New-ScheduledTaskAction `
    -Execute "py" `
    -Argument "run_weekly_fundamentals.py" `
    -WorkingDirectory "D:\Projects\NewsFetcher\MarketPulse\ToUseForData"

$weeklyTrigger = New-ScheduledTaskTrigger `
    -Weekly `
    -DaysOfWeek Sunday `
    -At "2:00AM"

$weeklySettings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RunOnlyIfNetworkAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Hours 8)

Register-ScheduledTask `
    -TaskName "StockData_WeeklyFundamentals" `
    -Description "Scrape comprehensive fundamental data every Sunday at 2:00 AM" `
    -Action $weeklyAction `
    -Trigger $weeklyTrigger `
    -Settings $weeklySettings `
    -User $env:USERNAME `
    -RunLevel Limited `
    -Force

Write-Host "✅ Weekly fundamentals task created successfully!" -ForegroundColor Green

# Display created tasks
Write-Host "`n📋 Scheduled Tasks Summary:" -ForegroundColor Cyan
Get-ScheduledTask -TaskName "StockData_*" | Format-Table TaskName, State, @{L="Next Run";E={$_.Triggers[0].StartBoundary}}

Write-Host "`n📌 To manage tasks:" -ForegroundColor Yellow
Write-Host "  - Open Task Scheduler: taskschd.msc"
Write-Host "  - View logs: D:\Projects\NewsFetcher\MarketPulse\logs"
Write-Host "  - Test daily run: Start-ScheduledTask -TaskName 'StockData_DailyPriceUpdate'"
Write-Host "  - Test weekly run: Start-ScheduledTask -TaskName 'StockData_WeeklyFundamentals'"
Write-Host "  - Disable: Disable-ScheduledTask -TaskName 'StockData_DailyPriceUpdate'"
Write-Host "  - Remove: Unregister-ScheduledTask -TaskName 'StockData_DailyPriceUpdate' -Confirm:`$false"
