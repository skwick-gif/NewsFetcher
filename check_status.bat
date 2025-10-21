@echo off
REM Check training status only

echo ========================================
echo   MarketPulse Training Status Check
echo ========================================

cd /d "D:\Projects\NewsFetcher"

py check_training_status.py

echo.
echo Press any key to exit...
pause >nul