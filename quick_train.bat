@echo off
REM Quick training launcher - resume with default settings

echo ========================================
echo   MarketPulse Quick Training Start
echo ========================================

cd /d "D:\Projects\NewsFetcher"

echo Current Status:
py check_training_status.py
echo.

echo Starting training with default settings:
echo - Batch Size: 32
echo - Resume from checkpoint: Yes
echo - Max Epochs: 100
echo.
echo Press Ctrl+C to stop safely...
echo.

REM Create timestamp for log
for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set "dt=%%a"
set "timestamp=%dt:~0,8%_%dt:~8,6%"

py "D:\Projects\NewsFetcher\app\ml\train_model.py" 2>&1 | tee logs\quick_training_%timestamp%.log

echo.
echo Training session ended.
pause