@echo off
echo ========================================
echo   MarketPulse Auto Training Resume
echo ========================================
echo.

REM Set the working directory to NewsFetcher
cd /d "D:\Projects\NewsFetcher"

REM Check if we're in the right directory
if not exist "app\ml\train_model.py" (
    echo ERROR: train_model.py not found!
    echo Current directory: %CD%
    pause
    exit /b 1
)

echo Current directory: %CD%
echo.

REM Display current training status
echo Checking current training status...
py check_training_status.py
echo.

REM Optimized settings for speed without losing quality
set BATCH_SIZE=128
set MAX_EPOCHS=100

echo ========================================
echo Auto-Starting Training with Optimized Settings:
echo   Batch Size: %BATCH_SIZE% (larger for faster training)
echo   Max Epochs: %MAX_EPOCHS%
echo   Mode: Resume from where we left off
echo   Optimization: Mixed precision, larger batches
echo ========================================
echo.

REM Create log directory if it doesn't exist
if not exist "logs" mkdir logs

REM Generate timestamp for log file
for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set "dt=%%a"
set "timestamp=%dt:~0,8%_%dt:~8,6%"

echo Starting optimized training at %date% %time%
echo Log file: logs\auto_training_%timestamp%.log
echo.
echo Training will continue from where it stopped...
echo Press Ctrl+C to stop training safely (checkpoints will be saved)
echo.

REM Start training with optimized settings
py "D:\Projects\NewsFetcher\ml\scripts\train_model.py" --batch-size %BATCH_SIZE% --epochs %MAX_EPOCHS% 2>&1 | tee logs\auto_training_%timestamp%.log

echo.
echo ========================================
echo Training completed or stopped.
echo Check the log file: logs\auto_training_%timestamp%.log
echo ========================================
echo.

REM Show final status
echo Final training status:
py check_training_status.py

echo.
echo Press any key to exit...
pause >nul