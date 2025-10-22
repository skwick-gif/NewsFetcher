@echo off
REM ================================================================
REM Fundamentals Download - NewsFetcher
REM ================================================================
REM This script runs the same function as clicking "Download Fundamentals"
REM button in the Data Management interface of the application
REM ================================================================

echo.
echo ================================
echo  📊 NewsFetcher - Fundamentals Download
echo ================================
echo.

REM Check if we're in the correct directory
if not exist "ToUse\download_fundamentals.py" (
    echo ❌ ERROR: File ToUse\download_fundamentals.py not found
    echo    Make sure you run this file from the main project directory: NewsFetcher
    echo.
    pause
    exit /b 1
)

REM Check if Python is installed
py --version >nul 2>&1
if errorlevel 1 (
    echo ❌ ERROR: Python is not installed or not found in PATH
    echo    Install Python or make sure it's available in Command Line
    echo.
    pause
    exit /b 1
)

echo ✅ Python is available
echo ✅ Download files found
echo.

REM Save current working directory
pushd "%~dp0"

echo 🚀 Starting fundamentals data download...
echo.
echo ⏳ This may take several minutes depending on the number of stocks
echo    Data will be saved to folder: stock_data\
echo.

REM Run the script with real-time output
py ToUse\download_fundamentals.py

REM Check results
if errorlevel 1 (
    echo.
    echo ❌ Fundamentals download completed with errors
    echo    Check the logs above for more details
    echo.
) else (
    echo.
    echo ✅ Fundamentals download completed successfully!
    echo    Data saved to folder: stock_data\
    echo.
)

REM Return to original working directory
popd

echo 📝 Additional logs available in file: stock_fetcher.log
echo.
echo Press any key to close the window...
pause >nul