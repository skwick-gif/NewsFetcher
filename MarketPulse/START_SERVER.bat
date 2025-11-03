@echo off
echo ================================================================================
echo Starting MarketPulse Financial Intelligence Platform
echo ================================================================================
cd /d "D:\Projects\NewsFetcher"
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
pause
