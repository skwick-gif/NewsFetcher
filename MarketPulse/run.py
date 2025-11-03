# MarketPulse Runner Script
# Run from NewsFetcher/ directory

import sys
import os

# Ensure project root is on the path so ``app.main`` can be imported
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)

# Now import and run the modular FastAPI application
from app.main import app
import uvicorn

if __name__ == "__main__":
    print("=" * 80)
    print("🚀 Starting MarketPulse Financial Intelligence Platform")
    print("=" * 80)
    print("   Dashboard: http://localhost:8000")
    print("   WebSocket: ws://localhost:8000/ws/alerts")
    print("   API Docs: http://localhost:8000/docs")
    print("   Health: http://localhost:8000/health")
    print("=" * 80)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
