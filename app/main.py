"""
Main entry point for the modular FastAPI application
Combines all routers into a single application
"""
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from app.core.lifespan import lifespan

# Import all the routers we've created
from app.routes.financial.market_data import router as financial_market_router
from app.routes.financial.historical import router as financial_historical_router
from app.routes.financial.analysis import router as financial_analysis_router
from app.api.routers.financial import router as api_financial_router
from app.routes.ai.analysis import router as ai_analysis_router
from app.routes.ai.market_intelligence import router as ai_market_intelligence_router
from app.api.routers.scanner import router as scanner_router
from app.api.routers.rl import router as rl_router
from app.api.routers.ml import router as ml_router
from app.api.routers.news import router as news_router
# Note: Removed duplicate ML and RL routers from app.routes.* to avoid conflicts
from app.routes.system import router as system_router
from app.routes.websocket import router as websocket_router

# Create FastAPI app with lifespan
app = FastAPI(
    title="NewsFetcher Modular API",
    description="Modular FastAPI backend for NewsFetcher with real-time market data, AI analysis, ML predictions, and RL trading",
    version="1.0.0",
    lifespan=lifespan
)

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="app/templates")

# Include all routers
# Financial endpoints
app.include_router(financial_market_router, prefix="/api", tags=["Financial"])
app.include_router(financial_historical_router, prefix="/api", tags=["Financial"])
app.include_router(financial_analysis_router, prefix="/api", tags=["Financial"])
app.include_router(api_financial_router, prefix="/api/financial", tags=["Financial"])

# AI endpoints
app.include_router(ai_analysis_router, prefix="/api", tags=["AI"])
app.include_router(ai_market_intelligence_router, prefix="/api", tags=["AI"])

# ML endpoints - use the main ML router with proper prefix
app.include_router(ml_router, tags=["ML"])  # This has prefix="/api/ml" built-in
# Note: ml_predictions_router, ml_training_router, ml_backtesting_router are duplicated by ml_router

# RL endpoints - use the main RL router with proper prefix  
app.include_router(rl_router, tags=["RL"])  # This has prefix="/api/rl" built-in
# Note: rl_simulation_router, rl_training_router are duplicated by rl_router

# Scanner and other endpoints
app.include_router(scanner_router, tags=["Scanner"])
app.include_router(news_router, tags=["News"])
app.include_router(system_router, tags=["System"])
app.include_router(websocket_router, tags=["WebSocket"])

# Future IBKR Integration (as per MODULARIZATION_PLAN.md):
# When IBKR connection is implemented, add here:
# from app.api.routers.ibkr import router as ibkr_router
# app.include_router(ibkr_router, prefix="/api/ibkr", tags=["IBKR Trading"])

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Modular FastAPI app is running"}

@app.get("/api/system/backend-info")
async def backend_info():
    """Backend system information for Flask frontend validation"""
    return {
        "status": "success",
        "backend_type": "modular_fastapi",
        "version": "2.0",
        "endpoints": {
            "ml_progressive_status": "/api/ml/progressive/status",
            "rl_status": "/api/rl/status",
            "rl_auto_tune_status": "/api/rl/auto-tune/status"
        },
        "modularization_complete": True,
        "ibkr_ready": False,  # Will be True when IBKR integration is complete
        "message": "Modular backend operational"
    }

@app.get("/scanner")
async def scanner_page(request: Request):
    """Serve scanner page"""
    return templates.TemplateResponse("scanner/scanner.html", {"request": request})

@app.get("/strategy")
async def strategy_page(request: Request):
    """Serve strategy lab page"""
    return templates.TemplateResponse("strategy/lab.html", {"request": request})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)