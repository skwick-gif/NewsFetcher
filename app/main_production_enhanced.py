"""
MarketPulse - Real-time Financial Intelligence Platform
Production FastAPI server with AI analysis, ML models, and live streaming
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json
import asyncio
import sys
import os
from contextlib import asynccontextmanager

# Add current directory to path for local imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Financial modules
try:
    from financial.market_data_clean import FinancialDataProvider  # Clean version - NO DEMO DATA
    from financial.news_impact import NewsImpactAnalyzer
    from financial.social_sentiment_enhanced import RealSocialMediaAnalyzer
    from financial.ai_models import AdvancedAIModels
    from financial.neural_networks import EnsembleNeuralNetwork
    from financial.ml_trainer import MLModelTrainer
    from financial.websocket_manager import WebSocketManager, MarketDataStreamer
    FINANCIAL_MODULES_AVAILABLE = True
    logger.info("✅ Financial modules loaded - REAL DATA ONLY MODE")
except ImportError as e:
    FINANCIAL_MODULES_AVAILABLE = False
    print(f"⚠️ Financial modules not available: {e}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global components
ml_trainer = None
websocket_manager = None
market_streamer = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    global ml_trainer, websocket_manager, market_streamer
    
    logger.info("🚀 Starting MarketPulse Production Server...")
    
    if FINANCIAL_MODULES_AVAILABLE:
        try:
            # Initialize ML trainer
            ml_trainer = MLModelTrainer()
            logger.info("✅ ML Trainer initialized")
            
            # Initialize WebSocket manager
            websocket_manager = WebSocketManager()
            logger.info("✅ WebSocket Manager initialized")
            
            # Initialize market data streamer
            market_streamer = MarketDataStreamer(websocket_manager)
            logger.info("✅ Market Data Streamer initialized")
            
            # Start background tasks
            asyncio.create_task(market_streamer.start_streaming())
            
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
    
    yield
    
    # Cleanup
    logger.info("🛑 Shutting down MarketPulse...")

# Create FastAPI app with lifespan
app = FastAPI(
    title="MarketPulse - Financial Intelligence Platform",
    description="Real-time financial data analysis with AI-powered insights",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
if FINANCIAL_MODULES_AVAILABLE:
    financial_provider = FinancialDataProvider()
    news_analyzer = NewsImpactAnalyzer()
    social_analyzer = RealSocialMediaAnalyzer()
    ai_models = AdvancedAIModels()
    neural_network = EnsembleNeuralNetwork()
else:
    financial_provider = None
    news_analyzer = None
    social_analyzer = None
    ai_models = None
    neural_network = None

# Setup templates and static files
try:
    templates = Jinja2Templates(directory="templates")
    app.mount("/static", StaticFiles(directory="static"), name="static")
except Exception as e:
    logger.warning(f"Templates/static files not available: {e}")
    templates = None

@app.get("/")
async def root():
    """Root endpoint - redirect to dashboard"""
    return RedirectResponse(url="/dashboard")

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main financial dashboard"""
    if templates:
        return templates.TemplateResponse("dashboard.html", {"request": request})
    else:
        return HTMLResponse(content="""
        <html>
            <head><title>MarketPulse Dashboard</title></head>
            <body>
                <h1>MarketPulse Financial Intelligence Platform</h1>
                <p>API endpoints available at: <a href="/docs">/docs</a></p>
                <p>Real-time financial data and AI analysis</p>
            </body>
        </html>
        """)

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0",
        "components": {
            "financial_modules": FINANCIAL_MODULES_AVAILABLE,
            "ml_trainer": ml_trainer is not None,
            "websocket_manager": websocket_manager is not None,
            "market_streamer": market_streamer is not None,
            "apis": {
                "social_media": social_analyzer.twitter_client is not None or social_analyzer.reddit_client is not None if social_analyzer else False,
                "financial_data": financial_provider is not None
            }
        }
    }

@app.get("/api/market/{symbol}")
async def get_market_data(symbol: str):
    """Get comprehensive market data for a symbol"""
    if not FINANCIAL_MODULES_AVAILABLE:
        raise HTTPException(status_code=503, detail="Financial modules not available")
    
    try:
        # Get basic market data
        market_data = await financial_provider.get_stock_data(symbol)
        
        # Get AI analysis
        ai_analysis = await ai_models.analyze_stock(symbol, market_data)
        
        # Get news impact
        news_impact = await news_analyzer.analyze_stock_impact(symbol)
        
        return {
            "symbol": symbol,
            "market_data": market_data,
            "ai_analysis": ai_analysis,
            "news_impact": news_impact,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error fetching market data for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch market data: {str(e)}")

@app.get("/api/sentiment/{symbol}")
async def get_social_sentiment(symbol: str):
    """Get social media sentiment analysis for a symbol"""
    if not FINANCIAL_MODULES_AVAILABLE:
        raise HTTPException(status_code=503, detail="Financial modules not available")
    
    try:
        sentiment_analysis = await social_analyzer.analyze_sentiment(symbol)
        return sentiment_analysis
    except Exception as e:
        logger.error(f"Error analyzing sentiment for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to analyze sentiment: {str(e)}")

@app.get("/api/ai/analysis/{symbol}")
async def get_ai_analysis(symbol: str):
    """Get comprehensive AI analysis for a symbol"""
    if not FINANCIAL_MODULES_AVAILABLE:
        raise HTTPException(status_code=503, detail="Financial modules not available")
    
    try:
        # Get market data first
        market_data = await financial_provider.get_stock_data(symbol)
        
        # Run AI analysis
        ai_analysis = await ai_models.analyze_stock(symbol, market_data)
        
        # Get neural network prediction
        neural_prediction = await neural_network.predict_price_movement(symbol, market_data)
        
        # Get sentiment analysis
        sentiment_analysis = await social_analyzer.analyze_sentiment(symbol)
        
        return {
            "symbol": symbol,
            "ai_analysis": ai_analysis,
            "neural_prediction": neural_prediction,
            "sentiment_analysis": sentiment_analysis,
            "market_data": market_data,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error in AI analysis for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"AI analysis failed: {str(e)}")

@app.get("/api/ml/predictions/{symbol}")
async def get_ml_predictions(symbol: str):
    """Get ML model predictions for a symbol"""
    if not FINANCIAL_MODULES_AVAILABLE:
        raise HTTPException(status_code=503, detail="Financial modules not available")
    
    try:
        if not ml_trainer:
            raise HTTPException(status_code=503, detail="ML trainer not initialized")
        
        # Get predictions from trained models
        predictions = await ml_trainer.get_predictions(symbol)
        return predictions
    except Exception as e:
        logger.error(f"Error getting ML predictions for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"ML prediction failed: {str(e)}")

@app.post("/api/ml/train/{symbol}")
async def train_ml_model(symbol: str, days_back: int = 365):
    """Train ML model for a specific symbol"""
    if not FINANCIAL_MODULES_AVAILABLE:
        raise HTTPException(status_code=503, detail="Financial modules not available")
    
    try:
        if not ml_trainer:
            raise HTTPException(status_code=503, detail="ML trainer not initialized")
        
        # Start model training
        training_result = await ml_trainer.train_models_for_symbol(symbol, days_back)
        return training_result
    except Exception as e:
        logger.error(f"Error training ML model for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"ML training failed: {str(e)}")

@app.get("/api/watchlist")
async def get_watchlist():
    """Get default stock watchlist with real-time data"""
    if not FINANCIAL_MODULES_AVAILABLE:
        return {
            "watchlist": [],
            "error": "Financial modules not available",
            "timestamp": datetime.now().isoformat()
        }
    
    symbols = ['AAPL', 'TSLA', 'NVDA', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NFLX']
    
    watchlist_data = []
    for symbol in symbols:
        try:
            market_data = await financial_provider.get_stock_data(symbol)
            ai_analysis = await ai_models.analyze_stock(symbol, market_data)
            
            watchlist_data.append({
                "symbol": symbol,
                "price": market_data.get('current_price', 0),
                "change": market_data.get('price_change', 0),
                "change_percent": market_data.get('price_change_percent', 0),
                "ai_score": ai_analysis.get('confidence_score', 0),
                "ai_recommendation": ai_analysis.get('recommendation', 'Hold')
            })
        except Exception as e:
            logger.warning(f"Error fetching data for {symbol}: {e}")
            continue
    
    return {
        "watchlist": watchlist_data,
        "timestamp": datetime.now().isoformat()
    }

@app.websocket("/ws/market/{symbol}")
async def websocket_market_data(websocket: WebSocket, symbol: str):
    """WebSocket endpoint for real-time market data"""
    await websocket.accept()
    
    if not FINANCIAL_MODULES_AVAILABLE or not websocket_manager:
        await websocket.send_json({"error": "WebSocket services not available"})
        await websocket.close()
        return
    
    try:
        # Subscribe to symbol updates
        await websocket_manager.subscribe_to_symbol(websocket, symbol)
        logger.info(f"WebSocket connected for {symbol}")
        
        while True:
            # Keep connection alive and handle messages
            try:
                message = await websocket.receive_text()
                data = json.loads(message)
                
                if data.get('type') == 'ping':
                    await websocket.send_json({'type': 'pong'})
                elif data.get('type') == 'subscribe':
                    new_symbol = data.get('symbol')
                    if new_symbol:
                        await websocket_manager.subscribe_to_symbol(websocket, new_symbol)
                        
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"WebSocket error for {symbol}: {e}")
                break
                
    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")
    finally:
        # Cleanup
        if websocket_manager:
            await websocket_manager.unsubscribe_from_symbol(websocket, symbol)
        logger.info(f"WebSocket disconnected for {symbol}")

@app.websocket("/ws/ai-analysis/{symbol}")
async def websocket_ai_analysis(websocket: WebSocket, symbol: str):
    """WebSocket endpoint for real-time AI analysis"""
    await websocket.accept()
    
    if not FINANCIAL_MODULES_AVAILABLE:
        await websocket.send_json({"error": "AI analysis services not available"})
        await websocket.close()
        return
    
    try:
        while True:
            # Get latest AI analysis
            market_data = await financial_provider.get_stock_data(symbol)
            ai_analysis = await ai_models.analyze_stock(symbol, market_data)
            sentiment_analysis = await social_analyzer.analyze_sentiment(symbol)
            
            # Send comprehensive analysis
            analysis_data = {
                "type": "ai_analysis",
                "symbol": symbol,
                "ai_analysis": ai_analysis,
                "sentiment": sentiment_analysis,
                "timestamp": datetime.now().isoformat()
            }
            
            await websocket.send_json(analysis_data)
            
            # Wait before next update
            await asyncio.sleep(30)  # Update every 30 seconds
            
    except WebSocketDisconnect:
        logger.info(f"AI analysis WebSocket disconnected for {symbol}")
    except Exception as e:
        logger.error(f"AI analysis WebSocket error: {e}")

@app.get("/api/performance/model-stats")
async def get_model_performance():
    """Get ML model performance statistics"""
    if not FINANCIAL_MODULES_AVAILABLE:
        return {"error": "Financial modules not available"}
    
    try:
        if not ml_trainer:
            return {"error": "ML trainer not available"}
        
        performance_stats = ml_trainer.get_model_performance_stats()
        return performance_stats
    except Exception as e:
        logger.error(f"Error getting model performance: {e}")
        return {"error": str(e)}

@app.get("/api/system/status")
async def get_system_status():
    """Get comprehensive system status"""
    return {
        "server_status": "running",
        "financial_modules": FINANCIAL_MODULES_AVAILABLE,
        "components": {
            "ml_trainer": {
                "status": "active" if ml_trainer else "inactive",
                "models_trained": ml_trainer.get_trained_models_count() if ml_trainer and hasattr(ml_trainer, 'get_trained_models_count') else 0
            },
            "websocket_manager": {
                "status": "active" if websocket_manager else "inactive",
                "active_connections": len(websocket_manager.connections) if websocket_manager and hasattr(websocket_manager, 'connections') else 0
            },
            "market_streamer": {
                "status": "active" if market_streamer else "inactive",
                "streaming": market_streamer.is_streaming if market_streamer and hasattr(market_streamer, 'is_streaming') else False
            }
        },
        "timestamp": datetime.now().isoformat()
    }

# API information endpoint
@app.get("/api/info")
async def get_api_info():
    """Get API information and available endpoints"""
    return {
        "name": "MarketPulse Financial Intelligence API",
        "version": "2.0.0",
        "description": "Real-time financial data analysis with AI-powered insights",
        "endpoints": {
            "market_data": "/api/market/{symbol}",
            "sentiment_analysis": "/api/sentiment/{symbol}",
            "ai_analysis": "/api/ai/analysis/{symbol}",
            "ml_predictions": "/api/ml/predictions/{symbol}",
            "watchlist": "/api/watchlist",
            "health_check": "/api/health",
            "system_status": "/api/system/status"
        },
        "websockets": {
            "market_data": "/ws/market/{symbol}",
            "ai_analysis": "/ws/ai-analysis/{symbol}"
        },
        "features": [
            "Real-time market data",
            "AI-powered stock analysis",
            "Social media sentiment analysis",
            "Machine learning predictions",
            "WebSocket streaming",
            "Neural network price forecasting"
        ]
    }

if __name__ == "__main__":
    logger.info("🚀 Starting MarketPulse Production Server...")
    uvicorn.run(
        "main_production_enhanced:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Production mode
        log_level="info"
    )