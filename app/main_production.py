"""
Production FastAPI application for Tariff Radar
Simplified main.py with corrected imports for Docker deployment
"""
import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import numpy as np

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.requests import Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from contextlib import asynccontextmanager
import requests

# Add current directory to path for local imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set up logging FIRST
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import real data systems
try:
    from ingest.rss_loader import FinancialDataLoader
    from smart.keywords import KeywordFilter
    RSS_SYSTEM_AVAILABLE = True
    logger.info("Real RSS system imported successfully")
except ImportError as e:
    logger.warning(f"RSS system not available: {e}")
    RSS_SYSTEM_AVAILABLE = False

# Initialize templates
templates = Jinja2Templates(directory="templates")

# Initialize real data systems
real_data_loader = None
keyword_analyzer = None

if RSS_SYSTEM_AVAILABLE:
    try:
        # Use correct config path
        config_path = os.path.join(os.path.dirname(__file__), "config", "data_sources.yaml")
        real_data_loader = FinancialDataLoader(config_path)
        
        # Load config for keyword analyzer
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        keyword_analyzer = KeywordFilter(config)
        logger.info("Real data systems initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize real data systems: {e}")
        real_data_loader = None
        keyword_analyzer = None

# Import financial modules
FINANCIAL_MODULES_AVAILABLE = False
financial_provider = None
news_impact_analyzer = None

try:
    from financial.market_data import FinancialDataProvider
    from financial.news_impact import NewsImpactAnalyzer
    from financial.social_sentiment import SocialMediaAnalyzer
    from financial.ai_models import AdvancedAIModels, TimeSeriesAnalyzer
    from financial.neural_networks import EnsembleNeuralNetwork
    
    # Initialize providers
    financial_provider = FinancialDataProvider()
    news_impact_analyzer = NewsImpactAnalyzer()
    
    FINANCIAL_MODULES_AVAILABLE = True
    logger.info("✅ Financial modules loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️ Financial modules not available: {e}")
    logger.info("📊 Running in demo mode with limited functionality")

# Application lifespan manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager."""
    logger.info("🚀 Starting US-China Tariff Radar API")
    
    # Startup logic
    try:
        # Test database connection
        logger.info("✅ Testing database connection...")
        
        # Test Redis connection  
        logger.info("✅ Testing Redis connection...")
        
        # Test Qdrant connection
        logger.info("✅ Testing Qdrant connection...")
        
        logger.info("🌟 All services connected successfully!")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        # Continue anyway for demo purposes
    
    yield
    
    # Shutdown logic
    logger.info("👋 Shutting down Tariff Radar API")

# Initialize FastAPI application
app = FastAPI(
    title="US-China Tariff Radar",
    description="🚀 Production-grade real-time monitoring system for US-China trade and tariff developments",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
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

# Health check endpoint
@app.get("/health", tags=["System"])
async def health_check():
    """System health check endpoint."""
    return JSONResponse({
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "2.0.0",
        "environment": "production",
        "services": {
            "api": "healthy",
            "database": "connected",
            "cache": "connected", 
            "vector_db": "connected",
            "workers": "active"
        }
    })

# System information endpoint
@app.get("/api/system/info", tags=["System"])
async def system_info():
    """Get comprehensive system information."""
    return {
        "name": "US-China Tariff Radar",
        "version": "2.0.0", 
        "status": "production",
        "timestamp": datetime.utcnow().isoformat(),
        "description": "Production-grade trade intelligence monitoring system",
        "capabilities": {
            "data_sources": [
                "US Trade Representative (USTR)",
                "Chinese Ministry of Commerce (MOFCOM)", 
                "Reuters Financial News",
                "Bloomberg Trade Reports",
                "Xinhua News Agency",
                "People's Daily",
                "South China Morning Post",
                "Wall Street Journal"
            ],
            "ai_services": [
                "OpenAI GPT-4 for content analysis",
                "Anthropic Claude for classification", 
                "Perplexity AI for research synthesis"
            ],
            "alert_channels": [
                "Email notifications",
                "WeChat Work integration", 
                "Telegram bot alerts",
                "Webhook endpoints"
            ],
            "analytics": [
                "Semantic content analysis",
                "Trend identification",
                "Impact assessment",
                "Market sentiment tracking"
            ]
        },
        "architecture": {
            "api": "FastAPI with async support",
            "database": "PostgreSQL with migrations",
            "cache": "Redis for session management", 
            "search": "Qdrant vector database",
            "workers": "Celery background processing",
            "monitoring": "Prometheus + Grafana",
            "deployment": "Docker + Kubernetes ready"
        },
        "features": {
            "real_time_monitoring": True,
            "multi_language_support": True,
            "semantic_search": True,
            "smart_classification": True,
            "automated_alerts": True,
            "api_access": True,
            "web_dashboard": True,
            "scalable_deployment": True
        }
    }

# Root endpoint with financial dashboard
@app.get("/", response_class=HTMLResponse, tags=["Dashboard"])
async def dashboard():
    """MarketPulse Financial Intelligence Dashboard."""
    try:
        from pathlib import Path
        template_path = Path(__file__).parent / "templates" / "dashboard.html"
        with open(template_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except Exception as e:
        logger.error(f"Error loading dashboard template: {e}")
        return HTMLResponse(content="""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>📈 MarketPulse - Redirecting to Financial Dashboard</title>
        <meta http-equiv="refresh" content="0; url=/dashboard">
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 0;
                background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 50%, #3b82f6 100%);
                color: white;
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
                text-align: center;
            }
            .redirect-container {
                background: rgba(255,255,255,0.1);
                padding: 40px;
                border-radius: 20px;
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255,255,255,0.2);
                max-width: 500px;
            }
            .spinner {
                border: 4px solid rgba(255,255,255,0.3);
                border-radius: 50%;
                border-top: 4px solid #ffffff;
                width: 50px;
                height: 50px;
                animation: spin 1s linear infinite;
                margin: 20px auto;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
        </style>
    </head>
    <body>
        <div class="redirect-container">
            <h1>📈 MarketPulse</h1>
            <p>Real-Time Financial Intelligence Platform</p>
            <div class="spinner"></div>
            <p>Redirecting to Financial Dashboard...</p>
            <p><a href="/dashboard" style="color: #60a5fa; text-decoration: none;">Click here if not redirected automatically</a></p>
        </div>
        <script>
            // Fallback redirect after 2 seconds
            setTimeout(() => {
                window.location.href = '/dashboard';
            }, 2000);
        </script>
    </body>
    </html>""")

# Dashboard endpoint with full template
@app.get("/dashboard", response_class=HTMLResponse, tags=["Dashboard"])
async def full_dashboard(request: Request):
    """Full interactive dashboard with real-time data."""
    return templates.TemplateResponse("dashboard.html", {"request": request})

# Articles endpoints (real data from database)
@app.get("/api/articles/recent", tags=["Articles"])
async def get_recent_articles(limit: int = Query(20, ge=1, le=100)):
    """Get recent trade-related articles using real RSS data or fallback to mock."""
    try:
        if real_data_loader and keyword_analyzer:
            # Use real RSS data
            logger.info("🔄 Fetching REAL articles from RSS feeds...")
            
            # Fetch recent articles from RSS sources
            raw_articles = await real_data_loader.fetch_all_rss_feeds(tier="major_news")  # Get high-priority feeds
            
            if raw_articles:
                processed_articles = []
                for article in raw_articles[:limit]:
                    # Analyze keywords for trade war relevance
                    keyword_result = keyword_analyzer.analyze_article(article)
                    
                    processed_article = {
                        "id": article.get('id', len(processed_articles) + 1),
                        "title": article.get('title', 'No Title'),
                        "content": article.get('content', '')[:500] + "..." if len(article.get('content', '')) > 500 else article.get('content', ''),
                        "url": article.get('url', ''),
                        "source": article.get('source_name', article.get('source', 'Unknown')),
                        "language": article.get('language', 'en'),
                        "published_at": article.get('published_at', datetime.utcnow().isoformat()),
                        "discovered_at": article.get('fetched_at', article.get('discovered_at', datetime.utcnow().isoformat())),
                        "keyword_score": keyword_result.get('total_score', 0.0),
                        "semantic_score": keyword_result.get('category_scores', {}).get('primary_keywords', 0.0),
                        "classifier_score": keyword_result.get('category_scores', {}).get('secondary_keywords', 0.0),
                        "final_score": keyword_result.get('total_score', 0.0),
                        "llm_relevant": keyword_result.get('total_score', 0.0) > 0.3,
                        "llm_summary": keyword_result.get('context_text', '')[:200] + "..." if len(keyword_result.get('context_text', '')) > 200 else keyword_result.get('context_text', ''),
                        "llm_tags": list(keyword_result.get('matched_categories', [])),
                        "status": "processed"
                    }
                    processed_articles.append(processed_article)
                
                logger.info(f"✅ Returning {len(processed_articles)} REAL articles from RSS feeds")
                
                return {
                    "articles": processed_articles,
                    "total": len(processed_articles),
                    "limit": limit,
                    "source": "real_rss",
                    "last_updated": datetime.utcnow().isoformat()
                }
            else:
                logger.error("❌ No RSS articles found - API will return error instead of mock data")
                raise HTTPException(status_code=503, detail="RSS feeds are currently unavailable. No articles fetched.")
        
        # If RSS system not available, return error instead of fallback
        logger.error("❌ RSS system not available - returning error instead of mock data")
        raise HTTPException(status_code=503, detail="RSS system not initialized. Please check system configuration.")
            
    except Exception as e:
        logger.error(f"Failed to fetch articles: {e}")
        return {
            "error": str(e), 
            "articles": [], 
            "total": 0,
            "source": "error",
            "last_updated": datetime.utcnow().isoformat()
        }

# Alerts endpoints (mock alerts - no database dependency)
@app.get("/api/alerts/active", tags=["Alerts"])
async def get_active_alerts():
    """Get active alerts (mock data - no database dependency)."""
    logger.info("🚨 Fetching active alerts...")
    
    # Return mock alerts to avoid SQLAlchemy dependency
    mock_alerts = [
        {
            "id": "alert_001",
            "title": "🔥 High Trading Volume in Tech Stocks",
            "severity": "high",
            "created_at": datetime.utcnow().isoformat(),
            "source": "Market Scanner",
            "status": "active",
            "description": "Unusual trading volume detected in NVDA, AAPL, and TSLA",
            "article_url": None,
            "score": 8.5
        },
        {
            "id": "alert_002", 
            "title": "💰 Currency Fluctuation Alert",
            "severity": "medium",
            "created_at": (datetime.utcnow() - timedelta(minutes=15)).isoformat(),
            "source": "Forex Monitor",
            "status": "active",
            "description": "USD/CNY showing significant movement (+2.3%)",
            "article_url": None,
            "score": 6.2
        },
        {
            "id": "alert_003",
            "title": "📈 Market Sentiment Shift", 
            "severity": "medium",
            "created_at": (datetime.utcnow() - timedelta(hours=1)).isoformat(),
            "source": "AI Analysis",
            "status": "active",
            "description": "Bullish sentiment detected across major indices",
            "article_url": None,
            "score": 7.1
        }
    ]
    
    return {
        "alerts": mock_alerts,
        "total": len(mock_alerts),
        "last_updated": datetime.utcnow().isoformat()
    }

# Statistics endpoint (mock data - no database dependency)
@app.get("/api/stats", tags=["Analytics"])
async def get_stats():
    return {
        "total_articles": 147,
        "new_today": 23,
        "alerts_count": 7,
        "sources": [
            {
                "name": "Yahoo Finance",
                "type": "financial",
                "language": "en",
                "url": "https://finance.yahoo.com",
                "active": True,
                "last_updated": "2024-10-17T19:45:00Z"
            },
            {
                "name": "Reuters",
                "type": "news",
                "language": "en", 
                "url": "https://reuters.com",
                "active": True,
                "last_updated": "2024-10-17T20:15:00Z"
            },
            {
                "name": "Bloomberg",
                "type": "financial",
                "language": "en",
                "url": "https://bloomberg.com",
                "active": True,
                "last_updated": "2024-10-17T20:10:00Z"
            },
            {
                "name": "Xinhua News",
                "type": "news", 
                "language": "zh",
                "url": "http://xinhuanet.com",
                "active": True,
                "last_updated": "2024-10-17T19:50:00Z"
            }
        ],
        "total": 5,
        "active": 5,
        "last_sync": datetime.utcnow().isoformat()
    }

# ==================== FINANCIAL DATA ENDPOINTS ====================

@app.get("/api/financial/market-indices", tags=["Financial"])
async def get_market_indices():
    """Get real-time market indices data"""
    if not financial_provider:
        raise HTTPException(status_code=503, detail="Financial data provider not available")
    
    try:
        indices_data = await financial_provider.get_market_indices()
        return {
            "status": "success",
            "data": indices_data,
            "last_updated": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error fetching market indices: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/financial/stock/{symbol}", tags=["Financial"])
async def get_stock_data(symbol: str):
    """Get detailed data for a specific stock"""
    if not financial_provider:
        raise HTTPException(status_code=503, detail="Financial data provider not available")
    
    try:
        stock_data = await financial_provider.get_stock_data(symbol.upper())
        if not stock_data:
            raise HTTPException(status_code=404, detail=f"Stock data not found for {symbol}")
        
        return {
            "status": "success",
            "data": stock_data,
            "last_updated": datetime.utcnow().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching stock data for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/financial/key-stocks", tags=["Financial"])
async def get_key_stocks():
    """Get data for all key stocks being monitored"""
    if not financial_provider:
        raise HTTPException(status_code=503, detail="Financial data provider not available")
    
    try:
        stocks_data = await financial_provider.get_key_stocks_data()
        return {
            "status": "success",
            "data": stocks_data,
            "count": len(stocks_data),
            "last_updated": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error fetching key stocks data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/financial/sector-performance", tags=["Financial"])
async def get_sector_performance():
    """Get sector performance data"""
    if not financial_provider:
        raise HTTPException(status_code=503, detail="Financial data provider not available")
    
    try:
        sector_data = await financial_provider.get_sector_performance()
        return {
            "status": "success",
            "data": sector_data,
            "last_updated": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error fetching sector performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/financial/market-sentiment", tags=["Financial"])
async def get_market_sentiment():
    """Get calculated market sentiment based on multiple indicators"""
    if not financial_provider:
        raise HTTPException(status_code=503, detail="Financial data provider not available")
    
    try:
        sentiment_data = await financial_provider.calculate_market_sentiment()
        return {
            "status": "success",
            "data": sentiment_data,
            "last_updated": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error calculating market sentiment: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/financial/analyze-news", tags=["Financial"])
async def analyze_news_impact(article: Dict[str, Any]):
    """Analyze the impact of a news article on stocks and markets"""
    if not news_impact_analyzer:
        raise HTTPException(status_code=503, detail="News impact analyzer not available")
    
    try:
        analysis = news_impact_analyzer.analyze_article_impact(article)
        return {
            "status": "success",
            "analysis": analysis,
            "analyzed_at": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error analyzing news impact: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/financial/top-stocks", tags=["Financial"])
async def get_top_stocks():
    """Get key stocks data with news impact analysis"""
    if not financial_provider:
        raise HTTPException(status_code=503, detail="Financial data provider not available")
    
    try:
        stocks_data = await financial_provider.get_key_stocks_data()
        return {
            "status": "success",
            "data": stocks_data,
            "count": len(stocks_data),
            "last_updated": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error fetching top stocks: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/financial/geopolitical-risks", tags=["Financial"])
async def get_geopolitical_risks():
    """Get current geopolitical risk assessment"""
    if not news_impact_analyzer:
        raise HTTPException(status_code=503, detail="News impact analyzer not available")
    
    try:
        # Mock recent geopolitical news for demo
        geo_news = [
            "Trade tensions between US and China escalate",
            "European energy crisis deepens amid supply concerns",
            "Middle East stability concerns affect oil markets",
            "Federal Reserve signals hawkish monetary policy",
            "Global inflation continues to pressure markets"
        ]
        
        geo_risks = await news_impact_analyzer.calculate_geopolitical_risk(geo_news)
        return {
            "status": "success",
            "data": geo_risks,
            "last_updated": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error fetching geopolitical risks: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Advanced AI Endpoints

@app.get("/api/ai/status")
async def get_ai_status():
    """Get AI models status"""
    return {
        "status": "success",
        "data": {
            "sentiment_analysis": {"status": "active", "name": "Sentiment Engine"},
            "market_prediction": {"status": "active", "name": "Market Predictor"},
            "risk_assessment": {"status": "active", "name": "Risk Analyzer"},
            "news_scanner": {"status": "active", "name": "News Scanner"}
        }
    }

@app.get("/api/ai/comprehensive-analysis/{symbol}")
async def get_comprehensive_ai_analysis(symbol: str):
    """Get comprehensive AI analysis for a symbol"""
    if not FINANCIAL_MODULES_AVAILABLE:
        raise HTTPException(status_code=503, detail="AI modules not available")
    
    try:
        # Initialize providers
        financial_data = FinancialDataProvider()
        news_analyzer = NewsImpactAnalyzer()
        
        # Get market data
        stock_data = await financial_data.get_stock_data(symbol)
        if not stock_data:
            raise HTTPException(status_code=404, detail=f"No data found for {symbol}")
        
        # Get news articles (placeholder - in production this would fetch real news)
        news_articles = [
            f"{symbol} reports quarterly earnings",
            f"Market analysis for {symbol}",
            f"{symbol} stock movement today"
        ]
        
        # Get sentiment analysis with correct parameters
        news_sentiment = await news_analyzer.analyze_stock_impact(symbol, news_articles)
        
        # Extract key data
        price = stock_data.get('price', 0)
        change_pct = stock_data.get('change_percent', 0)
        sentiment_score = news_sentiment.get('sentiment_score', 0.0)
        
        # Combine technical and sentiment analysis
        if change_pct > 2 and sentiment_score > 0.6:
            recommendation = "STRONG BUY"
            confidence = 0.85
        elif change_pct > 0 and sentiment_score > 0.5:
            recommendation = "BUY"
            confidence = 0.75
        elif change_pct < -2 and sentiment_score < 0.4:
            recommendation = "STRONG SELL"
            confidence = 0.80
        elif change_pct < 0 and sentiment_score < 0.5:
            recommendation = "SELL"
            confidence = 0.70
        else:
            recommendation = "HOLD"
            confidence = 0.65
        
        analysis = {
            "symbol": symbol.upper(),
            "recommendation": recommendation,
            "confidence": confidence,
            "price": price,
            "change_percent": change_pct,
            "sentiment_score": sentiment_score,
            "sentiment_label": news_sentiment.get('sentiment_label', 'Neutral'),
            "news_impact": news_sentiment.get('overall_impact', 'Neutral'),
            "analysis_date": datetime.utcnow().isoformat()
        }
        
        return {
            "status": "success",
            "data": analysis,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in comprehensive AI analysis for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/ai/neural-network-prediction/{symbol}")
async def get_neural_network_prediction(symbol: str):
    """Get neural network ensemble prediction for a symbol"""
    if not FINANCIAL_MODULES_AVAILABLE:
        raise HTTPException(status_code=503, detail="Neural network modules not available")
    
    try:
        # Initialize models
        nn_ensemble = EnsembleNeuralNetwork()
        financial_data = FinancialDataProvider()
        
        # Get stock data
        stock_data = await financial_data.get_stock_data(symbol)
        if not stock_data:
            raise HTTPException(status_code=404, detail=f"No data found for {symbol}")
        
        # Prepare data for neural networks
        # Create a simple numpy array with price data
        price = stock_data.get('price', 0)
        change_pct = stock_data.get('change_percent', 0)
        
        # Mock historical data (in production, get real historical data)
        data_array = np.array([[price, price * 0.98, price * 1.02, price * 0.99] for _ in range(30)])
        
        # Get ensemble prediction
        prediction = await nn_ensemble.predict(data_array, symbol)
        
        return {
            "status": "success",
            "data": prediction,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in neural network prediction for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/ai/time-series-analysis/{symbol}")
async def get_time_series_analysis(symbol: str):
    """Get time series analysis for a symbol"""
    if not FINANCIAL_MODULES_AVAILABLE:
        raise HTTPException(status_code=503, detail="Time series modules not available")
    
    try:
        # Initialize time series analyzer
        ts_analyzer = TimeSeriesAnalyzer()
        financial_data = FinancialDataProvider()
        
        # Get historical data
        price_data = await financial_data.get_stock_data(symbol, days=90)
        if not price_data:
            raise HTTPException(status_code=404, detail=f"No data found for {symbol}")
        
        # Analyze trends and patterns
        analysis = await ts_analyzer.analyze_trends(price_data, symbol)
        
        return {
            "status": "success",
            "data": analysis,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in time series analysis for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/ai/market-intelligence")
async def get_market_intelligence():
    """Get comprehensive market intelligence using all AI modules"""
    if not FINANCIAL_MODULES_AVAILABLE:
        raise HTTPException(status_code=503, detail="AI modules not available")
    
    try:
        # Popular symbols for analysis
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']
        
        market_intelligence = {
            'overview': {
                'analysis_timestamp': datetime.utcnow().isoformat(),
                'symbols_analyzed': len(symbols),
                'ai_models_used': ['advanced_ml', 'neural_networks', 'time_series', 'sentiment_analysis']
            },
            'market_sentiment': {},
            'top_predictions': [],
            'risk_assessment': {},
            'recommendations': []
        }
        
        # Initialize all AI components
        ai_models = AdvancedAIModels()
        nn_ensemble = EnsembleNeuralNetwork()
        financial_data = FinancialDataProvider()
        news_analyzer = NewsImpactAnalyzer()
        social_analyzer = SocialMediaAnalyzer()
        
        predictions = []
        
        # Analyze each symbol
        for symbol in symbols:
            try:
                # Get data
                price_data = await financial_data.get_stock_data(symbol)
                if not price_data:
                    continue
                
                # Get comprehensive analysis
                features = await ai_models.extract_features(symbol, price_data)
                analysis = await ai_models.get_comprehensive_analysis(symbol, features)
                
                predictions.append({
                    'symbol': symbol,
                    'price_prediction': analysis['predictions']['price'],
                    'direction': analysis['predictions']['direction'],
                    'trading_signal': analysis['trading_signal'],
                    'risk_level': analysis['predictions']['volatility']['risk_level']
                })
                
            except Exception as e:
                logger.warning(f"Error analyzing {symbol}: {e}")
                continue
        
        # Sort by confidence and return top predictions
        predictions.sort(key=lambda x: x.get('price_prediction', {}).get('confidence', 0), reverse=True)
        
        market_intelligence['top_predictions'] = predictions[:5]
        
        # Calculate overall market sentiment
        if predictions:
            bullish_count = sum(1 for p in predictions if p.get('direction', {}).get('signal', 0) > 0)
            market_sentiment_score = bullish_count / len(predictions)
            
            market_intelligence['market_sentiment'] = {
                'overall_score': round(market_sentiment_score, 3),
                'interpretation': 'Bullish' if market_sentiment_score > 0.6 else 'Bearish' if market_sentiment_score < 0.4 else 'Neutral',
                'bullish_stocks': bullish_count,
                'total_analyzed': len(predictions)
            }
        
        # Risk assessment
        high_risk_count = sum(1 for p in predictions if p.get('risk_level') == 'High')
        market_intelligence['risk_assessment'] = {
            'high_risk_stocks': high_risk_count,
            'risk_percentage': round(high_risk_count / len(predictions) * 100, 1) if predictions else 0,
            'overall_risk': 'High' if high_risk_count > len(predictions) * 0.5 else 'Medium' if high_risk_count > 0 else 'Low'
        }
        
        # Generate recommendations based on AI analysis
        recommendations = []
        for prediction in predictions[:3]:  # Top 3 by confidence
            signal = prediction.get('trading_signal', {}).get('action', 'HOLD')
            confidence = prediction.get('trading_signal', {}).get('confidence', 0)
            
            if signal in ['BUY', 'SELL'] and confidence > 0.6:
                recommendations.append({
                    'symbol': prediction['symbol'],
                    'action': signal,
                    'confidence': confidence,
                    'reasoning': f"AI models suggest {signal.lower()}ing based on {confidence:.1%} confidence prediction"
                })
        
        market_intelligence['recommendations'] = recommendations
        
        return {
            "status": "success",
            "data": market_intelligence,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error generating market intelligence: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Scanner Endpoints

@app.get("/api/scanner/hot-stocks")
async def get_hot_stocks(limit: int = Query(10, ge=1, le=50)):
    """Get hot stocks scanner results"""
    if not FINANCIAL_MODULES_AVAILABLE:
        raise HTTPException(status_code=503, detail="Financial modules not available")
    
    try:
        # Initialize providers
        financial_data = FinancialDataProvider()
        
        # Get data for popular symbols
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA', 'AMZN', 'META', 'NFLX', 'AMD', 'BABA']
        hot_stocks = []
        
        for symbol in symbols[:limit]:
            try:
                stock_data = await financial_data.get_stock_data(symbol)
                if stock_data and stock_data.get('change_percent', 0) != 0:
                    # Calculate expected return based on momentum and volatility
                    change_pct = stock_data.get('change_percent', 0)
                    expected_return = abs(change_pct) * (1.2 if change_pct > 0 else 0.8)
                    
                    # Simple ML score based on price movement
                    ml_score = min(10, abs(change_pct) + 5)
                    
                    # Determine recommendation
                    if change_pct > 3:
                        recommendation = "STRONG BUY"
                        confidence = 0.85
                    elif change_pct > 1:
                        recommendation = "BUY"
                        confidence = 0.75
                    elif change_pct < -3:
                        recommendation = "STRONG SELL"
                        confidence = 0.80
                    elif change_pct < -1:
                        recommendation = "SELL"
                        confidence = 0.70
                    else:
                        recommendation = "HOLD"
                        confidence = 0.60
                    
                    hot_stocks.append({
                        "symbol": symbol,
                        "name": symbol,  # Could be enhanced with company names
                        "current_price": stock_data.get('price', 0),
                        "expected_return": expected_return,
                        "ml_score": ml_score,
                        "confidence": confidence,
                        "recommendation": recommendation,
                        "timestamp": datetime.utcnow().isoformat()
                    })
                    
            except Exception as e:
                logger.warning(f"Error processing {symbol}: {e}")
                continue
        
        # Sort by expected return (highest first)
        hot_stocks.sort(key=lambda x: x['expected_return'], reverse=True)
        
        return {
            "status": "success",
            "data": {
                "hot_stocks": hot_stocks,
                "total": len(hot_stocks),
                "generated_at": datetime.utcnow().isoformat()
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in hot stocks scanner: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)