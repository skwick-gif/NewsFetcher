"""
Financial Market Data Endpoints
Real-time market data and sentiment analysis
"""
from fastapi import APIRouter, HTTPException
from datetime import datetime, timezone
from app.core.config import (
    logger, financial_provider, social_analyzer
)

router = APIRouter()

@router.get("/market/{symbol}")
async def get_market_data(symbol: str):
    """Get real-time market data for symbol"""
    try:
        if financial_provider:
            market_data = await financial_provider.get_stock_data(symbol)
            return {
                "status": "success",
                "symbol": symbol,
                "data": market_data,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        # Live only: no market data without provider
        raise HTTPException(status_code=503, detail="Financial provider unavailable")

    except Exception as e:
        logger.error(f"Failed to get market data for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get market data for {symbol}")

@router.get("/sentiment/{symbol}")
async def get_social_sentiment(symbol: str):
    """Get social media sentiment for symbol"""
    try:
        if social_analyzer:
            sentiment = await social_analyzer.get_sentiment(symbol)
            return {
                "status": "success",
                "symbol": symbol,
                "sentiment": sentiment,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        # Live only: social sentiment unavailable
        raise HTTPException(status_code=503, detail="Social sentiment analyzer unavailable")

    except Exception as e:
        logger.error(f"Failed to get sentiment for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get sentiment for {symbol}")

@router.get("/financial/market-indices")
async def get_market_indices():
    """Get major market indices data"""
    try:
        from app.financial.market_data import financial_provider
        indices_data = await financial_provider.get_market_indices()
        return {
            "status": "success",
            "data": indices_data,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get market indices: {e}")
        raise HTTPException(status_code=500, detail="Failed to get market indices")