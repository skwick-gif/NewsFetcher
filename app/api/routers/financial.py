"""
Financial Data Endpoints
Real-time market data, sentiment, and financial analysis
"""
from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any
from datetime import datetime
from app.core.config import (
    logger, financial_provider, FINANCIAL_MODULES_AVAILABLE,
    news_sentiment_provider, NEWS_SENTIMENT_AVAILABLE
)

router = APIRouter()

# ============================================================
# Financial Data Endpoints (REAL DATA ONLY)
# ============================================================
@router.get("/market-indices")
async def get_market_indices_endpoint():
    """
    Get REAL market indices data (S&P500, NASDAQ, DOW, VIX)
    NO DEMO DATA - Only real Yahoo Finance data
    """
    try:
        indices = await financial_provider.get_market_indices()

        if not indices:
            raise HTTPException(status_code=503, detail="Unable to fetch real market data")

        return {
            "status": "success",
            "data": indices,
            "timestamp": datetime.now().isoformat(),
            "source": "Yahoo Finance (Real-time)"
        }
    except Exception as e:
        logger.error(f"❌ Error fetching market indices: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/market-sentiment")
async def get_market_sentiment_endpoint():
    """
    Calculate market sentiment from REAL indices data
    NO DEMO DATA - Based on actual market performance
    """
    try:
        sentiment = await financial_provider.calculate_market_sentiment()

        return {
            "status": "success",
            "data": sentiment,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"❌ Error calculating market sentiment: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/top-stocks")
async def get_top_stocks_endpoint():
    """
    Get top performing stocks with REAL data
    NO DEMO DATA - Real Yahoo Finance data
    """
    try:
        stocks = await financial_provider.get_key_stocks_data()

        # Sort by change_percent to show top movers
        gainers = sorted([s for s in stocks if s.get('is_positive', False)],
                        key=lambda x: x.get('change_percent', 0), reverse=True)[:5]
        losers = sorted([s for s in stocks if not s.get('is_positive', True)],
                       key=lambda x: x.get('change_percent', 0))[:5]

        return {
            "status": "success",
            "data": {
                "gainers": gainers,
                "losers": losers,
                "all_stocks": stocks
            },
            "timestamp": datetime.now().isoformat(),
            "source": "Yahoo Finance"
        }
    except Exception as e:
        logger.error(f"❌ Error fetching top stocks: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/historical/{symbol}")
async def get_historical_data(symbol: str, timeframe: str = "1D"):
    """
    Get REAL historical price data from LOCAL CSV files
    Timeframes: 1D (1 day), 1W (5 days), 1M (1 month), 3M (3 months)
    """
    try:
        import pandas as pd
        from datetime import datetime, timedelta
        from pathlib import Path

        # Get stock_data directory (two levels up from app/)
        project_root = Path(__file__).parent.parent.parent
        stock_file = project_root / "stock_data" / symbol / f"{symbol}_price.csv"

        if not stock_file.exists():
            raise HTTPException(status_code=404, detail=f"No local data found for {symbol}")

        # Read CSV file (robust to index column and case-insensitive 'Date')
        try:
            df = pd.read_csv(stock_file)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to read CSV for {symbol}: {e}")

        # Normalize date column
        date_col = None
        for cand in ['Date', 'date']:
            if cand in df.columns:
                date_col = cand
                break
        # If Date not found as a named column, try first column as Date
        if date_col is None and df.shape[1] >= 1:
            date_col = df.columns[0]

        if date_col is None:
            raise HTTPException(status_code=500, detail=f"Invalid CSV: missing Date column for {symbol}")

        # Convert to datetime robustly and make timezone-naive
        try:
            # Prefer strict ISO-8601 when possible
            df[date_col] = pd.to_datetime(df[date_col], format='ISO8601', utc=True)
        except Exception:
            try:
                # Pandas >=2.0 supports mixed formats
                df[date_col] = pd.to_datetime(df[date_col], format='mixed', utc=True)
            except Exception:
                # Fallback: let pandas infer; coerce invalids to NaT
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce', utc=True)
        # Drop invalid and strip timezone
        df[date_col] = df[date_col].dt.tz_localize(None)
        df = df.dropna(subset=[date_col]).sort_values(date_col)

        # Map timeframe to number of days
        timeframe_days = {
            "1D": 1,    # Last 1 day
            "1W": 5,    # Last 5 trading days (1 week)
            "1M": 30,   # Last 30 days (1 month)
            "3M": 90    # Last 90 days (3 months)
        }

        days = timeframe_days.get(timeframe, 1)

        # Filter to last N days
        cutoff_date = datetime.now() - timedelta(days=days)
        df_filtered = df[df[date_col] >= cutoff_date].copy()

        if len(df_filtered) == 0:
            # If no data in timeframe, get last N rows
            df_filtered = df.tail(days)

        # Format data for chart (both line and candlestick)
        labels = []
        prices = []
        ohlc = []  # list of dicts: {t, o, h, l, c}

        for _, row in df_filtered.iterrows():
            date = pd.to_datetime(row[date_col])
            if timeframe == "1D":
                # For 1 day, show time if available, else show date
                labels.append(date.strftime("%H:%M" if date.hour != 0 else "%m/%d"))
            else:
                # For longer periods, show date
                labels.append(date.strftime("%m/%d"))

            # Handle different casings for Close
            close_val = None
            if 'Close' in df.columns:
                close_val = row['Close']
            elif 'close' in df.columns:
                close_val = row['close']
            else:
                raise HTTPException(status_code=500, detail=f"Invalid CSV: missing Close column for {symbol}")
            prices.append(float(close_val))

            # Build OHLC if columns exist
            def _get(colA, colB=None):
                if colA in df.columns:
                    return row[colA]
                if colB and colB in df.columns:
                    return row[colB]
                return None
            o = _get('Open', 'open')
            h = _get('High', 'high')
            l = _get('Low', 'low')
            c = close_val
            if o is not None and h is not None and l is not None and c is not None:
                # Use ISO-8601 with Z to avoid client-side parser format errors
                ohlc.append({
                    "t": date.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "o": float(o),
                    "h": float(h),
                    "l": float(l),
                    "c": float(c)
                })

        # Calculate statistics
        current_price = prices[-1] if prices else 0
        start_price = prices[0] if prices else 0
        change = current_price - start_price
        change_percent = (change / start_price * 100) if start_price != 0 else 0

        logger.info(f"✅ Loaded {symbol} from local CSV: {len(prices)} data points")

        return {
            "status": "success",
            "data": {
                "symbol": symbol,
                "timeframe": timeframe,
                "labels": labels,
                "prices": prices,
                "ohlc": ohlc,
                "current_price": round(current_price, 2),
                "change": round(change, 2),
                "change_percent": round(change_percent, 2),
                "is_positive": change >= 0,
                "data_points": len(prices)
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error loading local data for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading data: {str(e)}")

@router.get("/ai/market-intelligence")
async def get_market_intelligence():
    """
    Get market intelligence analysis
    Combines market data, sentiment, and risk assessment
    """
    try:
        # Get market indices and calculate sentiment
        indices = await financial_provider.get_market_indices()
        sentiment_data = await financial_provider.calculate_market_sentiment()

        # Harmonized sentiment fields
        sentiment_score = float(sentiment_data.get('score', 50.0))
        total_change = float(sentiment_data.get('total_change', 0.0))

        # Canonical interpretation used by frontend coloring logic
        if sentiment_score >= 70:
            sentiment_interpretation = "Bullish"
        elif sentiment_score >= 55:
            sentiment_interpretation = "Slightly Bullish"
        elif sentiment_score >= 45:
            sentiment_interpretation = "Neutral"
        elif sentiment_score >= 30:
            sentiment_interpretation = "Slightly Bearish"
        else:
            sentiment_interpretation = "Bearish"

        # Calculate risk assessment based on VIX and market volatility
        vix_data = indices.get('vix', {})
        vix_value = vix_data.get('price', 20.0)

        if vix_value > 30:
            risk_level = "High"
            risk_percentage = min(int((vix_value - 20) * 3), 100)
        elif vix_value > 20:
            risk_level = "Moderate"
            risk_percentage = int((vix_value - 10) * 2)
        else:
            risk_level = "Low"
            risk_percentage = int(vix_value)

        # Get top movers for recommendations and compute overview stats
        stocks = await financial_provider.get_key_stocks_data()
        total_analyzed = len(stocks)
        bullish_stocks = sum(1 for s in stocks if s.get('is_positive'))
        # Heuristic: consider high risk if daily drop >= 2.5%
        high_risk_stocks = sum(1 for s in stocks if s.get('change_percent', 0) <= -2.5)

        # Generate AI recommendations based on real data
        recommendations = []

        # Sort by performance
        sorted_stocks = sorted(stocks, key=lambda x: abs(x.get('change_percent', 0)), reverse=True)

        for stock in sorted_stocks[:3]:  # Top 3 movers
            change_pct = stock.get('change_percent', 0)

            if change_pct > 3:
                action = "STRONG BUY"
                reasoning = f"Strong upward momentum (+{change_pct:.1f}%). Technical indicators suggest continuation."
                confidence = 0.75
            elif change_pct > 1:
                action = "BUY"
                reasoning = f"Positive momentum (+{change_pct:.1f}%). Good entry point."
                confidence = 0.65
            elif change_pct < -3:
                action = "STRONG SELL"
                reasoning = f"Significant downward pressure ({change_pct:.1f}%). Risk of further decline."
                confidence = 0.70
            elif change_pct < -1:
                action = "SELL"
                reasoning = f"Negative momentum ({change_pct:.1f}%). Consider reducing position."
                confidence = 0.60
            else:
                action = "HOLD"
                reasoning = f"Consolidating around current levels. Wait for clearer signal."
                confidence = 0.55

            recommendations.append({
                "symbol": stock.get('symbol'),
                "action": action,
                "confidence": confidence,
                "reasoning": reasoning,
                "current_price": stock.get('price'),
                "change_percent": change_pct
            })

        # Derive simple trend signal from aggregate index change
        if total_change > 0.5:
            trend = "up"
        elif total_change < -0.5:
            trend = "down"
        else:
            trend = "neutral"

        return {
            "status": "success",
            "data": {
                "overview": {
                    "symbols_analyzed": total_analyzed,
                    "ai_models_used": ["sentiment", "risk", "recommendations"],
                },
                "market_sentiment": {
                    "overall_score": round(sentiment_score, 1),
                    "interpretation": sentiment_interpretation,
                    "trend": trend,
                    "bullish_stocks": bullish_stocks,
                    "total_analyzed": total_analyzed,
                },
                "risk_assessment": {
                    "overall_risk": risk_level,
                    "risk_percentage": risk_percentage,
                    "vix_level": vix_value,
                    "high_risk_stocks": high_risk_stocks,
                    "factors": [
                        f"VIX at {vix_value:.2f}",
                        f"Market sentiment: {sentiment_interpretation}",
                        f"Volatility: {risk_level}"
                    ],
                },
                "recommendations": recommendations,
                "last_updated": datetime.now().isoformat(),
            },
        }

    except Exception as e:
        logger.error(f"❌ Error getting market intelligence: {e}")
        raise HTTPException(status_code=500, detail=str(e))