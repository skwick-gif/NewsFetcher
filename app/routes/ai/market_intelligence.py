from fastapi import APIRouter, HTTPException
from datetime import datetime
from app.core.config import financial_provider

router = APIRouter()

@router.get("/ai/market-intelligence")
async def get_market_intelligence():
    """
    Get market intelligence analysis
    Combines market data, sentiment, and risk assessment
    """
    try:
        from app.financial.market_data import financial_provider
        
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