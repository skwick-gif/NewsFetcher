"""
News and Sentiment Endpoints
Real-time news analysis and sentiment tracking
"""
from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any
from datetime import datetime, timezone
import asyncio
from app.core.config import (
    logger, news_sentiment_provider, NEWS_SENTIMENT_AVAILABLE,
    real_data_loader, keyword_analyzer, TEMPLATES_AVAILABLE
)

router = APIRouter()

# ============================================================
# News & Sentiment Endpoints (Live-only, no mock data)
# ============================================================
@router.get("/sentiment/{symbol}", tags=["Sentiment"])
async def get_news_sentiment(symbol: str):
    """Get aggregated daily news sentiment for a symbol.

    Returns 200 with an empty data list and a friendly message when no data.
    """
    try:
        if not NEWS_SENTIMENT_AVAILABLE or not news_sentiment_provider:
            raise HTTPException(status_code=503, detail="News sentiment provider unavailable")

        # Run blocking analyzer in a thread
        result = await asyncio.to_thread(news_sentiment_provider.fetch_daily_sentiment, symbol)

        if not result.get("data"):
            return {
                "status": "success",
                "symbol": symbol,
                "data": [],
                "message": "No sentiment data available",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        return {
            "status": "success",
            "symbol": symbol,
            "data": result.get("data", []),
            "days": result.get("days", len(result.get("data", []))),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error getting news sentiment for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{symbol}", tags=["News"])
async def get_recent_news(symbol: str, limit: int = 20):
    """Get recent news articles for a symbol.

    Returns 200 with an empty articles array and a friendly message when no data.
    """
    try:
        if not NEWS_SENTIMENT_AVAILABLE or not news_sentiment_provider:
            raise HTTPException(status_code=503, detail="News sentiment provider unavailable")

        result = await asyncio.to_thread(news_sentiment_provider.fetch_recent_news, symbol, min(max(limit, 1), 50))

        if not result.get("articles"):
            return {
                "status": "success",
                "symbol": symbol,
                "articles": [],
                "message": "No sentiment data available",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        return {
            "status": "success",
            "symbol": symbol,
            "count": result.get("count", len(result.get("articles", []))),
            "articles": result.get("articles", []),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error getting news articles for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sentiment/providers", tags=["Sentiment"])
async def get_sentiment_providers_health():
    """List availability of sentiment/news providers and related LLM keys.

    This is informational for ops; it does not change behavior.
    """
    try:
        if not NEWS_SENTIMENT_AVAILABLE or not news_sentiment_provider:
            # Still report social/LLM availability even if news provider import failed
            from app.financial.news_sentiment_provider import NewsSentimentProvider as _NSP  # type: ignore
            tmp = _NSP(days_back=1, ttl_seconds=60)
            health = tmp.get_provider_health()
        else:
            health = news_sentiment_provider.get_provider_health()

        return {
            "status": "success",
            "providers": health,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.error(f"❌ Error getting providers health: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/recent", tags=["Articles"])
async def get_recent_articles(limit: int = Query(20, ge=1, le=100)):
    """Get recent financial articles"""
    try:
        if not real_data_loader:
            # Live only: return explicit error (no demo articles)
            raise HTTPException(status_code=503, detail="RSS system unavailable")
        # REAL data path: fetch RSS and enrich with keyword analysis
        fetched = await real_data_loader.fetch_all_rss_feeds(tier="major_news")
        articles = []

        # Use keyword analyzer if available
        use_keywords = keyword_analyzer is not None

        for idx, art in enumerate(fetched[:limit]):
            score = 0.0
            llm_relevant = False
            llm_tags = []

            if use_keywords:
                try:
                    ka = keyword_analyzer.analyze_article(art)
                    # Map keyword_score (-3..+3 typical) to 0..1 range for UI
                    raw = float(ka.get("keyword_score", 0.0))
                    norm = (raw + 3.0) / 6.0
                    score = max(0.0, min(1.0, norm))
                    # Basic relevance decision
                    llm_relevant = score >= 0.4
                    # Build tags from top categories or alert level
                    if ka.get("alert_level") and ka["alert_level"] != "none":
                        llm_tags.append(f"alert:{ka['alert_level']}")
                    if ka.get("sentiment"):
                        llm_tags.append(f"sentiment:{ka['sentiment']}")
                except Exception as e:
                    logger.warning(f"Keyword analysis failed: {e}")

            articles.append({
                "id": art.get("id", f"rss_{idx}"),
                "title": art.get("title", ""),
                "content": art.get("content", ""),
                "url": art.get("url", ""),
                "source": art.get("source_name", art.get("source", "rss")),
                # UI expects 'published' or 'timestamp'
                "published": art.get("published_at") or art.get("fetched_at"),
                "timestamp": art.get("published_at") or art.get("fetched_at"),
                # Scoring fields expected by UI
                "score": round(score, 2),
                "llm_checked": use_keywords,
                "llm_relevant": llm_relevant,
                "llm_summary": None,
                "llm_tags": llm_tags
            })

        return {
            "status": "success",
            "count": len(articles),
            "articles": articles
        }

    except Exception as e:
        logger.error(f"Failed to fetch articles: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch articles")