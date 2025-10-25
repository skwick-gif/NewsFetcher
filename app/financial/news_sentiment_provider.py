"""
News sentiment provider wrapper with simple in-memory caching and provider health.

Uses the existing app.data.sentiment_analyzer.SentimentAnalyzer to fetch real
news from NewsAPI, Yahoo, Alpha Vantage, and Bing and computes daily sentiment.

Policy:
- Live-only. No mock data. If providers yield no data, callers should surface a
  friendly "No sentiment data available" message to users.
- Lightweight in-memory cache to avoid rate limits (default TTL: 10 minutes).
"""
from __future__ import annotations

import os
import threading
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from app.data.sentiment_analyzer import SentimentAnalyzer
import pandas as pd
import math
import yaml


class NewsSentimentProvider:
    """Aggregates news and daily sentiment for a ticker with TTL cache."""

    def __init__(self, days_back: int = 14, ttl_seconds: int = 600) -> None:
        # Analyzer will auto-resolve keys from app/data/config.yaml or env
        env_days = os.getenv("NEWS_SENTIMENT_DAYS_BACK")
        if env_days and env_days.isdigit():
            days_back = int(env_days)

        env_ttl = os.getenv("NEWS_SENTIMENT_TTL_SECONDS")
        if env_ttl and env_ttl.isdigit():
            ttl_seconds = int(env_ttl)

        self._analyzer = SentimentAnalyzer(days_back=days_back)
        self._ttl = timedelta(seconds=ttl_seconds)
        self._cache: Dict[str, Tuple[datetime, Dict[str, Any]]] = {}
        self._cache_lock = threading.Lock()

    def get_provider_health(self) -> Dict[str, Any]:
        """Report which provider keys are available based on env/config resolution.

        Note: SentimentAnalyzer resolves keys internally from config/env for
        NewsAPI/Alpha Vantage/Bing. Here we only surface env-based presence for
        broader visibility, including social and LLM keys.
        """
        # Load keys from app/data/config.yaml as fallback
        cfg_newsapi = None
        cfg_alpha = None
        cfg_bing = None
        try:
            cfg_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data", "config.yaml"))
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            api_keys = (cfg.get("api_keys") or {})
            cfg_newsapi = api_keys.get("newsapi")
            cfg_alpha = api_keys.get("alpha_vantage")
            cfg_bing = api_keys.get("bing_news")
        except Exception:
            pass

        def resolve_cfg(val: Any) -> Optional[str]:
            if not val or not isinstance(val, str):
                return None
            v = val.strip()
            if v.startswith("${") and v.endswith("}"):
                # placeholder to env var
                env_name = v[2:-1]
                return os.getenv(env_name) or None
            return v

        newsapi_present = bool(os.getenv("NEWSAPI_KEY") or resolve_cfg(cfg_newsapi))
        alpha_present = bool(os.getenv("ALPHA_VANTAGE_KEY") or resolve_cfg(cfg_alpha))
        bing_present = bool(os.getenv("BING_NEWS_API_KEY") or resolve_cfg(cfg_bing))

        return {
            "newsapi": newsapi_present,
            "alpha_vantage": alpha_present,
            "bing_news": bing_present,
            # Social providers (used by RealSocialMediaAnalyzer)
            "twitter": bool(os.getenv("TWITTER_BEARER_TOKEN") or os.getenv("TWITTER_API_KEY")),
            "reddit": bool(os.getenv("REDDIT_CLIENT_ID") and os.getenv("REDDIT_CLIENT_SECRET")),
            # LLMs for optional enrichment
            "perplexity": bool(os.getenv("PERPLEXITY_API_KEY")),
            "gemini": bool(os.getenv("GEMINI_API_KEY")),
        }

    def _get_cached(self, symbol: str) -> Optional[Dict[str, Any]]:
        now = datetime.utcnow()
        with self._cache_lock:
            item = self._cache.get(symbol.upper())
            if not item:
                return None
            ts, data = item
            if now - ts <= self._ttl:
                return data
            # expired
            self._cache.pop(symbol.upper(), None)
            return None

    def _set_cache(self, symbol: str, data: Dict[str, Any]) -> None:
        with self._cache_lock:
            self._cache[symbol.upper()] = (datetime.utcnow(), data)

    # Blocking methods meant to be called in a thread from async routes
    def fetch_daily_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Compute daily sentiment dataframe for a symbol and format as json-serializable."""
        cached = self._get_cached(symbol)
        if cached is not None and cached.get("type") == "daily_sentiment":
            return cached

        df = self._analyzer.process_ticker_sentiment(symbol)
        if df is None or df.empty:
            result = {"type": "daily_sentiment", "symbol": symbol.upper(), "data": []}
            self._set_cache(symbol, result)
            return result

        # Convert DataFrame to list of dict rows
        def _clean_float(val: Any, default: float = 0.0) -> float:
            try:
                if val is None:
                    return default
                # Handle pandas NA/NaN and non-finite floats
                if isinstance(val, (float, int)):
                    f = float(val)
                    if math.isfinite(f):
                        return f
                    return default
                if pd.isna(val):
                    return default
                return float(val)
            except Exception:
                return default

        rows = []
        for _, row in df.iterrows():
            rows.append({
                "date": str(row.get("date")),
                "sentiment_score": _clean_float(row.get("sentiment_score", 0.0), 0.0),
                "article_count": int(row.get("article_count", 0) or 0),
                "sentiment_std": _clean_float(row.get("sentiment_std", 0.0), 0.0),
                "subjectivity_avg": _clean_float(row.get("subjectivity_avg", 0.0), 0.0),
            })

        result = {
            "type": "daily_sentiment",
            "symbol": symbol.upper(),
            "data": rows,
            "days": len(rows),
        }
        self._set_cache(symbol, result)
        return result

    def fetch_recent_news(self, symbol: str, limit: int = 20) -> Dict[str, Any]:
        """Return a list of recent news articles for a symbol."""
        cache_key = f"news:{symbol.upper()}"
        cached = self._get_cached(cache_key)
        if cached is not None and cached.get("type") == "recent_news":
            return cached

        articles = self._analyzer.get_news_for_ticker(symbol)
        if not articles:
            result = {"type": "recent_news", "symbol": symbol.upper(), "articles": []}
            self._set_cache(cache_key, result)
            return result

        # Normalize shape, keep only essential fields
        normalized: List[Dict[str, Any]] = []
        for a in articles[:limit]:
            normalized.append({
                "title": a.get("title", ""),
                "published": a.get("publishedAt") or a.get("published_at") or "",
                "source": (a.get("source") or {}).get("name", ""),
                "url": a.get("url", ""),
                "description": a.get("description", ""),
            })

        result = {
            "type": "recent_news",
            "symbol": symbol.upper(),
            "count": len(normalized),
            "articles": normalized,
        }
        self._set_cache(cache_key, result)
        return result
