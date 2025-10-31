from fastapi import APIRouter, HTTPException, Query
from typing import List, Dict, Any
import asyncio
from datetime import datetime, timezone
import logging

from app.core.config import (
    financial_provider,
    real_data_loader,
    news_impact_analyzer,
    keyword_analyzer,
    PROGRESSIVE_ML_AVAILABLE
)

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/articles/recent", tags=["Articles"])
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

@router.get("/financial/sector-performance", tags=["Financial"])
async def get_sector_performance():
    """Get sector performance analysis"""
    try:
        if financial_provider:
            return await financial_provider.get_sector_performance()
        # Live only: no sector performance without provider
        raise HTTPException(status_code=503, detail="Financial provider unavailable")

    except Exception as e:
        logger.error(f"Failed to get sector performance: {e}")
        raise HTTPException(status_code=500, detail="Failed to get sector performance")

@router.get("/financial/geopolitical-risks", tags=["Financial"])
async def get_geopolitical_risks():
    """Get geopolitical risk assessment"""
    try:
        # Validate required components
        if real_data_loader is None:
            raise HTTPException(status_code=503, detail="RSS loader unavailable")
        if news_impact_analyzer is None:
            raise HTTPException(status_code=503, detail="News impact analyzer unavailable")

        # Fetch recent RSS articles from relevant tiers (keep fast/lightweight)
        relevant_tiers = ["major_news", "global_markets", "chinese_news"]
        all_articles: List[Dict[str, Any]] = []
        try:
            for tier in relevant_tiers:
                articles = await real_data_loader.fetch_all_rss_feeds(tier=tier)
                all_articles.extend(articles)
                # Be a good citizen between tiers
                await asyncio.sleep(0.2)
        except Exception as e:
            logger.warning(f"Failed fetching RSS tiers for geopolitics: {e}")

        # If nothing fetched, return friendly 200 with no data
        if not all_articles:
            return {
                "status": "success",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data": {
                    "risk_level": "Low",
                    "risk_score": 0.0,
                    "factors": [],
                    "affected_sectors": [],
                    "events": [],
                    "overall_assessment": "No significant geopolitical signals detected in the last 48h"
                }
            }

        # Filter for geopolitically relevant articles
        geo_terms = {
            "war", "conflict", "sanction", "sanctions", "tariff", "trade", "trade war",
            "geopolitic", "geopolitical", "military", "missile", "cyberattack", "embargo",
            "taiwan", "ukraine", "russia", "china", "middle east", "israel", "gaza", "iran",
            "south china sea", "red sea", "strait", "blockade", "coup", "border clash", "naval"
        }

        def is_geo_article(a: Dict[str, Any]) -> bool:
            text = f"{a.get('title','')} {a.get('content','')}".lower()
            return any(term in text for term in geo_terms)

        geo_articles = [a for a in all_articles if is_geo_article(a)]

        # If none matched, provide low risk response
        if not geo_articles:
            return {
                "status": "success",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data": {
                    "risk_level": "Low",
                    "risk_score": 0.0,
                    "factors": [],
                    "affected_sectors": [],
                    "events": [],
                    "overall_assessment": "No geopolitically-relevant headlines detected in the last 48h"
                }
            }

        # Compute summary risk using analyzer (0-1 score, Low/Medium/High/Critical)
        news_texts = [f"{a.get('title','')}. {a.get('content','')}" for a in geo_articles]
        summary = await news_impact_analyzer.calculate_geopolitical_risk(news_texts)

        # Build top events ranked by per-article geopolitical risk (0-10)
        events: List[Dict[str, Any]] = []
        for art in geo_articles:
            try:
                # Map source field for analyzer
                mapped_article = {
                    "id": art.get("id", art.get("url", "")),
                    "title": art.get("title", ""),
                    "content": art.get("content", ""),
                    "source": art.get("source_name", art.get("source", "rss")),
                    "published_at": art.get("published_at"),
                }
                analysis = news_impact_analyzer.analyze_article_impact(mapped_article)
                events.append({
                    "title": art.get("title", ""),
                    "url": art.get("url", ""),
                    "source": art.get("source_name", art.get("source", "rss")),
                    "published_at": art.get("published_at"),
                    "risk_score": analysis.get("geopolitical_risk_score", 0.0),  # 0-10 scale
                    "affected_sectors": [s.get("sector") for s in analysis.get("affected_sectors", [])],
                    "symbols": art.get("symbols", []),
                })
            except Exception as e:
                logger.debug(f"Article analysis failed (geo): {e}")

        # Sort by risk score desc and take top N
        events.sort(key=lambda e: e.get("risk_score", 0.0), reverse=True)
        top_events = events[:10]

        # Compose response
        assessment = f"{summary.get('risk_level', 'Medium')} geopolitical risk (Score: {summary.get('risk_score', 0.0)}) based on {len(geo_articles)} articles"
        return {
            "status": "success",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": {
                "risk_level": summary.get("risk_level", "Medium"),
                "risk_score": summary.get("risk_score", 0.0),
                "factors": summary.get("factors", []),
                "affected_sectors": summary.get("affected_sectors", []),
                "events": top_events,
                "overall_assessment": assessment
            }
        }

    except Exception as e:
        logger.error(f"Failed to get geopolitical risks: {e}")
        raise HTTPException(status_code=500, detail="Failed to get geopolitical risks")

@router.get("/watchlist", tags=["Portfolio"])
async def get_watchlist():
    """Get user watchlist with live data"""
    try:
        from pathlib import Path as _Path
        symbols: list[str] = []

        # Prefer explicit watchlist file if exists
        watchlist_file = _Path('app/data/watchlist.json')
        if watchlist_file.exists():
            try:
                import json as _json
                with open(watchlist_file, 'r', encoding='utf-8') as f:
                    payload = _json.load(f)
                    if isinstance(payload, dict) and isinstance(payload.get('symbols'), list):
                        symbols = [s.upper() for s in payload['symbols'] if isinstance(s, str)]
            except Exception:
                symbols = []

        # Otherwise, derive from available champions (top 10 most recent)
        if not symbols:
            champions_root = _Path('app/ml/models/champions')
            if champions_root.exists():
                try:
                    syms = []
                    for p in champions_root.iterdir():
                        if p.is_dir():
                            syms.append((p.name, p.stat().st_mtime))
                    syms.sort(key=lambda x: x[1], reverse=True)
                    symbols = [s for s, _ in syms[:10]]
                except Exception:
                    symbols = []

        watchlist_items = []
        for sym in symbols:
            try:
                stock_data = await financial_provider.get_stock_data(sym) if financial_provider else None
                name = (stock_data or {}).get('name', sym)
                price = None
                change_str = '0.00%'
                if stock_data:
                    price = stock_data.get('price') or stock_data.get('current_price')
                    chg = stock_data.get('change_percent')
                    if isinstance(chg, (int, float)):
                        sign = '+' if chg >= 0 else ''
                        change_str = f"{sign}{chg:.2f}%"

                item = {
                    'symbol': sym,
                    'name': name,
                    'price': round(float(price), 2) if isinstance(price, (int, float)) else None,
                    'change': change_str
                }

                # Enrich with Progressive ML champion prediction (7d) and SL/TP
                try:
                    if PROGRESSIVE_ML_AVAILABLE:
                        from app.ml.progressive.predictor import ProgressivePredictor as _Pred
                        from app.ml.progressive.data_loader import ProgressiveDataLoader as _DL
                        champions_root = _Path('app/ml/models/champions') / sym
                        if champions_root.exists():
                            dirs = [p for p in champions_root.iterdir() if p.is_dir()]
                            dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                            if dirs:
                                champ_dir = str(dirs[0])
                                _loader = _DL(stock_data_dir=_Path('stock_data'))
                                _pred = _Pred(data_loader=_loader, model_dir=champ_dir)
                                _res = _pred.predict_ensemble(symbol=sym, mode='progressive')
                                if _res and _res.get('predictions'):
                                    p7 = _res['predictions'].get('7d') or _res['predictions'].get('1d')
                                    if p7:
                                        item['expected_return'] = float(p7.get('price_change_pct', 0.0) * 100.0)
                                        item['confidence'] = float(p7.get('confidence', 0.0))
                                        item['recommendation'] = p7.get('signal')
                                        # Risk enrich (ATR or volatility) similar to hot-stocks
                                        try:
                                            import pandas as __pd
                                            ind_path = _Path('stock_data') / sym / f"{sym}_indicators.csv"
                                            price_path = _Path('stock_data') / sym / f"{sym}_price.csv"
                                            close_price = float(_res.get('current_price', 0.0))
                                            atr_pct = None
                                            if ind_path.exists():
                                                ind_df = __pd.read_csv(ind_path, index_col=0)
                                                if 'ATR_14' in ind_df.columns and close_price > 0:
                                                    atr_val = float(__pd.to_numeric(ind_df['ATR_14'], errors='coerce').dropna().iloc[-1])
                                                    atr_pct = max(0.001, min(0.2, atr_val / close_price))
                                            if atr_pct is None and price_path.exists():
                                                dfp = __pd.read_csv(price_path, index_col=0)
                                                dfp['Close'] = __pd.to_numeric(dfp['Close'], errors='coerce')
                                                dfp = dfp.dropna(subset=['Close'])
                                                rets = dfp['Close'].pct_change().dropna()
                                                vol = float(rets.rolling(14).std().dropna().iloc[-1]) if len(rets) > 14 else float(rets.std())
                                                atr_pct = max(0.001, min(0.2, vol * 1.5))
                                            rr = 2.0
                                            risk_pct = max(0.005, min(0.2, atr_pct or 0.01))
                                            reward_pct = max(0.01, min(0.4, risk_pct * rr))
                                            change_pct = float(p7.get('price_change_pct', 0.0))
                                            sl = close_price * (1 - risk_pct)
                                            tp = close_price * (1 + reward_pct)
                                            if change_pct < 0:
                                                sl = close_price * (1 + risk_pct)
                                                tp = close_price * (1 - reward_pct)
                                            item['risk_7d'] = {
                                                'stop_loss': round(sl, 4),
                                                'take_profit': round(tp, 4),
                                                'stop_loss_pct': -risk_pct if change_pct >= 0 else risk_pct,
                                                'take_profit_pct': reward_pct if change_pct >= 0 else -reward_pct,
                                                'basis': 'ATR_14' if atr_pct is not None else 'volatility',
                                                'rr': rr
                                            }
                                        except Exception:
                                            pass
                except Exception as _we:
                    logger.debug(f"Watchlist enrich failed for {sym}: {_we}")

                watchlist_items.append(item)
            except Exception as _ie:
                logger.debug(f"Watchlist item build failed for {sym}: {_ie}")

        return {
            "status": "success",
            "watchlist": watchlist_items
        }

    except Exception as e:
        logger.error(f"Failed to get watchlist: {e}")
        raise HTTPException(status_code=500, detail="Failed to get watchlist")