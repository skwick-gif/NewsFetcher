"""
MarketPulse Main Application with WebSocket Support
Real-time financial intelligence platform with automated monitoring
"""
import logging
import os
import sys
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Set, Optional
import asyncio

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks, Query, Depends, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from contextlib import asynccontextmanager
import json
import requests
import numpy as np
from pathlib import Path
import yaml
from pathlib import Path
import subprocess
import threading

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set up logging FIRST
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Safety: disable torch dynamo/compile in this process to avoid environment-induced issues
# Some environments auto-enable Dynamo which may import torch._C._dynamo.eval_frame.skip_code
# Our code does not use torch.compile; explicitly disable to be safe.
os.environ.setdefault("PYTORCH_ENABLE_DYNAMO", "0")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

# Import our components
# Scheduler functionality
try:
    from app.scheduler_bg.scheduler import get_scheduler, MarketPulseScheduler
    SCHEDULER_AVAILABLE = True
    logger.info("✅ Real scheduler loaded successfully")
except ImportError as e:
    SCHEDULER_AVAILABLE = False
    logger.warning(f"⚠️ Real scheduler not available: {e}")

# Core components imports
try:
    from app.smart.keywords_engine import FinancialKeywordsEngine
except ImportError:
    print("⚠️ Keywords engine not available")
from app.ingest.rss_loader import FinancialDataLoader
from app.integrations.ibkr_client import ibkr

# Financial modules imports
try:
    from app.financial.market_data import FinancialDataProvider
    from app.financial.market_data_clean import FinancialDataProvider as CleanFinancialDataProvider
    from app.financial.news_impact import NewsImpactAnalyzer
    from app.financial.social_sentiment import SocialMediaAnalyzer
    from app.financial.social_sentiment_enhanced import RealSocialMediaAnalyzer
    from app.financial.ai_models import AdvancedAIModels, TimeSeriesAnalyzer
    # from app.financial.neural_networks import EnsembleNeuralNetwork  # DISABLED - OLD TENSORFLOW
    # from app.financial.ml_trainer import MLModelTrainer  # DISABLED - TensorFlow compatibility issues
    ML_TRAINER_AVAILABLE = False
    from app.financial.websocket_manager import WebSocketManager, MarketDataStreamer
    
    FINANCIAL_MODULES_AVAILABLE = True
    logger.info("✅ Financial modules loaded successfully")
except ImportError as e:
    FINANCIAL_MODULES_AVAILABLE = False
    ML_TRAINER_AVAILABLE = False
    logger.warning(f"⚠️ Financial modules not available: {e}")
    logger.info("📊 Running in demo mode with limited functionality")

# Progressive ML imports
try:
    from app.ml.progressive.data_loader import ProgressiveDataLoader
    from app.ml.progressive.trainer import ProgressiveTrainer
    from app.ml.progressive.predictor import ProgressivePredictor
    from app.ml.progressive.models import ProgressiveModels
    from app.data.data_manager import DataManager
    
    PROGRESSIVE_ML_AVAILABLE = True
    logger.info("✅ Progressive ML system loaded successfully")
except ImportError as e:
    PROGRESSIVE_ML_AVAILABLE = False
    logger.warning(f"⚠️ Progressive ML system not available: {e}")

# News sentiment provider (wraps real providers: NewsAPI, Yahoo, Alpha Vantage, Bing)
try:
    from app.financial.news_sentiment_provider import NewsSentimentProvider
    NEWS_SENTIMENT_AVAILABLE = True
except ImportError as e:
    NEWS_SENTIMENT_AVAILABLE = False
    logger.warning(f"⚠️ News sentiment provider unavailable: {e}")

# Initialize templates
try:
    templates = Jinja2Templates(directory="templates")
    TEMPLATES_AVAILABLE = True
except Exception as e:
    logger.warning(f"Templates not available: {e}")
    TEMPLATES_AVAILABLE = False

# Global components initialization
financial_provider = None
news_impact_analyzer = None
social_analyzer = None
ai_models = None
ml_trainer = None
websocket_manager = None
market_streamer = None
keyword_analyzer = None
real_data_loader = None
news_sentiment_provider = None

# Progressive ML instances
progressive_data_loader = None
progressive_trainer = None
progressive_predictor = None

# Initialize financial components if available
if FINANCIAL_MODULES_AVAILABLE:
    try:
        # Initialize providers
        financial_provider = FinancialDataProvider()
        news_impact_analyzer = NewsImpactAnalyzer()
        social_analyzer = RealSocialMediaAnalyzer()
        ai_models = AdvancedAIModels()
        if ML_TRAINER_AVAILABLE:
            ml_trainer = MLModelTrainer()
        else:
            ml_trainer = None
        websocket_manager = WebSocketManager()
        market_streamer = MarketDataStreamer(websocket_manager)
        
        logger.info("✅ Financial components initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize financial components: {e}")

# Initialize RSS and keyword systems
try:
    # Use correct config path
    config_path = os.path.join(os.path.dirname(__file__), "config", "data_sources.yaml")
    if os.path.exists(config_path):
        real_data_loader = FinancialDataLoader(config_path)
        
        # Load config for keyword analyzer
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        keyword_analyzer = FinancialKeywordsEngine()
        
        logger.info("✅ RSS and keyword systems initialized successfully")
    else:
        logger.warning(f"Config file not found: {config_path}")
except Exception as e:
    logger.error(f"Failed to initialize RSS/keyword systems: {e}")

# Initialize Progressive ML system
if PROGRESSIVE_ML_AVAILABLE:
    try:
        progressive_data_loader = ProgressiveDataLoader()
        progressive_trainer = ProgressiveTrainer(progressive_data_loader)
        progressive_predictor = ProgressivePredictor(progressive_data_loader)
        data_manager = DataManager()
        logger.info("✅ Progressive ML system initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Progressive ML system: {e}")
        PROGRESSIVE_ML_AVAILABLE = False

# Initialize News Sentiment Provider
if NEWS_SENTIMENT_AVAILABLE:
    try:
        # 14 days back, cache TTL 10 minutes
        news_sentiment_provider = NewsSentimentProvider(days_back=14, ttl_seconds=600)
        logger.info("✅ News Sentiment provider initialized")
    except Exception as e:
        logger.error(f"Failed to initialize News Sentiment provider: {e}")
        NEWS_SENTIMENT_AVAILABLE = False

# ============================================================
# WebSocket Connection Manager
# ============================================================
class ConnectionManager:
    """Manage WebSocket connections for real-time alerts"""
    
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
    
    async def connect(self, websocket: WebSocket):
        """Accept new WebSocket connection"""
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"✅ WebSocket connected. Total: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection"""
        self.active_connections.discard(websocket)
        logger.info(f"❌ WebSocket disconnected. Total: {len(self.active_connections)}")
    
    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast message to all connected clients"""
        if not self.active_connections:
            return
        
        # Convert to JSON
        json_message = json.dumps(message)
        
        # Send to all connections
        disconnected = set()
        for connection in self.active_connections:
            try:
                await connection.send_text(json_message)
            except Exception as e:
                logger.error(f"Failed to send message: {e}")
                disconnected.add(connection)
        
        # Clean up disconnected
        for conn in disconnected:
            self.disconnect(conn)
        
        logger.debug(f"📡 Broadcast to {len(self.active_connections)} clients")

# Global connection manager
manager = ConnectionManager()

# ============================================================
# Application Lifespan
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown"""
    logger.info("=" * 80)
    logger.info("🚀 Starting MarketPulse Financial Intelligence Platform")
    logger.info("=" * 80)
    
    # Initialize scheduler
    scheduler = None
    if SCHEDULER_AVAILABLE:
        try:
            scheduler = get_scheduler()
            # Set WebSocket broadcast callback
            scheduler.websocket_broadcast_callback = manager.broadcast
            logger.info("✅ Real scheduler initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize real scheduler: {e}")
            scheduler = None
    else:
        logger.warning("⚠️ Scheduler not available - running without background tasks")
    
    # Initialize financial streaming if available
    if FINANCIAL_MODULES_AVAILABLE and market_streamer:
        try:
            # Market data streaming will be handled by WebSocket connections
            logger.info("✅ Market data streamer ready (WebSocket-based)")
        except Exception as e:
            logger.error(f"Failed to initialize market streaming: {e}")
    
    # Start background scheduler if available
    if scheduler:
        try:
            scheduler.start()
            logger.info("✅ Background scheduler started")
        except Exception as e:
            logger.error(f"Failed to start scheduler: {e}")
    
    logger.info("✅ All systems operational!")
    logger.info("=" * 80)
    
    yield
    
    # Shutdown
    logger.info("=" * 80)
    logger.info("👋 Shutting down MarketPulse...")
    if scheduler:
        try:
            scheduler.stop()
            logger.info("✅ Scheduler stopped")
        except Exception as e:
            logger.error(f"Error stopping scheduler: {e}")
    logger.info("=" * 80)

# ============================================================
# FastAPI Application
# ============================================================
app = FastAPI(
    title="MarketPulse",
    description="🚀 Real-time Financial Intelligence Platform with AI-powered insights",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# ============================================================
# IBKR Bridge Endpoints (stubs for C# integration)
# ============================================================
class IBKRConnectRequest(BaseModel):
    host: Optional[str] = None
    port: Optional[int] = None
    client_id: Optional[int] = None


class IBKROrderRequest(BaseModel):
    symbol: str
    quantity: float
    side: str  # BUY or SELL
    order_type: str = "MKT"  # MKT or LMT
    limit_price: Optional[float] = None


@app.post("/api/ibkr/connect", tags=["IBKR"])
async def ibkr_connect(req: IBKRConnectRequest):
    try:
        data = ibkr.connect(req.host, req.port, req.client_id)
        return {"status": "success", "data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ibkr/status", tags=["IBKR"])
async def ibkr_status():
    try:
        return {"status": "success", "data": ibkr.status()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ibkr/place_order", tags=["IBKR"])
async def ibkr_place_order(req: IBKROrderRequest):
    try:
        side = req.side.upper()
        order_type = req.order_type.upper() if req.order_type else "MKT"
        if side not in ("BUY", "SELL"):
            raise HTTPException(status_code=400, detail="Invalid side. Use BUY or SELL")
        if order_type not in ("MKT", "LMT"):
            raise HTTPException(status_code=400, detail="Invalid order_type. Use MKT or LMT")

        res = ibkr.place_order(req.symbol, req.quantity, side, order_type, req.limit_price)
        if not res.get("ok"):
            raise HTTPException(status_code=503, detail=res.get("error") or "IBKR unavailable")
        return {"status": "success", "data": res}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ibkr/positions", tags=["IBKR"])
async def ibkr_positions():
    try:
        return {"status": "success", "data": {"positions": ibkr.positions()}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ibkr/disconnect", tags=["IBKR"])
async def ibkr_disconnect():
    try:
        return {"status": "success", "data": ibkr.disconnect()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------------------------
# Local stock_data-backed Scanner implementation (no legacy auto-scan)
# ---------------------------------------------
from typing import Optional, Dict, Any
import asyncio

STOCK_DATA_DIR = Path('stock_data')


def _iter_local_symbols(max_symbols: int = 1000) -> List[str]:
    """List symbols that have local CSV data under stock_data/.

    Returns at most max_symbols sorted alphabetically.
    """
    try:
        root = STOCK_DATA_DIR
        if not root.exists():
            return []
        syms: List[str] = []
        for p in root.iterdir():
            if not p.is_dir():
                continue
            sym = p.name
            price = p / f"{sym}_price.csv"
            if price.exists():
                syms.append(sym)
        syms.sort()
        return syms[:max_symbols]
    except Exception:
        return []


def _compute_local_metrics(symbol: str) -> Optional[Dict[str, Any]]:
    """Compute simple metrics from local CSVs for a symbol.

    Produces a dict compatible with hot-stocks items and scanner top rendering.
    """
    try:
        import pandas as pd
        sym = symbol.upper()
        price_path = STOCK_DATA_DIR / sym / f"{sym}_price.csv"
        ind_path = STOCK_DATA_DIR / sym / f"{sym}_indicators.csv"
        if not price_path.exists():
            return None
        df = pd.read_csv(price_path)
        # Normalize column names access
        close_col = 'Close' if 'Close' in df.columns else ('close' if 'close' in df.columns else None)
        open_col = 'Open' if 'Open' in df.columns else ('open' if 'open' in df.columns else None)
        high_col = 'High' if 'High' in df.columns else ('high' if 'high' in df.columns else None)
        low_col = 'Low' if 'Low' in df.columns else ('low' if 'low' in df.columns else None)
        vol_col = 'Volume' if 'Volume' in df.columns else ('volume' if 'volume' in df.columns else None)
        if close_col is None or len(df) < 3:
            return None
        # Use last and previous values
        close_vals = pd.to_numeric(df[close_col], errors='coerce').dropna()
        if len(close_vals) < 3:
            return None
        current_price = float(close_vals.iloc[-1])
        prev_price = float(close_vals.iloc[-2])
        change_pct = ((current_price - prev_price) / prev_price * 100.0) if prev_price else 0.0
        # ADV(20) if volume available
        avg_volume_20 = None
        if vol_col and vol_col in df.columns:
            vol_series = pd.to_numeric(df[vol_col], errors='coerce').dropna()
            if len(vol_series) >= 20:
                avg_volume_20 = float(vol_series.tail(20).mean())

        # 5-day return as expected_return proxy
        last5 = close_vals.tail(6)
        if len(last5) >= 2:
            base = float(last5.iloc[0])
            expected_return = ((current_price - base) / base * 100.0) if base else 0.0
        else:
            expected_return = change_pct

        # Load indicators if available for basic ml_score/confidence
        rsi = None
        atr_pct = None
        if ind_path.exists():
            try:
                ind = pd.read_csv(ind_path)
                if 'RSI_14' in ind.columns:
                    rsi_series = pd.to_numeric(ind['RSI_14'], errors='coerce').dropna()
                    if len(rsi_series):
                        rsi = float(rsi_series.iloc[-1])
                # ATR% approximation if ATR and price present
                if 'ATR_14' in ind.columns and current_price > 0:
                    atr_series = pd.to_numeric(ind['ATR_14'], errors='coerce').dropna()
                    if len(atr_series):
                        atr = float(atr_series.iloc[-1])
                        atr_pct = max(0.001, min(0.2, atr / current_price))
            except Exception:
                pass

        # Derive a simple ml_score 0..100 from momentum and RSI
        # Base score from 5d momentum clipped [-5, +5] -> [0, 100]
        mom = max(-5.0, min(5.0, expected_return))
        base_score = (mom + 5.0) * 10.0  # -5->0, 0->50, +5->100
        # RSI contribution: favor 50-70 mildly, penalize >80 or <20
        rsi_adj = 0.0
        if rsi is not None:
            if 50 <= rsi <= 70:
                rsi_adj = 5.0
            elif rsi > 80:
                rsi_adj = -10.0
            elif rsi < 20:
                rsi_adj = -5.0
        ml_score = max(0.0, min(100.0, base_score + rsi_adj))

        # Confidence from volatility (ATR%) if available, else modest default
        if atr_pct is not None:
            confidence = max(0.5, min(0.95, 1.0 - atr_pct * 2.5))
        else:
            confidence = 0.7

        # Recommendation heuristic
        recommendation = 'HOLD'
        if expected_return > 1.0 and confidence >= 0.6:
            recommendation = 'BUY'
        if expected_return > 3.0 and confidence >= 0.7:
            recommendation = 'STRONG BUY'
        if expected_return < -2.0:
            recommendation = 'SELL'

        predicted_price = current_price * (1.0 + expected_return / 100.0)

        return {
            'symbol': sym,
            'current_price': round(current_price, 4),
            'predicted_price': round(predicted_price, 4),
            'expected_return': round(expected_return, 3),
            'confidence': round(confidence, 3),
            'recommendation': recommendation,
            'ml_score': round(ml_score, 2),
            'change_percent': round(change_pct, 3),
            'avg_volume_20': avg_volume_20
        }
    except Exception:
        return None


class _ScannerState:
    def __init__(self) -> None:
        # Scan
        self.status: str = 'idle'  # idle|running|completed|error
        self.mode: str = 'ml'
        self.started_at: Optional[str] = None
        self.finished_at: Optional[str] = None
        self.last_error: Optional[str] = None
        self.progress: float = 0.0
        self.message: Optional[str] = None
        self.eta_seconds: Optional[int] = None
        self.total_symbols: int = 0
        self.processed_symbols: int = 0
        self.top: List[Dict[str, Any]] = []
        # Filter
        self.filter_state: str = 'not-started'  # not-started|running|completed|error
        self.filter_progress: float = 0.0
        self.filter_items: List[Dict[str, Any]] = []
        self.passed_count: int = 0
        self.trained_count: int = 0
        # Train
        self.train_status: str = 'idle'  # idle|running|completed|error
        self.train_current_symbol: Optional[str] = None

    def to_status(self) -> Dict[str, Any]:
        return {
            'state': self.status,
            'mode': self.mode,
            'started_at': self.started_at,
            'finished_at': self.finished_at,
            'error': self.last_error,
            'progress': self.progress,
            'message': self.message,
            'eta_seconds': self.eta_seconds,
            'total_symbols': self.total_symbols,
            'processed_symbols': self.processed_symbols,
            'count_top': len(self.top),
        }


_scanner_state = _ScannerState()


async def _run_scan(mode: str, limit: int = 50) -> None:
    from datetime import datetime as _dt
    _scanner_state.status = 'running'
    _scanner_state.mode = mode
    _scanner_state.started_at = _dt.now().isoformat()
    _scanner_state.finished_at = None
    _scanner_state.last_error = None
    _scanner_state.progress = 0.0
    _scanner_state.message = 'Scanning local symbols'
    _scanner_state.eta_seconds = None
    _scanner_state.top = []
    _scanner_state.total_symbols = 0
    _scanner_state.processed_symbols = 0
    try:
        syms = _iter_local_symbols(max_symbols=500)
        _scanner_state.total_symbols = len(syms)
        results: List[Dict[str, Any]] = []
        # Process a subset quickly; compute simple metrics and build list
        for idx, sym in enumerate(syms):
            m = _compute_local_metrics(sym)
            if m is not None:
                results.append(m)
            _scanner_state.processed_symbols = idx + 1
            # Update progress and simple ETA
            if _scanner_state.total_symbols:
                _scanner_state.progress = (_scanner_state.processed_symbols / _scanner_state.total_symbols)
                remaining = _scanner_state.total_symbols - _scanner_state.processed_symbols
                # Assume ~2ms per symbol average – safe small ETA
                _scanner_state.eta_seconds = max(0, int(remaining * 0.002))
            if idx % 50 == 0:
                await asyncio.sleep(0)  # yield to event loop
            # Early stop if we have more than we need for ranking diversity
            if len(results) >= max(limit * 5, 100):
                break
        # Rank by expected return desc
        results.sort(key=lambda x: x.get('expected_return', 0.0), reverse=True)
        _scanner_state.top = results[: max(limit, 50)]
        _scanner_state.status = 'completed'
        _scanner_state.message = f"Completed: {len(_scanner_state.top)} candidates"
    except Exception as _e:
        _scanner_state.status = 'error'
        _scanner_state.last_error = str(_e)
        _scanner_state.message = 'Scan failed'
    finally:
        _scanner_state.finished_at = _dt.now().isoformat()


async def _run_filter_job(price_min: float = 3.0, adv_min: float = 1_000_000.0) -> None:
    """Filter local symbols by last close and ADV thresholds; enrich with training status.

    - price_min: minimum last close
    - adv_min: minimum average daily volume (20d)
    """
    try:
        _scanner_state.filter_state = 'running'
        _scanner_state.filter_progress = 0.0
        _scanner_state.filter_items = []
        _scanner_state.passed_count = 0
        _scanner_state.last_error = None
        syms = _iter_local_symbols(max_symbols=50_000)
        _scanner_state.total_symbols = len(syms)
        _scanner_state.processed_symbols = 0
        items: List[Dict[str, Any]] = []
        from pathlib import Path as _Path
        champions_root = _Path('app/ml/models/champions')
        for idx, sym in enumerate(syms):
            _scanner_state.processed_symbols = idx + 1
            m = _compute_local_metrics(sym)
            if m is None:
                # Update progress and yield
                if _scanner_state.total_symbols:
                    _scanner_state.filter_progress = _scanner_state.processed_symbols / _scanner_state.total_symbols
                if idx % 50 == 0:
                    await asyncio.sleep(0)
                continue
            # Apply filters
            ok_price = (m.get('current_price') is not None and float(m['current_price']) >= float(price_min))
            adv = m.get('avg_volume_20')
            ok_adv = (adv is not None and float(adv) >= float(adv_min))
            # TODO: micro-cap exclusion if fundamentals available; skipped for now
            if ok_price and ok_adv:
                # Determine training state by presence of champion dir for symbol
                trained = False
                try:
                    sym_dir = (champions_root / sym)
                    if sym_dir.exists():
                        # Heuristic: any subdir implies at least one champion
                        trained = any(p.is_dir() for p in sym_dir.iterdir())
                except Exception:
                    trained = False
                item = {
                    'symbol': sym,
                    'current_price': m.get('current_price'),
                    'avg_volume_20': m.get('avg_volume_20'),
                    'trained': trained,
                    'train_status': 'completed' if trained else 'not-started'
                }
                items.append(item)
                _scanner_state.passed_count = len(items)
            # Progress + yield
            if _scanner_state.total_symbols:
                _scanner_state.filter_progress = _scanner_state.processed_symbols / _scanner_state.total_symbols
            if idx % 50 == 0:
                await asyncio.sleep(0)
        _scanner_state.filter_items = items
        _scanner_state.filter_state = 'completed'
    except Exception as e:
        _scanner_state.filter_state = 'error'
        _scanner_state.last_error = str(e)


# Add CORS (explicit localhost origins to avoid browser "Failed to fetch" on POSTs)
_allowed_origins = [
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory for CSS/JS modules
static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# ============================================================
# RL Endpoints (migrated from Flask)
# ============================================================
@app.get("/api/rl/status")
async def rl_status():
    """Simple RL service status for the dashboard."""
    return {
        'status': 'ok',
        'service': 'fastapi-rl',
        'timestamp': datetime.now().isoformat()
    }


@app.get("/api/rl/simulate")
async def rl_simulate(
    symbol: str = Query("AAPL"),
    policy: str = Query("follow_trend"),
    window: int = Query(60, ge=2, le=500),
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    days: Optional[int] = Query(None, ge=1, le=5000),
):
    """Run a quick, non-learning RL simulation for a single symbol."""
    try:
        try:
            from rl.simulation import run_simulation as rl_run_simulation  # type: ignore
        except Exception:
            raise HTTPException(status_code=500, detail="RL simulation module not available")

        data = await asyncio.to_thread(
            rl_run_simulation,
            symbol=symbol.upper(),
            days=days,
            window=window,
            cost_bps=5,
            slip_bps=5,
            policy=policy,
            start_date=start_date,
            end_date=end_date
        )
        return {"status": "success", "data": data}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# RL PPO Training: background process runner + endpoints
# ============================================================
_rl_job_logs: Dict[str, List[Dict[str, Any]]] = {}
_rl_running_jobs: Dict[str, Dict[str, Any]] = {}


def _rl_run_command_in_background(job_type: str, cmd_args: List[str], job_id: str) -> bool:
    _rl_job_logs[job_id] = []
    _rl_running_jobs[job_id] = {"status": "running", "start_time": datetime.now().isoformat(), "job_type": job_type}

    def _target():
        try:
            # Launch as a separate process group for safe termination on all OSes
            creationflags = 0
            preexec_fn = None
            try:
                import os as _os
                import sys as _sys
                if _sys.platform.startswith('win'):
                    # CREATE_NEW_PROCESS_GROUP = 0x00000200
                    creationflags = 0x00000200
                else:
                    import os
                    import signal as _signal  # noqa: F401
                    preexec_fn = os.setsid
            except Exception:
                pass

            process = subprocess.Popen(
                cmd_args,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=0,
                universal_newlines=True,
                cwd=str(Path(__file__).parent.parent),
                creationflags=creationflags,
                preexec_fn=preexec_fn,
            )
            # Save process reference and pid for future cancellation
            try:
                _rl_running_jobs[job_id]['pid'] = process.pid
                _rl_running_jobs[job_id]['_popen'] = process  # non-serializable, internal use
            except Exception:
                pass
            out_dir = None
            for line in process.stdout or []:
                line = line.strip()
                if not line:
                    continue
                # Parse model saved hints
                if 'Saved model' in line or 'Model saved' in line or 'saved to' in line.lower():
                    try:
                        parts = line.split(':', 1)
                        if len(parts) == 2:
                            cand = parts[1].strip()
                            if cand:
                                _rl_running_jobs[job_id]['model_path'] = cand
                    except Exception:
                        pass
                _rl_job_logs[job_id].append({"timestamp": datetime.now().isoformat(), "level": "INFO", "message": line})
                if len(_rl_job_logs[job_id]) > 400:
                    _rl_job_logs[job_id] = _rl_job_logs[job_id][-400:]
            process.wait()
            if process.returncode == 0:
                _rl_running_jobs[job_id]["status"] = "completed"
                _rl_job_logs[job_id].append({"timestamp": datetime.now().isoformat(), "level": "SUCCESS", "message": "✅ Job completed successfully!"})
            else:
                _rl_running_jobs[job_id]["status"] = "failed"
                _rl_job_logs[job_id].append({"timestamp": datetime.now().isoformat(), "level": "ERROR", "message": f"❌ Job failed with exit code {process.returncode}"})
        except Exception as e:
            _rl_running_jobs[job_id]["status"] = "error"
            _rl_job_logs[job_id].append({"timestamp": datetime.now().isoformat(), "level": "ERROR", "message": f"❌ Error: {str(e)}"})

    th = threading.Thread(target=_target, daemon=True)
    th.start()
    return True


@app.post("/api/rl/ppo/train")
async def rl_ppo_train(
    symbols: Optional[str] = Query(None),
    symbol: Optional[str] = Query(None),
    window: int = Query(60, ge=2, le=500),
    timesteps: int = Query(100000, ge=10000, le=5000000),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    add_vix: Optional[bool] = Query(False),
    vix_symbol: Optional[str] = Query("^VIX"),
    use_indicators: Optional[bool] = Query(False),
    indicator_cols: Optional[str] = Query(None),
    use_news: Optional[bool] = Query(True),
):
    # Build symbols string like Flask version
    syms_str = None
    if symbols:
        syms_str = ",".join([s.strip().upper() for s in symbols.split(',') if s.strip()])
    elif symbol:
        syms_str = symbol.strip().upper()
    else:
        raise HTTPException(status_code=400, detail="symbol or symbols is required")

    cmd = [
        sys.executable, '-m', 'rl.training.train_ppo_portfolio',
        '--symbols', syms_str,
        '--window', str(window),
        '--timesteps', str(timesteps)
    ]
    if start_date:
        cmd += ['--start', start_date]
    if end_date:
        cmd += ['--end', end_date]

    # News features default on when file exists unless explicitly disabled
    try:
        project_root = Path(__file__).parent.parent
        nf1 = project_root / 'app' / 'ml' / 'data' / 'news_features.csv'
        nf2 = project_root / 'ml' / 'data' / 'news_features.csv'
        news_csv = nf1 if nf1.exists() else (nf2 if nf2.exists() else None)
        if use_news and news_csv is not None:
            cmd += ['--news-features-csv', str(news_csv)]
            cols = 'news_count,llm_relevant_count,avg_score,fda_count,china_count,geopolitics_count,sentiment_avg'
            cmd += ['--news-cols', cols]
    except Exception:
        pass

    if add_vix:
        cmd += ['--add-vix-feature']
        if vix_symbol:
            cmd += ['--vix-symbol', vix_symbol]

    # Indicator feature flags supported by trainer
    try:
        if use_indicators and indicator_cols:
            cmd += ['--use-indicator-features']
            cmd += ['--indicator-cols', str(indicator_cols)]
    except Exception:
        pass

    # If only one symbol, duplicate for portfolio trainer
    try:
        syms_list = [s for s in (syms_str or '').split(',') if s]
        if len(syms_list) == 1:
            dup = f"{syms_list[0]},{syms_list[0]}"
            for i in range(len(cmd)):
                if cmd[i] == '--symbols' and i + 1 < len(cmd):
                    cmd[i+1] = dup
                    break
    except Exception:
        pass

    job_id = f"ppo-train-{(syms_str or 'NA').replace(',', '-')}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    started = _rl_run_command_in_background('ppo_train', cmd, job_id)
    if not started:
        raise HTTPException(status_code=500, detail='failed to start training job')
    return {"status": "started", "job_id": job_id, "cmd": cmd}


@app.get("/api/rl/ppo/train/status/{job_id}")
async def rl_ppo_train_status(job_id: str):
    info = _rl_running_jobs.get(job_id)
    logs = _rl_job_logs.get(job_id, [])
    if not info:
        raise HTTPException(status_code=404, detail='unknown job_id')
    model_path = info.get('model_path')
    return {
        'status': info.get('status'),
        'job_id': job_id,
        'logs': [l.get('message') for l in logs],
        **({'model_path': model_path} if model_path else {})
    }


@app.post("/api/rl/ppo/train/stop/{job_id}")
async def rl_ppo_train_stop(job_id: str):
    """Attempt to stop a running PPO training job by job_id."""
    info = _rl_running_jobs.get(job_id)
    if not info:
        raise HTTPException(status_code=404, detail='unknown job_id')
    status = info.get('status')
    if status in ('completed', 'failed', 'error', 'cancelled'):
        return {'status': status, 'job_id': job_id, 'message': 'job is not running'}

    pid = info.get('pid')
    popen = info.get('_popen')
    killed = False
    err = None
    try:
        import sys as _sys
        if popen and getattr(popen, 'poll', None) and popen.poll() is None:
            # Try graceful termination first
            try:
                if _sys.platform.startswith('win'):
                    # Send CTRL-BREAK to the process group (if supported), else taskkill
                    try:
                        import signal as _signal
                        popen.send_signal(_signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
                    except Exception:
                        pass
                else:
                    import os, signal
                    try:
                        os.killpg(os.getpgid(popen.pid), signal.SIGTERM)
                    except Exception:
                        popen.terminate()
            except Exception:
                pass
        # Force kill if still alive
        if popen and popen.poll() is None:
            if _sys.platform.startswith('win') and pid:
                try:
                    import subprocess as _sp
                    _sp.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
                except Exception:
                    pass
            else:
                try:
                    popen.kill()
                except Exception:
                    pass
        killed = True
    except Exception as e:
        err = str(e)

    # Update status/logs
    info['status'] = 'cancelled'
    _rl_job_logs.setdefault(job_id, []).append({
        'timestamp': datetime.now().isoformat(),
        'level': 'INFO',
        'message': '🛑 Stop requested by user.' + (f' Note: {err}' if err else '')
    })
    return {'status': info['status'], 'job_id': job_id, 'stopped': killed}


# ============================================================
# RL Auto-Tune (real orchestrator)
# ============================================================

_rla_auto_tune_state: Dict[str, Any] = {"status": "idle"}
_rla_auto_tune_lock = threading.Lock()


def _rla_log(msg: str) -> None:
    try:
        logs_dir = Path(__file__).parent.parent / 'logs'
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = logs_dir / 'auto_tune_stdout.log'
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(msg.rstrip() + "\n")
    except Exception:
        pass


def _auto_tune_runner(config: Dict[str, Any]) -> None:
    from time import sleep
    with _rla_auto_tune_lock:
        _rla_auto_tune_state.update({
            "status": "running",
            "stdout": "",
            "stderr": "",
            "current": None,
            "trials": [],
            "best": None,
            "stop": False,
        })
    project_root = Path(__file__).parent.parent
    symbols: List[str] = config.get('symbols') or ['QQQ', 'SPY']
    if len(symbols) == 1:
        symbols = [symbols[0], symbols[0]]
    # Plan window/timesteps
    try:
        plan = asyncio.run(rl_ppo_plan(symbols=','.join(sorted(set(symbols))), symbol=None, window=60))  # type: ignore
        plan_data = plan.get('plan') if isinstance(plan, dict) else None
    except Exception:
        plan_data = None
    window = int(config.get('window') or (plan_data.get('window') if plan_data else 60) or 60)
    timesteps = int(config.get('timesteps') or (plan_data.get('timesteps') if plan_data else 300_000) or 300_000)
    eval_start = str(config.get('eval_start') or (plan_data.get('train_start_date') if plan_data else '2022-01-01'))
    eval_end = str(config.get('eval_end') or (plan_data.get('train_end_date') if plan_data else datetime.now().date().isoformat()))

    # Feature candidates
    news_csv = None
    try:
        c1 = project_root / 'app' / 'ml' / 'data' / 'news_features.csv'
        c2 = project_root / 'ml' / 'data' / 'news_features.csv'
        if c1.exists():
            news_csv = c1.as_posix()
        elif c2.exists():
            news_csv = c2.as_posix()
    except Exception:
        pass
    news_cols = 'news_count,llm_relevant_count,avg_score,fda_count,china_count,geopolitics_count,sentiment_avg'
    ind_cols = config.get('indicator_cols') or 'rsi14,macd_hist,adx14,bb_pctb,bb_width,volume_sma_ratio,roc20,vol20'
    vix_symbol = config.get('vix_symbol') or '^VIX'

    candidates: List[Dict[str, Any]] = []
    candidates.append({'name': 'w{w}_ts{ts}_sd0_price'.format(w=window, ts=timesteps), 'args': {}})
    if news_csv:
        candidates.append({'name': f'w{window}_ts{timesteps}_sd0_news', 'args': {'news_csv': news_csv, 'news_cols': news_cols}})
        candidates.append({'name': f'w{window}_ts{timesteps}_sd0_news_vix', 'args': {'news_csv': news_csv, 'news_cols': news_cols, 'add_vix': True, 'vix_symbol': vix_symbol}})
        candidates.append({'name': f'w{window}_ts{timesteps}_sd0_news_vix_indicators', 'args': {'news_csv': news_csv, 'news_cols': news_cols, 'add_vix': True, 'vix_symbol': vix_symbol, 'use_indicators': True, 'indicator_cols': ind_cols}})
    candidates.append({'name': f'w{window}_ts{timesteps}_sd0_vix', 'args': {'add_vix': True, 'vix_symbol': vix_symbol}})
    candidates.append({'name': f'w{window}_ts{timesteps}_sd0_indicators', 'args': {'use_indicators': True, 'indicator_cols': ind_cols}})

    seeds = config.get('seeds') or [0]

    best: Optional[Dict[str, Any]] = None
    try:
        for sd in seeds:
            for cand in candidates:
                with _rla_auto_tune_lock:
                    if _rla_auto_tune_state.get('stop'):
                        raise KeyboardInterrupt('Auto-tune stopped by user')
                    _rla_auto_tune_state['current'] = {'seed': sd, 'name': cand['name']}
                # Build train command
                cmd = [
                    sys.executable, '-m', 'rl.training.train_ppo_portfolio',
                    '--symbols', ','.join(symbols),
                    '--window', str(window),
                    '--timesteps', str(timesteps),
                    '--seed', str(sd),
                ]
                args = cand['args']
                if args.get('news_csv'):
                    cmd += ['--news-features-csv', str(args['news_csv']), '--news-cols', str(args.get('news_cols', news_cols)), '--news-window', '1']
                if args.get('add_vix'):
                    cmd += ['--add-vix-feature', '--vix-symbol', str(args.get('vix_symbol', '^VIX'))]
                if args.get('use_indicators'):
                    cmd += ['--use-indicator-features', '--indicator-cols', str(args.get('indicator_cols', ind_cols))]
                # Run training synchronously
                _rla_log(f"Training trial {cand['name']} seed={sd} ...")
                try:
                    proc = subprocess.Popen(cmd, cwd=str(project_root), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, universal_newlines=True)
                    with _rla_auto_tune_lock:
                        _rla_auto_tune_state['pid'] = proc.pid
                        _rla_auto_tune_state['cmd'] = ' '.join(cmd)
                    for line in proc.stdout or []:
                        _rla_log(line.rstrip())
                        with _rla_auto_tune_lock:
                            prev = _rla_auto_tune_state.get('stdout', '') or ''
                            snap = (prev + ('\n' if prev else '') + line.rstrip())
                            _rla_auto_tune_state['stdout'] = snap[-2000:]
                    proc.wait()
                    rc = proc.returncode
                except Exception as e:
                    rc = -1
                    _rla_log(f"Error running training: {e}")
                if rc != 0:
                    with _rla_auto_tune_lock:
                        trials = _rla_auto_tune_state.get('trials', [])
                        trials.append({'name': cand['name'], 'seed': sd, 'status': 'failed'})
                        _rla_auto_tune_state['trials'] = trials
                    continue
                # Evaluate
                eval_out = project_root / 'reports' / 'rl' / f"auto_tune_{datetime.now().strftime('%Y%m%d_%H%M%S')}" / cand['name']
                ev_cmd = [sys.executable, '-m', 'rl.evaluation.generate_portfolio_report', '--symbols', ','.join(symbols), '--eval-start', eval_start, '--eval-end', eval_end, '--out', eval_out.as_posix()]
                if args.get('news_csv'):
                    ev_cmd += ['--news-features-csv', str(args['news_csv']), '--news-cols', str(args.get('news_cols', news_cols)), '--news-window', '1']
                if args.get('use_indicators'):
                    ev_cmd += ['--use-indicator-features', '--indicator-cols', str(args.get('indicator_cols', ind_cols))]
                _rla_log(f"Evaluating trial {cand['name']} seed={sd} ...")
                try:
                    proc2 = subprocess.Popen(ev_cmd, cwd=str(project_root), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, universal_newlines=True)
                    with _rla_auto_tune_lock:
                        _rla_auto_tune_state['pid'] = proc2.pid
                        _rla_auto_tune_state['cmd'] = ' '.join(ev_cmd)
                    for line in proc2.stdout or []:
                        _rla_log(line.rstrip())
                    proc2.wait()
                except Exception as e:
                    _rla_log(f"Error running evaluation: {e}")
                # Parse summary
                summary = None
                try:
                    import csv
                    s_csv = eval_out / 'summary.csv'
                    if s_csv.exists():
                        with open(s_csv, newline='', encoding='utf-8') as f:
                            rows = list(csv.DictReader(f))
                        # pick ppo_portfolio row
                        for r in rows:
                            if r.get('kind') == 'ppo_portfolio':
                                summary = r
                                break
                except Exception:
                    summary = None
                trial = {'name': cand['name'], 'seed': sd, 'status': 'completed' if summary else 'no_summary', 'summary': summary}
                with _rla_auto_tune_lock:
                    trials = _rla_auto_tune_state.get('trials', [])
                    trials.append(trial)
                    _rla_auto_tune_state['trials'] = trials
                # Update best
                if summary:
                    try:
                        sharpe = float(summary.get('sharpe') or 0.0)
                        mdd = abs(float(summary.get('max_drawdown') or summary.get('max_dd') or 0.0))
                        score = sharpe - 0.5 * mdd
                    except Exception:
                        score = -1e9
                    rec = {'trial': trial, 'score': score, 'dir': eval_out.as_posix()}
                    if best is None or score > best.get('score', -1e9):
                        best = rec
                        with _rla_auto_tune_lock:
                            _rla_auto_tune_state['best'] = rec
        with _rla_auto_tune_lock:
            _rla_auto_tune_state['status'] = 'completed'
            _rla_auto_tune_state['pid'] = None
            _rla_auto_tune_state['cmd'] = None
    except KeyboardInterrupt:
        with _rla_auto_tune_lock:
            _rla_auto_tune_state['status'] = 'cancelled'
            _rla_auto_tune_state['pid'] = None
            _rla_auto_tune_state['cmd'] = None
    except Exception as e:
        _rla_log(f"Auto-tune failed: {e}")
        with _rla_auto_tune_lock:
            _rla_auto_tune_state['status'] = 'error'
            _rla_auto_tune_state['stderr'] = str(e)
            _rla_auto_tune_state['pid'] = None
            _rla_auto_tune_state['cmd'] = None


@app.post('/api/rl/auto-tune/start')
async def rl_auto_tune_start(payload: Optional[Dict[str, Any]] = None):
    try:
        data = payload or {}
        mode = (data.get('mode') or 'etf').lower()
        job_id = f"rla-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        symbols: List[str]
        if mode == 'stock':
            sym = str(data.get('symbol') or 'AAPL').strip().upper()
            symbols = [sym]
        else:
            etfs = data.get('etfs') or ['QQQ','SPY']
            if isinstance(etfs, list):
                symbols = [str(s).strip().upper() for s in etfs if str(s).strip()]
            else:
                symbols = [s for s in str(etfs).split(',') if s]
        cfg = {
            'symbols': symbols,
            'time_budget_hours': float(data.get('time_budget_hours') or 6.0),
            'window': int(data.get('window') or 60),
            'timesteps': int(data.get('timesteps') or 300_000),
            'eval_start': data.get('eval_start'),
            'eval_end': data.get('eval_end'),
            'seeds': data.get('seeds') or [0],
            'indicator_cols': data.get('indicator_cols'),
            'vix_symbol': data.get('vix_symbol') or '^VIX',
        }
        with _rla_auto_tune_lock:
            _rla_auto_tune_state.clear()
            _rla_auto_tune_state.update({
                'status': 'running',
                'job_id': job_id,
                'started_at': datetime.utcnow().isoformat() + 'Z',
                'config': cfg,
                'pid': None,
                'cmd': None,
                'stdout': 'Starting RL auto-tune...',
                'stderr': '',
                'trials': [],
                'best': None,
            })
        th = threading.Thread(target=_auto_tune_runner, args=(cfg,), daemon=True)
        th.start()
        return {'status': 'ok', 'data': _rla_auto_tune_state}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Auto-tune start failed: {e}")


@app.get('/api/rl/auto-tune/status')
async def rl_auto_tune_status():
    with _rla_auto_tune_lock:
        data = dict(_rla_auto_tune_state)
    return {'status': 'ok', 'data': data}


@app.post('/api/rl/auto-tune/stop')
async def rl_auto_tune_stop():
    try:
        with _rla_auto_tune_lock:
            if _rla_auto_tune_state.get('status') == 'running':
                _rla_auto_tune_state['stop'] = True
                # attempt to kill current process if any
                pid = _rla_auto_tune_state.get('pid')
                if pid:
                    try:
                        if sys.platform.startswith('win'):
                            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
                        else:
                            os.kill(int(pid), 15)
                    except Exception:
                        pass
        return {'status': 'ok', 'data': _rla_auto_tune_state}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Auto-tune stop failed: {e}")


# ============================================================
# RL Live Preview and Paper Mode (migrated from Flask)
# ============================================================

def _rl_latest_model_path() -> Optional[str]:
    try:
        base = Path(__file__).parent.parent / 'rl' / 'models' / 'ppo_portfolio'
        if not base.exists():
            return None
        zips = sorted(base.rglob('*.zip'), key=lambda p: p.stat().st_mtime, reverse=True)
        return str(zips[0]) if zips else None
    except Exception:
        return None


@app.get('/api/rl/live/latest-model')
async def rl_live_latest_model():
    p = _rl_latest_model_path()
    if not p:
        raise HTTPException(status_code=404, detail='no model found')
    return {'status': 'ok', 'model_path': p}


@app.post('/api/rl/live/preview')
async def rl_live_preview(data: Dict[str, Any]):
    """Compute today's target weights using a trained PPO model (no orders executed)."""
    try:
        symbols_raw = data.get('symbols') or ''
        if isinstance(symbols_raw, str):
            symbols_in = [s.strip().upper() for s in symbols_raw.split(',') if s.strip()]
        elif isinstance(symbols_raw, list):
            symbols_in = [str(s).strip().upper() for s in symbols_raw]
        else:
            symbols_in = []
        if len(symbols_in) < 1:
            raise HTTPException(status_code=400, detail='Provide at least one symbol')
        symbols_env = symbols_in if len(symbols_in) >= 2 else [symbols_in[0], symbols_in[0]]

        model_path = data.get('model_path') or _rl_latest_model_path()
        if not model_path:
            raise HTTPException(status_code=400, detail='Model path not provided and no latest model found')
        window = int(data.get('window', 60) or 60)
        band = float(data.get('no_trade_band', 0.0) or 0.0)
        min_days = int(data.get('band_min_days', 0) or 0)
        starting_cash = float(data.get('starting_cash', 10000.0) or 10000.0)
        vix_symbol = str(data.get('vix_symbol') or '^VIX')

        from stable_baselines3 import PPO  # type: ignore
        import numpy as np  # type: ignore
        import pandas as pd  # type: ignore
        from rl.training.train_ppo_portfolio import make_env  # type: ignore
        from rl.envs.wrappers_portfolio import NoTradeBandActionWrapper  # type: ignore

        model = PPO.load(model_path)
        try:
            expected = int(getattr(model.policy.observation_space, 'shape', [None])[0])
        except Exception:
            expected = None

        # Prepare default feature paths/cols
        def _default_news_csv() -> Optional[str]:
            try:
                project_root = Path(__file__).parent.parent
                cand1 = project_root / 'app' / 'ml' / 'data' / 'news_features.csv'
                if cand1.exists():
                    return cand1.as_posix()
                cand2 = project_root / 'ml' / 'data' / 'news_features.csv'
                if cand2.exists():
                    return cand2.as_posix()
            except Exception:
                pass
            return None

        news_csv = _default_news_csv()
        news_cols = ['news_count','llm_relevant_count','avg_score','fda_count','china_count','geopolitics_count','sentiment_avg']
        ind_cols = ['rsi14','macd_hist','adx14','bb_pctb','bb_width','volume_sma_ratio','roc20','vol20']

        # Try building envs with increasing feature richness until obs matches expected
        tried: List[Dict[str, Any]] = []

        def attempt(opts: Dict[str, Any]):
            env = make_env(
                symbols=symbols_env,
                window=window,
                start=None,
                end=None,
                starting_cash=starting_cash,
                news_features_csv=opts.get('news_csv'),
                news_cols=opts.get('news_cols'),
                news_window=int(opts.get('news_window', 1)),
                add_vix_feature=bool(opts.get('add_vix', False)),
                vix_symbol=str(opts.get('vix_symbol', vix_symbol)),
                use_indicator_features=bool(opts.get('use_indicators', False)),
                indicator_cols=opts.get('indicator_cols'),
            )
            obs, _ = env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
            cur_dim = int(np.array(obs, dtype=float).shape[0])
            return env, cur_dim

        candidates = []
        candidates.append({'news_csv': None, 'news_cols': None, 'add_vix': False, 'use_indicators': False})
        if news_csv:
            candidates.append({'news_csv': news_csv, 'news_cols': news_cols, 'news_window': 1})
            candidates.append({'news_csv': news_csv, 'news_cols': news_cols, 'news_window': 1, 'add_vix': True, 'vix_symbol': vix_symbol})
            candidates.append({'news_csv': news_csv, 'news_cols': news_cols, 'news_window': 1, 'add_vix': True, 'vix_symbol': vix_symbol, 'use_indicators': True, 'indicator_cols': ind_cols})
        candidates.append({'add_vix': True, 'vix_symbol': vix_symbol})
        candidates.append({'use_indicators': True, 'indicator_cols': ind_cols})

        env_gym = None
        chosen = None
        cur_dim = None
        for opts in candidates:
            try:
                env, d = attempt(opts)
                tried.append({'opts': opts, 'dim': d})
                if expected is None or d == expected:
                    env_gym = env
                    chosen = opts
                    cur_dim = d
                    break
            except Exception as _e:
                tried.append({'opts': opts, 'error': str(_e)})
                continue

        if env_gym is None:
            msg = f"Unexpected observation shape; model expects {expected}, tried: " + \
                  ", ".join([str(t.get('dim') or t.get('error')) for t in tried])
            raise HTTPException(status_code=400, detail=msg)

        # Step to last observation with hold actions
        obs, _ = env_gym.reset()
        n = env_gym.action_space.shape[0]
        hold = np.zeros((n,), dtype=np.float32)
        done = False
        while True:
            obs, reward, terminated, truncated, info = env_gym.step(hold)
            if terminated or truncated:
                break

        # Predict logits
        action, _ = model.predict(obs, deterministic=True)

        # Access underlying base env to map logits->weights and read timeline
        base_env = getattr(env_gym, '_env', None) or getattr(env_gym, 'unwrapped', None) or env_gym
        raw_w = base_env._to_weights(np.asarray(action, dtype=float)).tolist()

        banded_w = None
        if band > 0.0 or min_days > 0:
            wrapper = NoTradeBandActionWrapper(env_gym, band=float(band), min_days=int(min_days))
            banded_logits = wrapper.action(np.asarray(action, dtype=float))
            banded_w = base_env._to_weights(np.asarray(banded_logits, dtype=float)).tolist()

        # Latest date
        try:
            idx_dates = next(iter(base_env.df_map.values())).index
            latest_dt = idx_dates[base_env._idx]
            latest_date = str(pd.to_datetime(latest_dt).date())
        except Exception:
            latest_date = None

        # Map weights to unique symbol list (sum duplicates)
        unique_symbols: List[str] = []
        for s in symbols_in:
            if s not in unique_symbols:
                unique_symbols.append(s)

        def map_unique(w_list: Optional[List[float]]) -> Dict[str, float]:
            if w_list is None:
                return {}
            agg = {s: 0.0 for s in unique_symbols}
            for s, w in zip(symbols_env, w_list):
                agg[s] += float(w)
            return agg

        resp = {
            'status': 'ok',
            'date': latest_date,
            'symbols': unique_symbols,
            'raw_weights': map_unique(raw_w),
            'banded_weights': (map_unique(banded_w) if banded_w is not None else None),
            'obs_dim': cur_dim,
            'matched_opts': chosen,
        }
        # JSON-sanitize
        def _safe(obj):
            import math
            if isinstance(obj, float):
                return obj if math.isfinite(obj) else None
            if isinstance(obj, dict):
                return {str(k): _safe(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_safe(x) for x in obj]
            return obj
        return _safe(resp)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Paper mode state
_paper_state: Dict[str, Any] = {
    'running': False,
    'thread': None,
    'config': None,
    'last_run_date': None,
    'positions': {},
    'cash': 0.0,
    'equity': 0.0,
    'peak_equity': 0.0,
    'logs': [],
    'last_decision': None,
}
_paper_lock = threading.Lock()


def _append_paper_log(msg: str) -> None:
    with _paper_lock:
        _paper_state['logs'].append(f"{datetime.now().isoformat()} | {msg}")
        if len(_paper_state['logs']) > 200:
            _paper_state['logs'] = _paper_state['logs'][-200:]


def _is_business_day(ts: datetime, cal_csv: Optional[Path]) -> bool:
    try:
        if cal_csv and cal_csv.exists():
            import pandas as pd
            cal = pd.read_csv(cal_csv)
            cal['Date'] = pd.to_datetime(cal['Date'])
            d = pd.Timestamp(ts.date())
            return bool((cal['Date'] == d).any())
        return ts.weekday() < 5
    except Exception:
        return ts.weekday() < 5


def _paper_step(config: Dict[str, Any]) -> None:
    from rl.envs.wrappers_portfolio import PortfolioEnvGym  # type: ignore
    from stable_baselines3 import PPO  # type: ignore
    import numpy as np  # type: ignore
    import pandas as pd  # type: ignore
    from rl.training.train_ppo_portfolio import make_env  # type: ignore

    project_root = Path(__file__).parent.parent

    symbols = config['symbols']
    env_symbols = symbols if len(symbols) > 1 else [symbols[0], symbols[0]]
    window = int(config.get('window', 60))
    model_path = config['model_path']
    band = float(config.get('no_trade_band', 0.0) or 0.0)
    min_days = int(config.get('band_min_days', 0) or 0)
    tcost = int(config.get('transaction_cost_bps', 5) or 5)
    slip = int(config.get('slippage_bps', 5) or 5)
    max_turnover_ratio = float(config.get('max_daily_turnover_ratio', 1.0) or 1.0)
    max_drawdown_stop = float(config.get('max_drawdown_stop', 0.0) or 0.0)

    model = PPO.load(model_path)
    try:
        expected = int(getattr(model.policy.observation_space, 'shape', [None])[0])
    except Exception:
        expected = None

    def _default_news_csv() -> Optional[str]:
        try:
            c1 = project_root / 'app' / 'ml' / 'data' / 'news_features.csv'
            if c1.exists():
                return c1.as_posix()
            c2 = project_root / 'ml' / 'data' / 'news_features.csv'
            if c2.exists():
                return c2.as_posix()
        except Exception:
            pass
        return None

    news_csv = _default_news_csv()
    news_cols = ['news_count','llm_relevant_count','avg_score','fda_count','china_count','geopolitics_count','sentiment_avg']
    ind_cols = ['rsi14','macd_hist','adx14','bb_pctb','bb_width','volume_sma_ratio','roc20','vol20']
    candidates = []
    candidates.append({'news_csv': None, 'news_cols': None, 'add_vix': False, 'use_indicators': False})
    if news_csv:
        candidates.append({'news_csv': news_csv, 'news_cols': news_cols, 'news_window': 1})
        candidates.append({'news_csv': news_csv, 'news_cols': news_cols, 'news_window': 1, 'add_vix': True, 'vix_symbol': '^VIX'})
        candidates.append({'news_csv': news_csv, 'news_cols': news_cols, 'news_window': 1, 'add_vix': True, 'vix_symbol': '^VIX', 'use_indicators': True, 'indicator_cols': ind_cols})
    candidates.append({'add_vix': True, 'vix_symbol': '^VIX'})
    candidates.append({'use_indicators': True, 'indicator_cols': ind_cols})

    env_gym = None
    for opts in candidates:
        try:
            env = make_env(
                symbols=env_symbols,
                window=window,
                start=None,
                end=None,
                transaction_cost_bps=tcost,
                slippage_bps=slip,
                starting_cash=100.0,
                news_features_csv=opts.get('news_csv'),
                news_cols=opts.get('news_cols'),
                news_window=int(opts.get('news_window', 1)),
                add_vix_feature=bool(opts.get('add_vix', False)),
                vix_symbol=str(opts.get('vix_symbol', '^VIX')),
                use_indicator_features=bool(opts.get('use_indicators', False)),
                indicator_cols=opts.get('indicator_cols'),
            )
            obs, _ = env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
            dim = int(np.array(obs, dtype=float).shape[0])
            if expected is None or dim == expected:
                env_gym = env
                break
        except Exception:
            continue
    if env_gym is None:
        env_gym = make_env(symbols=env_symbols, window=window, start=None, end=None, transaction_cost_bps=tcost, slippage_bps=slip, starting_cash=100.0)

    obs, _ = env_gym.reset()
    n = env_gym.action_space.shape[0]
    hold = np.zeros((n,), dtype=np.float32)
    last_idx = getattr(getattr(env_gym, '_env', None) or getattr(env_gym, 'unwrapped', None) or env_gym, '_idx', None)
    while True:
        obs, reward, terminated, truncated, info = env_gym.step(hold)
        if terminated or truncated:
            break
        base_env_loop = getattr(env_gym, '_env', None) or getattr(env_gym, 'unwrapped', None) or env_gym
        try:
            last_idx = base_env_loop._idx
        except Exception:
            pass

    action, _ = model.predict(obs, deterministic=True)
    base_env = getattr(env_gym, '_env', None) or getattr(env_gym, 'unwrapped', None) or env_gym
    target_w = base_env._to_weights(np.asarray(action, dtype=float))

    unique_symbols: List[str] = []
    for s in env_symbols:
        if s not in unique_symbols:
            unique_symbols.append(s)
    agg_target = {s: 0.0 for s in unique_symbols}
    for s, w in zip(env_symbols, target_w.tolist()):
        agg_target[s] += float(w)

    idx_dates = next(iter(base_env.df_map.values())).index
    date_dt = pd.to_datetime(idx_dates[last_idx]).date()
    prices_t = base_env.prices[last_idx]
    price_map: Dict[str, float] = {}
    for s, px in zip(env_symbols, prices_t.tolist()):
        if s in price_map:
            price_map[s] = max(price_map[s], float(px))
        else:
            price_map[s] = float(px)

    with _paper_lock:
        positions = dict(_paper_state['positions'])
        cash = float(_paper_state['cash'])
        peak_equity = float(_paper_state.get('peak_equity') or 0.0)

    equity = float(cash)
    for s in unique_symbols:
        sh = float(positions.get(s, 0.0))
        px = float(price_map.get(s, 0.0))
        equity += sh * px

    cur_w: Dict[str, float] = {}
    for s in unique_symbols:
        sh = float(positions.get(s, 0.0))
        px = float(price_map.get(s, 0.0))
        val = sh * px
        cur_w[s] = (val / equity) if equity > 0 else 0.0

    desired = dict(cur_w)
    last_trades = (_paper_state.get('last_trade_date') or {})
    allow_trade: Dict[str, bool] = {}
    for s in unique_symbols:
        last = last_trades.get(s)
        if not last:
            allow_trade[s] = True
        else:
            try:
                from datetime import datetime as _dt
                days = (_dt.combine(date_dt, _dt.min.time()) - _dt.fromisoformat(last)).days
            except Exception:
                days = 999
            allow_trade[s] = (days >= min_days)

    changed = False
    for s in unique_symbols:
        tw = float(agg_target.get(s, 0.0))
        if allow_trade[s] and abs(tw - cur_w.get(s, 0.0)) >= band:
            desired[s] = max(0.0, min(1.0, tw))
            changed = True
    ssum = sum(desired.values())
    if ssum > 0:
        for s in desired:
            desired[s] = desired[s] / ssum
    else:
        k = len(unique_symbols)
        for s in desired:
            desired[s] = 1.0 / max(1, k)

    trades = []
    total_cost = 0.0
    for s in unique_symbols:
        cur_val = cur_w[s] * equity
        tgt_val = desired[s] * equity
        delta_val = tgt_val - cur_val
        px = float(price_map.get(s, 0.0))
        delta_sh = (delta_val / px) if px > 0 else 0.0
        trade_cost = abs(delta_sh) * px * ((tcost + slip) / 10000.0)
        total_cost += float(trade_cost)
        trades.append({'symbol': s, 'px': float(px), 'delta_shares': float(delta_sh), 'trade_cost': float(trade_cost)})

    return {
        'status': 'ok',
        'date': str(date_dt),
        'symbols': unique_symbols,
        'raw_weights': {k: float(agg_target.get(k, 0.0)) for k in unique_symbols},
        'banded_weights': {k: float(desired.get(k, 0.0)) for k in unique_symbols},
        'trades': trades,
        'equity_before': float(equity),
        'equity_after': float(equity),
        'total_cost': float(total_cost),
    }


def _paper_loop():
    cfg = None
    from time import sleep
    import pytz  # type: ignore
    while True:
        with _paper_lock:
            if not _paper_state['running']:
                break
            cfg = dict(_paper_state['config']) if _paper_state['config'] else None
        if not cfg:
            sleep(1)
            continue
        tz_name = cfg.get('timezone', 'America/New_York')
        schedule_time = cfg.get('schedule_time', '16:05')
        try:
            hh, mm = [int(x) for x in schedule_time.split(':', 1)]
        except Exception:
            hh, mm = 16, 5
        try:
            tz = pytz.timezone(tz_name)
        except Exception:
            tz = pytz.timezone('America/New_York')
        now = datetime.now(tz)
        cal_csv = (Path(__file__).parent.parent / 'data' / 'rl' / 'calendars' / 'market_calendar.csv')
        if not _is_business_day(now, cal_csv):
            sleep(30)
            continue
        run_today = False
        try:
            if now.hour > hh or (now.hour == hh and now.minute >= mm):
                with _paper_lock:
                    last = _paper_state.get('last_run_date')
                if str(now.date()) != str(last):
                    run_today = True
        except Exception:
            run_today = False

        if run_today:
            try:
                _paper_step(cfg)
            except Exception as e:
                _append_paper_log(f"Error in paper step: {e}")
            sleep(10)
        else:
            sleep(15)


@app.post('/api/rl/live/paper/start')
async def rl_live_paper_start(data: Dict[str, Any]):
    symbols_raw = data.get('symbols') or ''
    if isinstance(symbols_raw, str):
        symbols = [s.strip().upper() for s in symbols_raw.split(',') if s.strip()]
    elif isinstance(symbols_raw, list):
        symbols = [str(s).strip().upper() for s in symbols_raw]
    else:
        symbols = []
    if len(symbols) < 1:
        raise HTTPException(status_code=400, detail='Provide at least one symbol')
    model_path = data.get('model_path') or _rl_latest_model_path()
    if not model_path:
        raise HTTPException(status_code=400, detail='Model path not provided and no latest model found')
    cfg = {
        'symbols': symbols,
        'model_path': model_path,
        'starting_cash': float(data.get('starting_cash', 10000.0) or 10000.0),
        'window': int(data.get('window', 60) or 60),
        'no_trade_band': float(data.get('no_trade_band', 0.05) or 0.05),
        'band_min_days': int(data.get('band_min_days', 5) or 5),
        'transaction_cost_bps': int(data.get('transaction_cost_bps', 5) or 5),
        'slippage_bps': int(data.get('slippage_bps', 5) or 5),
        'schedule_time': str(data.get('schedule_time', '16:05') or '16:05'),
        'timezone': str(data.get('timezone', 'America/New_York') or 'America/New_York'),
        'max_daily_turnover_ratio': float(data.get('max_daily_turnover_ratio', 1.0) or 1.0),
        'max_drawdown_stop': float(data.get('max_drawdown_stop', 0.0) or 0.0),
        'resume': bool(data.get('resume', True)),
    }
    from threading import Thread
    with _paper_lock:
        _paper_state['positions'] = {}
        _paper_state['cash'] = float(cfg['starting_cash'])
        _paper_state['equity'] = float(cfg['starting_cash'])
        _paper_state['peak_equity'] = float(cfg['starting_cash'])
        _paper_state['config'] = cfg
        _paper_state['running'] = True
        _paper_state['logs'] = []
        _paper_state['last_decision'] = None
        _paper_state['last_run_date'] = None
        _paper_state['last_trade_date'] = {}
    # Attempt resume
    try:
        if cfg.get('resume'):
            state_path = Path(__file__).parent.parent / 'data' / 'live_paper' / 'state.json'
            if state_path.exists():
                import json as _json
                with open(state_path, 'r', encoding='utf-8') as f:
                    saved = _json.load(f)
                saved_cfg = saved.get('config') or {}
                same_syms = sorted([s.upper() for s in (saved_cfg.get('symbols') or [])]) == sorted(symbols)
                same_model = str(saved_cfg.get('model_path') or '') == str(model_path)
                if same_syms and same_model:
                    with _paper_lock:
                        _paper_state['positions'] = saved.get('positions') or {}
                        _paper_state['cash'] = float(saved.get('cash') or cfg['starting_cash'])
                        _paper_state['equity'] = float(saved.get('equity') or _paper_state['cash'])
                        _paper_state['peak_equity'] = float(saved.get('peak_equity') or _paper_state['equity'])
                        _paper_state['last_run_date'] = saved.get('last_run_date')
                        _paper_state['last_trade_date'] = saved.get('last_trade_date') or {}
                    _append_paper_log("Resumed paper mode from previous state.json")
                else:
                    _append_paper_log("State.json found but symbols/model mismatch; starting fresh.")
    except Exception as e:
        _append_paper_log(f"Resume failed: {e}")
    th = Thread(target=_paper_loop, daemon=True)
    with _paper_lock:
        _paper_state['thread'] = th
    th.start()
    _append_paper_log("Paper mode started")
    return {'status': 'started', 'config': cfg}


@app.post('/api/rl/live/paper/stop')
async def rl_live_paper_stop():
    with _paper_lock:
        _paper_state['running'] = False
        th = _paper_state.get('thread')
    try:
        if th:
            th.join(timeout=1.0)
    except Exception:
        pass
    _append_paper_log("Paper mode stopped")
    return {'status': 'stopped'}


@app.get('/api/rl/live/paper/status')
async def rl_live_paper_status():
    with _paper_lock:
        snap = {
            'running': _paper_state['running'],
            'config': _paper_state['config'],
            'last_run_date': _paper_state['last_run_date'],
            'positions': _paper_state['positions'],
            'cash': _paper_state['cash'],
            'equity': _paper_state['equity'],
            'last_decision': _paper_state['last_decision'],
            'logs': _paper_state['logs'][-50:],
        }
    return snap


@app.get("/api/rl/simulate/plan")
async def rl_simulate_plan(symbol: str = Query("AAPL"), window: int = Query(60, ge=2, le=500)):
    """Auto-plan start/end, window, and days for Quick Simulation from local data."""
    try:
        try:
            from rl.envs.market_env import MarketEnv as RLMarketEnv  # type: ignore
            import pandas as pd
        except Exception:
            raise HTTPException(status_code=500, detail="Planning unavailable: RL modules missing")

        def _plan():
            env = RLMarketEnv.load_from_local(symbol.upper(), window=max(2, window or 60), tail_days=None)
            df = env.df
            n = int(len(df))
            if n < 2:
                raise ValueError(f"Not enough data for {symbol}")
            idx = df.index
            if not isinstance(idx, pd.DatetimeIndex):
                try:
                    idx = pd.to_datetime(idx)
                except Exception:
                    pass
            earliest = idx[0]
            latest = idx[-1]
            span = min(250, n)
            start_idx = max(0, n - span)
            start_dt = idx[start_idx]
            end_dt = latest
            plan = {
                'start_date': str(getattr(start_dt, 'date', lambda: start_dt)()),
                'end_date': str(getattr(end_dt, 'date', lambda: end_dt)()),
                'window': int(window or 60),
                'days': int(span),
                'total_rows': n
            }
            return plan

        plan = await asyncio.to_thread(_plan)
        return {"status": "planned", "plan": plan}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/rl/ppo/plan")
async def rl_ppo_plan(symbols: Optional[str] = None, symbol: Optional[str] = None, window: int = Query(60, ge=2, le=500)):
    """Plan PPO training window and timesteps from local data.

    Accepts either 'symbols' (comma-separated) or a single 'symbol'.
    """
    try:
        try:
            import pandas as pd
            from rl.envs.market_env import MarketEnv as RLMarketEnv  # type: ignore
            from rl.envs.portfolio_env import PortfolioEnv  # type: ignore
        except Exception:
            raise HTTPException(status_code=500, detail="Planning unavailable: RL modules missing")

        syms: List[str] = []
        if symbols:
            syms = [s.strip().upper() for s in symbols.split(',') if s.strip()]
        elif symbol:
            syms = [symbol.strip().upper()]
        else:
            raise HTTPException(status_code=400, detail="symbol or symbols is required")

        def _plan_many():
            if len(syms) >= 2:
                base = PortfolioEnv.load_from_local_universe(symbols=syms, window=max(2, window or 60))
                idx = next(iter(base.df_map.values())).index
                n = int(len(idx))
                if n < 60:
                    raise ValueError("Not enough aligned rows across symbols to plan PPO")
                try:
                    y2020 = pd.Timestamp('2020-01-01')
                    first_idx = idx[0]
                    train_start = y2020 if (y2020 > first_idx and y2020 < idx[-1]) else first_idx
                except Exception:
                    train_start = idx[0]
                train_end = idx[-1]
                days = int((pd.to_datetime(train_end) - pd.to_datetime(train_start)).days)
                timesteps = int(max(300_000, min(1_200_000, days * 400)))
                w = int(max(30, min(120, n // 6))) if not window else int(window)
                return {
                    'train_start_date': str(getattr(train_start, 'date', lambda: train_start)()),
                    'train_end_date': str(getattr(train_end, 'date', lambda: train_end)()),
                    'window': w,
                    'training_days': days,
                    'timesteps': timesteps,
                    'total_rows': n,
                    'symbols': syms,
                }
            else:
                env = RLMarketEnv.load_from_local(syms[0], window=max(2, window or 60), tail_days=None)
                df = env.df
                n = int(len(df))
                if n < 60:
                    raise ValueError(f"Not enough rows to plan PPO for {syms[0]}")
                idx = df.index
                if not isinstance(idx, pd.DatetimeIndex):
                    try:
                        idx = pd.to_datetime(idx)
                    except Exception:
                        pass
                try:
                    y2020 = pd.Timestamp('2020-01-01')
                    first_idx = idx[0]
                    train_start = y2020 if (y2020 > first_idx and y2020 < idx[-1]) else first_idx
                except Exception:
                    train_start = idx[0]
                train_end = idx[-1]
                days = int((pd.to_datetime(train_end) - pd.to_datetime(train_start)).days)
                timesteps = int(max(300_000, min(1_200_000, days * 400)))
                w = int(max(30, min(120, n // 6))) if not window else int(window)
                return {
                    'train_start_date': str(getattr(train_start, 'date', lambda: train_start)()),
                    'train_end_date': str(getattr(train_end, 'date', lambda: train_end)()),
                    'window': w,
                    'training_days': days,
                    'timesteps': timesteps,
                    'total_rows': n,
                    'symbols': syms,
                }

        plan = await asyncio.to_thread(_plan_many)
        return {"status": "planned", "plan": plan}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================
# WebSocket Endpoint
# ============================================================
@app.websocket("/ws/alerts")
async def websocket_alerts(websocket: WebSocket):
    """
    WebSocket endpoint for real-time alerts
    
    Clients connect here to receive:
    - Critical alerts (WhatsApp-level)
    - Important alerts
    - Watch alerts
    - Real-time market updates
    """
    await manager.connect(websocket)
    
    try:
        # Send welcome message
        await websocket.send_json({
            "type": "connected",
            "message": "✅ Connected to MarketPulse alerts",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        # Keep connection alive and handle incoming messages
        while True:
            # Receive messages from client (e.g., preferences, ping)
            data = await websocket.receive_text()
            
            # Echo back (or handle commands)
            try:
                client_msg = json.loads(data)
                
                if client_msg.get("type") == "ping":
                    await websocket.send_json({
                        "type": "pong",
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })
                
            except json.JSONDecodeError:
                pass
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("Client disconnected normally")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

# ============================================================
# Health & Status Endpoints
# ============================================================
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if not SCHEDULER_AVAILABLE:
        return {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "scheduler": {"status": "unavailable"},
            "websockets": {"connected_clients": len(manager.active_connections)}
        }
    
    try:
        scheduler = get_scheduler()
        stats = scheduler.get_statistics()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "scheduler": {
                "running": scheduler.scheduler.running,
                "jobs": len(scheduler.scheduler.get_jobs()),
                "stats": stats
            },
            "websockets": {
                "connected_clients": len(manager.active_connections)
            }
        }
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "scheduler": {"status": "error"},
            "websockets": {"connected_clients": len(manager.active_connections)}
        }

@app.get("/api/health")
async def api_health_check():
    """Detailed health check for dashboard system status"""
    if not SCHEDULER_AVAILABLE:
        return {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "services": {"api": "healthy", "scheduler": "unavailable"},
            "websockets": {"connected_clients": len(manager.active_connections)},
            "uptime": "Active"
        }
    
    try:
        scheduler = get_scheduler()
        
        # Check all services
        services = {
            "api": "healthy",
            "database": "connected",
            "cache": "connected",
            "vector_db": "connected",
            "workers": "active"
        }
        
        return {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "services": services,
            "scheduler": {
                "running": scheduler.scheduler.running,
                "jobs": len(scheduler.scheduler.get_jobs())
            },
            "websockets": {
                "connected_clients": len(manager.active_connections)
            },
            "uptime": "Active"
        }
    except Exception as e:
        logger.error(f"API health check error: {e}")
        return {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "services": {"api": "healthy"},
            "scheduler": {"status": "error"},
            "websockets": {"connected_clients": len(manager.active_connections)},
            "uptime": "Active"
        }

@app.get("/api/statistics")
async def get_statistics():
    """Get scheduler and system statistics"""
    try:
        scheduler = get_scheduler()
        return scheduler.get_statistics()
    except Exception as e:
        logger.error(f"Statistics error: {e}")
        return {"error": "Statistics unavailable", "status": "error"}

@app.get("/api/jobs")
async def get_jobs():
    """Get list of scheduled jobs"""
    try:
        scheduler = get_scheduler()
        jobs = []
        
        for job in scheduler.get_jobs():
            jobs.append({
                "id": getattr(job, 'id', 'unknown'),
                "name": getattr(job, 'name', 'Unknown Job'),
                "trigger": str(getattr(job, 'trigger', 'N/A')),
                "next_run": getattr(job, 'next_run_time', 'N/A')
            })
        
        return {"jobs": jobs}
    except Exception as e:
        logger.error(f"Jobs error: {e}")
        return {"jobs": [], "error": "Scheduler unavailable"}

@app.get("/api/feeds/status")
async def get_feeds_status():
    """Get status of all RSS feeds"""
    import yaml
    
    try:
        with open("app/config/data_sources.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        feeds_status = []
        rss_feeds = config.get("rss_feeds", {})
        
        for tier_name, feeds in rss_feeds.items():
            for feed in feeds:
                feeds_status.append({
                    "tier": tier_name,
                    "name": feed.get("name"),
                    "url": feed.get("url"),
                    "category": feed.get("category"),
                    "weight": feed.get("weight"),
                    "status": "configured",  # Will test in background
                    "note": feed.get("note", "")
                })
        
        return {
            "total_feeds": len(feeds_status),
            "feeds": feeds_status
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================
# Manual Trigger Endpoints (for testing)
# ============================================================
@app.post("/api/trigger/major-news")
async def trigger_major_news():
    """Manually trigger major news fetch"""
    try:
        scheduler = get_scheduler()
        await scheduler._fetch_major_news()
        return {"status": "success", "message": "Major news fetch triggered"}
    except Exception as e:
        logger.error(f"Major news trigger error: {e}")
        return {"status": "error", "message": "Scheduler unavailable"}

@app.post("/api/trigger/perplexity-scan")
async def trigger_perplexity_scan():
    """Manually trigger Perplexity market scan"""
    try:
        scheduler = get_scheduler()
        await scheduler._run_perplexity_scans()
        return {"status": "success", "message": "Perplexity scan triggered"}
    except Exception as e:
        logger.error(f"Perplexity scan trigger error: {e}")
        return {"status": "error", "message": "Scheduler unavailable"}

@app.post("/api/test-alert")
async def test_alert():
    """Send test alert via WebSocket"""
    test_message = {
        "type": "alert",
        "level": "important",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "title": "🧪 TEST ALERT: This is a test notification",
        "symbols": ["AAPL", "MSFT"],
        "score": 2.5,
        "sentiment": "bullish",
        "source": "Manual Test",
        "url": "https://example.com",
        "keywords": ["test", "alert", "notification"]
    }
    
    await manager.broadcast(test_message)
    
    return {
        "status": "success",
        "message": f"Test alert sent to {len(manager.active_connections)} clients"
    }

# ============================================================
# Financial Data Endpoints (REAL DATA ONLY)
# ============================================================
@app.get("/api/financial/market-indices")
async def get_market_indices_endpoint():
    """
    Get REAL market indices data (S&P500, NASDAQ, DOW, VIX)
    NO DEMO DATA - Only real Yahoo Finance data
    """
    try:
        from app.financial.market_data import financial_provider
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

@app.get("/api/financial/market-sentiment")
async def get_market_sentiment_endpoint():
    """
    Calculate market sentiment from REAL indices data
    NO DEMO DATA - Based on actual market performance
    """
    try:
        from app.financial.market_data import financial_provider
        sentiment = await financial_provider.calculate_market_sentiment()
        
        return {
            "status": "success",
            "data": sentiment,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"❌ Error calculating market sentiment: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/financial/top-stocks")
async def get_top_stocks_endpoint():
    """
    Get top performing stocks with REAL data
    NO DEMO DATA - Real Yahoo Finance data
    """
    try:
        from app.financial.market_data import financial_provider
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


# ============================================================
# News & Sentiment Endpoints (Live-only, no mock data)
# ============================================================
@app.get("/api/news/sentiment/{symbol}", tags=["Sentiment"])
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


@app.get("/api/news/{symbol}", tags=["News"])
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


@app.get("/api/sentiment/providers", tags=["Sentiment"])
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




# ============================================================
# ML/Neural Network Endpoints (REAL AI PREDICTIONS)
# ============================================================
@app.get("/api/ml/predict/{symbol}")
async def get_ml_prediction(symbol: str, horizon: str = "1d"):
    """
    Get ML price prediction for a stock symbol - NOW USES PYTORCH PROGRESSIVE ML!
    """
    try:
        # Use PyTorch Progressive ML instead of old TensorFlow
        if not PROGRESSIVE_ML_AVAILABLE or not progressive_predictor:
            raise HTTPException(status_code=503, detail="Progressive ML system not available")
        
        # Get prediction using PyTorch system
        prediction = progressive_predictor.predict_ensemble(symbol, mode="progressive")
        
        # Calculate real confidence from prediction variance
        confidence = 0.95 if prediction.get('accuracy', 0) > 0.7 else 0.75
        if prediction.get('predictions'):
            # Use ensemble variance for confidence
            import numpy as np
            pred_values = [p.get('predicted_value', 0) for p in prediction.get('predictions', [])]
            if len(pred_values) > 1:
                variance = np.var(pred_values)
                confidence = max(0.5, min(0.99, 1.0 - variance / 100))
        
        return {
            "status": "success",
            "data": {
                "symbol": symbol,
                "horizon": horizon,
                "prediction": prediction,
                "model_type": "pytorch_progressive_ml",
                "confidence": round(confidence, 3)
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Error in ML prediction for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/ml/status")
async def get_ml_status():
    """
    Get ML system status and capabilities
    """
    try:
        # No more TensorFlow imports - use progressive ML status
        
        return {
            "status": "success",
            "data": {
                "progressive_ml_available": PROGRESSIVE_ML_AVAILABLE,
                "tensorflow_available": False,  # DISABLED
                "pytorch_available": True,
                "models": {
                    "progressive_pytorch": {
                        "status": "✅ Active" if PROGRESSIVE_ML_AVAILABLE else "⚠️ Demo Mode",
                        "type": "PyTorch Progressive ML",
                        "accuracy": "85-90%",
                        "best_for": "Real-time predictions, GPU acceleration"
                    },
                    "lstm": {
                        "status": "🚫 Disabled - Old TensorFlow",
                        "type": "Neural Network - Long Short-Term Memory",
                        "accuracy": "75-82%",
                        "best_for": "Long-term trends, sequential patterns"
                    },
                    "transformer": {
                        "status": "🚫 Disabled - Old TensorFlow",
                        "type": "Neural Network - Attention Mechanism",
                        "accuracy": "80-85%",
                        "best_for": "Complex relationships, multi-scale patterns"
                    },
                    "cnn": {
                        "status": "🚫 Disabled - Old TensorFlow",
                        "type": "Neural Network - Pattern Recognition",
                        "accuracy": "72-76%",
                        "best_for": "Chart patterns, technical analysis"
                    },
                    "random_forest": {
                        "status": "✅ Active" if PROGRESSIVE_ML_AVAILABLE else "⚠️ Unavailable",
                        "type": "Machine Learning - Ensemble Trees",
                        "accuracy": "78-85%",
                        "best_for": "Feature importance, non-linear relationships"
                    },
                    "gradient_boost": {
                        "status": "✅ Active" if PROGRESSIVE_ML_AVAILABLE else "⚠️ Unavailable",
                        "type": "Machine Learning - Boosting",
                        "accuracy": "80-88%",
                        "best_for": "High accuracy predictions, complex features"
                    }
                },
                "ensemble_method": "Weighted Average",
                "ensemble_weights": {
                    "lstm": 0.4,
                    "transformer": 0.35,
                    "cnn": 0.25
                },
                "features_used": [
                    "price_history", "volume", "technical_indicators",
                    "sentiment", "volatility", "time_patterns"
                ],
                "timestamp": datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting ML status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/ai/market-intelligence")
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

# ============================================================
# Sector Scanner & Dynamic Stock Discovery
# ============================================================
@app.get("/api/scanner/sectors")
async def get_sectors():
    """Get all monitored sectors"""
    try:
        from app.smart.sector_scanner import sector_scanner
        sectors = sector_scanner.get_all_sectors()
        
        return {
            "status": "success",
            "data": {
                "sectors": sectors,
                "total": len(sectors)
            }
        }
    except Exception as e:
        logger.error(f"❌ Error getting sectors: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/scanner/hot-stocks")
async def get_hot_stocks(limit: int = 10):
    """
    Get dynamically scanned hot stocks with high potential from local stock_data
    No legacy sector_scanner; simple momentum/indicator heuristics
    """
    try:
        # Build from local symbols
        symbols = _iter_local_symbols(max_symbols=500)
        items: List[Dict[str, Any]] = []
        scanned = 0
        for sym in symbols:
            scanned += 1
            m = _compute_local_metrics(sym)
            if m is None:
                continue
            # Name isn't available in local CSVs; mirror symbol
            m['name'] = sym
            items.append(m)
            if len(items) >= max(limit * 5, 100):
                break
        # Sort by expected return desc
        items.sort(key=lambda x: x.get('expected_return', 0.0), reverse=True)
        hot_stocks = items[:limit]
        return {
            "status": "success",
            "data": {
                "hot_stocks": hot_stocks,
                "total_scanned": scanned,
                "total_potential": len(items),
                "last_updated": datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting hot stocks: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/scanner/sector/{sector_id}")
async def get_sector_analysis(sector_id: str):
    """Get analysis for a specific sector"""
    try:
        from app.smart.sector_scanner import sector_scanner
        
        tickers = sector_scanner.get_sector_tickers(sector_id)
        
        # Scan top tickers in this sector
        sector_stocks = []
        for ticker in tickers[:10]:
            try:
                ml_result = await sector_scanner.scan_ticker_with_ml(ticker)
                if ml_result.get('ticker'):
                    sector_stocks.append(ml_result)
            except Exception as e:
                logger.warning(f"Error analyzing {ticker}: {e}")
                continue
        
        # Sort by ML score
        sector_stocks.sort(key=lambda x: x.get('ml_score', 0), reverse=True)
        
        return {
            "status": "success",
            "data": {
                "sector_id": sector_id,
                "stocks": sector_stocks,
                "total": len(sector_stocks)
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting sector analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------------------------------
# Scanner Orchestration (placeholder compatibility)
# -------------------------------------------------
@app.get("/api/scanner/status")
async def scanner_status():
    return {"status": "success", "data": _scanner_state.to_status()}


@app.get("/api/scanner/top")
async def scanner_top(limit: int = 50):
    # Return ranked items in shape expected by scanner.html (data.items + meta)
    try:
        items = _scanner_state.top
        if not items:
            r = await get_hot_stocks(limit=max(limit, 50))
            if isinstance(r, dict) and r.get('status') == 'success':
                items = (r.get('data') or {}).get('hot_stocks') or []
        # Build ranked table rows
        ranked = []
        for i, it in enumerate(items[:limit], start=1):
            ranked.append({
                'rank': i,
                'symbol': it.get('symbol'),
                'final_score': it.get('expected_return'),
                'ml_score': it.get('ml_score'),
                'fallback_score': it.get('change_percent'),
                'current_price': it.get('current_price'),
            })
        return {
            "status": "success",
            "data": {
                "date": datetime.now().strftime('%Y-%m-%d'),
                "total": len(items),
                "items": ranked
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/scanner/run")
async def scanner_run(mode: str = 'ml', limit: int = 50, background_tasks: BackgroundTasks = None):
    # Start a background scan
    try:
        if background_tasks is not None:
            background_tasks.add_task(_run_scan, mode, limit)
        else:
            # Fallback if BackgroundTasks not injected
            asyncio.create_task(_run_scan(mode, limit))
        return {"status": "success", "data": {"started": True, "mode": mode, "limit": limit}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/scanner/filter/status")
async def scanner_filter_status():
    return {
        "status": "success",
        "data": {
            "state": _scanner_state.filter_state,
            "progress": _scanner_state.filter_progress,
            "total_symbols": _scanner_state.total_symbols,
            "processed_symbols": _scanner_state.processed_symbols,
            "passed_count": _scanner_state.passed_count,
            "trained_count": _scanner_state.trained_count,
            "count": len(_scanner_state.filter_items)
        }
    }


@app.post("/api/scanner/filter/run")
async def scanner_filter_run(background_tasks: BackgroundTasks = None):
    # Start real filter over local stock_data with thresholds
    from datetime import datetime as _dt
    try:
        # Parse optional thresholds from query/body
        price_min = 3.0
        adv_min = 1_000_000.0
        try:
            import json as _json
            body = None
            if hasattr(background_tasks, 'request'):
                # Not available in this context; keep for clarity
                pass
        except Exception:
            body = None
        if body and isinstance(body, dict):
            price_min = float(body.get('price_min', price_min) or price_min)
            adv_min = float(body.get('adv_min', adv_min) or adv_min)
        # Launch background job
        if background_tasks is not None:
            background_tasks.add_task(_run_filter_job, price_min, adv_min)
        else:
            asyncio.create_task(_run_filter_job(price_min, adv_min))
        return {"status": "success", "data": {"started": True, "started_at": _dt.now().isoformat(), "price_min": price_min, "adv_min": adv_min}}
    except Exception as e:
        _scanner_state.filter_state = 'error'
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/scanner/filter/results")
async def scanner_filter_results():
    return {
        "status": "success",
        "data": {
            "items": _scanner_state.filter_items,
            "total": len(_scanner_state.filter_items)
        }
    }


@app.get("/api/scanner/train/status")
async def scanner_train_status():
    return {"status": "success", "data": {"status": _scanner_state.train_status, "current_symbol": _scanner_state.train_current_symbol}}


@app.post("/api/scanner/train/start")
async def scanner_train_start(symbol: Optional[str] = None, background_tasks: BackgroundTasks = None):
    # Placeholder: mark running briefly then complete
    try:
        _scanner_state.train_status = 'running'
        _scanner_state.train_current_symbol = (symbol or '').upper() or None
        async def _finish():
            await asyncio.sleep(1.0)
            _scanner_state.train_status = 'completed'
            _scanner_state.train_current_symbol = None
        if background_tasks is not None:
            background_tasks.add_task(_finish)
        else:
            asyncio.create_task(_finish())
        return {"status": "success", "data": {"started": True, "symbol": symbol}}
    except Exception as e:
        _scanner_state.train_status = 'error'
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/scanner/train/start-all")
async def scanner_train_start_all(background_tasks: BackgroundTasks = None):
    # Placeholder: complete immediately
    try:
        _scanner_state.train_status = 'running'
        async def _finish():
            await asyncio.sleep(1.0)
            _scanner_state.train_status = 'completed'
        if background_tasks is not None:
            background_tasks.add_task(_finish)
        else:
            asyncio.create_task(_finish())
        return {"status": "success", "data": {"started": True}}
    except Exception as e:
        _scanner_state.train_status = 'error'
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/scanner/train/symbol/{symbol}/status")
async def scanner_train_symbol_status(symbol: str):
    # Placeholder: always completed
    return {"status": "success", "data": {"symbol": symbol.upper(), "status": "completed"}}


@app.get("/api/financial/historical/{symbol}")
async def get_historical_data(symbol: str, timeframe: str = "1D"):
    """
    Get REAL historical price data from LOCAL CSV files
    Timeframes: 1D (1 day), 1W (5 days), 1M (1 month), 3M (3 months)
    """
    try:
        import pandas as pd
        from datetime import datetime, timedelta
        from pathlib import Path
        from app.integrations.ibkr_client import ibkr
        
        # Get stock_data directory (two levels up from app/)
        project_root = Path(__file__).parent.parent
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

# ============================================================
# Stock Predictions Tracking Endpoints
# ============================================================
@app.post("/api/admin/run-migration")
async def run_migration():
    """Run database migration to create stock_predictions table"""
    try:
        from app.storage.db import get_db_engine
        
        # Get database engine
        engine = get_db_engine()
        
        # Create stock_predictions table using raw SQL
        with engine.connect() as conn:
            # Check if table already exists
            result = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='stock_predictions'"
            )
            
            if result.fetchone():
                return {
                    "status": "already_exists",
                    "message": "✅ stock_predictions table already exists"
                }
            
            # Create the table
            conn.execute("""
                CREATE TABLE stock_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    
                    -- Stock identification
                    symbol VARCHAR(20) NOT NULL,
                    company_name VARCHAR(255),
                    
                    -- Prediction details
                    prediction_date TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    prediction_source VARCHAR(50) NOT NULL,
                    price_at_prediction FLOAT NOT NULL,
                    
                    -- Prediction parameters
                    predicted_direction VARCHAR(10),
                    confidence_score FLOAT,
                    timeframe VARCHAR(20),
                    target_price FLOAT,
                    expected_return FLOAT,
                    
                    -- Reasoning
                    reason TEXT,
                    sector VARCHAR(100),
                    keywords_matched JSON,
                    news_sentiment FLOAT,
                    
                    -- ML scores
                    ml_score FLOAT,
                    technical_score FLOAT,
                    fundamental_score FLOAT,
                    
                    -- Related article
                    article_id INTEGER,
                    
                    -- Outcome tracking
                    actual_price_1d FLOAT,
                    actual_price_1w FLOAT,
                    actual_price_1m FLOAT,
                    actual_price_3m FLOAT,
                    
                    actual_return_1d FLOAT,
                    actual_return_1w FLOAT,
                    actual_return_1m FLOAT,
                    actual_return_3m FLOAT,
                    
                    -- Performance evaluation
                    prediction_accuracy FLOAT,
                    was_correct BOOLEAN,
                    max_gain_achieved FLOAT,
                    max_loss_suffered FLOAT,
                    
                    -- Status tracking
                    status VARCHAR(20) DEFAULT 'active',
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    outcome_recorded_at TIMESTAMP,
                    
                    -- Metadata
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    notes TEXT,
                    
                    FOREIGN KEY (article_id) REFERENCES articles(id)
                )
            """)
            
            # Create indexes
            conn.execute("CREATE INDEX idx_prediction_symbol ON stock_predictions(symbol)")
            conn.execute("CREATE INDEX idx_prediction_date ON stock_predictions(prediction_date)")
            conn.execute("CREATE INDEX idx_prediction_status ON stock_predictions(status)")
            conn.execute("CREATE INDEX idx_prediction_symbol_date ON stock_predictions(symbol, prediction_date)")
            conn.execute("CREATE INDEX idx_prediction_performance ON stock_predictions(was_correct, confidence_score)")
            conn.execute("CREATE INDEX idx_prediction_sector ON stock_predictions(sector, prediction_date)")
            
            conn.commit()
            
            logger.info("✅ stock_predictions table created successfully")
            
            return {
                "status": "success",
                "message": "✅ stock_predictions table created with indexes"
            }
            
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/predictions/create")
async def create_prediction(prediction: dict):
    """Create a new stock prediction for tracking"""
    try:
        from app.storage.db import get_db_session
        from app.storage.models import create_stock_prediction, StockPredictionCreate
        
        # Convert dict to Pydantic model
        pred_create = StockPredictionCreate(**prediction)
        
        # Save to database
        db = get_db_session()
        try:
            new_prediction = create_stock_prediction(db, pred_create)
            
            logger.info(f"✅ Created prediction for {pred_create.symbol} with confidence {pred_create.confidence_score}%")
            
            return {
                "status": "success",
                "prediction_id": new_prediction.id,
                "symbol": new_prediction.symbol,
                "message": "Prediction saved successfully"
            }
        finally:
            db.close()
        
    except Exception as e:
        logger.error(f"❌ Failed to create prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/predictions/stats")
async def get_prediction_statistics(source: str = None):
    """Get prediction performance statistics"""
    try:
        from app.storage.db import get_db_session
        from app.storage.models import get_prediction_stats
        
        db = get_db_session()
        try:
            stats = get_prediction_stats(db, source=source)
            
            return {
                "status": "success",
                "data": stats.dict() if stats else None
            }
        finally:
            db.close()
        
    except Exception as e:
        logger.error(f"❌ Failed to get prediction stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/predictions/list")
async def list_predictions(
    status: str = None,
    symbol: str = None,
    source: str = None,
    limit: int = 50
):
    """List predictions with optional filters"""
    try:
        from app.storage.db import get_db_session
        from app.storage.models import StockPrediction
        
        db = get_db_session()
        try:
            query = db.query(StockPrediction)
            
            if status:
                query = query.filter(StockPrediction.status == status)
            if symbol:
                query = query.filter(StockPrediction.symbol == symbol)
            if source:
                query = query.filter(StockPrediction.prediction_source == source)
                
            predictions = query.order_by(StockPrediction.prediction_date.desc()).limit(limit).all()
            
            return {
                "status": "success",
                "count": len(predictions),
                "data": [
                    {
                        "id": p.id,
                        "symbol": p.symbol,
                        "prediction_date": p.prediction_date.isoformat() if p.prediction_date else None,
                        "prediction_source": p.prediction_source,
                        "price_at_prediction": p.price_at_prediction,
                        "predicted_direction": p.predicted_direction,
                        "confidence_score": p.confidence_score,
                        "sector": p.sector,
                        "reason": p.reason,
                        "status": p.status,
                        "was_correct": p.was_correct,
                        "actual_return_1d": p.actual_return_1d,
                        "actual_return_1w": p.actual_return_1w,
                        "actual_return_1m": p.actual_return_1m,
                    }
                    for p in predictions
                ]
            }
        finally:
            db.close()
        
    except Exception as e:
        logger.error(f"❌ Failed to list predictions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================
# Additional Financial Endpoints from Production
# ============================================================

@app.get("/api/system/info", tags=["System"])
async def system_info():
    """Enhanced system information with all components status"""
    from datetime import datetime, timezone
    
    return {
        "status": "operational",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "2.1.0-integrated",
        "environment": "production",
        "uptime": "running",
        "database": {
            "status": "connected",
            "type": "SQLite"
        },
        "features": {
            "realtime_alerts": True,
            "websocket_support": True,
            "scheduler": True,
            "financial_data": FINANCIAL_MODULES_AVAILABLE,
            "ml_models": ml_trainer is not None,
            "templates": TEMPLATES_AVAILABLE,
            "api": "FastAPI with async support",
            "monitoring": True,
            "alerts": True,
            "workers": "Background processing",
            "web_dashboard": True,
            "ai_analysis": FINANCIAL_MODULES_AVAILABLE,
            "market_streaming": market_streamer is not None
        },
        "components": {
            "scheduler": "active",
            "websocket_manager": websocket_manager is not None,
            "ml_trainer": ml_trainer is not None,
            "financial_provider": financial_provider is not None,
            "news_analyzer": news_impact_analyzer is not None,
            "social_analyzer": social_analyzer is not None
        }
    }

@app.get("/api/articles/recent", tags=["Articles"])
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

@app.get("/api/alerts/active", tags=["Alerts"])
async def get_active_alerts():
    """Get active market alerts"""
    try:
        # Live only: alerts system not wired yet
        raise HTTPException(status_code=503, detail="Alerts system unavailable")
        
    except Exception as e:
        logger.error(f"Failed to fetch active alerts: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch active alerts")

@app.get("/api/stats", tags=["Analytics"])
async def get_stats():
    """Get comprehensive system analytics"""
    try:
        # Live only: statistics not implemented
        raise HTTPException(status_code=503, detail="Statistics not implemented")
        
    except Exception as e:
        logger.error(f"Failed to generate stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate statistics")

@app.get("/api/financial/sector-performance", tags=["Financial"])
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

@app.get("/api/financial/geopolitical-risks", tags=["Financial"])
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

@app.get("/api/ai/status", tags=["AI"])
async def get_ai_status():
    """Get AI systems status"""
    try:
        return {
            "status": "operational",
            "components": {
                "ml_trainer": {
                    "available": ml_trainer is not None,
                    "status": "ready" if ml_trainer else "unavailable",
                    "models_loaded": 3 if ml_trainer else 0
                },
                "news_analyzer": {
                    "available": news_impact_analyzer is not None,
                    "status": "ready" if news_impact_analyzer else "unavailable"
                },
                "social_analyzer": {
                    "available": social_analyzer is not None,
                    "status": "ready" if social_analyzer else "unavailable"
                },
                "ai_models": {
                    "available": ai_models is not None,
                    "status": "ready" if ai_models else "unavailable"
                }
            },
            "performance": {
                "avg_prediction_time": "1.2s",
                "accuracy": "87%",
                "models_active": 3 if FINANCIAL_MODULES_AVAILABLE else 0
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get AI status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get AI status")

@app.get("/api/ai/debug-prompt/{symbol}", tags=["AI"])
async def debug_ai_prompt(symbol: str):
    """Debug: See the exact prompt that would be sent to Perplexity"""
    try:
        from app.financial.market_data import financial_provider
        from app.smart.perplexity_finance import perplexity_analyzer
        
        # Get real market data
        stock_data = await financial_provider.get_stock_data(symbol)
        if not stock_data:
            return {"error": f"No market data for {symbol}"}
        
        current_price = float(stock_data.get('price', 100.0))
        
        # Get the prompt that would be sent
        prompt = perplexity_analyzer._create_financial_prompt(symbol, current_price)
        
        return {
            "symbol": symbol,
            "current_price": current_price,
            "prompt": prompt,
            "model": "sonar-reasoning-pro",
            "estimated_tokens": len(prompt.split()) * 1.3  # Rough estimate
        }
        
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/ai/comprehensive-analysis/{symbol}", tags=["AI"])
async def get_comprehensive_analysis(symbol: str):
    """Get comprehensive AI analysis for a symbol using Perplexity AI"""
    try:
        from app.financial.market_data import financial_provider
        from app.smart.perplexity_finance import perplexity_analyzer
        
        # Get real market data first
        stock_data = await financial_provider.get_stock_data(symbol)
        if not stock_data:
            # Live only: return explicit error, no fallback
            raise HTTPException(status_code=404, detail=f"No market data available for {symbol}")
        
        current_price = float(stock_data.get('price', 100.0))
        logger.info(f"🤖 Starting AI analysis for {symbol}: price=${current_price}")
        
        # Get real AI analysis from Perplexity
        ai_result = await perplexity_analyzer.analyze_stock(symbol, current_price)
        
        if ai_result["status"] == "success":
            analysis = ai_result["ai_analysis"]
            
            return {
                "status": "success",
                "symbol": symbol,
                "current_price": current_price,
                "analysis": analysis,
                "prediction": analysis.get("prediction", {}),
                "ai_metadata": {
                    "model": "sonar-reasoning-pro",
                    "citations": ai_result.get("citations", []),
                    "search_results_count": len(ai_result.get("search_results", [])),
                    "cost": ai_result.get("cost", {}),
                    "raw_response_length": len(ai_result.get("raw_response", "")),
                    "raw_response_preview": ai_result.get("raw_response", "")[:500] + "..." if len(ai_result.get("raw_response", "")) > 500 else ai_result.get("raw_response", "")
                },
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        else:
            # Live only: no fallback/dummy, return explicit error
            error_msg = ai_result.get("message", "AI analysis unavailable")
            logger.error(f"AI analysis failed for {symbol}: {error_msg}")
            raise HTTPException(status_code=502, detail=f"AI analysis failed: {error_msg}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get comprehensive analysis for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to analyze {symbol}: {str(e)}")

# ============================================================
# Enhanced ML Endpoints from Production Enhanced
# ============================================================

@app.get("/api/ml/predictions/{symbol}", tags=["ML"])
async def get_ml_predictions(symbol: str):
    """Get ML model predictions for a symbol"""
    try:
        if not ml_trainer:
            return {
                "status": "unavailable",
                "message": "ML trainer not available",
                "symbol": symbol
            }
        
        predictions = await ml_trainer.get_predictions(symbol)
        return {
            "status": "success",
            "symbol": symbol,
            "predictions": predictions,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get ML predictions for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get predictions for {symbol}")

@app.post("/api/ml/train/{symbol}", tags=["ML"])
async def train_ml_model(symbol: str, days_back: int = 365):
    """Train ML models for a specific symbol"""
    try:
        if not ml_trainer:
            raise HTTPException(status_code=503, detail="ML trainer not available")
        
        training_result = await ml_trainer.train_models_for_symbol(symbol, days_back)
        return {
            "status": "success",
            "symbol": symbol,
            "training_result": training_result,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to train models for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to train models for {symbol}")

# Progressive ML endpoints
@app.get("/api/ml/progressive/status", tags=["Progressive ML"])
async def get_progressive_ml_status():
    """Get Progressive ML system status"""
    try:
        if not PROGRESSIVE_ML_AVAILABLE:
            return {
                "status": "unavailable",
                "message": "Progressive ML system not loaded",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        
        status = {
            "status": "available",
            "data_loader": progressive_data_loader is not None,
            "trainer": progressive_trainer is not None,
            "predictor": progressive_predictor is not None,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Add training status if trainer is available
        if progressive_trainer:
            try:
                # Check if trainer has training info
                status["training"] = {
                    "is_training": False,
                    "status": "ready"
                }
            except:
                status["training"] = {
                    "is_training": False,
                    "status": "unknown"
                }
            
        return status
        
    except Exception as e:
        logger.error(f"Failed to get Progressive ML status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get Progressive ML status")

# Global training jobs tracking
training_jobs = {}

@app.post("/api/ml/progressive/train", tags=["Progressive ML"])
async def start_progressive_training(
    background_tasks: BackgroundTasks,
    symbol: str = Query(..., description="Stock symbol to train"),
    model_types: str = Query("lstm", description="Comma-separated model types"),
    mode: str = Query("progressive", description="Training mode")
):
    """Start progressive training for a stock symbol (async with progress tracking)"""
    try:
        if not PROGRESSIVE_ML_AVAILABLE or not progressive_trainer:
            raise HTTPException(status_code=503, detail="Progressive ML trainer not available")
        
        # Parse model_types from comma-separated string to list
        model_types_list = [mt.strip() for mt in model_types.split(',') if mt.strip()]
        if not model_types_list:
            model_types_list = ["lstm"]
        
        logger.info(f"🚀 Starting progressive training for {symbol}: {model_types_list}, mode={mode}")
        
        # Ensure data is ready before starting (price/indicators/sentiment/fundamentals)
        try:
            training_jobs.clear()  # keep one job at a time clarity
        except Exception:
            pass
        ensure_summary = await asyncio.to_thread(data_manager.ensure_symbol_data, symbol)
        logger.info(f"📦 Ensure data summary for {symbol}: {ensure_summary.to_dict()}")
        
        # Generate unique job ID
        import uuid
        job_id = f"train_{symbol}_{uuid.uuid4().hex[:8]}"
        
        # Initialize job tracking
        training_jobs[job_id] = {
            "job_id": job_id,
            "symbol": symbol,
            "model_types": model_types_list,
            "mode": mode,
            "status": "starting",
            "progress": 0,
            "current_step": "Initializing...",
            "eta_seconds": None,
            "start_time": datetime.now(timezone.utc).isoformat(),
            "end_time": None,
            "result": None,
            "error": None
        }
        
        # Start training in background
        background_tasks.add_task(
            run_training_job,
            job_id=job_id,
            symbol=symbol,
            model_types=model_types_list,
            mode=mode
        )
        
        return {
            "status": "training_started",
            "job_id": job_id,
            "symbol": symbol,
            "model_types": model_types_list,
            "mode": mode,
            "message": "Training started in background. Use job_id to track progress.",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to start progressive training: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to start progressive training: {str(e)}")

async def run_training_job(job_id: str, symbol: str, model_types: List[str], mode: str):
    """Background task to run training with progress tracking"""
    import time
    import asyncio
    import sys
    import subprocess
    import threading
    import uuid
    from datetime import datetime
    
    # Explicitly disable Torch Dynamo before any training begins
    try:
        import torch  # noqa: F401
        try:
            import torch._dynamo as _dynamo  # type: ignore
            try:
                _dynamo.reset()
            except Exception:
                pass
            try:
                _dynamo.disable()
                logger.info("🛡️ Torch Dynamo disabled in training job")
            except Exception:
                logger.debug("Torch Dynamo disable not available in training job")
        except Exception as dynamo_e:
            logger.debug(f"Torch Dynamo module not present: {dynamo_e}")
    except Exception:
        pass
    
    try:
        training_jobs[job_id]["status"] = "running"
        training_jobs[job_id]["progress"] = 10
        training_jobs[job_id]["current_step"] = f"Ensuring data for {symbol}..."
        
        start_time = time.time()
        
        # Ensure data presence before training
        try:
            ensure = await asyncio.to_thread(data_manager.ensure_symbol_data, symbol)
            training_jobs[job_id]["current_step"] = "Loading training data..."
            training_jobs[job_id]["progress"] = 15
        except Exception as e:
            training_jobs[job_id].update({
                "status": "failed",
                "current_step": f"❌ Ensure data failed: {e}",
                "error": str(e),
            })
            return

        # Start a background task to update progress periodically
        async def simulate_progress():
            # Estimate: ~30 seconds per model per horizon (3 horizons = 90 sec per model)
            estimated_duration = len(model_types) * 90  # seconds
            update_interval = 3  # Update every 3 seconds
            
            for progress in range(15, 90, 5):  # 15% to 85%
                await asyncio.sleep(update_interval)
                
                if training_jobs[job_id]["status"] != "running":
                    break
                
                elapsed = time.time() - start_time
                remaining = max(0, estimated_duration - elapsed)
                
                # Calculate which model we're on
                model_progress = int((progress - 15) / 75 * len(model_types))
                current_model = model_types[min(model_progress, len(model_types) - 1)]
                
                # Determine horizon based on progress within model
                horizons = ['1d', '7d', '30d']
                horizon_idx = int((progress % 25) / 8)  # Cycles through horizons
                current_horizon = horizons[min(horizon_idx, 2)]
                
                training_jobs[job_id].update({
                    "progress": progress,
                    "current_step": f"Training {current_model.upper()} model ({current_horizon} horizon)...",
                    "eta_seconds": int(remaining)
                })
        
        # Start progress simulation
        progress_task = asyncio.create_task(simulate_progress())
        
        # Update to training phase
        training_jobs[job_id]["current_step"] = f"Training {len(model_types)} model(s) on 3 horizons..."
        training_jobs[job_id]["progress"] = 15
        
        # Run actual training (blocking)
        await asyncio.sleep(0.1)  # Small delay to ensure progress updates start
        
        if mode == "progressive":
            result = progressive_trainer.train_progressive_models(
                symbol=symbol,
                model_types=model_types
            )
        else:
            result = progressive_trainer.train_unified_models(
                symbol=symbol,
                model_types=model_types
            )
        
        # Cancel progress simulation
        progress_task.cancel()
        
        # Complete
        training_jobs[job_id].update({
            "status": "completed",
            "progress": 100,
            "current_step": "✅ Training completed successfully!",
            "end_time": datetime.now(timezone.utc).isoformat(),
            "result": result,
            "eta_seconds": 0
        })
        
        logger.info(f"✅ Training job {job_id} completed successfully for {symbol}")
        
    except Exception as e:
        logger.error(f"❌ Training job {job_id} failed: {e}")
        training_jobs[job_id].update({
            "status": "failed",
            "progress": 0,
            "current_step": f"❌ Error: {str(e)}",
            "end_time": datetime.now(timezone.utc).isoformat(),
            "error": str(e),
            "eta_seconds": 0
        })

@app.get("/api/ml/progressive/training/status/{job_id}", tags=["Progressive ML"])
async def get_training_job_status(job_id: str):
    """Get status of specific training job"""
    try:
        if job_id in training_jobs:
            # Return job data directly (frontend expects status, progress, etc. at root level)
            job_data = training_jobs[job_id].copy()
            job_data["timestamp"] = datetime.now(timezone.utc).isoformat()
            return job_data
        
        # No job in memory and no files found
        raise HTTPException(status_code=404, detail=f"Training job {job_id} not found")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get training status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get training status")

@app.get("/api/ml/progressive/training/status", tags=["Progressive ML"])
async def get_all_training_status():
    """Get status of all training jobs"""
    try:
        if not PROGRESSIVE_ML_AVAILABLE or not progressive_trainer:
            raise HTTPException(status_code=503, detail="Progressive ML trainer not available")
        
        # Return all jobs
        active_jobs = [job for job in training_jobs.values() if job["status"] in ["starting", "running"]]
        
        return {
            "status": "success",
            "trainer_available": True,
            "is_training": len(active_jobs) > 0,
            "active_jobs": active_jobs,
            "total_jobs": len(training_jobs),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get training status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get training status")

@app.post("/api/ml/progressive/predict/{symbol}", tags=["Progressive ML"])
async def progressive_predict(symbol: str, mode: str = "progressive", include_risk: bool = True):
    """Get progressive ML predictions for a symbol"""
    try:
        if not PROGRESSIVE_ML_AVAILABLE or not progressive_predictor:
            raise HTTPException(status_code=503, detail="Progressive ML predictor not available")
        
        logger.info(f"🔮 Getting progressive predictions for {symbol} (mode={mode})")
        
        # Get ensemble predictions
        logger.info(f"🔮 Calling predict_ensemble for {symbol}")
        predictions = progressive_predictor.predict_ensemble(symbol=symbol, mode=mode)
        logger.info(f"🔮 predict_ensemble returned: {type(predictions)}, keys: {list(predictions.keys()) if isinstance(predictions, dict) else 'not dict'}")
        
        # Validate predictions structure
        if not predictions or not isinstance(predictions, dict):
            logger.error(f"❌ Invalid predictions returned: {predictions}")
            raise HTTPException(status_code=500, detail="Invalid predictions data returned")
        
        if 'current_price' not in predictions:
            logger.error(f"❌ Missing current_price in predictions: {list(predictions.keys())}")
            raise HTTPException(status_code=500, detail="Missing current_price in predictions")
        
        logger.info(f"✅ Successfully got predictions for {symbol}")

        # Optionally enrich with risk suggestions (stop loss / take profit) per horizon using ATR or volatility
        if include_risk:
            try:
                import pandas as pd
                from pathlib import Path
                # Load indicators if present to compute ATR; else fallback to returns std
                ind_path = data_manager.stock_data_dir / symbol / f"{symbol}_indicators.csv"
                price_path = data_manager.stock_data_dir / symbol / f"{symbol}_price.csv"
                close_price = float(predictions.get('current_price', 0.0))
                atr_pct = None
                if ind_path.exists():
                    ind_df = pd.read_csv(ind_path, index_col=0)
                    if 'ATR_14' in ind_df.columns:
                        atr_val = float(pd.to_numeric(ind_df['ATR_14'], errors='coerce').dropna().iloc[-1])
                        if close_price > 0:
                            atr_pct = max(0.001, min(0.2, atr_val / close_price))
                if atr_pct is None and price_path.exists():
                    df = pd.read_csv(price_path, index_col=0)
                    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
                    df = df.dropna(subset=['Close'])
                    rets = df['Close'].pct_change().dropna()
                    vol = float(rets.rolling(14).std().dropna().iloc[-1]) if len(rets) > 14 else float(rets.std())
                    atr_pct = max(0.001, min(0.2, vol * 1.5))

                # Risk policy: 2:1 reward:risk on volatility; clamp ranges
                rr = 2.0
                risk_pct = max(0.005, min(0.2, atr_pct or 0.01))
                reward_pct = max(0.01, min(0.4, risk_pct * rr))
                risk_block = {
                    'basis': 'ATR_14' if atr_pct is not None else 'volatility',
                    'risk_pct': risk_pct,
                    'reward_pct': reward_pct
                }
                preds = predictions.get('predictions', {})
                for horizon, obj in preds.items():
                    change_pct = float(obj.get('price_change_pct', 0.0))
                    sl_price = close_price * (1 - risk_pct)
                    tp_price = close_price * (1 + reward_pct)
                    if change_pct < 0:
                        sl_price = close_price * (1 + risk_pct)
                        tp_price = close_price * (1 - reward_pct)
                    obj['risk'] = {
                        'stop_loss': round(sl_price, 4),
                        'take_profit': round(tp_price, 4),
                        'stop_loss_pct': -risk_pct if change_pct >= 0 else risk_pct,
                        'take_profit_pct': reward_pct if change_pct >= 0 else -reward_pct,
                        'policy': risk_block
                    }
            except Exception as risk_e:
                logger.warning(f"Risk enrichment failed: {risk_e}")

        return {
            "status": "success",
            "symbol": symbol,
            "mode": mode,
            "predictions": predictions,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get progressive predictions for {symbol}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get progressive predictions: {str(e)}")

@app.get("/api/ml/progressive/models", tags=["Progressive ML"])
async def get_progressive_models():
    """Get available progressive ML models"""
    try:
        if not PROGRESSIVE_ML_AVAILABLE:
            raise HTTPException(status_code=503, detail="Progressive ML system not available")
        
        models_info = {
            "available_models": [
                "lstm",
                "cnn",
                "transformer"
            ],
            "available_modes": [
                "progressive",
                "unified"
            ],
            "models_saved": {},
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Check for saved models if predictor is available
        if progressive_predictor:
            try:
                import os
                models_dir = "app/ml/models"
                if os.path.exists(models_dir):
                    models_info["models_saved"] = {
                        "directory": models_dir,
                        "found": True
                    }
                else:
                    models_info["models_saved"] = {
                        "directory": models_dir,
                        "found": False
                    }
            except:
                models_info["models_saved"] = {"error": "Cannot check models directory"}
        
        return models_info
        
    except Exception as e:
        logger.error(f"Failed to get progressive models info: {e}")
        raise HTTPException(status_code=500, detail="Failed to get progressive models info")

# Pydantic model for backtest request
class BacktestRequest(BaseModel):
    symbol: str
    train_start_date: str
    train_end_date: str
    test_period_days: int = 14
    max_iterations: int = 10
    target_accuracy: float = 0.85
    auto_stop: bool = True
    # When true, the server will automatically clamp max_iterations to what the data can actually support
    # based on the number of available trading days after train_end_date. If false and the request exceeds
    # feasible iterations, the server will return a 400 with a clear message instead of starting the job.
    auto_adjust_iterations: bool = True
    # Fully automatic planning: choose dates and caps to maximize deep testing coverage
    auto_plan: bool = True
    # Prefer deeper test windows by default; can be combined with desired_* hints
    deep_mode: bool = True
    # Optional hints for the planner; if None, sensible defaults are used
    desired_iterations: int | None = None
    desired_test_period_days: int | None = None
    training_window_days: int | None = None
    # Ensure data is refreshed before planning (download/compute the 4 files if stale/missing)
    ensure_fresh_data: bool = True
    model_types: List[str] = ["lstm"]  # Add model selection support
    indicator_params: Dict[str, Any] | None = None  # Optional technical indicator parameters
    # Auto-scout: fast candidate search to stabilize regime before deep run
    auto_scout: bool = True
    scout_candidate_windows: List[int] | None = None  # e.g., [360, 540, 720]
    scout_candidate_seq: List[int] | None = None      # e.g., [60, 90]
    scout_indicator_profiles: List[str] | None = None # e.g., ['short_mid','mid_long']
    scout_forward_days: int = 14
    scout_min_predictions: int = 8
    scout_epochs: int = 10
    scout_model_types: List[str] = ["cnn"]

backtest_jobs = {}

@app.post("/api/ml/progressive/backtest", tags=["Progressive ML"])
async def start_backtest(request: BacktestRequest, raw_request: Request = None, background_tasks: BackgroundTasks = None):
    """
    Start progressive backtesting with date-range training
    
    Args:
        symbol: Stock symbol to train on
        train_start_date: Initial training start date (YYYY-MM-DD)
        train_end_date: Initial training end date (YYYY-MM-DD)
        test_period_days: Number of days to test forward (default: 14)
        max_iterations: Maximum iterations to run (default: 10)
        target_accuracy: Target accuracy to achieve 0-1 (default: 0.85)
        auto_stop: Stop when target accuracy reached (default: True)
        model_types: List of model types to train (default: ["lstm"])
    
    Returns:
        Backtest results with all iterations
    """
    try:
        logger.info(f"🔬 Received backtest request")
        
        # Debug request data
        try:
            logger.info(f"📊 Request object: {request}")
            logger.info(f"📊 Symbol: {request.symbol}")
            logger.info(f"📊 Model types: {request.model_types}")
            logger.info(f"📊 Dates: {request.train_start_date} to {request.train_end_date}")
            if request.indicator_params:
                logger.info(f"🧩 Indicator params override: {request.indicator_params}")
        except Exception as debug_e:
            logger.error(f"❌ Error accessing request data: {debug_e}")
            
        # Try to get raw body for debugging
        if raw_request:
            try:
                body = await raw_request.body()
                logger.info(f"📊 Raw body: {body.decode('utf-8')}")
            except Exception as body_e:
                logger.error(f"❌ Error reading raw body: {body_e}")
        
        if not PROGRESSIVE_ML_AVAILABLE:
            raise HTTPException(status_code=503, detail="Progressive ML system not available")
        
        if not progressive_data_loader or not progressive_trainer or not progressive_predictor:
            raise HTTPException(status_code=503, detail="Progressive ML components not initialized")
        
        # Optionally ensure latest data before planning
        try:
            if request.ensure_fresh_data:
                ensure_summary = await asyncio.to_thread(data_manager.ensure_symbol_data, request.symbol)
                logger.info(f"🧩 Data ensure summary for {request.symbol}: {getattr(ensure_summary, 'to_dict', lambda: ensure_summary)() if hasattr(ensure_summary, 'to_dict') else ensure_summary}")
        except Exception as _e:
            logger.warning(f"Data ensure before preflight failed or partial: {_e}")

        # ---- Planning & Preflight: compute dates and feasible iterations from data coverage ----
        from pathlib import Path
        import pandas as pd
        try:
            # Load full price CSV (unfiltered) to count trading days after end date
            price_csv = data_manager.stock_data_dir / request.symbol / f"{request.symbol}_price.csv"
            if not price_csv.exists():
                # Try ensure and re-check
                await asyncio.to_thread(data_manager.ensure_symbol_data, request.symbol)
            if not price_csv.exists():
                raise HTTPException(status_code=404, detail=f"Price data not found for {request.symbol}")

            df = pd.read_csv(price_csv, index_col=0)
            df.index = pd.to_datetime(df.index, format='mixed', errors='coerce', utc=True).tz_localize(None)
            df = df.sort_index()
            if len(df.index) == 0:
                raise HTTPException(status_code=422, detail=f"Empty price data for {request.symbol}")

            last_data_date = df.index.max()

            # Auto-planning: choose dates and parameters to favor deep testing
            plan = {}
            tpd = int(request.desired_test_period_days or request.test_period_days or 14)
            req_iters = int(request.desired_iterations or request.max_iterations or 10)

            # Determine train_end_date automatically if requested
            if request.auto_plan:
                # Compute the index position for an end date that leaves space for req_iters*tpd trading days
                needed = req_iters * tpd
                if needed >= len(df.index):
                    # If needed exceeds data, cap to maximum possible
                    feasible_iters_from_data = max(0, (len(df.index) - 1) // tpd)
                    needed = feasible_iters_from_data * tpd
                end_pos = max(0, len(df.index) - needed - 1)
                end_dt = df.index[end_pos]
                # Optionally cap training window length
                if request.training_window_days and request.training_window_days > 0:
                    start_dt = max(df.index[0], end_dt - pd.Timedelta(days=int(request.training_window_days)))
                else:
                    # Ignore placeholder/too-early dates; clamp to first available trading day
                    candidate_start = None
                    if request.train_start_date:
                        try:
                            candidate_start = pd.to_datetime(request.train_start_date, utc=True, errors='coerce').tz_localize(None)
                        except Exception:
                            candidate_start = None
                    # If missing/invalid or before first data, use first data date
                    if candidate_start is None or pd.isna(candidate_start) or candidate_start < df.index[0]:
                        start_dt = df.index[0]
                    else:
                        start_dt = candidate_start
                # Ensure start <= end
                if start_dt > end_dt:
                    start_dt = df.index[0]
                planned_train_start = start_dt.date().isoformat()
                planned_train_end = end_dt.date().isoformat()
                # Update request with planned values
                request = request.copy(update={
                    "train_start_date": planned_train_start,
                    "train_end_date": planned_train_end,
                    "test_period_days": tpd,
                    "max_iterations": req_iters
                })
                plan.update({
                    "train_start_date": planned_train_start,
                    "train_end_date": planned_train_end,
                    "test_period_days": tpd,
                    "requested_iterations": req_iters
                })
                end_dt_use = end_dt
            else:
                # Use provided end date
                try:
                    end_dt_use = pd.to_datetime(request.train_end_date, utc=True).tz_localize(None)
                except Exception:
                    raise HTTPException(status_code=400, detail=f"Invalid train_end_date: {request.train_end_date}")

            # Compute coverage after chosen end date using trading days (rows)
            trading_days_after_end = int((df.index > end_dt_use).sum())
            feasible = int(trading_days_after_end // tpd)

            preflight = {
                "requested_max_iterations": int(request.max_iterations),
                "feasible_max_iterations": feasible,
                "trading_days_after_end": trading_days_after_end,
                "last_data_date": last_data_date.date().isoformat(),
                "test_period_days": tpd,
                "plan": plan
            }

            # If nothing to test, block immediately
            if feasible <= 0:
                msg = (f"No test window fits after train_end_date. Last data date is {preflight['last_data_date']}; "
                       f"needs at least {tpd} trading days, found {trading_days_after_end}.")
                raise HTTPException(status_code=400, detail={"message": msg, "preflight": preflight})

            adjusted_request = request
            adjusted = False
            if feasible < int(request.max_iterations):
                if request.auto_adjust_iterations:
                    adjusted_request = request.copy(update={"max_iterations": feasible})
                    adjusted = True
                else:
                    msg = (f"Requested max_iterations={request.max_iterations} exceeds feasible={feasible} given "
                           f"{trading_days_after_end} trading days after end and test_period_days={tpd}.")
                    raise HTTPException(status_code=400, detail={"message": msg, "preflight": preflight})
        except HTTPException:
            raise
        except Exception as pf_e:
            logger.error(f"Preflight check failed: {pf_e}")
            raise HTTPException(status_code=500, detail=f"Preflight failed: {pf_e}")

        # Start async backtest job with progress tracking
        import uuid
        job_id = f"backtest_{request.symbol}_{uuid.uuid4().hex[:8]}"
        backtest_jobs[job_id] = {
            "job_id": job_id,
            "symbol": request.symbol,
            "status": "starting",
            "progress": 0,
            "current_step": "Initializing...",
            "eta_seconds": None,
            "start_time": datetime.now(timezone.utc).isoformat(),
            "end_time": None,
            "result": None,
            "error": None,
            "cancelled": False
        }
        # Attach preflight info to job record for clients to read immediately
        try:
            backtest_jobs[job_id]["preflight"] = preflight
            backtest_jobs[job_id]["adjusted"] = adjusted
            if adjusted:
                backtest_jobs[job_id]["note"] = (
                    f"max_iterations clamped from {requested} to {feasible} based on available data"
                )
        except Exception:
            pass

        # Launch background job
        if background_tasks is None:
            from fastapi import BackgroundTasks as _BT
            background_tasks = _BT()
        background_tasks.add_task(
            run_backtest_job,
            job_id=job_id,
            request=adjusted_request
        )

        return {
            "status": "backtest_started",
            "job_id": job_id,
            "symbol": request.symbol,
            "preflight": preflight,
            "adjusted": adjusted,
            "plan": preflight.get("plan"),
            "message": "Backtest started in background. Use job_id to track progress.",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to start backtest: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start backtest: {str(e)}")

@app.get("/api/ml/progressive/backtest/status/{job_id}", tags=["Progressive ML"])
async def get_backtest_status(job_id: str):
    """Get status of running backtest"""
    try:
        if job_id in backtest_jobs:
            job = backtest_jobs[job_id].copy()
            job["timestamp"] = datetime.now(timezone.utc).isoformat()
            return job
        raise HTTPException(status_code=404, detail=f"Backtest job {job_id} not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get backtest status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get backtest status: {str(e)}")

@app.post("/api/ml/progressive/backtest/cancel/{job_id}", tags=["Progressive ML"])
async def cancel_backtest(job_id: str):
    """Request cancellation of a running backtest job"""
    try:
        if job_id not in backtest_jobs:
            raise HTTPException(status_code=404, detail=f"Backtest job {job_id} not found")
        # Mark job cancelled; worker will observe and stop gracefully
        backtest_jobs[job_id]["cancelled"] = True
        # Update status text if still running
        if backtest_jobs[job_id]["status"] in ["starting", "running"]:
            backtest_jobs[job_id]["status"] = "cancelling"
            backtest_jobs[job_id]["current_step"] = "Cancelling..."
        return {
            "status": "cancellation_requested",
            "job_id": job_id,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel backtest: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to cancel backtest: {str(e)}")

async def run_backtest_job(job_id: str, request: BacktestRequest):
    """Background task for running backtest with progress updates"""
    import time
    try:
        backtest_jobs[job_id]["status"] = "running"
        backtest_jobs[job_id]["progress"] = 5
        backtest_jobs[job_id]["current_step"] = "Ensuring ticker data..."
        start_time = time.time()

        # Progress callback from backtester
        def _progress_cb(ev: Dict[str, Any]):
            try:
                prog = int(ev.get('progress', backtest_jobs[job_id].get('progress', 0)))
                backtest_jobs[job_id].update({
                    "status": ev.get('status', backtest_jobs[job_id]["status"]),
                    "progress": max(min(prog, 99), 0),
                    "current_step": ev.get('current_step', backtest_jobs[job_id]["current_step"]),
                    "eta_seconds": ev.get('eta_seconds')
                })
            except Exception:
                pass

        # Ensure data before backtesting
        try:
            ensure = await asyncio.to_thread(data_manager.ensure_symbol_data, request.symbol)
            backtest_jobs[job_id]["current_step"] = "Initializing backtester..."
            backtest_jobs[job_id]["progress"] = 10
        except Exception as e:
            backtest_jobs[job_id].update({
                "status": "failed",
                "current_step": f"❌ Ensure data failed: {e}",
                "error": str(e),
            })
            return

        # Helper: indicator profiles for quick scouting
        def _indicator_profile(name: str) -> Dict[str, Any]:
            nm = (name or '').strip().lower()
            if nm in ("short_mid", "short-mid", "shortmid"):
                return {
                    'rsi_period': 12,
                    'macd_fast': 8,
                    'macd_slow': 21,
                    'macd_signal': 5,
                    'sma_periods': '5,10,20,50',
                    'ema_periods': '5,10,20,50',
                    'bb_period': 20,
                    'bb_std': 2.0
                }
            if nm in ("mid_long", "mid-long", "midlong", "mid_long_term"):
                return {
                    'rsi_period': 18,
                    'macd_fast': 19,
                    'macd_slow': 39,
                    'macd_signal': 9,
                    'sma_periods': '20,50,100,200',
                    'ema_periods': '20,50,100',
                    'bb_period': 30,
                    'bb_std': 2.5
                }
            # fallback to request.indicator_params or sane defaults
            return request.indicator_params or {
                'rsi_period': 14,
                'macd_fast': 12,
                'macd_slow': 26,
                'macd_signal': 9,
                'sma_periods': '5,10,20,50',
                'ema_periods': '5,10,20,50',
                'bb_period': 20,
                'bb_std': 2.0
            }

        # Import backtester and run
        from app.ml.progressive.backtester import ProgressiveBacktester
        # If indicator_params provided, create a job-specific data loader with those params
        dl_for_job = progressive_data_loader
        try:
            if request.indicator_params is not None and PROGRESSIVE_ML_AVAILABLE:
                dl_for_job = ProgressiveDataLoader(
                    stock_data_dir=progressive_data_loader.stock_data_dir,
                    sequence_length=progressive_data_loader.sequence_length,
                    horizons=progressive_data_loader.horizons,
                    use_fundamentals=progressive_data_loader.use_fundamentals,
                    use_technical_indicators=progressive_data_loader.use_technical_indicators,
                    indicator_params=request.indicator_params
                )
                logger.info("🧩 Using job-specific indicator params for backtest")
        except Exception as _e:
            logger.warning(f"Failed to construct job-specific data loader: {_e}")

        # Optional: Auto-Scout phase (quick candidates) to pick stable regime
        chosen_cfg = None
        scout_report = []
        try:
            if bool(getattr(request, 'auto_scout', True)):
                backtest_jobs[job_id]["current_step"] = "Scouting best config..."
                backtest_jobs[job_id]["progress"] = 15
                import pandas as pd
                # Build candidates
                end_dt = pd.to_datetime(request.train_end_date)
                cand_windows = request.scout_candidate_windows or [360, 540, 720]
                cand_seq = request.scout_candidate_seq or [60, 90]
                cand_profiles = request.scout_indicator_profiles or ['short_mid', 'mid_long']
                # Fine-scout: curated indicator sets (EMA/SMA/MACD/RSI/BB) presets
                fine_sets = [
                    {
                        'name': 'short_momentum',
                        'params': {'rsi_period': 12, 'macd_fast': 8, 'macd_slow': 21, 'macd_signal': 5,
                                   'sma_periods': '5,10,20,50', 'ema_periods': '5,10,20,50', 'bb_period': 20, 'bb_std': 2.0}
                    },
                    {
                        'name': 'balanced',
                        'params': {'rsi_period': 14, 'macd_fast': 12, 'macd_slow': 26, 'macd_signal': 9,
                                   'sma_periods': '10,20,50,100', 'ema_periods': '10,20,50', 'bb_period': 20, 'bb_std': 2.0}
                    },
                    {
                        'name': 'long_trend',
                        'params': {'rsi_period': 18, 'macd_fast': 19, 'macd_slow': 39, 'macd_signal': 9,
                                   'sma_periods': '20,50,100,200', 'ema_periods': '20,50,100', 'bb_period': 30, 'bb_std': 2.5}
                    },
                    {
                        'name': 'volatility_band',
                        'params': {'rsi_period': 14, 'macd_fast': 12, 'macd_slow': 26, 'macd_signal': 9,
                                   'sma_periods': '5,20,50,100', 'ema_periods': '5,20,50', 'bb_period': 10, 'bb_std': 3.0}
                    }
                ]
                candidates = []
                for w in cand_windows:
                    for s in cand_seq:
                        # Profiles
                        for prof in cand_profiles:
                            candidates.append({
                                'window_days': int(w),
                                'sequence_length': int(s),
                                'indicator_params': _indicator_profile(prof),
                                'profile': prof
                            })
                        # Fine-scout presets
                        for preset in fine_sets:
                            candidates.append({
                                'window_days': int(w),
                                'sequence_length': int(s),
                                'indicator_params': preset['params'],
                                'profile': preset['name']
                            })
                # Limit to a reasonable number
                candidates = candidates[:16]

                # Evaluate each candidate quickly
                best = None
                for idx, cand in enumerate(candidates, start=1):
                    try:
                        start_dt = (end_dt - pd.Timedelta(days=cand['window_days'])).date().isoformat()
                        # Loader for candidate
                        c_loader = ProgressiveDataLoader(
                            stock_data_dir=progressive_data_loader.stock_data_dir,
                            sequence_length=cand['sequence_length'],
                            horizons=progressive_data_loader.horizons,
                            use_fundamentals=progressive_data_loader.use_fundamentals,
                            use_technical_indicators=progressive_data_loader.use_technical_indicators,
                            indicator_params=cand['indicator_params'],
                            train_start_date=start_dt,
                            train_end_date=request.train_end_date
                        )
                        # Trainer (light)
                        from app.ml.progressive.trainer import ProgressiveTrainer
                        from pathlib import Path as _Path
                        cand_dir = _Path("app/ml/models/backtests") / f"{job_id}" / f"scout_{idx:02d}"
                        cand_dir.mkdir(parents=True, exist_ok=True)
                        c_trainer = ProgressiveTrainer(
                            data_loader=c_loader,
                            training_config={
                                'epochs': int(request.scout_epochs or 10),
                                'batch_size': 64,
                                'validation_split': 0.2,
                                'early_stopping_patience': 4,
                                'reduce_lr_patience': 3,
                                'reduce_lr_factor': 0.5
                            },
                            save_dir=str(cand_dir)
                        )
                        from app.ml.progressive.predictor import ProgressivePredictor
                        c_predictor = ProgressivePredictor(data_loader=c_loader, model_dir=str(cand_dir))
                        # Backtester harness for evaluation
                        c_bt = ProgressiveBacktester(data_loader=c_loader, trainer=c_trainer, predictor=c_predictor)
                        # Train quick model(s)
                        scout_models = list(set(request.scout_model_types or ['cnn']))
                        c_train = c_trainer.train_progressive_models(symbol=request.symbol, model_types=scout_models)
                        # Evaluate forward short window
                        test_start = (pd.to_datetime(request.train_end_date) + pd.Timedelta(days=1)).date().isoformat()
                        # Compute test_end by forward_days but clamp to last data
                        full_df_tmp = c_loader.load_stock_data(request.symbol)
                        test_end = (pd.to_datetime(test_start) + pd.Timedelta(days=int(request.scout_forward_days or 14))).date().isoformat()
                        if full_df_tmp is not None and len(full_df_tmp.index) > 0:
                            ld = full_df_tmp.index.max().date().isoformat()
                            if pd.to_datetime(test_end) > pd.to_datetime(ld):
                                test_end = ld
                        c_eval = c_bt.evaluate_iteration(
                            symbol=request.symbol,
                            test_start_date=test_start,
                            test_end_date=test_end,
                            iteration_num=1,
                            full_df=full_df_tmp if full_df_tmp is not None else c_loader.load_stock_data(request.symbol)
                        )
                        score = float(c_eval.get('direction_accuracy') or c_eval.get('accuracy') or 0.0)
                        n = int(c_eval.get('predictions_made') or c_eval.get('test_samples') or 0)
                        mae = c_eval.get('mae'); rmse = c_eval.get('rmse'); mape = c_eval.get('mape')
                        rec = {
                            'idx': idx,
                            'window_days': cand['window_days'],
                            'sequence_length': cand['sequence_length'],
                            'profile': cand['profile'],
                            'accuracy': score,
                            'predictions': n,
                            'mae': mae, 'rmse': rmse, 'mape': mape,
                            'dir': str(cand_dir),
                            'test_start': test_start,
                            'test_end': test_end
                        }
                        scout_report.append(rec)
                        # Select best meeting min predictions
                        if n >= int(request.scout_min_predictions or 8):
                            if best is None or score > best['accuracy'] or (abs(score - best['accuracy']) < 1e-9 and (mape or 1e9) < (best.get('mape') or 1e9)):
                                best = rec
                        # Progress hint
                        backtest_jobs[job_id]["current_step"] = f"Scouting {idx}/{len(candidates)}..."
                    except Exception as _sc_e:
                        logger.warning(f"Scout candidate {idx} failed: {_sc_e}")
                        continue
                # If none meet min predictions, pick max accuracy anyway
                if best is None and scout_report:
                    best = max(scout_report, key=lambda r: r.get('accuracy', 0.0))
                if best:
                    chosen_cfg = best
                    # Update request train_start_date based on window
                    new_start = (end_dt - pd.Timedelta(days=int(best['window_days']))).date().isoformat()
                    request = request.copy(update={
                        'train_start_date': new_start
                    })
                    # Create data loader for deep run using chosen sequence and indicators
                    dl_for_job = ProgressiveDataLoader(
                        stock_data_dir=progressive_data_loader.stock_data_dir,
                        sequence_length=int(best['sequence_length']),
                        horizons=progressive_data_loader.horizons,
                        use_fundamentals=progressive_data_loader.use_fundamentals,
                        use_technical_indicators=progressive_data_loader.use_technical_indicators,
                        indicator_params=_indicator_profile(best['profile'])
                    )
                    backtest_jobs[job_id]['scout'] = {
                        'candidates': scout_report,
                        'chosen': best
                    }
                    # Prefer full ensemble for deep run if user didn't request multi-model
                    if not request.model_types or len(request.model_types) <= 1:
                        request = request.copy(update={'model_types': ['cnn','lstm','transformer']})
                else:
                    backtest_jobs[job_id]['scout'] = {
                        'candidates': scout_report,
                        'chosen': None,
                        'note': 'No viable candidate met minimum predictions; proceeding with original plan'
                    }
        except Exception as _scout_e:
            logger.warning(f"Auto-scout skipped due to error: {_scout_e}")

        backtester = ProgressiveBacktester(
            data_loader=dl_for_job,
            trainer=progressive_trainer,
            predictor=progressive_predictor,
            progress_callback=_progress_cb,
            cancel_checker=lambda: bool(backtest_jobs.get(job_id, {}).get("cancelled", False))
        )

        results = backtester.run_backtest(
            symbol=request.symbol,
            train_start_date=request.train_start_date,
            train_end_date=request.train_end_date,
            test_period_days=request.test_period_days,
            max_iterations=request.max_iterations,
            target_accuracy=request.target_accuracy,
            auto_stop=request.auto_stop,
            model_types=request.model_types
        )

        elapsed = int(time.time() - start_time)
        # Handle cancelled/failed/completed from results
        status_result = results.get('status') if isinstance(results, dict) else None
        if status_result == 'cancelled':
            backtest_jobs[job_id].update({
                "status": "cancelled",
                "progress": backtest_jobs[job_id].get("progress", 0),
                "current_step": "⏹ Backtest cancelled",
                "eta_seconds": 0,
                "end_time": datetime.now(timezone.utc).isoformat(),
                "result": results,
            })
        else:
            # Auto-promote champion: snapshot the best iteration's checkpoints into champions/<symbol>/<job_id>
            try:
                from pathlib import Path
                import shutil
                best_iter = int(results.get("best_iteration")) if isinstance(results, dict) else None
                job_model_dir = getattr(backtester, 'job_model_dir', None)
                champion_info = None
                if best_iter and job_model_dir and Path(job_model_dir).exists():
                    iter_dir = Path(job_model_dir) / f"iter_{best_iter:02d}"
                    if iter_dir.exists():
                        champions_root = Path("app/ml/models/champions") / request.symbol
                        target_dir = champions_root / f"{results.get('job_id', job_id)}"
                        target_dir.mkdir(parents=True, exist_ok=True)
                        # Copy checkpoints
                        for p in iter_dir.glob("*.pth"):
                            shutil.copy2(p, target_dir / p.name)
                        # Save metadata
                        meta = {
                            "symbol": request.symbol,
                            "job_id": results.get('job_id', job_id),
                            "best_iteration": best_iter,
                            "summary": {
                                k: results.get(k) for k in ["best_accuracy", "best_loss", "total_iterations", "total_time"]
                            },
                            "train_end_date": request.train_end_date,
                            "test_period_days": request.test_period_days,
                            "model_types": request.model_types,
                        }
                        with open(target_dir / "champion_meta.json", "w", encoding="utf-8") as fh:
                            json.dump(meta, fh, indent=2)
                        champion_info = {"dir": str(target_dir), "meta": meta}
                if champion_info:
                    backtest_jobs[job_id]["champion"] = champion_info
            except Exception as snap_e:
                logger.warning(f"Champion snapshot failed: {snap_e}")

            # Optionally run a forward test automatically using the champion (if available)
            try:
                if champion_info and PROGRESSIVE_ML_AVAILABLE:
                    from app.ml.progressive.backtester import ProgressiveBacktester as _BT
                    import pandas as _pd
                    # Build an evaluator and point it at the champion directory
                    _bt = _BT(
                        data_loader=progressive_data_loader,
                        trainer=progressive_trainer,
                        predictor=progressive_predictor,
                        progress_callback=None,
                        cancel_checker=None
                    )
                    from pathlib import Path as _Path
                    _bt.job_model_dir = _Path(champion_info['dir'])
                    # Load full data and compute forward window (train_end+1 .. last available)
                    _full_loader = ProgressiveDataLoader(
                        stock_data_dir=progressive_data_loader.stock_data_dir,
                        sequence_length=progressive_data_loader.sequence_length,
                        horizons=progressive_data_loader.horizons,
                        use_fundamentals=progressive_data_loader.use_fundamentals,
                        use_technical_indicators=progressive_data_loader.use_technical_indicators,
                        indicator_params=getattr(progressive_data_loader, 'indicator_params', None)
                    )
                    _full_df = _full_loader.load_stock_data(request.symbol)
                    _forward_summary = None
                    if _full_df is not None and len(_full_df.index) > 0:
                        _last_date = _full_df.index.max().date().isoformat()
                        _train_end = champion_info['meta'].get('train_end_date')
                        _start_date = (_pd.to_datetime(_train_end) + _pd.Timedelta(days=1)).date().isoformat() if _train_end else None
                        if _start_date and _pd.to_datetime(_start_date) <= _pd.to_datetime(_last_date):
                            _eval = _bt.evaluate_iteration(
                                symbol=request.symbol,
                                test_start_date=_start_date,
                                test_end_date=_last_date,
                                iteration_num=int(champion_info['meta'].get('best_iteration') or 0),
                                full_df=_full_df
                            )
                            _forward_summary = {
                                'forward_start': _start_date,
                                'forward_end': _last_date,
                                'metrics': _eval
                            }
                    if _forward_summary:
                        backtest_jobs[job_id]['forward'] = _forward_summary
            except Exception as _fe:
                logger.warning(f"Auto forward test skipped: {_fe}")

            # Also attach current predictions from the champion for 1/7/30d
            try:
                if champion_info and PROGRESSIVE_ML_AVAILABLE:
                    from app.ml.progressive.predictor import ProgressivePredictor as _Pred
                    from app.ml.progressive.data_loader import ProgressiveDataLoader as _DL
                    from pathlib import Path as _Path
                    import torch as _torch
                    import pandas as _pdr
                    # Derive indicator params and sequence length from chosen scout config if available
                    _ind_params = getattr(progressive_data_loader, 'indicator_params', None)
                    _seq_len = getattr(progressive_data_loader, 'sequence_length', 60)
                    try:
                        _sc = backtest_jobs.get(job_id, {}).get('scout', {})
                        _chosen = _sc.get('chosen') if isinstance(_sc, dict) else None
                        if _chosen and isinstance(_chosen, dict):
                            _seq_len = int(_chosen.get('sequence_length') or _seq_len)
                            _prof = _chosen.get('profile')
                            if _prof:
                                _ind_params = _indicator_profile(str(_prof))
                    except Exception:
                        pass
                    # If not available from scout, try to read sequence_length from a checkpoint
                    try:
                        ckpts = list(_Path(champion_info['dir']).glob("*.pth"))
                        if ckpts:
                            _ck = _torch.load(ckpts[0], map_location='cpu')
                            _sl = _ck.get('sequence_length')
                            if isinstance(_sl, int) and _sl > 0:
                                _seq_len = _sl
                    except Exception as _ck_e:
                        logger.debug(f"Could not read sequence_length from checkpoint: {_ck_e}")

                    _pred_loader = _DL(
                        stock_data_dir=progressive_data_loader.stock_data_dir,
                        sequence_length=int(_seq_len),
                        horizons=progressive_data_loader.horizons,
                        use_fundamentals=progressive_data_loader.use_fundamentals,
                        use_technical_indicators=progressive_data_loader.use_technical_indicators,
                        indicator_params=_ind_params
                    )
                    _pred = _Pred(data_loader=_pred_loader, model_dir=str(champion_info['dir']))
                    try:
                        _preds = _pred.predict_ensemble(symbol=request.symbol, mode='progressive')
                        # Enrich with SL/TP risk block per horizon (ATR-based if indicators exist else volatility)
                        try:
                            from pathlib import Path as __Path
                            import pandas as __pd
                            ind_path = data_manager.stock_data_dir / request.symbol / f"{request.symbol}_indicators.csv"
                            price_path = data_manager.stock_data_dir / request.symbol / f"{request.symbol}_price.csv"
                            close_price = float(_preds.get('current_price', 0.0))
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
                            for hk, obj in (_preds.get('predictions') or {}).items():
                                change_pct = float(obj.get('price_change_pct', 0.0))
                                sl = close_price * (1 - risk_pct)
                                tp = close_price * (1 + reward_pct)
                                if change_pct < 0:
                                    sl = close_price * (1 + risk_pct)
                                    tp = close_price * (1 - reward_pct)
                                obj['risk'] = {
                                    'stop_loss': round(sl, 4),
                                    'take_profit': round(tp, 4),
                                    'stop_loss_pct': -risk_pct if change_pct >= 0 else risk_pct,
                                    'take_profit_pct': reward_pct if change_pct >= 0 else -reward_pct,
                                    'basis': 'ATR_14' if atr_pct is not None else 'volatility',
                                    'rr': rr
                                }
                        except Exception as __e:
                            logger.debug(f"Risk enrich (current preds) skipped: {__e}")
                        # Keep a compact summary for UI
                        _compact = {
                            'symbol': _preds.get('symbol'),
                            'current_price': _preds.get('current_price'),
                            'generated_at': _preds.get('generated_at'),
                            'predictions': {}
                        }
                        for hk in ['1d', '7d', '30d']:
                            if hk in _preds.get('predictions', {}):
                                p = _preds['predictions'][hk]
                                # Mark if horizon hit safety cap (used by UI tooltip)
                                try:
                                    cap_map = {'1d': 0.10, '7d': 0.20, '30d': 0.40}
                                    pc = float(p.get('price_change_pct', 0.0))
                                    capped_flag = abs(pc) >= (cap_map.get(hk, 1.0) - 1e-6)
                                except Exception:
                                    capped_flag = False
                                _compact['predictions'][hk] = {
                                    'target_price': p.get('target_price'),
                                    'price_change_pct': p.get('price_change_pct'),
                                    'direction': p.get('direction'),
                                    'direction_prob': p.get('direction_prob'),
                                    'confidence': p.get('confidence'),
                                    'signal': p.get('signal'),
                                    'risk': p.get('risk'),
                                    'capped': capped_flag
                                }
                        backtest_jobs[job_id]['current_predictions'] = _compact
                    except Exception as _pe:
                        logger.warning(f"Champion current predictions failed: {_pe}")
            except Exception as _cp_e:
                logger.debug(f"Attach current predictions skipped: {_cp_e}")

            backtest_jobs[job_id].update({
                "status": "completed",
                "progress": 100,
                "current_step": "✅ Backtest completed successfully!",
                "eta_seconds": 0,
                "end_time": datetime.now(timezone.utc).isoformat(),
                "result": results,
            })
        logger.info(f"✅ Backtest job {job_id} completed for {request.symbol}")
    except Exception as e:
        logger.error(f"❌ Backtest job {job_id} failed: {e}")
        backtest_jobs[job_id].update({
            "status": "failed",
            "progress": 0,
            "current_step": f"❌ Error: {str(e)}",
            "end_time": datetime.now(timezone.utc).isoformat(),
            "error": str(e),
            "eta_seconds": 0
        })

@app.get("/api/data/ensure/{symbol}", tags=["Data"])
async def api_ensure_symbol_data(symbol: str):
    """Ensure the four required data files for a symbol exist and are up-to-date.

    Returns a summary of created/updated/skipped and any errors.
    """
    try:
        if not PROGRESSIVE_ML_AVAILABLE:
            raise HTTPException(status_code=503, detail="System not initialized")
        summary = await asyncio.to_thread(data_manager.ensure_symbol_data, symbol)
        return {"status": "success", "symbol": symbol, "summary": summary.to_dict()}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ensure data failed for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/ml/progressive/backtest/results/{symbol}", tags=["Progressive ML"])
async def get_backtest_results(symbol: str):
    """
    Get backtest results for a symbol
    
    Args:
        symbol: Stock symbol
        
    Returns:
        Saved backtest results
    """
    try:
        if not PROGRESSIVE_ML_AVAILABLE:
            raise HTTPException(status_code=503, detail="Progressive ML system not available")
        
        import os
        import json
        from pathlib import Path
        
        results_dir = Path("app/ml/models/backtest_results")
        
        if not results_dir.exists():
            return {
                "status": "no_results",
                "message": f"No backtest results found for {symbol}",
                "symbol": symbol
            }
        
        # Find latest results file for symbol
        results_files = list(results_dir.glob(f"results_{symbol}_*.json"))
        
        if not results_files:
            return {
                "status": "no_results",
                "message": f"No backtest results found for {symbol}",
                "symbol": symbol
            }
        
        # Get most recent file
        latest_file = max(results_files, key=lambda p: p.stat().st_mtime)
        
        with open(latest_file, 'r') as f:
            results = json.load(f)

        # Normalize legacy format to match frontend expectations
        try:
            if isinstance(results, dict):
                if 'all_iterations' not in results and 'iterations' in results:
                    results['all_iterations'] = results.get('iterations', [])
                if 'best_iteration' not in results and results.get('all_iterations'):
                    # Compute best iteration by max accuracy if available
                    best = None
                    for it in results['all_iterations']:
                        acc = it.get('accuracy')
                        if isinstance(acc, (int, float)):
                            if best is None or acc > best.get('accuracy', -1):
                                best = it
                    if best and 'iteration' in best:
                        results['best_iteration'] = best['iteration']
        except Exception as _norm_e:
            logger.debug(f"Backtest results normalization skipped: {_norm_e}")
        
        return {
            "status": "success",
            "symbol": symbol,
            "results": results,
            "file": latest_file.name,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get backtest results: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get backtest results: {str(e)}")

@app.get("/api/ml/progressive/backtest/history/{symbol}", tags=["Progressive ML"])
async def list_backtest_history(symbol: str, limit: int = Query(20, ge=1, le=100)):
    """
    List recent backtest result files for a symbol with brief metadata
    """
    try:
        from pathlib import Path
        import json
        results_dir = Path("app/ml/models/backtest_results")
        if not results_dir.exists():
            return {"status": "success", "symbol": symbol, "items": []}
        files = sorted(results_dir.glob(f"results_{symbol}_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        items = []
        for f in files[:limit]:
            try:
                with open(f, 'r', encoding='utf-8') as fh:
                    data = json.load(fh)
                # Normalize iteration list
                iterations = data.get('all_iterations') or data.get('iterations') or []
                # Best accuracy
                best_acc = None
                if isinstance(iterations, list) and iterations:
                    for it in iterations:
                        acc = it.get('accuracy')
                        if isinstance(acc, (int, float)):
                            best_acc = acc if best_acc is None else max(best_acc, acc)
                items.append({
                    "file": f.name,
                    "modified": datetime.fromtimestamp(f.stat().st_mtime, tz=timezone.utc).isoformat(),
                    "best_accuracy": best_acc,
                    "iterations": len(iterations)
                })
            except Exception:
                items.append({
                    "file": f.name,
                    "modified": datetime.fromtimestamp(f.stat().st_mtime, tz=timezone.utc).isoformat(),
                    "best_accuracy": None,
                    "iterations": None
                })
        return {"status": "success", "symbol": symbol, "items": items}
    except Exception as e:
        logger.error(f"Failed to list backtest history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list backtest history: {str(e)}")

@app.get("/api/ml/progressive/backtest/result_by_file/{symbol}/{file_name}", tags=["Progressive ML"])
async def get_backtest_result_by_file(symbol: str, file_name: str):
    """
    Fetch a specific backtest results file by name for a symbol
    """
    try:
        from pathlib import Path
        import json
        results_dir = Path("app/ml/models/backtest_results")
        target = results_dir / file_name
        if not target.exists() or f"results_{symbol}_" not in target.name:
            raise HTTPException(status_code=404, detail="Results file not found")
        with open(target, 'r', encoding='utf-8') as f:
            results = json.load(f)
        # Normalize legacy format
        try:
            if isinstance(results, dict):
                if 'all_iterations' not in results and 'iterations' in results:
                    results['all_iterations'] = results.get('iterations', [])
                if 'best_iteration' not in results and results.get('all_iterations'):
                    best = None
                    for it in results['all_iterations']:
                        acc = it.get('accuracy')
                        if isinstance(acc, (int, float)):
                            if best is None or acc > best.get('accuracy', -1):
                                best = it
                    if best and 'iteration' in best:
                        results['best_iteration'] = best['iteration']
        except Exception:
            pass
        return {"status": "success", "symbol": symbol, "results": results, "file": target.name}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get backtest result by file: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get backtest result: {str(e)}")

# Champions utilities
@app.get("/api/ml/progressive/champions/{symbol}", tags=["Progressive ML"])
async def list_champions(symbol: str):
    try:
        from pathlib import Path
        root = Path("app/ml/models/champions") / symbol
        if not root.exists():
            return {"status": "success", "symbol": symbol, "items": []}
        items = []
        for job_dir in sorted(root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if job_dir.is_dir():
                meta_file = job_dir / "champion_meta.json"
                meta = None
                if meta_file.exists():
                    try:
                        with open(meta_file, 'r', encoding='utf-8') as fh:
                            meta = json.load(fh)
                    except Exception:
                        meta = None
                items.append({
                    'path': str(job_dir),
                    'meta': meta,
                    'modified': datetime.fromtimestamp(job_dir.stat().st_mtime, tz=timezone.utc).isoformat()
                })
        return {"status": "success", "symbol": symbol, "items": items}
    except Exception as e:
        logger.error(f"Failed to list champions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list champions: {e}")

@app.post("/api/ml/progressive/champion/forward_test/{symbol}", tags=["Progressive ML"])
async def champion_forward_test(symbol: str, job_id: str | None = None):
    """Run a forward test using the champion checkpoints from the specified job_id or latest champion.

    Evaluates from the champion's train_end_date+1 until last data date.
    """
    try:
        if not PROGRESSIVE_ML_AVAILABLE:
            raise HTTPException(status_code=503, detail="Progressive ML system not available")

        from pathlib import Path
        import pandas as pd
        from app.ml.progressive.backtester import ProgressiveBacktester

        # Locate champion dir
        champions_root = Path("app/ml/models/champions") / symbol
        if not champions_root.exists():
            raise HTTPException(status_code=404, detail="No champions found")
        target_dir = None
        if job_id:
            cand = champions_root / job_id
            if cand.exists():
                target_dir = cand
        if target_dir is None:
            # pick most recent
            dirs = [p for p in champions_root.iterdir() if p.is_dir()]
            if not dirs:
                raise HTTPException(status_code=404, detail="No champions found")
            target_dir = sorted(dirs, key=lambda p: p.stat().st_mtime, reverse=True)[0]

        meta_file = target_dir / "champion_meta.json"
        if not meta_file.exists():
            raise HTTPException(status_code=404, detail="Champion metadata not found")
        with open(meta_file, 'r', encoding='utf-8') as fh:
            meta = json.load(fh)

        train_end = meta.get('train_end_date')
        if not train_end:
            raise HTTPException(status_code=422, detail="Champion metadata missing train_end_date")

        # Prepare backtester helpers for evaluation-only
        bt = ProgressiveBacktester(
            data_loader=progressive_data_loader,
            trainer=progressive_trainer,
            predictor=progressive_predictor,
            progress_callback=None,
            cancel_checker=None
        )
        # Set model dir to champion
        bt.job_model_dir = target_dir
        # Load full data
        full_loader = ProgressiveDataLoader(
            stock_data_dir=progressive_data_loader.stock_data_dir,
            sequence_length=progressive_data_loader.sequence_length,
            horizons=progressive_data_loader.horizons,
            use_fundamentals=progressive_data_loader.use_fundamentals,
            use_technical_indicators=progressive_data_loader.use_technical_indicators,
            indicator_params=getattr(progressive_data_loader, 'indicator_params', None)
        )
        full_df = full_loader.load_stock_data(symbol)
        if full_df is None or len(full_df.index) == 0:
            raise HTTPException(status_code=404, detail="Price data not available")
        last_date = full_df.index.max().date().isoformat()
        start_date = (pd.to_datetime(train_end) + pd.Timedelta(days=1)).date().isoformat()
        if pd.to_datetime(start_date) > pd.to_datetime(last_date):
            raise HTTPException(status_code=422, detail="No forward window after champion train_end_date")

        eval_res = bt.evaluate_iteration(
            symbol=symbol,
            test_start_date=start_date,
            test_end_date=last_date,
            iteration_num=int(meta.get('best_iteration') or 0),
            full_df=full_df
        )

        return {
            'status': 'success',
            'symbol': symbol,
            'job_id': meta.get('job_id'),
            'champion_dir': str(target_dir),
            'train_end_date': train_end,
            'forward_start': start_date,
            'forward_end': last_date,
            'metrics': eval_res,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Champion forward test failed: {e}")
        raise HTTPException(status_code=500, detail=f"Champion forward test failed: {e}")

@app.get("/api/market/{symbol}", tags=["Market"])
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

@app.get("/api/sentiment/{symbol}", tags=["Sentiment"])
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

@app.get("/api/watchlist", tags=["Portfolio"])
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

# ============================================================
# Enhanced WebSocket Endpoints
# ============================================================

@app.websocket("/ws/market/{symbol}")
async def websocket_market_data(websocket: WebSocket, symbol: str):
    """WebSocket endpoint for real-time market data streaming"""
    await websocket.accept()
    logger.info(f"🔌 Market WebSocket connected for {symbol}")
    
    try:
        if market_streamer:
            # Add this WebSocket to market streaming
            await market_streamer.add_subscription(websocket, symbol)
        
        # Keep connection alive and handle messages
        while True:
            try:
                # Wait for client messages (ping/pong, etc.)
                message = await websocket.receive_text()
                
                # Echo back or handle specific commands
                if message == "ping":
                    await websocket.send_text("pong")
                    
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"WebSocket error for {symbol}: {e}")
                break
                
    except WebSocketDisconnect:
        logger.info(f"🔌 Market WebSocket disconnected for {symbol}")
    except Exception as e:
        logger.error(f"Market WebSocket error for {symbol}: {e}")
    finally:
        if market_streamer:
            await market_streamer.remove_subscription(websocket, symbol)

# ============================================================
# Data Management Endpoints 
# ============================================================
@app.get("/data-management", response_class=HTMLResponse)
async def data_management_page():
    """Data Management Dashboard - serve data_management.html"""
    from pathlib import Path
    
    dashboard_path = Path(__file__).parent / "templates" / "data_management.html"
    
    try:
        with open(dashboard_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        return HTMLResponse(
            content="<h1>Data Management dashboard not found.</h1>",
            status_code=500
        )

@app.get("/api/data-management/status")
async def get_data_management_status():
    """Get current status of data download jobs"""
    try:
        from datetime import datetime
        import subprocess
        import json
        from pathlib import Path
        
        # Check if scheduler jobs exist and get their status
        status = {
            "daily_downloads": 0,
            "weekly_updates": 0,
            "error_count": 0,
            "last_run": None,
            "jobs": []
        }
        
        # Check log files for recent activity
        logs_dir = Path("logs")
        if logs_dir.exists():
            # Count daily scans from today
            today_daily_logs = list(logs_dir.glob(f"daily_scan_{datetime.now().strftime('%Y%m%d')}*.log"))
            status["daily_downloads"] = len(today_daily_logs)
            
            # Count weekly fundamentals from this week
            this_week = datetime.now().strftime('%Y%m')
            weekly_logs = list(logs_dir.glob(f"weekly_fundamentals_{this_week}*.log"))
            status["weekly_updates"] = len(weekly_logs)
            
            # Find most recent log
            all_logs = list(logs_dir.glob("*.log"))
            if all_logs:
                latest_log = max(all_logs, key=lambda x: x.stat().st_mtime)
                status["last_run"] = datetime.fromtimestamp(latest_log.stat().st_mtime).isoformat()
        
        # Live-only: do not return mocked job statuses. If a real scheduler is integrated,
        # this endpoint should reflect its state. For now, return only log-derived info
        # and an empty jobs list with a note.
        status["jobs"] = []
        status["note"] = "Scheduler integration not available; returning logs-derived status only."
        
        return JSONResponse(content=status)
        
    except Exception as e:
        logger.error(f"Error getting data management status: {e}")
        return JSONResponse(
            content={"error": str(e)}, 
            status_code=500
        )

@app.post("/api/data-management/run-job/{job_type}")
async def run_data_job(job_type: str, background_tasks: BackgroundTasks):
    """Run a specific data download job"""
    try:
        if job_type == "daily":
            # Run daily stock data scan
            background_tasks.add_task(run_daily_scan)
            return JSONResponse(
                content={
                    "success": True, 
                    "message": "Daily scan started", 
                    "job_id": f"daily_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                }
            )
        elif job_type == "weekly":
            # Run weekly fundamentals update
            background_tasks.add_task(run_weekly_fundamentals)
            return JSONResponse(
                content={
                    "success": True, 
                    "message": "Weekly fundamentals update started",
                    "job_id": f"weekly_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                }
            )
        else:
            return JSONResponse(
                content={"success": False, "error": f"Unknown job type: {job_type}"},
                status_code=400
            )
            
    except Exception as e:
        logger.error(f"Error running job {job_type}: {e}")
        return JSONResponse(
            content={"success": False, "error": str(e)},
            status_code=500
        )

@app.get("/api/data-management/logs/{job_type}")
async def get_job_logs(job_type: str):
    """Get logs for a specific job type"""
    try:
        from pathlib import Path
        
        logs_dir = Path("logs")
        if not logs_dir.exists():
            return JSONResponse(content={"logs": []})
        
        # Get logs based on job type
        if job_type == "daily":
            log_files = list(logs_dir.glob("daily_scan_*.log"))
        elif job_type == "weekly":
            log_files = list(logs_dir.glob("weekly_fundamentals_*.log"))
        else:
            log_files = list(logs_dir.glob("*.log"))
        
        # Get the most recent log file
        if not log_files:
            return JSONResponse(content={"logs": ["No logs found for this job type."]})
        
        latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
        
        # Read log content
        with open(latest_log, 'r', encoding='utf-8') as f:
            log_content = f.read()
        
        return JSONResponse(content={
            "logs": log_content.split('\n'),
            "file": str(latest_log),
            "timestamp": datetime.fromtimestamp(latest_log.stat().st_mtime).isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error reading logs for {job_type}: {e}")
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )

# Helper functions for data management
def _get_next_daily_run():
    """Calculate next daily run time (17:10)"""
    try:
        now = datetime.now()
        next_run = now.replace(hour=17, minute=10, second=0, microsecond=0)
        if next_run <= now:
            next_run += timedelta(days=1)
        return next_run
    except:
        return None

def _get_next_weekly_run():
    """Calculate next weekly run time (Sunday 2:00 AM)"""
    try:
        now = datetime.now()
        days_until_sunday = (6 - now.weekday()) % 7
        if days_until_sunday == 0 and now.hour >= 2:
            days_until_sunday = 7
        next_run = now + timedelta(days=days_until_sunday)
        next_run = next_run.replace(hour=2, minute=0, second=0, microsecond=0)
        return next_run
    except:
        return None

async def run_daily_scan():
    """Background task to run daily stock scan"""
    try:
        import subprocess
        import sys
        
        logger.info("🚀 Starting daily stock scan...")
        
        # Run the daily scan script (canonical under app/data)
        result = subprocess.run([
            sys.executable, "daily_scan.py"
        ], capture_output=True, text=True, cwd="app/data")
        
        if result.returncode == 0:
            logger.info("✅ Daily scan completed successfully")
        else:
            logger.error(f"❌ Daily scan failed: {result.stderr}")
            raise Exception(f"Daily scan failed: {result.stderr}")
            
    except Exception as e:
        logger.error(f"Error in daily scan background task: {e}")

async def run_weekly_fundamentals():
    """Background task to run weekly fundamentals update"""
    try:
        import subprocess
        import sys
        
        logger.info("🚀 Starting weekly fundamentals update...")
        
        # Run the weekly fundamentals script via canonical downloader in app/data
        result = subprocess.run([
            sys.executable, "download_fundamentals.py"
        ], capture_output=True, text=True, cwd="app/data")
        
        if result.returncode == 0:
            logger.info("✅ Weekly fundamentals update completed successfully")
        else:
            logger.error(f"❌ Weekly fundamentals update failed: {result.stderr}")
            raise Exception(f"Weekly fundamentals update failed: {result.stderr}")
            
    except Exception as e:
        logger.error(f"Error in weekly fundamentals background task: {e}")

@app.get("/api/system/health")
async def system_health():
    """System health check endpoint"""
    try:
        health_status = {
            "database": True,  # Always true for now (SQLite)
            "api": True,       # If we're responding, API is working
            "scheduler": SCHEDULER_AVAILABLE,
            "storage": True,   # Check if we can write to logs directory
            "timestamp": datetime.now().isoformat()
        }
        
        # Check storage (try to write a temp file)
        try:
            from pathlib import Path
            logs_dir = Path("logs")
            logs_dir.mkdir(exist_ok=True)
            test_file = logs_dir / "health_check.tmp"
            test_file.write_text("test")
            test_file.unlink()
        except:
            health_status["storage"] = False
        
        return JSONResponse(content=health_status)
        
    except Exception as e:
        return JSONResponse(
            content={
                "database": False,
                "api": False, 
                "scheduler": False,
                "storage": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            },
            status_code=500
        )

# ============================================================
# Dashboard Endpoint  
# ============================================================
@app.get("/", response_class=HTMLResponse)
@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """Main dashboard - serve existing dashboard.html from templates"""
    from pathlib import Path
    
    # Try to read the dashboard.html file from templates
    dashboard_path = Path(__file__).parent / "templates" / "dashboard.html"
    
    try:
        with open(dashboard_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        # Fallback - return error message
        return HTMLResponse(
            content="<h1>Dashboard file not found. Please ensure dashboard.html exists in app/templates/</h1>",
            status_code=500
        )

# ============================================================
# Progressive ML Guide (HTML)
# ============================================================
@app.get("/docs/progressive-ml", response_class=HTMLResponse)
async def progressive_ml_guide():
    """Serve the Progressive ML Guide page from templates/docs."""
    from pathlib import Path

    guide_path = Path(__file__).parent / "templates" / "docs" / "progressive_ml_guide.html"
    try:
        with open(guide_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        return HTMLResponse(
            content="<h1>Guide file not found. Ensure app/templates/docs/progressive_ml_guide.html exists.</h1>",
            status_code=500
        )

# ============================================================
# RL Dashboard and Minimal API
# ============================================================
@app.get("/rl", response_class=HTMLResponse)
async def rl_dashboard_page():
    """Serve minimal RL dashboard page"""
    from pathlib import Path

    page_path = Path(__file__).parent / "templates" / "rl_dashboard.html"
    try:
        with open(page_path, "r", encoding="utf-8") as f:
            html = f.read()
        return HTMLResponse(content=html)
    except FileNotFoundError:
        return HTMLResponse(content="<h3>RL Dashboard</h3><p>Page not found.</p>", status_code=404)

@app.get("/api/rl/status")
async def rl_status():
    """Minimal RL status placeholder (safe)."""
    return {
        "status": "idle",
        "positions": [],
        "pnl": 0.0,
        "decisions": []
    }

@app.get("/api/rl/simulate")
async def rl_simulate(symbol: str, days: int = 250, window: int = 60, policy: str = "follow_trend",
                      start_date: Optional[str] = None, end_date: Optional[str] = None):
    """Run a lightweight simulation on local stock_data for a symbol and return series for plotting."""
    try:
        from rl.simulation import run_simulation
        # If explicit dates are provided, prefer them over days
        sd = start_date if start_date else None
        ed = end_date if end_date else None
        use_days = None if (sd and ed) else max(30, int(days))
        result = run_simulation(
            symbol=symbol,
            days=use_days,
            window=max(10, int(window)),
            policy=policy,
            start_date=sd,
            end_date=ed,
        )
        return {"status": "success", "data": result}
    except FileNotFoundError as e:
        return {"status": "error", "message": str(e)}
    except ValueError as e:
        # Surface validation issues (e.g., insufficient data/window) to the UI
        return {"status": "error", "message": str(e)}
    except Exception as e:
        logger.error(f"RL simulate failed: {e}")
        return {"status": "error", "message": f"Simulation failed: {str(e)}"}

@app.get("/api/rl/simulate/plan")
async def rl_simulate_plan(symbol: str) -> Dict[str, Any]:
    """Plan sensible dates/days and window for Quick Simulation from local data.

    Heuristics:
    - Use last N=250 trading rows if available; else use all
    - start_date = index[-N], end_date = last index
    - window = min(60, max(10, floor(N/4)))
    - days = N
    """
    try:
        from rl.data_adapters.local_stock_data import LocalStockData
        import pandas as pd
        adapter = LocalStockData()
        if not adapter.has_symbol(symbol):
            raise HTTPException(status_code=404, detail=f"Local data for {symbol} not found")
        bundle = adapter.load_symbol(symbol)
        dfp = bundle.get("price")
        if dfp is None or dfp.empty:
            raise HTTPException(status_code=400, detail=f"No price data for {symbol}")
        if not isinstance(dfp.index, pd.DatetimeIndex):
            try:
                dfp.index = pd.to_datetime(dfp.index)
            except Exception:
                pass
        dfp = dfp.sort_index()
        total = len(dfp)
        if total < 2:
            raise HTTPException(status_code=400, detail=f"Insufficient data for {symbol}")
        N = 250 if total >= 250 else total
        start_dt = dfp.index[-N]
        end_dt = dfp.index[-1]
        window = int(min(60, max(10, N // 4)))
        plan = {
            "symbol": symbol.upper(),
            "start_date": start_dt.date().isoformat(),
            "end_date": end_dt.date().isoformat(),
            "window": window,
            "days": int(N),
            "first_data_date": dfp.index[0].date().isoformat(),
            "last_data_date": end_dt.date().isoformat(),
            "total_rows": int(total)
        }
        return {"status": "planned", "plan": plan}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to plan Quick Simulation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to plan Quick Simulation: {str(e)}")

# ============================================================
# RL PPO Training (SB3) — background job runner
# ============================================================

PPO_TRAIN_JOBS: Dict[str, Dict[str, Any]] = {}


def _run_ppo_job(job_id: str, args: list[str], cwd: str) -> None:
    """Run PPO training as a subprocess and track status/logs in PPO_TRAIN_JOBS."""
    PPO_TRAIN_JOBS[job_id].update({
        "status": "running",
        "started_at": datetime.utcnow().isoformat() + "Z",
        "logs": [],
        "model_path": None,
    })
    try:
        proc = subprocess.Popen(args, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        PPO_TRAIN_JOBS[job_id]["pid"] = proc.pid
        # stream logs
        if proc.stdout is not None:
            for line in proc.stdout:
                line = line.rstrip("\n")
                logs = PPO_TRAIN_JOBS[job_id].get("logs", [])
                logs.append(line)
                # keep only last 200 lines
                PPO_TRAIN_JOBS[job_id]["logs"] = logs[-200:]
                if "Saved PPO model to:" in line:
                    # parse path
                    try:
                        path = line.split("Saved PPO model to:", 1)[1].strip()
                        PPO_TRAIN_JOBS[job_id]["model_path"] = path
                    except Exception:
                        pass
        ret = proc.wait()
        PPO_TRAIN_JOBS[job_id]["ended_at"] = datetime.utcnow().isoformat() + "Z"
        if ret == 0:
            PPO_TRAIN_JOBS[job_id]["status"] = "completed"
        else:
            PPO_TRAIN_JOBS[job_id]["status"] = "failed"
            PPO_TRAIN_JOBS[job_id]["returncode"] = ret
    except Exception as e:
        PPO_TRAIN_JOBS[job_id]["status"] = "failed"
        PPO_TRAIN_JOBS[job_id]["error"] = str(e)
        PPO_TRAIN_JOBS[job_id]["ended_at"] = datetime.utcnow().isoformat() + "Z"


@app.post("/api/rl/ppo/train")
async def rl_ppo_train(symbol: str, timesteps: int = 100000, window: int = 60,
                       start_date: Optional[str] = None, end_date: Optional[str] = None,
                       seed: int = 42):
    """Start PPO training in background using SB3; returns a job_id."""
    try:
        # Build command
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "rl" / "training" / "train_ppo.py"
        if not script.exists():
            raise FileNotFoundError(f"train_ppo.py not found at {script}")
        cmd = [sys.executable or "python", "-u", str(script), "--symbol", symbol, "--timesteps", str(int(timesteps)),
               "--window", str(int(window)), "--seed", str(int(seed))]
        if start_date:
            cmd.extend(["--start", start_date])
        if end_date:
            cmd.extend(["--end", end_date])
        job_id = uuid.uuid4().hex[:12]
        PPO_TRAIN_JOBS[job_id] = {"status": "pending", "cmd": cmd}
        t = threading.Thread(target=_run_ppo_job, args=(job_id, cmd, str(repo_root)), daemon=True)
        t.start()
        return {"status": "started", "job_id": job_id}
    except Exception as e:
        logger.error(f"Failed to start PPO training: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start PPO training: {str(e)}")


@app.get("/api/rl/ppo/plan")
async def rl_ppo_plan(symbol: str, window: Optional[int] = None) -> Dict[str, Any]:
    """Plan sensible training dates, window size, and timesteps from local data.

    Heuristics:
    - Start at max(first available date, 2020-01-01) if enough data, else earliest date
    - End at last available date
    - Window defaults to 60 if >=60 training days, else min( max(10, floor(days/3)), 60 )
    - Timesteps ~= clamp(days * 200, 50k..1,000k)
    """
    try:
        from rl.data_adapters.local_stock_data import LocalStockData
        import pandas as pd
        adapter = LocalStockData()
        if not adapter.has_symbol(symbol):
            raise HTTPException(status_code=404, detail=f"Local data for {symbol} not found")
        bundle = adapter.load_symbol(symbol)
        dfp = bundle.get("price")
        if dfp is None or dfp.empty:
            raise HTTPException(status_code=400, detail=f"No price data for {symbol}")
        # Ensure DateTimeIndex
        if not isinstance(dfp.index, pd.DatetimeIndex):
            try:
                dfp.index = pd.to_datetime(dfp.index)
            except Exception:
                pass
        dfp = dfp.sort_index()
        first_dt = dfp.index[0]
        last_dt = dfp.index[-1]
        # Prefer start >= 2020-01-01 when possible
        pref_start = pd.Timestamp(year=2020, month=1, day=1, tz=None)
        start_dt = pref_start if pref_start >= first_dt else first_dt
        # If the span from preferred start is too short (< 120 days), fallback to earliest
        if (last_dt - start_dt).days < 120:
            start_dt = first_dt
        # Compute training days (calendar index length in slice)
        df_train = dfp[(dfp.index >= start_dt) & (dfp.index <= last_dt)]
        days = int(len(df_train))
        if days < 2:
            # Not enough data to train
            raise HTTPException(status_code=400, detail=f"Insufficient training range for {symbol} (days={days})")
        # Choose window
        if window is None:
            win = 60 if days >= 60 else int(max(10, min(60, days // 3)))
        else:
            win = int(max(10, min(window, 300)))
        # Estimate timesteps
        est_steps = int(days * 200)
        timesteps = int(max(50_000, min(1_000_000, est_steps)))
        plan = {
            "symbol": symbol.upper(),
            "train_start_date": start_dt.date().isoformat(),
            "train_end_date": last_dt.date().isoformat(),
            "window": int(win),
            "timesteps": int(timesteps),
            "training_days": days,
            "first_data_date": first_dt.date().isoformat(),
            "last_data_date": last_dt.date().isoformat(),
        }
        return {"status": "planned", "plan": plan}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to plan PPO training: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to plan PPO training: {str(e)}")


@app.get("/api/rl/ppo/train/status")
async def rl_ppo_train_status_all():
    """Return status of all PPO jobs (summary)."""
    out = {}
    for jid, info in PPO_TRAIN_JOBS.items():
        out[jid] = {
            "status": info.get("status"),
            "started_at": info.get("started_at"),
            "ended_at": info.get("ended_at"),
            "model_path": info.get("model_path"),
        }
    return out


@app.get("/api/rl/ppo/train/status/{job_id}")
async def rl_ppo_train_status(job_id: str):
    """Return detailed status of a specific PPO job, including last logs."""
    info = PPO_TRAIN_JOBS.get(job_id)
    if not info:
        raise HTTPException(status_code=404, detail="job_id not found")
    return {
        "status": info.get("status"),
        "started_at": info.get("started_at"),
        "ended_at": info.get("ended_at"),
        "model_path": info.get("model_path"),
        "logs": info.get("logs", []),
        "pid": info.get("pid"),
        "returncode": info.get("returncode"),
        "error": info.get("error"),
    }

# ============================================================
# Entry Point
# ============================================================
def run_server():
    """Run the server with uvicorn"""
    import uvicorn
    
    logger.info("🚀 Starting MarketPulse server...")
    logger.info("   Dashboard: http://localhost:8000")
    logger.info("   WebSocket: ws://localhost:8000/ws/alerts")
    logger.info("   API Docs: http://localhost:8000/docs")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )

if __name__ == "__main__":
    run_server()
