"""
Scanner Router - Stock Filtering, Training & Scanning
Migrated from main_realtime.py to keep the main file clean
"""
import logging
import asyncio
import threading
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import json

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, BackgroundTasks

# Set up logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/scanner", tags=["Scanner"])

# ============================================================
# Scanner State Management
# ============================================================
class ScannerState:
    """Global state for scanner operations"""
    def __init__(self):
        self.top: List[Dict[str, Any]] = []
        self.filter_state: str = 'idle'
        self.filter_progress: float = 0.0
        self.total_symbols: int = 0
        self.processed_symbols: int = 0
        self.passed_count: int = 0
        self.trained_count: int = 0
        self.filter_items: List[Dict[str, Any]] = []
        self.train_status: str = 'idle'
        self.train_current_symbol: Optional[str] = None
        self.train_queue: List[str] = []
        self.train_total: int = 0
        self.train_completed: int = 0
        self.train_failed_symbols: Dict[str, str] = {}
        self.train_symbols_status: Dict[str, str] = {}
        self.train_started_at: Optional[float] = None
        self.train_last_persisted_at: Optional[float] = None
    
    def to_status(self) -> Dict[str, Any]:
        """Return scanner status summary"""
        return {
            "state": "idle" if not self.top else "ready",
            "top_count": len(self.top),
            "filter_state": self.filter_state,
            "train_status": self.train_status,
        }

    def reset_training_tracking(self) -> None:
        """Reset all training queue bookkeeping fields."""
        self.train_status = 'idle'
        self.train_current_symbol = None
        self.train_queue = []
        self.train_total = 0
        self.train_completed = 0
        self.train_failed_symbols = {}
        self.train_symbols_status = {}
        self.train_started_at = None
        self.train_last_persisted_at = None

_scanner_state = ScannerState()

# Training queue coordination
_train_queue_lock: Optional[asyncio.Lock] = None
_train_worker_task: Optional[asyncio.Task] = None


def _get_train_queue_lock() -> asyncio.Lock:
    """Lazily create and return the training queue lock."""
    global _train_queue_lock
    if _train_queue_lock is None:
        _train_queue_lock = asyncio.Lock()
    return _train_queue_lock

MICRO_CAP_THRESHOLD = 300_000_000  # $300M

TRAIN_SEQUENCE_LENGTH = 126
TRAIN_HORIZONS = [126]
TRAIN_HISTORY_BUFFER = 10  # Matches loader guard (seq + max horizon + buffer)
MIN_HISTORY_ROWS = TRAIN_SEQUENCE_LENGTH + max(TRAIN_HORIZONS) + TRAIN_HISTORY_BUFFER
TRAIN_MODEL_TYPES = ["transformer"]

# ------------------------------------------------------------
# Model artifacts detection
# NOTE (SCANNER): מודלי 126d נשמרים תחת app/ml/models בתבנית
#   transformer_{SYMBOL}_126d_best.pth
# העמודה "Model" בטבלת הסינון משקפת אך ורק קיום של מודל 126d עבור הסימבול.
# ------------------------------------------------------------
def _model_126d_exists(symbol: str) -> bool:
    """Return True only if a 126d checkpoint exists for the symbol.

    Pattern checked: app/ml/models/transformer_{SYMBOL}_126d_best.pth
    (We intentionally ignore 1d/7d/30d/unified and legacy progressive.pt here
    to ensure UI reflects 126d training status precisely.)
    """
    try:
        project_root = Path(__file__).resolve().parents[3]
        model_dir = project_root / 'app' / 'ml' / 'models'
        if not model_dir.exists():
            return False
        sym = symbol.upper()
        p = model_dir / f"transformer_{sym}_126d_best.pth"
        return p.exists()
    except Exception:
        return False


def _count_price_rows(symbol: str) -> int:
    """Return number of price rows for symbol (excluding CSV header)."""
    try:
        project_root = Path(__file__).resolve().parents[3]
        price_path = project_root / 'stock_data' / symbol / f"{symbol}_price.csv"
        if not price_path.exists():
            return 0

        with open(price_path, 'r', encoding='utf-8') as fh:
            line_count = sum(1 for _ in fh)
        return max(0, line_count - 1)
    except Exception:
        return 0


def _has_minimum_history(symbol: str) -> bool:
    """Return True if the symbol has sufficient samples for 126d training."""
    rows = _count_price_rows(symbol.upper())
    if rows < MIN_HISTORY_ROWS:
        logger.debug(
            "Insufficient history for %s: %s rows (requires %s)",
            symbol, rows, MIN_HISTORY_ROWS
        )
        return False
    return True


def _load_basic_symbol_snapshot(symbol: str) -> Optional[Dict[str, Any]]:
    """Load only the minimal data needed for filter checks.

    Avoids the heavy metrics pipeline so the filter stage stays lightweight.
    """
    try:
        project_root = Path(__file__).resolve().parents[3]
        stock_dir = project_root / 'stock_data' / symbol
        price_path = stock_dir / f"{symbol}_price.csv"

        if not price_path.exists():
            return None

        df = pd.read_csv(price_path)
        if df.empty or 'Close' not in df.columns:
            return None

        close_series = pd.to_numeric(df['Close'], errors='coerce').dropna()
        if close_series.empty:
            return None
        current_price = float(close_series.iloc[-1])

        recent = df.tail(20).copy()
        avg_volume = None
        avg_dollar_volume = None
        if 'Volume' in recent.columns:
            vol_series = pd.to_numeric(recent['Volume'], errors='coerce').dropna()
            if not vol_series.empty:
                avg_volume_val = float(vol_series.mean())
                avg_volume = avg_volume_val
                avg_dollar_volume = current_price * avg_volume_val

        fundamentals_path = stock_dir / f"{symbol}_advanced.json"
        market_cap = None
        if fundamentals_path.exists():
            try:
                with open(fundamentals_path, 'r', encoding='utf-8') as f:
                    fundamentals = json.load(f)
                market_cap_raw = fundamentals.get('marketCap')
                if market_cap_raw is not None:
                    try:
                        raw_str = str(market_cap_raw).strip()
                        if raw_str == '' or raw_str.lower() in {'na', 'n/a', 'none', 'null', '-'}:
                            market_cap = None
                        else:
                            sanitized = raw_str.replace(',', '').replace('$', '')
                            market_cap = float(sanitized)
                    except (ValueError, TypeError):
                        market_cap = None
            except (ValueError, TypeError, json.JSONDecodeError):
                market_cap = None
        return {
            'symbol': symbol,
            'current_price': current_price,
            'avg_volume': avg_volume,
            'avg_dollar_volume': avg_dollar_volume,
            'market_cap': market_cap
        }
    except Exception as err:
        logger.debug(f"Basic snapshot failed for {symbol}: {err}")
        return None

# Global Progressive ML predictor (lazy initialization)
_progressive_predictor = None
_progressive_data_loader = None
_progressive_trainer = None
_progressive_trainer_loader = None

def _get_progressive_predictor():
    """Get or create Progressive ML predictor instance (singleton)"""
    global _progressive_predictor, _progressive_data_loader
    
    if _progressive_predictor is None:
        try:
            from app.ml.progressive.predictor import ProgressivePredictor
            from app.ml.progressive.data_loader import ProgressiveDataLoader
            
            _progressive_data_loader = ProgressiveDataLoader()
            _progressive_predictor = ProgressivePredictor(data_loader=_progressive_data_loader)
            logger.info("✅ Progressive ML predictor initialized for Scanner")
        except Exception as e:
            logger.warning(f"⚠️ Could not initialize Progressive ML predictor: {e}")
            return None
    
    return _progressive_predictor


def _get_progressive_trainer():
    """Get or create Progressive ML trainer instance (singleton)"""
    global _progressive_trainer, _progressive_trainer_loader
    
    if _progressive_trainer is None:
        try:
            from app.ml.progressive.trainer import ProgressiveTrainer
            from app.ml.progressive.data_loader import ProgressiveDataLoader

            if _progressive_trainer_loader is None:
                _progressive_trainer_loader = ProgressiveDataLoader(
                    sequence_length=TRAIN_SEQUENCE_LENGTH,
                    horizons=TRAIN_HORIZONS
                )

            training_config = {
                'epochs': 60,
                'validation_split': 0.2,
                'batch_size': 32,
                'early_stopping_patience': 10,
                'reduce_lr_patience': 6,
            }

            _progressive_trainer = ProgressiveTrainer(
                _progressive_trainer_loader,
                training_config=training_config
            )
            logger.info("✅ Progressive ML trainer initialized for Scanner (126d transformer focus)")
        except Exception as e:
            logger.warning(f"⚠️ Could not initialize Progressive ML trainer: {e}")
            return None
    
    return _progressive_trainer


async def _train_symbol_job(symbol: str, trainer) -> Tuple[bool, Optional[str]]:
    """Run the heavy 126d transformer training for a single symbol."""
    symbol = symbol.upper()
    success = False
    failure_reason: Optional[str] = 'training-error'

    if not _has_minimum_history(symbol):
        logger.warning(
            "Skipping training for %s: insufficient history (< %s rows)",
            symbol,
            MIN_HISTORY_ROWS
        )
        return False, 'insufficient-history'

    try:
        logger.info(f"Starting Progressive ML training (126d transformer) for {symbol}")
        result = await asyncio.to_thread(
            trainer.train_progressive_models,
            symbol,
            TRAIN_MODEL_TYPES
        )
        if not result:
            logger.error(f"Training returned empty result for {symbol}")
            failure_reason = 'empty-result'
        elif not _model_126d_exists(symbol):
            logger.error(f"Expected 126d checkpoint missing after training {symbol}")
            failure_reason = 'missing-checkpoint'
        else:
            updated = False
            for item in _scanner_state.filter_items:
                if item.get('symbol') == symbol:
                    item['has_model'] = True
                    updated = True
                    break
            if updated:
                try:
                    _scanner_state.trained_count = sum(1 for it in _scanner_state.filter_items if it.get('has_model'))
                except Exception:
                    pass

            success = True
            logger.info(f"✅ Training completed for {symbol}")
    except Exception as exc:
        logger.error(f"Training error for {symbol}: {exc}")
        if isinstance(exc, ValueError) and 'Could not prepare features' in str(exc):
            failure_reason = 'insufficient-history'
        else:
            failure_reason = 'training-error'
    finally:
        _scanner_state.train_last_persisted_at = time.time()
        if not success:
            logger.debug(f"Training failed for {symbol}; queue will mark status accordingly")

    return success, None if success else failure_reason


async def _drain_training_queue():
    """Background worker that processes the training queue sequentially."""
    global _train_worker_task

    trainer = _get_progressive_trainer()
    if not trainer:
        logger.error("Progressive trainer unavailable; marking queued jobs as failed")
        lock = _get_train_queue_lock()
        async with lock:
            while _scanner_state.train_queue:
                sym = _scanner_state.train_queue.pop(0)
                _scanner_state.train_symbols_status[sym] = 'failed'
                _scanner_state.train_failed_symbols[sym] = 'trainer-unavailable'
            _scanner_state.train_status = 'error'
            _scanner_state.train_current_symbol = None
            _train_worker_task = None
        return

    try:
        while True:
            lock = _get_train_queue_lock()
            async with lock:
                if not _scanner_state.train_queue:
                    if _scanner_state.train_total == 0:
                        _scanner_state.train_status = 'idle'
                    elif _scanner_state.train_failed_symbols:
                        _scanner_state.train_status = 'error'
                    else:
                        _scanner_state.train_status = 'completed'
                    _scanner_state.train_current_symbol = None
                    _train_worker_task = None
                    return

                symbol = _scanner_state.train_queue.pop(0)
                _scanner_state.train_current_symbol = symbol
                _scanner_state.train_symbols_status[symbol] = 'running'

            success, failure_reason = await _train_symbol_job(symbol, trainer)

            lock = _get_train_queue_lock()
            async with lock:
                if success:
                    _scanner_state.train_symbols_status[symbol] = 'completed'
                    _scanner_state.train_completed += 1
                else:
                    status_value = 'skipped' if failure_reason == 'insufficient-history' else 'failed'
                    _scanner_state.train_symbols_status[symbol] = status_value
                    if failure_reason:
                        _scanner_state.train_failed_symbols[symbol] = failure_reason
                _scanner_state.train_current_symbol = None

            await asyncio.sleep(0)
    except Exception as exc:
        logger.error(f"Training worker crashed: {exc}")
        lock = _get_train_queue_lock()
        async with lock:
            if _scanner_state.train_current_symbol:
                current = _scanner_state.train_current_symbol
                _scanner_state.train_symbols_status[current] = 'failed'
                _scanner_state.train_failed_symbols[current] = 'worker-crashed'
            while _scanner_state.train_queue:
                sym = _scanner_state.train_queue.pop(0)
                _scanner_state.train_symbols_status[sym] = 'failed'
                _scanner_state.train_failed_symbols[sym] = 'worker-crashed'
            _scanner_state.train_status = 'error'
            _scanner_state.train_current_symbol = None
            _train_worker_task = None
    else:
        lock = _get_train_queue_lock()
        async with lock:
            if _train_worker_task is not None and _train_worker_task.done():
                _train_worker_task = None


async def _enqueue_training(symbols: List[str], *, force: bool = False) -> Dict[str, Any]:
    """Queue symbols for training and optionally force retraining existing models."""
    global _train_worker_task

    if not symbols:
        return {"enqueued": [], "skipped": {}}

    enqueued: List[str] = []
    skipped: Dict[str, str] = {}

    lock = _get_train_queue_lock()

    async with lock:
        if _scanner_state.train_status not in {'running'} and not _scanner_state.train_queue:
            _scanner_state.reset_training_tracking()

        for raw_symbol in symbols:
            symbol = (raw_symbol or '').upper()
            if not symbol:
                continue

            already_running = _scanner_state.train_symbols_status.get(symbol) in {'running', 'queued'} or symbol in _scanner_state.train_queue
            if already_running:
                logger.info(f"Training queue: {symbol} already queued or running; skipping duplicate")
                continue

            has_model = _model_126d_exists(symbol)
            if has_model and not force:
                _scanner_state.train_symbols_status[symbol] = 'completed'
                skipped[symbol] = 'already-trained'
                logger.info(f"Training queue: {symbol} already has 126d checkpoint; skipping")
                continue

            if not _has_minimum_history(symbol):
                _scanner_state.train_symbols_status[symbol] = 'skipped'
                skipped[symbol] = 'insufficient-history'
                _scanner_state.train_failed_symbols[symbol] = 'insufficient-history'
                logger.info(
                    "Training queue: %s lacks minimum history (%s required)",
                    symbol,
                    MIN_HISTORY_ROWS
                )
                continue

            if force and has_model:
                logger.info(f"Training queue: forcing retrain for {symbol} despite existing model")
                for item in _scanner_state.filter_items:
                    if (item.get('symbol') or '').upper() == symbol:
                        item['has_model'] = False
                        break
                try:
                    _scanner_state.trained_count = sum(1 for it in _scanner_state.filter_items if it.get('has_model'))
                except Exception:
                    pass

            _scanner_state.train_queue.append(symbol)
            _scanner_state.train_symbols_status[symbol] = 'queued'
            _scanner_state.train_total += 1
            enqueued.append(symbol)

        if enqueued:
            if _scanner_state.train_status != 'running':
                _scanner_state.train_status = 'running'
                _scanner_state.train_started_at = time.time()

            loop = asyncio.get_running_loop()
            if _train_worker_task is None or _train_worker_task.done():
                _train_worker_task = loop.create_task(_drain_training_queue())

    return {"enqueued": enqueued, "skipped": skipped}


from app.strategies.metrics import _compute_local_metrics, _iter_local_symbols


async def _get_hot_stocks_internal(limit: int = 10) -> Dict[str, Any]:
    """
    Get dynamically scanned hot stocks with high potential from local stock_data
    Scans ALL symbols (10,889), not limited to 500
    """
    try:
        # Build from ALL local symbols
        symbols = _iter_local_symbols(max_symbols=None)
        items: List[Dict[str, Any]] = []
        scanned = 0
        
        logger.info(f"🔥 Scanning {len(symbols)} symbols for hot stocks...")
        
        for sym in symbols:
            scanned += 1
            m = _compute_local_metrics(sym)
            if m is None:
                continue
            # Name isn't available in local CSVs; mirror symbol
            m['name'] = sym
            
            # Ensure all data is JSON serializable before storing
            def make_json_serializable(obj):
                if isinstance(obj, dict):
                    return {k: make_json_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [make_json_serializable(item) for item in obj]
                elif isinstance(obj, (np.integer, np.int64, np.int32)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float64, np.float32)):
                    return float(obj)
                elif isinstance(obj, (np.bool_,)):
                    return bool(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                else:
                    return obj
            
            serializable_m = make_json_serializable(m)
            items.append(serializable_m)
            
            # Log progress every 1000
            if scanned % 1000 == 0:
                logger.info(f"   Scanned {scanned}/{len(symbols)} symbols, found {len(items)} valid")
        
        logger.info(f"✅ Scan complete: {len(items)} valid stocks from {scanned} symbols")
        
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


async def _run_scan(mode: str, limit: Optional[int]):
    """Background task: run scanner and update _scanner_state.top"""
    try:
        limit_desc = limit if (limit is not None and limit > 0) else 'all'
        logger.info(f"Starting scanner run (mode={mode}, limit={limit_desc})")
        _scanner_state.filter_state = 'running'

        filtered_items = list(_scanner_state.filter_items or [])
        filtered_symbols: List[str] = []
        for item in filtered_items:
            symbol = (item.get('symbol') or '').upper()
            if symbol:
                filtered_symbols.append(symbol)

        candidates: Optional[List[str]] = None
        if filtered_symbols:
            if mode.lower() == 'ml':
                candidates = []
                for item in filtered_items:
                    symbol = (item.get('symbol') or '').upper()
                    if symbol and item.get('has_model'):
                        candidates.append(symbol)
                logger.info(
                    "Scanner ML mode using %s symbols with trained models from filter results",
                    len(candidates)
                )
            elif mode.lower() == 'technical':
                candidates = filtered_symbols.copy()
                logger.info(
                    "Scanner technical mode using %s filtered symbols",
                    len(candidates)
                )
        else:
            logger.info("No cached filter results; scanner will use full universe")

        # Import scan strategies
        if mode.lower() == 'technical':
            from app.strategies.technical_scan import scan_technical
            result = await asyncio.to_thread(
                scan_technical,
                limit=limit if limit and limit > 0 else None,
                symbols=candidates
            )
        elif mode.lower() == 'ml':
            from app.strategies.ml_scan import scan_ml
            result = await asyncio.to_thread(
                scan_ml,
                limit=limit if limit and limit > 0 else None,
                symbols=candidates
            )
        else:
            # Default to ML scan
            from app.strategies.ml_scan import scan_ml
            result = await asyncio.to_thread(
                scan_ml,
                limit=limit if limit and limit > 0 else None,
                symbols=candidates
            )

        if isinstance(result, dict) and result.get('status') == 'success':
            stocks = result.get('data', {}).get('stocks', [])
            _scanner_state.top = stocks
            logger.info(f"Scanner completed: found {len(stocks)} stocks")
        
        _scanner_state.filter_state = 'completed'
    except Exception as e:
        logger.error(f"Scanner run failed: {e}")
        _scanner_state.filter_state = 'error'


def _run_filter_job(price_min: float, adv_min: float):
    """
    Background task: filter ALL local symbols by criteria:
    - Price >= $3
    - Average Dollar Volume >= $1M
    - EXCLUDE micro-cap stocks (focus on liquid, non-micro caps)
    """
    try:
        logger.info(f"🔍 Starting filter job (price_min=${price_min}, adv_min=${adv_min:,.0f})")
        _scanner_state.filter_state = 'running'
        _scanner_state.filter_progress = 0.0
        _scanner_state.filter_items = []
        
        # Get ALL symbols (no limit)
        symbols = _iter_local_symbols(max_symbols=None)
        _scanner_state.total_symbols = len(symbols)
        _scanner_state.processed_symbols = 0
        _scanner_state.passed_count = 0
        _scanner_state.trained_count = 0
        
        logger.info(f"📊 Processing {len(symbols)} symbols from stock_data...")
        
        for i, sym in enumerate(symbols):
            snapshot = _load_basic_symbol_snapshot(sym)

            if snapshot:
                raw_price = snapshot.get('current_price')
                try:
                    current_price = float(raw_price)
                    if not math.isfinite(current_price):
                        current_price = 0.0
                except (TypeError, ValueError):
                    current_price = 0.0

                raw_adv = snapshot.get('avg_dollar_volume')
                try:
                    avg_dollar_volume = float(raw_adv)
                    if not math.isfinite(avg_dollar_volume):
                        avg_dollar_volume = 0.0
                except (TypeError, ValueError):
                    avg_dollar_volume = 0.0

                raw_market_cap = snapshot.get('market_cap')
                try:
                    market_cap_val = float(raw_market_cap) if raw_market_cap is not None else None
                    if market_cap_val is not None and not math.isfinite(market_cap_val):
                        market_cap_val = None
                except (TypeError, ValueError):
                    market_cap_val = None

                passes_price = current_price >= price_min
                passes_volume = avg_dollar_volume >= adv_min
                passes_micro_cap = (market_cap_val is not None) and (market_cap_val >= MICRO_CAP_THRESHOLD)

                if passes_price and passes_volume and passes_micro_cap:
                    has_model = _model_126d_exists(sym)

                    avg_volume_val = snapshot.get('avg_volume')
                    try:
                        avg_volume_float = float(avg_volume_val)
                        avg_volume_int = int(avg_volume_float) if math.isfinite(avg_volume_float) else 0
                    except (TypeError, ValueError):
                        avg_volume_int = 0

                    item = {
                        'symbol': sym,
                        'current_price': current_price,
                        'avg_volume': avg_volume_int,
                        'avg_dollar_volume': avg_dollar_volume,
                        'market_cap': market_cap_val,
                        'has_model': has_model,
                    }
                    _scanner_state.filter_items.append(item)
                    _scanner_state.passed_count += 1
                    if has_model:
                        _scanner_state.trained_count += 1
            
            _scanner_state.processed_symbols = i + 1
            # Report progress as 0..1 fraction (UI multiplies by 100)
            _scanner_state.filter_progress = (i + 1) / len(symbols)
            
            # Log progress every 1000 symbols (print human-friendly percent)
            if (i + 1) % 1000 == 0:
                logger.info(
                    f"   Progress: {i+1}/{len(symbols)} ({_scanner_state.filter_progress*100:.1f}%) - Passed: {_scanner_state.passed_count}"
                )
        
        _scanner_state.filter_state = 'completed'
        logger.info(f"✅ Filter completed: {_scanner_state.passed_count}/{len(symbols)} stocks passed all criteria")
        
    except Exception as e:
        logger.error(f"❌ Filter job failed: {e}")
        _scanner_state.filter_state = 'error'


# ============================================================
# Scanner Endpoints
# ============================================================
@router.get("/status")
async def scanner_status():
    """Get scanner system status"""
    return {"status": "success", "data": _scanner_state.to_status()}


@router.get("/hot-stocks")
async def get_hot_stocks(limit: int = 10):
    """Get dynamically scanned hot stocks with high potential"""
    return await _get_hot_stocks_internal(limit=limit)


@router.get("/top")
async def scanner_top(limit: Optional[int] = None):
    """Return ranked items in shape expected by scanner.html"""
    try:
        items = _scanner_state.top
        if not items:
            fallback_limit = None
            if limit and limit > 0:
                fallback_limit = limit
            r = await _get_hot_stocks_internal(limit=fallback_limit or 100)
            if isinstance(r, dict) and r.get('status') == 'success':
                items = (r.get('data') or {}).get('hot_stocks') or []
        
        # Build ranked table rows
        ranked = []
        visible_items = items
        if limit and limit > 0:
            visible_items = items[:limit]

        for i, it in enumerate(visible_items, start=1):
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
                "returned": len(ranked),
                "items": ranked
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/run")
async def scanner_run(mode: str = 'ml', limit: Optional[int] = None, background_tasks: BackgroundTasks = None):
    """Start a background scan"""
    try:
        norm_limit = limit if limit and limit > 0 else None

        if background_tasks is not None:
            background_tasks.add_task(_run_scan, mode, norm_limit)
        else:
            asyncio.create_task(_run_scan(mode, norm_limit))
        return {"status": "success", "data": {"started": True, "mode": mode, "limit": norm_limit}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/filter/status")
async def scanner_filter_status():
    """Get filter job status"""
    # When not running, refresh has_model (126d) from filesystem so counters reflect reality without requiring /filter/results
    try:
        if _scanner_state.filter_state != 'running' and _scanner_state.filter_items:
            for item in _scanner_state.filter_items:
                sym = item.get('symbol')
                if sym:
                    try:
                        item['has_model'] = _model_126d_exists(sym)
                    except Exception:
                        pass
        current_trained = sum(1 for it in _scanner_state.filter_items if it.get('has_model'))
        _scanner_state.trained_count = current_trained
    except Exception:
        pass

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


@router.post("/filter/run")
async def scanner_filter_run(background_tasks: BackgroundTasks = None):
    """Start filter job to scan local symbols"""
    from datetime import datetime as _dt
    try:
        # Parse optional thresholds
        price_min = 3.0
        adv_min = 1_000_000.0
        
        # Always run the heavy filter job in a separate thread so the main event loop stays responsive.
        def _bg_runner(pmin: float, amin: float):
            try:
                _run_filter_job(pmin, amin)
            except Exception as _e:
                logger.error(f"Background filter thread failed: {_e}")

        threading.Thread(target=_bg_runner, args=(price_min, adv_min), daemon=True).start()
        
        return {"status": "success", "data": {"started": True, "started_at": _dt.now().isoformat(), "price_min": price_min, "adv_min": adv_min}}
    except Exception as e:
        _scanner_state.filter_state = 'error'
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/filter/results")
async def scanner_filter_results():
    """Get filter results"""
    try:
        # Ensure all data is JSON serializable (convert numpy types to native Python)
        def make_json_serializable(obj):
            if isinstance(obj, dict):
                return {k: make_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_json_serializable(item) for item in obj]
            elif isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, (np.bool_,)):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        # Refresh has_model flag (126d only) based on file system to reflect models trained elsewhere
        for item in _scanner_state.filter_items:
            sym = item.get('symbol')
            if sym:
                try:
                    # Only consider 126d transformer checkpoints for UI status
                    item['has_model'] = _model_126d_exists(sym)
                except Exception:
                    pass
        serializable_items = [make_json_serializable(item) for item in _scanner_state.filter_items]
        # Update trained_count after refreshing has_model flags
        try:
            _scanner_state.trained_count = sum(1 for it in _scanner_state.filter_items if it.get('has_model'))
        except Exception:
            pass
        
        return {
            "status": "success",
            "data": {
                "items": serializable_items,
                "total": len(serializable_items)
            }
        }
    except Exception as e:
        logger.error(f"Error serializing filter results: {e}")
        raise HTTPException(status_code=500, detail=f"Serialization error: {str(e)}")


@router.get("/train/status")
async def scanner_train_status():
    """Get training queue status"""
    data = {
        "status": _scanner_state.train_status,
        "state": _scanner_state.train_status,
        "current_symbol": _scanner_state.train_current_symbol,
        "queued": len(_scanner_state.train_queue),
        "completed": _scanner_state.train_completed,
        "failed": len(_scanner_state.train_failed_symbols),
        "failed_symbols": dict(_scanner_state.train_failed_symbols),
        "symbols": dict(_scanner_state.train_symbols_status),
        "total": _scanner_state.train_total,
        "restored": False,
        "last_persisted_at": _scanner_state.train_last_persisted_at,
    }
    return {"status": "success", "data": data}


@router.post("/train/start")
async def scanner_train_start(symbol: Optional[str] = None):
    """Queue Progressive ML training for a single symbol"""
    try:
        if not symbol:
            raise HTTPException(status_code=400, detail="symbol parameter required")

        symbol = symbol.strip().upper()

        trainer = _get_progressive_trainer()
        if not trainer:
            raise HTTPException(status_code=503, detail="Progressive ML trainer not available")

        enqueue_result = await _enqueue_training([symbol], force=True)
        enqueued = enqueue_result.get("enqueued", [])
        skipped = enqueue_result.get("skipped", {})

        started = bool(enqueued)
        message = None

        if skipped:
            reason = skipped.get(symbol)
            if reason == 'insufficient-history':
                message = f"Insufficient price history for {symbol} (requires >= {MIN_HISTORY_ROWS} rows)"
            elif reason == 'already-trained':
                message = f"Model for {symbol} already up to date"
            else:
                message = f"Skipped {symbol}: {reason}"
        elif not started:
            if _model_126d_exists(symbol):
                message = "Model already trained"
            else:
                message = "Symbol already queued"

        return {
            "status": "success",
            "data": {
                "started": started,
                "symbol": symbol,
                "queued": len(enqueued),
                "skipped": skipped,
                "skipped_count": len(skipped),
                "message": message
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to queue training for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/train/start-all")
async def scanner_train_start_all():
    """Queue training for all currently filtered symbols missing a 126d model."""
    try:
        trainer = _get_progressive_trainer()

        if not trainer:
            raise HTTPException(status_code=503, detail="Progressive ML trainer not available")

        symbols_to_train = [item['symbol'] for item in _scanner_state.filter_items if not item.get('has_model')]

        if not symbols_to_train:
            return {
                "status": "success",
                "data": {"started": False, "message": "No symbols require training"}
            }

        enqueue_result = await _enqueue_training(symbols_to_train)
        enqueued = enqueue_result.get("enqueued", [])
        skipped = enqueue_result.get("skipped", {})

        started = bool(enqueued)
        skipped_count = len(skipped)
        message = None

        if skipped_count and started:
            message = f"Enqueued {len(enqueued)} symbols; skipped {skipped_count} with insufficient history"
        elif skipped_count and not started:
            message = f"Skipped {skipped_count} symbols (insufficient history)"
        elif not started:
            message = "All symbols already queued or trained"

        return {
            "status": "success",
            "data": {
                "started": started,
                "total": len(symbols_to_train),
                "queued": len(enqueued),
                "skipped": skipped,
                "skipped_count": skipped_count,
                "message": message
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to queue train-all: {e}")
        _scanner_state.train_status = 'error'
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/train/symbol/{symbol}/status")
async def scanner_train_symbol_status(symbol: str):
    """Check if a symbol has trained models"""
    try:
        symbol = symbol.upper()
        
        # Check if 126d model exists (UI reflects 126d training only)
        has_model = _model_126d_exists(symbol)

        status = _scanner_state.train_symbols_status.get(symbol)
        if status is None:
            status = 'completed' if has_model else 'not-started'
        elif has_model:
            status = 'completed'
        elif status == 'completed' and not has_model:
            status = 'not-started'
        elif status == 'running' and _scanner_state.train_current_symbol != symbol:
            status = 'queued'

        return {
            "status": "success",
            "data": {
                "symbol": symbol,
                "status": status,
                "has_model": has_model
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting train status for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/technical/convergence")
async def scanner_technical_convergence(min_score: float = 60.0, limit: int = 50):
    """
    Scan for stocks meeting MACD Convergence technical criteria.
    More flexible filtering to catch setups early.
    
    Args:
        min_score: Minimum technical score (0-100), default 60
        limit: Maximum results to return
    
    Returns stocks sorted by technical_score descending.
    """
    try:
        logger.info(f"🔍 Technical scan: Convergence setup (min_score={min_score})")
        
        # Use filter_items if available, otherwise scan all
        if _scanner_state.filter_items:
            candidates = _scanner_state.filter_items
            logger.info(f"   Using {len(candidates)} pre-filtered stocks")
        else:
            # Quick scan - sample from available symbols
            all_symbols = _iter_local_symbols(max_symbols=None)
            logger.info(f"   Sampling from {len(all_symbols)} total symbols...")
            
            # Sample every 10th symbol for speed (still ~1000 stocks)
            sampled = all_symbols[::10] if len(all_symbols) > 500 else all_symbols
            
            candidates = []
            for sym in sampled:
                m = _compute_local_metrics(sym)
                if m:
                    m['name'] = sym
                    candidates.append(m)
        
        # Filter by technical score
        matches = []
        for item in candidates:
            tech_score = item.get('technical_score', 0.0)
            convergence_data = item.get('convergence_data', {})
            
            if tech_score >= min_score and convergence_data.get('meets_criteria', False):
                matches.append(item)
        
        # Sort by technical score descending
        matches.sort(key=lambda x: x.get('technical_score', 0.0), reverse=True)
        results = matches[:limit]
        
        logger.info(f"✅ Found {len(results)} stocks with convergence setup (min_score={min_score})")
        
        return {
            "status": "success",
            "data": {
                "stocks": results,
                "total": len(results),
                "total_scanned": len(candidates),
                "min_score": min_score,
                "criteria": "MACD Convergence (ADX, Negative Zone, Volume Dry, Histogram Rising, Conv Ratio)"
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Technical convergence scan failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
