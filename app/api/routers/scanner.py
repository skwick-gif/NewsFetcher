"""
Scanner Router - Stock Filtering, Training & Scanning
Migrated from main_realtime.py to keep the main file clean
"""
import logging
import asyncio
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Set, Optional

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
    
    def to_status(self) -> Dict[str, Any]:
        """Return scanner status summary"""
        return {
            "state": "idle" if not self.top else "ready",
            "top_count": len(self.top),
            "filter_state": self.filter_state,
            "train_status": self.train_status,
        }

_scanner_state = ScannerState()

# Global Progressive ML predictor (lazy initialization)
_progressive_predictor = None
_progressive_data_loader = None
_progressive_trainer = None

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
    global _progressive_trainer, _progressive_data_loader
    
    if _progressive_trainer is None:
        try:
            from app.ml.progressive.trainer import ProgressiveTrainer
            from app.ml.progressive.data_loader import ProgressiveDataLoader
            
            if _progressive_data_loader is None:
                _progressive_data_loader = ProgressiveDataLoader()
            
            _progressive_trainer = ProgressiveTrainer(_progressive_data_loader)
            logger.info("✅ Progressive ML trainer initialized for Scanner")
        except Exception as e:
            logger.warning(f"⚠️ Could not initialize Progressive ML trainer: {e}")
            return None
    
    return _progressive_trainer


from app.strategies.technical_scan import _iter_local_symbols, _compute_local_metrics
def _iter_local_symbols(max_symbols: int = None) -> List[str]:
    """Iterate over ALL symbols in local stock_data directory (10,889 stocks)"""
    try:
        project_root = Path(__file__).resolve().parents[3]
        stock_data_dir = project_root / 'stock_data'
        if not stock_data_dir.exists():
            logger.warning(f"stock_data directory not found: {stock_data_dir}")
            return []
        
        symbols = []
        for item in stock_data_dir.iterdir():
            if item.is_dir() and item.name.isupper():
                symbols.append(item.name)
                if max_symbols and len(symbols) >= max_symbols:
                    break
        
        logger.info(f"📊 Found {len(symbols)} total symbols in stock_data")
        return symbols
    except Exception as e:
        logger.error(f"Error iterating local symbols: {e}")
        return []


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
                elif isinstance(obj, (np.bool_, np.bool8)):
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


async def _run_scan(mode: str, limit: int):
    """Background task: run scanner and update _scanner_state.top"""
    try:
        logger.info(f"Starting scanner run (mode={mode}, limit={limit})")
        _scanner_state.filter_state = 'running'

        # Import scan strategies
        if mode.lower() == 'technical':
            from app.strategies.technical_scan import scan_technical
            result = await asyncio.to_thread(scan_technical, limit=limit)
        elif mode.lower() == 'ml':
            from app.strategies.ml_scan import scan_ml
            result = await asyncio.to_thread(scan_ml, limit=limit)
        else:
            # Default to ML scan
            from app.strategies.ml_scan import scan_ml
            result = await asyncio.to_thread(scan_ml, limit=limit)

        if isinstance(result, dict) and result.get('status') == 'success':
            stocks = result.get('data', {}).get('stocks', [])
            _scanner_state.top = stocks
            logger.info(f"Scanner completed: found {len(stocks)} stocks")
        
        _scanner_state.filter_state = 'completed'
    except Exception as e:
        logger.error(f"Scanner run failed: {e}")
        _scanner_state.filter_state = 'error'


async def _run_filter_job(price_min: float, adv_min: float):
    """
    Background task: filter ALL local symbols by criteria:
    - Price >= $3
    - Average Dollar Volume >= $1M
    - Micro-cap stocks only
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
        
        logger.info(f"📊 Processing {len(symbols)} symbols from stock_data...")
        
        for i, sym in enumerate(symbols):
            metrics = _compute_local_metrics(sym)
            
            # Apply all filters:
            # 1. Price >= $3
            # 2. Avg Dollar Volume >= $1M
            # 3. Micro-cap stocks
            if metrics:
                passes_price = metrics.get('current_price', 0) >= price_min
                passes_volume = metrics.get('avg_dollar_volume', 0) >= adv_min
                is_micro = metrics.get('is_micro_cap', False)
                
                if passes_price and passes_volume and is_micro:
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
                        elif isinstance(obj, (np.bool_, np.bool8)):
                            return bool(obj)
                        elif isinstance(obj, np.ndarray):
                            return obj.tolist()
                        else:
                            return obj
                    
                    serializable_metrics = make_json_serializable(metrics)
                    _scanner_state.filter_items.append(serializable_metrics)
                    _scanner_state.passed_count += 1
            
            _scanner_state.processed_symbols = i + 1
            _scanner_state.filter_progress = (i + 1) / len(symbols) * 100.0
            
            # Log progress every 1000 symbols
            if (i + 1) % 1000 == 0:
                logger.info(f"   Progress: {i+1}/{len(symbols)} ({_scanner_state.filter_progress:.1f}%) - Passed: {_scanner_state.passed_count}")
        
        _scanner_state.filter_state = 'completed'
        logger.info(f"✅ Filter completed: {_scanner_state.passed_count}/{len(symbols)} stocks passed all criteria")
        
    except Exception as e:
        logger.error(f"❌ Filter job failed: {e}")
        _scanner_state.filter_state = 'error'


async def _train_symbol_background(symbol: str, progressive_trainer):
    """Background task: train Progressive ML models for a symbol"""
    try:
        logger.info(f"Starting Progressive ML training for {symbol}")
        _scanner_state.train_status = 'running'
        _scanner_state.train_current_symbol = symbol
        
        if not progressive_trainer:
            logger.error("Progressive ML trainer not available")
            _scanner_state.train_status = 'error'
            return
        
        # Train all model types (transformer, lstm, cnn)
        result = progressive_trainer.train_progressive_models(
            symbol=symbol,
            model_types=["transformer", "lstm", "cnn"]
        )
        
        if result and result.get('success'):
            logger.info(f"✅ Training completed for {symbol}")
            _scanner_state.train_status = 'completed'
            
            # Update filter_items to mark as trained
            for item in _scanner_state.filter_items:
                if item.get('symbol') == symbol:
                    item['has_model'] = True
                    break
        else:
            logger.error(f"Training failed for {symbol}: {result}")
            _scanner_state.train_status = 'error'
            
    except Exception as e:
        logger.error(f"Training error for {symbol}: {e}")
        _scanner_state.train_status = 'error'
    finally:
        _scanner_state.train_current_symbol = None


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
async def scanner_top(limit: int = 50):
    """Return ranked items in shape expected by scanner.html"""
    try:
        items = _scanner_state.top
        if not items:
            r = await _get_hot_stocks_internal(limit=max(limit, 50))
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


@router.post("/run")
async def scanner_run(mode: str = 'ml', limit: int = 50, background_tasks: BackgroundTasks = None):
    """Start a background scan"""
    try:
        if background_tasks is not None:
            background_tasks.add_task(_run_scan, mode, limit)
        else:
            asyncio.create_task(_run_scan(mode, limit))
        return {"status": "success", "data": {"started": True, "mode": mode, "limit": limit}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/filter/status")
async def scanner_filter_status():
    """Get filter job status"""
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
        
        # Launch background job. Prefer FastAPI BackgroundTasks when available
        # but fall back to a daemon thread that runs the async job to ensure
        # the endpoint returns immediately even if the server's event loop
        # is busy or blocking.
        if background_tasks is not None:
            background_tasks.add_task(_run_filter_job, price_min, adv_min)
        else:
            def _bg_runner(pmin: float, amin: float):
                try:
                    # Run the async filter job in a fresh event loop inside the thread
                    asyncio.run(_run_filter_job(pmin, amin))
                except Exception as _e:
                    logger.error(f"Background filter thread failed: {_e}")

            th = threading.Thread(target=_bg_runner, args=(price_min, adv_min), daemon=True)
            th.start()
        
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
            elif isinstance(obj, (np.bool_, np.bool8)):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        serializable_items = [make_json_serializable(item) for item in _scanner_state.filter_items]
        
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
    return {"status": "success", "data": {"status": _scanner_state.train_status, "current_symbol": _scanner_state.train_current_symbol}}


@router.post("/train/start")
async def scanner_train_start(symbol: Optional[str] = None, background_tasks: BackgroundTasks = None):
    """Start Progressive ML training for a single symbol"""
    try:
        if not symbol:
            raise HTTPException(status_code=400, detail="symbol parameter required")
        
        symbol = symbol.strip().upper()
        
        # Get cached trainer instance
        trainer = _get_progressive_trainer()
        
        if not trainer:
            raise HTTPException(status_code=503, detail="Progressive ML trainer not available")
        
        if background_tasks is not None:
            background_tasks.add_task(_train_symbol_background, symbol, trainer)
        else:
            asyncio.create_task(_train_symbol_background(symbol, trainer))
            
        return {"status": "success", "data": {"started": True, "symbol": symbol}}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start training for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/train/start-all")
async def scanner_train_start_all(background_tasks: BackgroundTasks = None):
    """Train all symbols in filter results"""
    try:
        # Get cached trainer instance
        trainer = _get_progressive_trainer()
        
        if not trainer:
            raise HTTPException(status_code=503, detail="Progressive ML trainer not available")
        
        symbols_to_train = [item['symbol'] for item in _scanner_state.filter_items if not item.get('has_model')]
        
        if not symbols_to_train:
            return {"status": "success", "data": {"started": False, "message": "No symbols to train"}}
        
        _scanner_state.train_status = 'running'
        _scanner_state.trained_count = 0
        
        async def _train_all():
            for symbol in symbols_to_train:
                await _train_symbol_background(symbol, trainer)
                _scanner_state.trained_count += 1
            _scanner_state.train_status = 'completed'
        
        if background_tasks is not None:
            background_tasks.add_task(_train_all)
        else:
            asyncio.create_task(_train_all())
            
        return {"status": "success", "data": {"started": True, "total": len(symbols_to_train)}}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start train-all: {e}")
        _scanner_state.train_status = 'error'
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/train/symbol/{symbol}/status")
async def scanner_train_symbol_status(symbol: str):
    """Check if a symbol has trained models"""
    try:
        symbol = symbol.upper()
        
        # Check if model exists
        has_model = False
        try:
            project_root = Path(__file__).resolve().parents[3]
            model_dir = project_root / 'app' / 'ml' / 'models'
            
            # Check for any model type
            for model_type in ['transformer', 'lstm', 'cnn']:
                model_file = model_dir / f"{symbol}_{model_type}_progressive.pt"
                if model_file.exists():
                    has_model = True
                    break
                    
        except Exception as e:
            logger.debug(f"Error checking model for {symbol}: {e}")
        
        status = 'completed' if has_model else 'not-started'
        if _scanner_state.train_current_symbol == symbol:
            status = 'running'
            
        return {"status": "success", "data": {"symbol": symbol, "status": status, "has_model": has_model}}
        
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
