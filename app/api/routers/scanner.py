"""
Scanner Router - Stock Filtering, Training & Scanning
Migrated from main_realtime.py to keep the main file clean
"""
import logging
import asyncio
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


# ============================================================
# Scanner Helper Functions
# ============================================================
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


def _compute_convergence_score(symbol: str, df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate technical score based on MACD Convergence strategy criteria.
    Returns dict with score and individual filter results.
    More flexible thresholds to catch stocks early.
    """
    from app.strategies.indicators import macd_series, adx_series
    
    try:
        # Need at least 100 bars for reliable indicators
        if len(df) < 100:
            return {'score': 0.0, 'meets_criteria': False, 'reason': 'insufficient_data'}
        
        # Get last 100 bars
        recent = df.tail(100).copy()
        
        # Extract OHLCV
        close = recent['Close'].values
        high = recent['High'].values if 'High' in recent.columns else close
        low = recent['Low'].values if 'Low' in recent.columns else close
        volume = recent['Volume'].values if 'Volume' in recent.columns else None
        
        # Calculate MACD (12-26-9 for stocks)
        macd_df = macd_series(close, fast=12, slow=26, signal=9)
        macd = macd_df['macd'].values
        signal = macd_df['macd_signal'].values
        hist = macd_df['macd_hist'].values
        
        # Calculate ADX
        adx = adx_series(high, low, close, period=14).values
        
        # Calculate Volume SMA
        vol_sma = None
        if volume is not None:
            vol_sma = pd.Series(volume).rolling(window=20).mean().values
        
        # Get current values (last bar)
        m_t = macd[-1]
        s_t = signal[-1]
        h_t = hist[-1]
        adx_t = adx[-1]
        vol_t = volume[-1] if volume is not None else 0
        vol_sma_t = vol_sma[-1] if vol_sma is not None else 0
        
        # Calculate convergence ratio
        if abs(m_t) > 0.001:
            conv_ratio = abs(m_t - s_t) / abs(m_t)
        else:
            conv_ratio = 1.0  # Wide gap if MACD near zero
        
        # === FILTERS (with flexibility) ===
        filters = {}
        
        # Filter 1: ADX > 15 (more flexible than 20 to catch early trends)
        filters['adx'] = adx_t > 15 if np.isfinite(adx_t) else False
        
        # Filter 2: MACD < 0 AND Signal < 0 (negative zone)
        # Allow MACD slightly positive if very close to zero (within 5% of recent range)
        macd_range = np.ptp(macd[-20:]) if len(macd) >= 20 else 1.0
        tolerance = macd_range * 0.05
        filters['negative_zone'] = (m_t < tolerance) and (s_t < 0)
        
        # Filter 3: Volume < VOL_SMA (seller exhaustion)
        if vol_sma_t > 0:
            filters['volume_dry'] = vol_t < vol_sma_t * 1.1  # Allow up to 10% above SMA
        else:
            filters['volume_dry'] = True  # Skip if no volume data
        
        # Filter 4: Histogram rising (check last 2-3 bars for flexibility)
        k_buy = 2  # Reduced from 3 for earlier signals
        hist_rising = True
        if len(hist) >= k_buy + 1:
            for i in range(-k_buy, 0):
                if not (np.isfinite(hist[i]) and np.isfinite(hist[i-1])):
                    hist_rising = False
                    break
                if not (hist[i] > hist[i-1]):
                    hist_rising = False
                    break
        else:
            hist_rising = False
        filters['hist_rising'] = hist_rising
        
        # Filter 5: conv_ratio <= 50% (more flexible than 40%)
        filters['convergence'] = conv_ratio <= 0.50
        
        # === SCORING ===
        # Count how many filters pass
        passed_filters = sum(filters.values())
        total_filters = len(filters)
        
        # Base score from filter pass rate
        base_score = (passed_filters / total_filters) * 100.0
        
        # Bonus points for strong signals
        bonus = 0.0
        
        # Strong convergence bonus (tighter = better)
        if conv_ratio <= 0.25:
            bonus += 15.0
        elif conv_ratio <= 0.40:
            bonus += 10.0
        
        # Strong ADX bonus (higher = stronger trend)
        if adx_t > 25:
            bonus += 10.0
        elif adx_t > 20:
            bonus += 5.0
        
        # Histogram momentum bonus
        if len(hist) >= 3:
            recent_slope = hist[-1] - hist[-3]
            if recent_slope > 0:
                bonus += 5.0
        
        final_score = min(base_score + bonus, 100.0)
        
        # Determine if meets criteria (at least 4/5 filters + reasonable score)
        meets_criteria = (passed_filters >= 4) and (final_score >= 60.0)
        
        return {
            'score': float(final_score),
            'meets_criteria': meets_criteria,
            'filters': filters,
            'passed_filters': f"{passed_filters}/{total_filters}",
            'conv_ratio': float(conv_ratio * 100.0),  # as percentage
            'adx': float(adx_t) if np.isfinite(adx_t) else 0.0,
            'macd': float(m_t),
            'signal': float(s_t),
            'histogram': float(h_t),
            'reason': 'convergence_setup' if meets_criteria else 'weak_setup'
        }
        
    except Exception as e:
        logger.debug(f"Error computing convergence score for {symbol}: {e}")
        return {'score': 0.0, 'meets_criteria': False, 'reason': f'error: {str(e)}'}


def _compute_local_metrics(symbol: str) -> Optional[Dict[str, Any]]:
    """Compute basic metrics from local stock data CSV and fundamentals"""
    try:
        project_root = Path(__file__).resolve().parents[3]
        csv_path = project_root / 'stock_data' / symbol / f'{symbol}_historical_data.csv'
        
        if not csv_path.exists():
            return None
        
        import pandas as pd
        df = pd.read_csv(csv_path)
        
        if df.empty or len(df) < 5:
            return None
        
        # Ensure we have required columns
        if 'Close' not in df.columns:
            return None
        
        # Get recent data
        recent = df.tail(20)
        current_price = recent['Close'].iloc[-1]
        prev_price = recent['Close'].iloc[-2] if len(recent) >= 2 else current_price
        
        # Calculate simple metrics
        change = current_price - prev_price
        change_percent = (change / prev_price * 100) if prev_price != 0 else 0.0
        
        # Volume if available
        volume = recent['Volume'].iloc[-1] if 'Volume' in recent.columns else 0
        
        # Calculate average daily volume (ADV) - last 20 days
        avg_volume = recent['Volume'].mean() if 'Volume' in recent.columns else 0
        
        # Calculate average dollar volume (price * volume)
        avg_dollar_volume = current_price * avg_volume if avg_volume > 0 else 0
        
        # Load fundamentals for accurate market cap
        market_cap = 0
        is_micro_cap = False
        sector = None
        industry = None
        
        try:
            import json
            fundamentals_path = project_root / 'stock_data' / symbol / f'{symbol}_advanced.json'
            if fundamentals_path.exists():
                with open(fundamentals_path, 'r', encoding='utf-8') as f:
                    fundamentals = json.load(f)
                    
                # Get market cap from fundamentals
                market_cap = fundamentals.get('marketCap', 0)
                
                # Micro-cap definition: Market Cap < $300M
                is_micro_cap = (market_cap > 0 and market_cap < 300_000_000)
                
                # Get sector and industry
                sector = fundamentals.get('sector', None)
                industry = fundamentals.get('industry', None)
        except Exception as e:
            logger.debug(f"Could not load fundamentals for {symbol}: {e}")
            # Fallback to estimation if fundamentals unavailable
            is_micro_cap = (current_price < 10.0 and avg_dollar_volume < 10_000_000)
        
        # Simple momentum (5-day vs 20-day average)
        ma5 = recent['Close'].tail(5).mean()
        ma20 = recent['Close'].mean()
        momentum = ((ma5 - ma20) / ma20 * 100) if ma20 != 0 else 0.0
        
        # Expected return heuristic (combine change% and momentum)
        expected_return = (change_percent * 0.6) + (momentum * 0.4)
        
        # Check if trained model exists and get ML prediction if available
        has_model = False
        ml_score = float(expected_return)  # Default to heuristic score
        
        try:
            model_dir = project_root / 'app' / 'ml' / 'models'
            for model_type in ['transformer', 'lstm', 'cnn']:
                model_file = model_dir / f"{symbol}_{model_type}_progressive.pt"
                if model_file.exists():
                    has_model = True
                    break
            
            # If model exists, try to get real ML prediction
            if has_model:
                try:
                    # Get cached predictor instance
                    predictor = _get_progressive_predictor()
                    
                    if predictor:
                        # Get ensemble prediction
                        pred_result = predictor.predict_ensemble(symbol=symbol, mode="progressive")
                        
                        # Extract 1-day prediction as ml_score
                        if '1d' in pred_result and 'return_pct' in pred_result['1d']:
                            ml_score = float(pred_result['1d']['return_pct'])
                            logger.debug(f"✅ Got ML score for {symbol}: {ml_score:.2f}%")
                    
                except Exception as ml_err:
                    logger.debug(f"Could not get ML prediction for {symbol}: {ml_err}")
                    # Fall back to heuristic score
                    pass
                    
        except Exception:
            pass
        
        # Calculate technical convergence score
        technical_score = 0.0
        convergence_data = {}
        try:
            convergence_data = _compute_convergence_score(symbol, df)
            technical_score = convergence_data.get('score', 0.0)
        except Exception as tech_err:
            logger.debug(f"Could not compute technical score for {symbol}: {tech_err}")
        
        return {
            'symbol': symbol,
            'current_price': float(current_price),
            'change': float(change),
            'change_percent': float(change_percent),
            'volume': int(volume) if volume else 0,
            'avg_volume': int(avg_volume) if avg_volume else 0,
            'avg_dollar_volume': float(avg_dollar_volume),
            'market_cap': int(market_cap),
            'is_micro_cap': is_micro_cap,
            'sector': sector,
            'industry': industry,
            'momentum': float(momentum),
            'expected_return': float(expected_return),
            'ml_score': ml_score,  # Real ML prediction if available, else heuristic
            'technical_score': technical_score,  # Convergence strategy score (0-100)
            'convergence_data': convergence_data,  # Detailed breakdown
            'has_model': has_model,
        }
        
    except Exception as e:
        logger.debug(f"Error computing metrics for {symbol}: {e}")
        return None


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
            items.append(m)
            
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
        
        # Get hot stocks
        result = await _get_hot_stocks_internal(limit=limit)
        if isinstance(result, dict) and result.get('status') == 'success':
            hot_stocks = result.get('data', {}).get('hot_stocks', [])
            _scanner_state.top = hot_stocks
            logger.info(f"Scanner completed: found {len(hot_stocks)} stocks")
        
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
                    _scanner_state.filter_items.append(metrics)
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
        
        # Launch background job
        if background_tasks is not None:
            background_tasks.add_task(_run_filter_job, price_min, adv_min)
        else:
            asyncio.create_task(_run_filter_job(price_min, adv_min))
        
        return {"status": "success", "data": {"started": True, "started_at": _dt.now().isoformat(), "price_min": price_min, "adv_min": adv_min}}
    except Exception as e:
        _scanner_state.filter_state = 'error'
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/filter/results")
async def scanner_filter_results():
    """Get filter results"""
    return {
        "status": "success",
        "data": {
            "items": _scanner_state.filter_items,
            "total": len(_scanner_state.filter_items)
        }
    }


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
