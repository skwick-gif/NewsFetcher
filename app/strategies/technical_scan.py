"""
Technical Scan Strategy
Scans for stocks meeting MACD Convergence technical criteria.
"""

import logging
from typing import List, Dict, Any, Optional
import pandas as pd

from app.api.routers.scanner import _iter_local_symbols, _compute_local_metrics

logger = logging.getLogger(__name__)

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

        # Ensure all filter values are native Python types (no numpy.bool_ etc.)
        try:
            filters = {k: bool(v) for k, v in filters.items()}
        except Exception:
            # Fallback: coerce via simple truthiness
            filters = {k: (True if v else False) for k, v in filters.items()}

        return {
            'score': float(final_score),
            'meets_criteria': bool(meets_criteria),
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
        csv_path = project_root / 'stock_data' / symbol / f'{symbol}_price.csv'
        
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
            'is_micro_cap': bool(is_micro_cap),
            'sector': sector,
            'industry': industry,
            'momentum': float(momentum),
            'expected_return': float(expected_return),
            'ml_score': float(ml_score),
            'technical_score': float(technical_score),
            'convergence_data': convergence_data,
            'has_model': bool(has_model),
        }
        
    except Exception as e:
        logger.debug(f"Error computing metrics for {symbol}: {e}")
        return None


def scan_technical(limit: int = 50, min_score: float = 60.0) -> Dict[str, Any]:
    """
    Scan for stocks meeting technical convergence criteria.

    Args:
        limit: Maximum results to return
        min_score: Minimum technical score (0-100)

    Returns:
        Dict with status, data containing stocks list
    """
    try:
        logger.info(f"🔍 Technical scan: Convergence setup (min_score={min_score})")

        # Get symbols to scan
        symbols = _iter_local_symbols(max_symbols=None)
        logger.info(f"   Scanning {len(symbols)} symbols...")

        matches = []
        for sym in symbols:
            metrics = _compute_local_metrics(sym)
            if not metrics:
                continue

            tech_score = metrics.get('technical_score', 0.0)
            convergence_data = metrics.get('convergence_data', {})

            if tech_score >= min_score and convergence_data.get('meets_criteria', False):
                matches.append(metrics)

        # Sort by technical score descending
        matches.sort(key=lambda x: x.get('technical_score', 0.0), reverse=True)
        results = matches[:limit]

        # Ensure all data is JSON serializable
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

        serializable_results = [make_json_serializable(item) for item in results]

        logger.info(f"✅ Found {len(results)} stocks with convergence setup (min_score={min_score})")

        return {
            "status": "success",
            "data": {
                "stocks": serializable_results,
                "total": len(serializable_results),
                "total_scanned": len(symbols),
                "min_score": min_score,
                "criteria": "MACD Convergence (ADX, Negative Zone, Volume Dry, Histogram Rising, Conv Ratio)"
            }
        }

    except Exception as e:
        logger.error(f"❌ Technical scan failed: {e}")
        return {
            "status": "error",
            "message": str(e)
        }