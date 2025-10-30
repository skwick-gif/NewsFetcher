"""
ML Scan Strategy
Scans for stocks using ML predictions and expected returns.
"""

import logging
from typing import List, Dict, Any, Optional
import pandas as pd

from app.strategies.technical_scan import _iter_local_symbols, _compute_local_metrics

logger = logging.getLogger(__name__)


def scan_ml(limit: int = 50) -> Dict[str, Any]:
    """
    Scan for stocks using ML predictions and expected returns.

    Args:
        limit: Maximum results to return

    Returns:
        Dict with status, data containing stocks list
    """
    try:
        logger.info("🔍 ML scan: Using ML predictions and expected returns")

        # Get symbols to scan
        symbols = _iter_local_symbols(max_symbols=None)
        logger.info(f"   Scanning {len(symbols)} symbols...")

        items = []
        for sym in symbols:
            metrics = _compute_local_metrics(sym)
            if metrics:
                items.append(metrics)

        logger.info(f"   Found {len(items)} valid stocks")

        # Sort by expected return descending (which includes ML score)
        items.sort(key=lambda x: x.get('expected_return', 0.0), reverse=True)
        results = items[:limit]

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

        logger.info(f"✅ ML scan completed: {len(results)} stocks")

        return {
            "status": "success",
            "data": {
                "stocks": serializable_results,
                "total": len(serializable_results),
                "total_scanned": len(symbols),
                "criteria": "ML predictions + expected returns"
            }
        }

    except Exception as e:
        logger.error(f"❌ ML scan failed: {e}")
        return {
            "status": "error",
            "message": str(e)
        }