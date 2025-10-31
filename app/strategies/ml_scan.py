"""
ML Scan Strategy
Scans for stocks using ML predictions and expected returns.
"""

import logging
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np

from app.strategies.metrics import _iter_local_symbols, _compute_local_metrics

logger = logging.getLogger(__name__)


def scan_ml(limit: Optional[int] = None, symbols: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Scan for stocks using ML predictions and expected returns.

    Args:
        limit: Maximum results to return

    Returns:
        Dict with status, data containing stocks list
    """
    try:
        logger.info("🔍 ML scan: Using ML predictions and expected returns")

        # Determine the symbol universe to scan
        if symbols is None:
            universe = _iter_local_symbols(max_symbols=None)
            logger.info(f"   Scanning {len(universe)} symbols (full universe)...")
        else:
            # Normalize symbols and drop duplicates while preserving order
            seen = set()
            universe = []
            for sym in symbols:
                sym_upper = (sym or '').upper()
                if not sym_upper or sym_upper in seen:
                    continue
                seen.add(sym_upper)
                universe.append(sym_upper)
            logger.info(f"   Scanning {len(universe)} symbols from filtered set...")

        if not universe:
            logger.info("   No symbols to scan; returning empty result")
            return {
                "status": "success",
                "data": {
                    "stocks": [],
                    "total": 0,
                    "total_scanned": 0,
                    "criteria": "ML predictions + expected returns"
                }
            }

        items = []
        total_scanned = 0
        for sym in universe:
            total_scanned += 1
            metrics = _compute_local_metrics(sym)
            if metrics:
                items.append(metrics)

        logger.info(f"   Found {len(items)} valid stocks out of {total_scanned} scanned")

        # Sort by expected return descending (which includes ML score)
        items.sort(key=lambda x: x.get('expected_return', 0.0), reverse=True)
        if limit is None or limit <= 0:
            results = items
        else:
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
            elif isinstance(obj, (np.bool_,)):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj

        serializable_results = [make_json_serializable(item) for item in results]

        logger.info(f"✅ ML scan completed: {len(results)} stocks (limit={limit if limit else 'all'})")

        return {
            "status": "success",
            "data": {
                "stocks": serializable_results,
                "total": len(items),
                "returned": len(serializable_results),
                "total_scanned": total_scanned,
                "criteria": "ML predictions + expected returns"
            }
        }

    except Exception as e:
        logger.error(f"❌ ML scan failed: {e}")
        return {
            "status": "error",
            "message": str(e)
        }