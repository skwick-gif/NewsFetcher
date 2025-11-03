"""Value and income scan identifying discounted cash generative names."""

import logging
from typing import List, Dict, Any, Optional

from app.strategies.metrics import _iter_local_symbols, _compute_local_metrics

logger = logging.getLogger(__name__)


def _norm(value: Optional[float], lower: float, upper: float) -> float:
    if value is None:
        return 0.0
    span = upper - lower
    if span <= 0:
        return 0.0
    scaled = (value - lower) / span
    if scaled < 0:
        return 0.0
    if scaled > 1:
        return 1.0
    return float(scaled)


def _inverse_norm(value: Optional[float], lower: float, upper: float) -> float:
    if value is None:
        return 0.0
    span = upper - lower
    if span <= 0:
        return 0.0
    scaled = (upper - value) / span
    if scaled < 0:
        return 0.0
    if scaled > 1:
        return 1.0
    return float(scaled)


def _make_json_serializable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_json_serializable(item) for item in obj]
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    return obj


def scan_value(limit: Optional[int] = None, symbols: Optional[List[str]] = None) -> Dict[str, Any]:
    """Return value & income ideas ranked by valuation, analyst conviction, and cash yield."""
    min_price = 5.0
    min_adv = 800_000.0
    min_mcap = 600_000_000
    max_price_to_sales = 6.0
    max_peg_ratio = 2.5
    min_fcf_yield = 0.03
    min_dividend_yield = 0.01

    try:
        if symbols is None:
            universe = _iter_local_symbols(max_symbols=None)
            logger.info("💰 Value scan: using full universe (%s symbols)", len(universe))
        else:
            seen = set()
            universe = []
            for sym in symbols:
                sym_up = (sym or "").upper()
                if sym_up and sym_up not in seen:
                    seen.add(sym_up)
                    universe.append(sym_up)
            logger.info("💰 Value scan: using filtered universe (%s symbols)", len(universe))

        if not universe:
            return {
                "status": "success",
                "data": {
                    "stocks": [],
                    "total": 0,
                    "returned": 0,
                    "total_scanned": 0,
                    "criteria": "Discounted valuations with positive cash yield"
                }
            }

        matches: List[Dict[str, Any]] = []
        total_scanned = 0
        for sym in universe:
            total_scanned += 1
            metrics = _compute_local_metrics(sym)
            if not metrics:
                continue

            price = metrics.get("current_price")
            adv = metrics.get("avg_dollar_volume")
            mcap = metrics.get("market_cap")
            price_to_sales = metrics.get("price_to_sales")
            peg_ratio = metrics.get("peg_ratio")
            fcf_yield = metrics.get("free_cash_flow_yield")
            dividend_yield = metrics.get("dividend_yield")
            analyst_rating = metrics.get("analyst_rating")
            expected_return = metrics.get("expected_return")
            momentum = metrics.get("momentum")

            filters = {
                "price": price is not None and price >= min_price,
                "avg_dollar_volume": adv is not None and adv >= min_adv,
                "market_cap": mcap is not None and mcap >= min_mcap,
                "price_to_sales": price_to_sales is not None and price_to_sales <= max_price_to_sales,
                "peg_ratio": (peg_ratio is None) or (peg_ratio <= max_peg_ratio),
                "free_cash_flow_yield": fcf_yield is not None and fcf_yield >= min_fcf_yield,
                "dividend_yield": (dividend_yield is None) or (dividend_yield >= min_dividend_yield),
                "analyst_rating": (analyst_rating is None) or (analyst_rating <= 3.0),
                "momentum": momentum is None or momentum >= -3.0,
            }

            if not all(filters.values()):
                continue

            score = (
                _inverse_norm(price_to_sales, 1.0, 6.0) * 30.0 +
                _inverse_norm(peg_ratio, 0.5, 3.0) * 20.0 +
                _norm(fcf_yield, 0.03, 0.10) * 20.0 +
                _norm(dividend_yield, 0.01, 0.05) * 10.0 +
                _inverse_norm(analyst_rating, 1.5, 3.0) * 10.0 +
                _norm(expected_return, 0.0, 10.0) * 10.0
            )

            score_components = {
                "price_to_sales": round(_inverse_norm(price_to_sales, 1.0, 6.0) * 30.0, 2),
                "peg_ratio": round(_inverse_norm(peg_ratio, 0.5, 3.0) * 20.0, 2),
                "free_cash_flow_yield": round(_norm(fcf_yield, 0.03, 0.10) * 20.0, 2),
                "dividend_yield": round(_norm(dividend_yield, 0.01, 0.05) * 10.0, 2),
                "analyst_rating": round(_inverse_norm(analyst_rating, 1.5, 3.0) * 10.0, 2),
                "expected_return": round(_norm(expected_return, 0.0, 10.0) * 10.0, 2),
            }

            matches.append({
                **metrics,
                "strategy": "value",
                "strategy_score": round(score, 2),
                "score_components": score_components,
                "filters": filters,
            })

        matches.sort(key=lambda x: x.get("strategy_score", 0.0), reverse=True)
        results = matches if limit in (None, 0) else matches[:limit]
        serializable = [_make_json_serializable(item) for item in results]

        return {
            "status": "success",
            "data": {
                "stocks": serializable,
                "total": len(matches),
                "returned": len(serializable),
                "total_scanned": total_scanned,
                "criteria": "Discounted valuations with positive cash yield",
            }
        }

    except Exception as exc:
        logger.error("❌ Value scan failed: %s", exc)
        return {"status": "error", "message": str(exc)}
