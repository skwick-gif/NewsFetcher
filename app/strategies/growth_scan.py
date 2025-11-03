"""Growth-oriented scan that favours revenue acceleration with momentum confirmation."""

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


def scan_growth(limit: Optional[int] = None, symbols: Optional[List[str]] = None) -> Dict[str, Any]:
    """Return growth leaders ranked by combined revenue, earnings, and momentum metrics."""
    min_price = 5.0
    min_adv = 1_500_000.0
    min_mcap = 750_000_000  # favour mid/large cap for robustness
    min_revenue_growth = 0.12
    min_earnings_growth = 0.08
    min_momentum = 2.0

    try:
        if symbols is None:
            universe = _iter_local_symbols(max_symbols=None)
            logger.info("📈 Growth scan: using full universe (%s symbols)", len(universe))
        else:
            seen = set()
            universe = []
            for sym in symbols:
                sym_up = (sym or "").upper()
                if sym_up and sym_up not in seen:
                    seen.add(sym_up)
                    universe.append(sym_up)
            logger.info("📈 Growth scan: using filtered universe (%s symbols)", len(universe))

        if not universe:
            return {
                "status": "success",
                "data": {
                    "stocks": [],
                    "total": 0,
                    "returned": 0,
                    "total_scanned": 0,
                    "criteria": "Revenue & earnings growth with positive momentum"
                }
            }

        matches = []
        total_scanned = 0
        for sym in universe:
            total_scanned += 1
            metrics = _compute_local_metrics(sym)
            if not metrics:
                continue

            price = metrics.get("current_price")
            adv = metrics.get("avg_dollar_volume")
            mcap = metrics.get("market_cap")
            rev_growth = metrics.get("revenue_growth")
            earn_growth = metrics.get("earnings_growth")
            momentum = metrics.get("momentum")
            profit_margin = metrics.get("profit_margin")
            expected_return = metrics.get("expected_return")

            filters = {
                "price": price is not None and price >= min_price,
                "avg_dollar_volume": adv is not None and adv >= min_adv,
                "market_cap": mcap is not None and mcap >= min_mcap,
                "revenue_growth": rev_growth is not None and rev_growth >= min_revenue_growth,
                "earnings_growth": earn_growth is not None and earn_growth >= min_earnings_growth,
                "momentum": momentum is not None and momentum >= min_momentum,
            }

            if not all(filters.values()):
                continue

            score = (
                _norm(rev_growth, 0.12, 0.45) * 40.0 +
                _norm(earn_growth, 0.08, 0.35) * 25.0 +
                _norm(momentum, 2.0, 12.0) * 15.0 +
                _norm(expected_return, 2.0, 12.0) * 10.0 +
                _norm(profit_margin, 0.08, 0.30) * 10.0
            )

            score_components = {
                "revenue_growth": round(_norm(rev_growth, 0.12, 0.45) * 40.0, 2),
                "earnings_growth": round(_norm(earn_growth, 0.08, 0.35) * 25.0, 2),
                "momentum": round(_norm(momentum, 2.0, 12.0) * 15.0, 2),
                "expected_return": round(_norm(expected_return, 2.0, 12.0) * 10.0, 2),
                "profit_margin": round(_norm(profit_margin, 0.08, 0.30) * 10.0, 2),
            }

            matches.append({
                **metrics,
                "strategy": "growth",
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
                "criteria": "Revenue & earnings growth with momentum confirmation",
            }
        }

    except Exception as exc:
        logger.error("❌ Growth scan failed: %s", exc)
        return {"status": "error", "message": str(exc)}
