"""Quality compounder scan emphasising profitability, cash generation, and balance sheet strength."""

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


def scan_quality(limit: Optional[int] = None, symbols: Optional[List[str]] = None) -> Dict[str, Any]:
    """Return quality compounders ranked by profitability and balance-sheet strength."""
    min_price = 8.0
    min_adv = 1_000_000.0
    min_mcap = 1_000_000_000
    min_profit_margin = 0.12
    min_return_on_equity = 0.12
    max_debt_to_equity = 200.0
    min_fcf_yield = 0.025

    try:
        if symbols is None:
            universe = _iter_local_symbols(max_symbols=None)
            logger.info("🏆 Quality scan: using full universe (%s symbols)", len(universe))
        else:
            seen = set()
            universe = []
            for sym in symbols:
                sym_up = (sym or "").upper()
                if sym_up and sym_up not in seen:
                    seen.add(sym_up)
                    universe.append(sym_up)
            logger.info("🏆 Quality scan: using filtered universe (%s symbols)", len(universe))

        if not universe:
            return {
                "status": "success",
                "data": {
                    "stocks": [],
                    "total": 0,
                    "returned": 0,
                    "total_scanned": 0,
                    "criteria": "High-margin, cash-generative companies with manageable leverage"
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
            profit_margin = metrics.get("profit_margin")
            operating_margin = metrics.get("operating_margin")
            roe = metrics.get("return_on_equity")
            roa = metrics.get("return_on_assets")
            debt_to_equity = metrics.get("debt_to_equity")
            fcf_yield = metrics.get("free_cash_flow_yield")
            current_ratio = metrics.get("current_ratio")
            quick_ratio = metrics.get("quick_ratio")

            filters = {
                "price": price is not None and price >= min_price,
                "avg_dollar_volume": adv is not None and adv >= min_adv,
                "market_cap": mcap is not None and mcap >= min_mcap,
                "profit_margin": profit_margin is not None and profit_margin >= min_profit_margin,
                "return_on_equity": roe is not None and roe >= min_return_on_equity,
                "debt_to_equity": (debt_to_equity is None) or (debt_to_equity <= max_debt_to_equity),
                "free_cash_flow_yield": fcf_yield is not None and fcf_yield >= min_fcf_yield,
            }

            if not all(filters.values()):
                continue

            score = (
                _norm(profit_margin, 0.12, 0.35) * 25.0 +
                _norm(operating_margin, 0.10, 0.30) * 15.0 +
                _norm(roe, 0.12, 0.45) * 25.0 +
                _norm(roa, 0.06, 0.25) * 10.0 +
                _norm(fcf_yield, 0.025, 0.09) * 15.0 +
                _inverse_norm(debt_to_equity, 40.0, 200.0) * 10.0
            )

            score_components = {
                "profit_margin": round(_norm(profit_margin, 0.12, 0.35) * 25.0, 2),
                "operating_margin": round(_norm(operating_margin, 0.10, 0.30) * 15.0, 2),
                "return_on_equity": round(_norm(roe, 0.12, 0.45) * 25.0, 2),
                "return_on_assets": round(_norm(roa, 0.06, 0.25) * 10.0, 2),
                "free_cash_flow_yield": round(_norm(fcf_yield, 0.025, 0.09) * 15.0, 2),
                "leverage": round(_inverse_norm(debt_to_equity, 40.0, 200.0) * 10.0, 2),
            }

            matches.append({
                **metrics,
                "strategy": "quality",
                "strategy_score": round(score, 2),
                "score_components": score_components,
                "filters": filters,
                "liquidity": {
                    "avg_dollar_volume": adv,
                    "current_ratio": current_ratio,
                    "quick_ratio": quick_ratio,
                }
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
                "criteria": "High-margin, cash-generative companies with manageable leverage",
            }
        }

    except Exception as exc:
        logger.error("❌ Quality scan failed: %s", exc)
        return {"status": "error", "message": str(exc)}
