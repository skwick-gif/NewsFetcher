from fastapi import APIRouter, HTTPException, Query
from typing import Optional, Dict, Any
import logging

from app.core.config import logger

router = APIRouter()

@router.get("/rl/status")
async def rl_status():
    """Minimal RL status placeholder (safe)."""
    return {
        "status": "idle",
        "positions": [],
        "pnl": 0.0,
        "decisions": []
    }

@router.get("/rl/simulate")
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

@router.get("/rl/simulate/plan")
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