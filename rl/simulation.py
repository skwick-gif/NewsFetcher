"""
Simple RL simulation runner for a single symbol using MarketEnv.
Returns time series suitable for plotting in the UI.
"""

from typing import Dict, Any, List, Optional
import numpy as np

from rl.envs.market_env import MarketEnv


def run_simulation(symbol: str, days: Optional[int] = 250, window: int = 60,
                   cost_bps: int = 5, slip_bps: int = 5,
                   policy: str = "follow_trend",
                   start_date: Optional[str] = None,
                   end_date: Optional[str] = None) -> Dict[str, Any]:
    """
    Run a simple non-learning simulation to visualize behavior.
    policy options:
      - follow_trend: long if last return > 0 else flat
      - buy_and_hold: always full long
      - half_long: always half long
    """
    env = MarketEnv.load_from_local(symbol, window=window, tail_days=days,
                                    start_date=start_date, end_date=end_date,
                                    transaction_cost_bps=cost_bps, slippage_bps=slip_bps)
    obs = env.reset()

    times: List[int] = []
    dates: List[str] = []
    prices: List[float] = []
    ohlc: List[Dict[str, Any]] = []
    equities: List[float] = []
    pos_fracs: List[float] = []
    actions: List[int] = []

    done = False
    step_count = 0
    decisions: List[Dict[str, Any]] = []
    while not done and step_count < 100000:
        # Simple policy
        act = 2  # default full long
        if policy == "follow_trend":
            last_ret = 0.0
            try:
                r = obs['features_window']['ret']
                last_ret = float(r[-1]) if r is not None and len(r) > 0 else 0.0
            except Exception:
                last_ret = 0.0
            act = 2 if last_ret > 0 else 0
        elif policy == "buy_and_hold":
            act = 2
        elif policy == "half_long":
            act = 1

        # Step environment
        obs, reward, done, info = env.step(act)

        # Time/index
        t = obs.get('t', step_count)
        times.append(t)
        try:
            dt = str(env.df.index[t].date()) if hasattr(env.df.index, 'date') else str(env.df.index[t])
        except Exception:
            dt = str(t)
        dates.append(dt)

        # Series values
        prices.append(float(info.get('price', 0.0)))
        # Capture OHLC if available for candlestick rendering
        try:
            row = env.df.iloc[t]
            o = float(row['Open']) if 'Open' in env.df.columns else None
            h = float(row['High']) if 'High' in env.df.columns else None
            l = float(row['Low']) if 'Low' in env.df.columns else None
            c = float(row['Close']) if 'Close' in env.df.columns else None
            # also try to fetch Volume
            v = None
            try:
                if 'Volume' in env.df.columns:
                    v = float(row['Volume'])
                elif 'volume' in env.df.columns:
                    v = float(row['volume'])
            except Exception:
                v = None
            if None not in (o, h, l, c):
                # include timestamp string for tooltip and axes, and volume if present
                ohlc.append({"o": o, "h": h, "l": l, "c": c, "t": dt, **({"v": v} if v is not None else {})})
            else:
                ohlc.append({})
        except Exception:
            ohlc.append({})
        equities.append(float(info.get('equity', 0.0)))
        pos_fracs.append(float(info.get('position_frac', 0.0)))
        actions.append(int(info.get('action', act)))

        # Progressive snapshot (if available) — capture 1d, 7d, 30d
        prog = obs.get('progressive', {}) if isinstance(obs, dict) else {}
        p1 = prog.get('1d', {}) if isinstance(prog, dict) else {}
        p7 = prog.get('7d', {}) if isinstance(prog, dict) else {}
        p30 = prog.get('30d', {}) if isinstance(prog, dict) else {}

        def _extract(p: Dict[str, Any], key: str) -> Optional[float]:
            v = p.get(key, None)
            return float(v) if isinstance(v, (int, float)) else None

        er1, cf1, sl1, tp1 = _extract(p1, 'expected_return'), _extract(p1, 'confidence'), _extract(p1, 'sl'), _extract(p1, 'tp')
        er7, cf7, sl7, tp7 = _extract(p7, 'expected_return'), _extract(p7, 'confidence'), _extract(p7, 'sl'), _extract(p7, 'tp')
        er30, cf30, sl30, tp30 = _extract(p30, 'expected_return'), _extract(p30, 'confidence'), _extract(p30, 'sl'), _extract(p30, 'tp')
        sg1 = p1.get('signal', None)
        sg7 = p7.get('signal', None)
        sg30 = p30.get('signal', None)

        # append decision row now (aligned with this step)
        decisions.append({
            "time": dates[-1] if dates else str(t),
            "action": int(info.get('action', act)),
            "price": float(info.get('price', 0.0)),
            "equity": float(info.get('equity', 0.0)),
            "position_frac": float(pos_fracs[-1]) if pos_fracs else 0.0,
            # Progressive 1d
            "prog_1d_er": er1,
            "prog_1d_conf": cf1,
            "prog_1d_sig": str(sg1) if isinstance(sg1, str) else ("" if sg1 is None else str(sg1)),
            "prog_1d_sl": sl1,
            "prog_1d_tp": tp1,
            # Progressive 7d
            "prog_7d_er": er7,
            "prog_7d_conf": cf7,
            "prog_7d_sig": str(sg7) if isinstance(sg7, str) else ("" if sg7 is None else str(sg7)),
            "prog_7d_sl": sl7,
            "prog_7d_tp": tp7,
            # Progressive 30d
            "prog_30d_er": er30,
            "prog_30d_conf": cf30,
            "prog_30d_sig": str(sg30) if isinstance(sg30, str) else ("" if sg30 is None else str(sg30)),
            "prog_30d_sl": sl30,
            "prog_30d_tp": tp30
        })
        step_count += 1

    # Metrics
    metrics: Dict[str, Any] = {}
    if equities:
        eq = np.array(equities, dtype=float)
        ret = np.diff(eq) / np.maximum(eq[:-1], 1e-8)
        total_return = (eq[-1] / max(eq[0], 1e-8)) - 1.0
        # Max Drawdown
        peak = np.maximum.accumulate(eq)
        dd = (eq - peak) / np.maximum(peak, 1e-8)
        max_dd = float(dd.min()) if dd.size else 0.0
        # Sharpe approx (daily)
        if ret.size > 1:
            sharpe = float((np.mean(ret) / (np.std(ret) + 1e-8)) * np.sqrt(252.0))
        else:
            sharpe = 0.0
        metrics = {
            "final_equity": float(eq[-1]),
            "total_return": float(total_return),
            "max_drawdown": float(max_dd),
            "sharpe": float(sharpe)
        }

    # Coverage metrics using market calendar if available and explicit dates provided
    try:
        if start_date and end_date:
            from pathlib import Path
            import pandas as pd
            repo_root = Path(__file__).resolve().parents[1]
            cal_csv = repo_root / "data" / "rl" / "calendars" / "market_calendar.csv"
            if cal_csv.exists():
                cal = pd.read_csv(cal_csv)
                cal['Date'] = pd.to_datetime(cal['Date'])
                sdt = pd.to_datetime(start_date)
                edt = pd.to_datetime(end_date)
                cal_range = cal[(cal['Date'] >= sdt) & (cal['Date'] <= edt)]
                calendar_days = int(len(cal_range))
                # Count symbol days present in env.df within range
                df_idx = env.df.index
                if not isinstance(df_idx, pd.DatetimeIndex):
                    try:
                        idx = pd.to_datetime(df_idx)
                    except Exception:
                        idx = pd.Index(df_idx)
                else:
                    idx = df_idx
                symbol_days = int(((idx >= sdt) & (idx <= edt)).sum())
                coverage_ratio = float(symbol_days / calendar_days) if calendar_days > 0 else None
                metrics.update({
                    "calendar_days": calendar_days,
                    "symbol_days": symbol_days,
                    "coverage_ratio": coverage_ratio
                })
    except Exception:
        pass

    # Decisions log
    # decisions already collected inside the loop

    return {
        "symbol": symbol,
        "times": times,
        "dates": dates,
        "prices": prices,
        "equities": equities,
    "ohlc": ohlc,
        "position_fractions": pos_fracs,
        "actions": actions,
        "metrics": metrics,
        "decisions": decisions
    }
