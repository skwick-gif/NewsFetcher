from __future__ import annotations

from typing import Dict, List

import numpy as np

from .base import StrategyResult


def _notional(qty: float, px: float) -> float:
    return abs(float(qty) * float(px))


def _apply_costs(notional: float, tcost_bps: int, slip_bps: int) -> float:
    return float(notional) * ((float(tcost_bps or 0) + float(slip_bps or 0)) / 10000.0)


def run(df, params: Dict) -> StrategyResult:
    initial_cash = float(params.get('initial_cash', 10000.0) or 10000.0)
    tcost = int(params.get('transaction_cost_bps', 5) or 5)
    slip = int(params.get('slippage_bps', 5) or 5)

    cash = float(initial_cash)
    shares = 0.0
    equity_curve: List[float] = []
    trades: List[dict] = []
    in_position: List[int] = []
    stops_count = 0

    prev_macd = None
    prev_sig = None

    dates = df['Date'].dt.date.tolist() if 'Date' in df.columns else [None] * len(df)

    for i in range(len(df)):
        px = float(df.loc[i, 'Close'])
        m = float(df.loc[i, 'macd'])
        s = float(df.loc[i, 'macd_signal'])
        action = None
        reason = None

        if np.isfinite(m) and np.isfinite(s) and prev_macd is not None and prev_sig is not None:
            crossed_up = (prev_macd <= prev_sig) and (m > s)
            crossed_dn = (prev_macd >= prev_sig) and (m < s)
            if crossed_up and cash > 0:
                qty = cash / px
                fee = _apply_costs(_notional(qty, px), tcost, slip)
                cash -= (qty * px + fee)
                shares += qty
                action = 'BUY'; reason = 'cross_up'
            elif crossed_dn and shares > 0:
                fee = _apply_costs(_notional(shares, px), tcost, slip)
                cash += (shares * px - fee)
                shares = 0.0
                action = 'SELL'; reason = 'cross_down'

        equity = cash + shares * px
        equity_curve.append(float(equity))
        in_position.append(1 if shares > 0 else 0)
        if action:
            trades.append({
                'date': str(dates[i]) if dates[i] is not None else None,
                'action': action,
                'reason': reason,
                'price': px,
                'shares': float(shares),
                'cash': float(cash),
                'equity': float(equity),
            })

        prev_macd = m if np.isfinite(m) else prev_macd
        prev_sig = s if np.isfinite(s) else prev_sig

    return StrategyResult(trades=trades, in_position=in_position, equity_curve=equity_curve, stops_count=stops_count)
