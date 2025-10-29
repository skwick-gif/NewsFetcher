from __future__ import annotations

from typing import Dict, List

import numpy as np

from .base import StrategyResult


def _notional(qty: float, px: float) -> float:
    return abs(float(qty) * float(px))


def _apply_costs(notional: float, tcost_bps: int, slip_bps: int) -> float:
    return float(notional) * ((float(tcost_bps or 0) + float(slip_bps or 0)) / 10000.0)


def _rising_hist(hist, i: int, k: int) -> bool:
    if i - k < 1:
        return False
    for j in range(i - k + 1, i + 1):
        if not (np.isfinite(hist.iloc[j]) and np.isfinite(hist.iloc[j-1])):
            return False
        if not (hist.iloc[j] > hist.iloc[j-1]):
            return False
    return True


def _falling_hist(hist, i: int, k: int) -> bool:
    if i - k < 1:
        return False
    for j in range(i - k + 1, i + 1):
        if not (np.isfinite(hist.iloc[j]) and np.isfinite(hist.iloc[j-1])):
            return False
        if not (hist.iloc[j] < hist.iloc[j-1]):
            return False
    return True


def run(df, params: Dict) -> StrategyResult:
    initial_cash = float(params.get('initial_cash', 10000.0) or 10000.0)
    tcost = int(params.get('transaction_cost_bps', 5) or 5)
    slip = int(params.get('slippage_bps', 5) or 5)
    pre_bars = int(params.get('pre_bars', 3) or 3)
    sell_bars = int(params.get('sell_bars', 2) or 2)
    adx_min = float(params.get('adx_min', 25) or 25)
    stop_loss_pct = float(params.get('stop_loss_pct', 0) or 0)
    trailing_stop = bool(params.get('trailing_stop', True))
    p_buy = float(params.get('p_buy', 35) or 35)
    e_buy = float(params.get('e_buy', 20) or 20)
    p_sell = float(params.get('p_sell', 40) or 40)

    cash = float(initial_cash)
    shares = 0.0
    equity_curve: List[float] = []
    trades: List[dict] = []
    in_position: List[int] = []
    stops_count = 0
    entry_px = None
    peak_px_in_pos = None

    prev_macd = None
    prev_sig = None
    dates = df['Date'].dt.date.tolist() if 'Date' in df.columns else [None] * len(df)
    hist = (df['macd'] - df['macd_signal']).astype(float)

    for i in range(len(df)):
        px = float(df.loc[i, 'Close'])
        m = float(df.loc[i, 'macd'])
        s = float(df.loc[i, 'macd_signal'])
        action = None
        reason = None

        if np.isfinite(m) and np.isfinite(s) and prev_macd is not None and prev_sig is not None:
            crossed_up = (prev_macd <= prev_sig) and (m > s)
            crossed_dn = (prev_macd >= prev_sig) and (m < s)

            # Stop-loss check first (if in position)
            if shares > 0 and stop_loss_pct > 0:
                basis = peak_px_in_pos if (trailing_stop and peak_px_in_pos is not None) else (entry_px if entry_px is not None else px)
                trigger_px = float(basis) * (1.0 - float(stop_loss_pct)/100.0)
                if px <= trigger_px:
                    fee = _apply_costs(_notional(shares, px), tcost, slip)
                    cash += (shares * px - fee)
                    shares = 0.0
                    action = 'SELL'; reason = 'stoploss'
                    stops_count += 1

            # If not stopped, evaluate SELL rules
            if action is None and shares > 0 and crossed_dn:
                fee = _apply_costs(_notional(shares, px), tcost, slip)
                cash += (shares * px - fee)
                shares = 0.0
                action = 'SELL'; reason = 'cross_down'

            if action is None and shares > 0 and (m > s):
                cr = df.loc[i, 'conv_ratio'] if 'conv_ratio' in df.columns else np.nan
                if _falling_hist(hist, i, max(2, sell_bars)) and (np.isfinite(cr) and cr <= float(p_sell)/100.0):
                    fee = _apply_costs(_notional(shares, px), tcost, slip)
                    cash += (shares * px - fee)
                    shares = 0.0
                    action = 'SELL'; reason = 'hist_falling_conv'

            # BUY rule (pre-cross below zero + rising hist + ADX filter + convergence small)
            if action is None and (m <= s) and (m < 0.0) and (s < 0.0):
                cr = df.loc[i, 'conv_ratio'] if 'conv_ratio' in df.columns else np.nan
                buy_setup = _rising_hist(hist, i, max(2, pre_bars)) and (np.isfinite(cr) and cr <= float(p_buy)/100.0)
                buy_trigger = crossed_up or (np.isfinite(cr) and cr <= float(e_buy)/100.0)
                adx_ok = (('adx' not in df.columns) or (not np.isfinite(df.loc[i, 'adx'])) or (float(df.loc[i, 'adx']) >= float(adx_min)))
                if buy_setup and buy_trigger and adx_ok and cash > 0:
                    qty = cash / px
                    fee = _apply_costs(_notional(qty, px), tcost, slip)
                    cash -= (qty * px + fee)
                    shares += qty
                    action = 'BUY'; reason = 'pre_cross_convergence'
                    entry_px = px
                    peak_px_in_pos = px

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

        # Track peak while in position for trailing stop
        if shares > 0:
            if peak_px_in_pos is None or px > peak_px_in_pos:
                peak_px_in_pos = px
        else:
            entry_px = None
            peak_px_in_pos = None

        prev_macd = m if np.isfinite(m) else prev_macd
        prev_sig = s if np.isfinite(s) else prev_sig

    return StrategyResult(trades=trades, in_position=in_position, equity_curve=equity_curve, stops_count=stops_count)
