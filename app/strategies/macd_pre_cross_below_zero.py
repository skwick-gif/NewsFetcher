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
    """
    MACD Pre-Cross Below Zero Strategy for ETFs - Enhanced Version
    
    Entry Logic (Original - Unchanged):
    - ADX > 20 (trending market)
    - MACD < 0 AND Signal < 0 (below zero area)
    - Histogram rising for k_buy days
    - Convergence ratio <= 40%
    - Trigger: conv_ratio <= 25% OR bullish cross
    
    Exit Logic (Enhanced):
    - Trailing Stop: Price - (1.5 * ATR) - tight protection
    - Take Profit: 12% gain - conservative target
    - Logical Exit (MACD < 0): Failed rally - histogram drops k_sell days
    - Logical Exit (MACD >= 0): Trend reversal - bearish cross
    """
    initial_cash = float(params.get('initial_cash', 3000.0) or 3000.0)
    tcost = int(params.get('transaction_cost_bps', 5) or 5)
    slip = int(params.get('slippage_bps', 5) or 5)
    pre_bars = int(params.get('pre_bars', 3) or 3)  # k_buy
    sell_bars = int(params.get('sell_bars', 2) or 2)  # k_sell
    adx_min = float(params.get('adx_min', 20) or 20)  # ADX threshold
    
    # Enhanced exit parameters
    atr_multiplier = float(params.get('atr_multiplier', 1.5) or 1.5)  # Tight trailing stop
    take_profit_pct = float(params.get('take_profit_pct', 12.0) or 12.0)  # 12% target
    
    # Entry filters
    p_buy = float(params.get('p_buy', 40) or 40)  # conv_ratio max for setup
    e_buy = float(params.get('e_buy', 25) or 25)  # conv_ratio trigger

    cash = float(initial_cash)
    shares = 0.0
    equity_curve: List[float] = []
    trades: List[dict] = []
    in_position: List[int] = []
    stops_count = 0
    entry_px = None
    trailing_stop_px = None
    
    # Calculate ATR for dynamic stops
    if 'High' in df.columns and 'Low' in df.columns and 'Close' in df.columns:
        df['TR'] = np.maximum(
            df['High'] - df['Low'],
            np.maximum(
                abs(df['High'] - df['Close'].shift(1)),
                abs(df['Low'] - df['Close'].shift(1))
            )
        )
        df['ATR'] = df['TR'].rolling(14, min_periods=1).mean()
    else:
        df['ATR'] = 0.0

    prev_macd = None
    prev_sig = None
    dates = df['Date'].dt.date.tolist() if 'Date' in df.columns else [None] * len(df)
    hist = (df['macd'] - df['macd_signal']).astype(float)

    for i in range(len(df)):
        px = float(df.loc[i, 'Close'])
        m = float(df.loc[i, 'macd'])
        s = float(df.loc[i, 'macd_signal'])
        atr = float(df.loc[i, 'ATR']) if 'ATR' in df.columns else 0.0
        
        action = None
        reason = None

        if np.isfinite(m) and np.isfinite(s) and prev_macd is not None and prev_sig is not None:
            crossed_up = (prev_macd <= prev_sig) and (m > s)
            crossed_dn = (prev_macd >= prev_sig) and (m < s)

            # ========================================
            # STATE 2: MANAGING POSITION (Exit Logic)
            # ========================================
            if shares > 0:
                # Update Trailing Stop: T_Stop = max(T_Stop, Price - (1.5 * ATR))
                new_stop = px - (atr_multiplier * atr) if atr > 0 else (px * 0.91)  # fallback 9% if no ATR
                if trailing_stop_px is None:
                    trailing_stop_px = new_stop
                else:
                    trailing_stop_px = max(trailing_stop_px, new_stop)
                
                # CHECK 1: Hit Trailing Stop?
                if px <= trailing_stop_px:
                    fee = _apply_costs(_notional(shares, px), tcost, slip)
                    cash += (shares * px - fee)
                    shares = 0.0
                    action = 'SELL'
                    reason = 'trailing_stop_hit'
                    stops_count += 1
                
                # CHECK 2: Hit Take Profit? (12%)
                elif entry_px is not None and px >= entry_px * (1.0 + take_profit_pct / 100.0):
                    fee = _apply_costs(_notional(shares, px), tcost, slip)
                    cash += (shares * px - fee)
                    shares = 0.0
                    action = 'SELL'
                    reason = 'take_profit_12pct'
                
                # CHECK 3: Logical Exit - depends on MACD position
                elif m < 0:
                    # MACD still negative - check for failed rally
                    if _falling_hist(hist, i, max(2, sell_bars)):
                        fee = _apply_costs(_notional(shares, px), tcost, slip)
                        cash += (shares * px - fee)
                        shares = 0.0
                        action = 'SELL'
                        reason = 'failed_rally_macd_negative'
                
                elif m >= 0:
                    # MACD turned positive - check for bearish cross (end of trend)
                    if crossed_dn:
                        fee = _apply_costs(_notional(shares, px), tcost, slip)
                        cash += (shares * px - fee)
                        shares = 0.0
                        action = 'SELL'
                        reason = 'end_of_trend_bearish_cross'

            # ========================================
            # STATE 1: LOOKING FOR BUY (Entry Logic)
            # ========================================
            if action is None and shares == 0:
                # Filter 1: ADX > 20 (trending market)
                adx_ok = True
                if 'adx' in df.columns and np.isfinite(df.loc[i, 'adx']):
                    adx_ok = float(df.loc[i, 'adx']) >= adx_min
                
                # Filter 2: MACD < 0 AND Signal < 0 (target area)
                in_target_area = (m < 0.0) and (s < 0.0)
                
                # Filter 3: Histogram rising for k_buy days
                hist_rising = _rising_hist(hist, i, max(2, pre_bars))
                
                # Filter 4: Convergence ratio <= 40%
                cr = df.loc[i, 'conv_ratio'] if 'conv_ratio' in df.columns else np.nan
                weak_convergence = np.isfinite(cr) and (cr <= p_buy / 100.0)
                
                # Trigger: conv_ratio <= 25% OR bullish cross
                trigger = (np.isfinite(cr) and cr <= e_buy / 100.0) or crossed_up  # RESTORED
                
                # Execute Buy if all conditions met
                if adx_ok and in_target_area and hist_rising and weak_convergence and trigger and cash > 0:
                    qty = cash / px
                    fee = _apply_costs(_notional(qty, px), tcost, slip)
                    cash -= (qty * px + fee)
                    shares += qty
                    action = 'BUY'
                    reason = 'pre_cross_convergence_etf'
                    entry_px = px
                    trailing_stop_px = px - (atr_multiplier * atr) if atr > 0 else (px * 0.91)

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
                'trailing_stop': float(trailing_stop_px) if trailing_stop_px is not None else None,
            })

        # Reset tracking when not in position
        if shares == 0:
            entry_px = None
            trailing_stop_px = None

        prev_macd = m if np.isfinite(m) else prev_macd
        prev_sig = s if np.isfinite(s) else prev_sig

    return StrategyResult(trades=trades, in_position=in_position, equity_curve=equity_curve, stops_count=stops_count)
