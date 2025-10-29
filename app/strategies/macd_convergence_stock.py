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
    """Check if histogram has been falling for k consecutive days."""
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
    MACD Convergence Stock Strategy - Enhanced with ATR-based trailing stops.
    
    Entry Logic (State: Looking for Buy):
    - Filter 1: ADX > 20 (trending market)
    - Filter 2: MACD < 0 AND Signal < 0 (negative zone)
    - Filter 3: Volume < VOL_SMA (seller exhaustion)
    - Filter 4: Histogram rising for k_buy days
    - Filter 5: conv_ratio <= 40%
    - Trigger: conv_ratio <= 25% OR MACD > Signal (bullish cross)
    
    Exit Logic (State: Managing Position):
    - Check 1: Price <= Trailing Stop (T_Stop = max(T_Stop, Price - 2*ATR))
    - Check 2: Price >= buy_price * 1.20 (20% take profit)
    - Check 3 (MACD < 0): Histogram falling for k_sell days → Failed Rally
    - Check 3 (MACD >= 0): MACD < Signal (bearish cross) → End of Trend
    """
    initial_cash = float(params.get('initial_cash', 10000.0) or 10000.0)
    tcost = int(params.get('transaction_cost_bps', 5) or 5)
    slip = int(params.get('slippage_bps', 5) or 5)
    k_buy = int(params.get('pre_bars', 3) or 3)
    k_sell = int(params.get('sell_bars', 2) or 2)
    adx_min = float(params.get('adx_min', 20) or 20)
    atr_multiplier = float(params.get('atr_multiplier', 2.0) or 2.0)
    take_profit_pct = float(params.get('take_profit_pct', 20.0) or 20.0)
    p_buy = float(params.get('p_buy', 40) or 40)
    e_buy = float(params.get('e_buy', 25) or 25)
    vol_sma_period = int(params.get('vol_sma_period', 20) or 20)

    cash = float(initial_cash)
    shares = 0.0
    equity_curve: List[float] = []
    trades: List[dict] = []
    in_position: List[int] = []
    stops_count = 0
    entry_px = None
    trailing_stop_px = None

    prev_macd = None
    prev_sig = None
    dates = df['Date'].dt.date.tolist() if 'Date' in df.columns else [None] * len(df)
    hist = (df['macd'] - df['macd_signal']).astype(float)

    # Pre-calculate ATR and Volume SMA
    if 'High' in df.columns and 'Low' in df.columns:
        df['TR'] = df[['High', 'Low', 'Close']].apply(
            lambda row: max(row['High'] - row['Low'], 
                          abs(row['High'] - row['Close']), 
                          abs(row['Low'] - row['Close'])), 
            axis=1
        )
        df['ATR'] = df['TR'].rolling(window=14).mean()
    else:
        df['ATR'] = np.nan

    if 'Volume' in df.columns:
        df['VOL_SMA'] = df['Volume'].rolling(window=vol_sma_period).mean()
    else:
        df['VOL_SMA'] = np.nan

    for i in range(len(df)):
        px = float(df.loc[i, 'Close'])
        m = float(df.loc[i, 'macd'])
        s = float(df.loc[i, 'macd_signal'])
        h = float(hist.iloc[i])
        atr = float(df.loc[i, 'ATR']) if np.isfinite(df.loc[i, 'ATR']) else 0.0
        action = None
        reason = None

        # ===== STATE 2: Managing Position (IN_MARKET) =====
        if shares > 0:
            # Update trailing stop: T_Stop = max(T_Stop, Price - (atr_multiplier * ATR))
            new_stop = px - (atr_multiplier * atr) if atr > 0 else px * 0.92  # fallback 8% fixed stop
            if trailing_stop_px is None:
                trailing_stop_px = new_stop
            else:
                trailing_stop_px = max(trailing_stop_px, new_stop)

            # Check 1: Trailing Stop Hit
            if px <= trailing_stop_px:
                fee = _apply_costs(_notional(shares, px), tcost, slip)
                cash += (shares * px - fee)
                shares = 0.0
                action = 'SELL'
                reason = 'trailing_stop_hit'
                stops_count += 1

            # Check 2: Take Profit Hit
            elif entry_px is not None and px >= entry_px * (1.0 + take_profit_pct / 100.0):
                fee = _apply_costs(_notional(shares, px), tcost, slip)
                cash += (shares * px - fee)
                shares = 0.0
                action = 'SELL'
                reason = 'take_profit_hit'

            # Check 3: Logical Exit (MACD-based)
            elif np.isfinite(m) and np.isfinite(s):
                if m < 0:
                    # MACD still negative: Check for failed rally (histogram falling)
                    if _falling_hist(hist, i, k_sell):
                        fee = _apply_costs(_notional(shares, px), tcost, slip)
                        cash += (shares * px - fee)
                        shares = 0.0
                        action = 'SELL'
                        reason = 'failed_rally_macd_negative'
                else:
                    # MACD turned positive: Check for bearish cross (trend reversal)
                    if prev_macd is not None and prev_sig is not None:
                        crossed_dn = (prev_macd >= prev_sig) and (m < s)
                        if crossed_dn:
                            fee = _apply_costs(_notional(shares, px), tcost, slip)
                            cash += (shares * px - fee)
                            shares = 0.0
                            action = 'SELL'
                            reason = 'end_of_trend_bearish_cross'

        # ===== STATE 1: Looking for Buy (NO POSITION) =====
        elif cash > 0 and action is None:
            if not (np.isfinite(m) and np.isfinite(s)):
                pass  # Skip if MACD not available
            else:
                # Filter 1: ADX > 20 (trending market)
                adx_val = float(df.loc[i, 'adx']) if 'adx' in df.columns and np.isfinite(df.loc[i, 'adx']) else 0.0
                adx_ok = (adx_val >= adx_min)

                # Filter 2: MACD < 0 AND Signal < 0 (negative zone)
                in_negative_zone = (m < 0.0) and (s < 0.0)

                # Filter 3: Volume < VOL_SMA (seller exhaustion)
                vol = float(df.loc[i, 'Volume']) if 'Volume' in df.columns and np.isfinite(df.loc[i, 'Volume']) else 0.0
                vol_sma = float(df.loc[i, 'VOL_SMA']) if np.isfinite(df.loc[i, 'VOL_SMA']) else 0.0
                seller_exhaustion = (vol < vol_sma) if (vol > 0 and vol_sma > 0) else False

                # Filter 4: Histogram rising for k_buy days
                hist_rising = _rising_hist(hist, i, k_buy)

                # Filter 5: conv_ratio <= 40%
                cr = float(df.loc[i, 'conv_ratio']) if 'conv_ratio' in df.columns and np.isfinite(df.loc[i, 'conv_ratio']) else 100.0
                conv_ok = (cr <= p_buy / 100.0)

                # All filters must pass
                setup_ok = adx_ok and in_negative_zone and seller_exhaustion and hist_rising and conv_ok

                # Trigger: conv_ratio <= 25% OR MACD > Signal (bullish cross)
                if setup_ok:
                    trigger_conv = (cr <= e_buy / 100.0)
                    trigger_cross = False
                    if prev_macd is not None and prev_sig is not None:
                        trigger_cross = (prev_macd <= prev_sig) and (m > s)
                    
                    if trigger_conv or trigger_cross:
                        # Execute Buy
                        qty = cash / px
                        fee = _apply_costs(_notional(qty, px), tcost, slip)
                        cash -= (qty * px + fee)
                        shares += qty
                        action = 'BUY'
                        reason = 'stock_conv_entry'
                        entry_px = px
                        # Initialize trailing stop: Price - (atr_multiplier * ATR)
                        trailing_stop_px = px - (atr_multiplier * atr) if atr > 0 else px * 0.92

        # ===== Record Equity and Trades =====
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

        # Reset state when exiting position
        if shares == 0:
            entry_px = None
            trailing_stop_px = None

        # Track previous MACD values for cross detection
        prev_macd = m if np.isfinite(m) else prev_macd
        prev_sig = s if np.isfinite(s) else prev_sig

    return StrategyResult(trades=trades, in_position=in_position, equity_curve=equity_curve, stops_count=stops_count)
