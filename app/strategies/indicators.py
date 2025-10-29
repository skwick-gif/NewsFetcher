from __future__ import annotations

import numpy as np
import pandas as pd


def macd_series(close, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    close = pd.Series(close, dtype=float)
    ema_fast = close.ewm(span=int(fast), adjust=False).mean()
    ema_slow = close.ewm(span=int(slow), adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_sig = macd.ewm(span=int(signal), adjust=False).mean()
    macd_hist = macd - macd_sig
    return pd.DataFrame({'macd': macd, 'macd_signal': macd_sig, 'macd_hist': macd_hist})


def rsi_series(close, period: int = 14) -> pd.Series:
    close = pd.Series(close, dtype=float)
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(int(period), min_periods=int(period)).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(int(period), min_periods=int(period)).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.rename('rsi')


def stoch_series(high, low, close, k_period: int = 14, d_period: int = 3) -> tuple[pd.Series, pd.Series]:
    h = pd.Series(high, dtype=float) if high is not None else None
    l = pd.Series(low, dtype=float) if low is not None else None
    c = pd.Series(close, dtype=float)
    if h is None or l is None:
        k = pd.Series([np.nan] * len(c), name='stoch_k')
        d = pd.Series([np.nan] * len(c), name='stoch_d')
        return k, d
    lowest_low = l.rolling(int(k_period), min_periods=int(k_period)).min()
    highest_high = h.rolling(int(k_period), min_periods=int(k_period)).max()
    denom = (highest_high - lowest_low).replace(0, np.nan)
    k = (100.0 * (c - lowest_low) / denom).rename('stoch_k')
    d = k.rolling(int(d_period), min_periods=int(d_period)).mean().rename('stoch_d')
    return k, d


def adx_series(high, low, close, period: int = 14) -> pd.Series:
    high = pd.Series(high, dtype=float)
    low = pd.Series(low, dtype=float)
    close = pd.Series(close, dtype=float)
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr_s = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_dm_s = pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean()
    minus_dm_s = pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100 * (plus_dm_s / tr_s).replace({0: np.nan})
    minus_di = 100 * (minus_dm_s / tr_s).replace({0: np.nan})
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).abs()) * 100.0
    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    return pd.Series(adx, name='adx')


def kama_series(close, n: int = 10, fast: int = 2, slow: int = 30) -> pd.Series:
    prices = pd.Series(close, dtype=float)
    if len(prices) == 0:
        return pd.Series([], dtype=float, name='kama')
    change = prices.diff(n).abs()
    volatility = prices.diff().abs().rolling(n, min_periods=n).sum()
    er = change / volatility.replace(0, np.nan)
    fast_sc = 2.0 / (fast + 1.0)
    slow_sc = 2.0 / (slow + 1.0)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    kama = pd.Series(index=prices.index, dtype='float64')
    kama.iloc[0] = float(prices.iloc[0])
    for i in range(1, len(prices)):
        prev = kama.iloc[i-1]
        sci = sc.iloc[i]
        if not np.isfinite(sci):
            sci = slow_sc ** 2
        kama.iloc[i] = prev + sci * (prices.iloc[i] - prev)
    return kama.rename('kama')
