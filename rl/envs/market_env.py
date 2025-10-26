"""
Gym-like Market Environment (single-asset, discrete actions) — MVP implementation.

Contract:
- reset() -> observation dict
- step(action) -> (observation, reward, done, info)

Observation:
- features_window: list[float] or a tiny dict (MVP uses Close and returns)
- portfolio: {position_frac, cash, equity}
- price: current price

Action (MVP):
- discrete: {0, 1, 2} -> {flat, half long, full long}

Reward:
- change in equity (PnL) between steps minus transaction costs
"""

from typing import Any, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np
import pandas as pd

from rl.data_adapters.local_stock_data import LocalStockData


@dataclass
class Portfolio:
    cash: float
    shares: float

    def equity(self, price: float) -> float:
        return float(self.cash + self.shares * price)


class MarketEnv:
    def __init__(self, symbol: str, data: pd.DataFrame, window: int = 60,
                 transaction_cost_bps: int = 5, slippage_bps: int = 5,
                 starting_cash: float = 10000.0,
                 # Broker/commission model
                 broker: str = 'bps',
                 ibkr_per_share: float = 0.0035,
                 ibkr_min_per_order: float = 0.35,
                 sec_fee_rate: float = 0.000008) -> None:
        self.symbol = symbol
        self.df = data.copy()
        self.window = int(max(2, window))
        self.cost_bps = int(max(0, transaction_cost_bps))
        self.slip_bps = int(max(0, slippage_bps))
        self.total_bps = self.cost_bps + self.slip_bps
        # Broker commission settings
        self.broker = str(broker or 'bps').lower()
        self.ibkr_per_share = float(max(0.0, ibkr_per_share))
        self.ibkr_min_per_order = float(max(0.0, ibkr_min_per_order))
        self.sec_fee_rate = float(max(0.0, sec_fee_rate))

        # precompute simple returns
        self.df['ret'] = self.df['Close'].pct_change().fillna(0.0)

        # choose a safe start index even if there are fewer rows than window
        n = len(self.df)
        if n < 2:
            raise ValueError(f"Not enough data for {symbol}: need >=2 rows, got {n}")
        self._start_idx = int(min(self.window, n - 1))
        self._idx = self._start_idx
        self._done = False
        self.portfolio = Portfolio(cash=starting_cash, shares=0.0)
        self._last_equity = self.portfolio.equity(float(self.df['Close'].iloc[self._idx]))

    @staticmethod
    def load_from_local(symbol: str, window: int = 60, tail_days: Optional[int] = None,
                        start_date: Optional[str] = None, end_date: Optional[str] = None,
                        transaction_cost_bps: int = 5, slippage_bps: int = 5,
                        starting_cash: float = 10000.0,
                        broker: str = 'bps',
                        ibkr_per_share: float = 0.0035,
                        ibkr_min_per_order: float = 0.35,
                        sec_fee_rate: float = 0.000008) -> "MarketEnv":
        adapter = LocalStockData()
        bundle = adapter.load_symbol(symbol, tail_days=tail_days, start_date=start_date, end_date=end_date)
        dfm = bundle['merged']
        # Merge progressive signals if available (wide pivot)
        sigs = adapter.load_progressive_signals_pivot(symbol)
        if sigs is not None:
            # align on index; ensure dfm has DateTimeIndex
            if not isinstance(dfm.index, pd.DatetimeIndex):
                dfm = dfm.copy()
                if 'Date' in dfm.columns:
                    dfm['Date'] = pd.to_datetime(dfm['Date'])
                    dfm = dfm.set_index('Date')
            dfm = dfm.join(sigs, how='left')
        # keep OHLCV when present + progressive features
        base_cols = {'Open', 'High', 'Low', 'Close', 'Volume'}
        keep_cols = [c for c in dfm.columns if (c in base_cols) or c.startswith('expected_return_') or c.startswith('confidence_') or c.startswith('signal_') or c.startswith('sl_') or c.startswith('tp_')]
        df = dfm[keep_cols].copy() if keep_cols else dfm[['Close']].copy()
        return MarketEnv(symbol, df, window, transaction_cost_bps, slippage_bps, starting_cash,
                         broker=broker, ibkr_per_share=ibkr_per_share,
                         ibkr_min_per_order=ibkr_min_per_order, sec_fee_rate=sec_fee_rate)

    def _obs(self) -> Dict[str, Any]:
        w = self.df.iloc[self._idx - self.window:self._idx]
        close_window = w['Close'].values.astype(float)
        ret_window = w['ret'].values.astype(float) if 'ret' in w.columns else np.zeros_like(close_window)
        price = float(self.df['Close'].iloc[self._idx])
        eq = self.portfolio.equity(price)
        pos_value = self.portfolio.shares * price
        pos_frac = float(0.0 if eq <= 0 else pos_value / eq)
        # Progressive signals at t-1 (as-of)
        prog = {}
        tminus1 = self._idx - 1
        if tminus1 >= 0:
            row = self.df.iloc[tminus1]
            for horizon in ('1d', '7d', '30d'):
                er_col = f"expected_return_{horizon}"
                cf_col = f"confidence_{horizon}"
                sg_col = f"signal_{horizon}"
                sl_col = f"sl_{horizon}"
                tp_col = f"tp_{horizon}"
                if er_col in self.df.columns or cf_col in self.df.columns or sg_col in self.df.columns or sl_col in self.df.columns or tp_col in self.df.columns:
                    prog[horizon] = {
                        "expected_return": float(row.get(er_col, 0.0)) if er_col in self.df.columns and pd.notna(row.get(er_col, None)) else 0.0,
                        "confidence": float(row.get(cf_col, 0.0)) if cf_col in self.df.columns and pd.notna(row.get(cf_col, None)) else 0.0,
                        "signal": (str(row.get(sg_col)) if sg_col in self.df.columns and pd.notna(row.get(sg_col, None)) else ""),
                        "sl": float(row.get(sl_col)) if sl_col in self.df.columns and pd.notna(row.get(sl_col, None)) else None,
                        "tp": float(row.get(tp_col)) if tp_col in self.df.columns and pd.notna(row.get(tp_col, None)) else None
                    }
        return {
            "features_window": {
                "close": close_window.tolist(),
                "ret": ret_window.tolist()
            },
            "portfolio": {
                "position_frac": pos_frac,
                "cash": float(self.portfolio.cash),
                "equity": float(eq)
            },
            "price": price,
            "progressive": prog,
            "t": int(self._idx)
        }

    def reset(self) -> Dict[str, Any]:
        self._idx = self._start_idx
        self._done = False
        # reset portfolio but keep starting cash
        self.portfolio.shares = 0.0
        # last equity at start
        price = float(self.df['Close'].iloc[self._idx])
        self._last_equity = self.portfolio.equity(price)
        return self._obs()

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if self._done:
            return self._obs(), 0.0, True, {"reason": "episode_done"}

        action = int(action)
        # Map discrete action to target position fraction
        mapping = {0: 0.0, 1: 0.5, 2: 1.0}
        target_frac = mapping.get(action, 0.0)

        # Current price and equity
        price = float(self.df['Close'].iloc[self._idx])
        equity = self.portfolio.equity(price)
        target_value = target_frac * equity
        current_value = self.portfolio.shares * price
        delta_value = target_value - current_value

        # Convert to shares change
        delta_shares = 0.0 if price <= 0 else (delta_value / price)
        # Commission model
        if self.broker == 'ibkr':
            commission = max(self.ibkr_min_per_order, self.ibkr_per_share * abs(delta_shares))
            sell_notional = max(0.0, -delta_shares) * price
            sec_fee = self.sec_fee_rate * sell_notional
            trade_cost = commission + sec_fee
        else:
            trade_cost = abs(delta_shares) * price * (self.total_bps / 10000.0)

        # Execute trade
        cash_change = -(delta_shares * price) - trade_cost
        self.portfolio.cash += cash_change
        self.portfolio.shares += delta_shares

        # Advance time
        prev_equity = equity
        self._idx += 1
        if self._idx >= len(self.df):
            self._idx = len(self.df) - 1
            self._done = True

        # Compute reward on next price
        next_price = float(self.df['Close'].iloc[self._idx])
        next_equity = self.portfolio.equity(next_price)
        reward = float(next_equity - prev_equity)
        self._last_equity = next_equity

        info = {
            "price": next_price,
            "equity": next_equity,
            "action": action,
            "position_frac": 0.0 if next_equity <= 0 else (self.portfolio.shares * next_price) / next_equity
        }

        return self._obs(), reward, self._done, info
