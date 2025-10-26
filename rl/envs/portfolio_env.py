"""
Multi-asset Portfolio Environment (continuous actions) — additive to existing single-asset env.

Contract:
- reset() -> observation dict
- step(action_vector) -> (observation, reward, done, info)

Action: continuous vector a in R^n mapped to portfolio weights via softmax.
Constraints: optional max per-asset weight and optional cash buffer (future work).

Reward: delta equity minus linear bps transaction+slippage costs based on turnover.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd

from rl.data_adapters.local_stock_data import LocalStockData


@dataclass
class PortfolioState:
    cash: float
    shares: np.ndarray  # shape (n_assets,)

    def equity(self, prices: np.ndarray) -> float:
        return float(self.cash + float(np.dot(self.shares, prices)))


class PortfolioEnv:
    def __init__(
        self,
        symbols: List[str],
        data: Dict[str, pd.DataFrame],
        window: int = 60,
        transaction_cost_bps: int = 5,
        slippage_bps: int = 5,
        starting_cash: float = 10000.0,
        max_weight: Optional[float] = None,
        extra_features: Optional[Dict[str, pd.DataFrame]] = None,
        extra_feature_cols: Optional[List[str]] = None,
        extra_window: int = 1,
        turnover_penalty_bps: int = 0,
        # Broker/commission model
        broker: str = 'bps',  # 'bps' or 'ibkr'
        ibkr_per_share: float = 0.0035,
        ibkr_min_per_order: float = 0.35,
        sec_fee_rate: float = 0.000008,
    ) -> None:
        assert len(symbols) > 1, "PortfolioEnv requires at least two symbols"
        self.symbols = symbols
        self.n = len(symbols)
        self.window = int(max(2, window))
        self.cost_bps = int(max(0, transaction_cost_bps))
        self.slip_bps = int(max(0, slippage_bps))
        self.total_bps = self.cost_bps + self.slip_bps
        self.max_weight = float(max_weight) if max_weight is not None else None
        self.turnover_penalty_bps = int(max(0, turnover_penalty_bps))
        # Broker commission settings
        self.broker = str(broker or 'bps').lower()
        self.ibkr_per_share = float(max(0.0, ibkr_per_share))
        self.ibkr_min_per_order = float(max(0.0, ibkr_min_per_order))
        self.sec_fee_rate = float(max(0.0, sec_fee_rate))

        # Align data on intersection of dates to avoid NaNs
        # Keep only Close, Volume if present
        dfs: List[pd.DataFrame] = []
        for s in symbols:
            df = data[s].copy()
            cols = [c for c in df.columns if c in ("Open", "High", "Low", "Close", "Volume")]
            if not cols:
                cols = ["Close"]
            df = df[cols].copy()
            # ensure DateTimeIndex
            if not isinstance(df.index, pd.DatetimeIndex):
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                    df = df.set_index('Date')
                else:
                    # cannot proceed safely
                    raise ValueError(f"Data for {s} must be indexed by Date or include a Date column")
            dfs.append(df)

        # Intersect indices
        idx = dfs[0].index
        for d in dfs[1:]:
            idx = idx.intersection(d.index)
        # reindex and drop NaNs in Close
        aligned: Dict[str, pd.DataFrame] = {}
        for s, df in zip(symbols, dfs):
            x = df.reindex(idx).sort_index()
            x['Close'] = pd.to_numeric(x['Close'], errors='coerce')
            x = x.dropna(subset=['Close'])
            aligned[s] = x

        # After dropping NaNs, recompute common index
        idx = None
        for s in symbols:
            idx = aligned[s].index if idx is None else idx.intersection(aligned[s].index)
        if idx is None or len(idx) < 2:
            raise ValueError("Not enough aligned data across symbols for PortfolioEnv")
        for s in symbols:
            aligned[s] = aligned[s].reindex(idx)

        self.df_map = aligned

        # Optional extra features per symbol (e.g., news features)
        self.extra_window = max(1, int(extra_window))
        self.extra_cols: List[str] = list(extra_feature_cols or [])
        self.extra_map: Dict[str, pd.DataFrame] = {}
        if extra_features:
            # Align extra features to common idx and retain selected columns
            for s in symbols:
                if s in extra_features:
                    df_ex = extra_features[s].copy()
                    if not isinstance(df_ex.index, pd.DatetimeIndex):
                        if 'Date' in df_ex.columns:
                            df_ex['Date'] = pd.to_datetime(df_ex['Date'], errors='coerce')
                            df_ex = df_ex.set_index('Date')
                        else:
                            continue
                    if self.extra_cols:
                        keep = [c for c in self.extra_cols if c in df_ex.columns]
                        df_ex = df_ex[keep].copy()
                    # numeric
                    for c in df_ex.columns:
                        df_ex[c] = pd.to_numeric(df_ex[c], errors='coerce')
                    df_ex = df_ex.sort_index().reindex(idx).fillna(0.0)
                    self.extra_map[s] = df_ex

        # Precompute returns matrix and prices matrix
        prices = np.stack([aligned[s]['Close'].values.astype(float) for s in symbols], axis=1)  # shape (T, n)
        rets = (prices[1:] / np.maximum(prices[:-1], 1e-12)) - 1.0
        rets = np.concatenate([np.zeros((1, self.n), dtype=float), rets], axis=0)
        self.prices = prices
        self.rets = rets

        # Start index respecting window
        T = prices.shape[0]
        self._start_idx = int(min(max(1, self.window), T - 1))
        self._idx = self._start_idx
        self._done = False

        self.state = PortfolioState(cash=float(starting_cash), shares=np.zeros(self.n, dtype=float))
        self._last_equity = self.state.equity(self.prices[self._idx])

    @staticmethod
    def load_from_local_universe(
        symbols: List[str],
        window: int = 60,
        tail_days: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        transaction_cost_bps: int = 5,
        slippage_bps: int = 5,
        starting_cash: float = 10000.0,
        max_weight: Optional[float] = None,
        extra_features: Optional[Dict[str, pd.DataFrame]] = None,
        extra_feature_cols: Optional[List[str]] = None,
        extra_window: int = 1,
        turnover_penalty_bps: int = 0,
        broker: str = 'bps',
        ibkr_per_share: float = 0.0035,
        ibkr_min_per_order: float = 0.35,
        sec_fee_rate: float = 0.000008,
    ) -> "PortfolioEnv":
        adapter = LocalStockData()
        data: Dict[str, pd.DataFrame] = {}
        use_syms: List[str] = []
        for s in symbols:
            if not adapter.has_symbol(s):
                continue
            bundle = adapter.load_symbol(s, tail_days=tail_days, start_date=start_date, end_date=end_date)
            dfm = bundle['merged']
            # ensure DataFrame has Close and Date index
            if 'Close' not in dfm.columns and 'close' in dfm.columns:
                dfm['Close'] = pd.to_numeric(dfm['close'], errors='coerce')
            if not isinstance(dfm.index, pd.DatetimeIndex):
                if 'Date' in dfm.columns:
                    dfm['Date'] = pd.to_datetime(dfm['Date'], errors='coerce')
                    dfm = dfm.set_index('Date')
                else:
                    continue
            dfm = dfm.sort_index()
            if dfm['Close'].dropna().shape[0] >= 2:
                data[s] = dfm
                use_syms.append(s)
        if len(use_syms) < 2:
            raise ValueError("Need at least two valid symbols with price data to build PortfolioEnv")
        return PortfolioEnv(
            use_syms,
            data,
            window,
            transaction_cost_bps,
            slippage_bps,
            starting_cash,
            max_weight,
            extra_features=extra_features,
            extra_feature_cols=extra_feature_cols,
            extra_window=extra_window,
            turnover_penalty_bps=turnover_penalty_bps,
            broker=broker,
            ibkr_per_share=ibkr_per_share,
            ibkr_min_per_order=ibkr_min_per_order,
            sec_fee_rate=sec_fee_rate,
        )

    def _obs(self) -> Dict[str, Any]:
        t0 = self._idx - self.window
        # Ensure fixed-length windows by left-padding when there is not enough history
        i0 = max(0, t0)
        w_prices_raw = self.prices[i0:self._idx, :]  # (m, n)
        w_rets_raw = self.rets[i0:self._idx, :]      # (m, n)
        m = int(w_prices_raw.shape[0])
        if m < self.window:
            pad = self.window - m
            if m > 0:
                first_prices = w_prices_raw[0:1, :]
            else:
                # If even current index has no history, synthesize from current price row
                cur = self.prices[self._idx:self._idx+1, :]
                first_prices = cur
            w_prices = np.vstack([np.repeat(first_prices, pad, axis=0), w_prices_raw])
            w_rets = np.vstack([np.zeros((pad, self.n), dtype=float), (w_rets_raw if m > 0 else np.zeros((0, self.n), dtype=float))])
        else:
            w_prices = w_prices_raw
            w_rets = w_rets_raw
        # normalize price windows by last-price per asset for scale invariance
        base = np.where(w_prices[-1, :] == 0.0, np.maximum(1.0, np.mean(w_prices, axis=0)), w_prices[-1, :])
        close_norm = (w_prices / base[np.newaxis, :]).astype(float)

        current_prices = self.prices[self._idx]
        eq = self.state.equity(current_prices)
        pos_val = self.state.shares * current_prices
        weights = (pos_val / eq) if eq > 0 else np.zeros_like(pos_val)

        # Build observation dict
        feat_map: Dict[str, Dict[str, List[float]]] = {}
        for i, s in enumerate(self.symbols):
            feat_map[s] = {
                "ret": w_rets[:, i].astype(float).tolist(),
                "close": close_norm[:, i].astype(float).tolist(),
            }
            # Append extra features window if present
            if s in self.extra_map and len(self.extra_cols) > 0:
                # Take the last extra_window rows up to current index; align by df_map index
                idx_dates = self.df_map[s].index
                cur_date = idx_dates[self._idx]
                ex_df = self.extra_map[s]
                # up to current date inclusive
                ex_df = ex_df.loc[:cur_date]
                if not ex_df.empty:
                    ex_slice = ex_df.iloc[-self.extra_window:]
                    # Flatten as [col1_window..., col2_window..., ...]
                    for c in self.extra_cols:
                        if c in ex_slice.columns:
                            feat_map[s][f"news_{c}"] = ex_slice[c].astype(float).to_numpy().tolist()
        return {
            "features_window": feat_map,
            "portfolio": {
                "weights": weights.astype(float).tolist(),
                "cash": float(self.state.cash),
                "equity": float(eq),
            },
            "t": int(self._idx),
        }

    def reset(self) -> Dict[str, Any]:
        self._idx = self._start_idx
        self._done = False
        self.state.shares[:] = 0.0
        # keep initial cash
        self._last_equity = self.state.equity(self.prices[self._idx])
        return self._obs()

    def _to_weights(self, action: np.ndarray) -> np.ndarray:
        # softmax to simplex, guard for numerical stability
        a = np.asarray(action, dtype=float).reshape(-1)
        if a.shape[0] != self.n:
            raise ValueError("Action size must match number of assets")
        a = a - np.max(a)  # shift
        exp_a = np.exp(np.clip(a, -20, 20))  # avoid overflow
        w = exp_a / np.maximum(np.sum(exp_a), 1e-12)
        if self.max_weight is not None:
            w = np.minimum(w, self.max_weight)
            s = np.sum(w)
            if s > 0:
                w = w / s
            else:
                # fallback to equal weight
                w = np.ones_like(w) / float(self.n)
        return w

    def step(self, action: Any) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if self._done:
            return self._obs(), 0.0, True, {"reason": "episode_done"}

        # Map action to weights
        if isinstance(action, (list, tuple)):
            action = np.asarray(action, dtype=float)
        elif not isinstance(action, np.ndarray):
            action = np.array([float(action)] * self.n, dtype=float)
        target_w = self._to_weights(action)

        # Current prices and equity
        prices_t = self.prices[self._idx]  # (n,)
        equity_t = self.state.equity(prices_t)
        target_value = target_w * equity_t
        current_value = self.state.shares * prices_t
        delta_value = target_value - current_value

        # Compute trades and linear costs
        # Convert delta_value to shares
        with np.errstate(divide='ignore', invalid='ignore'):
            delta_shares = np.where(prices_t > 0, delta_value / prices_t, 0.0)
        turnover_notional = np.sum(np.abs(delta_shares) * prices_t)
        # Commission model
        if self.broker == 'ibkr':
            # Per-share commission with per-order minimum, per asset
            per_share_comm = self.ibkr_per_share * np.abs(delta_shares)
            # Convert to dollars (per-share fee already in $ per share)
            comm_per_asset = np.maximum(self.ibkr_min_per_order, per_share_comm)
            commission = float(np.sum(comm_per_asset))
            # SEC fee on sells only (approx): rate * sell notional
            sell_notional = float(np.sum(np.maximum(0.0, -delta_shares) * prices_t))
            sec_fee = self.sec_fee_rate * sell_notional
            trade_cost = commission + sec_fee
        else:
            trade_cost = turnover_notional * (self.total_bps / 10000.0)

        # Execute trades
        cash_change = -np.sum(delta_shares * prices_t) - trade_cost
        self.state.cash += float(cash_change)
        self.state.shares = self.state.shares + delta_shares

        # Advance time
        prev_equity = equity_t
        self._idx += 1
        if self._idx >= self.prices.shape[0]:
            self._idx = self.prices.shape[0] - 1
            self._done = True

        prices_tp1 = self.prices[self._idx]
        next_equity = self.state.equity(prices_tp1)
        reward = float(next_equity - prev_equity)
        if self.turnover_penalty_bps > 0:
            reward -= float(turnover_notional) * (self.turnover_penalty_bps / 10000.0)
        self._last_equity = next_equity

        info = {
            "equity": float(next_equity),
            "trade_cost": float(trade_cost),
            "weights": target_w.astype(float).tolist(),
        }
        return self._obs(), reward, self._done, info
