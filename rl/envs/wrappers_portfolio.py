"""
Gymnasium wrapper for PortfolioEnv, exposing a vector observation and Box action space.
The env itself accepts any array-like and maps it to weights via softmax internally.
"""

from __future__ import annotations

from typing import Tuple, Any
import numpy as np
import gymnasium as gym  # type: ignore

from rl.envs.portfolio_env import PortfolioEnv


class PortfolioEnvGym(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, env: PortfolioEnv) -> None:
        super().__init__()
        self._env = env
        # derive observation size from reset
        obs = self._env.reset()
        feats = self._to_features(obs)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(feats.shape[0],), dtype=np.float32)
        # action: one scalar per asset (unbounded, will be softmaxed)
        self.action_space = gym.spaces.Box(low=-5.0, high=5.0, shape=(len(self._env.symbols),), dtype=np.float32)

    def _to_features(self, obs: dict) -> np.ndarray:
        fw = obs.get('features_window', {}) if isinstance(obs, dict) else {}
        # concatenate per-symbol ret and close_norm windows
        parts = []
        # Keep symbol order stable
        # PortfolioEnv exposes env.symbols
        # ret first, then close
        for s in getattr(self._env, 'symbols', sorted(fw.keys())):
            f = fw.get(s, {})
            r = np.asarray(f.get('ret', []), dtype=np.float32)
            c = np.asarray(f.get('close', []), dtype=np.float32)
            parts.append(r)
            parts.append(c)
            # Append any extra feature arrays present (e.g., 'news_*') in stable key order
            for k in sorted([k for k in f.keys() if k not in ('ret', 'close')]):
                v = np.asarray(f.get(k, []), dtype=np.float32)
                parts.append(v)
        feats = np.concatenate(parts).astype(np.float32) if parts else np.zeros((1,), dtype=np.float32)
        return feats

    def reset(self, seed: int | None = None, options: dict | None = None) -> Tuple[np.ndarray, dict]:
        if seed is not None:
            pass
        obs = self._env.reset()
        feats = self._to_features(obs)
        return feats, {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        obs, reward, done, info = self._env.step(action)
        feats = self._to_features(obs)
        terminated = bool(done)
        truncated = False
        return feats, float(reward), terminated, truncated, info

    def render(self) -> Any:  # pragma: no cover
        return None

    def close(self) -> None:  # pragma: no cover
        return None


class NoTradeBandActionWrapper(gym.ActionWrapper):
    """
    Action wrapper that suppresses small rebalances (per-symbol Δweight below band)
    and enforces a minimum number of business days between trades.

    This works by transforming the action vector into logits that produce the
    desired post-band target weights through the env's softmax mapping.
    """

    def __init__(self, env: gym.Env, band: float = 0.05, min_days: int = 0) -> None:
        super().__init__(env)
        self.band = float(max(0.0, band))
        self.min_days = int(max(0, min_days))
        self._last_trade_idx = -10_000

    def action(self, action: np.ndarray) -> np.ndarray:
        # Access underlying PortfolioEnv
        base = getattr(self.env, "_env", None)
        if base is None or not isinstance(base, PortfolioEnv) or (self.band <= 0.0 and self.min_days <= 0):
            return action

        # Compute target weights from incoming action
        try:
            target_w = base._to_weights(np.asarray(action, dtype=float))
        except Exception:
            return action

        # Compute current weights
        prices_t = base.prices[base._idx]
        eq = base.state.equity(prices_t)
        pos_val = base.state.shares * prices_t
        cur_w = (pos_val / eq) if eq > 0 else np.zeros_like(pos_val)

        # Enforce band and min-days
        desired = cur_w.copy()
        allow_trade_day = (base._idx - self._last_trade_idx) >= self.min_days
        if allow_trade_day:
            delta = target_w - cur_w
            mask = np.abs(delta) >= self.band
            if mask.any():
                desired[mask] = target_w[mask]
                # normalize and clip
                desired = np.clip(desired, 0.0, 1.0)
                s = desired.sum()
                if s > 0:
                    desired = desired / s
                self._last_trade_idx = base._idx
        # else: hold cur_w

        # Map desired weights back to logits for softmax
        eps = 1e-8
        w = np.clip(desired, eps, 1.0)
        w = w / w.sum()
        logits = np.log(w)
        return logits.astype(np.float32)
