"""
Gymnasium wrapper for the custom MarketEnv so it can be used with Stable-Baselines3.

Observation: flatten windowed features into a single float32 vector.
Currently uses two series from MarketEnv observation:
- ret window (returns)
- close window normalized by last close in the window

Action space: Discrete(3) -> {flat, half long, full long}
"""

from __future__ import annotations

from typing import Tuple, Any
import numpy as np

from rl.envs.market_env import MarketEnv

import gymnasium as gym  # type: ignore


class MarketEnvGym(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, env: MarketEnv) -> None:
        super().__init__()
        self._env = env

        # derive feature size from a fresh reset
        obs = self._env.reset()
        feats = self._to_features(obs)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(feats.shape[0],), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(3)

    def _to_features(self, obs: dict) -> np.ndarray:
        fw = obs.get('features_window', {}) if isinstance(obs, dict) else {}
        ret = np.asarray(fw.get('ret', []), dtype=np.float32)
        close = np.asarray(fw.get('close', []), dtype=np.float32)
        if close.size > 0:
            base = close[-1] if close[-1] != 0 else (np.mean(close) if np.mean(close) != 0 else 1.0)
            close_norm = close / float(base)
        else:
            close_norm = close
        feats = np.concatenate([ret, close_norm]).astype(np.float32)
        return feats

    def reset(self, seed: int | None = None, options: dict | None = None) -> Tuple[np.ndarray, dict]:
        if seed is not None:
            # no RNG inside env yet
            pass
        obs = self._env.reset()
        feats = self._to_features(obs)
        return feats, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        obs, reward, done, info = self._env.step(int(action))
        feats = self._to_features(obs)
        terminated = bool(done)
        truncated = False
        return feats, float(reward), terminated, truncated, info

    # Optional helpers for SB3 compatibility
    def render(self) -> Any:  # pragma: no cover
        return None

    def close(self) -> None:  # pragma: no cover
        return None
