# RL Data & Environment Pipeline

This document records the current RL stack components and how to validate them locally while IBKR connectivity is still pending.

## Components Overview

| Layer | Location | Notes |
| --- | --- | --- |
| Data adapter | `rl/data_adapters/local_stock_data.py` | Loads price/indicator CSVs from `stock_data/<SYMBOL>/`. Parses dates, prunes invalid rows, merges indicators without duplicating OHLCV columns. Optional progressive signal pivot via `data/rl/progressive_signals/`. |
| Single-asset env | `rl/envs/market_env.py` | Discrete action space {flat, half long, full long}. Handles per-step commissions (bps or IBKR-like per-share) and exposes a Gym-ready observation dict. |
| Portfolio env | `rl/envs/portfolio_env.py` | Continuous actions mapped to portfolio weights via softmax; supports turnover penalties and IBKR fee model. |
| Gym wrappers | `rl/envs/wrappers*.py` | Convert custom env observations/actions to Gymnasium spaces for stable-baselines3. |
| Simulation API | `rl/simulation.py` | Runs deterministic policies (e.g., follow_trend) for UI charts (`/api/rl/simulate`). |
| PPO training | `rl/training/train_ppo*.py` | Command-line entry points for single-symbol and multi-asset PPO training with checkpointing/eval Callbacks. |

## Validation Workflow

1. **Unit tests**: `pytest -q tests/test_rl_envs.py` builds synthetic price data to verify `LocalStockData`, `MarketEnv`, `PortfolioEnv`, and their Gym wrappers.
2. **Simulation check**:
   ```powershell
   py -c "from rl.simulation import run_simulation; import json; res = run_simulation('MBLY', days=120); print(len(res['prices'])); print(json.dumps(res['metrics'], indent=2))"
   ```
   Confirm that the function returns price/equity series and metric dict.
3. **PPO smoke test**:
   ```powershell
   py -m rl.training.train_ppo --symbol MBLY --timesteps 1000 --window 30 --checkpoint-freq 0 --start 2020-01-01 --end 2024-01-01 --eval-start 2024-01-02 --eval-end 2024-06-30
   ```
   Should finish quickly and store a model under `rl/models/ppo/`.

## Next Steps Before IBKR Integration

- Automate PPO job logging & surface last-run metadata on RL Dashboard.
- Formalize data QA (missing dates, NaNs) in `LocalStockData` before live feeds.
- Define risk guardrails for future IBKR order routing (max leverage, kill switches).
- Keep `docs/RL_DASHBOARD_GUIDE_HE.md` aligned with the new pipeline.
