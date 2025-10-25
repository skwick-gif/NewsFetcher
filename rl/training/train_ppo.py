"""
PPO training entrypoint for MarketEnv using Stable-Baselines3 and Gymnasium.

Minimal usage (Windows PowerShell):
    python rl/training/train_ppo.py --symbol AAPL --timesteps 100000 --window 60 --start 2018-01-01 --end 2020-12-31

Extras added:
- Periodic checkpoint saving via SB3 CheckpointCallback (configurable frequency)
- Optional evaluation during training via SB3 EvalCallback (separate eval date range)
- Optional TensorBoard logging directory

Dependencies:
    - gymnasium
    - stable-baselines3 (uses PyTorch under the hood; torch is already in requirements)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime


def make_env(symbol: str, window: int, start: str | None, end: str | None):
    from rl.envs.market_env import MarketEnv
    from rl.envs.wrappers import MarketEnvGym
    env = MarketEnv.load_from_local(symbol=symbol, window=window, tail_days=None,
                                    start_date=start, end_date=end)
    return MarketEnvGym(env)


def main():
    parser = argparse.ArgumentParser(description="Train PPO on MarketEnv (single-symbol)")
    parser.add_argument('--symbol', required=True, help='Ticker symbol (must exist under stock_data/<SYMBOL>/)')
    parser.add_argument('--timesteps', type=int, default=100000, help='Total training timesteps')
    parser.add_argument('--window', type=int, default=60, help='Feature window size')
    parser.add_argument('--start', type=str, default=None, help='Start date YYYY-MM-DD (optional)')
    parser.add_argument('--end', type=str, default=None, help='End date YYYY-MM-DD (optional)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    # Optional evaluation split (use a separate date range if provided)
    parser.add_argument('--eval-start', type=str, default=None, help='Eval start date YYYY-MM-DD (optional)')
    parser.add_argument('--eval-end', type=str, default=None, help='Eval end date YYYY-MM-DD (optional)')
    # Checkpointing and logging
    parser.add_argument('--checkpoint-freq', type=int, default=50000, help='Save a checkpoint every N steps (0 to disable)')
    parser.add_argument('--tensorboard-log', type=str, default=None, help='TensorBoard log dir (optional)')
    args = parser.parse_args()

    try:
        from stable_baselines3 import PPO  # type: ignore
        from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, EvalCallback  # type: ignore
    except Exception as e:
        raise ImportError("stable-baselines3 is required. Please install it to run PPO training.") from e

    # Build env
    env = make_env(args.symbol, args.window, args.start, args.end)

    # Construct model
    model = PPO("MlpPolicy", env, verbose=1, seed=args.seed, tensorboard_log=args.tensorboard_log)

    # Prepare callbacks
    callbacks = []
    out_dir = Path(__file__).resolve().parents[1] / "models" / "ppo"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.checkpoint_freq and args.checkpoint_freq > 0:
        ckpt_dir = out_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        callbacks.append(CheckpointCallback(save_freq=int(args.checkpoint_freq), save_path=ckpt_dir.as_posix(), name_prefix=f"ppo_{args.symbol}"))

    # Optional evaluation callback using a separate date range if provided; otherwise, skip to avoid leakage
    if args.eval_start or args.eval_end:
        eval_env = make_env(args.symbol, args.window, args.eval_start, args.eval_end)
        callbacks.append(EvalCallback(eval_env, best_model_save_path=(out_dir / "best").as_posix(),
                                      log_path=(out_dir / "eval_logs").as_posix(), eval_freq=max(10000, args.checkpoint_freq or 10000),
                                      deterministic=True, render=False))

    # Train
    model.learn(total_timesteps=int(args.timesteps), callback=(CallbackList(callbacks) if callbacks else None))

    # Save
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"ppo_{args.symbol}_{stamp}.zip"
    model.save(out_path.as_posix())
    print(f"Saved PPO model to: {out_path}")


if __name__ == "__main__":
    main()
