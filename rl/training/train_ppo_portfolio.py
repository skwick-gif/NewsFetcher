"""
Train PPO on the multi-asset PortfolioEnv (continuous actions).

Usage (Windows PowerShell):
  py -3 -m rl.training.train_ppo_portfolio --symbols QQQ,MBLY,TNA \
     --timesteps 200000 --window 60 --start 2020-01-01 --end 2023-12-31 \
     --eval-start 2024-01-01 --eval-end 2024-12-31 --checkpoint-freq 50000 \
     --tensorboard-log rl\tb
"""

from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import pandas as pd


def _load_news_features(csv_path: str, symbols: List[str], start: str | None, end: str | None, cols: List[str]) -> Dict[str, pd.DataFrame]:
    df = pd.read_csv(csv_path)
    # Normalize columns
    colmap = {c.lower(): c for c in df.columns}
    date_col = colmap.get('date') or colmap.get('timestamp') or colmap.get('published_at')
    sym_col = colmap.get('symbol') or colmap.get('ticker')
    if not date_col:
        raise ValueError("news-features CSV must have a Date-like column")
    if not sym_col:
        raise ValueError("news-features CSV must have a Symbol column")
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.rename(columns={date_col: 'Date', sym_col: 'Symbol'})
    # Keep requested columns if present
    keep = ['Date', 'Symbol'] + [c for c in cols if c in df.columns]
    df = df[keep].copy()
    if start:
        df = df[df['Date'] >= pd.to_datetime(start)]
    if end:
        df = df[df['Date'] <= pd.to_datetime(end)]
    # Build per-symbol DataFrames indexed by Date
    out: Dict[str, pd.DataFrame] = {}
    for s in symbols:
        sub = df[df['Symbol'] == s].drop(columns=['Symbol']).copy()
        if not sub.empty:
            sub = sub.sort_values('Date').set_index('Date')
            out[s] = sub
    return out


def _load_vix_features(vix_symbol: str, start: str | None, end: str | None) -> pd.DataFrame:
    """Load VIX daily features as exogenous inputs (not tradable).
    Returns a DataFrame indexed by Date with columns like 'vix_ret'.
    Uses yfinance for simplicity and broad availability.
    """
    import pandas as pd
    import yfinance as yf
    df = yf.download(vix_symbol, start=start, end=end, progress=False)
    if df is None or len(df) == 0:
        return pd.DataFrame()
    df = df.reset_index()
    # Handle possible MultiIndex columns from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        def pick(col):
            parts = [str(x).strip() for x in (list(col) if isinstance(col, tuple) else [col])]
            for p in parts:
                if p.lower() in ('open','high','low','close','adj close','volume','date','datetime'):
                    return p.title() if p.lower() != 'adj close' else 'Adj Close'
            return parts[0] if parts else str(col)
        df.columns = [pick(c) for c in df.columns]
    # Normalize date and compute returns
    date_col = 'Date' if 'Date' in df.columns else ('Datetime' if 'Datetime' in df.columns else None)
    if date_col is None:
        return pd.DataFrame()
    df['Date'] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=['Date']).set_index('Date').sort_index()
    if 'Close' not in df.columns and 'Adj Close' in df.columns:
        df['Close'] = df['Adj Close']
    if 'Close' not in df.columns:
        return pd.DataFrame()
    df['vix_ret'] = df['Close'].pct_change().fillna(0.0)
    return df[['vix_ret']]


def make_env(
    symbols: List[str],
    window: int,
    start: str | None,
    end: str | None,
    transaction_cost_bps: int = 5,
    slippage_bps: int = 5,
    starting_cash: float = 10000.0,
    max_weight: float | None = None,
    news_features_csv: str | None = None,
    news_cols: List[str] | None = None,
    news_window: int = 1,
    no_trade_band: float = 0.0,
    band_min_days: int = 0,
    turnover_penalty_bps: int = 0,
    # Broker/commission model
    broker: str = 'bps',
    ibkr_per_share: float = 0.0035,
    ibkr_min_per_order: float = 0.35,
    sec_fee_rate: float = 0.000008,
    # VIX feature
    add_vix_feature: bool = False,
    vix_symbol: str = '^VIX',
):
    from rl.envs.portfolio_env import PortfolioEnv
    from rl.envs.wrappers_portfolio import PortfolioEnvGym, NoTradeBandActionWrapper
    extra_map: Dict[str, pd.DataFrame] | None = None
    # Load news features per symbol
    if news_features_csv:
        extra_map = _load_news_features(news_features_csv, symbols, start, end, news_cols or [])
    # Optionally load VIX as exogenous feature and replicate to all symbols
    vix_cols: List[str] = []
    if add_vix_feature:
        vdf = _load_vix_features(vix_symbol, start, end)
        if not vdf.empty:
            vix_cols = ['vix_ret']
            if extra_map is None:
                extra_map = {}
            for s in symbols:
                if s in (extra_map or {}):
                    extra_map[s] = extra_map[s].join(vdf, how='left').fillna(0.0)
                else:
                    extra_map[s] = vdf.copy()
    env = PortfolioEnv.load_from_local_universe(
        symbols=symbols,
        window=window,
        start_date=start,
        end_date=end,
        transaction_cost_bps=int(transaction_cost_bps),
        slippage_bps=int(slippage_bps),
        starting_cash=float(starting_cash),
        max_weight=(float(max_weight) if max_weight is not None else None),
        extra_features=extra_map,
        extra_feature_cols=((news_cols or []) + vix_cols),
        extra_window=int(news_window),
        turnover_penalty_bps=int(turnover_penalty_bps),
        broker=broker,
        ibkr_per_share=float(ibkr_per_share),
        ibkr_min_per_order=float(ibkr_min_per_order),
        sec_fee_rate=float(sec_fee_rate),
    )
    base = PortfolioEnvGym(env)
    if float(no_trade_band) > 0.0 or int(band_min_days) > 0:
        base = NoTradeBandActionWrapper(base, band=float(no_trade_band), min_days=int(band_min_days))
    return base


def main():
    parser = argparse.ArgumentParser(description="Train PPO on PortfolioEnv (multi-asset)")
    parser.add_argument('--symbols', required=True, help='Comma-separated list of tickers (must exist under stock_data/<SYMBOL>/)')
    parser.add_argument('--timesteps', type=int, default=200000, help='Total training timesteps')
    parser.add_argument('--window', type=int, default=60, help='Feature window size')
    parser.add_argument('--start', type=str, default=None, help='Start date YYYY-MM-DD (optional)')
    parser.add_argument('--end', type=str, default=None, help='End date YYYY-MM-DD (optional)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'], help="Compute device: 'auto' picks cuda if available, else cpu")
    # Costs and constraints
    parser.add_argument('--transaction-cost-bps', type=int, default=5, help='Per-trade transaction cost in bps')
    parser.add_argument('--slippage-bps', type=int, default=5, help='Per-trade slippage in bps')
    parser.add_argument('--starting-cash', type=float, default=10000.0, help='Initial portfolio cash')
    parser.add_argument('--max-weight', type=float, default=None, help='Optional max weight per asset (e.g., 0.6)')
    parser.add_argument('--turnover-penalty-bps', type=int, default=0, help='Extra reward penalty per bps of turnover notional (0=disabled)')
    parser.add_argument('--eval-start', type=str, default=None, help='Eval start date YYYY-MM-DD (optional)')
    parser.add_argument('--eval-end', type=str, default=None, help='Eval end date YYYY-MM-DD (optional)')
    parser.add_argument('--checkpoint-freq', type=int, default=50000, help='Save a checkpoint every N steps (0 to disable)')
    parser.add_argument('--tensorboard-log', type=str, default=None, help='TensorBoard log dir (optional)')
    # Optional news features
    parser.add_argument('--news-features-csv', type=str, default=None, help='Path to ml/data/news_features.csv (optional)')
    parser.add_argument('--news-cols', type=str, default=None, help='Comma list of news feature columns to include (e.g., news_count,llm_relevant_count,avg_score,fda_count,china_count,geopolitics_count)')
    parser.add_argument('--news-window', type=int, default=1, help='Number of recent days of news features to include per column')
    # Broker commission model
    parser.add_argument('--broker', type=str, default='bps', choices=['bps', 'ibkr'], help="Commission model: 'bps' linear bps or 'ibkr' per-share + SEC fee")
    parser.add_argument('--ibkr-per-share', type=float, default=0.0035, help='IBKR commission $ per share (e.g., 0.0035)')
    parser.add_argument('--ibkr-min-per-order', type=float, default=0.35, help='IBKR minimum commission $ per order per asset per step')
    parser.add_argument('--sec-fee-rate', type=float, default=0.000008, help='SEC fee rate applied to sell notional (approx)')
    # VIX feature
    parser.add_argument('--add-vix-feature', action='store_true', help='Include VIX as exogenous feature (vix_ret)')
    parser.add_argument('--vix-symbol', type=str, default='^VIX', help='Symbol to use for VIX features (default ^VIX)')
    # Optional no-trade band (training-time constraint)
    parser.add_argument('--no-trade-band', type=float, default=0.0, help='Suppress rebalances below this per-symbol Δweight during training (0=disabled)')
    parser.add_argument('--band-min-days', type=int, default=0, help='Min business days between trades during training (0=disabled)')
    args = parser.parse_args()

    try:
        from stable_baselines3 import PPO  # type: ignore
        from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, EvalCallback  # type: ignore
        import torch  # type: ignore
    except Exception as e:
        raise ImportError("stable-baselines3 (and torch) are required. Please install them to run PPO training.") from e

    symbols = [s.strip() for s in args.symbols.split(',') if s.strip()]
    if len(symbols) < 2:
        raise ValueError("Provide at least two symbols for portfolio training")

    # Build env
    env = make_env(
        symbols,
        args.window,
        args.start,
        args.end,
        transaction_cost_bps=args.transaction_cost_bps,
        slippage_bps=args.slippage_bps,
        starting_cash=args.starting_cash,
        max_weight=args.max_weight,
        news_features_csv=args.news_features_csv,
        news_cols=([c.strip() for c in args.news_cols.split(',')] if args.news_cols else None),
        news_window=args.news_window,
        no_trade_band=args.no_trade_band,
        band_min_days=args.band_min_days,
        turnover_penalty_bps=args.turnover_penalty_bps,
        broker=args.broker,
        ibkr_per_share=args.ibkr_per_share,
        ibkr_min_per_order=args.ibkr_min_per_order,
        sec_fee_rate=args.sec_fee_rate,
        add_vix_feature=bool(args.add_vix_feature),
        vix_symbol=args.vix_symbol,
    )

    # Model; PPO handles Box action spaces
    # Select device
    if args.device == 'auto':
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    # Log device info for visibility
    try:
        avail = torch.cuda.is_available()
        n_gpus = torch.cuda.device_count()
        msg = f"Using device: {device} | torch.cuda.is_available()={avail} | cuda_device_count={n_gpus}"
        if device == 'cuda' and avail and n_gpus > 0:
            try:
                name0 = torch.cuda.get_device_name(0)
                msg += f" | cuda[0]={name0}"
            except Exception:
                pass
        print(msg)
    except Exception:
        pass
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=args.seed,
        tensorboard_log=args.tensorboard_log,
        device=device,
        # Reasonable defaults; can be tuned further
        n_steps=2048,
        batch_size=64,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        clip_range=0.2,
        vf_coef=0.5,
        policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),
    )

    # Callbacks
    callbacks = []
    out_dir = Path(__file__).resolve().parents[1] / "models" / "ppo_portfolio"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.checkpoint_freq and args.checkpoint_freq > 0:
        ckpt_dir = out_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        callbacks.append(CheckpointCallback(save_freq=int(args.checkpoint_freq), save_path=ckpt_dir.as_posix(), name_prefix=f"ppo_portfolio_{'-'.join(symbols)}"))

    # Optional evaluation callback on separate date range
    if args.eval_start or args.eval_end:
        eval_env = make_env(
            symbols,
            args.window,
            args.eval_start,
            args.eval_end,
            transaction_cost_bps=args.transaction_cost_bps,
            slippage_bps=args.slippage_bps,
            starting_cash=args.starting_cash,
            max_weight=args.max_weight,
            news_features_csv=args.news_features_csv,
            news_cols=([c.strip() for c in args.news_cols.split(',')] if args.news_cols else None),
            news_window=args.news_window,
            no_trade_band=args.no_trade_band,
            band_min_days=args.band_min_days,
            turnover_penalty_bps=args.turnover_penalty_bps,
            broker=args.broker,
            ibkr_per_share=args.ibkr_per_share,
            ibkr_min_per_order=args.ibkr_min_per_order,
            sec_fee_rate=args.sec_fee_rate,
            add_vix_feature=bool(args.add_vix_feature),
            vix_symbol=args.vix_symbol,
        )
        callbacks.append(EvalCallback(eval_env, best_model_save_path=(out_dir / "best").as_posix(),
                                      log_path=(out_dir / "eval_logs").as_posix(), eval_freq=max(10000, args.checkpoint_freq or 10000),
                                      deterministic=True, render=False))

    model.learn(total_timesteps=int(args.timesteps), callback=(CallbackList(callbacks) if callbacks else None))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"ppo_portfolio_{'-'.join(symbols)}_{stamp}.zip"
    model.save(out_path.as_posix())
    print(f"Saved PPO portfolio model to: {out_path}")


if __name__ == "__main__":
    main()
