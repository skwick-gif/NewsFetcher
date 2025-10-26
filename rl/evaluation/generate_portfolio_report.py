from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def max_drawdown(equity: np.ndarray) -> float:
    if equity.size == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / np.maximum(peak, 1e-8)
    return float(dd.min())


def cagr(equity: np.ndarray, periods_per_year: int = 252) -> float:
    if equity.size == 0:
        return 0.0
    years = max(len(equity) / periods_per_year, 1e-8)
    return float((equity[-1] / max(equity[0], 1e-8)) ** (1 / years) - 1.0)


def sharpe_ratio(returns: np.ndarray, rf_daily: Optional[np.ndarray] = None) -> float:
    if returns.size < 2:
        return 0.0
    if rf_daily is not None and rf_daily.size == returns.size:
        ex = returns - rf_daily
    else:
        ex = returns
    mu = ex.mean()
    sd = ex.std() + 1e-8
    return float((mu / sd) * np.sqrt(252.0))


def load_rf_series(repo_root: Path, start: pd.Timestamp, end: pd.Timestamp) -> Optional[pd.Series]:
    econ_csv = repo_root / "stock_data" / "economic_data" / "combined_economic_data.csv"
    if not econ_csv.exists():
        return None
    try:
        df = pd.read_csv(econ_csv)
        if "Date" not in df.columns or "IRX" not in df.columns:
            return None
        df["Date"] = pd.to_datetime(df["Date"])  # robust
        ser = df.set_index("Date")["IRX"].sort_index()
        rf_daily = (ser / 100.0) / 252.0
        rf_daily = rf_daily[(rf_daily.index >= start) & (rf_daily.index <= end)].ffill()
        return rf_daily
    except Exception:
        return None


@dataclass
class SeriesResult:
    label: str
    dates: List[pd.Timestamp]
    equity: List[float]
    total_return: float
    max_dd: float
    sharpe: float
    cagr: float
    kind: str  # 'ppo_portfolio' | 'equal_weight' | 'qqq_bh'


def _find_model_for_symbols(model_dir: Path, symbols: List[str]) -> Path:
    """Pick the most recent model whose embedded symbol set matches exactly.
    Model filename pattern from training: ppo_portfolio_<SYM1-SYM2-...>_<STAMP>.zip
    We compare sets (order-insensitive). If none match, raise with guidance.
    """
    cand = sorted(model_dir.glob("ppo_portfolio_*.zip"))
    if not cand:
        raise FileNotFoundError(f"No portfolio models found in {model_dir}")
    target_set = set(symbols)
    matches: List[Path] = []
    for p in cand:
        name = p.stem  # without .zip
        # Expected tokens: ['ppo', 'portfolio', '<SYMS>', '<DATE>', '<TIME>']
        try:
            tokens = name.split('_')
            if len(tokens) < 5 or tokens[0] != 'ppo' or tokens[1] != 'portfolio':
                continue
            sym_part = tokens[2]
            sym_list = [s for s in sym_part.split('-') if s]
            if set(sym_list) == target_set:
                matches.append(p)
        except Exception:
            continue
    if matches:
        return sorted(matches)[-1]
    # No exact symbol-set match; provide a clear error instead of silently picking a wrong model
    raise FileNotFoundError(
        f"No model found matching symbols {sorted(target_set)}. Train a model for this exact set or adjust symbols."
    )


def run_portfolio_policy(
    repo_root: Path,
    symbols: List[str],
    start: str,
    end: str,
    news_features_csv: Optional[str] = None,
    news_cols: Optional[List[str]] = None,
    news_window: int = 1,
) -> Tuple[List[pd.Timestamp], List[float]]:
    from rl.training.train_ppo_portfolio import make_env
    from stable_baselines3 import PPO  # type: ignore

    # Find latest portfolio model matching the symbol set
    model_dir = repo_root / "rl" / "models" / "ppo_portfolio"
    model_path = _find_model_for_symbols(model_dir, symbols)

    env = make_env(
        symbols,
        60,
        start,
        end,
        news_features_csv=news_features_csv,
        news_cols=news_cols,
        news_window=news_window,
    )
    model = PPO.load(model_path.as_posix())

    # Validate observation size matches model expectation; if not, retry without news extras,
    # optionally try default news features, then fail fast with a helpful error.
    try:
        expected = int(getattr(model.policy.observation_space, 'shape', [None])[0])
    except Exception:
        expected = None  # fallback
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    cur_dim = int(np.array(obs, dtype=float).shape[0])
    if expected is not None and cur_dim != expected:
        # Try rebuild with no news extras as a common mismatch source
        env = make_env(symbols, 60, start, end, news_features_csv=None, news_cols=None, news_window=1)
        obs2 = env.reset()
        if isinstance(obs2, tuple):
            obs2 = obs2[0]
        cur_dim2 = int(np.array(obs2, dtype=float).shape[0])
        if cur_dim2 != expected:
            # Heuristic: attempt default news cols if available and looks plausible
            base_dim = len(symbols) * (2 * 60)
            needed = expected - base_dim
            tried_default = False
            if needed > 0 and needed % max(1, len(symbols)) == 0:
                default_csv = (repo_root / 'ml' / 'data' / 'news_features.csv')
                default_cols = ['news_count','llm_relevant_count','avg_score','fda_count','china_count','geopolitics_count','sentiment_avg']
                if default_csv.exists():
                    tried_default = True
                    env = make_env(symbols, 60, start, end, news_features_csv=default_csv.as_posix(), news_cols=default_cols, news_window=1)
                    obs3 = env.reset()
                    if isinstance(obs3, tuple):
                        obs3 = obs3[0]
                    cur_dim3 = int(np.array(obs3, dtype=float).shape[0])
                    if cur_dim3 == expected:
                        # matched using defaults; proceed
                        pass
                    else:
                        raise ValueError(
                            f"Observation shape mismatch: model expects {expected}, but env produced {cur_dim} (with provided), {cur_dim2} (no news) and {cur_dim3} (default news). "
                            f"Please pass --news-features-csv and --news-cols matching training, or retrain."
                        )
            if not tried_default:
                raise ValueError(
                    f"Observation shape mismatch: model expects {expected}, but env produced {cur_dim} (with provided) and {cur_dim2} (no news). "
                    f"Ensure you evaluate with the same feature config (news_cols/news_window/VIX) and window as training."
                )
        # else: proceed with env without news

    # Refresh obs after potential rebuild
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]

    equities: List[float] = []
    dates: List[pd.Timestamp] = []
    weights_rows: List[List[float]] = []
    done = False
    steps = 0

    while not done and steps < 2_000_000:
        action, _ = model.predict(obs, deterministic=True)
        res = env.step(action)
        if isinstance(res, tuple) and len(res) == 4:
            obs, reward, done, info = res
        elif isinstance(res, tuple) and len(res) == 5:
            obs, reward, terminated, truncated, info = res
            done = bool(terminated) or bool(truncated)
        else:
            raise RuntimeError("Unexpected env.step() return format")
        if isinstance(obs, tuple):
            obs = obs[0]
        equities.append(float(info.get("equity", 0.0)))
        w = info.get("weights")
        if isinstance(w, (list, tuple)):
            weights_rows.append([float(x) for x in w])
        else:
            weights_rows.append([np.nan] * len(symbols))
        # extract date from underlying env if possible
        try:
            base_env = getattr(env, "_env", None) or getattr(env, "unwrapped", None) or env
            idx = base_env.df_map[symbols[0]].index  # type: ignore[attr-defined]
            t = int(info.get("t", steps))
            dates.append(pd.Timestamp(idx[t]))
        except Exception:
            dates.append(pd.NaT)
        steps += 1

    # try repair dates via market calendar if NaT
    if any(pd.isna(d) for d in dates):
        cal_csv = repo_root / "data" / "rl" / "calendars" / "market_calendar.csv"
        if cal_csv.exists():
            cal = pd.read_csv(cal_csv)
            cal["Date"] = pd.to_datetime(cal["Date"])  # robust
            sdt, edt = pd.to_datetime(start), pd.to_datetime(end)
            rng = cal[(cal["Date"] >= sdt) & (cal["Date"] <= edt)]["Date"].tolist()
            if len(rng) >= len(dates):
                dates = rng[: len(dates)]

    return dates, equities


def run_portfolio_policy_with_weights(
    repo_root: Path,
    symbols: List[str],
    start: str,
    end: str,
    news_features_csv: Optional[str] = None,
    news_cols: Optional[List[str]] = None,
    news_window: int = 1,
) -> Tuple[List[pd.Timestamp], List[float], pd.DataFrame]:
    dates, equities = run_portfolio_policy(
        repo_root,
        symbols,
        start,
        end,
        news_features_csv=news_features_csv,
        news_cols=news_cols,
        news_window=news_window,
    )
    # Re-roll to extract weights aligned to dates
    from rl.training.train_ppo_portfolio import make_env
    from stable_baselines3 import PPO  # type: ignore
    model_dir = repo_root / "rl" / "models" / "ppo_portfolio"
    model_path = _find_model_for_symbols(model_dir, symbols)
    env = make_env(symbols, 60, start, end, news_features_csv=news_features_csv, news_cols=news_cols, news_window=news_window)
    model = PPO.load(model_path.as_posix())
    # If obs mismatch, try drop news extras or default news (consistent with run_portfolio_policy)
    try:
        expected = int(getattr(model.policy.observation_space, 'shape', [None])[0])
    except Exception:
        expected = None
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    if expected is not None and int(np.array(obs).shape[0]) != expected:
        env = make_env(symbols, 60, start, end, news_features_csv=None, news_cols=None, news_window=1)
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        if int(np.array(obs).shape[0]) != expected:
            # Try default news
            default_csv = (repo_root / 'ml' / 'data' / 'news_features.csv')
            default_cols = ['news_count','llm_relevant_count','avg_score','fda_count','china_count','geopolitics_count','sentiment_avg']
            if default_csv.exists():
                env = make_env(symbols, 60, start, end, news_features_csv=default_csv.as_posix(), news_cols=default_cols, news_window=1)
                obs = env.reset()
                if isinstance(obs, tuple):
                    obs = obs[0]
    rows: List[List[float]] = []
    done = False
    steps = 0
    while not done and steps < len(dates):
        action, _ = model.predict(obs, deterministic=True)
        res = env.step(action)
        if isinstance(res, tuple) and len(res) == 4:
            obs, reward, done, info = res
        else:
            obs, reward, terminated, truncated, info = res  # type: ignore[misc]
            done = bool(terminated) or bool(truncated)
        if isinstance(obs, tuple):
            obs = obs[0]
        w = info.get("weights")
        if isinstance(w, (list, tuple)):
            rows.append([float(x) for x in w])
        else:
            rows.append([np.nan] * len(symbols))
        steps += 1
    df_w = pd.DataFrame(rows, columns=[f"Weight_{s}" for s in symbols])
    df_w.insert(0, "Date", pd.to_datetime(dates[: len(rows)]))
    return dates, equities, df_w


def compute_banded_equity(repo_root: Path, symbols: List[str], dates: List[pd.Timestamp], weights_df: pd.DataFrame,
                          band: float, band_min_days: int, tc_bps: float, slip_bps: float, starting_equity: float = 10000.0) -> List[float]:
    # Build prices matrix aligned to given dates
    idx, prices = load_prices_matrix(repo_root, symbols, str(dates[0].date()), str(dates[-1].date()))
    # Align to dates list
    idx_target = pd.to_datetime(pd.Series(dates))
    px_df = pd.DataFrame(prices, index=idx, columns=symbols).reindex(idx_target).ffill().bfill()
    rets = px_df.pct_change().fillna(0.0).to_numpy(dtype=float)
    # Align weights to idx_target
    w_cols = [c for c in weights_df.columns if c.startswith("Weight_")]
    w = weights_df.set_index("Date")[w_cols].reindex(idx_target).ffill().fillna(0.0).to_numpy(dtype=float)
    # simulate
    band = float(max(0.0, band))
    band_min_days = int(max(0, band_min_days))
    total_bps = float(tc_bps) + float(slip_bps)
    equity = np.zeros((len(idx_target),), dtype=float)
    equity[0] = float(starting_equity)
    prev_w = np.zeros((w.shape[1],), dtype=float)
    last_trade = -10_000
    for t in range(len(idx_target)):
        target = w[t]
        new_w = prev_w.copy()
        if (t - last_trade) >= band_min_days:
            delta = target - prev_w
            mask = np.abs(delta) >= band
            if mask.any():
                new_w[mask] = target[mask]
                new_w = np.clip(new_w, 0.0, 1.0)
                s = new_w.sum()
                if s > 0:
                    new_w = new_w / s
                turnover = np.abs(new_w - prev_w).sum()
                if turnover > 0 and t > 0:
                    equity[t-1] = max(0.0, equity[t-1] * (1.0 - turnover * (total_bps / 10_000.0)))
                last_trade = t
        prev_w = new_w
        if t > 0:
            r = float(np.dot(prev_w, rets[t]))
            equity[t] = equity[t-1] * (1.0 + r)
    return list(equity)


def load_prices_matrix(repo_root: Path, symbols: List[str], start: str, end: str) -> Tuple[pd.DatetimeIndex, np.ndarray]:
    # Load Close prices for symbols, align by intersection
    from rl.data_adapters.local_stock_data import LocalStockData
    adapter = LocalStockData(root=repo_root / "stock_data")
    dfs: List[pd.DataFrame] = []
    use: List[str] = []
    sdt, edt = pd.to_datetime(start), pd.to_datetime(end)
    for s in symbols:
        if not adapter.has_symbol(s):
            continue
        dfm = adapter.load_symbol(s, start_date=start, end_date=end)["merged"]
        if not isinstance(dfm.index, pd.DatetimeIndex):
            continue
        if "Close" not in dfm.columns:
            continue
        df = dfm[["Close"]].copy().sort_index()
        df = df[(df.index >= sdt) & (df.index <= edt)]
        dfs.append(df)
        use.append(s)
    if len(dfs) < 1:
        raise ValueError("No valid price data for baseline computation")
    idx = dfs[0].index
    for d in dfs[1:]:
        idx = idx.intersection(d.index)
    aligned = [d.reindex(idx).sort_index() for d in dfs]
    mat = np.stack([d["Close"].to_numpy(dtype=float) for d in aligned], axis=1)
    return idx, mat


def baseline_equal_weight(repo_root: Path, symbols: List[str], start: str, end: str) -> Tuple[List[pd.Timestamp], List[float]]:
    idx, prices = load_prices_matrix(repo_root, symbols, start, end)
    rets = (prices[1:] / np.maximum(prices[:-1], 1e-12)) - 1.0  # (T-1, N)
    ew_rets = rets.mean(axis=1)  # daily rebalanced equal-weight
    equity = [10000.0]
    for r in ew_rets:
        equity.append(equity[-1] * (1.0 + float(r)))
    dates = list(idx)
    return dates, equity


def baseline_buy_hold(repo_root: Path, symbol: str, start: str, end: str) -> Tuple[List[pd.Timestamp], List[float]]:
    idx, prices = load_prices_matrix(repo_root, [symbol], start, end)
    p = prices[:, 0]
    equity = [10000.0]
    for t in range(1, len(p)):
        r = (p[t] / max(p[t - 1], 1e-12)) - 1.0
        equity.append(equity[-1] * (1.0 + float(r)))
    dates = list(idx)
    return dates, equity


def baseline_equal_weight_weekly(repo_root: Path, symbols: List[str], start: str, end: str) -> Tuple[List[pd.Timestamp], List[float]]:
    """Equal-weight portfolio rebalanced weekly (every Monday or first day)."""
    idx, prices = load_prices_matrix(repo_root, symbols, start, end)
    rets = (prices[1:] / np.maximum(prices[:-1], 1e-12)) - 1.0  # (T-1, N)
    equity = [10000.0]
    n = prices.shape[1]
    w = np.ones((n,), dtype=float) / float(n)
    for t in range(1, len(idx)):
        # Rebalance on Mondays or first step
        if t == 1 or idx[t].weekday() == 0:
            w = np.ones((n,), dtype=float) / float(n)
        r_vec = rets[t - 1]
        pr = float(np.dot(w, r_vec))
        equity.append(equity[-1] * (1.0 + pr))
        # Drift weights (no transaction costs modeled here)
        # After daily return, weights become w * (1+r) normalized
        gross = w * (1.0 + r_vec)
        s = float(np.sum(gross))
        if s > 0:
            w = gross / s
        else:
            w = np.ones((n,), dtype=float) / float(n)
    dates = list(idx)
    return dates, equity


def summarize_series(repo_root: Path, label: str, kind: str, dates: List[pd.Timestamp], equity_list: List[float], rf: Optional[pd.Series]) -> SeriesResult:
    equity = np.array(equity_list, dtype=float)
    rets = np.diff(equity) / np.maximum(equity[:-1], 1e-8) if len(equity) > 1 else np.array([])
    rf_daily = None
    if rf is not None and len(dates) > 1:
        ds = pd.Series(dates)
        if not isinstance(ds.iloc[0], pd.Timestamp):
            ds = pd.to_datetime(ds)
        rf_aligned = rf.reindex(ds, method="ffill")
        rf_daily = rf_aligned.iloc[1:].to_numpy(dtype=float)
    tr = float(equity[-1] / max(equity[0], 1e-8) - 1.0) if len(equity) else 0.0
    sr = sharpe_ratio(rets, rf_daily)
    mdd = max_drawdown(equity)
    cg = cagr(equity)
    return SeriesResult(label=label, dates=dates, equity=equity_list, total_return=tr, max_dd=mdd, sharpe=sr, cagr=cg, kind=kind)


def save_outputs(repo_root: Path, out_dir: Path, results: List[SeriesResult]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in results:
        rows.append({
            "label": r.label,
            "kind": r.kind,
            "total_return": r.total_return,
            "max_drawdown": r.max_dd,
            "sharpe": r.sharpe,
            "cagr": r.cagr,
            "points": len(r.equity),
        })
    pd.DataFrame(rows).to_csv(out_dir / "summary.csv", index=False)

    for r in results:
        df = pd.DataFrame({"Date": r.dates, "Equity": r.equity})
        df.to_csv(out_dir / f"equity_{r.kind}_{r.label.replace('/', '-')}.csv", index=False)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for r in results:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(r.dates, r.equity, label=f"{r.kind}: {r.label}")
            ax.set_title(f"Equity Curve — {r.label} ({r.kind})")
            ax.set_xlabel("Date")
            ax.set_ylabel("Equity")
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.autofmt_xdate()
            fig.tight_layout()
            fig.savefig(out_dir / f"equity_{r.kind}_{r.label.replace('/', '-')}.png", dpi=150)
            plt.close(fig)
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="Generate evaluation report for PPO Portfolio model with baselines")
    parser.add_argument("--symbols", required=True, help="Comma-separated symbols, e.g., QQQ,MBLY,TNA")
    parser.add_argument("--eval-start", default="2024-01-01")
    parser.add_argument("--eval-end", default="2024-12-31")
    parser.add_argument("--out", default=None)
    # Optional risk overlay arguments
    parser.add_argument("--risk-overlay", action="store_true", help="Apply evaluation-time risk manager overlay")
    parser.add_argument("--regime-symbol", default=None, help="Symbol to drive regime filter (default: QQQ if present, else first symbol)")
    parser.add_argument("--sma-window", type=int, default=200, help="SMA window for trend filter")
    parser.add_argument("--vol-window", type=int, default=30, help="Window for realized vol (days)")
    parser.add_argument("--vol-threshold", type=float, default=0.25, help="Annualized vol threshold to scale down")
    parser.add_argument("--vol-scale", type=float, default=0.5, help="Scale factor when vol above threshold")
    parser.add_argument("--trend-risk-off", type=float, default=0.3, help="Scale factor when price below SMA (risk-off)")
    parser.add_argument("--kill-mdd", type=float, default=0.20, help="Kill-switch drawdown threshold (fraction, e.g., 0.2=20%)")
    # Optional news features to align env obs with training
    parser.add_argument("--news-features-csv", default=None, help="Path to ml/data/news_features.csv to include in observations")
    parser.add_argument("--news-cols", default=None, help="Comma list of news feature columns to include")
    parser.add_argument("--news-window", type=int, default=1, help="Number of recent days of news features to include per column")
    # Optional banded rebalancing at evaluation
    parser.add_argument("--no-trade-band", type=float, default=0.0, help="Suppress rebalances below this per-symbol Δweight during evaluation (0=disabled)")
    parser.add_argument("--band-min-days", type=int, default=0, help="Min business days between banded rebalances (0=disabled)")
    parser.add_argument("--band-transaction-cost-bps", type=float, default=5.0, help="Transaction cost bps for banded simulation")
    parser.add_argument("--band-slippage-bps", type=float, default=5.0, help="Slippage bps for banded simulation")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    symbols = [s.strip() for s in args.symbols.split(',') if s.strip()]

    if args.out:
        out_dir = Path(args.out)
        if not out_dir.is_absolute():
            out_dir = repo_root / "reports" / "rl" / args.out
    else:
        stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
        out_dir = repo_root / "reports" / "rl" / f"portfolio_report_{stamp}"

    sdt, edt = pd.to_datetime(args.eval_start), pd.to_datetime(args.eval_end)
    rf = load_rf_series(repo_root, sdt, edt)

    results: List[SeriesResult] = []

    # PPO Portfolio (raw)
    news_cols = [c.strip() for c in args.news_cols.split(',')] if args.news_cols else None
    news_csv = args.news_features_csv
    p_dates, p_eq, p_w = run_portfolio_policy_with_weights(
        repo_root,
        symbols,
        args.eval_start,
        args.eval_end,
        news_features_csv=news_csv,
        news_cols=news_cols,
        news_window=int(args.news_window),
    )
    results.append(summarize_series(repo_root, label='-'.join(symbols), kind='ppo_portfolio', dates=p_dates, equity_list=p_eq, rf=rf))

    # PPO Portfolio (banded) — evaluation-time no-trade band
    if float(args.no_trade_band) > 0.0 or int(args.band_min_days) > 0:
        try:
            p_eq_b = compute_banded_equity(
                repo_root,
                symbols,
                p_dates,
                p_w,
                band=float(args.no_trade_band),
                band_min_days=int(args.band_min_days),
                tc_bps=float(args.band_transaction_cost_bps),
                slip_bps=float(args.band_slippage_bps),
                starting_equity=float(p_eq[0] if len(p_eq) else 10000.0),
            )
            results.append(summarize_series(repo_root, label='-'.join(symbols), kind='ppo_portfolio_banded', dates=p_dates, equity_list=p_eq_b, rf=rf))
        except Exception:
            pass

    # Optional: Apply Risk Manager overlay at evaluation time (no retraining)
    if args.risk_overlay:
        regime_symbol = args.regime_symbol or ("QQQ" if "QQQ" in symbols else symbols[0])
        try:
            # Build scaling series aligned to dates[1:] (returns timeline)
            idx, prices = load_prices_matrix(repo_root, [regime_symbol], args.eval_start, args.eval_end)
            px = pd.Series(prices[:, 0], index=idx)
            daily_ret = px.pct_change().fillna(0.0)
            sma = px.rolling(args.sma_window, min_periods=1).mean()
            trend_scale = pd.Series(np.where(px > sma, 1.0, float(args.trend_risk_off)), index=idx)
            # Realized vol (annualized) over vol_window
            # Use daily std * sqrt(252)
            vol_ann = daily_ret.rolling(args.vol_window, min_periods=1).std().fillna(0.0) * np.sqrt(252.0)
            vol_scale = pd.Series(np.where(vol_ann > args.vol_threshold, float(args.vol_scale), 1.0), index=idx)
            scale = (trend_scale * vol_scale).clip(lower=0.0, upper=1.0)
            # Align to p_dates
            date_idx = pd.to_datetime(pd.Series(p_dates))
            scale_aligned = scale.reindex(date_idx, method="ffill").fillna(1.0)
            # Apply to returns of PPO equity
            eq = np.array(p_eq, dtype=float)
            if eq.size >= 2:
                rets = np.diff(eq) / np.maximum(eq[:-1], 1e-8)
                # returns correspond to dates[1:]
                mult = scale_aligned.iloc[1:].to_numpy(dtype=float)
                if mult.size != rets.size:
                    # Fallback: length mismatch, broadcast last value
                    mult = np.resize(mult, rets.size)
                scaled_rets = rets * mult
                # Kill-switch: if drawdown exceeds threshold, flatten thereafter
                new_eq = [eq[0]]
                peak = eq[0]
                killed = False
                for i, r in enumerate(scaled_rets, start=1):
                    if not killed:
                        cur = new_eq[-1] * (1.0 + float(r))
                        peak = max(peak, cur)
                        dd = (cur - peak) / max(peak, 1e-8)
                        if dd <= -abs(args.kill_mdd):
                            killed = True
                            cur = new_eq[-1]  # flatten from this step
                    else:
                        cur = new_eq[-1]
                    new_eq.append(cur)
                p_eq_risk = new_eq
            else:
                p_eq_risk = p_eq
            results.append(summarize_series(repo_root, label='-'.join(symbols), kind='ppo_portfolio_risk', dates=p_dates, equity_list=p_eq_risk, rf=rf))
        except Exception as e:
            # If overlay fails, continue without blocking report
            results.append(summarize_series(repo_root, label='-'.join(symbols)+" (overlay_failed)", kind='ppo_portfolio_risk', dates=p_dates, equity_list=p_eq, rf=rf))

    # Equal-Weight baseline (daily rebalanced)
    ew_dates, ew_eq = baseline_equal_weight(repo_root, symbols, args.eval_start, args.eval_end)
    results.append(summarize_series(repo_root, label='EW(' + ','.join(symbols) + ')', kind='equal_weight', dates=ew_dates, equity_list=ew_eq, rf=rf))

    # Equal-Weight baseline (weekly rebalanced)
    ew_w_dates, ew_w_eq = baseline_equal_weight_weekly(repo_root, symbols, args.eval_start, args.eval_end)
    results.append(summarize_series(repo_root, label='EW_weekly(' + ','.join(symbols) + ')', kind='equal_weight_weekly', dates=ew_w_dates, equity_list=ew_w_eq, rf=rf))

    # QQQ Buy-and-Hold baseline
    if 'QQQ' in symbols:
        bh_dates, bh_eq = baseline_buy_hold(repo_root, 'QQQ', args.eval_start, args.eval_end)
        results.append(summarize_series(repo_root, label='QQQ', kind='qqq_bh', dates=bh_dates, equity_list=bh_eq, rf=rf))

    save_outputs(repo_root, out_dir, results)
    print(f"Portfolio evaluation report written to: {out_dir}")


if __name__ == "__main__":
    main()
