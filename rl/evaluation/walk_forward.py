from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Dict
import pandas as pd

from rl.evaluation.generate_portfolio_report import (
    run_portfolio_policy_with_weights,
    summarize_series,
    compute_banded_equity,
    load_rf_series,
)


def daterange_segments(start: pd.Timestamp, end: pd.Timestamp, segments: int) -> List[tuple[str, str, str]]:
    # Split [start, end] into roughly equal contiguous segments
    segments = max(1, int(segments))
    total_days = (end - start).days
    if total_days <= 1 or segments == 1:
        label = f"{start.date()}_{end.date()}"
        return [(label, str(start.date()), str(end.date()))]
    edges = [start + (end - start) * (i / segments) for i in range(segments + 1)]
    out = []
    for i in range(segments):
        s = pd.to_datetime(edges[i]).normalize()
        e = pd.to_datetime(edges[i+1]).normalize()
        # ensure inclusive end
        if i < segments - 1:
            e = e - pd.Timedelta(days=1)
        label = f"{s.date()}_{e.date()}"
        out.append((label, str(s.date()), str(e.date())))
    return out


def main():
    parser = argparse.ArgumentParser(description="Walk-forward evaluation across multiple sub-windows (uses latest trained PPO portfolio model)")
    parser.add_argument("--symbols", required=True, help="Comma-separated symbols, e.g., QQQ,MBLY,TNA")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--segments", type=int, default=4, help="Number of sub-windows to split [start,end] into (default 4)")
    parser.add_argument("--out", default=None, help="Optional output dir name (under reports/rl)")
    # News features (optional) to align env with training
    parser.add_argument("--news-features-csv", default=None, help="Path to ml/data/news_features.csv")
    parser.add_argument("--news-cols", default=None, help="Comma list of news feature columns to include")
    parser.add_argument("--news-window", type=int, default=1, help="Days of news features to include")
    # Evaluation-time banded rebalancing
    parser.add_argument("--no-trade-band", type=float, default=0.0, help="Band threshold for evaluation-time rebalancing (0=disabled)")
    parser.add_argument("--band-min-days", type=int, default=0, help="Min days between rebalances (evaluation)")
    parser.add_argument("--band-transaction-cost-bps", type=float, default=5.0, help="Transaction cost bps for banded")
    parser.add_argument("--band-slippage-bps", type=float, default=5.0, help="Slippage bps for banded")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    symbols = [s.strip() for s in args.symbols.split(',') if s.strip()]

    if args.out:
        out_dir = Path(args.out)
        if not out_dir.is_absolute():
            out_dir = repo_root / "reports" / "rl" / args.out
    else:
        stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
        out_dir = repo_root / "reports" / "rl" / f"walk_forward_{'-'.join(symbols)}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    sdt, edt = pd.to_datetime(args.start), pd.to_datetime(args.end)
    splits = daterange_segments(sdt, edt, int(args.segments))

    news_cols = [c.strip() for c in args.news_cols.split(',')] if args.news_cols else None
    news_csv = args.news_features_csv

    # Risk-free for Sharpe
    rf = load_rf_series(repo_root, sdt, edt)

    # Aggregate results
    summary_rows: List[Dict] = []

    for label, s, e in splits:
        try:
            dates, eq, w = run_portfolio_policy_with_weights(
                repo_root,
                symbols,
                s,
                e,
                news_features_csv=news_csv,
                news_cols=news_cols,
                news_window=int(args.news_window),
            )
            series = summarize_series(repo_root, label=label, kind="ppo_portfolio", dates=dates, equity_list=eq, rf=rf)
            summary_rows.append({
                "window": label,
                "kind": "ppo_portfolio",
                "total_return": series.total_return,
                "max_drawdown": series.max_dd,
                "sharpe": series.sharpe,
                "cagr": series.cagr,
                "points": len(series.equity),
            })
            # Save per-window equity CSV
            pd.DataFrame({"Date": series.dates, "Equity": series.equity}).to_csv(out_dir / f"equity_ppo_{label}.csv", index=False)

            # Optional banded replay
            if float(args.no_trade_band) > 0.0 or int(args.band_min_days) > 0:
                try:
                    eq_b = compute_banded_equity(
                        repo_root,
                        symbols,
                        dates,
                        w,
                        band=float(args.no_trade_band),
                        band_min_days=int(args.band_min_days),
                        tc_bps=float(args.band_transaction_cost_bps),
                        slip_bps=float(args.band_slippage_bps),
                        starting_equity=float(eq[0] if len(eq) else 10000.0),
                    )
                    series_b = summarize_series(repo_root, label=label, kind="ppo_portfolio_banded", dates=dates, equity_list=eq_b, rf=rf)
                    summary_rows.append({
                        "window": label,
                        "kind": "ppo_portfolio_banded",
                        "total_return": series_b.total_return,
                        "max_drawdown": series_b.max_dd,
                        "sharpe": series_b.sharpe,
                        "cagr": series_b.cagr,
                        "points": len(series_b.equity),
                    })
                    pd.DataFrame({"Date": series_b.dates, "Equity": series_b.equity}).to_csv(out_dir / f"equity_ppo_banded_{label}.csv", index=False)
                except Exception:
                    pass
        except Exception as e:
            # Record failure row
            summary_rows.append({
                "window": label,
                "kind": "ppo_portfolio",
                "total_return": 0.0,
                "max_drawdown": 0.0,
                "sharpe": 0.0,
                "cagr": 0.0,
                "points": 0,
                "error": str(e),
            })

    # Save summary
    pd.DataFrame(summary_rows).to_csv(out_dir / "summary.csv", index=False)
    print(f"Walk-forward report written to: {out_dir}")


if __name__ == "__main__":
    main()
