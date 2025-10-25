from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
PRICING_DIR = BASE_DIR / "data" / "rl" / "pricing"
CAL_DIR = BASE_DIR / "data" / "rl" / "calendars"
CAL_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = CAL_DIR / "market_calendar.csv"


def main() -> int:
    if not PRICING_DIR.exists():
        print(f"Pricing dir not found: {PRICING_DIR}")
        return 1
    all_dates = []
    active_counts = {}
    files = list(PRICING_DIR.glob('*.csv'))
    if not files:
        print("No pricing files found")
        return 2
    for p in files:
        try:
            df = pd.read_csv(p, usecols=["Date"])  # type: ignore
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.dropna(subset=['Date'])
            dates = pd.to_datetime(df['Date'].unique())
            for d in dates:
                active_counts[d] = active_counts.get(d, 0) + 1
        except Exception as e:
            print(f"[WARN] Skipping {p.name}: {e}")
            continue
    if not active_counts:
        print("No dates aggregated from pricing")
        return 3
    dates_sorted = sorted(active_counts.keys())
    out_df = pd.DataFrame({
        'Date': dates_sorted,
        'active_symbols': [active_counts[d] for d in dates_sorted]
    })
    out_df['Date'] = out_df['Date'].dt.strftime('%Y-%m-%d')
    out_df.to_csv(OUT_CSV, index=False)
    print(f"Market calendar written: {OUT_CSV} (rows={len(out_df)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
