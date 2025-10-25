import os
import shutil
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
STOCK_DATA = BASE_DIR / "stock_data"
OUT_BASE = BASE_DIR / "data" / "rl" / "pricing"
OUT_BASE.mkdir(parents=True, exist_ok=True)

REQUIRED_COLS = {"Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"}


def normalize_price_csv(src_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(src_csv)
    # Basic normalization: ensure required columns, sort by Date, drop duplicates
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {src_csv}: {missing}")
    df = df.sort_values("Date").drop_duplicates(subset=["Date"]).reset_index(drop=True)
    return df


def process_symbol(symbol_dir: Path, out_dir: Path) -> None:
    price_csv = symbol_dir / "_price.csv"
    if not price_csv.exists():
        return
    df = normalize_price_csv(price_csv)
    out_csv = out_dir / f"{symbol_dir.name}.csv"
    df.to_csv(out_csv, index=False)


def main() -> int:
    if not STOCK_DATA.exists():
        print(f"No stock_data directory at {STOCK_DATA}")
        return 0
    count = 0
    for entry in STOCK_DATA.iterdir():
        if entry.is_dir():
            try:
                process_symbol(entry, OUT_BASE)
                count += 1
            except Exception as e:
                print(f"[WARN] {entry.name}: {e}")
    print(f"Pricing dataset built: {OUT_BASE} ({count} symbols processed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
