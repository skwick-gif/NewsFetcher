import os
import csv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]  # repo root
STOCK_DATA = BASE_DIR / "stock_data"
OUTPUT_DIR = BASE_DIR / "data" / "rl"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_CSV = OUTPUT_DIR / "universe.csv"
PRICING_DIR = OUTPUT_DIR / "pricing"


def discover_symbols(stock_data_dir: Path) -> list[str]:
    if not stock_data_dir.exists():
        return []
    # Each subfolder under stock_data is a symbol
    symbols = []
    for entry in stock_data_dir.iterdir():
        if entry.is_dir():
            symbols.append(entry.name)
    symbols.sort()
    return symbols


def write_universe_csv(symbols: list[str], output_csv: Path) -> None:
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["symbol", "start_date", "end_date", "source"])  # only symbols with pricing are included
        for sym in symbols:
            start_date = ""
            end_date = ""
            p_csv = PRICING_DIR / f"{sym}.csv"
            if not p_csv.exists():
                # skip symbols without normalized pricing
                continue
            try:
                import pandas as pd
                df = pd.read_csv(p_csv, usecols=["Date"])  # type: ignore
                if not df.empty:
                    start_date = str(pd.to_datetime(df["Date"].min()).date())
                    end_date = str(pd.to_datetime(df["Date"].max()).date())
            except Exception:
                # if we can't read, skip symbol
                continue
            writer.writerow([sym, start_date, end_date, "local-stock_data"])


def main() -> int:
    symbols = discover_symbols(STOCK_DATA)
    write_universe_csv(symbols, OUTPUT_CSV)
    print(f"Universe written: {OUTPUT_CSV} ({len(symbols)} symbols)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
