from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
PRICING_DIR = BASE_DIR / "data" / "rl" / "pricing"
SIG_DIR = BASE_DIR / "data" / "rl" / "progressive_signals"
CAL_DIR = BASE_DIR / "data" / "rl" / "calendars"


def validate_pricing(pricing_dir: Path) -> tuple[int, int, int]:
    total_files = 0
    issues = 0
    rows = 0
    for csv_path in pricing_dir.glob("*.csv"):
        total_files += 1
        try:
            df = pd.read_csv(csv_path)
            rows += len(df)
            # Basic checks
            if "Date" not in df.columns:
                print(f"[FAIL] {csv_path.name}: missing Date column")
                issues += 1
                continue
            if df["Date"].duplicated().any():
                print(f"[FAIL] {csv_path.name}: duplicate dates")
                issues += 1
            # sort check
            if not df["Date"].is_monotonic_increasing:
                print(f"[WARN] {csv_path.name}: dates not sorted ascending")
            required = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
            missing = required - set(df.columns)
            if missing:
                print(f"[FAIL] {csv_path.name}: missing columns {missing}")
                issues += 1
            if df[["Open","High","Low","Close","Adj Close","Volume"]].isnull().any().any():
                print(f"[WARN] {csv_path.name}: NaNs present in OHLCV")
        except Exception as e:
            print(f"[FAIL] {csv_path.name}: {e}")
            issues += 1
    return total_files, rows, issues


def main() -> int:
    if not PRICING_DIR.exists():
        print(f"No pricing dir found at {PRICING_DIR}")
        return 1
    files, rows, issues = validate_pricing(PRICING_DIR)
    status = "PASS" if issues == 0 else "FAIL"
    print(f"Pricing: files={files}, rows={rows}, issues={issues} => {status}")

    # Optional: validate progressive signals if exist
    sig_issues = 0
    if SIG_DIR.exists():
        import pandas as pd
        for sig_csv in SIG_DIR.glob('*.csv'):
            try:
                df = pd.read_csv(sig_csv)
                required = {"Date","horizon","signal","confidence","expected_return","sl","tp","capped","model_version","data_version"}
                missing = required - set(df.columns)
                if missing:
                    print(f"[FAIL] {sig_csv.name}: missing columns {missing}")
                    sig_issues += 1
                    continue
                # types and values
                if df.empty:
                    print(f"[FAIL] {sig_csv.name}: empty signals file")
                    sig_issues += 1
                    continue
                # horizon domain
                bad_hz = df[~df['horizon'].isin(['1d','7d','30d'])]
                if not bad_hz.empty:
                    print(f"[FAIL] {sig_csv.name}: invalid horizon values present")
                    sig_issues += 1
                # monotonic dates after sort
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                if df['Date'].isna().any():
                    print(f"[FAIL] {sig_csv.name}: invalid Date values")
                    sig_issues += 1
                df = df.sort_values(['Date','horizon'])
                # sanity: expected_return within caps
                if (df['expected_return'].abs() > 0.5).any():
                    print(f"[WARN] {sig_csv.name}: expected_return has values > 50%")
                # Cross-check against pricing dates for the same symbol
                symbol = sig_csv.stem
                p_csv = PRICING_DIR / f"{symbol}.csv"
                if not p_csv.exists():
                    print(f"[FAIL] {sig_csv.name}: pricing file missing for symbol {symbol}")
                    sig_issues += 1
                else:
                    dp = pd.read_csv(p_csv, usecols=["Date"])  # type: ignore
                    dp['Date'] = pd.to_datetime(dp['Date'], errors='coerce')
                    if dp['Date'].isna().any():
                        print(f"[FAIL] pricing {p_csv.name}: invalid Date values")
                        sig_issues += 1
                    pricing_dates = set(dp['Date'].unique())
                    sig_dates = set(df['Date'].unique())
                    missing_in_pricing = [d for d in sig_dates if d not in pricing_dates]
                    if missing_in_pricing:
                        print(f"[FAIL] {sig_csv.name}: {len(missing_in_pricing)} signal dates not present in pricing")
                        sig_issues += 1
                # Optional: verify against market calendar if present
                cal_csv = CAL_DIR / 'market_calendar.csv'
                if cal_csv.exists():
                    try:
                        cal = pd.read_csv(cal_csv)
                        if 'Date' in cal.columns:
                            cal['Date'] = pd.to_datetime(cal['Date'], errors='coerce')
                            cal_dates = set(cal['Date'].dropna().unique())
                            sig_dates = set(df['Date'].unique())
                            off_calendar = [d for d in sig_dates if d not in cal_dates]
                            if off_calendar:
                                print(f"[FAIL] {sig_csv.name}: {len(off_calendar)} signal dates not present in market_calendar")
                                sig_issues += 1
                    except Exception as e:
                        print(f"[WARN] calendar check skipped: {e}")
            except Exception as e:
                print(f"[FAIL] {sig_csv.name}: {e}")
                sig_issues += 1

    overall_fail = (issues > 0) or (sig_issues > 0)
    if SIG_DIR.exists():
        print(f"Signals: issues={sig_issues} => {'PASS' if sig_issues==0 else 'FAIL'}")
    return 0 if not overall_fail else 2


if __name__ == "__main__":
    raise SystemExit(main())
