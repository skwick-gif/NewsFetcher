"""
Local stock data adapter.
Reads from stock_data/<SYMBOL>/<SYMBOL>_price.csv and _indicators.csv,
parses to pandas DataFrame with DateTimeIndex, merges indicators if present.
"""

from pathlib import Path
from typing import Dict, Optional, List
import pandas as pd


class LocalStockData:
    def __init__(self, root: str | Path = "stock_data") -> None:
        self.root = Path(root)

    def list_symbols(self) -> List[str]:
        if not self.root.exists():
            return []
        return [p.name for p in self.root.iterdir() if p.is_dir()]

    def has_symbol(self, symbol: str) -> bool:
        pdir = self.root / symbol
        return (pdir / f"{symbol}_price.csv").exists()

    def load_symbol(self, symbol: str, tail_days: Optional[int] = None,
                    start_date: Optional[str] = None, end_date: Optional[str] = None) -> Dict:
        pdir = self.root / symbol
        price_csv = pdir / f"{symbol}_price.csv"
        ind_csv = pdir / f"{symbol}_indicators.csv"

        if not price_csv.exists():
            raise FileNotFoundError(f"Price CSV not found: {price_csv}")

        dfp = pd.read_csv(price_csv)
        # Accept either Date column or unnamed index
        if 'Date' in dfp.columns:
            # Robust parse for different date formats
            try:
                dfp['Date'] = pd.to_datetime(dfp['Date'], format='ISO8601')
            except Exception:
                try:
                    dfp['Date'] = pd.to_datetime(dfp['Date'], format='mixed')
                except Exception:
                    dfp['Date'] = pd.to_datetime(dfp['Date'], errors='coerce')
            dfp = dfp.set_index('Date')
        else:
            # Try to parse first column
            try:
                dfp.iloc[:, 0] = pd.to_datetime(dfp.iloc[:, 0], format='ISO8601')
            except Exception:
                try:
                    dfp.iloc[:, 0] = pd.to_datetime(dfp.iloc[:, 0], format='mixed')
                except Exception:
                    dfp.iloc[:, 0] = pd.to_datetime(dfp.iloc[:, 0], errors='coerce')
            dfp = dfp.set_index(dfp.columns[0])
        dfp = dfp.sort_index()

        # Ensure Close exists
        if 'Close' not in dfp.columns and 'close' in dfp.columns:
            dfp['Close'] = pd.to_numeric(dfp['close'], errors='coerce')
        else:
            dfp['Close'] = pd.to_numeric(dfp['Close'], errors='coerce')
        dfp = dfp.dropna(subset=['Close'])

        # Indicators
        dfi = None
        if ind_csv.exists():
            dfi = pd.read_csv(ind_csv)
            # parse index
            if 'Date' in dfi.columns:
                try:
                    dfi['Date'] = pd.to_datetime(dfi['Date'], format='ISO8601')
                except Exception:
                    try:
                        dfi['Date'] = pd.to_datetime(dfi['Date'], format='mixed')
                    except Exception:
                        dfi['Date'] = pd.to_datetime(dfi['Date'], errors='coerce')
                dfi = dfi.set_index('Date')
            else:
                try:
                    dfi.iloc[:, 0] = pd.to_datetime(dfi.iloc[:, 0], format='ISO8601')
                except Exception:
                    try:
                        dfi.iloc[:, 0] = pd.to_datetime(dfi.iloc[:, 0], format='mixed')
                    except Exception:
                        dfi.iloc[:, 0] = pd.to_datetime(dfi.iloc[:, 0], errors='coerce')
                dfi = dfi.set_index(dfi.columns[0])
            dfi = dfi.sort_index()
            # make numeric
            for c in dfi.columns:
                dfi[c] = pd.to_numeric(dfi[c], errors='coerce')

        # Optional date slicing (PIT-safe slicing on index)
        if start_date:
            try:
                start_ts = pd.to_datetime(start_date)
                dfp = dfp[dfp.index >= start_ts]
            except Exception:
                pass
        if end_date:
            try:
                end_ts = pd.to_datetime(end_date)
                dfp = dfp[dfp.index <= end_ts]
            except Exception:
                pass

        # Merge (left join on price index)
        dfm = dfp.copy()
        if dfi is not None:
            # Avoid overlapping OHLCV columns from indicators by dropping duplicates
            overlap = [c for c in dfi.columns if c in dfm.columns]
            if overlap:
                # common overlaps typically include Open, High, Low, Close, Volume
                dfi_use = dfi.drop(columns=overlap)
            else:
                dfi_use = dfi
            if len(dfi_use.columns) > 0:
                try:
                    dfm = dfm.join(dfi_use, how='left')
                except ValueError:
                    # Fallback: enforce unique columns with suffixes
                    dedup = dfi_use.copy()
                    new_cols = []
                    seen = set(dfm.columns)
                    for c in dedup.columns:
                        nc = c
                        k = 1
                        while nc in seen:
                            nc = f"{c}_ind{k}"
                            k += 1
                        new_cols.append(nc)
                        seen.add(nc)
                    dedup.columns = new_cols
                    dfm = dfm.join(dedup, how='left')

        if tail_days is not None and tail_days > 0:
            dfm = dfm.tail(tail_days)

        return {
            "symbol": symbol,
            "price": dfp,
            "indicators": dfi,
            "merged": dfm
        }

    def load_progressive_signals_pivot(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load progressive signals from data/rl/progressive_signals/<SYMBOL>.csv and pivot by horizon.

        Returns a DataFrame indexed by Date with columns like:
          expected_return_1d, confidence_1d, signal_1d, ... for 7d, 30d.
        If the file does not exist or is empty, returns None.
        """
        # repo root = this file ../../..
        repo_root = Path(__file__).resolve().parents[2]
        sig_csv = repo_root / "data" / "rl" / "progressive_signals" / f"{symbol}.csv"
        if not sig_csv.exists():
            return None
        try:
            df = pd.read_csv(sig_csv)
            if df.empty:
                return None
            # Expect Date, horizon, signal, confidence, expected_return, sl, tp, capped, model_version, data_version
            if 'Date' not in df.columns or 'horizon' not in df.columns:
                return None
            try:
                df['Date'] = pd.to_datetime(df['Date'], format='ISO8601')
            except Exception:
                try:
                    df['Date'] = pd.to_datetime(df['Date'], format='mixed')
                except Exception:
                    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.dropna(subset=['Date'])
            # pivot by horizon for expected_return, confidence, signal, sl, tp
            piv_er = df.pivot_table(index='Date', columns='horizon', values='expected_return', aggfunc='last')
            piv_cf = df.pivot_table(index='Date', columns='horizon', values='confidence', aggfunc='last')
            piv_sg = df.pivot_table(index='Date', columns='horizon', values='signal', aggfunc='last')
            piv_sl = df.pivot_table(index='Date', columns='horizon', values='sl', aggfunc='last')
            piv_tp = df.pivot_table(index='Date', columns='horizon', values='tp', aggfunc='last')
            # flatten columns
            er_cols = {c: f"expected_return_{c}" for c in piv_er.columns}
            cf_cols = {c: f"confidence_{c}" for c in piv_cf.columns}
            sg_cols = {c: f"signal_{c}" for c in piv_sg.columns}
            sl_cols = {c: f"sl_{c}" for c in piv_sl.columns}
            tp_cols = {c: f"tp_{c}" for c in piv_tp.columns}
            piv_er = piv_er.rename(columns=er_cols)
            piv_cf = piv_cf.rename(columns=cf_cols)
            piv_sg = piv_sg.rename(columns=sg_cols)
            piv_sl = piv_sl.rename(columns=sl_cols)
            piv_tp = piv_tp.rename(columns=tp_cols)
            out = piv_er.join(piv_cf, how='outer').join(piv_sg, how='outer').join(piv_sl, how='outer').join(piv_tp, how='outer')
            out = out.sort_index()
            return out
        except Exception:
            return None
