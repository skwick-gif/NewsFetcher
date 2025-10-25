"""
Production pipeline for generating Progressive ML predictions as-of each bar (no placeholders).

Contract:
- Input: data/rl/pricing/<SYMBOL>.csv (built by build_pricing_dataset.py)
- Output: data/rl/progressive_signals/<SYMBOL>.csv with columns:
    Date, horizon, signal, confidence, expected_return, sl, tp, capped, model_version, data_version

Notes:
- Requires Progressive ML modules and trained checkpoints. If unavailable, the symbol is skipped without creating any file.
- SL/TP are computed via ATR(14) as-of each bar; capped indicates if expected_return was clipped by safety caps.
"""
from pathlib import Path
import csv
import sys
from typing import List, Optional
import hashlib
import pandas as pd

# Ensure repo root on sys.path for app.* imports
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from app.ml.progressive.data_loader import ProgressiveDataLoader
    from app.ml.progressive.predictor import ProgressivePredictor
    _PROGRESSIVE_AVAILABLE = True
except Exception as _e:
    print(f"[WARN] Progressive ML modules not available: {_e}")
    _PROGRESSIVE_AVAILABLE = False

BASE_DIR = Path(__file__).resolve().parents[2]
PRICING_DIR = BASE_DIR / "data" / "rl" / "pricing"
OUT_DIR = BASE_DIR / "data" / "rl" / "progressive_signals"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SCHEMA = [
    "Date",
    "horizon",
    "signal",
    "confidence",
    "expected_return",
    "sl",
    "tp",
    "capped",
    "model_version",
    "data_version",
]

def _write_rows(symbol: str, rows: List[List]) -> None:
    out_csv = OUT_DIR / f"{symbol}.csv"
    # Production: always write a fresh file with header
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(SCHEMA)
        writer.writerows(rows)


def _compute_atr_14(price_csv: Path) -> Optional[pd.Series]:
    try:
        dfp = pd.read_csv(price_csv)
        if 'Date' not in dfp.columns:
            return None
        for c in ('Open','High','Low','Close'):
            if c not in dfp.columns:
                return None
        dfp['Date'] = pd.to_datetime(dfp['Date'])
        dfp = dfp.sort_values('Date').reset_index(drop=True)
        high = pd.to_numeric(dfp['High'], errors='coerce')
        low = pd.to_numeric(dfp['Low'], errors='coerce')
        close = pd.to_numeric(dfp['Close'], errors='coerce')
        prev_close = close.shift(1)
        tr = pd.concat([
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(14, min_periods=14).mean()
        atr.index = pd.to_datetime(dfp['Date'])
        return atr
    except Exception:
        return None


def _sha256_file(path: Path) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def _compute_model_version(model_dir: Path, symbol: str, horizon: str, available_types: List[str]) -> str:
    """Compute a deterministic version hash for the set of checkpoints used for this horizon.
    Hashes the bytes of all existing files {type}_{symbol}_{horizon}_best.pth in sorted order.
    Returns hex digest (first 16 chars for brevity) or 'unknown' if none found.
    """
    files: List[Path] = []
    for mtype in sorted(available_types):
        f = model_dir / f"{mtype}_{symbol}_{horizon}_best.pth"
        if f.exists():
            files.append(f)
    if not files:
        return "unknown"
    h = hashlib.sha256()
    for fp in files:
        try:
            with fp.open('rb') as f:
                for chunk in iter(lambda: f.read(8192), b''):
                    h.update(chunk)
        except Exception:
            continue
    return h.hexdigest()[:16]


def _build_for_symbol(symbol: str, dl: ProgressiveDataLoader, predictor: ProgressivePredictor) -> int:
    # prepare full features dataframe
    df = dl.prepare_features(symbol, for_prediction=True)
    if df is None or df.empty:
        return 0
    feature_cols = dl.get_feature_columns(df)
    seq_len = dl.sequence_length
    if len(df) < seq_len:
        return 0
    # ensure price column
    price_col = 'Close' if 'Close' in df.columns else ('close' if 'close' in df.columns else None)
    if price_col is None:
        return 0
    rows = []
    # ATR series for SL/TP as-of
    price_src = REPO_ROOT / 'stock_data' / symbol / f'{symbol}_price.csv'
    atr_series = _compute_atr_14(price_src)
    # data version from normalized pricing file
    p_norm = PRICING_DIR / f"{symbol}.csv"
    data_version = _sha256_file(p_norm) or "unknown"
    # Load models once
    try:
        predictor.load_models(symbol, ['cnn', 'transformer', 'lstm'])
    except Exception as e:
        print(f"[WARN] {symbol}: load_models failed: {e}")
        return 0
    # iterate over time (PIT: use up to index i)
    for i in range(seq_len - 1, len(df)):
        # build sequence X (1, seq_len, F)
        try:
            X = df[feature_cols].iloc[i - seq_len + 1:i + 1].values.reshape(1, seq_len, len(feature_cols))
            current_price = float(df[price_col].iloc[i])
        except Exception:
            continue
        date_val = df.index[i] if isinstance(df.index, pd.DatetimeIndex) else df.iloc[i].get('Date', None)
        date_str = pd.to_datetime(date_val).strftime('%Y-%m-%d') if date_val is not None else ''
        # ensemble using available models per horizon
        horizons = ["1d", "7d", "30d"]
        for hz in horizons:
            # collect single model preds
            preds = []
            weights = []
            loaded = predictor.loaded_models.get(symbol, {})
            if not loaded:
                continue
            for mtype, models in loaded.items():
                if hz in models:
                    try:
                        pred = predictor.predict_single_model(models[hz], X, hz)
                        preds.append(pred)
                        w = predictor._compute_model_weight(symbol, mtype, hz)  # type: ignore
                        weights.append(float(w))
                    except Exception:
                        continue
            if not preds:
                continue
            tw = sum(weights) if sum(weights) > 0 else 1.0
            weights = [w / tw for w in weights]
            raw_expret = float(sum(p['price_change_pct'] * w for p, w in zip(preds, weights)))
            # hard caps to be safe
            cap = {"1d": 0.10, "7d": 0.20, "30d": 0.40}.get(hz, 0.50)
            capped_flag = 0
            expret = raw_expret
            if expret > cap:
                expret = cap
                capped_flag = 1
            if expret < -cap:
                expret = -cap
                capped_flag = 1
            conf = float(sum(p['confidence'] * w for p, w in zip(preds, weights)))
            # simple signal rule consistent with predictor
            signal = 'HOLD'
            if abs(expret) > 0.02:
                signal = 'BUY' if expret > 0 else 'SELL'
            # SL/TP from ATR as-of
            sl_val = ''
            tp_val = ''
            if atr_series is not None and date_str:
                series = atr_series.loc[:pd.to_datetime(date_str)]
                if not series.empty and pd.notna(series.iloc[-1]):
                    atr = float(series.iloc[-1])
                    sl_k, tp_k = 1.5, 2.0
                    if signal == 'BUY':
                        sl_val = round(current_price - sl_k * atr, 6)
                        tp_abs_pred = current_price * (1.0 + expret)
                        tp_abs_atr = current_price + tp_k * atr
                        tp_val = round(max(tp_abs_pred, tp_abs_atr), 6)
                    elif signal == 'SELL':
                        sl_val = round(current_price + sl_k * atr, 6)
                        tp_abs_pred = current_price * (1.0 + expret)
                        tp_abs_atr = current_price - tp_k * atr
                        tp_val = round(min(tp_abs_pred, tp_abs_atr), 6)
            # model version for this horizon from actual checkpoint files present
            available_types = list(loaded.keys())
            model_version = _compute_model_version(Path(predictor.model_dir), symbol, hz, available_types)
            rows.append([
                date_str,
                hz,
                signal,
                round(conf, 6),
                round(expret, 6),
                sl_val,
                tp_val,
                int(capped_flag),
                model_version,
                data_version,
            ])
    if rows:
        _write_rows(symbol, rows)
    return len(rows)


def main() -> int:
    if not PRICING_DIR.exists():
        print(f"Pricing dir not found: {PRICING_DIR}. Run build_pricing_dataset.py first.")
        return 1
    symbols = [p.stem for p in PRICING_DIR.glob("*.csv")]
    if not _PROGRESSIVE_AVAILABLE:
        print("[ERROR] Progressive ML is required to build as-of signals. Aborting without writing any files.")
        return 2
    # With Progressive available, build real as-of rows
    dl = ProgressiveDataLoader()
    predictor = ProgressivePredictor(dl)
    total_rows = 0
    for sym in symbols:
        try:
            written = _build_for_symbol(sym, dl, predictor)
            total_rows += written
            if written > 0:
                print(f"{sym}: wrote {written} rows")
            else:
                print(f"{sym}: skipped (no models or insufficient data)")
        except Exception as e:
            print(f"[WARN] {sym}: {e}")
    print(f"Progressive signals completed at: {OUT_DIR} (total rows: {total_rows})")
    return 0 if total_rows > 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())
