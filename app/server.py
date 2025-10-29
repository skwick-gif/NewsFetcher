import os
import sys
from pathlib import Path
from datetime import datetime
import requests
import threading
import json

# הוספת הנתיב של הפרויקט ל-sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from flask import Flask, render_template, jsonify, request

app = Flask(__name__)
app.config['SECRET_KEY'] = 'development-key'

# Register UI Blueprint (template-only routes)
try:
    from app.routes.ui import ui_bp
    app.register_blueprint(ui_bp)
except Exception:
    # Fallback if blueprint not available; keep server usable
    pass

from app.config.runtime import get_backend_base_url
from app.utils.proxy import proxy_to_backend
from app.services.jobs import job_logs, running_jobs, log_job as _log_job, run_script_in_background, run_command_in_background

# Blueprints moved routes
try:
    from app.routes.ibkr import ibkr_bp
    app.register_blueprint(ibkr_bp)
except Exception:
    pass
try:
    from app.routes.rl import rl_bp
    app.register_blueprint(rl_bp)
except Exception:
    pass
try:
    from app.routes.financial import financial_bp
    app.register_blueprint(financial_bp)
except Exception:
    pass
try:
    from app.routes.ml import ml_bp
    app.register_blueprint(ml_bp)
except Exception:
    pass
try:
    from app.routes.scanner import scanner_bp
    app.register_blueprint(scanner_bp)
except Exception:
    pass
try:
    from app.routes.strategy import strategy_bp
    app.register_blueprint(strategy_bp)
except Exception:
    pass
try:
    from app.routes.rl_tools import rl_tools_bp
    app.register_blueprint(rl_tools_bp)
except Exception:
    pass
try:
    from app.routes.system import system_bp
    app.register_blueprint(system_bp)
except Exception:
    pass
try:
    from app.routes.predictions import predictions_bp
    app.register_blueprint(predictions_bp)
except Exception:
    pass
try:
    from app.routes.triggers import triggers_bp
    app.register_blueprint(triggers_bp)
except Exception:
    pass
try:
    from app.routes.data import data_bp
    app.register_blueprint(data_bp)
except Exception:
    pass

last_success_cache = {}

# NOTE: RL paper/live logic has been migrated to FastAPI; Flask now proxies only.

# JSON safety: recursively sanitize NaN/Inf and non-serializable values
def _to_json_safe(obj):
    try:
        import numpy as _np
        import math as _math
    except Exception:
        _np = None
        import math as _math

    if obj is None:
        return None
    # Basic scalars
    if isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, (int,)):
        return int(obj)
    if isinstance(obj, float):
        if _math.isfinite(obj):
            return float(obj)
        return None
    # Numpy scalars
    if _np is not None and isinstance(obj, (_np.integer,)):
        try:
            return int(obj)
        except Exception:
            return None
    if _np is not None and isinstance(obj, (_np.floating,)):
        try:
            val = float(obj)
            return val if _math.isfinite(val) else None
        except Exception:
            return None
    # Datetime-like
    try:
        from datetime import date, datetime as _dt
        if isinstance(obj, (date, _dt)):
            try:
                return obj.isoformat()
            except Exception:
                return str(obj)
    except Exception:
        pass
    # Lists/tuples
    if isinstance(obj, (list, tuple)):
        return [ _to_json_safe(x) for x in obj ]
    # Dicts
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            try:
                ks = str(k)
            except Exception:
                ks = repr(k)
            out[ks] = _to_json_safe(v)
        return out
    # Fallback: try float, then str
    try:
        f = float(obj)
        if _math.isfinite(f):
            return f
        return None
    except Exception:
        try:
            return str(obj)
        except Exception:
            return None


def _load_strategy_defaults() -> dict:
    """Load strategy defaults from app/strategies/strategies.yaml. Returns dict; empty on failure."""
    try:
        import yaml  # type: ignore
        path = project_root / 'app' / 'strategies' / 'strategies.yaml'
        if not path.exists():
            return {}
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}
        if isinstance(data, dict):
            return data
        return {}
    except Exception:
        return {}

# Ensure API endpoints return JSON on errors instead of HTML error pages
@app.errorhandler(500)
def _handle_500(e):
    try:
        from traceback import format_exc
        detail = str(e) or 'internal server error'
        if request.path.startswith('/api/'):
            return jsonify({'status': 'error', 'detail': detail}), 500
    except Exception:
        pass
    # Fallback: default HTML for non-API paths
    return jsonify({'status': 'error', 'detail': 'internal server error'}), 500

@app.errorhandler(404)
def _handle_404(e):
    if request.path.startswith('/api/'):
        return jsonify({'status': 'error', 'detail': 'not found'}), 404
    return jsonify({'status': 'error', 'detail': 'not found'}), 404




    

# ===========================
# RL Auto-Tune Proxies (to FastAPI)
# ===========================

@app.route('/api/rl/auto-tune/start', methods=['POST'])
def api_rl_auto_tune_start_proxy():
    """Proxy to FastAPI: Start RL auto-tune job"""
    data = request.get_json(silent=True) or {}
    return proxy_to_backend('/api/rl/auto-tune/start', method='POST', json=data)

@app.route('/api/rl/auto-tune/status')
def api_rl_auto_tune_status_proxy():
    """Proxy to FastAPI: RL auto-tune status"""
    return proxy_to_backend('/api/rl/auto-tune/status')

@app.route('/api/rl/auto-tune/stop', methods=['POST'])
def api_rl_auto_tune_stop_proxy():
    """Proxy to FastAPI: Stop RL auto-tune job"""
    return proxy_to_backend('/api/rl/auto-tune/stop', method='POST')


# ===========================
# Data Ensure pipeline (prices + indicators + basic news counts)
# ===========================

def _log_job(job_id: str, level: str, msg: str):
    entry = {"timestamp": datetime.now().isoformat(), "level": level, "message": msg}
    job_logs.setdefault(job_id, []).append(entry)
    if len(job_logs[job_id]) > 400:
        job_logs[job_id] = job_logs[job_id][-400:]


def _ensure_symbol_data(symbol: str, start: str | None, end: str | None, root_dir: Path, job_id: str, indicator_params: dict | None = None):
    import pandas as pd
    import numpy as np
    import yfinance as yf
    sym = symbol.upper()
    try:
        _log_job(job_id, 'INFO', f"[{sym}] Downloading OHLCV from yfinance {start or ''} -> {end or ''}")
        df = yf.download(sym, start=start, end=end, progress=False)
        if df is None or df.empty:
            _log_job(job_id, 'WARN', f"[{sym}] No data returned from yfinance")
            return False
        df = df.reset_index()
        # Normalize columns: handle MultiIndex from yfinance and deduplicate
        if isinstance(df.columns, pd.MultiIndex):
            std_fields = ['Open','High','Low','Close','Adj Close','Volume']
            std_lower = [s.lower() for s in std_fields]
            def _pick_name(col):
                parts = [str(x).strip() for x in (list(col) if isinstance(col, tuple) else [col])]
                # Prefer a part matching a standard OHLCV field (case-insensitive)
                for p in parts:
                    pl = p.lower()
                    if pl in std_lower:
                        # return canonical casing
                        return std_fields[std_lower.index(pl)]
                # Fallback: if first part looks like a field appended with extra text
                if parts:
                    return parts[0]
                return str(col)
            df.columns = [_pick_name(c) for c in df.columns]
        # strip whitespace and de-duplicate columns
        df.columns = [str(c).strip() for c in df.columns]
        df = df.loc[:, ~pd.Index(df.columns).duplicated()]
        # Fallback: if Close missing but Adj Close exists, synthesize Close
        if 'Close' not in df.columns and 'Adj Close' in df.columns:
            df['Close'] = df['Adj Close']
        # Ensure a 'Date' column exists (yfinance may use 'Date', 'Datetime', or 'index')
        date_col = None
        for cand in ['Date','Datetime','date','datetime','index']:
            if cand in df.columns:
                date_col = cand
                break
        if date_col is None:
            # fallback: use the current index as dates
            df['Date'] = pd.to_datetime(df.index)
        elif date_col != 'Date':
            df = df.rename(columns={date_col: 'Date'})
        # Coerce Date and select required columns
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        # Build price dataframe from required columns; skip 'Adj Close'
        missing = [c for c in ['Open','High','Low','Close','Volume'] if c not in df.columns]
        if missing:
            _log_job(job_id, 'WARN', f"[{sym}] Missing columns: {missing} — proceeding with available data")
        price_cols = ['Date'] + [c for c in ['Open','High','Low','Close','Volume'] if c in df.columns]
        price_df = df[price_cols].copy()
        # Merge with existing CSV (if any) and ensure coverage from 2020-01-01
        sdir = root_dir / sym
        sdir.mkdir(parents=True, exist_ok=True)
        price_path = sdir / f"{sym}_price.csv"

        def _normalize_price_df(df_in):
            import pandas as _pd
            df2 = df_in.copy()
            # Normalize date column name and dtype
            date_col = None
            for cand in ['Date','Datetime','date','datetime','index']:
                if cand in df2.columns:
                    date_col = cand
                    break
            if date_col is None and df2.shape[1] >= 1:
                date_col = df2.columns[0]
            if date_col != 'Date':
                df2 = df2.rename(columns={date_col: 'Date'})
            df2['Date'] = _pd.to_datetime(df2['Date'], errors='coerce')
            # Keep only standard OHLCV columns (ignore Adj Close)
            keep_cols = ['Date'] + [c for c in ['Open','High','Low','Close','Volume'] if c in df2.columns]
            df2 = df2[keep_cols].dropna(subset=['Date']).drop_duplicates(subset=['Date'])
            return df2

        merged_df = _normalize_price_df(price_df)
        # Merge with existing prices if exists
        try:
            if price_path.exists():
                import pandas as _pd
                old_df = _pd.read_csv(price_path)
                old_df = _normalize_price_df(old_df)
                merged_df = _pd.concat([old_df, merged_df], ignore_index=True)
                merged_df = merged_df.drop_duplicates(subset=['Date']).sort_values('Date')
        except Exception as _e:
            _log_job(job_id, 'WARN', f"[{sym}] Failed to merge with existing CSV; overwriting. Error: {_e}")

        # Ensure coverage from 2020-01-01 by backfilling if needed
        try:
            import pandas as _pd
            min_needed = _pd.Timestamp('2020-01-01')
            if not merged_df.empty:
                cur_min = _pd.to_datetime(merged_df['Date']).min()
                if _pd.isna(cur_min) or cur_min > min_needed:
                    # Backfill from 2020-01-01 to current earliest date (exclusive)
                    back_end = None
                    try:
                        back_end = cur_min.date().isoformat()
                    except Exception:
                        back_end = cur_min.strftime('%Y-%m-%d') if cur_min is not None else None
                    _log_job(job_id, 'INFO', f"[{sym}] Backfilling history from 2020-01-01 to {back_end}")
                    df_back = yf.download(sym, start='2020-01-01', end=back_end, progress=False)
                    if df_back is not None and not df_back.empty:
                        df_back = df_back.reset_index()
                        if isinstance(df_back.columns, _pd.MultiIndex):
                            std_fields = ['Open','High','Low','Close','Adj Close','Volume']
                            std_lower = [s.lower() for s in std_fields]
                            def _pick_name(col):
                                parts = [str(x).strip() for x in (list(col) if isinstance(col, tuple) else [col])]
                                for p in parts:
                                    pl = p.lower()
                                    if pl in std_lower:
                                        return std_fields[std_lower.index(pl)]
                                return str(parts[0]) if parts else str(col)
                            df_back.columns = [_pick_name(c) for c in df_back.columns]
                        df_back.columns = [str(c).strip() for c in df_back.columns]
                        df_back = df_back.loc[:, ~_pd.Index(df_back.columns).duplicated()]
                        if 'Close' not in df_back.columns and 'Adj Close' in df_back.columns:
                            df_back['Close'] = df_back['Adj Close']
                        back_cols = ['Date'] + [c for c in ['Open','High','Low','Close','Volume'] if c in df_back.columns]
                        try:
                            # Normalize 'Date'
                            if 'Date' not in df_back.columns:
                                # Detect date-col
                                for cand in ['Date','Datetime','date','datetime','index']:
                                    if cand in df_back.columns:
                                        df_back = df_back.rename(columns={cand: 'Date'})
                                        break
                            df_back['Date'] = _pd.to_datetime(df_back['Date'], errors='coerce')
                        except Exception:
                            pass
                        back_df = df_back[back_cols].dropna(subset=['Date'])
                        merged_df = _pd.concat([back_df, merged_df], ignore_index=True)
                        merged_df = merged_df.drop_duplicates(subset=['Date']).sort_values('Date')
                    else:
                        _log_job(job_id, 'WARN', f"[{sym}] Backfill returned no data")
        except Exception as _e:
            _log_job(job_id, 'WARN', f"[{sym}] Backfill step failed: {_e}")

        # Write merged price CSV
        merged_df.to_csv(price_path, index=False)
        _log_job(job_id, 'INFO', f"[{sym}] Wrote price CSV: {price_path} (rows={len(merged_df)})")

        # Indicators (SMA20, EMA50, RSI14, MACD) + extras; optional override via indicator_params
        ip = indicator_params or {}
        def _as_int(x, d):
            try:
                return int(x)
            except Exception:
                return d
        def _as_float(x, d):
            try:
                return float(x)
            except Exception:
                return d
        # defaults
        rsi_p = _as_int(ip.get('rsi_period', 14), 14)
        macd_fast = _as_int(ip.get('macd_fast', 12), 12)
        macd_slow = _as_int(ip.get('macd_slow', 26), 26)
        macd_sig = _as_int(ip.get('macd_signal', 9), 9)
        atr_p = _as_int(ip.get('atr_period', 14), 14)
        bb_p = _as_int(ip.get('bb_period', 20), 20)
        bb_std_k = _as_float(ip.get('bb_std', 2.0), 2.0)
        sma_short = _as_int(ip.get('sma_short', 20), 20)
        sma_mid = _as_int(ip.get('sma_mid', 50), 50)
        sma_long = _as_int(ip.get('sma_long', 200), 200)
        ema_main = _as_int(ip.get('ema_main', 50), 50)
        roc_p = _as_int(ip.get('roc_period', 20), 20)
        vol_p = _as_int(ip.get('vol_period', 20), 20)
        vol_ma_p = _as_int(ip.get('volume_ma_period', 20), 20)
        adx_p = _as_int(ip.get('adx_period', 14), 14)
        kama_n = _as_int(ip.get('kama_n', 10), 10)
        kama_fast = _as_int(ip.get('kama_fast', 2), 2)
        kama_slow = _as_int(ip.get('kama_slow', 30), 30)
        stoch_k_p = _as_int(ip.get('stoch_k', 14), 14)
        stoch_d_p = _as_int(ip.get('stoch_d', 3), 3)

        if indicator_params:
            try:
                _log_job(job_id, 'INFO', f"[{sym}] Computing indicators with params: {json.dumps(ip)}")
            except Exception:
                _log_job(job_id, 'INFO', f"[{sym}] Computing indicators with custom params (logging failed)")
        else:
            _log_job(job_id, 'INFO', f"[{sym}] Computing indicators")
        import traceback as _tb
        try:
            # Ensure all columns are clean 1D Series of float
            def _col_to_series(df_, col_name):
                x = df_[col_name]
                if isinstance(x, pd.DataFrame):
                    x = x.iloc[:, 0]
                arr = np.asarray(x)
                if arr.ndim != 1:
                    arr = arr.reshape(-1)
                return pd.Series(arr, index=df_.index)

            # Use merged_df for indicators to ensure full history
            price_df_full = merged_df.copy()
            close = pd.to_numeric(_col_to_series(price_df_full, 'Close'), errors='coerce').astype(float)
            high = pd.to_numeric(_col_to_series(price_df_full, 'High'), errors='coerce').astype(float)
            low = pd.to_numeric(_col_to_series(price_df_full, 'Low'), errors='coerce').astype(float)
            volume = pd.to_numeric(_col_to_series(price_df_full, 'Volume'), errors='coerce').astype(float)
            def sma(series, n):
                return series.rolling(n, min_periods=n).mean()
            def ema(series, n):
                return series.ewm(span=n, adjust=False).mean()
            def rsi(series, n=rsi_p):
                delta = series.diff()
                gain = (delta.where(delta > 0, 0)).rolling(n).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(n).mean()
                rs = gain / (loss.replace(0, np.nan))
                out = 100 - (100 / (1 + rs))
                return out.fillna(50)
            # ATR
            prev_close = close.shift(1)
            tr = (high - low).abs()
            tr = pd.concat([
                tr,
                (high - prev_close).abs(),
                (low - prev_close).abs()
            ], axis=1).max(axis=1)
            atr_14 = tr.rolling(atr_p, min_periods=atr_p).mean()
            ema12 = ema(close, macd_fast)
            ema26 = ema(close, macd_slow)
            macd = ema12 - ema26
            signal = ema(macd, macd_sig)
            hist = macd - signal
            # Bollinger
            sma20 = sma(close, bb_p)
            std20 = close.rolling(bb_p, min_periods=bb_p).std()
            bb_upper = sma20 + bb_std_k * std20
            bb_lower = sma20 - bb_std_k * std20
            bb_width = (bb_upper - bb_lower) / sma20.replace(0, np.nan)
            bb_pctb = (close - bb_lower) / (bb_upper - bb_lower)
            # Trend and momentum extras
            sma50 = sma(close, sma_mid)
            sma200 = sma(close, sma_long)
            roc20 = close.pct_change(roc_p)
            vol20 = close.pct_change().rolling(vol_p, min_periods=vol_p).std()
            # Volume features
            volume_ma = volume.rolling(vol_ma_p, min_periods=vol_ma_p).mean()
            volume_sma_ratio = volume / volume_ma.replace(0, np.nan)

            # ADX(14)
            def _adx(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
                # ensure 1D series
                h = pd.to_numeric(high, errors='coerce').astype(float)
                l = pd.to_numeric(low, errors='coerce').astype(float)
                c = pd.to_numeric(close, errors='coerce').astype(float)
                up_move = h.diff()
                down_move = -l.diff()
                cond_plus = (up_move > down_move) & (up_move > 0)
                cond_minus = (down_move > up_move) & (down_move > 0)
                plus_dm = pd.Series(np.where(cond_plus.to_numpy(), up_move.to_numpy(), 0.0), index=h.index)
                minus_dm = pd.Series(np.where(cond_minus.to_numpy(), down_move.to_numpy(), 0.0), index=h.index)
                tr = pd.concat([
                    (h - l).abs(),
                    (h - c.shift(1)).abs(),
                    (l - c.shift(1)).abs()
                ], axis=1).max(axis=1)
                atr = tr.ewm(alpha=1.0/n, adjust=False).mean()
                plus_di = 100 * (plus_dm.ewm(alpha=1.0/n, adjust=False).mean() / atr.replace(0, np.nan))
                minus_di = 100 * (minus_dm.ewm(alpha=1.0/n, adjust=False).mean() / atr.replace(0, np.nan))
                dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
                adx_series = dx.ewm(alpha=1.0/n, adjust=False).mean()
                return adx_series

            adx14 = _adx(high, low, close, adx_p)

            # OBV
            close_diff = close.diff()
            # ensure np arrays for np.where to keep it strictly 1D
            cond_up = (close_diff > 0).to_numpy()
            cond_down = (close_diff < 0).to_numpy()
            vol_np = volume.to_numpy()
            obv_base = np.where(cond_down, -vol_np, 0.0)
            obv_arr = np.where(cond_up, vol_np, obv_base)
            obv_delta = pd.Series(obv_arr, index=close.index)
            obv = obv_delta.fillna(0).cumsum()

            # KAMA (params)
            def _kama(prices: pd.Series, n: int = 10, fast: int = 2, slow: int = 30) -> pd.Series:
                change = prices.diff(n).abs()
                volatility = prices.diff().abs().rolling(n, min_periods=n).sum()
                er = change / volatility.replace(0, np.nan)
                fast_sc = 2.0 / (fast + 1.0)
                slow_sc = 2.0 / (slow + 1.0)
                sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
                kama_series = pd.Series(index=prices.index, dtype='float64')
                if len(prices) == 0:
                    return kama_series
                kama_series.iloc[0] = float(prices.iloc[0])
                for i in range(1, len(prices)):
                    prev = kama_series.iloc[i-1]
                    sci = sc.iloc[i]
                    if not np.isfinite(sci):
                        sci = slow_sc ** 2
                    kama_series.iloc[i] = prev + sci * (prices.iloc[i] - prev)
                return kama_series

            kama = _kama(close, kama_n, kama_fast, kama_slow)

            # Stochastic Oscillator %K and %D
            lowest_low = low.rolling(stoch_k_p, min_periods=stoch_k_p).min()
            highest_high = high.rolling(stoch_k_p, min_periods=stoch_k_p).max()
            stoch_k14 = 100.0 * (close - lowest_low) / (highest_high - lowest_low)
            stoch_d14_3 = stoch_k14.rolling(stoch_d_p, min_periods=stoch_d_p).mean()

            ind = pd.DataFrame({
                'Date': price_df_full['Date'],
                'sma20': sma20,
                'sma50': sma50,
                'sma200': sma200,
                'ema50': ema(close, ema_main),
                'rsi14': rsi(close, rsi_p),
                'macd': macd,
                'macd_signal': signal,
                'macd_hist': hist,
                'atr_14': atr_14,
                'bb_upper': bb_upper,
                'bb_lower': bb_lower,
                'bb_width': bb_width,
                'bb_pctb': bb_pctb,
                'roc20': roc20,
                'vol20': vol20,
                'volume_ma': volume_ma,
                'volume_sma_ratio': volume_sma_ratio,
                'adx14': adx14,
                'obv': obv,
                'kama10': kama,
                'stoch_k14': stoch_k14,
                'stoch_d14_3': stoch_d14_3,
            })
            ind_path = sdir / f"{sym}_indicators.csv"
            ind.to_csv(ind_path, index=False)
            _log_job(job_id, 'INFO', f"[{sym}] Wrote indicators CSV: {ind_path}")
        except Exception as e:
            _log_job(job_id, 'ERROR', f"[{sym}] Indicator computation failed: {e}\n{_tb.format_exc()}")
            return False

        # Basic news counts using yfinance news (best-effort)
        try:
            t = yf.Ticker(sym)
            news = t.news or []
            rows = []
            for item in news:
                ts = item.get('providerPublishTime') or item.get('published')
                if ts is None:
                    continue
                try:
                    dt = datetime.fromtimestamp(int(ts))
                except Exception:
                    try:
                        dt = pd.to_datetime(ts).to_pydatetime()
                    except Exception:
                        continue
                rows.append({'Date': dt.date(), 'Symbol': sym, 'news_count': 1})
            if rows:
                nf_dir = project_root / 'ml' / 'data'
                nf_dir.mkdir(parents=True, exist_ok=True)
                nf_path = nf_dir / 'news_features.csv'
                nf_cols = ['Date','Symbol','news_count','llm_relevant_count','avg_score','fda_count','china_count','geopolitics_count','sentiment_avg']
                try:
                    nf = pd.read_csv(nf_path)
                except Exception:
                    nf = pd.DataFrame(columns=nf_cols)
                add = pd.DataFrame(rows)
                add['llm_relevant_count']=0
                add['avg_score']=0.0
                add['fda_count']=0
                add['china_count']=0
                add['geopolitics_count']=0
                add['sentiment_avg']=0.0
                add['Date'] = pd.to_datetime(add['Date'])
                nf['Date'] = pd.to_datetime(nf['Date'], errors='coerce')
                nf = pd.concat([nf, add[nf_cols]], ignore_index=True)
                nf = nf.dropna(subset=['Date','Symbol']).sort_values(['Symbol','Date'])
                nf.to_csv(nf_path, index=False)
                _log_job(job_id, 'INFO', f"[{sym}] Updated news features: {nf_path}")
            else:
                _log_job(job_id, 'INFO', f"[{sym}] No news items found via yfinance")
        except Exception as e:
            _log_job(job_id, 'WARN', f"[{sym}] News fetch failed: {e}")

        return True
    except Exception as e:
        _log_job(job_id, 'ERROR', f"[{sym}] Ensure failed: {e}")
        return False


@app.route('/api/rl/data/ensure', methods=['POST'])
def api_rl_data_ensure():
    """Ensure data for a list of symbols: prices+indicators+basic news counts. Runs in background."""
    data = request.get_json(silent=True) or {}
    symbols_raw = data.get('symbols') or ''
    if isinstance(symbols_raw, str):
        symbols = [s.strip().upper() for s in symbols_raw.split(',') if s.strip()]
    elif isinstance(symbols_raw, list):
        symbols = [str(s).strip().upper() for s in symbols_raw]
    else:
        symbols = []
    if not symbols:
        return jsonify({'status':'error','detail':'symbols required'}), 400
    start = data.get('start')
    end = data.get('end')
    indicator_params = data.get('indicator_params') or None

    job_id = f"ensure-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    running_jobs[job_id] = {"status":"running","start_time": datetime.now().isoformat(), "job_type": "ensure"}
    job_logs[job_id] = []

    def _runner():
        root_dir = project_root / 'stock_data'
        ok = True
        for sym in symbols:
            res = _ensure_symbol_data(sym, start, end, root_dir, job_id, indicator_params=indicator_params)
            ok = ok and bool(res)
        running_jobs[job_id]["status"] = "completed" if ok else "failed"
        _log_job(job_id, 'INFO', f"Done ensure symbols: {','.join(symbols)} status={running_jobs[job_id]['status']}")

    th = threading.Thread(target=_runner, daemon=True)
    th.start()
    return jsonify({'status':'started','job_id': job_id, 'symbols': symbols, 'start': start, 'end': end, 'indicator_params': indicator_params}), 202


@app.route('/api/rl/data/ensure/status/<job_id>')
def api_rl_data_ensure_status(job_id):
    info = running_jobs.get(job_id)
    logs = job_logs.get(job_id, [])
    if not info:
        return jsonify({'status':'error','detail':'unknown job_id'}), 404
    return jsonify({'status': info.get('status'), 'job_id': job_id, 'logs': [l.get('message') for l in logs]})

@app.route('/api/rl/data/ensure-all', methods=['POST'])
def api_rl_data_ensure_all():
    """Ensure data for ALL symbols under stock_data directory.
    Incremental: for each symbol, start from last available date + 1 day if CSV exists; otherwise full history.
    Runs in background and reuses the same status endpoint.
    """
    data = request.get_json(silent=True) or {}
    indicator_params = data.get('indicator_params') or None

    job_id = f"ensure-all-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    running_jobs[job_id] = {"status":"running","start_time": datetime.now().isoformat(), "job_type": "ensure-all"}
    job_logs[job_id] = []

    def _runner():
        root_dir = project_root / 'stock_data'
        syms = []
        try:
            for p in root_dir.iterdir():
                if p.is_dir():
                    try:
                        syms.append(p.name.upper())
                    except Exception:
                        continue
        except Exception as e:
            _log_job(job_id, 'ERROR', f"Failed to list stock_data: {e}")
        total = len(syms)
        if total == 0:
            running_jobs[job_id]["status"] = "completed"
            _log_job(job_id, 'INFO', "No symbols found under stock_data")
            return
        ok_all = True
        success = 0
        for i, sym in enumerate(sorted(syms)):
            # Compute incremental start date
            start = None
            try:
                import pandas as pd
                price_path = root_dir / sym / f"{sym}_price.csv"
                if price_path.exists():
                    df = pd.read_csv(price_path)
                    # Find date column robustly
                    date_col = None
                    for cand in ['Date','Datetime','date','datetime','index']:
                        if cand in df.columns:
                            date_col = cand
                            break
                    if date_col is None and df.shape[1] >= 1:
                        date_col = df.columns[0]
                    if date_col is not None and len(df) > 0:
                        try:
                            ds = pd.to_datetime(df[date_col], errors='coerce').dropna()
                            if len(ds) > 0:
                                last_dt = pd.to_datetime(ds.iloc[-1]).date()
                                from datetime import timedelta as _td
                                start = (last_dt + _td(days=1)).isoformat()
                        except Exception:
                            start = None
            except Exception:
                start = None

            _log_job(job_id, 'INFO', f"[{i+1}/{total}] Ensuring {sym} from {start or 'beginning'}")
            try:
                res = _ensure_symbol_data(sym, start, None, root_dir, job_id, indicator_params=indicator_params)
                if res:
                    success += 1
                else:
                    ok_all = False
            except Exception as e:
                ok_all = False
                _log_job(job_id, 'ERROR', f"[{sym}] Ensure failed: {e}")
        # Mark done with summary
        running_jobs[job_id]["status"] = "completed" if ok_all else "completed"
        _log_job(job_id, 'INFO', f"Done ensure-all: {success}/{total} succeeded")

    th = threading.Thread(target=_runner, daemon=True)
    th.start()
    return jsonify({'status':'started','job_id': job_id, 'mode': 'all', 'indicator_params': indicator_params}), 202

# ===========================
# Dashboard Routes (moved to UI blueprint in app.routes.ui)
# ===========================

# Routes '/', '/rl', '/scanner', '/strategy' now live in app.routes.ui
# NOTE: Strategy defaults route moved to app.routes.strategy


# NOTE: Strategy symbol helper moved to app.routes.strategy


# NOTE: Strategy symbols route moved to app.routes.strategy


# NOTE: Strategy price loader moved to app.routes.strategy


# NOTE: Strategy indicator imports moved to app.routes.strategy


# NOTE: Strategy backtest route moved to app.routes.strategy

@app.route('/health')
def health():
    """בדיקת תקינות השרת"""
    return jsonify({
        'status': 'healthy',
        'service': 'MarketPulse Dashboard',
        'backend': get_backend_base_url()
    })

# Moved: Financial routes are now under app.routes.financial

# NOTE: AI API proxies moved to app.routes.scanner

# Moved: ML routes are now under app.routes.ml

# NOTE: Data ensure proxy moved to app.routes.data

# Moved: additional financial/market routes under app.routes.financial

# NOTE: System API proxies moved to app.routes.system

# NOTE: Alerts/Stats/Articles proxies moved to app.routes.system and app.routes.predictions

# Moved: scanner/AI analysis routes to app.routes.scanner

# NOTE: Predictions proxies moved to app.routes.predictions

# Moved: scanner routes to app.routes.scanner

# NOTE: Jobs/Feeds/Statistics proxies moved to app.routes.system

# NOTE: Trigger proxies moved to app.routes.triggers

# ===========================
# Legacy Routes (removed)
# ===========================
# Note: Legacy routes using BACKEND_URL were removed to prevent NameError and
# duplicated endpoints. All clients should use the standardized proxies above
# which forward to FASTAPI_BACKEND.

"""Data Management Routes and job utilities now use app.services.jobs for state and runners."""

# ===========================
# News Exporter (T-1) — aggregate per-symbol daily features from FastAPI/DB
# ===========================



def _export_news_features_job(symbols: list[str], start: str, end: str, refresh: bool, job_id: str):
    _log_job(job_id, 'INFO', f"Exporting news features for {','.join(symbols)} {start}->{end} (T-1)")
    try:
        articles = _try_fetch_articles(start, end, symbols)
        _log_job(job_id, 'INFO', f"Fetched {len(articles)} articles from backend")
    except Exception as e:
        running_jobs[job_id]["status"] = "failed"
        _log_job(job_id, 'ERROR', f"Article fetch failed: {e}")
        return
    try:
        rows = _compute_news_features_rows(articles, symbols)
        _log_job(job_id, 'INFO', f"Aggregated into {len(rows)} (Date,Symbol) rows (after T-1 shift)")
        import pandas as pd
        nf_dir = project_root / 'ml' / 'data'
        nf_dir.mkdir(parents=True, exist_ok=True)
        nf_path = nf_dir / 'news_features.csv'
        cols = ["Date","Symbol","news_count","llm_relevant_count","avg_score","fda_count","china_count","geopolitics_count","sentiment_avg"]
        try:
            df_old = pd.read_csv(nf_path)
        except Exception:
            df_old = pd.DataFrame(columns=cols)
        df_old['Date'] = pd.to_datetime(df_old.get('Date', pd.Series([], dtype='datetime64[ns]')), errors='coerce')
        df_new = pd.DataFrame(rows)
        df_new['Date'] = pd.to_datetime(df_new['Date'])
        if refresh:
            # remove overlapping (Date,Symbol) in new set, then append
            key = df_new[['Date','Symbol']].drop_duplicates()
            merged = df_old.merge(key.assign(_rm=1), on=['Date','Symbol'], how='left')
            df_old = merged[merged['_rm'].isna()].drop(columns=['_rm'])
        df_all = pd.concat([df_old, df_new], ignore_index=True)
        df_all = df_all.dropna(subset=['Date','Symbol']).drop_duplicates(subset=['Date','Symbol'], keep='last').sort_values(['Symbol','Date'])
        df_all.to_csv(nf_path, index=False)
        _log_job(job_id, 'SUCCESS', f"Wrote {len(df_all)} total rows to {nf_path}")
        running_jobs[job_id]["status"] = "completed"
    except Exception as e:
        running_jobs[job_id]["status"] = "failed"
        _log_job(job_id, 'ERROR', f"Export failed: {e}")


"""RL news export endpoints moved to app.routes.rl_tools"""

@app.route('/api/data-management/status')
def data_management_status():
    """Get data management status and job information"""
    return jsonify({
        "status": "error",
        "message": "Data management status not implemented (no demo data)",
        "timestamp": datetime.now().isoformat()
    }), 503

@app.route('/api/data-management/run-job/<job_type>', methods=['POST'])
def run_data_job(job_type):
    """Trigger a data download job"""
    return jsonify({
        "status": "error",
        "message": "Data job triggering not implemented (no demo execution)",
        "job_type": job_type
    }), 501

@app.route('/api/data-management/job-status/<job_id>')
def get_job_status(job_id):
    """Get the status and logs of a running job"""
    return jsonify({
        "status": "error",
        "message": "Job status not implemented (no background job runner in Flask)",
        "job_id": job_id
    }), 501

@app.route('/api/system/health')
def system_health():
    """Get system health status"""
    return proxy_to_backend('/health')

@app.route('/api/data-management/logs')
def data_management_logs():
    """Get recent logs from data management operations"""
    return jsonify({
        "status": "error",
        "message": "Data management logs not implemented (no demo logs)",
        "timestamp": datetime.now().isoformat()
    }), 503

"""RL portfolio evaluate endpoints moved to app.routes.rl_tools"""

"""RL walk-forward endpoints moved to app.routes.rl_tools"""

if __name__ == '__main__':
    print("🚀 Starting MarketPulse Dashboard Server...")
    print("📊 Dashboard will be available at: http://localhost:5000")
    print("🔗 Health check: http://localhost:5000/health")
    print("📈 API endpoint: http://localhost:5000/api/market-data")
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )