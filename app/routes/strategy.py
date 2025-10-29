import sys
from pathlib import Path
from datetime import datetime

from flask import Blueprint, jsonify, request

# Ensure project root on path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from app.strategies.indicators import (  # noqa: E402
    macd_series as _macd_series,
    rsi_series as _rsi_series,
    stoch_series as _stoch_series,
    adx_series as _adx_series,
    kama_series as _kama_series,
)


strategy_bp = Blueprint('strategy', __name__)


def _to_json_safe(obj):
    try:
        import numpy as _np
        import math as _math
    except Exception:
        _np = None
        import math as _math

    if obj is None:
        return None
    if isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, (int,)):
        return int(obj)
    if isinstance(obj, float):
        if _math.isfinite(obj):
            return float(obj)
        return None
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
    try:
        from datetime import date, datetime as _dt
        if isinstance(obj, (date, _dt)):
            try:
                return obj.isoformat()
            except Exception:
                return str(obj)
    except Exception:
        pass
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(x) for x in obj]
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            try:
                ks = str(k)
            except Exception:
                ks = repr(k)
            out[ks] = _to_json_safe(v)
        return out
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


@strategy_bp.route('/api/strategy/defaults')
def api_strategy_defaults():
    """Return strategy defaults from strategies.yaml. Optional ?id=<strategy_id> filters to single one."""
    data = _load_strategy_defaults()
    all_defs = data.get('strategies') if isinstance(data, dict) else {}
    sid = request.args.get('id')
    if sid:
        s = all_defs.get(sid)
        if not s:
            return jsonify({'status': 'error', 'detail': f'unknown strategy id: {sid}'}), 404
        return jsonify({'status': 'ok', 'id': sid, 'label': s.get('label'), 'params': s.get('params', {})})
    out = {}
    if isinstance(all_defs, dict):
        for k, v in all_defs.items():
            if isinstance(v, dict):
                out[k] = {'label': v.get('label'), 'params': v.get('params', {})}
    return jsonify({'status': 'ok', 'strategies': out})


def _list_local_symbols_with_prices(max_symbols: int | None = None) -> list[str]:
    """Return symbols that have a price CSV under stock_data/<SYM>/<SYM>_price.csv"""
    syms: list[str] = []
    try:
        root = project_root / 'stock_data'
        if not root.exists():
            return []
        for p in root.iterdir():
            if not p.is_dir():
                continue
            sym = p.name.upper()
            price_path = p / f"{sym}_price.csv"
            if price_path.exists():
                syms.append(sym)
    except Exception:
        return []
    syms = sorted(set(syms))
    if max_symbols is not None and max_symbols > 0:
        return syms[:max_symbols]
    return syms


@strategy_bp.route('/api/strategy/symbols')
def api_strategy_symbols():
    try:
        n = request.args.get('limit')
        limit = int(n) if n is not None else None
    except Exception:
        limit = None
    syms = _list_local_symbols_with_prices(limit)
    return jsonify({'status': 'ok', 'symbols': syms, 'count': len(syms)})


def _load_price_df(sym: str):
    import pandas as pd
    sym = sym.upper().strip()
    root = project_root / 'stock_data' / sym
    price_path = root / f"{sym}_price.csv"
    if not price_path.exists():
        raise FileNotFoundError(f"price csv not found: {price_path}")
    df = pd.read_csv(price_path)
    date_col = None
    for cand in ['Date','Datetime','date','datetime','index']:
        if cand in df.columns:
            date_col = cand
            break
    if date_col is None and df.shape[1] >= 1:
        date_col = df.columns[0]
    if date_col != 'Date':
        df = df.rename(columns={date_col: 'Date'})
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    keep = ['Date'] + [c for c in ['Open','High','Low','Close','Volume'] if c in df.columns]
    out = df[keep].dropna(subset=['Date']).drop_duplicates(subset=['Date']).sort_values('Date').reset_index(drop=True)
    if 'Close' not in out.columns and 'Adj Close' in df.columns:
        out['Close'] = df['Adj Close']
    return out


@strategy_bp.route('/api/strategy/backtest', methods=['POST'])
def api_strategy_backtest():
    try:
        data = request.get_json(silent=True) or {}
        sym = (data.get('symbol') or '').strip().upper()
        if not sym:
            return jsonify({'status': 'error', 'detail': 'symbol is required'}), 400
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        strat = (data.get('strategy') or 'macd_cross').strip()
        macd_fast = int(data.get('macd_fast', 12) or 12)
        macd_slow = int(data.get('macd_slow', 26) or 26)
        macd_sig = int(data.get('macd_signal', 9) or 9)
        initial_cash = float(data.get('initial_cash', 10000.0) or 10000.0)
        tcost = int(data.get('transaction_cost_bps', 5) or 5)
        slip = int(data.get('slippage_bps', 5) or 5)
        adv_period = int(data.get('adv_period', 20) or 20)
        pre_bars = int(data.get('pre_bars', 3) or 3)
        sell_bars = int(data.get('sell_bars', 2) or 2)
        adx_min = float(data.get('adx_min', 25) or 25)
        stop_loss_pct = float(data.get('stop_loss_pct', 0) or 0)
        trailing_stop = bool(data.get('trailing_stop', True))
        conv_window = int(data.get('conv_window', 60) or 60)
        p_buy = float(data.get('p_buy', 35) or 35)
        e_buy = float(data.get('e_buy', 20) or 20)
        p_sell = float(data.get('p_sell', 40) or 40)
        vol_sma_period = int(data.get('vol_sma_period', 20) or 20)
        vol_down_strict = bool(data.get('vol_down_strict', False))
        macd_zero_stop = bool(data.get('macd_zero_stop', False))
        rsi_period = int(data.get('rsi_period', 14) or 14)
        stoch_k_p = int(data.get('stoch_k', 14) or 14)
        stoch_d_p = int(data.get('stoch_d', 3) or 3)
        # New enhanced strategy parameters
        atr_multiplier = float(data.get('atr_multiplier', 1.5) or 1.5)
        take_profit_pct = float(data.get('take_profit_pct', 12.0) or 12.0)

        import pandas as pd
        import numpy as np

        df = _load_price_df(sym)
        if start_date:
            df = df[df['Date'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['Date'] <= pd.to_datetime(end_date)]
        df = df.dropna(subset=['Close']).reset_index(drop=True)
        if len(df) < max(macd_fast, macd_slow, macd_sig) + 5:
            return jsonify({'status': 'error', 'detail': f'Not enough rows for MACD on {sym}'}), 400

        macd_df = _macd_series(df['Close'].astype(float), macd_fast, macd_slow, macd_sig)
        rsi_ser = _rsi_series(df['Close'].astype(float), rsi_period)
        ema_fast_p = int(data.get('ema_fast', 20) or 20)
        ema_slow_p = int(data.get('ema_slow', 50) or 50)
        kama_n_p = int(data.get('kama_n', 10) or 10)
        ema_fast_ser = df['Close'].astype(float).ewm(span=ema_fast_p, adjust=False).mean().rename('ema_fast')
        ema_slow_ser = df['Close'].astype(float).ewm(span=ema_slow_p, adjust=False).mean().rename('ema_slow')
        kama_ser = _kama_series(df['Close'].astype(float), kama_n_p, 2, 30)
        if 'High' in df.columns and 'Low' in df.columns:
            stoch_k_ser, stoch_d_ser = _stoch_series(
                df['High'].astype(float), df['Low'].astype(float), df['Close'].astype(float), stoch_k_p, stoch_d_p
            )
        else:
            import pandas as _pd
            stoch_k_ser = _pd.Series([float('nan')] * len(df))
            stoch_d_ser = _pd.Series([float('nan')] * len(df))
        if 'High' in df.columns and 'Low' in df.columns:
            adx = _adx_series(df['High'].astype(float), df['Low'].astype(float), df['Close'].astype(float), 14)
        else:
            import pandas as _pd
            adx = _pd.Series([float('nan')]*len(df), name='adx')
        df = pd.concat([
            df, macd_df, adx,
            rsi_ser.rename('rsi'),
            stoch_k_ser.rename('stoch_k'), stoch_d_ser.rename('stoch_d'),
            ema_fast_ser, ema_slow_ser, kama_ser
        ], axis=1)
        if 'Volume' not in df.columns:
            df['Volume'] = np.nan
        df['ADV'] = df['Volume'].rolling(adv_period, min_periods=adv_period).mean()
        try:
            df['VOL_SMA'] = df['Volume'].rolling(vol_sma_period, min_periods=vol_sma_period).mean()
        except Exception:
            df['VOL_SMA'] = np.nan

        dates = df['Date'].dt.date.tolist()
        hist = (df['macd'] - df['macd_signal']).astype(float)
        g = hist.abs()
        minp = max(3, int(conv_window//3)) if conv_window and conv_window > 0 else 3
        rollmax = g.rolling(int(conv_window or 60), min_periods=minp).max().shift(1)
        with np.errstate(divide='ignore', invalid='ignore'):
            conv_ratio = g / rollmax
        df['conv_ratio'] = conv_ratio.replace([np.inf, -np.inf], np.nan)

        from app.strategies import get_strategy as _get_strategy
        strat_fn = _get_strategy(strat) or _get_strategy('macd_cross')
        strat_params = {
            'initial_cash': float(initial_cash),
            'transaction_cost_bps': int(tcost),
            'slippage_bps': int(slip),
            'pre_bars': int(pre_bars),
            'sell_bars': int(sell_bars),
            'adx_min': float(adx_min),
            'stop_loss_pct': float(stop_loss_pct),
            'trailing_stop': bool(trailing_stop),
            'p_buy': float(p_buy),
            'e_buy': float(e_buy),
            'p_sell': float(p_sell),
            'vol_down_strict': bool(vol_down_strict),
            'macd_zero_stop': bool(macd_zero_stop),
            'atr_multiplier': float(atr_multiplier),
            'take_profit_pct': float(take_profit_pct),
            'vol_sma_period': int(vol_sma_period),
        }
        result = strat_fn(df, strat_params) if strat_fn else None
        if result is None:
            equity_curve = []
            trades = []
            in_position = []
            stops_count = 0
        else:
            equity_curve = result.equity_curve
            trades = result.trades
            in_position = result.in_position
            stops_count = result.stops_count

        import numpy as np
        eq = np.array(equity_curve, dtype=float)
        ret_total = (eq[-1] / float(initial_cash)) - 1.0 if len(eq) else 0.0
        if len(eq):
            peaks = np.maximum.accumulate(eq)
            dd = (eq - peaks) / peaks
            max_dd = float(dd.min()) if np.isfinite(dd).any() else 0.0
        else:
            max_dd = 0.0
        win = 0
        tot = 0
        last_buy_equity = None
        for t in trades:
            if t['action'] == 'BUY':
                last_buy_equity = t['equity']
            elif t['action'] == 'SELL' and last_buy_equity is not None:
                tot += 1
                if t['equity'] > last_buy_equity:
                    win += 1
                last_buy_equity = None
        win_rate = (win / tot) if tot > 0 else None

        def _series_to_list_none(s):
            import pandas as _pd
            return [(None if _pd.isna(x) else float(x)) for x in s.tolist()]

        if 'Open' in df.columns:
            open_arr = _series_to_list_none(df['Open'])
        else:
            open_arr = [None] * len(df)
        if 'High' in df.columns:
            high_arr = _series_to_list_none(df['High'])
        else:
            high_arr = [None] * len(df)
        if 'Low' in df.columns:
            low_arr = _series_to_list_none(df['Low'])
        else:
            low_arr = [None] * len(df)
        close_arr = _series_to_list_none(df['Close'])
        vol_arr = df['Volume'].astype(float).fillna(0.0).tolist()
        adv_arr = _series_to_list_none(df['ADV'])

        resp = {
            'status': 'ok',
            'symbol': sym,
            'dates': [str(d) for d in dates],
            'open': open_arr,
            'high': high_arr,
            'low': low_arr,
            'close': close_arr,
            'volume': vol_arr,
            'adv': adv_arr,
            'equity': [float(x) for x in equity_curve],
            'macd': df['macd'].astype(float).fillna(0.0).tolist(),
            'macd_signal': df['macd_signal'].astype(float).fillna(0.0).tolist(),
            'rsi': df['rsi'].astype(float).where(df['rsi'].notna(), None).tolist(),
            'stoch_k': df['stoch_k'].astype(float).where(df['stoch_k'].notna(), None).tolist(),
            'stoch_d': df['stoch_d'].astype(float).where(df['stoch_d'].notna(), None).tolist(),
            'ema_fast': df['ema_fast'].astype(float).where(df['ema_fast'].notna(), None).tolist(),
            'ema_slow': df['ema_slow'].astype(float).where(df['ema_slow'].notna(), None).tolist(),
            'kama': df['kama'].astype(float).where(df['kama'].notna(), None).tolist(),
            'trades': trades,
            'in_position': in_position,
            'metrics': {
                'initial_cash': float(initial_cash),
                'final_equity': float(eq[-1]) if len(eq) else float(initial_cash),
                'total_return_pct': float(ret_total * 100.0),
                'max_drawdown_pct': float(max_dd * 100.0),
                'num_trades': int(len(trades)),
                'round_trips': int(tot),
                'win_rate': float(win_rate) if win_rate is not None else None,
                'tcost_bps': int(tcost),
                'slippage_bps': int(slip),
                'adv_period': int(adv_period),
                'strategy': strat,
                'pre_bars': int(pre_bars),
                'sell_bars': int(sell_bars),
                'adx_min': float(adx_min),
                'stop_loss_pct': float(stop_loss_pct),
                'trailing_stop': bool(trailing_stop),
                'num_stops': int(stops_count),
                'conv_window': int(conv_window),
                'p_buy': float(p_buy),
                'e_buy': float(e_buy),
                'p_sell': float(p_sell),
                'vol_sma_period': int(vol_sma_period),
                'vol_down_strict': bool(vol_down_strict),
                'macd_zero_stop': bool(macd_zero_stop),
                'rsi_period': int(rsi_period),
                'stoch_k': int(stoch_k_p),
                'stoch_d': int(stoch_d_p),
                'ema_fast': int(ema_fast_p),
                'ema_slow': int(ema_slow_p),
                'kama_n': int(kama_n_p),
            }
        }
        return jsonify(_to_json_safe(resp))
    except FileNotFoundError as e:
        return jsonify({'status': 'error', 'detail': str(e)}), 404
    except Exception as e:
        return jsonify({'status': 'error', 'detail': str(e)}), 500
