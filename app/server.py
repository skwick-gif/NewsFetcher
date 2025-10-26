import os
import sys
from pathlib import Path
import requests
from datetime import datetime
import subprocess
import threading
import json

# הוספת הנתיב של הפרויקט ל-sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from flask import Flask, render_template, jsonify, request
from flask import send_file

app = Flask(__name__)
app.config['SECRET_KEY'] = 'development-key'

# FastAPI Backend URL
FASTAPI_BACKEND = os.getenv('FASTAPI_BACKEND', 'http://localhost:8000')

last_success_cache = {}
live_state = {}

# Paper trading state
paper_state = {
    'running': False,
    'thread': None,
    'config': None,
    'last_run_date': None,
    'positions': {},  # symbol -> shares
    'cash': 0.0,
    'equity': 0.0,
    'peak_equity': 0.0,
    'logs': [],
    'last_decision': None,  # dict with date, raw_weights, banded_weights, trades
}
from threading import Lock
paper_lock = Lock()

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


def proxy_to_backend(endpoint, method='GET', timeout=None, **kwargs):
    """
    Proxy request to FastAPI backend
    """
    try:
        url = f"{FASTAPI_BACKEND}{endpoint}"

        # Default timeouts: GET 6s, POST 12s unless overridden
        if timeout is None:
            timeout = 12 if method == 'POST' else 6

        if method == 'GET':
            response = requests.get(url, timeout=timeout, **kwargs)
        elif method == 'POST':
            response = requests.post(url, timeout=timeout, **kwargs)
        else:
            return jsonify({'error': f'Unsupported method: {method}'}), 400
        
        # Return the JSON response from backend
        try:
            payload = response.json()
        except Exception:
            payload = {'status': 'error', 'message': f'Invalid JSON from backend ({response.status_code})'}

        # Always forward backend response as-is (no demo/fallback)
        return jsonify(payload), response.status_code
    
    except requests.exceptions.ConnectionError:
        # Return explicit error (no fallback)
        return jsonify({
            'status': 'error',
            'message': 'Backend service unavailable',
            'endpoint': endpoint,
            'timestamp': datetime.now().isoformat()
        }), 503
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

# ===========================
# RL Quick Simulation + Status (Flask-native)
# ===========================

# Import lightweight RL helpers locally to avoid import cost on cold start
try:
    from rl.simulation import run_simulation as rl_run_simulation
    from rl.envs.market_env import MarketEnv as RLMarketEnv
    import pandas as pd  # used for date handling in planning
except Exception:
    rl_run_simulation = None
    RLMarketEnv = None
    pd = None


@app.route('/api/rl/status')
def api_rl_status():
    """Simple RL service status for the dashboard."""
    return jsonify({
        'status': 'ok',
        'service': 'flask-rl-ui',
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/rl/simulate')
def api_rl_simulate():
    """Run a quick, non-learning simulation for a single symbol (UI helper)."""
    if rl_run_simulation is None:
        return jsonify({'status': 'error', 'message': 'RL simulation module not available'}), 500

    symbol = (request.args.get('symbol') or 'AAPL').upper()
    policy = request.args.get('policy', 'follow_trend')
    try:
        window = int(request.args.get('window', '60'))
    except Exception:
        window = 60
    # Either days or explicit start/end
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    days = request.args.get('days')
    days_int = None
    try:
        if days is not None:
            days_int = int(days)
    except Exception:
        days_int = None

    try:
        data = rl_run_simulation(
            symbol=symbol,
            days=days_int,
            window=window,
            cost_bps=5,
            slip_bps=5,
            policy=policy,
            start_date=start_date,
            end_date=end_date
        )
        return jsonify({'status': 'success', 'data': data})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/rl/simulate/plan')
def api_rl_simulate_plan():
    """Auto-plan reasonable start/end, window, and days for Quick Simulation."""
    if RLMarketEnv is None or pd is None:
        return jsonify({'status': 'error', 'detail': 'Planning unavailable: RL modules missing'}), 500
    symbol = (request.args.get('symbol') or 'AAPL').upper()
    # Optional window hint
    try:
        win_hint = int(request.args.get('window', '0'))
    except Exception:
        win_hint = 0
    try:
        env = RLMarketEnv.load_from_local(symbol, window=max(2, win_hint or 60), tail_days=None)
        df = env.df
        n = int(len(df))
        if n < 2:
            return jsonify({'status': 'error', 'detail': f'Not enough data for {symbol}'}), 400
        # Ensure index is datetime for date slicing
        idx = df.index
        if not isinstance(idx, pd.DatetimeIndex):
            try:
                idx = pd.to_datetime(idx)
            except Exception:
                pass
        earliest = idx[0]
        latest = idx[-1]
        # Plan last ~250 rows if available
        span = min(250, n)
        start_idx = max(0, n - span)
        start_dt = idx[start_idx]
        end_dt = latest
        # Window: respect hint else derive from data length
        window = win_hint if win_hint > 0 else int(max(10, min(60, n // 4)))
        # Days equals span
        plan = {
            'start_date': str(getattr(start_dt, 'date', lambda: start_dt)()),
            'end_date': str(getattr(end_dt, 'date', lambda: end_dt)()),
            'window': window,
            'days': int(span),
            'total_rows': n
        }
        return jsonify({'status': 'planned', 'plan': plan})
    except Exception as e:
        return jsonify({'status': 'error', 'detail': str(e)}), 500


@app.route('/api/rl/ppo/plan')
def api_rl_ppo_plan():
    """Plan PPO training window and timesteps from local data (single-symbol helper for UI)."""
    if RLMarketEnv is None or pd is None:
        return jsonify({'status': 'error', 'detail': 'Planning unavailable: RL modules missing'}), 500
    # Accept either multiple symbols or a single symbol
    symbols_param = request.args.get('symbols')
    symbol_single = (request.args.get('symbol') or '').upper()
    symbols: list[str] = []
    if symbols_param:
        symbols = [s.strip().upper() for s in symbols_param.split(',') if s.strip()]
    elif symbol_single:
        symbols = [symbol_single]
    else:
        return jsonify({'status': 'error', 'detail': 'symbol or symbols is required'}), 400
    try:
        win_hint = int(request.args.get('window', '0'))
    except Exception:
        win_hint = 0
    try:
        # If multiple symbols, plan based on PortfolioEnv with intersected index
        if len(symbols) >= 2:
            from rl.envs.portfolio_env import PortfolioEnv
            base = PortfolioEnv.load_from_local_universe(symbols=symbols, window=max(2, win_hint or 60))
            # Use the common index via any df_map entry (pd available from top-level import)
            idx = next(iter(base.df_map.values())).index
            n = int(len(idx))
            if n < 60:
                return jsonify({'status': 'error', 'detail': 'Not enough aligned rows across symbols to plan PPO'}), 400
            # Prefer start at 2020-01-01 if available
            try:
                y2020 = pd.Timestamp('2020-01-01')
                first_idx = idx[0]
                if y2020 > first_idx and y2020 < idx[-1]:
                    train_start = y2020
                else:
                    train_start = first_idx
            except Exception:
                train_start = idx[0]
            train_end = idx[-1]
            days = int((pd.to_datetime(train_end) - pd.to_datetime(train_start)).days)
            timesteps = int(max(300_000, min(1_200_000, days * 400)))
            window = win_hint if win_hint > 0 else int(max(30, min(120, n // 6)))
            plan = {
                'train_start_date': str(getattr(train_start, 'date', lambda: train_start)()),
                'train_end_date': str(getattr(train_end, 'date', lambda: train_end)()),
                'window': window,
                'training_days': days,
                'timesteps': timesteps,
                'total_rows': n,
                'symbols': symbols,
            }
            return jsonify({'status': 'planned', 'plan': plan})
        else:
            # Single symbol: use MarketEnv
            env = RLMarketEnv.load_from_local(symbols[0], window=max(2, win_hint or 60), tail_days=None)
            df = env.df
            n = int(len(df))
            if n < 60:
                return jsonify({'status': 'error', 'detail': f'Not enough rows to plan PPO for {symbols[0]}'}), 400
            idx = df.index
            if not isinstance(idx, pd.DatetimeIndex):
                try:
                    idx = pd.to_datetime(idx)
                except Exception:
                    pass
            try:
                y2020 = pd.Timestamp('2020-01-01')
                first_idx = idx[0]
                if y2020 > first_idx and y2020 < idx[-1]:
                    train_start = y2020
                else:
                    train_start = first_idx
            except Exception:
                train_start = idx[0]
            train_end = idx[-1]
            days = int((pd.to_datetime(train_end) - pd.to_datetime(train_start)).days)
            timesteps = int(max(300_000, min(1_200_000, days * 400)))
            window = win_hint if win_hint > 0 else int(max(30, min(120, n // 6)))
            plan = {
                'train_start_date': str(getattr(train_start, 'date', lambda: train_start)()),
                'train_end_date': str(getattr(train_end, 'date', lambda: train_end)()),
                'window': window,
                'training_days': days,
                'timesteps': timesteps,
                'total_rows': n,
                'symbols': symbols,
            }
            return jsonify({'status': 'planned', 'plan': plan})
    except Exception as e:
        return jsonify({'status': 'error', 'detail': str(e)}), 500


@app.route('/api/rl/ppo/train', methods=['POST'])
def api_rl_ppo_train():
    """Start PPO training in background using the CLI module rl.training.train_ppo_portfolio.
    UI expects JSON: { status: 'started', job_id }
    """
    # Accept either 'symbols' (comma-separated) or a single 'symbol'
    symbols_param = request.args.get('symbols')
    symbol_single = (request.args.get('symbol') or '').upper()
    symbols_str = None
    if symbols_param:
        symbols_str = ','.join([s.strip().upper() for s in symbols_param.split(',') if s.strip()])
    elif symbol_single:
        symbols_str = symbol_single
    else:
        return jsonify({'status': 'error', 'detail': 'symbol or symbols is required'}), 400
    try:
        window = int(request.args.get('window', '60'))
    except Exception:
        window = 60
    try:
        timesteps = int(request.args.get('timesteps', '100000'))
    except Exception:
        timesteps = 100000
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')

    # Build command
    cmd = [
        sys.executable, '-m', 'rl.training.train_ppo_portfolio',
    '--symbols', symbols_str,
        '--window', str(window),
        '--timesteps', str(timesteps)
    ]
    if start_date:
        cmd += ['--start', start_date]
    if end_date:
        cmd += ['--end', end_date]

    # Optional: if news features file exists, pass it along with default cols
    try:
        nf = (project_root / 'ml' / 'data' / 'news_features.csv')
        if nf.exists():
            cmd += ['--news-features-csv', str(nf)]
            cols = 'news_count,llm_relevant_count,avg_score,fda_count,china_count,geopolitics_count,sentiment_avg'
            cmd += ['--news-cols', cols]
    except Exception:
        pass

    # If only one symbol supplied to a portfolio trainer, duplicate it to satisfy >=2 requirement
    try:
        syms_list = [s for s in (symbols_str or '').split(',') if s]
        if len(syms_list) == 1:
            # Rebuild cmd with duplicated symbol
            dup = f"{syms_list[0]},{syms_list[0]}"
            # Replace the symbols value in cmd
            for i in range(len(cmd)):
                if cmd[i] == '--symbols' and i+1 < len(cmd):
                    cmd[i+1] = dup
                    break
    except Exception:
        pass

    job_id = f"ppo-train-{(symbols_str or 'NA').replace(',', '-')}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    started = run_command_in_background('ppo_train', cmd, job_id)
    if not started:
        return jsonify({'status': 'error', 'detail': 'failed to start training job'}), 500
    return jsonify({'status': 'started', 'job_id': job_id, 'cmd': cmd}), 202


@app.route('/api/rl/ppo/train/status/<job_id>')
def api_rl_ppo_train_status(job_id):
    info = running_jobs.get(job_id)
    logs = job_logs.get(job_id, [])
    if not info:
        return jsonify({'status': 'error', 'detail': 'unknown job_id'}), 404
    # Try to parse model path from logs if any line indicates saving
    model_path = None
    for entry in logs:
        try:
            msg = entry.get('message', '')
            if 'Saved model' in msg or 'Model saved' in msg or 'saved to' in msg.lower():
                # naive extraction: take substring after ':'
                parts = msg.split(':', 1)
                if len(parts) == 2:
                    candidate = parts[1].strip()
                    if candidate:
                        model_path = candidate
        except Exception:
            pass
    return jsonify({
        'status': info.get('status'),
        'job_id': job_id,
        'logs': [l.get('message') for l in logs],
        **({'model_path': model_path} if model_path else {})
    })


# ===========================
# RL Live (Paper) Preview
# ===========================

def _latest_model_path() -> str | None:
    try:
        base = project_root / 'rl' / 'models' / 'ppo_portfolio'
        if not base.exists():
            return None
        zips = sorted(base.rglob('*.zip'), key=lambda p: p.stat().st_mtime, reverse=True)
        return str(zips[0]) if zips else None
    except Exception:
        return None


@app.route('/api/rl/live/latest-model')
def api_rl_live_latest_model():
    p = _latest_model_path()
    if not p:
        return jsonify({'status': 'error', 'detail': 'no model found'}), 404
    return jsonify({'status': 'ok', 'model_path': p})


@app.route('/api/rl/live/preview', methods=['POST'])
def api_rl_live_preview():
    """Compute today's target weights using a trained PPO model (no orders executed)."""
    try:
        data = request.get_json(silent=True) or {}
        symbols_raw = data.get('symbols') or ''
        if isinstance(symbols_raw, str):
            symbols = [s.strip().upper() for s in symbols_raw.split(',') if s.strip()]
        elif isinstance(symbols_raw, list):
            symbols = [str(s).strip().upper() for s in symbols_raw]
        else:
            symbols = []
        if len(symbols) < 2:
            # portfolio env requires >=2; if single provided, duplicate
            if len(symbols) == 1:
                symbols = [symbols[0], symbols[0]]
            else:
                return jsonify({'status': 'error', 'detail': 'Provide at least one symbol'}), 400
        model_path = data.get('model_path') or _latest_model_path()
        if not model_path:
            return jsonify({'status': 'error', 'detail': 'Model path not provided and no latest model found'}), 400
        window = int(data.get('window', 60))
        band = float(data.get('no_trade_band', 0.0) or 0.0)
        min_days = int(data.get('band_min_days', 0) or 0)
        starting_cash = float(data.get('starting_cash', 10000.0) or 10000.0)

        # Build env
        from rl.envs.portfolio_env import PortfolioEnv
        from rl.envs.wrappers_portfolio import PortfolioEnvGym, NoTradeBandActionWrapper
        from stable_baselines3 import PPO  # type: ignore
        import numpy as np
        import pandas as pd

        base = PortfolioEnv.load_from_local_universe(
            symbols=symbols,
            window=window,
            start_date=None,
            end_date=None,
            starting_cash=starting_cash,
        )
        env_gym = PortfolioEnvGym(base)

        # Step to last observation with hold actions
        obs, _ = env_gym.reset()
        n = env_gym.action_space.shape[0]
        hold = np.zeros((n,), dtype=np.float32)
        done = False
        while True:
            obs, reward, terminated, truncated, info = env_gym.step(hold)
            if terminated or truncated:
                break

        # Load model and predict action logits
        model = PPO.load(model_path)
        action, _ = model.predict(obs, deterministic=True)

        # Raw weights
        raw_w = base._to_weights(np.asarray(action, dtype=float)).tolist()

        # Banded weights (optional): transform logits using wrapper logic
        banded_w = None
        if band > 0.0 or min_days > 0:
            wrapper = NoTradeBandActionWrapper(env_gym, band=band, min_days=min_days)
            banded_logits = wrapper.action(np.asarray(action, dtype=float))
            banded_w = base._to_weights(np.asarray(banded_logits, dtype=float)).tolist()

        # Latest date
        try:
            idx_dates = next(iter(base.df_map.values())).index
            latest_dt = idx_dates[base._idx]
            latest_date = str(pd.to_datetime(latest_dt).date())
        except Exception:
            latest_date = None

        # If the list was duplicated for single symbol, show unique in response mapping
        unique_symbols = []
        for s in symbols:
            if s not in unique_symbols:
                unique_symbols.append(s)
        # Map weights to unique symbols by summing duplicates
        def map_unique(w_list: list[float]) -> dict:
            if w_list is None:
                return {}
            agg = {s: 0.0 for s in unique_symbols}
            for s, w in zip(symbols, w_list):
                agg[s] += float(w)
            return agg

        resp = {
            'status': 'ok',
            'date': latest_date,
            'symbols': unique_symbols,
            'raw_weights': map_unique(raw_w),
            'banded_weights': (map_unique(banded_w) if banded_w is not None else None)
        }
        return jsonify(resp)
    except Exception as e:
        return jsonify({'status': 'error', 'detail': str(e)}), 500


# ===========================
# Paper Mode: daily scheduler (no broker)
# ===========================

def _append_paper_log(msg: str) -> None:
    with paper_lock:
        paper_state['logs'].append(f"{datetime.now().isoformat()} | {msg}")
        if len(paper_state['logs']) > 200:
            paper_state['logs'] = paper_state['logs'][-200:]


def _is_business_day(ts: datetime, cal_csv: Path | None) -> bool:
    try:
        if cal_csv and cal_csv.exists():
            import pandas as pd
            cal = pd.read_csv(cal_csv)
            cal['Date'] = pd.to_datetime(cal['Date'])
            d = pd.Timestamp(ts.date())
            return bool((cal['Date'] == d).any())
        # Fallback: weekdays only
        return ts.weekday() < 5
    except Exception:
        return ts.weekday() < 5


def _paper_step(config: dict) -> None:
    """Run one paper step: compute target weights and simulate trades, update positions/equity."""
    from rl.envs.portfolio_env import PortfolioEnv
    from rl.envs.wrappers_portfolio import PortfolioEnvGym
    from stable_baselines3 import PPO  # type: ignore
    import numpy as np
    import pandas as pd

    symbols = config['symbols']
    if len(symbols) == 1:
        # duplicate for env, but we'll aggregate back
        env_symbols = [symbols[0], symbols[0]]
    else:
        env_symbols = symbols
    window = int(config.get('window', 60))
    model_path = config['model_path']
    band = float(config.get('no_trade_band', 0.0) or 0.0)
    min_days = int(config.get('band_min_days', 0) or 0)
    tcost = int(config.get('transaction_cost_bps', 5) or 5)
    slip = int(config.get('slippage_bps', 5) or 5)
    max_turnover_ratio = float(config.get('max_daily_turnover_ratio', 1.0) or 1.0)  # fraction of equity
    max_drawdown_stop = float(config.get('max_drawdown_stop', 0.0) or 0.0)  # e.g., 0.25 for 25%

    # Build env to the latest date
    base = PortfolioEnv.load_from_local_universe(
        symbols=env_symbols,
        window=window,
        start_date=None,
        end_date=None,
        starting_cash=100.0,  # placeholder; we'll use our own cash/positions bookkeeping
        transaction_cost_bps=tcost,
        slippage_bps=slip,
    )
    env_gym = PortfolioEnvGym(base)

    # Step to last observation
    obs, _ = env_gym.reset()
    n = env_gym.action_space.shape[0]
    hold = np.zeros((n,), dtype=np.float32)
    last_idx = base._idx
    while True:
        obs, reward, terminated, truncated, info = env_gym.step(hold)
        if terminated or truncated:
            break
        last_idx = base._idx

    # Load model and predict logits
    model = PPO.load(model_path)
    action, _ = model.predict(obs, deterministic=True)
    target_w = base._to_weights(np.asarray(action, dtype=float))  # env order

    # Map to unique symbols
    unique_symbols = []
    for s in env_symbols:
        if s not in unique_symbols:
            unique_symbols.append(s)
    agg_target = {s: 0.0 for s in unique_symbols}
    for s, w in zip(env_symbols, target_w.tolist()):
        agg_target[s] += float(w)

    # Get latest prices
    idx_dates = next(iter(base.df_map.values())).index
    date_dt = pd.to_datetime(idx_dates[last_idx]).date()
    prices_t = base.prices[last_idx]  # aligned to env_symbols
    price_map = {}
    for s, px in zip(env_symbols, prices_t.tolist()):
        if s in price_map:
            price_map[s] = max(price_map[s], float(px))  # duplicate symbol: keep max to avoid zeros
        else:
            price_map[s] = float(px)

    # Load current positions/cash
    with paper_lock:
        positions = dict(paper_state['positions'])  # symbol->shares
        cash = float(paper_state['cash'])
        peak_equity = float(paper_state.get('peak_equity') or 0.0)

    # Compute equity using unique symbol price
    equity = float(cash)
    for s in unique_symbols:
        sh = float(positions.get(s, 0.0))
        px = float(price_map.get(s, 0.0))
        equity += sh * px

    # Current weights and target with band/min-days
    cur_w = {}
    for s in unique_symbols:
        sh = float(positions.get(s, 0.0))
        px = float(price_map.get(s, 0.0))
        val = sh * px
        cur_w[s] = (val / equity) if equity > 0 else 0.0

    desired = dict(cur_w)
    # min-days: we track last trade date per symbol
    last_trades = (paper_state.get('last_trade_date') or {})
    allow_trade = {}
    for s in unique_symbols:
        last = last_trades.get(s)
        if not last:
            allow_trade[s] = True
        else:
            try:
                days = (datetime.combine(date_dt, datetime.min.time()) - datetime.fromisoformat(last)).days
            except Exception:
                days = 999
            allow_trade[s] = (days >= min_days)

    # Apply band
    changed = False
    for s in unique_symbols:
        tw = float(agg_target.get(s, 0.0))
        if allow_trade[s] and abs(tw - cur_w.get(s, 0.0)) >= band:
            desired[s] = max(0.0, min(1.0, tw))
            changed = True
    # Normalize desired
    ssum = sum(desired.values())
    if ssum > 0:
        for s in desired:
            desired[s] = desired[s] / ssum
    else:
        # fallback equal-weight
        k = len(unique_symbols)
        for s in desired:
            desired[s] = 1.0 / max(1, k)

    # Translate to trades and apply costs
    # First compute desired deltas and total notional to enforce turnover cap
    planned = []  # list of tuples (s, px, delta_sh)
    total_notional = 0.0
    for s in unique_symbols:
        cur_val = cur_w[s] * equity
        tgt_val = desired[s] * equity
        delta_val = tgt_val - cur_val
        px = float(price_map.get(s, 0.0))
        delta_sh = (delta_val / px) if px > 0 else 0.0
        notional = abs(delta_sh) * px
        total_notional += notional
        planned.append((s, px, delta_sh))

    scale = 1.0
    cap_notional = equity * max(1e-9, max_turnover_ratio)
    if total_notional > cap_notional:
        scale = cap_notional / total_notional if total_notional > 0 else 1.0
        _append_paper_log(f"Turnover cap applied: planned={total_notional:.2f} > cap={cap_notional:.2f}, scale={scale:.4f}")

    trades = []
    total_cost = 0.0
    for (s, px, delta_sh0) in planned:
        delta_sh = float(delta_sh0) * float(scale)
        trade_cost = abs(delta_sh) * px * ((tcost + slip) / 10000.0)
        total_cost += float(trade_cost)
        positions[s] = float(positions.get(s, 0.0)) + float(delta_sh)
        cash = float(cash) - float(delta_sh) * float(px) - float(trade_cost)
        trades.append({'symbol': s, 'px': float(px), 'delta_shares': float(delta_sh), 'trade_cost': float(trade_cost)})
        if abs(delta_sh) > 1e-9:
            last_trades[s] = datetime.combine(date_dt, datetime.min.time()).isoformat()

    # Recompute equity after trades at same px
    equity2 = float(cash)
    for s in unique_symbols:
        equity2 += float(positions.get(s, 0.0)) * float(price_map.get(s, 0.0))

    # Update peak equity and enforce drawdown stop if configured
    peak_equity = max(peak_equity, equity2)
    dd = 0.0
    stopped = False
    if peak_equity > 0:
        dd = 1.0 - (equity2 / peak_equity)
    if max_drawdown_stop > 0.0 and dd >= max_drawdown_stop:
        with paper_lock:
            paper_state['running'] = False
        _append_paper_log(f"Max drawdown reached ({dd:.2%} >= {max_drawdown_stop:.2%}). Stopping paper mode.")
        stopped = True

    # Store state and append to ledger
    dec = {
        'date': str(date_dt),
        'raw_weights': {k: float(agg_target.get(k, 0.0)) for k in unique_symbols},
        'banded_weights': {k: float(desired.get(k, 0.0)) for k in unique_symbols},
        'trades': trades,
        'equity_before': float(equity),
        'equity_after': float(equity2),
        'total_cost': float(total_cost),
        'peak_equity': float(peak_equity),
        'drawdown': float(dd),
    }
    out_dir = project_root / 'data' / 'live_paper'
    out_dir.mkdir(parents=True, exist_ok=True)
    # Append JSONL decision
    try:
        with open(out_dir / 'decisions.jsonl', 'a', encoding='utf-8') as f:
            import json
            f.write(json.dumps(dec) + "\n")
    except Exception:
        pass
    # Update state
    with paper_lock:
        paper_state['positions'] = positions
        paper_state['cash'] = float(cash)
        paper_state['equity'] = float(equity2)
        paper_state['peak_equity'] = float(peak_equity)
        paper_state['last_run_date'] = str(date_dt)
        paper_state['last_decision'] = dec
        paper_state['last_trade_date'] = last_trades
    # Persist state for resume
    try:
        state_path = out_dir / 'state.json'
        with open(state_path, 'w', encoding='utf-8') as f:
            json.dump({
                'positions': positions,
                'cash': float(cash),
                'equity': float(equity2),
                'peak_equity': float(peak_equity),
                'last_run_date': str(date_dt),
                'last_trade_date': last_trades,
                'config': config,
            }, f)
    except Exception:
        pass
    _append_paper_log(f"Paper step {date_dt}: equity {equity:.2f} -> {equity2:.2f}, trades={len(trades)}{' [STOPPED]' if stopped else ''}")


def _paper_loop():
    cfg = None
    from time import sleep
    import pytz
    while True:
        with paper_lock:
            if not paper_state['running']:
                break
            cfg = dict(paper_state['config']) if paper_state['config'] else None
        if not cfg:
            sleep(1)
            continue
        # Determine timezone and schedule time
        tz_name = cfg.get('timezone', 'America/New_York')
        schedule_time = cfg.get('schedule_time', '16:05')  # HH:MM local to tz
        try:
            hh, mm = [int(x) for x in schedule_time.split(':', 1)]
        except Exception:
            hh, mm = 16, 5
        try:
            tz = pytz.timezone(tz_name)
        except Exception:
            tz = pytz.timezone('America/New_York')
        now = datetime.now(tz)
        # Check business day
        cal_csv = (project_root / 'data' / 'rl' / 'calendars' / 'market_calendar.csv')
        if not _is_business_day(now, cal_csv):
            sleep(30)
            continue
        run_today = False
        try:
            if now.hour > hh or (now.hour == hh and now.minute >= mm):
                # only run once per date
                with paper_lock:
                    last = paper_state.get('last_run_date')
                if str(now.date()) != str(last):
                    run_today = True
        except Exception:
            run_today = False

        if run_today:
            try:
                _paper_step(cfg)
            except Exception as e:
                _append_paper_log(f"Error in paper step: {e}")
            # Sleep a bit to avoid double-run
            sleep(10)
        else:
            sleep(15)


@app.route('/api/rl/live/paper/start', methods=['POST'])
def api_rl_live_paper_start():
    data = request.get_json(silent=True) or {}
    symbols_raw = data.get('symbols') or ''
    if isinstance(symbols_raw, str):
        symbols = [s.strip().upper() for s in symbols_raw.split(',') if s.strip()]
    elif isinstance(symbols_raw, list):
        symbols = [str(s).strip().upper() for s in symbols_raw]
    else:
        symbols = []
    if len(symbols) < 1:
        return jsonify({'status': 'error', 'detail': 'Provide at least one symbol'}), 400
    model_path = data.get('model_path') or _latest_model_path()
    if not model_path:
        return jsonify({'status': 'error', 'detail': 'Model path not provided and no latest model found'}), 400
    cfg = {
        'symbols': symbols,
        'model_path': model_path,
        'starting_cash': float(data.get('starting_cash', 10000.0) or 10000.0),
        'window': int(data.get('window', 60) or 60),
        'no_trade_band': float(data.get('no_trade_band', 0.05) or 0.05),
        'band_min_days': int(data.get('band_min_days', 5) or 5),
        'transaction_cost_bps': int(data.get('transaction_cost_bps', 5) or 5),
        'slippage_bps': int(data.get('slippage_bps', 5) or 5),
        'schedule_time': str(data.get('schedule_time', '16:05') or '16:05'),
        'timezone': str(data.get('timezone', 'America/New_York') or 'America/New_York'),
        'max_daily_turnover_ratio': float(data.get('max_daily_turnover_ratio', 1.0) or 1.0),
        'max_drawdown_stop': float(data.get('max_drawdown_stop', 0.0) or 0.0),
        'resume': bool(data.get('resume', True)),
    }
    # Initialize or resume positions/cash
    from threading import Thread
    with paper_lock:
        # Reset state on start (may be replaced if resume)
        paper_state['positions'] = {}
        paper_state['cash'] = float(cfg['starting_cash'])
        paper_state['equity'] = float(cfg['starting_cash'])
        paper_state['peak_equity'] = float(cfg['starting_cash'])
        paper_state['config'] = cfg
        paper_state['running'] = True
        paper_state['logs'] = []
        paper_state['last_decision'] = None
        paper_state['last_run_date'] = None
        paper_state['last_trade_date'] = {}
    # Attempt to resume from saved state
    try:
        if cfg.get('resume'):
            state_path = project_root / 'data' / 'live_paper' / 'state.json'
            if state_path.exists():
                with open(state_path, 'r', encoding='utf-8') as f:
                    saved = json.load(f)
                # Only resume if same symbols and model_path
                saved_cfg = saved.get('config') or {}
                same_syms = sorted([s.upper() for s in (saved_cfg.get('symbols') or [])]) == sorted(symbols)
                same_model = str(saved_cfg.get('model_path') or '') == str(model_path)
                if same_syms and same_model:
                    with paper_lock:
                        paper_state['positions'] = saved.get('positions') or {}
                        paper_state['cash'] = float(saved.get('cash') or cfg['starting_cash'])
                        paper_state['equity'] = float(saved.get('equity') or paper_state['cash'])
                        paper_state['peak_equity'] = float(saved.get('peak_equity') or paper_state['equity'])
                        paper_state['last_run_date'] = saved.get('last_run_date')
                        paper_state['last_trade_date'] = saved.get('last_trade_date') or {}
                    _append_paper_log("Resumed paper mode from previous state.json")
                else:
                    _append_paper_log("State.json found but symbols/model mismatch; starting fresh.")
    except Exception as e:
        _append_paper_log(f"Resume failed: {e}")
    th = Thread(target=_paper_loop, daemon=True)
    with paper_lock:
        paper_state['thread'] = th
    th.start()
    _append_paper_log("Paper mode started")
    return jsonify({'status': 'started', 'config': cfg})


@app.route('/api/rl/live/paper/stop', methods=['POST'])
def api_rl_live_paper_stop():
    with paper_lock:
        paper_state['running'] = False
        th = paper_state.get('thread')
    try:
        if th:
            th.join(timeout=1.0)
    except Exception:
        pass
    _append_paper_log("Paper mode stopped")
    return jsonify({'status': 'stopped'})


@app.route('/api/rl/live/paper/status')
def api_rl_live_paper_status():
    with paper_lock:
        snap = {
            'running': paper_state['running'],
            'config': paper_state['config'],
            'last_run_date': paper_state['last_run_date'],
            'positions': paper_state['positions'],
            'cash': paper_state['cash'],
            'equity': paper_state['equity'],
            'last_decision': paper_state['last_decision'],
            'logs': paper_state['logs'][-50:],
        }
    return jsonify(snap)


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
        # Write price CSV
        sdir = root_dir / sym
        sdir.mkdir(parents=True, exist_ok=True)
        price_path = sdir / f"{sym}_price.csv"
        price_df.to_csv(price_path, index=False)
        _log_job(job_id, 'INFO', f"[{sym}] Wrote price CSV: {price_path}")

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

            close = pd.to_numeric(_col_to_series(price_df, 'Close'), errors='coerce').astype(float)
            high = pd.to_numeric(_col_to_series(price_df, 'High'), errors='coerce').astype(float)
            low = pd.to_numeric(_col_to_series(price_df, 'Low'), errors='coerce').astype(float)
            volume = pd.to_numeric(_col_to_series(price_df, 'Volume'), errors='coerce').astype(float)
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
                'Date': price_df['Date'],
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

# ===========================
# Dashboard Routes
# ===========================

@app.route('/')
def dashboard():
    """מציג את דשבורד MarketPulse הראשי"""
    return render_template('dashboard.html')

@app.route('/rl')
def rl_dashboard():
    """RL Dashboard page (experimental)"""
    return render_template('rl_dashboard.html')

@app.route('/docs/progressive-ml')
def docs_progressive_ml():
    """Serve the Progressive ML Guide (static HTML) from templates/docs."""
    return render_template('docs/progressive_ml_guide.html')

@app.route('/health')
def health():
    """בדיקת תקינות השרת"""
    return jsonify({
        'status': 'healthy', 
        'service': 'MarketPulse Dashboard',
        'backend': FASTAPI_BACKEND
    })

# ===========================
# Financial API Proxies
# ===========================

@app.route('/api/financial/market-indices')
def api_market_indices():
    """Proxy to FastAPI: Market indices"""
    return proxy_to_backend('/api/financial/market-indices')

@app.route('/api/financial/market-sentiment')
def api_market_sentiment():
    """Proxy to FastAPI: Market sentiment"""
    return proxy_to_backend('/api/financial/market-sentiment')

@app.route('/api/financial/stock/<symbol>')
def api_stock_data(symbol):
    """Proxy to FastAPI: Stock data"""
    return proxy_to_backend(f'/api/financial/stock/{symbol}')

@app.route('/api/financial/top-stocks')
def api_top_stocks():
    """Proxy to FastAPI: Top performing stocks"""
    return proxy_to_backend('/api/financial/top-stocks')

@app.route('/api/financial/geopolitical-risks')
def api_geopolitical_risks():
    """Proxy to FastAPI: Geopolitical risk analysis"""
    return proxy_to_backend('/api/financial/geopolitical-risks')

# ===========================
# AI API Proxies
# ===========================

@app.route('/api/ai/status')
def api_ai_status():
    """Proxy to FastAPI: AI models status"""
    return proxy_to_backend('/api/ai/status')

@app.route('/api/ai/comprehensive-analysis')
def api_comprehensive_analysis():
    """Proxy to FastAPI: Comprehensive stock analysis"""
    symbol = request.args.get('symbol', 'AAPL')
    return proxy_to_backend(f'/api/ai/comprehensive-analysis/{symbol}')

@app.route('/api/ai/comprehensive-analysis/<symbol>')
def api_comprehensive_analysis_path(symbol):
    """Proxy to FastAPI: Comprehensive stock analysis (path version)"""
    return proxy_to_backend(f'/api/ai/comprehensive-analysis/{symbol}')

# ===========================
# ML API Proxies (NEW)
# ===========================

@app.route('/api/ml/predictions/<symbol>')
def api_ml_predictions(symbol):
    """Proxy to FastAPI: ML predictions for symbol"""
    return proxy_to_backend(f'/api/ml/predictions/{symbol}')

@app.route('/api/ml/train/<symbol>', methods=['POST'])
def api_ml_train(symbol):
    """Proxy to FastAPI: Train ML models for symbol"""
    days_back = request.args.get('days_back', 365)
    return proxy_to_backend(f'/api/ml/train/{symbol}?days_back={days_back}', method='POST')

@app.route('/api/ml/status')
def api_ml_status():
    """Proxy to FastAPI: ML system status"""
    return proxy_to_backend('/api/ml/status')

# Progressive ML endpoints
@app.route('/api/ml/progressive/status')
def api_progressive_ml_status():
    """Proxy to FastAPI: Progressive ML system status"""
    return proxy_to_backend('/api/ml/progressive/status')

@app.route('/api/ml/progressive/models')
def api_progressive_ml_models():
    """Proxy to FastAPI: Progressive ML models info"""
    return proxy_to_backend('/api/ml/progressive/models')

@app.route('/api/ml/progressive/training/status')
def api_progressive_training_status():
    """Proxy to FastAPI: Progressive ML training status"""
    return proxy_to_backend('/api/ml/progressive/training/status')

@app.route('/api/ml/progressive/training/status/<job_id>')
def api_progressive_training_job_status(job_id):
    """Proxy to FastAPI: Progressive ML training job status"""
    return proxy_to_backend(f'/api/ml/progressive/training/status/{job_id}')

@app.route('/api/ml/progressive/train', methods=['POST'])
def api_progressive_train():
    """Proxy to FastAPI: Start Progressive ML training"""
    symbol = request.args.get('symbol', 'AAPL')
    model_types = request.args.get('model_types', 'lstm').split(',')
    mode = request.args.get('mode', 'progressive')
    return proxy_to_backend(f'/api/ml/progressive/train?symbol={symbol}&model_types={",".join(model_types)}&mode={mode}', method='POST')

@app.route('/api/ml/progressive/predict/<symbol>', methods=['POST'])
def api_progressive_predict(symbol):
    """Proxy to FastAPI: Get Progressive ML predictions"""
    mode = request.args.get('mode', 'progressive')
    return proxy_to_backend(f'/api/ml/progressive/predict/{symbol}?mode={mode}', method='POST')

@app.route('/api/ml/progressive/backtest', methods=['POST'])
def api_progressive_backtest():
    """Proxy to FastAPI: Start progressive backtesting"""
    data = request.get_json(silent=True) or {}
    # Forward JSON body as-is; FastAPI expects a Pydantic model in the request body.
    return proxy_to_backend('/api/ml/progressive/backtest', method='POST', timeout=60, json=data)

@app.route('/api/ml/progressive/backtest/status/<job_id>')
def api_progressive_backtest_status(job_id):
    """Proxy to FastAPI: Get backtest status"""
    return proxy_to_backend(f'/api/ml/progressive/backtest/status/{job_id}')

@app.route('/api/ml/progressive/backtest/results/<symbol>')
def api_progressive_backtest_results(symbol):
    """Proxy to FastAPI: Get backtest results"""
    return proxy_to_backend(f'/api/ml/progressive/backtest/results/{symbol}')

# ===========================
# Data Ensure Proxy (NEW)
# ===========================

@app.route('/api/data/ensure/<symbol>')
def api_data_ensure(symbol):
    """Proxy to FastAPI: Ensure per-symbol data exists and is fresh"""
    return proxy_to_backend(f'/api/data/ensure/{symbol}', method='GET', timeout=30)

# ===========================
# Enhanced Financial API Proxies (NEW)
# ===========================

@app.route('/api/financial/sector-performance')
def api_sector_performance():
    """Proxy to FastAPI: Sector performance analysis"""
    return proxy_to_backend('/api/financial/sector-performance')

@app.route('/api/market/<symbol>')
def api_market_data(symbol):
    """Proxy to FastAPI: Real-time market data for symbol"""
    return proxy_to_backend(f'/api/market/{symbol}')

@app.route('/api/sentiment/<symbol>')
def api_sentiment(symbol):
    """Proxy to FastAPI: Social sentiment for symbol"""
    return proxy_to_backend(f'/api/sentiment/{symbol}')

@app.route('/api/watchlist')
def api_watchlist():
    """Proxy to FastAPI: User watchlist"""
    return proxy_to_backend('/api/watchlist')

# ===========================
# System API Proxies (NEW)
# ===========================

@app.route('/api/system/info')
def api_system_info():
    """Proxy to FastAPI: Enhanced system information"""
    return proxy_to_backend('/api/system/info')

# ===========================
# Alerts API Proxies
# ===========================

@app.route('/api/alerts/active')
def api_active_alerts():
    """Proxy to FastAPI: Active alerts"""
    return proxy_to_backend('/api/alerts/active')

@app.route('/api/stats')
def api_stats():
    """Proxy to FastAPI: System statistics"""
    return proxy_to_backend('/api/stats')

@app.route('/api/articles')
def api_articles():
    """Proxy to FastAPI: News articles"""
    limit = request.args.get('limit', 50)
    return proxy_to_backend(f'/api/articles/recent?limit={limit}')

@app.route('/api/articles/recent')
def api_articles_recent():
    """Proxy to FastAPI: Recent news articles"""
    limit = request.args.get('limit', 20)
    return proxy_to_backend(f'/api/articles/recent?limit={limit}')

# ===========================
# Scanner API Proxies
# ===========================

@app.route('/api/scanner/hot-stocks')
def api_hot_stocks():
    """Proxy to FastAPI: Hot stocks scanner"""
    limit = request.args.get('limit', 10)
    return proxy_to_backend(f'/api/scanner/hot-stocks?limit={limit}')

@app.route('/api/ai/market-intelligence')
def api_market_intelligence():
    """Proxy to FastAPI: Market intelligence analysis"""
    return proxy_to_backend('/api/ai/market-intelligence')

@app.route('/api/ai/neural-network-prediction/<symbol>')
def api_neural_network_prediction(symbol):
    """Proxy to FastAPI: Neural network prediction for symbol"""
    return proxy_to_backend(f'/api/ai/neural-network-prediction/{symbol}')

@app.route('/api/ai/time-series-analysis/<symbol>')
def api_time_series_analysis(symbol):
    """Proxy to FastAPI: Time series analysis for symbol"""
    return proxy_to_backend(f'/api/ai/time-series-analysis/{symbol}')

# ===========================
# Predictions API Proxies (NEW)
# ===========================

@app.route('/api/predictions/create', methods=['POST'])
def api_predictions_create():
    """Proxy to FastAPI: Create new prediction"""
    return proxy_to_backend('/api/predictions/create', method='POST', json=request.json)

@app.route('/api/predictions/stats')
def api_predictions_stats():
    """Proxy to FastAPI: Prediction statistics"""
    source = request.args.get('source')
    endpoint = '/api/predictions/stats'
    if source:
        endpoint += f'?source={source}'
    return proxy_to_backend(endpoint)

@app.route('/api/predictions/list')
def api_predictions_list():
    """Proxy to FastAPI: List predictions with filters"""
    params = []
    for param in ['status', 'symbol', 'source', 'limit']:
        value = request.args.get(param)
        if value:
            params.append(f'{param}={value}')
    
    endpoint = '/api/predictions/list'
    if params:
        endpoint += '?' + '&'.join(params)
    return proxy_to_backend(endpoint)

# ===========================
# Scanner API Proxies (Enhanced)
# ===========================

@app.route('/api/scanner/sectors')
def api_scanner_sectors():
    """Proxy to FastAPI: Sector analysis"""
    return proxy_to_backend('/api/scanner/sectors')

@app.route('/api/scanner/sector/<sector_id>')
def api_scanner_sector_detail(sector_id):
    """Proxy to FastAPI: Detailed sector analysis"""
    return proxy_to_backend(f'/api/scanner/sector/{sector_id}')

# ===========================
# Jobs & Feeds API Proxies
# ===========================

@app.route('/api/jobs')
def api_jobs():
    """Proxy to FastAPI: System jobs status"""
    return proxy_to_backend('/api/jobs')

@app.route('/api/feeds/status')
def api_feeds_status():
    """Proxy to FastAPI: RSS feeds status"""
    return proxy_to_backend('/api/feeds/status')

@app.route('/api/statistics')
def api_statistics():
    """Proxy to FastAPI: System statistics"""
    return proxy_to_backend('/api/statistics')

# ===========================
# Trigger API Proxies (NEW)
# ===========================

@app.route('/api/trigger/major-news', methods=['POST'])
def api_trigger_major_news():
    """Proxy to FastAPI: Trigger major news scan"""
    return proxy_to_backend('/api/trigger/major-news', method='POST')

@app.route('/api/trigger/perplexity-scan', methods=['POST'])
def api_trigger_perplexity_scan():
    """Proxy to FastAPI: Trigger Perplexity scan"""
    return proxy_to_backend('/api/trigger/perplexity-scan', method='POST')

@app.route('/api/test-alert', methods=['POST'])
def api_test_alert():
    """Proxy to FastAPI: Send test alert"""
    return proxy_to_backend('/api/test-alert', method='POST')

# ===========================
# Legacy Routes (removed)
# ===========================
# Note: Legacy routes using BACKEND_URL were removed to prevent NameError and
# duplicated endpoints. All clients should use the standardized proxies above
# which forward to FASTAPI_BACKEND.

# ===========================
# Data Management Routes
# ===========================

# Global storage for job logs
job_logs = {}
running_jobs = {}

def run_script_in_background(job_type, script_name, job_id):
    """Run a Python script in the background and capture output"""
    global job_logs, running_jobs
    
    job_logs[job_id] = []
    running_jobs[job_id] = {"status": "running", "start_time": datetime.now().isoformat()}
    
    try:
        # Run the script
        process = subprocess.Popen(
            [sys.executable, script_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=0,  # Unbuffered for real-time output
            universal_newlines=True
        )
        
        # Capture output line by line
        for line in process.stdout:
            line = line.strip()
            if line:
                log_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "level": "INFO",
                    "message": line
                }
                job_logs[job_id].append(log_entry)
                
                # Keep only last 200 lines
                if len(job_logs[job_id]) > 200:
                    job_logs[job_id] = job_logs[job_id][-200:]
        
        process.wait()
        
        if process.returncode == 0:
            running_jobs[job_id]["status"] = "completed"
            job_logs[job_id].append({
                "timestamp": datetime.now().isoformat(),
                "level": "SUCCESS",
                "message": f"✅ Job completed successfully!"
            })
        else:
            running_jobs[job_id]["status"] = "failed"
            job_logs[job_id].append({
                "timestamp": datetime.now().isoformat(),
                "level": "ERROR",
                "message": f"❌ Job failed with exit code {process.returncode}"
            })
    
    except Exception as e:
        running_jobs[job_id]["status"] = "error"
        job_logs[job_id].append({
            "timestamp": datetime.now().isoformat(),
            "level": "ERROR",
            "message": f"❌ Error: {str(e)}"
        })

def run_command_in_background(job_type, cmd_args, job_id):
    """Run an arbitrary command (list of args) in the background and capture output"""
    global job_logs, running_jobs
    job_logs[job_id] = []
    running_jobs[job_id] = {"status": "running", "start_time": datetime.now().isoformat(), "job_type": job_type}

    def _target():
        try:
            process = subprocess.Popen(
                cmd_args,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=0,
                universal_newlines=True,
                cwd=str(project_root)
            )
            out_dir = None
            for line in process.stdout:
                line = line.strip()
                if not line:
                    continue
                # Parse out_dir from known message
                if 'Portfolio evaluation report written to:' in line:
                    try:
                        out_dir = line.split('Portfolio evaluation report written to:')[-1].strip()
                    except Exception:
                        pass
                log_entry = {"timestamp": datetime.now().isoformat(), "level": "INFO", "message": line}
                job_logs[job_id].append(log_entry)
                if len(job_logs[job_id]) > 300:
                    job_logs[job_id] = job_logs[job_id][-300:]
            process.wait()
            if process.returncode == 0:
                running_jobs[job_id]["status"] = "completed"
                if out_dir:
                    running_jobs[job_id]["out_dir"] = out_dir
                job_logs[job_id].append({"timestamp": datetime.now().isoformat(), "level": "SUCCESS", "message": "✅ Job completed successfully!"})
            else:
                running_jobs[job_id]["status"] = "failed"
                job_logs[job_id].append({"timestamp": datetime.now().isoformat(), "level": "ERROR", "message": f"❌ Job failed with exit code {process.returncode}"})
        except Exception as e:
            running_jobs[job_id]["status"] = "error"
            job_logs[job_id].append({"timestamp": datetime.now().isoformat(), "level": "ERROR", "message": f"❌ Error: {str(e)}"})

    th = threading.Thread(target=_target, daemon=True)
    th.start()
    return True

# ===========================
# News Exporter (T-1) — aggregate per-symbol daily features from FastAPI/DB
# ===========================

def _try_fetch_articles(start_iso: str, end_iso: str, symbols: list[str] | None) -> list[dict]:
    """Best-effort fetch of articles from the FastAPI backend.
    Tries multiple endpoints for compatibility. Returns a list of article dicts with keys including
    at least 'published' (ISO), 'title', 'symbol' or 'symbols' (list), 'sentiment' or 'score'.
    """
    items: list[dict] = []
    try:
        # Preferred query endpoint
        params = {"start": start_iso, "end": end_iso}
        if symbols:
            params["symbols"] = ",".join(symbols)
        r = requests.get(f"{FASTAPI_BACKEND}/api/articles/query", params=params, timeout=20)
        if r.status_code == 200:
            j = r.json()
            if isinstance(j, dict) and "items" in j and isinstance(j["items"], list):
                return j["items"]
            if isinstance(j, list):
                return j
    except Exception:
        pass
    try:
        # Fallback by-date endpoint
        r = requests.get(f"{FASTAPI_BACKEND}/api/articles/by-date", params={"start": start_iso, "end": end_iso}, timeout=20)
        if r.status_code == 200:
            j = r.json()
            if isinstance(j, dict) and "items" in j and isinstance(j["items"], list):
                items = j["items"]
            elif isinstance(j, list):
                items = j
    except Exception:
        pass
    if not items:
        try:
            # Last resort: recent with large limit, then client-side filter
            r = requests.get(f"{FASTAPI_BACKEND}/api/articles/recent", params={"limit": 10000}, timeout=20)
            if r.status_code == 200:
                j = r.json()
                if isinstance(j, dict) and "items" in j and isinstance(j["items"], list):
                    items = j["items"]
                elif isinstance(j, list):
                    items = j
        except Exception:
            pass
    # Filter client-side by date and symbols
    try:
        import pandas as pd
        sdt = pd.to_datetime(start_iso)
        edt = pd.to_datetime(end_iso)
        out = []
        for it in items:
            ts = it.get("published") or it.get("published_at") or it.get("Date") or it.get("timestamp")
            if ts is None:
                continue
            try:
                t = pd.to_datetime(ts)
            except Exception:
                continue
            if t < sdt or t > edt:
                continue
            if symbols:
                # Accept if any overlap between provided symbols and item symbols fields
                sy = set()
                sym1 = it.get("symbol")
                if isinstance(sym1, str):
                    sy.add(sym1.upper())
                sy_list = it.get("symbols") or it.get("tickers") or []
                if isinstance(sy_list, list):
                    for s in sy_list:
                        try:
                            sy.add(str(s).upper())
                        except Exception:
                            pass
                # Lightweight fallback: match symbols in title text
                if not sy:
                    title = (it.get("title") or "").upper()
                    for s in symbols:
                        if s.upper() in title:
                            sy.add(s.upper())
                if not sy.intersection({s.upper() for s in symbols}):
                    continue
            out.append(it)
        return out
    except Exception:
        return items


def _compute_news_features_rows(articles: list[dict], wanted_symbols: list[str]) -> list[dict]:
    """Aggregate per-symbol per-publish-day, then shift by +1 day to enforce T-1 availability."""
    import pandas as pd
    import numpy as np
    rows: list[dict] = []
    if not articles:
        return rows
    # Normalize records to (date, symbol, sentiment, score, title/summary)
    norm = []
    for it in articles:
        ts = it.get("published") or it.get("published_at") or it.get("Date") or it.get("timestamp")
        try:
            dt = pd.to_datetime(ts).normalize()
        except Exception:
            continue
        # Extract symbols from item
        sy = set()
        sym1 = it.get("symbol")
        if isinstance(sym1, str):
            sy.add(sym1.upper())
        for f in ("symbols", "tickers", "related"):
            val = it.get(f)
            if isinstance(val, list):
                for s in val:
                    try:
                        sy.add(str(s).upper())
                    except Exception:
                        pass
        # Text for keyword tags
        text = f"{it.get('title') or ''} {it.get('summary') or it.get('description') or ''}".lower()
        sentiment = it.get("sentiment")
        try:
            sentiment = float(sentiment)
        except Exception:
            sentiment = np.nan
        score = it.get("score") or it.get("avg_score") or it.get("confidence")
        try:
            score = float(score)
        except Exception:
            score = np.nan
        # If no symbol attached in article, try to infer from title
        if not sy and wanted_symbols:
            up = text.upper()
            for s in wanted_symbols:
                if s.upper() in up:
                    sy.add(s.upper())
        if not sy:
            continue
        for s in sy:
            norm.append({
                "Date": dt,
                "Symbol": s,
                "sentiment": sentiment,
                "score": score,
                "title_text": text,
            })
    if not norm:
        return rows
    df = pd.DataFrame(norm)
    # Keyword categories
    def has_kw(txt: str, kws: list[str]) -> bool:
        return any(k in txt for k in kws)
    df["fda_hit"] = df["title_text"].apply(lambda t: bool(has_kw(t, ["fda", "drug", "trial"])) )
    df["china_hit"] = df["title_text"].apply(lambda t: bool(has_kw(t, ["china", "beijing", "cpc"])) )
    df["geopol_hit"] = df["title_text"].apply(lambda t: bool(has_kw(t, ["war", "geopolit", "sanction", "tariff"])) )
    # Group by (Date, Symbol)
    agg = df.groupby(["Date", "Symbol"]).agg(
        news_count=("Symbol", "count"),
        sentiment_avg=("sentiment", "mean"),
        avg_score=("score", "mean"),
        fda_count=("fda_hit", "sum"),
        china_count=("china_hit", "sum"),
        geopolitics_count=("geopol_hit", "sum"),
    ).reset_index()
    # LLM relevant: placeholder — if score available, treat score>0 as relevant; else 0
    agg["llm_relevant_count"] = 0
    if "avg_score" in agg.columns:
        agg["llm_relevant_count"] = (agg["avg_score"].fillna(0.0) > 0.0).astype(int)
    # Enforce T-1: shift Date by +1 day
    agg["Date"] = pd.to_datetime(agg["Date"]).dt.tz_localize(None) + pd.Timedelta(days=1)
    # Clip to today-1 to avoid future leakage
    today = pd.Timestamp.now().normalize()
    agg = agg[agg["Date"] <= today]
    # Fill NaNs
    for c in ["avg_score", "sentiment_avg"]:
        if c in agg.columns:
            agg[c] = agg[c].fillna(0.0)
    # Output rows
    cols = ["Date","Symbol","news_count","llm_relevant_count","avg_score","fda_count","china_count","geopolitics_count","sentiment_avg"]
    agg = agg[cols].sort_values(["Symbol","Date"]).reset_index(drop=True)
    rows = agg.to_dict(orient="records")
    return rows


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


@app.route('/api/rl/news/export', methods=['POST'])
def api_rl_news_export():
    data = request.get_json(silent=True) or {}
    symbols_raw = data.get('symbols') or ''
    if isinstance(symbols_raw, str):
        symbols = [s.strip().upper() for s in symbols_raw.split(',') if s.strip()]
    elif isinstance(symbols_raw, list):
        symbols = [str(s).strip().upper() for s in symbols_raw]
    else:
        symbols = []
    start = (data.get('start') or '').strip()
    end = (data.get('end') or '').strip()
    refresh = bool(data.get('refresh', False))
    if not symbols or not start or not end:
        return jsonify({'status':'error','detail':'symbols, start and end are required'}), 400
    job_id = f"news-export-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    running_jobs[job_id] = {"status":"running","start_time": datetime.now().isoformat(), "job_type": "news_export"}
    job_logs[job_id] = []
    th = threading.Thread(target=_export_news_features_job, args=(symbols, start, end, refresh, job_id), daemon=True)
    th.start()
    return jsonify({'status':'started','job_id': job_id}), 202


@app.route('/api/rl/news/export/status/<job_id>')
def api_rl_news_export_status(job_id):
    info = running_jobs.get(job_id)
    logs = job_logs.get(job_id, [])
    if not info:
        return jsonify({'status':'error','detail':'unknown job_id'}), 404
    return jsonify({'status': info.get('status'), 'job_id': job_id, 'logs': [l.get('message') for l in logs]})

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

# ===========================
# RL Portfolio Evaluation (UI helper)
# ===========================

@app.route('/api/rl/portfolio/evaluate', methods=['POST'])
def api_rl_portfolio_evaluate():
    """Start portfolio evaluation report generation in background.
    Runs: python -m rl.evaluation.generate_portfolio_report --symbols ... --eval-start ... --eval-end ...
    """
    symbols = request.args.get('symbols', '').strip()
    eval_start = request.args.get('eval_start', '2024-01-01')
    eval_end = request.args.get('eval_end', '2024-12-31')
    if not symbols:
        return jsonify({"status": "error", "detail": "symbols parameter is required (comma-separated)"}), 400
    job_id = f"portfolio-eval-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    cmd = [sys.executable, "-m", "rl.evaluation.generate_portfolio_report",
           "--symbols", symbols,
           "--eval-start", eval_start,
           "--eval-end", eval_end]
    started = run_command_in_background("portfolio_evaluate", cmd, job_id)
    if not started:
        return jsonify({"status": "error", "detail": "failed to start job"}), 500
    return jsonify({"status": "started", "job_id": job_id, "cmd": cmd}), 202

@app.route('/api/rl/portfolio/evaluate/status/<job_id>')
def api_rl_portfolio_evaluate_status(job_id):
    info = running_jobs.get(job_id)
    logs = job_logs.get(job_id, [])
    if not info:
        return jsonify({"status": "error", "detail": "unknown job_id"}), 404
    resp = {"status": info.get("status"), "job_id": job_id, "logs": [l.get("message") for l in logs]}
    out_dir = info.get("out_dir")
    if out_dir:
        resp["out_dir"] = out_dir
        # try include summary.csv content
        try:
            summary_csv = Path(out_dir) / "summary.csv"
            if summary_csv.exists():
                import csv
                rows = []
                with open(summary_csv, newline='', encoding='utf-8') as f:
                    r = csv.DictReader(f)
                    for row in r:
                        rows.append(row)
                resp["summary"] = rows
        except Exception:
            pass
    return jsonify(resp)

# ===========================
# RL Walk-Forward Evaluation (UI helper)
# ===========================

@app.route('/api/rl/portfolio/walkforward', methods=['POST'])
def api_rl_portfolio_walkforward():
    """Start walk-forward evaluation in background.
    Runs: python -m rl.evaluation.walk_forward --symbols ... --start ... --end ... --segments N
    Optional: band, news args
    """
    symbols = request.args.get('symbols', '').strip()
    start = request.args.get('start', '').strip()
    end = request.args.get('end', '').strip()
    segments = request.args.get('segments', '4').strip()
    if not symbols or not start or not end:
        return jsonify({"status": "error", "detail": "symbols,start,end are required"}), 400
    cmd = [sys.executable, "-m", "rl.evaluation.walk_forward",
           "--symbols", symbols,
           "--start", start,
           "--end", end,
           "--segments", segments]
    # Optional pass-throughs
    for name in [
        ("news-features-csv", request.args.get('news_features_csv')),
        ("news-cols", request.args.get('news_cols')),
        ("news-window", request.args.get('news_window')),
        ("no-trade-band", request.args.get('no_trade_band')),
        ("band-min-days", request.args.get('band_min_days')),
        ("band-transaction-cost-bps", request.args.get('band_transaction_cost_bps')),
        ("band-slippage-bps", request.args.get('band_slippage_bps')),
    ]:
        k, v = name
        if v is not None and str(v).strip() != '':
            cmd += [f"--{k}", str(v)]
    job_id = f"walkforward-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    started = run_command_in_background("walkforward", cmd, job_id)
    if not started:
        return jsonify({"status": "error", "detail": "failed to start walk-forward job"}), 500
    return jsonify({"status": "started", "job_id": job_id, "cmd": cmd}), 202

@app.route('/api/rl/portfolio/walkforward/status/<job_id>')
def api_rl_portfolio_walkforward_status(job_id):
    info = running_jobs.get(job_id)
    logs = job_logs.get(job_id, [])
    if not info:
        return jsonify({"status": "error", "detail": "unknown job_id"}), 404
    resp = {"status": info.get("status"), "job_id": job_id, "logs": [l.get("message") for l in logs]}
    out_dir = info.get("out_dir")
    # Try to infer out_dir from printed line
    if not out_dir:
        for l in logs:
            msg = l.get("message", "")
            if "Walk-forward report written to:" in msg:
                try:
                    info["out_dir"] = msg.split(":", 1)[1].strip()
                    out_dir = info["out_dir"]
                except Exception:
                    pass
    if out_dir:
        resp["out_dir"] = out_dir
        # try include summary.csv
        try:
            summary_csv = Path(out_dir) / "summary.csv"
            if summary_csv.exists():
                import csv
                rows = []
                with open(summary_csv, newline='', encoding='utf-8') as f:
                    r = csv.DictReader(f)
                    for row in r:
                        rows.append(row)
                resp["summary"] = rows
        except Exception:
            pass
    return jsonify(resp)

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