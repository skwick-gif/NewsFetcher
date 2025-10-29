from __future__ import annotations

import os
import sys
import subprocess
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query


router = APIRouter(prefix="/api/rl", tags=["RL"])

# ============================================================
# PPO Training: background process runner + endpoints
# ============================================================

_rl_job_logs: Dict[str, List[Dict[str, Any]]] = {}
_rl_running_jobs: Dict[str, Dict[str, Any]] = {}


def _rl_run_command_in_background(job_type: str, cmd_args: List[str], job_id: str) -> bool:
    _rl_job_logs[job_id] = []
    _rl_running_jobs[job_id] = {"status": "running", "start_time": datetime.now().isoformat(), "job_type": job_type}

    def _target():
        try:
            creationflags = 0
            preexec_fn = None
            try:
                if sys.platform.startswith('win'):
                    creationflags = 0x00000200  # CREATE_NEW_PROCESS_GROUP
                else:
                    import os as _os
                    import signal as _signal  # noqa: F401
                    preexec_fn = _os.setsid
            except Exception:
                pass

            process = subprocess.Popen(
                cmd_args,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=0,
                universal_newlines=True,
                cwd=str(Path(__file__).resolve().parents[2]),  # repo root
                creationflags=creationflags,
                preexec_fn=preexec_fn,
            )
            try:
                _rl_running_jobs[job_id]['pid'] = process.pid
                _rl_running_jobs[job_id]['_popen'] = process
            except Exception:
                pass
            for line in process.stdout or []:
                line = (line or '').strip()
                if not line:
                    continue
                if 'Saved model' in line or 'Model saved' in line or 'saved to' in line.lower():
                    try:
                        parts = line.split(':', 1)
                        if len(parts) == 2:
                            cand = parts[1].strip()
                            if cand:
                                _rl_running_jobs[job_id]['model_path'] = cand
                    except Exception:
                        pass
                _rl_job_logs[job_id].append({"timestamp": datetime.now().isoformat(), "level": "INFO", "message": line})
                if len(_rl_job_logs[job_id]) > 400:
                    _rl_job_logs[job_id] = _rl_job_logs[job_id][-400:]
            process.wait()
            if process.returncode == 0:
                _rl_running_jobs[job_id]["status"] = "completed"
                _rl_job_logs[job_id].append({"timestamp": datetime.now().isoformat(), "level": "SUCCESS", "message": "✅ Job completed successfully!"})
            else:
                _rl_running_jobs[job_id]["status"] = "failed"
                _rl_job_logs[job_id].append({"timestamp": datetime.now().isoformat(), "level": "ERROR", "message": f"❌ Job failed with exit code {process.returncode}"})
        except Exception as e:
            _rl_running_jobs[job_id]["status"] = "error"
            _rl_job_logs[job_id].append({"timestamp": datetime.now().isoformat(), "level": "ERROR", "message": f"❌ Error: {str(e)}"})

    th = threading.Thread(target=_target, daemon=True)
    th.start()
    return True


@router.post("/ppo/train")
async def rl_ppo_train(
    symbols: Optional[str] = Query(None),
    symbol: Optional[str] = Query(None),
    window: int = Query(60, ge=2, le=500),
    timesteps: int = Query(100000, ge=10000, le=5000000),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    add_vix: Optional[bool] = Query(False),
    vix_symbol: Optional[str] = Query("^VIX"),
    use_indicators: Optional[bool] = Query(False),
    indicator_cols: Optional[str] = Query(None),
    use_news: Optional[bool] = Query(True),
):
    syms_str = None
    if symbols:
        syms_str = ",".join([s.strip().upper() for s in symbols.split(',') if s.strip()])
    elif symbol:
        syms_str = symbol.strip().upper()
    else:
        raise HTTPException(status_code=400, detail="symbol or symbols is required")

    cmd = [
        sys.executable, '-m', 'rl.training.train_ppo_portfolio',
        '--symbols', syms_str,
        '--window', str(window),
        '--timesteps', str(timesteps),
    ]
    if start_date:
        cmd += ['--start', start_date]
    if end_date:
        cmd += ['--end', end_date]

    # News CSV auto-detect
    try:
        project_root = Path(__file__).resolve().parents[2]
        nf1 = project_root / 'app' / 'ml' / 'data' / 'news_features.csv'
        nf2 = project_root / 'ml' / 'data' / 'news_features.csv'
        news_csv = nf1 if nf1.exists() else (nf2 if nf2.exists() else None)
        if use_news and news_csv is not None:
            cmd += ['--news-features-csv', str(news_csv)]
            cols = 'news_count,llm_relevant_count,avg_score,fda_count,china_count,geopolitics_count,sentiment_avg'
            cmd += ['--news-cols', cols]
    except Exception:
        pass

    if add_vix:
        cmd += ['--add-vix-feature']
        if vix_symbol:
            cmd += ['--vix-symbol', vix_symbol]

    # Indicators
    try:
        if use_indicators and indicator_cols:
            cmd += ['--use-indicator-features', '--indicator-cols', str(indicator_cols)]
    except Exception:
        pass

    # Duplicate single symbol to satisfy portfolio trainer
    try:
        syms_list = [s for s in (syms_str or '').split(',') if s]
        if len(syms_list) == 1:
            dup = f"{syms_list[0]},{syms_list[0]}"
            for i in range(len(cmd)):
                if cmd[i] == '--symbols' and i + 1 < len(cmd):
                    cmd[i+1] = dup
                    break
    except Exception:
        pass

    job_id = f"ppo-train-{(syms_str or 'NA').replace(',', '-')}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    started = _rl_run_command_in_background('ppo_train', cmd, job_id)
    if not started:
        raise HTTPException(status_code=500, detail='failed to start training job')
    return {"status": "started", "job_id": job_id, "cmd": cmd}


@router.get("/ppo/train/status/{job_id}")
async def rl_ppo_train_status(job_id: str):
    info = _rl_running_jobs.get(job_id)
    logs = _rl_job_logs.get(job_id, [])
    if not info:
        raise HTTPException(status_code=404, detail='unknown job_id')
    model_path = info.get('model_path')
    return {
        'status': info.get('status'),
        'job_id': job_id,
        'logs': [l.get('message') for l in logs],
        **({'model_path': model_path} if model_path else {})
    }


@router.post("/ppo/train/stop/{job_id}")
async def rl_ppo_train_stop(job_id: str):
    info = _rl_running_jobs.get(job_id)
    if not info:
        raise HTTPException(status_code=404, detail='unknown job_id')
    status = info.get('status')
    if status in ('completed', 'failed', 'error', 'cancelled'):
        return {'status': status, 'job_id': job_id, 'message': 'job is not running'}

    pid = info.get('pid')
    popen = info.get('_popen')
    try:
        if popen and getattr(popen, 'poll', None) and popen.poll() is None:
            try:
                if sys.platform.startswith('win'):
                    import signal as _signal
                    popen.send_signal(_signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
                else:
                    import os as _os, signal
                    try:
                        _os.killpg(_os.getpgid(popen.pid), signal.SIGTERM)
                    except Exception:
                        popen.terminate()
            except Exception:
                pass
        if popen and popen.poll() is None:
            if sys.platform.startswith('win') and pid:
                subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
            else:
                try:
                    popen.kill()
                except Exception:
                    pass
    except Exception:
        pass

    info['status'] = 'cancelled'
    _rl_job_logs.setdefault(job_id, []).append({'timestamp': datetime.now().isoformat(), 'level': 'INFO', 'message': '🛑 Stop requested by user.'})
    return {'status': info['status'], 'job_id': job_id, 'stopped': True}


# ============================================================
# RL Auto-Tune (orchestrator)
# ============================================================

_rla_auto_tune_state: Dict[str, Any] = {"status": "idle"}
_rla_auto_tune_lock = threading.Lock()


def _rla_log(msg: str) -> None:
    try:
        logs_dir = Path(__file__).resolve().parents[2] / 'logs'
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = logs_dir / 'auto_tune_stdout.log'
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(msg.rstrip() + "\n")
    except Exception:
        pass


def _auto_tune_runner(config: Dict[str, Any]) -> None:
    from time import sleep
    with _rla_auto_tune_lock:
        _rla_auto_tune_state.update({
            "status": "running",
            "stdout": "",
            "stderr": "",
            "current": None,
            "trials": [],
            "best": None,
            "stop": False,
        })
    project_root = Path(__file__).resolve().parents[2]
    symbols: List[str] = config.get('symbols') or ['QQQ', 'SPY']
    if len(symbols) == 1:
        symbols = [symbols[0], symbols[0]]

    # Simple plan defaults
    window = int(config.get('window') or 60)
    timesteps = int(config.get('timesteps') or 300_000)
    eval_start = str(config.get('eval_start') or '2024-01-01')
    eval_end = str(config.get('eval_end') or datetime.now().date().isoformat())

    # Feature candidates
    news_csv = None
    try:
        c1 = project_root / 'app' / 'ml' / 'data' / 'news_features.csv'
        c2 = project_root / 'ml' / 'data' / 'news_features.csv'
        if c1.exists():
            news_csv = c1.as_posix()
        elif c2.exists():
            news_csv = c2.as_posix()
    except Exception:
        pass
    news_cols = 'news_count,llm_relevant_count,avg_score,fda_count,china_count,geopolitics_count,sentiment_avg'
    ind_cols = config.get('indicator_cols') or 'rsi14,macd_hist,adx14,bb_pctb,bb_width,volume_sma_ratio,roc20,vol20'
    vix_symbol = config.get('vix_symbol') or '^VIX'

    candidates: List[Dict[str, Any]] = []
    candidates.append({'name': f'w{window}_ts{timesteps}_sd0_price', 'args': {}})
    if news_csv:
        candidates.append({'name': f'w{window}_ts{timesteps}_sd0_news', 'args': {'news_csv': news_csv, 'news_cols': news_cols}})
        candidates.append({'name': f'w{window}_ts{timesteps}_sd0_news_vix', 'args': {'news_csv': news_csv, 'news_cols': news_cols, 'add_vix': True, 'vix_symbol': vix_symbol}})
        candidates.append({'name': f'w{window}_ts{timesteps}_sd0_news_vix_indicators', 'args': {'news_csv': news_csv, 'news_cols': news_cols, 'add_vix': True, 'vix_symbol': vix_symbol, 'use_indicators': True, 'indicator_cols': ind_cols}})
    candidates.append({'name': f'w{window}_ts{timesteps}_sd0_vix', 'args': {'add_vix': True, 'vix_symbol': vix_symbol}})
    candidates.append({'name': f'w{window}_ts{timesteps}_sd0_indicators', 'args': {'use_indicators': True, 'indicator_cols': ind_cols}})

    seeds = config.get('seeds') or [0]
    best: Optional[Dict[str, Any]] = None

    try:
        for sd in seeds:
            for cand in candidates:
                with _rla_auto_tune_lock:
                    if _rla_auto_tune_state.get('stop'):
                        raise KeyboardInterrupt('Auto-tune stopped by user')
                    _rla_auto_tune_state['current'] = {'seed': sd, 'name': cand['name']}

                cmd = [
                    sys.executable, '-m', 'rl.training.train_ppo_portfolio',
                    '--symbols', ','.join(symbols),
                    '--window', str(window),
                    '--timesteps', str(timesteps),
                    '--seed', str(sd),
                ]
                args = cand['args']
                if args.get('news_csv'):
                    cmd += ['--news-features-csv', str(args['news_csv']), '--news-cols', str(args.get('news_cols', news_cols)), '--news-window', '1']
                if args.get('add_vix'):
                    cmd += ['--add-vix-feature', '--vix-symbol', str(args.get('vix_symbol', '^VIX'))]
                if args.get('use_indicators'):
                    cmd += ['--use-indicator-features', '--indicator-cols', str(args.get('indicator_cols', ind_cols))]

                _rla_log(f"Training trial {cand['name']} seed={sd} ...")
                try:
                    proc = subprocess.Popen(cmd, cwd=str(project_root), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, universal_newlines=True)
                    with _rla_auto_tune_lock:
                        _rla_auto_tune_state['pid'] = proc.pid
                        _rla_auto_tune_state['cmd'] = ' '.join(cmd)
                    for line in proc.stdout or []:
                        _rla_log(line.rstrip())
                        with _rla_auto_tune_lock:
                            prev = _rla_auto_tune_state.get('stdout', '') or ''
                            snap = (prev + ('\n' if prev else '') + line.rstrip())
                            _rla_auto_tune_state['stdout'] = snap[-2000:]
                    proc.wait()
                    rc = proc.returncode
                except Exception as e:
                    rc = -1
                    _rla_log(f"Error running training: {e}")
                if rc != 0:
                    with _rla_auto_tune_lock:
                        trials = _rla_auto_tune_state.get('trials', [])
                        trials.append({'name': cand['name'], 'seed': sd, 'status': 'failed'})
                        _rla_auto_tune_state['trials'] = trials
                    continue

                eval_out = project_root / 'reports' / 'rl' / f"auto_tune_{datetime.now().strftime('%Y%m%d_%H%M%S')}" / cand['name']
                ev_cmd = [sys.executable, '-m', 'rl.evaluation.generate_portfolio_report', '--symbols', ','.join(symbols), '--eval-start', eval_start, '--eval-end', eval_end, '--out', eval_out.as_posix()]
                if args.get('news_csv'):
                    ev_cmd += ['--news-features-csv', str(args['news_csv']), '--news-cols', str(args.get('news_cols', news_cols)), '--news-window', '1']
                if args.get('use_indicators'):
                    ev_cmd += ['--use-indicator-features', '--indicator-cols', str(args.get('indicator_cols', ind_cols))]
                _rla_log(f"Evaluating trial {cand['name']} seed={sd} ...")
                try:
                    proc2 = subprocess.Popen(ev_cmd, cwd=str(project_root), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, universal_newlines=True)
                    with _rla_auto_tune_lock:
                        _rla_auto_tune_state['pid'] = proc2.pid
                        _rla_auto_tune_state['cmd'] = ' '.join(ev_cmd)
                    for line in proc2.stdout or []:
                        _rla_log(line.rstrip())
                    proc2.wait()
                except Exception as e:
                    _rla_log(f"Error running evaluation: {e}")

                summary = None
                try:
                    import csv
                    s_csv = eval_out / 'summary.csv'
                    if s_csv.exists():
                        with open(s_csv, newline='', encoding='utf-8') as f:
                            rows = list(csv.DictReader(f))
                        for r in rows:
                            if r.get('kind') == 'ppo_portfolio':
                                summary = r
                                break
                except Exception:
                    summary = None

                trial = {'name': cand['name'], 'seed': sd, 'status': 'completed' if summary else 'no_summary', 'summary': summary}
                with _rla_auto_tune_lock:
                    trials = _rla_auto_tune_state.get('trials', [])
                    trials.append(trial)
                    _rla_auto_tune_state['trials'] = trials

                if summary:
                    try:
                        sharpe = float(summary.get('sharpe') or 0.0)
                        mdd = abs(float(summary.get('max_drawdown') or summary.get('max_dd') or 0.0))
                        score = sharpe - 0.5 * mdd
                    except Exception:
                        score = -1e9
                    rec = {'trial': trial, 'score': score, 'dir': eval_out.as_posix()}
                    if best is None or score > best.get('score', -1e9):
                        best = rec
                        with _rla_auto_tune_lock:
                            _rla_auto_tune_state['best'] = rec

        with _rla_auto_tune_lock:
            _rla_auto_tune_state['status'] = 'completed'
            _rla_auto_tune_state['pid'] = None
            _rla_auto_tune_state['cmd'] = None
    except KeyboardInterrupt:
        with _rla_auto_tune_lock:
            _rla_auto_tune_state['status'] = 'cancelled'
            _rla_auto_tune_state['pid'] = None
            _rla_auto_tune_state['cmd'] = None
    except Exception as e:
        _rla_log(f"Auto-tune failed: {e}")
        with _rla_auto_tune_lock:
            _rla_auto_tune_state['status'] = 'error'
            _rla_auto_tune_state['stderr'] = str(e)
            _rla_auto_tune_state['pid'] = None
            _rla_auto_tune_state['cmd'] = None


@router.post('/auto-tune/start')
async def rl_auto_tune_start(payload: Optional[Dict[str, Any]] = None):
    try:
        data = payload or {}
        mode = (data.get('mode') or 'etf').lower()
        job_id = f"rla-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        if mode == 'stock':
            sym = str(data.get('symbol') or 'AAPL').strip().upper()
            symbols = [sym]
        else:
            etfs = data.get('etfs') or ['QQQ','SPY']
            if isinstance(etfs, list):
                symbols = [str(s).strip().upper() for s in etfs if str(s).strip()]
            else:
                symbols = [s for s in str(etfs).split(',') if s]
        cfg = {
            'symbols': symbols,
            'time_budget_hours': float(data.get('time_budget_hours') or 6.0),
            'window': int(data.get('window') or 60),
            'timesteps': int(data.get('timesteps') or 300_000),
            'eval_start': data.get('eval_start'),
            'eval_end': data.get('eval_end'),
            'seeds': data.get('seeds') or [0],
            'indicator_cols': data.get('indicator_cols'),
            'vix_symbol': data.get('vix_symbol') or '^VIX',
        }
        with _rla_auto_tune_lock:
            _rla_auto_tune_state.clear()
            _rla_auto_tune_state.update({
                'status': 'running',
                'job_id': job_id,
                'started_at': datetime.utcnow().isoformat() + 'Z',
                'config': cfg,
                'pid': None,
                'cmd': None,
                'stdout': 'Starting RL auto-tune...',
                'stderr': '',
                'trials': [],
                'best': None,
            })
        th = threading.Thread(target=_auto_tune_runner, args=(cfg,), daemon=True)
        th.start()
        return {'status': 'ok', 'data': _rla_auto_tune_state}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Auto-tune start failed: {e}")


@router.get('/auto-tune/status')
async def rl_auto_tune_status():
    with _rla_auto_tune_lock:
        data = dict(_rla_auto_tune_state)
    return {'status': 'ok', 'data': data}


@router.post('/auto-tune/stop')
async def rl_auto_tune_stop():
    try:
        with _rla_auto_tune_lock:
            if _rla_auto_tune_state.get('status') == 'running':
                _rla_auto_tune_state['stop'] = True
                pid = _rla_auto_tune_state.get('pid')
                if pid:
                    try:
                        if sys.platform.startswith('win'):
                            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
                        else:
                            os.kill(int(pid), 15)
                    except Exception:
                        pass
        return {'status': 'ok', 'data': _rla_auto_tune_state}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Auto-tune stop failed: {e}")


# ============================================================
# RL Live Endpoints (latest-model, preview, paper mode)
# ============================================================
def _rl_latest_model_path() -> Optional[str]:
    try:
        base = Path(__file__).resolve().parents[2] / 'rl' / 'models' / 'ppo_portfolio'
        if not base.exists():
            return None
        zips = sorted(base.rglob('*.zip'), key=lambda p: p.stat().st_mtime, reverse=True)
        return str(zips[0]) if zips else None
    except Exception:
        return None


@router.get('/live/latest-model')
async def rl_live_latest_model():
    p = _rl_latest_model_path()
    if not p:
        raise HTTPException(status_code=404, detail='no model found')
    return {'status': 'ok', 'model_path': p}


@router.post('/live/preview')
async def rl_live_preview(data: Dict[str, Any]):
    try:
        symbols_raw = data.get('symbols') or ''
        if isinstance(symbols_raw, str):
            symbols_in = [s.strip().upper() for s in symbols_raw.split(',') if s.strip()]
        elif isinstance(symbols_raw, list):
            symbols_in = [str(s).strip().upper() for s in symbols_raw]
        else:
            symbols_in = []
        if len(symbols_in) < 1:
            raise HTTPException(status_code=400, detail='Provide at least one symbol')
        symbols_env = symbols_in if len(symbols_in) >= 2 else [symbols_in[0], symbols_in[0]]

        model_path = data.get('model_path') or _rl_latest_model_path()
        if not model_path:
            raise HTTPException(status_code=400, detail='Model path not provided and no latest model found')
        window = int(data.get('window', 60) or 60)
        band = float(data.get('no_trade_band', 0.0) or 0.0)
        min_days = int(data.get('band_min_days', 0) or 0)
        starting_cash = float(data.get('starting_cash', 10000.0) or 10000.0)
        vix_symbol = str(data.get('vix_symbol') or '^VIX')

        from stable_baselines3 import PPO  # type: ignore
        import numpy as np  # type: ignore
        import pandas as pd  # type: ignore
        from rl.training.train_ppo_portfolio import make_env  # type: ignore
        from rl.envs.wrappers_portfolio import NoTradeBandActionWrapper  # type: ignore

        model = PPO.load(model_path)
        try:
            expected = int(getattr(model.policy.observation_space, 'shape', [None])[0])
        except Exception:
            expected = None

        def _default_news_csv() -> Optional[str]:
            try:
                project_root = Path(__file__).resolve().parents[2]
                cand1 = project_root / 'app' / 'ml' / 'data' / 'news_features.csv'
                if cand1.exists():
                    return cand1.as_posix()
                cand2 = project_root / 'ml' / 'data' / 'news_features.csv'
                if cand2.exists():
                    return cand2.as_posix()
            except Exception:
                pass
            return None

        news_csv = _default_news_csv()
        news_cols = ['news_count','llm_relevant_count','avg_score','fda_count','china_count','geopolitics_count','sentiment_avg']
        ind_cols = ['rsi14','macd_hist','adx14','bb_pctb','bb_width','volume_sma_ratio','roc20','vol20']

        tried: List[Dict[str, Any]] = []

        def attempt(opts: Dict[str, Any]):
            env = make_env(
                symbols=symbols_env,
                window=window,
                start=None,
                end=None,
                starting_cash=starting_cash,
                news_features_csv=opts.get('news_csv'),
                news_cols=opts.get('news_cols'),
                news_window=int(opts.get('news_window', 1)),
                add_vix_feature=bool(opts.get('add_vix', False)),
                vix_symbol=str(opts.get('vix_symbol', vix_symbol)),
                use_indicator_features=bool(opts.get('use_indicators', False)),
                indicator_cols=opts.get('indicator_cols'),
            )
            obs, _ = env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
            cur_dim = int(np.array(obs, dtype=float).shape[0])
            return env, cur_dim

        candidates = []
        candidates.append({'news_csv': None, 'news_cols': None, 'add_vix': False, 'use_indicators': False})
        if news_csv:
            candidates.append({'news_csv': news_csv, 'news_cols': news_cols, 'news_window': 1})
            candidates.append({'news_csv': news_csv, 'news_cols': news_cols, 'news_window': 1, 'add_vix': True, 'vix_symbol': vix_symbol})
            candidates.append({'news_csv': news_csv, 'news_cols': news_cols, 'news_window': 1, 'add_vix': True, 'vix_symbol': vix_symbol, 'use_indicators': True, 'indicator_cols': ind_cols})
        candidates.append({'add_vix': True, 'vix_symbol': vix_symbol})
        candidates.append({'use_indicators': True, 'indicator_cols': ind_cols})

        env_gym = None
        chosen = None
        cur_dim = None
        for opts in candidates:
            try:
                env, d = attempt(opts)
                tried.append({'opts': opts, 'dim': d})
                if expected is None or d == expected:
                    env_gym = env
                    chosen = opts
                    cur_dim = d
                    break
            except Exception as _e:
                tried.append({'opts': opts, 'error': str(_e)})
                continue

        if env_gym is None:
            msg = f"Unexpected observation shape; model expects {expected}, tried: " + \
                  ", ".join([str(t.get('dim') or t.get('error')) for t in tried])
            raise HTTPException(status_code=400, detail=msg)

        obs, _ = env_gym.reset()
        n = env_gym.action_space.shape[0]
        hold = np.zeros((n,), dtype=np.float32)
        done = False
        while True:
            obs, reward, terminated, truncated, info = env_gym.step(hold)
            if terminated or truncated:
                break

        action, _ = model.predict(obs, deterministic=True)
        base_env = getattr(env_gym, '_env', None) or getattr(env_gym, 'unwrapped', None) or env_gym
        raw_w = base_env._to_weights(np.asarray(action, dtype=float)).tolist()

        banded_w = None
        if band > 0.0 or min_days > 0:
            wrapper = NoTradeBandActionWrapper(env_gym, band=float(band), min_days=int(min_days))
            banded_logits = wrapper.action(np.asarray(action, dtype=float))
            banded_w = base_env._to_weights(np.asarray(banded_logits, dtype=float)).tolist()

        try:
            idx_dates = next(iter(base_env.df_map.values())).index
            latest_dt = idx_dates[base_env._idx]
            latest_date = str(pd.to_datetime(latest_dt).date())
        except Exception:
            latest_date = None

        unique_symbols: List[str] = []
        for s in symbols_in:
            if s not in unique_symbols:
                unique_symbols.append(s)

        def map_unique(w_list: Optional[List[float]]) -> Dict[str, float]:
            if w_list is None:
                return {}
            agg = {s: 0.0 for s in unique_symbols}
            for s, w in zip(symbols_env, w_list):
                agg[s] += float(w)
            return agg

        resp = {
            'status': 'ok',
            'date': latest_date,
            'symbols': unique_symbols,
            'raw_weights': map_unique(raw_w),
            'banded_weights': (map_unique(banded_w) if banded_w is not None else None),
            'obs_dim': cur_dim,
            'matched_opts': chosen,
        }
        def _safe(obj):
            import math
            if isinstance(obj, float):
                return obj if math.isfinite(obj) else None
            if isinstance(obj, dict):
                return {str(k): _safe(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_safe(x) for x in obj]
            return obj
        return _safe(resp)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Paper Mode state and endpoints
_paper_state: Dict[str, Any] = {
    'running': False,
    'thread': None,
    'config': None,
    'last_run_date': None,
    'positions': {},
    'cash': 0.0,
    'equity': 0.0,
    'peak_equity': 0.0,
    'logs': [],
    'last_decision': None,
}
_paper_lock = threading.Lock()


def _append_paper_log(msg: str) -> None:
    with _paper_lock:
        _paper_state['logs'].append(f"{datetime.now().isoformat()} | {msg}")
        if len(_paper_state['logs']) > 200:
            _paper_state['logs'] = _paper_state['logs'][-200:]


def _is_business_day(ts: datetime, cal_csv: Optional[Path]) -> bool:
    try:
        if cal_csv and cal_csv.exists():
            import pandas as pd
            cal = pd.read_csv(cal_csv)
            cal['Date'] = pd.to_datetime(cal['Date'])
            d = pd.Timestamp(ts.date())
            return bool((cal['Date'] == d).any())
        return ts.weekday() < 5
    except Exception:
        return ts.weekday() < 5


def _paper_step(config: Dict[str, Any]) -> None:
    from stable_baselines3 import PPO  # type: ignore
    import numpy as np  # type: ignore
    import pandas as pd  # type: ignore
    from rl.training.train_ppo_portfolio import make_env  # type: ignore

    project_root = Path(__file__).resolve().parents[2]

    symbols = config['symbols']
    env_symbols = symbols if len(symbols) > 1 else [symbols[0], symbols[0]]
    window = int(config.get('window', 60))
    model_path = config['model_path']
    band = float(config.get('no_trade_band', 0.0) or 0.0)
    min_days = int(config.get('band_min_days', 0) or 0)
    tcost = int(config.get('transaction_cost_bps', 5) or 5)
    slip = int(config.get('slippage_bps', 5) or 5)
    max_turnover_ratio = float(config.get('max_daily_turnover_ratio', 1.0) or 1.0)
    max_drawdown_stop = float(config.get('max_drawdown_stop', 0.0) or 0.0)

    model = PPO.load(model_path)
    try:
        expected = int(getattr(model.policy.observation_space, 'shape', [None])[0])
    except Exception:
        expected = None

    def _default_news_csv() -> Optional[str]:
        try:
            c1 = project_root / 'app' / 'ml' / 'data' / 'news_features.csv'
            if c1.exists():
                return c1.as_posix()
            c2 = project_root / 'ml' / 'data' / 'news_features.csv'
            if c2.exists():
                return c2.as_posix()
        except Exception:
            pass
        return None

    news_csv = _default_news_csv()
    news_cols = ['news_count','llm_relevant_count','avg_score','fda_count','china_count','geopolitics_count','sentiment_avg']
    ind_cols = ['rsi14','macd_hist','adx14','bb_pctb','bb_width','volume_sma_ratio','roc20','vol20']
    candidates = []
    candidates.append({'news_csv': None, 'news_cols': None, 'add_vix': False, 'use_indicators': False})
    if news_csv:
        candidates.append({'news_csv': news_csv, 'news_cols': news_cols, 'news_window': 1})
        candidates.append({'news_csv': news_csv, 'news_cols': news_cols, 'news_window': 1, 'add_vix': True, 'vix_symbol': '^VIX'})
        candidates.append({'news_csv': news_csv, 'news_cols': news_cols, 'news_window': 1, 'add_vix': True, 'vix_symbol': '^VIX', 'use_indicators': True, 'indicator_cols': ind_cols})
    candidates.append({'add_vix': True, 'vix_symbol': '^VIX'})
    candidates.append({'use_indicators': True, 'indicator_cols': ind_cols})

    env_gym = None
    for opts in candidates:
        try:
            env = make_env(
                symbols=env_symbols,
                window=window,
                start=None,
                end=None,
                transaction_cost_bps=tcost,
                slippage_bps=slip,
                starting_cash=100.0,
                news_features_csv=opts.get('news_csv'),
                news_cols=opts.get('news_cols'),
                news_window=int(opts.get('news_window', 1)),
                add_vix_feature=bool(opts.get('add_vix', False)),
                vix_symbol=str(opts.get('vix_symbol', '^VIX')),
                use_indicator_features=bool(opts.get('use_indicators', False)),
                indicator_cols=opts.get('indicator_cols'),
            )
            obs, _ = env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
            dim = int(np.array(obs, dtype=float).shape[0])
            if expected is None or dim == expected:
                env_gym = env
                break
        except Exception:
            continue
    if env_gym is None:
        env_gym = make_env(symbols=env_symbols, window=window, start=None, end=None, transaction_cost_bps=tcost, slippage_bps=slip, starting_cash=100.0)

    obs, _ = env_gym.reset()
    n = env_gym.action_space.shape[0]
    hold = np.zeros((n,), dtype=np.float32)
    last_idx = getattr(getattr(env_gym, '_env', None) or getattr(env_gym, 'unwrapped', None) or env_gym, '_idx', None)
    while True:
        obs, reward, terminated, truncated, info = env_gym.step(hold)
        if terminated or truncated:
            break
        base_env_loop = getattr(env_gym, '_env', None) or getattr(env_gym, 'unwrapped', None) or env_gym
        try:
            last_idx = base_env_loop._idx
        except Exception:
            pass

    action, _ = model.predict(obs, deterministic=True)
    base_env = getattr(env_gym, '_env', None) or getattr(env_gym, 'unwrapped', None) or env_gym
    target_w = base_env._to_weights(np.asarray(action, dtype=float))

    unique_symbols: List[str] = []
    for s in env_symbols:
        if s not in unique_symbols:
            unique_symbols.append(s)
    agg_target = {s: 0.0 for s in unique_symbols}
    for s, w in zip(env_symbols, target_w.tolist()):
        agg_target[s] += float(w)

    idx_dates = next(iter(base_env.df_map.values())).index
    date_dt = pd.to_datetime(idx_dates[last_idx]).date()
    prices_t = base_env.prices[last_idx]
    price_map: Dict[str, float] = {}
    for s, px in zip(env_symbols, prices_t.tolist()):
        if s in price_map:
            price_map[s] = max(price_map[s], float(px))
        else:
            price_map[s] = float(px)

    with _paper_lock:
        positions = dict(_paper_state['positions'])
        cash = float(_paper_state['cash'])
        peak_equity = float(_paper_state.get('peak_equity') or 0.0)

    equity = float(cash)
    for s in unique_symbols:
        sh = float(positions.get(s, 0.0))
        px = float(price_map.get(s, 0.0))
        equity += sh * px

    cur_w: Dict[str, float] = {}
    for s in unique_symbols:
        sh = float(positions.get(s, 0.0))
        px = float(price_map.get(s, 0.0))
        val = sh * px
        cur_w[s] = (val / equity) if equity > 0 else 0.0

    desired = dict(cur_w)
    last_trades = (_paper_state.get('last_trade_date') or {})
    allow_trade: Dict[str, bool] = {}
    for s in unique_symbols:
        last = last_trades.get(s)
        if not last:
            allow_trade[s] = True
        else:
            try:
                from datetime import datetime as _dt
                days = (_dt.combine(date_dt, _dt.min.time()) - _dt.fromisoformat(last)).days
            except Exception:
                days = 999
            allow_trade[s] = (days >= min_days)

    changed = False
    for s in unique_symbols:
        tw = float(agg_target.get(s, 0.0))
        if allow_trade[s] and abs(tw - cur_w.get(s, 0.0)) >= band:
            desired[s] = max(0.0, min(1.0, tw))
            changed = True
    ssum = sum(desired.values())
    if ssum > 0:
        for s in desired:
            desired[s] = desired[s] / ssum
    else:
        k = len(unique_symbols)
        for s in desired:
            desired[s] = 1.0 / max(1, k)

    trades = []
    total_cost = 0.0
    for s in unique_symbols:
        cur_val = cur_w[s] * equity
        tgt_val = desired[s] * equity
        delta_val = tgt_val - cur_val
        px = float(price_map.get(s, 0.0))
        delta_sh = (delta_val / px) if px > 0 else 0.0
        trade_cost = abs(delta_sh) * px * ((tcost + slip) / 10000.0)
        total_cost += float(trade_cost)
        trades.append({'symbol': s, 'px': float(px), 'delta_shares': float(delta_sh), 'trade_cost': float(trade_cost)})

    return {
        'status': 'ok',
        'date': str(date_dt),
        'symbols': unique_symbols,
        'raw_weights': {k: float(agg_target.get(k, 0.0)) for k in unique_symbols},
        'banded_weights': {k: float(desired.get(k, 0.0)) for k in unique_symbols},
        'trades': trades,
        'equity_before': float(equity),
        'equity_after': float(equity),
        'total_cost': float(total_cost),
    }


def _paper_loop():
    cfg = None
    from time import sleep
    import pytz  # type: ignore
    while True:
        with _paper_lock:
            if not _paper_state['running']:
                break
            cfg = dict(_paper_state['config']) if _paper_state['config'] else None
        if not cfg:
            sleep(1)
            continue
        tz_name = cfg.get('timezone', 'America/New_York')
        schedule_time = cfg.get('schedule_time', '16:05')
        try:
            hh, mm = [int(x) for x in schedule_time.split(':', 1)]
        except Exception:
            hh, mm = 16, 5
        try:
            tz = pytz.timezone(tz_name)
        except Exception:
            tz = pytz.timezone('America/New_York')
        now = datetime.now(tz)
        cal_csv = (Path(__file__).resolve().parents[2] / 'data' / 'rl' / 'calendars' / 'market_calendar.csv')
        if not _is_business_day(now, cal_csv):
            sleep(30)
            continue
        run_today = False
        try:
            if now.hour > hh or (now.hour == hh and now.minute >= mm):
                with _paper_lock:
                    last = _paper_state.get('last_run_date')
                if str(now.date()) != str(last):
                    run_today = True
        except Exception:
            run_today = False

        if run_today:
            try:
                _paper_step(cfg)
            except Exception as e:
                _append_paper_log(f"Error in paper step: {e}")
            sleep(10)
        else:
            sleep(15)


@router.post('/live/paper/start')
async def rl_live_paper_start(data: Dict[str, Any]):
    symbols_raw = data.get('symbols') or ''
    if isinstance(symbols_raw, str):
        symbols = [s.strip().upper() for s in symbols_raw.split(',') if s.strip()]
    elif isinstance(symbols_raw, list):
        symbols = [str(s).strip().upper() for s in symbols_raw]
    else:
        symbols = []
    if len(symbols) < 1:
        raise HTTPException(status_code=400, detail='Provide at least one symbol')
    model_path = data.get('model_path') or _rl_latest_model_path()
    if not model_path:
        raise HTTPException(status_code=400, detail='Model path not provided and no latest model found')
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
    from threading import Thread
    with _paper_lock:
        _paper_state['positions'] = {}
        _paper_state['cash'] = float(cfg['starting_cash'])
        _paper_state['equity'] = float(cfg['starting_cash'])
        _paper_state['peak_equity'] = float(cfg['starting_cash'])
        _paper_state['config'] = cfg
        _paper_state['running'] = True
        _paper_state['logs'] = []
        _paper_state['last_decision'] = None
        _paper_state['last_run_date'] = None
        _paper_state['last_trade_date'] = {}
    try:
        if cfg.get('resume'):
            state_path = Path(__file__).resolve().parents[2] / 'data' / 'live_paper' / 'state.json'
            if state_path.exists():
                import json as _json
                with open(state_path, 'r', encoding='utf-8') as f:
                    saved = _json.load(f)
                saved_cfg = saved.get('config') or {}
                same_syms = sorted([s.upper() for s in (saved_cfg.get('symbols') or [])]) == sorted(symbols)
                same_model = str(saved_cfg.get('model_path') or '') == str(model_path)
                if same_syms and same_model:
                    with _paper_lock:
                        _paper_state['positions'] = saved.get('positions') or {}
                        _paper_state['cash'] = float(saved.get('cash') or cfg['starting_cash'])
                        _paper_state['equity'] = float(saved.get('equity') or _paper_state['cash'])
                        _paper_state['peak_equity'] = float(saved.get('peak_equity') or _paper_state['equity'])
                        _paper_state['last_run_date'] = saved.get('last_run_date')
                        _paper_state['last_trade_date'] = saved.get('last_trade_date') or {}
                    _append_paper_log("Resumed paper mode from previous state.json")
                else:
                    _append_paper_log("State.json found but symbols/model mismatch; starting fresh.")
    except Exception as e:
        _append_paper_log(f"Resume failed: {e}")
    th = Thread(target=_paper_loop, daemon=True)
    with _paper_lock:
        _paper_state['thread'] = th
    th.start()
    _append_paper_log("Paper mode started")
    return {'status': 'started', 'config': cfg}


@router.post('/live/paper/stop')
async def rl_live_paper_stop():
    with _paper_lock:
        _paper_state['running'] = False
        th = _paper_state.get('thread')
    try:
        if th:
            th.join(timeout=1.0)
    except Exception:
        pass
    _append_paper_log("Paper mode stopped")
    return {'status': 'stopped'}


@router.get('/live/paper/status')
async def rl_live_paper_status():
    with _paper_lock:
        snap = {
            'running': _paper_state['running'],
            'config': _paper_state['config'],
            'last_run_date': _paper_state['last_run_date'],
            'positions': _paper_state['positions'],
            'cash': _paper_state['cash'],
            'equity': _paper_state['equity'],
            'last_decision': _paper_state['last_decision'],
            'logs': _paper_state['logs'][-50:],
        }
    return snap


# ============================================================
# PPO Plan endpoint
# ============================================================
@router.get("/ppo/plan")
async def rl_ppo_plan(symbols: Optional[str] = None, symbol: Optional[str] = None, window: int = Query(60, ge=2, le=500)):
    try:
        try:
            import pandas as pd
            from rl.envs.market_env import MarketEnv as RLMarketEnv  # type: ignore
            from rl.envs.portfolio_env import PortfolioEnv  # type: ignore
        except Exception:
            raise HTTPException(status_code=500, detail="Planning unavailable: RL modules missing")

        syms: List[str] = []
        if symbols:
            syms = [s.strip().upper() for s in symbols.split(',') if s.strip()]
        elif symbol:
            syms = [symbol.strip().upper()]
        else:
            raise HTTPException(status_code=400, detail="symbol or symbols is required")

        def _plan_many():
            if len(syms) >= 2:
                base = PortfolioEnv.load_from_local_universe(symbols=syms, window=max(2, window or 60))
                idx = next(iter(base.df_map.values())).index
                n = int(len(idx))
                if n < 60:
                    raise ValueError("Not enough aligned rows across symbols to plan PPO")
                try:
                    y2020 = pd.Timestamp('2020-01-01')
                    first_idx = idx[0]
                    train_start = y2020 if (y2020 > first_idx and y2020 < idx[-1]) else first_idx
                except Exception:
                    train_start = idx[0]
                train_end = idx[-1]
                days = int((pd.to_datetime(train_end) - pd.to_datetime(train_start)).days)
                timesteps = int(max(300_000, min(1_200_000, days * 400)))
                w = int(max(30, min(120, n // 6))) if not window else int(window)
                return {
                    'train_start_date': str(getattr(train_start, 'date', lambda: train_start)()),
                    'train_end_date': str(getattr(train_end, 'date', lambda: train_end)()),
                    'window': w,
                    'training_days': days,
                    'timesteps': timesteps,
                    'total_rows': n,
                    'symbols': syms,
                }
            else:
                env = RLMarketEnv.load_from_local(syms[0], window=max(2, window or 60), tail_days=None)
                df = env.df
                n = int(len(df))
                if n < 60:
                    raise ValueError(f"Not enough rows to plan PPO for {syms[0]}")
                idx = df.index
                if not isinstance(idx, pd.DatetimeIndex):
                    try:
                        idx = pd.to_datetime(idx)
                    except Exception:
                        pass
                try:
                    y2020 = pd.Timestamp('2020-01-01')
                    first_idx = idx[0]
                    train_start = y2020 if (y2020 > first_idx and y2020 < idx[-1]) else first_idx
                except Exception:
                    train_start = idx[0]
                train_end = idx[-1]
                days = int((pd.to_datetime(train_end) - pd.to_datetime(train_start)).days)
                timesteps = int(max(300_000, min(1_200_000, days * 400)))
                w = int(max(30, min(120, n // 6))) if not window else int(window)
                return {
                    'train_start_date': str(getattr(train_start, 'date', lambda: train_start)()),
                    'train_end_date': str(getattr(train_end, 'date', lambda: train_end)()),
                    'window': w,
                    'training_days': days,
                    'timesteps': timesteps,
                    'total_rows': n,
                    'symbols': syms,
                }

        import asyncio
        plan = await asyncio.to_thread(_plan_many)
        return {"status": "planned", "plan": plan}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
