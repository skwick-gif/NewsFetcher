from fastapi import APIRouter, HTTPException, Request, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import uuid
import time
import threading
import subprocess
import sys

from app.core.config import (
    PROGRESSIVE_ML_AVAILABLE,
    data_manager,
    progressive_data_loader,
    progressive_trainer,
    progressive_predictor
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Pydantic model for backtest request
class BacktestRequest(BaseModel):
    symbol: str
    train_start_date: str
    train_end_date: str
    test_period_days: int = 14
    max_iterations: int = 10
    target_accuracy: float = 0.85
    auto_stop: bool = True
    # When true, the server will automatically clamp max_iterations to what the data can actually support
    # based on the number of available trading days after train_end_date. If false and the request exceeds
    # feasible iterations, the server will return a 400 with a clear message instead of starting the job.
    auto_adjust_iterations: bool = True
    # Fully automatic planning: choose dates and caps to maximize deep testing coverage
    auto_plan: bool = True
    # Prefer deeper test windows by default; can be combined with desired_* hints
    deep_mode: bool = True
    # Optional hints for the planner; if None, sensible defaults are used
    desired_iterations: int | None = None
    desired_test_period_days: int | None = None
    training_window_days: int | None = None
    # Ensure data is refreshed before planning (download/compute the 4 files if stale/missing)
    ensure_fresh_data: bool = True
    model_types: List[str] = ["lstm"]  # Add model selection support
    indicator_params: Dict[str, Any] | None = None  # Optional technical indicator parameters
    # Auto-scout: fast candidate search to stabilize regime before deep run
    auto_scout: bool = True
    scout_candidate_windows: List[int] | None = None  # e.g., [360, 540, 720]
    scout_candidate_seq: List[int] | None = None      # e.g., [60, 90]
    scout_indicator_profiles: List[str] | None = None # e.g., ['short_mid','mid_long']
    scout_forward_days: int = 14
    scout_min_predictions: int = 8
    scout_epochs: int = 10
    scout_model_types: List[str] = ["cnn"]

backtest_jobs = {}

@router.post("/ml/progressive/backtest", tags=["Progressive ML"])
async def start_backtest(request: BacktestRequest, raw_request: Request = None, background_tasks: BackgroundTasks = None):
    """
    Start progressive backtesting with date-range training

    Args:
        symbol: Stock symbol to train on
        train_start_date: Initial training start date (YYYY-MM-DD)
        train_end_date: Initial training end date (YYYY-MM-DD)
        test_period_days: Number of days to test forward (default: 14)
        max_iterations: Maximum iterations to run (default: 10)
        target_accuracy: Target accuracy to achieve 0-1 (default: 0.85)
        auto_stop: Stop when target accuracy reached (default: True)
        model_types: List of model types to train (default: ["lstm"])

    Returns:
        Backtest results with all iterations
    """
    try:
        logger.info(f"🔬 Received backtest request")

        # Debug request data
        try:
            logger.info(f"📊 Request object: {request}")
            logger.info(f"📊 Symbol: {request.symbol}")
            logger.info(f"📊 Model types: {request.model_types}")
            logger.info(f"📊 Dates: {request.train_start_date} to {request.train_end_date}")
            if request.indicator_params:
                logger.info(f"🧩 Indicator params override: {request.indicator_params}")
        except Exception as debug_e:
            logger.error(f"❌ Error accessing request data: {debug_e}")

        # Try to get raw body for debugging
        if raw_request:
            try:
                body = await raw_request.body()
                logger.info(f"📊 Raw body: {body.decode('utf-8')}")
            except Exception as body_e:
                logger.error(f"❌ Error reading raw body: {body_e}")

        if not PROGRESSIVE_ML_AVAILABLE:
            raise HTTPException(status_code=503, detail="Progressive ML system not available")

        if not progressive_data_loader or not progressive_trainer or not progressive_predictor:
            raise HTTPException(status_code=503, detail="Progressive ML components not initialized")

        # Optionally ensure latest data before planning
        try:
            if request.ensure_fresh_data:
                ensure_summary = await asyncio.to_thread(data_manager.ensure_symbol_data, request.symbol)
                logger.info(f"🧩 Data ensure summary for {request.symbol}: {getattr(ensure_summary, 'to_dict', lambda: ensure_summary)() if hasattr(ensure_summary, 'to_dict') else ensure_summary}")
        except Exception as _e:
            logger.warning(f"Data ensure before preflight failed or partial: {_e}")

        # ---- Planning & Preflight: compute dates and feasible iterations from data coverage ----
        from pathlib import Path
        import pandas as pd
        try:
            # Load full price CSV (unfiltered) to count trading days after end date
            price_csv = data_manager.stock_data_dir / request.symbol / f"{request.symbol}_price.csv"
            if not price_csv.exists():
                # Try ensure and re-check
                await asyncio.to_thread(data_manager.ensure_symbol_data, request.symbol)
            if not price_csv.exists():
                raise HTTPException(status_code=404, detail=f"Price data not found for {request.symbol}")

            df = pd.read_csv(price_csv, index_col=0)
            df.index = pd.to_datetime(df.index, format='mixed', errors='coerce', utc=True).tz_localize(None)
            df = df.sort_index()
            if len(df.index) == 0:
                raise HTTPException(status_code=422, detail=f"Empty price data for {request.symbol}")

            last_data_date = df.index.max()

            # Auto-planning: choose dates and parameters to favor deep testing
            plan = {}
            tpd = int(request.desired_test_period_days or request.test_period_days or 14)
            req_iters = int(request.desired_iterations or request.max_iterations or 10)

            # Determine train_end_date automatically if requested
            if request.auto_plan:
                # Compute the index position for an end date that leaves space for req_iters*tpd trading days
                needed = req_iters * tpd
                if needed >= len(df.index):
                    # If needed exceeds data, cap to maximum possible
                    feasible_iters_from_data = max(0, (len(df.index) - 1) // tpd)
                    needed = feasible_iters_from_data * tpd
                end_pos = max(0, len(df.index) - needed - 1)
                end_dt = df.index[end_pos]
                # Optionally cap training window length
                if request.training_window_days and request.training_window_days > 0:
                    start_dt = max(df.index[0], end_dt - pd.Timedelta(days=int(request.training_window_days)))
                else:
                    # Ignore placeholder/too-early dates; clamp to first available trading day
                    candidate_start = None
                    if request.train_start_date:
                        try:
                            candidate_start = pd.to_datetime(request.train_start_date, utc=True, errors='coerce').tz_localize(None)
                        except Exception:
                            candidate_start = None
                    # If missing/invalid or before first data, use first data date
                    if candidate_start is None or pd.isna(candidate_start) or candidate_start < df.index[0]:
                        start_dt = df.index[0]
                    else:
                        start_dt = candidate_start
                # Ensure start <= end
                if start_dt > end_dt:
                    start_dt = df.index[0]
                planned_train_start = start_dt.date().isoformat()
                planned_train_end = end_dt.date().isoformat()
                # Update request with planned values
                request = request.copy(update={
                    "train_start_date": planned_train_start,
                    "train_end_date": planned_train_end,
                    "test_period_days": tpd,
                    "max_iterations": req_iters
                })
                plan.update({
                    "train_start_date": planned_train_start,
                    "train_end_date": planned_train_end,
                    "test_period_days": tpd,
                    "requested_iterations": req_iters
                })
                end_dt_use = end_dt
            else:
                # Use provided end date
                try:
                    end_dt_use = pd.to_datetime(request.train_end_date, utc=True).tz_localize(None)
                except Exception:
                    raise HTTPException(status_code=400, detail=f"Invalid train_end_date: {request.train_end_date}")

            # Compute coverage after chosen end date using trading days (rows)
            trading_days_after_end = int((df.index > end_dt_use).sum())
            feasible = int(trading_days_after_end // tpd)

            preflight = {
                "requested_max_iterations": int(request.max_iterations),
                "feasible_max_iterations": feasible,
                "trading_days_after_end": trading_days_after_end,
                "last_data_date": last_data_date.date().isoformat(),
                "test_period_days": tpd,
                "plan": plan
            }

            # If nothing to test, block immediately
            if feasible <= 0:
                msg = (f"No test window fits after train_end_date. Last data date is {preflight['last_data_date']}; "
                       f"needs at least {tpd} trading days, found {trading_days_after_end}.")
                raise HTTPException(status_code=400, detail={"message": msg, "preflight": preflight})

            adjusted_request = request
            adjusted = False
            if feasible < int(request.max_iterations):
                if request.auto_adjust_iterations:
                    adjusted_request = request.copy(update={"max_iterations": feasible})
                    adjusted = True
                else:
                    msg = (f"Requested max_iterations={request.max_iterations} exceeds feasible={feasible} given "
                           f"{trading_days_after_end} trading days after end and test_period_days={tpd}.")
                    raise HTTPException(status_code=400, detail={"message": msg, "preflight": preflight})
        except HTTPException:
            raise
        except Exception as pf_e:
            logger.error(f"Preflight check failed: {pf_e}")
            raise HTTPException(status_code=500, detail=f"Preflight failed: {pf_e}")

        # Start async backtest job with progress tracking
        import uuid
        job_id = f"backtest_{request.symbol}_{uuid.uuid4().hex[:8]}"
        backtest_jobs[job_id] = {
            "job_id": job_id,
            "symbol": request.symbol,
            "status": "starting",
            "progress": 0,
            "current_step": "Initializing...",
            "eta_seconds": None,
            "start_time": datetime.now(timezone.utc).isoformat(),
            "end_time": None,
            "result": None,
            "error": None,
            "cancelled": False
        }
        # Attach preflight info to job record for clients to read immediately
        try:
            backtest_jobs[job_id]["preflight"] = preflight
            backtest_jobs[job_id]["adjusted"] = adjusted
            if adjusted:
                backtest_jobs[job_id]["note"] = (
                    f"max_iterations clamped from {requested} to {feasible} based on available data"
                )
        except Exception:
            pass

        # Launch background job
        if background_tasks is None:
            from fastapi import BackgroundTasks as _BT
            background_tasks = _BT()
        background_tasks.add_task(
            run_backtest_job,
            job_id=job_id,
            request=adjusted_request
        )

        return {
            "status": "backtest_started",
            "job_id": job_id,
            "symbol": request.symbol,
            "preflight": preflight,
            "adjusted": adjusted,
            "plan": preflight.get("plan"),
            "message": "Backtest started in background. Use job_id to track progress.",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    except Exception as e:
        logger.error(f"Failed to start backtest: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start backtest: {str(e)}")

@router.get("/ml/progressive/backtest/status/{job_id}", tags=["Progressive ML"])
async def get_backtest_status(job_id: str):
    """Get status of running backtest"""
    try:
        if job_id in backtest_jobs:
            job = backtest_jobs[job_id].copy()
            job["timestamp"] = datetime.now(timezone.utc).isoformat()
            return job
        raise HTTPException(status_code=404, detail=f"Backtest job {job_id} not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get backtest status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get backtest status: {str(e)}")

@router.post("/ml/progressive/backtest/cancel/{job_id}", tags=["Progressive ML"])
async def cancel_backtest(job_id: str):
    """Request cancellation of a running backtest job"""
    try:
        if job_id not in backtest_jobs:
            raise HTTPException(status_code=404, detail=f"Backtest job {job_id} not found")
        # Mark job cancelled; worker will observe and stop gracefully
        backtest_jobs[job_id]["cancelled"] = True
        # Update status text if still running
        if backtest_jobs[job_id]["status"] in ["starting", "running"]:
            backtest_jobs[job_id]["status"] = "cancelling"
            backtest_jobs[job_id]["current_step"] = "Cancelling..."
        return {
            "status": "cancellation_requested",
            "job_id": job_id,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel backtest: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to cancel backtest: {str(e)}")

async def run_backtest_job(job_id: str, request: BacktestRequest):
    """Background task for running backtest with progress updates"""
    import time
    try:
        backtest_jobs[job_id]["status"] = "running"
        backtest_jobs[job_id]["progress"] = 5
        backtest_jobs[job_id]["current_step"] = "Ensuring ticker data..."
        start_time = time.time()

        # Progress callback from backtester
        def _progress_cb(ev: Dict[str, Any]):
            try:
                prog = int(ev.get('progress', backtest_jobs[job_id].get('progress', 0)))
                backtest_jobs[job_id].update({
                    "status": ev.get('status', backtest_jobs[job_id]["status"]),
                    "progress": max(min(prog, 99), 0),
                    "current_step": ev.get('current_step', backtest_jobs[job_id]["current_step"]),
                    "eta_seconds": ev.get('eta_seconds')
                })
            except Exception:
                pass

        # Ensure data before backtesting
        try:
            ensure = await asyncio.to_thread(data_manager.ensure_symbol_data, request.symbol)
            backtest_jobs[job_id]["current_step"] = "Initializing backtester..."
            backtest_jobs[job_id]["progress"] = 10
        except Exception as e:
            backtest_jobs[job_id].update({
                "status": "failed",
                "current_step": f"❌ Ensure data failed: {e}",
                "error": str(e),
            })
            return

        # Helper: indicator profiles for quick scouting
        def _indicator_profile(name: str) -> Dict[str, Any]:
            nm = (name or '').strip().lower()
            if nm in ("short_mid", "short-mid", "shortmid"):
                return {
                    'rsi_period': 12,
                    'macd_fast': 8,
                    'macd_slow': 21,
                    'macd_signal': 5,
                    'sma_periods': '5,10,20,50',
                    'ema_periods': '5,10,20,50',
                    'bb_period': 20,
                    'bb_std': 2.0
                }
            if nm in ("mid_long", "mid-long", "midlong", "mid_long_term"):
                return {
                    'rsi_period': 18,
                    'macd_fast': 19,
                    'macd_slow': 39,
                    'macd_signal': 9,
                    'sma_periods': '20,50,100,200',
                    'ema_periods': '20,50,100',
                    'bb_period': 30,
                    'bb_std': 2.5
                }
            # fallback to request.indicator_params or sane defaults
            return request.indicator_params or {
                'rsi_period': 14,
                'macd_fast': 12,
                'macd_slow': 26,
                'macd_signal': 9,
                'sma_periods': '5,10,20,50',
                'ema_periods': '5,10,20,50',
                'bb_period': 20,
                'bb_std': 2.0
            }

        # Optional: Auto-Scout phase (quick candidates) to pick stable regime
        chosen_cfg = None
        scout_report = []
        try:
            if bool(getattr(request, 'auto_scout', True)):
                backtest_jobs[job_id]["current_step"] = "Scouting best config..."
                backtest_jobs[job_id]["progress"] = 15
                import pandas as pd
                # Build candidates
                end_dt = pd.to_datetime(request.train_end_date)
                cand_windows = request.scout_candidate_windows or [360, 540, 720]
                cand_seq = request.scout_candidate_seq or [60, 90]
                cand_profiles = request.scout_indicator_profiles or ['short_mid', 'mid_long']
                # Fine-scout: curated indicator sets (EMA/SMA/MACD/RSI/BB) presets
                fine_sets = [
                    {
                        'name': 'short_momentum',
                        'params': {'rsi_period': 12, 'macd_fast': 8, 'macd_slow': 21, 'macd_signal': 5,
                                   'sma_periods': '5,10,20,50', 'ema_periods': '5,10,20,50', 'bb_period': 20, 'bb_std': 2.0}
                    },
                    {
                        'name': 'balanced',
                        'params': {'rsi_period': 14, 'macd_fast': 12, 'macd_slow': 26, 'macd_signal': 9,
                                   'sma_periods': '10,20,50,100', 'ema_periods': '10,20,50', 'bb_period': 20, 'bb_std': 2.0}
                    },
                    {
                        'name': 'long_trend',
                        'params': {'rsi_period': 18, 'macd_fast': 19, 'macd_slow': 39, 'macd_signal': 9,
                                   'sma_periods': '20,50,100,200', 'ema_periods': '20,50,100', 'bb_period': 30, 'bb_std': 2.5}
                    },
                    {
                        'name': 'volatility_band',
                        'params': {'rsi_period': 14, 'macd_fast': 12, 'macd_slow': 26, 'macd_signal': 9,
                                   'sma_periods': '5,20,50,100', 'ema_periods': '5,20,50', 'bb_period': 10, 'bb_std': 3.0}
                    }
                ]
                candidates = []
                for w in cand_windows:
                    for s in cand_seq:
                        # Profiles
                        for prof in cand_profiles:
                            candidates.append({
                                'window_days': int(w),
                                'sequence_length': int(s),
                                'indicator_params': _indicator_profile(prof),
                                'profile': prof
                            })
                        # Fine-scout presets
                        for preset in fine_sets:
                            candidates.append({
                                'window_days': int(w),
                                'sequence_length': int(s),
                                'indicator_params': preset['params'],
                                'profile': preset['name']
                            })
                # Limit to a reasonable number
                candidates = candidates[:16]

                # Evaluate each candidate quickly
                best = None
                for idx, cand in enumerate(candidates, start=1):
                    try:
                        start_dt = (end_dt - pd.Timedelta(days=cand['window_days'])).date().isoformat()
                        # Loader for candidate
                        c_loader = ProgressiveDataLoader(
                            stock_data_dir=progressive_data_loader.stock_data_dir,
                            sequence_length=cand['sequence_length'],
                            horizons=progressive_data_loader.horizons,
                            use_fundamentals=progressive_data_loader.use_fundamentals,
                            use_technical_indicators=progressive_data_loader.use_technical_indicators,
                            indicator_params=cand['indicator_params'],
                            train_start_date=start_dt,
                            train_end_date=request.train_end_date
                        )
                        # Trainer (light)
                        from app.ml.progressive.trainer import ProgressiveTrainer
                        from pathlib import Path as _Path
                        cand_dir = _Path("app/ml/models/backtests") / f"{job_id}" / f"scout_{idx:02d}"
                        cand_dir.mkdir(parents=True, exist_ok=True)
                        c_trainer = ProgressiveTrainer(
                            data_loader=c_loader,
                            training_config={
                                'epochs': int(request.scout_epochs or 10),
                                'batch_size': 64,
                                'validation_split': 0.2,
                                'early_stopping_patience': 4,
                                'reduce_lr_patience': 3,
                                'reduce_lr_factor': 0.5
                            },
                            save_dir=str(cand_dir)
                        )
                        from app.ml.progressive.predictor import ProgressivePredictor
                        c_predictor = ProgressivePredictor(data_loader=c_loader, model_dir=str(cand_dir))
                        # Backtester harness for evaluation
                        c_bt = ProgressiveBacktester(data_loader=c_loader, trainer=c_trainer, predictor=c_predictor)
                        # Train quick model(s)
                        scout_models = list(set(request.scout_model_types or ['cnn']))
                        c_train = c_trainer.train_progressive_models(symbol=request.symbol, model_types=scout_models)
                        # Evaluate forward short window
                        test_start = (pd.to_datetime(request.train_end_date) + pd.Timedelta(days=1)).date().isoformat()
                        # Compute test_end by forward_days but clamp to last data
                        full_df_tmp = c_loader.load_stock_data(request.symbol)
                        test_end = (pd.to_datetime(test_start) + pd.Timedelta(days=int(request.scout_forward_days or 14))).date().isoformat()
                        if full_df_tmp is not None and len(full_df_tmp.index) > 0:
                            ld = full_df_tmp.index.max().date().isoformat()
                            if pd.to_datetime(test_end) > pd.to_datetime(ld):
                                test_end = ld
                        c_eval = c_bt.evaluate_iteration(
                            symbol=request.symbol,
                            test_start_date=test_start,
                            test_end_date=test_end,
                            iteration_num=1,
                            full_df=full_df_tmp if full_df_tmp is not None else c_loader.load_stock_data(request.symbol)
                        )
                        score = float(c_eval.get('direction_accuracy') or c_eval.get('accuracy') or 0.0)
                        n = int(c_eval.get('predictions_made') or c_eval.get('test_samples') or 0)
                        mae = c_eval.get('mae'); rmse = c_eval.get('rmse'); mape = c_eval.get('mape')
                        rec = {
                            'idx': idx,
                            'window_days': cand['window_days'],
                            'sequence_length': cand['sequence_length'],
                            'profile': cand['profile'],
                            'accuracy': score,
                            'predictions': n,
                            'mae': mae, 'rmse': rmse, 'mape': mape,
                            'dir': str(cand_dir),
                            'test_start': test_start,
                            'test_end': test_end
                        }
                        scout_report.append(rec)
                        # Select best meeting min predictions
                        if n >= int(request.scout_min_predictions or 8):
                            if best is None or score > best['accuracy'] or (abs(score - best['accuracy']) < 1e-9 and (mape or 1e9) < (best.get('mape') or 1e9)):
                                best = rec
                        # Progress hint
                        backtest_jobs[job_id]["current_step"] = f"Scouting {idx}/{len(candidates)}..."
                    except Exception as _sc_e:
                        logger.warning(f"Scout candidate {idx} failed: {_sc_e}")
                        continue
                # If none meet min predictions, pick max accuracy anyway
                if best is None and scout_report:
                    best = max(scout_report, key=lambda r: r.get('accuracy', 0.0))
                if best:
                    chosen_cfg = best
                    # Update request train_start_date based on window
                    new_start = (end_dt - pd.Timedelta(days=int(best['window_days']))).date().isoformat()
                    request = request.copy(update={
                        'train_start_date': new_start
                    })
                    # Create data loader for deep run using chosen sequence and indicators
                    dl_for_job = ProgressiveDataLoader(
                        stock_data_dir=progressive_data_loader.stock_data_dir,
                        sequence_length=int(best['sequence_length']),
                        horizons=progressive_data_loader.horizons,
                        use_fundamentals=progressive_data_loader.use_fundamentals,
                        use_technical_indicators=progressive_data_loader.use_technical_indicators,
                        indicator_params=_indicator_profile(best['profile'])
                    )
                    backtest_jobs[job_id]['scout'] = {
                        'candidates': scout_report,
                        'chosen': best
                    }
                    # Prefer full ensemble for deep run if user didn't request multi-model
                    if not request.model_types or len(request.model_types) <= 1:
                        request = request.copy(update={'model_types': ['cnn','lstm','transformer']})
                else:
                    backtest_jobs[job_id]['scout'] = {
                        'candidates': scout_report,
                        'chosen': None,
                        'note': 'No viable candidate met minimum predictions; proceeding with original plan'
                    }
        except Exception as _scout_e:
            logger.warning(f"Auto-scout skipped due to error: {_scout_e}")

        backtester = ProgressiveBacktester(
            data_loader=dl_for_job,
            trainer=progressive_trainer,
            predictor=progressive_predictor,
            progress_callback=_progress_cb,
            cancel_checker=lambda: bool(backtest_jobs.get(job_id, {}).get("cancelled", False))
        )

        results = backtester.run_backtest(
            symbol=request.symbol,
            train_start_date=request.train_start_date,
            train_end_date=request.train_end_date,
            test_period_days=request.test_period_days,
            max_iterations=request.max_iterations,
            target_accuracy=request.target_accuracy,
            auto_stop=request.auto_stop,
            model_types=request.model_types
        )

        elapsed = int(time.time() - start_time)
        # Handle cancelled/failed/completed from results
        status_result = results.get('status') if isinstance(results, dict) else None
        if status_result == 'cancelled':
            backtest_jobs[job_id].update({
                "status": "cancelled",
                "progress": backtest_jobs[job_id].get("progress", 0),
                "current_step": "⏹ Backtest cancelled",
                "eta_seconds": 0,
                "end_time": datetime.now(timezone.utc).isoformat(),
                "result": results,
            })
        else:
            # Auto-promote champion: snapshot the best iteration's checkpoints into champions/<symbol>/<job_id>
            try:
                from pathlib import Path
                import shutil
                best_iter = int(results.get("best_iteration")) if isinstance(results, dict) else None
                job_model_dir = getattr(backtester, 'job_model_dir', None)
                champion_info = None
                if best_iter and job_model_dir and Path(job_model_dir).exists():
                    iter_dir = Path(job_model_dir) / f"iter_{best_iter:02d}"
                    if iter_dir.exists():
                        champions_root = Path("app/ml/models/champions") / request.symbol
                        target_dir = champions_root / f"{results.get('job_id', job_id)}"
                        target_dir.mkdir(parents=True, exist_ok=True)
                        # Copy checkpoints
                        for p in iter_dir.glob("*.pth"):
                            shutil.copy2(p, target_dir / p.name)
                        # Save metadata
                        meta = {
                            "symbol": request.symbol,
                            "job_id": results.get('job_id', job_id),
                            "best_iteration": best_iter,
                            "summary": {
                                k: results.get(k) for k in ["best_accuracy", "best_loss", "total_iterations", "total_time"]
                            },
                            "train_end_date": request.train_end_date,
                            "test_period_days": request.test_period_days,
                            "model_types": request.model_types,
                        }
                        with open(target_dir / "champion_meta.json", "w", encoding="utf-8") as fh:
                            json.dump(meta, fh, indent=2)
                        champion_info = {"dir": str(target_dir), "meta": meta}
                if champion_info:
                    backtest_jobs[job_id]["champion"] = champion_info
            except Exception as snap_e:
                logger.warning(f"Champion snapshot failed: {snap_e}")

            # Optionally run a forward test automatically using the champion (if available)
            try:
                if champion_info and PROGRESSIVE_ML_AVAILABLE:
                    from app.ml.progressive.backtester import ProgressiveBacktester as _BT
                    import pandas as _pd
                    # Build an evaluator and point it at the champion directory
                    _bt = _BT(
                        data_loader=progressive_data_loader,
                        trainer=progressive_trainer,
                        predictor=progressive_predictor,
                        progress_callback=None,
                        cancel_checker=None
                    )
                    from pathlib import Path as _Path
                    _bt.job_model_dir = _Path(champion_info['dir'])
                    # Load full data and compute forward window (train_end+1 .. last available)
                    _full_loader = ProgressiveDataLoader(
                        stock_data_dir=progressive_data_loader.stock_data_dir,
                        sequence_length=progressive_data_loader.sequence_length,
                        horizons=progressive_data_loader.horizons,
                        use_fundamentals=progressive_data_loader.use_fundamentals,
                        use_technical_indicators=progressive_data_loader.use_technical_indicators,
                        indicator_params=getattr(progressive_data_loader, 'indicator_params', None)
                    )
                    _full_df = _full_loader.load_stock_data(request.symbol)
                    _forward_summary = None
                    if _full_df is not None and len(_full_df.index) > 0:
                        _last_date = _full_df.index.max().date().isoformat()
                        _train_end = champion_info['meta'].get('train_end_date')
                        _start_date = (_pd.to_datetime(_train_end) + _pd.Timedelta(days=1)).date().isoformat() if _train_end else None
                        if _start_date and _pd.to_datetime(_start_date) <= _pd.to_datetime(_last_date):
                            _eval = _bt.evaluate_iteration(
                                symbol=request.symbol,
                                test_start_date=_start_date,
                                test_end_date=_last_date,
                                iteration_num=int(champion_info['meta'].get('best_iteration') or 0),
                                full_df=_full_df
                            )
                            _forward_summary = {
                                'forward_start': _start_date,
                                'forward_end': _last_date,
                                'metrics': _eval
                            }
                    if _forward_summary:
                        backtest_jobs[job_id]['forward'] = _forward_summary
            except Exception as _fe:
                logger.warning(f"Auto forward test skipped: {_fe}")

            # Also attach current predictions from the champion for 1/7/30d
            try:
                if champion_info and PROGRESSIVE_ML_AVAILABLE:
                    from app.ml.progressive.predictor import ProgressivePredictor as _Pred
                    from app.ml.progressive.data_loader import ProgressiveDataLoader as _DL
                    from pathlib import Path as _Path
                    import torch as _torch
                    import pandas as _pdr
                    # Derive indicator params and sequence length from chosen scout config if available
                    _ind_params = getattr(progressive_data_loader, 'indicator_params', None)
                    _seq_len = getattr(progressive_data_loader, 'sequence_length', 60)
                    try:
                        _sc = backtest_jobs.get(job_id, {}).get('scout', {})
                        _chosen = _sc.get('chosen') if isinstance(_sc, dict) else None
                        if _chosen and isinstance(_chosen, dict):
                            _seq_len = int(_chosen.get('sequence_length') or _seq_len)
                            _prof = _chosen.get('profile')
                            if _prof:
                                _ind_params = _indicator_profile(str(_prof))
                    except Exception:
                        pass
                    # If not available from scout, try to read sequence_length from a checkpoint
                    try:
                        ckpts = list(_Path(champion_info['dir']).glob("*.pth"))
                        if ckpts:
                            _ck = _torch.load(ckpts[0], map_location='cpu')
                            _sl = _ck.get('sequence_length')
                            if isinstance(_sl, int) and _sl > 0:
                                _seq_len = _sl
                    except Exception as _ck_e:
                        logger.debug(f"Could not read sequence_length from checkpoint: {_ck_e}")

                    _pred_loader = _DL(
                        stock_data_dir=progressive_data_loader.stock_data_dir,
                        sequence_length=int(_seq_len),
                        horizons=progressive_data_loader.horizons,
                        use_fundamentals=progressive_data_loader.use_fundamentals,
                        use_technical_indicators=progressive_data_loader.use_technical_indicators,
                        indicator_params=_ind_params
                    )
                    _pred = _Pred(data_loader=_pred_loader, model_dir=str(champion_info['dir']))
                    try:
                        _preds = _pred.predict_ensemble(symbol=request.symbol, mode='progressive')
                        # Enrich with SL/TP risk block per horizon (ATR-based if indicators exist else volatility)
                        try:
                            from pathlib import Path as __Path
                            import pandas as __pd
                            ind_path = data_manager.stock_data_dir / request.symbol / f"{request.symbol}_indicators.csv"
                            price_path = data_manager.stock_data_dir / request.symbol / f"{request.symbol}_price.csv"
                            close_price = float(_preds.get('current_price', 0.0))
                            atr_pct = None
                            if ind_path.exists():
                                ind_df = __pd.read_csv(ind_path, index_col=0)
                                if 'ATR_14' in ind_df.columns and close_price > 0:
                                    atr_val = float(__pd.to_numeric(ind_df['ATR_14'], errors='coerce').dropna().iloc[-1])
                                    atr_pct = max(0.001, min(0.2, atr_val / close_price))
                            if atr_pct is None and price_path.exists():
                                dfp = __pd.read_csv(price_path, index_col=0)
                                dfp['Close'] = __pd.to_numeric(dfp['Close'], errors='coerce')
                                dfp = dfp.dropna(subset=['Close'])
                                rets = dfp['Close'].pct_change().dropna()
                                vol = float(rets.rolling(14).std().dropna().iloc[-1]) if len(rets) > 14 else float(rets.std())
                                atr_pct = max(0.001, min(0.2, vol * 1.5))
                            rr = 2.0
                            risk_pct = max(0.005, min(0.2, atr_pct or 0.01))
                            reward_pct = max(0.01, min(0.4, risk_pct * rr))
                            for hk, obj in (_preds.get('predictions') or {}).items():
                                change_pct = float(obj.get('price_change_pct', 0.0))
                                sl = close_price * (1 - risk_pct)
                                tp = close_price * (1 + reward_pct)
                                if change_pct < 0:
                                    sl = close_price * (1 + risk_pct)
                                    tp = close_price * (1 - reward_pct)
                                obj['risk'] = {
                                    'stop_loss': round(sl, 4),
                                    'take_profit': round(tp, 4),
                                    'stop_loss_pct': -risk_pct if change_pct >= 0 else risk_pct,
                                    'take_profit_pct': reward_pct if change_pct >= 0 else -reward_pct,
                                    'basis': 'ATR_14' if atr_pct is not None else 'volatility',
                                    'rr': rr
                                }
                        except Exception as __e:
                            logger.debug(f"Risk enrich (current preds) skipped: {__e}")
                        # Keep a compact summary for UI
                        _compact = {
                            'symbol': _preds.get('symbol'),
                            'current_price': _preds.get('current_price'),
                            'generated_at': _preds.get('generated_at'),
                            'predictions': {}
                        }
                        for hk in ['1d', '7d', '30d']:
                            if hk in _preds.get('predictions', {}):
                                p = _preds['predictions'][hk]
                                # Mark if horizon hit safety cap (used by UI tooltip)
                                try:
                                    cap_map = {'1d': 0.10, '7d': 0.20, '30d': 0.40}
                                    pc = float(p.get('price_change_pct', 0.0))
                                    capped_flag = abs(pc) >= (cap_map.get(hk, 1.0) - 1e-6)
                                except Exception:
                                    capped_flag = False
                                _compact['predictions'][hk] = {
                                    'target_price': p.get('target_price'),
                                    'price_change_pct': p.get('price_change_pct'),
                                    'direction': p.get('direction'),
                                    'direction_prob': p.get('direction_prob'),
                                    'confidence': p.get('confidence'),
                                    'signal': p.get('signal'),
                                    'risk': p.get('risk'),
                                    'capped': capped_flag
                                }
                        backtest_jobs[job_id]['current_predictions'] = _compact
                    except Exception as _pe:
                        logger.warning(f"Champion current predictions failed: {_pe}")
            except Exception as _cp_e:
                logger.debug(f"Attach current predictions skipped: {_cp_e}")

            backtest_jobs[job_id].update({
                "status": "completed",
                "progress": 100,
                "current_step": "✅ Backtest completed successfully!",
                "eta_seconds": 0,
                "end_time": datetime.now(timezone.utc).isoformat(),
                "result": results,
            })
        logger.info(f"✅ Backtest job {job_id} completed for {request.symbol}")
    except Exception as e:
        logger.error(f"❌ Backtest job {job_id} failed: {e}")
        backtest_jobs[job_id].update({
            "status": "failed",
            "progress": 0,
            "current_step": f"❌ Error: {str(e)}",
            "end_time": datetime.now(timezone.utc).isoformat(),
            "error": str(e),
            "eta_seconds": 0
        })

@router.get("/data/ensure/{symbol}", tags=["Data"])
async def api_ensure_symbol_data(symbol: str):
    """Ensure the four required data files for a symbol exist and are up-to-date.

    Returns a summary of created/updated/skipped and any errors.
    """
    try:
        if not PROGRESSIVE_ML_AVAILABLE:
            raise HTTPException(status_code=503, detail="System not initialized")
        summary = await asyncio.to_thread(data_manager.ensure_symbol_data, symbol)
        return {"status": "success", "symbol": symbol, "summary": summary.to_dict()}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ensure data failed for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/ml/progressive/backtest/results/{symbol}", tags=["Progressive ML"])
async def get_backtest_results(symbol: str):
    """
    Get backtest results for a symbol

    Args:
        symbol: Stock symbol

    Returns:
        Saved backtest results
    """
    try:
        if not PROGRESSIVE_ML_AVAILABLE:
            raise HTTPException(status_code=503, detail="Progressive ML system not available")

        import os
        import json
        from pathlib import Path

        results_dir = Path("app/ml/models/backtest_results")

        if not results_dir.exists():
            return {
                "status": "no_results",
                "message": f"No backtest results found for {symbol}",
                "symbol": symbol
            }

        # Find latest results file for symbol
        results_files = list(results_dir.glob(f"results_{symbol}_*.json"))

        if not results_files:
            return {
                "status": "no_results",
                "message": f"No backtest results found for {symbol}",
                "symbol": symbol
            }

        # Get most recent file
        latest_file = max(results_files, key=lambda p: p.stat().st_mtime)

        with open(latest_file, 'r') as f:
            results = json.load(f)

        # Normalize legacy format to match frontend expectations
        try:
            if isinstance(results, dict):
                if 'all_iterations' not in results and 'iterations' in results:
                    results['all_iterations'] = results.get('iterations', [])
                if 'best_iteration' not in results and results.get('all_iterations'):
                    # Compute best iteration by max accuracy if available
                    best = None
                    for it in results['all_iterations']:
                        acc = it.get('accuracy')
                        if isinstance(acc, (int, float)):
                            if best is None or acc > best.get('accuracy', -1):
                                best = it
                    if best and 'iteration' in best:
                        results['best_iteration'] = best['iteration']
        except Exception as _norm_e:
            logger.debug(f"Backtest results normalization skipped: {_norm_e}")

        return {
            "status": "success",
            "symbol": symbol,
            "results": results,
            "file": latest_file.name,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    except Exception as e:
        logger.error(f"Failed to get backtest results: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get backtest results: {str(e)}")

@router.get("/ml/progressive/backtest/history/{symbol}", tags=["Progressive ML"])
async def list_backtest_history(symbol: str, limit: int = Query(20, ge=1, le=100)):
    """
    List recent backtest result files for a symbol with brief metadata
    """
    try:
        from pathlib import Path
        import json
        results_dir = Path("app/ml/models/backtest_results")
        if not results_dir.exists():
            return {"status": "success", "symbol": symbol, "items": []}
        files = sorted(results_dir.glob(f"results_{symbol}_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        items = []
        for f in files[:limit]:
            try:
                with open(f, 'r', encoding='utf-8') as fh:
                    data = json.load(fh)
                # Normalize iteration list
                iterations = data.get('all_iterations') or data.get('iterations') or []
                # Best accuracy
                best_acc = None
                if isinstance(iterations, list) and iterations:
                    for it in iterations:
                        acc = it.get('accuracy')
                        if isinstance(acc, (int, float)):
                            best_acc = acc if best_acc is None else max(best_acc, acc)
                items.append({
                    "file": f.name,
                    "modified": datetime.fromtimestamp(f.stat().st_mtime, tz=timezone.utc).isoformat(),
                    "best_accuracy": best_acc,
                    "iterations": len(iterations)
                })
            except Exception:
                items.append({
                    "file": f.name,
                    "modified": datetime.fromtimestamp(f.stat().st_mtime, tz=timezone.utc).isoformat(),
                    "best_accuracy": None,
                    "iterations": None
                })
        return {"status": "success", "symbol": symbol, "items": items}
    except Exception as e:
        logger.error(f"Failed to list backtest history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list backtest history: {str(e)}")

@router.get("/ml/progressive/backtest/result_by_file/{symbol}/{file_name}", tags=["Progressive ML"])
async def get_backtest_result_by_file(symbol: str, file_name: str):
    """
    Fetch a specific backtest results file by name for a symbol
    """
    try:
        from pathlib import Path
        import json
        results_dir = Path("app/ml/models/backtest_results")
        target = results_dir / file_name
        if not target.exists() or f"results_{symbol}_" not in target.name:
            raise HTTPException(status_code=404, detail="Results file not found")
        with open(target, 'r', encoding='utf-8') as f:
            results = json.load(f)
        # Normalize legacy format
        try:
            if isinstance(results, dict):
                if 'all_iterations' not in results and 'iterations' in results:
                    results['all_iterations'] = results.get('iterations', [])
                if 'best_iteration' not in results and results.get('all_iterations'):
                    best = None
                    for it in results['all_iterations']:
                        acc = it.get('accuracy')
                        if isinstance(acc, (int, float)):
                            if best is None or acc > best.get('accuracy', -1):
                                best = it
                    if best and 'iteration' in best:
                        results['best_iteration'] = best['iteration']
        except Exception:
            pass
        return {"status": "success", "symbol": symbol, "results": results, "file": target.name}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get backtest result by file: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get backtest result: {str(e)}")

# Champions utilities
@router.get("/ml/progressive/champions/{symbol}", tags=["Progressive ML"])
async def list_champions(symbol: str):
    try:
        from pathlib import Path
        root = Path("app/ml/models/champions") / symbol
        if not root.exists():
            return {"status": "success", "symbol": symbol, "items": []}
        items = []
        for job_dir in sorted(root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if job_dir.is_dir():
                meta_file = job_dir / "champion_meta.json"
                meta = None
                if meta_file.exists():
                    try:
                        with open(meta_file, 'r', encoding='utf-8') as fh:
                            meta = json.load(fh)
                    except Exception:
                        meta = None
                items.append({
                    'path': str(job_dir),
                    'meta': meta,
                    'modified': datetime.fromtimestamp(job_dir.stat().st_mtime, tz=timezone.utc).isoformat()
                })
        return {"status": "success", "symbol": symbol, "items": items}
    except Exception as e:
        logger.error(f"Failed to list champions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list champions: {e}")

@router.post("/ml/progressive/champion/forward_test/{symbol}", tags=["Progressive ML"])
async def champion_forward_test(symbol: str, job_id: str | None = None):
    """Run a forward test using the champion checkpoints from the specified job_id or latest champion.

    Evaluates from the champion's train_end_date+1 until last data date.
    """
    try:
        if not PROGRESSIVE_ML_AVAILABLE:
            raise HTTPException(status_code=503, detail="Progressive ML system not available")

        from pathlib import Path
        import pandas as pd
        from app.ml.progressive.backtester import ProgressiveBacktester

        # Locate champion dir
        champions_root = Path("app/ml/models/champions") / symbol
        if not champions_root.exists():
            raise HTTPException(status_code=404, detail="No champions found")
        target_dir = None
        if job_id:
            cand = champions_root / job_id
            if cand.exists():
                target_dir = cand
        if target_dir is None:
            # pick most recent
            dirs = [p for p in champions_root.iterdir() if p.is_dir()]
            if not dirs:
                raise HTTPException(status_code=404, detail="No champions found")
            target_dir = sorted(dirs, key=lambda p: p.stat().st_mtime, reverse=True)[0]

        meta_file = target_dir / "champion_meta.json"
        if not meta_file.exists():
            raise HTTPException(status_code=404, detail="Champion metadata not found")
        with open(meta_file, 'r', encoding='utf-8') as fh:
            meta = json.load(fh)

        train_end = meta.get('train_end_date')
        if not train_end:
            raise HTTPException(status_code=422, detail="Champion metadata missing train_end_date")

        # Prepare backtester helpers for evaluation-only
        bt = ProgressiveBacktester(
            data_loader=progressive_data_loader,
            trainer=progressive_trainer,
            predictor=progressive_predictor,
            progress_callback=None,
            cancel_checker=None
        )
        # Set model dir to champion
        bt.job_model_dir = target_dir
        # Load full data
        full_loader = ProgressiveDataLoader(
            stock_data_dir=progressive_data_loader.stock_data_dir,
            sequence_length=progressive_data_loader.sequence_length,
            horizons=progressive_data_loader.horizons,
            use_fundamentals=progressive_data_loader.use_fundamentals,
            use_technical_indicators=progressive_data_loader.use_technical_indicators,
            indicator_params=getattr(progressive_data_loader, 'indicator_params', None)
        )
        full_df = full_loader.load_stock_data(symbol)
        if full_df is None or len(full_df.index) == 0:
            raise HTTPException(status_code=404, detail="Price data not available")
        last_date = full_df.index.max().date().isoformat()
        start_date = (pd.to_datetime(train_end) + pd.Timedelta(days=1)).date().isoformat()
        if pd.to_datetime(start_date) > pd.to_datetime(last_date):
            raise HTTPException(status_code=422, detail="No forward window after champion train_end_date")

        eval_res = bt.evaluate_iteration(
            symbol=symbol,
            test_start_date=start_date,
            test_end_date=last_date,
            iteration_num=int(meta.get('best_iteration') or 0),
            full_df=full_df
        )

        return {
            'status': 'success',
            'symbol': symbol,
            'job_id': meta.get('job_id'),
            'champion_dir': str(target_dir),
            'train_end_date': train_end,
            'forward_start': start_date,
            'forward_end': last_date,
            'metrics': eval_res,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Champion forward test failed: {e}")
        raise HTTPException(status_code=500, detail=f"Champion forward test failed: {e}")