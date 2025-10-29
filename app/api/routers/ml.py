from __future__ import annotations

from datetime import datetime, timezone
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel

router = APIRouter(prefix="/api/ml", tags=["ML"])


@router.get("/status")
async def get_ml_status():
    """Get ML system status and capabilities (mirrors main endpoint)."""
    try:
        # Lazy import to avoid circular import at app startup
        from app.main_realtime import PROGRESSIVE_ML_AVAILABLE, progressive_predictor, progressive_trainer

        return {
            "status": "success",
            "data": {
                "progressive_ml_available": bool(PROGRESSIVE_ML_AVAILABLE),
                "tensorflow_available": False,
                "pytorch_available": True,
                "models": {
                    "progressive_pytorch": {
                        "status": "✅ Active" if PROGRESSIVE_ML_AVAILABLE else "⚠️ Demo Mode",
                        "type": "PyTorch Progressive ML",
                        "accuracy": "85-90%",
                        "best_for": "Real-time predictions, GPU acceleration",
                    },
                    "lstm": {
                        "status": "🚫 Disabled - Old TensorFlow",
                        "type": "Neural Network - Long Short-Term Memory",
                        "accuracy": "75-82%",
                        "best_for": "Long-term trends, sequential patterns",
                    },
                    "transformer": {
                        "status": "🚫 Disabled - Old TensorFlow",
                        "type": "Neural Network - Attention Mechanism",
                        "accuracy": "80-85%",
                        "best_for": "Complex relationships, multi-scale patterns",
                    },
                    "cnn": {
                        "status": "🚫 Disabled - Old TensorFlow",
                        "type": "Neural Network - Pattern Recognition",
                        "accuracy": "72-76%",
                        "best_for": "Chart patterns, technical analysis",
                    },
                    "random_forest": {
                        "status": "✅ Active" if PROGRESSIVE_ML_AVAILABLE else "⚠️ Unavailable",
                        "type": "Machine Learning - Ensemble Trees",
                        "accuracy": "78-85%",
                        "best_for": "Feature importance, non-linear relationships",
                    },
                    "gradient_boost": {
                        "status": "✅ Active" if PROGRESSIVE_ML_AVAILABLE else "⚠️ Unavailable",
                        "type": "Machine Learning - Boosting",
                        "accuracy": "80-88%",
                        "best_for": "High accuracy predictions, complex features",
                    },
                },
                "ensemble_method": "Weighted Average",
                "ensemble_weights": {"lstm": 0.4, "transformer": 0.35, "cnn": 0.25},
                "features_used": [
                    "price_history",
                    "volume",
                    "technical_indicators",
                    "sentiment",
                    "volatility",
                    "time_patterns",
                ],
                "timestamp": datetime.now().isoformat(),
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get ML status: {e}")


@router.get("/progressive/status")
async def get_progressive_ml_status():
    """Get Progressive ML system status"""
    try:
        from app.main_realtime import PROGRESSIVE_ML_AVAILABLE, progressive_data_loader, progressive_trainer, progressive_predictor

        if not PROGRESSIVE_ML_AVAILABLE:
            return {
                "status": "unavailable",
                "message": "Progressive ML system not loaded",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        status = {
            "status": "available",
            "data_loader": progressive_data_loader is not None,
            "trainer": progressive_trainer is not None,
            "predictor": progressive_predictor is not None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if progressive_trainer:
            try:
                status["training"] = {"is_training": False, "status": "ready"}
            except Exception:
                status["training"] = {"is_training": False, "status": "unknown"}
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get Progressive ML status: {e}")


@router.get("/progressive/models")
async def get_progressive_models():
    """Get available progressive ML models"""
    try:
        from app.main_realtime import PROGRESSIVE_ML_AVAILABLE, progressive_predictor

        if not PROGRESSIVE_ML_AVAILABLE:
            raise HTTPException(status_code=503, detail="Progressive ML system not available")

        models_info: Dict[str, Any] = {
            "available_models": ["lstm", "cnn", "transformer"],
            "available_modes": ["progressive", "unified"],
            "models_saved": {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if progressive_predictor:
            try:
                import os

                models_dir = "app/ml/models"
                if os.path.exists(models_dir):
                    models_info["models_saved"] = {"directory": models_dir, "found": True}
                else:
                    models_info["models_saved"] = {"directory": models_dir, "found": False}
            except Exception:
                models_info["models_saved"] = {"error": "Cannot check models directory"}

        return models_info
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get progressive models info: {e}")


@router.get("/progressive/training/status/{job_id}")
async def get_training_job_status(job_id: str):
    """Get status of specific training job"""
    try:
        from app.main_realtime import training_jobs

        if job_id in training_jobs:
            job_data = training_jobs[job_id].copy()
            job_data["timestamp"] = datetime.now(timezone.utc).isoformat()
            return job_data
        raise HTTPException(status_code=404, detail=f"Training job {job_id} not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get training status: {e}")


@router.get("/progressive/training/status")
async def get_all_training_status():
    """Get status of all training jobs"""
    try:
        from app.main_realtime import PROGRESSIVE_ML_AVAILABLE, progressive_trainer, training_jobs

        if not PROGRESSIVE_ML_AVAILABLE or not progressive_trainer:
            raise HTTPException(status_code=503, detail="Progressive ML trainer not available")

        active_jobs = [job for job in training_jobs.values() if job.get("status") in ["starting", "running"]]
        return {
            "status": "success",
            "trainer_available": True,
            "is_training": len(active_jobs) > 0,
            "active_jobs": active_jobs,
            "total_jobs": len(training_jobs),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get training status: {e}")


# ==============================================
# Batch 2: Training and Prediction endpoints
# ==============================================

@router.post("/progressive/train")
async def start_progressive_training(
    background_tasks: BackgroundTasks,
    symbol: str = Query(..., description="Stock symbol to train"),
    model_types: str = Query("lstm", description="Comma-separated model types"),
    mode: str = Query("progressive", description="Training mode"),
):
    """Start progressive training for a stock symbol (async with progress tracking)."""
    try:
        from app.main_realtime import PROGRESSIVE_ML_AVAILABLE, progressive_trainer, data_manager, training_jobs, run_training_job

        if not PROGRESSIVE_ML_AVAILABLE or not progressive_trainer:
            raise HTTPException(status_code=503, detail="Progressive ML trainer not available")

        model_types_list = [mt.strip() for mt in model_types.split(',') if mt.strip()] or ["lstm"]

        # Ensure data is ready
        try:
            training_jobs.clear()
        except Exception:
            pass
        ensure_summary = await asyncio.to_thread(data_manager.ensure_symbol_data, symbol)

        import uuid
        job_id = f"train_{symbol}_{uuid.uuid4().hex[:8]}"
        from datetime import datetime as _dt, timezone as _tz
        training_jobs[job_id] = {
            "job_id": job_id,
            "symbol": symbol,
            "model_types": model_types_list,
            "mode": mode,
            "status": "starting",
            "progress": 0,
            "current_step": "Initializing...",
            "eta_seconds": None,
            "start_time": _dt.now(_tz.utc).isoformat(),
            "end_time": None,
            "result": None,
            "error": None,
        }

        background_tasks.add_task(
            run_training_job,
            job_id=job_id,
            symbol=symbol,
            model_types=model_types_list,
            mode=mode,
        )

        return {
            "status": "training_started",
            "job_id": job_id,
            "symbol": symbol,
            "model_types": model_types_list,
            "mode": mode,
            "message": "Training started in background. Use job_id to track progress.",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start progressive training: {e}")


@router.post("/progressive/predict/{symbol}")
async def progressive_predict(symbol: str, mode: str = "progressive", include_risk: bool = True):
    """Get progressive ML predictions for a symbol."""
    try:
        from app.main_realtime import PROGRESSIVE_ML_AVAILABLE, progressive_predictor, data_manager

        if not PROGRESSIVE_ML_AVAILABLE or not progressive_predictor:
            raise HTTPException(status_code=503, detail="Progressive ML predictor not available")

        # Predictions
        predictions = progressive_predictor.predict_ensemble(symbol=symbol, mode=mode)
        if not predictions or not isinstance(predictions, dict):
            raise HTTPException(status_code=500, detail="Invalid predictions data returned")
        if 'current_price' not in predictions:
            raise HTTPException(status_code=500, detail="Missing current_price in predictions")

        # Optional risk enrichment
        if include_risk:
            try:
                import pandas as pd
                close_price = float(predictions.get('current_price', 0.0))
                ind_path = data_manager.stock_data_dir / symbol / f"{symbol}_indicators.csv"
                price_path = data_manager.stock_data_dir / symbol / f"{symbol}_price.csv"
                atr_pct = None
                if ind_path.exists():
                    ind_df = pd.read_csv(ind_path, index_col=0)
                    if 'ATR_14' in ind_df.columns:
                        atr_val = float(pd.to_numeric(ind_df['ATR_14'], errors='coerce').dropna().iloc[-1])
                        if close_price > 0:
                            atr_pct = max(0.001, min(0.2, atr_val / close_price))
                if atr_pct is None and price_path.exists():
                    df = pd.read_csv(price_path, index_col=0)
                    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
                    df = df.dropna(subset=['Close'])
                    rets = df['Close'].pct_change().dropna()
                    vol = float(rets.rolling(14).std().dropna().iloc[-1]) if len(rets) > 14 else float(rets.std())
                    atr_pct = max(0.001, min(0.2, vol * 1.5))

                rr = 2.0
                risk_pct = max(0.005, min(0.2, atr_pct or 0.01))
                reward_pct = max(0.01, min(0.4, risk_pct * rr))
                risk_block = {'basis': 'ATR_14' if atr_pct is not None else 'volatility', 'risk_pct': risk_pct, 'reward_pct': reward_pct}
                preds = predictions.get('predictions', {})
                for horizon, obj in preds.items():
                    change_pct = float(obj.get('price_change_pct', 0.0))
                    sl_price = close_price * (1 - risk_pct)
                    tp_price = close_price * (1 + reward_pct)
                    if change_pct < 0:
                        sl_price = close_price * (1 + risk_pct)
                        tp_price = close_price * (1 - reward_pct)
                    obj['risk'] = {
                        'stop_loss': round(sl_price, 4),
                        'take_profit': round(tp_price, 4),
                        'stop_loss_pct': -risk_pct if change_pct >= 0 else risk_pct,
                        'take_profit_pct': reward_pct if change_pct >= 0 else -reward_pct,
                        'policy': risk_block,
                    }
            except Exception:
                pass

        return {
            "status": "success",
            "symbol": symbol,
            "mode": mode,
            "predictions": predictions,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get progressive predictions: {e}")


# ==============================================
# Batch 3: Backtest, Champions, Forward-test
# ==============================================

class BacktestRequest(BaseModel):
    symbol: str
    train_start_date: str
    train_end_date: str
    test_period_days: int = 14
    max_iterations: int = 10
    target_accuracy: float = 0.85
    auto_stop: bool = True
    auto_adjust_iterations: bool = True
    auto_plan: bool = True
    deep_mode: bool = True
    desired_iterations: int | None = None
    desired_test_period_days: int | None = None
    training_window_days: int | None = None
    ensure_fresh_data: bool = True
    model_types: List[str] = ["lstm"]
    indicator_params: Dict[str, Any] | None = None
    auto_scout: bool = True
    scout_candidate_windows: List[int] | None = None
    scout_candidate_seq: List[int] | None = None
    scout_indicator_profiles: List[str] | None = None
    scout_forward_days: int = 14
    scout_min_predictions: int = 8
    scout_epochs: int = 10
    scout_model_types: List[str] = ["cnn"]


@router.post("/progressive/backtest")
async def start_backtest(request: BacktestRequest, background_tasks: BackgroundTasks):
    """Start progressive backtesting with date-range training."""
    try:
        # Lazy import shared state and worker
        from app.main_realtime import (
            PROGRESSIVE_ML_AVAILABLE,
            progressive_data_loader,
            progressive_trainer,
            progressive_predictor,
            data_manager,
            backtest_jobs,
            run_backtest_job,
        )

        if not PROGRESSIVE_ML_AVAILABLE:
            raise HTTPException(status_code=503, detail="Progressive ML system not available")
        if not progressive_data_loader or not progressive_trainer or not progressive_predictor:
            raise HTTPException(status_code=503, detail="Progressive ML components not initialized")

        # Optional preflight ensure
        try:
            if request.ensure_fresh_data:
                await asyncio.to_thread(data_manager.ensure_symbol_data, request.symbol)
        except Exception:
            pass

        # Preflight feasibility computation mirrors main; re-use main worker for the heavy lifting
        import pandas as pd
        from pathlib import Path
        price_csv = data_manager.stock_data_dir / request.symbol / f"{request.symbol}_price.csv"
        if not price_csv.exists():
            await asyncio.to_thread(data_manager.ensure_symbol_data, request.symbol)
        if not price_csv.exists():
            raise HTTPException(status_code=404, detail=f"Price data not found for {request.symbol}")
        df = pd.read_csv(price_csv, index_col=0)
        df.index = pd.to_datetime(df.index, format='mixed', errors='coerce', utc=True).tz_localize(None)
        df = df.sort_index()
        if len(df.index) == 0:
            raise HTTPException(status_code=422, detail=f"Empty price data for {request.symbol}")
        last_data_date = df.index.max()

        plan: Dict[str, Any] = {}
        tpd = int(request.desired_test_period_days or request.test_period_days or 14)
        req_iters = int(request.desired_iterations or request.max_iterations or 10)
        if request.auto_plan:
            needed = req_iters * tpd
            if needed >= len(df.index):
                feasible_iters_from_data = max(0, (len(df.index) - 1) // tpd)
                needed = feasible_iters_from_data * tpd
            end_pos = max(0, len(df.index) - needed - 1)
            end_dt = df.index[end_pos]
            if request.training_window_days and request.training_window_days > 0:
                start_dt = max(df.index[0], end_dt - pd.Timedelta(days=int(request.training_window_days)))
            else:
                candidate_start = None
                if request.train_start_date:
                    try:
                        candidate_start = pd.to_datetime(request.train_start_date, utc=True, errors='coerce').tz_localize(None)
                    except Exception:
                        candidate_start = None
                if candidate_start is None or pd.isna(candidate_start) or candidate_start < df.index[0]:
                    start_dt = df.index[0]
                else:
                    start_dt = candidate_start
            if start_dt > end_dt:
                start_dt = df.index[0]
            planned_train_start = start_dt.date().isoformat()
            planned_train_end = end_dt.date().isoformat()
            request = request.copy(update={
                "train_start_date": planned_train_start,
                "train_end_date": planned_train_end,
                "test_period_days": tpd,
                "max_iterations": req_iters,
            })
            plan.update({
                "train_start_date": planned_train_start,
                "train_end_date": planned_train_end,
                "test_period_days": tpd,
                "requested_iterations": req_iters,
            })
            end_dt_use = end_dt
        else:
            try:
                end_dt_use = pd.to_datetime(request.train_end_date, utc=True).tz_localize(None)
            except Exception:
                raise HTTPException(status_code=400, detail=f"Invalid train_end_date: {request.train_end_date}")

        trading_days_after_end = int((df.index > end_dt_use).sum())
        feasible = int(trading_days_after_end // tpd)
        preflight = {
            "requested_max_iterations": int(request.max_iterations),
            "feasible_max_iterations": feasible,
            "trading_days_after_end": trading_days_after_end,
            "last_data_date": last_data_date.date().isoformat(),
            "test_period_days": tpd,
            "plan": plan,
        }
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
            "cancelled": False,
            "preflight": preflight,
            "adjusted": adjusted,
        }

        background_tasks.add_task(run_backtest_job, job_id=job_id, request=adjusted_request)

        return {
            "status": "backtest_started",
            "job_id": job_id,
            "symbol": request.symbol,
            "preflight": preflight,
            "adjusted": adjusted,
            "plan": preflight.get("plan"),
            "message": "Backtest started in background. Use job_id to track progress.",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start backtest: {e}")


@router.get("/progressive/backtest/status/{job_id}")
async def get_backtest_status(job_id: str):
    try:
        from app.main_realtime import backtest_jobs
        if job_id in backtest_jobs:
            job = backtest_jobs[job_id].copy()
            job["timestamp"] = datetime.now(timezone.utc).isoformat()
            return job
        raise HTTPException(status_code=404, detail=f"Backtest job {job_id} not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get backtest status: {e}")


@router.post("/progressive/backtest/cancel/{job_id}")
async def cancel_backtest(job_id: str):
    try:
        from app.main_realtime import backtest_jobs
        if job_id not in backtest_jobs:
            raise HTTPException(status_code=404, detail=f"Backtest job {job_id} not found")
        backtest_jobs[job_id]["cancelled"] = True
        if backtest_jobs[job_id]["status"] in ["starting", "running"]:
            backtest_jobs[job_id]["status"] = "cancelling"
            backtest_jobs[job_id]["current_step"] = "Cancelling..."
        return {"status": "cancellation_requested", "job_id": job_id, "timestamp": datetime.now(timezone.utc).isoformat()}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to cancel backtest: {e}")


@router.get("/progressive/backtest/results/{symbol}")
async def get_backtest_results(symbol: str):
    try:
        from app.main_realtime import PROGRESSIVE_ML_AVAILABLE
        if not PROGRESSIVE_ML_AVAILABLE:
            raise HTTPException(status_code=503, detail="Progressive ML system not available")
        import json
        from pathlib import Path
        results_dir = Path("app/ml/models/backtest_results")
        if not results_dir.exists():
            return {"status": "no_results", "message": f"No backtest results found for {symbol}", "symbol": symbol}
        files = list(results_dir.glob(f"results_{symbol}_*.json"))
        if not files:
            return {"status": "no_results", "message": f"No backtest results found for {symbol}", "symbol": symbol}
        latest_file = max(files, key=lambda p: p.stat().st_mtime)
        with open(latest_file, "r", encoding="utf-8") as f:
            results = json.load(f)
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
        return {"status": "success", "symbol": symbol, "results": results, "file": latest_file.name, "timestamp": datetime.now(timezone.utc).isoformat()}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get backtest results: {e}")


@router.get("/progressive/backtest/history/{symbol}")
async def list_backtest_history(symbol: str, limit: int = Query(20, ge=1, le=100)):
    try:
        from pathlib import Path
        import json
        results_dir = Path("app/ml/models/backtest_results")
        if not results_dir.exists():
            return {"status": "success", "symbol": symbol, "items": []}
        files = sorted(results_dir.glob(f"results_{symbol}_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        items: List[Dict[str, Any]] = []
        for f in files[:limit]:
            try:
                with open(f, 'r', encoding='utf-8') as fh:
                    data = json.load(fh)
                iterations = data.get('all_iterations') or data.get('iterations') or []
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
                    "iterations": len(iterations),
                })
            except Exception:
                items.append({
                    "file": f.name,
                    "modified": datetime.fromtimestamp(f.stat().st_mtime, tz=timezone.utc).isoformat(),
                    "best_accuracy": None,
                    "iterations": None,
                })
        return {"status": "success", "symbol": symbol, "items": items}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list backtest history: {e}")


@router.get("/progressive/backtest/result_by_file/{symbol}/{file_name}")
async def get_backtest_result_by_file(symbol: str, file_name: str):
    try:
        from pathlib import Path
        import json
        results_dir = Path("app/ml/models/backtest_results")
        target = results_dir / file_name
        if not target.exists() or f"results_{symbol}_" not in target.name:
            raise HTTPException(status_code=404, detail="Results file not found")
        with open(target, 'r', encoding='utf-8') as f:
            results = json.load(f)
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
        raise HTTPException(status_code=500, detail=f"Failed to get backtest result: {e}")


@router.get("/progressive/champions/{symbol}")
async def list_champions(symbol: str):
    try:
        from pathlib import Path
        import json
        root = Path("app/ml/models/champions") / symbol
        if not root.exists():
            return {"status": "success", "symbol": symbol, "items": []}
        items: List[Dict[str, Any]] = []
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
                    'modified': datetime.fromtimestamp(job_dir.stat().st_mtime, tz=timezone.utc).isoformat(),
                })
        return {"status": "success", "symbol": symbol, "items": items}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list champions: {e}")


@router.post("/progressive/champion/forward_test/{symbol}")
async def champion_forward_test(symbol: str, job_id: str | None = None):
    try:
        from app.main_realtime import (
            PROGRESSIVE_ML_AVAILABLE,
            progressive_data_loader,
            progressive_trainer,
            progressive_predictor,
        )
        if not PROGRESSIVE_ML_AVAILABLE:
            raise HTTPException(status_code=503, detail="Progressive ML system not available")
        from pathlib import Path
        import pandas as pd
        from app.ml.progressive.backtester import ProgressiveBacktester
        champions_root = Path("app/ml/models/champions") / symbol
        if not champions_root.exists():
            raise HTTPException(status_code=404, detail="No champions found")
        target_dir = None
        if job_id:
            cand = champions_root / job_id
            if cand.exists():
                target_dir = cand
        if target_dir is None:
            dirs = [p for p in champions_root.iterdir() if p.is_dir()]
            if not dirs:
                raise HTTPException(status_code=404, detail="No champions found")
            target_dir = sorted(dirs, key=lambda p: p.stat().st_mtime, reverse=True)[0]
        import json
        meta_file = target_dir / "champion_meta.json"
        if not meta_file.exists():
            raise HTTPException(status_code=404, detail="Champion metadata not found")
        with open(meta_file, 'r', encoding='utf-8') as fh:
            meta = json.load(fh)
        train_end = meta.get('train_end_date')
        if not train_end:
            raise HTTPException(status_code=422, detail="Champion metadata missing train_end_date")

        bt = ProgressiveBacktester(
            data_loader=progressive_data_loader,
            trainer=progressive_trainer,
            predictor=progressive_predictor,
            progress_callback=None,
            cancel_checker=None,
        )
        bt.job_model_dir = target_dir
        from app.ml.progressive.data_loader import ProgressiveDataLoader as _DL
        full_loader = _DL(
            stock_data_dir=progressive_data_loader.stock_data_dir,
            sequence_length=progressive_data_loader.sequence_length,
            horizons=progressive_data_loader.horizons,
            use_fundamentals=progressive_data_loader.use_fundamentals,
            use_technical_indicators=progressive_data_loader.use_technical_indicators,
            indicator_params=getattr(progressive_data_loader, 'indicator_params', None),
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
            full_df=full_df,
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
            'timestamp': datetime.now(timezone.utc).isoformat(),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Champion forward test failed: {e}")
