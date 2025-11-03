"""Shared state and helpers for progressive ML backtests.

Routers and template routes use these helpers to schedule long-running
progressive jobs without depending on legacy entrypoints.
"""
from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Any, Dict

from fastapi import HTTPException
from pydantic import BaseModel

from app.core.config import (
    logger,
    PROGRESSIVE_ML_AVAILABLE,
    progressive_data_loader,
    progressive_trainer,
    progressive_predictor,
    data_manager,
)

backtest_jobs: Dict[str, Dict[str, Any]] = {}


def ensure_progressive_ready() -> None:
    """Validate that progressive ML components are available before running jobs."""
    if not PROGRESSIVE_ML_AVAILABLE:
        raise HTTPException(status_code=503, detail="Progressive ML system not available")
    if not progressive_data_loader or not progressive_trainer or not progressive_predictor:
        raise HTTPException(status_code=503, detail="Progressive ML components not initialized")


async def run_backtest_job(job_id: str, request: BaseModel) -> None:
    """Execute a progressive backtest and update shared job state."""
    import json
    import shutil
    from pathlib import Path

    start_time = time.time()
    job_state = backtest_jobs.setdefault(job_id, {"job_id": job_id})

    try:
        ensure_progressive_ready()

        job_state.update({
            "status": "running",
            "progress": 5,
            "current_step": "Ensuring ticker data...",
        })

        def _progress_cb(event: Dict[str, Any]) -> None:
            try:
                progress = int(event.get("progress", job_state.get("progress", 0)))
                job_state.update({
                    "status": event.get("status", job_state.get("status", "running")),
                    "progress": max(min(progress, 99), 0),
                    "current_step": event.get("current_step", job_state.get("current_step", "")),
                    "eta_seconds": event.get("eta_seconds"),
                })
            except Exception:
                pass

        try:
            await asyncio.to_thread(data_manager.ensure_symbol_data, request.symbol)
            job_state["current_step"] = "Initializing backtester..."
            job_state["progress"] = 10
        except Exception as exc:
            job_state.update({
                "status": "failed",
                "current_step": f"❌ Ensure data failed: {exc}",
                "error": str(exc),
            })
            return

        def _indicator_profile(name: str) -> Dict[str, Any]:
            alias = (name or "").strip().lower()
            if alias in ("short_mid", "short-mid", "shortmid"):
                return {
                    "rsi_period": 12,
                    "macd_fast": 8,
                    "macd_slow": 21,
                    "macd_signal": 5,
                    "sma_periods": "5,10,20,50",
                    "ema_periods": "5,10,20,50",
                    "bb_period": 20,
                    "bb_std": 2.0,
                }
            if alias in ("mid_long", "mid-long", "midlong", "mid_long_term"):
                return {
                    "rsi_period": 18,
                    "macd_fast": 19,
                    "macd_slow": 39,
                    "macd_signal": 9,
                    "sma_periods": "20,50,100,200",
                    "ema_periods": "20,50,100",
                    "bb_period": 30,
                    "bb_std": 2.5,
                }
            return request.indicator_params or {
                "rsi_period": 14,
                "macd_fast": 12,
                "macd_slow": 26,
                "macd_signal": 9,
                "sma_periods": "5,10,20,50",
                "ema_periods": "5,10,20,50",
                "bb_period": 20,
                "bb_std": 2.0,
            }

        dl_for_job = progressive_data_loader
        try:
            if getattr(request, "indicator_params", None) is not None and PROGRESSIVE_ML_AVAILABLE:
                from app.ml.progressive.data_loader import ProgressiveDataLoader

                dl_for_job = ProgressiveDataLoader(
                    stock_data_dir=progressive_data_loader.stock_data_dir,
                    sequence_length=progressive_data_loader.sequence_length,
                    horizons=progressive_data_loader.horizons,
                    use_fundamentals=progressive_data_loader.use_fundamentals,
                    use_technical_indicators=progressive_data_loader.use_technical_indicators,
                    indicator_params=request.indicator_params,
                )
                logger.info("🧩 Using job-specific indicator params for backtest")
        except Exception as exc:
            logger.warning("Failed to construct job-specific data loader: %s", exc)

        scout_report: list[Dict[str, Any]] = []
        try:
            if bool(getattr(request, "auto_scout", True)):
                job_state["current_step"] = "Scouting best config..."
                job_state["progress"] = 15

                import pandas as pd
                from pathlib import Path as PathAlias
                from app.ml.progressive.data_loader import ProgressiveDataLoader as ScoutLoader
                from app.ml.progressive.predictor import ProgressivePredictor
                from app.ml.progressive.trainer import ProgressiveTrainer
                from app.ml.progressive.backtester import ProgressiveBacktester as ScoutBacktester

                end_dt = pd.to_datetime(request.train_end_date)
                candidate_windows = request.scout_candidate_windows or [360, 540, 720]
                candidate_sequences = request.scout_candidate_seq or [60, 90]
                candidate_profiles = request.scout_indicator_profiles or ["short_mid", "mid_long"]
                fine_sets = [
                    {
                        "name": "short_momentum",
                        "params": {
                            "rsi_period": 12,
                            "macd_fast": 8,
                            "macd_slow": 21,
                            "macd_signal": 5,
                            "sma_periods": "5,10,20,50",
                            "ema_periods": "5,10,20,50",
                            "bb_period": 20,
                            "bb_std": 2.0,
                        },
                    },
                    {
                        "name": "balanced",
                        "params": {
                            "rsi_period": 14,
                            "macd_fast": 12,
                            "macd_slow": 26,
                            "macd_signal": 9,
                            "sma_periods": "10,20,50,100",
                            "ema_periods": "10,20,50",
                            "bb_period": 20,
                            "bb_std": 2.0,
                        },
                    },
                    {
                        "name": "long_trend",
                        "params": {
                            "rsi_period": 18,
                            "macd_fast": 19,
                            "macd_slow": 39,
                            "macd_signal": 9,
                            "sma_periods": "20,50,100,200",
                            "ema_periods": "20,50,100",
                            "bb_period": 30,
                            "bb_std": 2.5,
                        },
                    },
                    {
                        "name": "volatility_band",
                        "params": {
                            "rsi_period": 14,
                            "macd_fast": 12,
                            "macd_slow": 26,
                            "macd_signal": 9,
                            "sma_periods": "5,20,50,100",
                            "ema_periods": "5,20,50",
                            "bb_period": 10,
                            "bb_std": 3.0,
                        },
                    },
                ]

                candidates: list[Dict[str, Any]] = []
                for window_days in candidate_windows:
                    for seq_len in candidate_sequences:
                        for profile in candidate_profiles:
                            candidates.append({
                                "window_days": int(window_days),
                                "sequence_length": int(seq_len),
                                "indicator_params": _indicator_profile(profile),
                                "profile": profile,
                            })
                        for preset in fine_sets:
                            candidates.append({
                                "window_days": int(window_days),
                                "sequence_length": int(seq_len),
                                "indicator_params": preset["params"],
                                "profile": preset["name"],
                            })

                candidates = candidates[:16]
                best_choice: Dict[str, Any] | None = None

                for idx, candidate in enumerate(candidates, start=1):
                    try:
                        start_dt = (end_dt - pd.Timedelta(days=candidate["window_days"])).date().isoformat()
                        loader = ScoutLoader(
                            stock_data_dir=progressive_data_loader.stock_data_dir,
                            sequence_length=candidate["sequence_length"],
                            horizons=progressive_data_loader.horizons,
                            use_fundamentals=progressive_data_loader.use_fundamentals,
                            use_technical_indicators=progressive_data_loader.use_technical_indicators,
                            indicator_params=candidate["indicator_params"],
                            train_start_date=start_dt,
                            train_end_date=request.train_end_date,
                        )
                        job_dir = PathAlias("app/ml/models/backtests") / job_id / f"scout_{idx:02d}"
                        job_dir.mkdir(parents=True, exist_ok=True)

                        trainer = ProgressiveTrainer(
                            data_loader=loader,
                            training_config={
                                "epochs": int(request.scout_epochs or 10),
                                "batch_size": 64,
                                "validation_split": 0.2,
                                "early_stopping_patience": 4,
                                "reduce_lr_patience": 3,
                                "reduce_lr_factor": 0.5,
                            },
                            save_dir=str(job_dir),
                        )
                        predictor = ProgressivePredictor(data_loader=loader, model_dir=str(job_dir))
                        scout_backtester = ScoutBacktester(
                            data_loader=loader,
                            trainer=trainer,
                            predictor=predictor,
                        )

                        scout_models = list(set(request.scout_model_types or ["cnn"]))
                        trainer.train_progressive_models(symbol=request.symbol, model_types=scout_models)

                        test_start = (pd.to_datetime(request.train_end_date) + pd.Timedelta(days=1)).date().isoformat()
                        full_df = loader.load_stock_data(request.symbol)
                        test_end = (pd.to_datetime(test_start) + pd.Timedelta(days=int(request.scout_forward_days or 14))).date().isoformat()
                        if full_df is not None and len(full_df.index) > 0:
                            last_dt = full_df.index.max().date().isoformat()
                            if pd.to_datetime(test_end) > pd.to_datetime(last_dt):
                                test_end = last_dt

                        evaluation = scout_backtester.evaluate_iteration(  # type: ignore[arg-type]
                            symbol=request.symbol,
                            test_start_date=test_start,
                            test_end_date=test_end,
                            iteration_num=1,
                            full_df=full_df if full_df is not None else loader.load_stock_data(request.symbol),
                        )
                        accuracy = float(evaluation.get("direction_accuracy") or evaluation.get("accuracy") or 0.0)
                        predictions = int(evaluation.get("predictions_made") or evaluation.get("test_samples") or 0)
                        scout_report.append({
                            "idx": idx,
                            "window_days": candidate["window_days"],
                            "sequence_length": candidate["sequence_length"],
                            "profile": candidate["profile"],
                            "accuracy": accuracy,
                            "predictions": predictions,
                            "mae": evaluation.get("mae"),
                            "rmse": evaluation.get("rmse"),
                            "mape": evaluation.get("mape"),
                            "dir": str(job_dir),
                            "test_start": test_start,
                            "test_end": test_end,
                        })

                        if predictions >= int(request.scout_min_predictions or 8):
                            if best_choice is None or accuracy > best_choice["accuracy"] or (
                                abs(accuracy - best_choice["accuracy"]) < 1e-9
                                and (evaluation.get("mape") or 1e9) < (best_choice.get("mape") or 1e9)
                            ):
                                best_choice = scout_report[-1]

                        job_state["current_step"] = f"Scouting {idx}/{len(candidates)}..."
                    except Exception as candidate_exc:
                        logger.warning("Scout candidate %s failed: %s", idx, candidate_exc)
                        continue

                if best_choice is None and scout_report:
                    best_choice = max(scout_report, key=lambda record: record.get("accuracy", 0.0))

                if best_choice:
                    from app.ml.progressive.data_loader import ProgressiveDataLoader as JobLoader

                    request = request.copy(update={
                        "train_start_date": (
                            (end_dt - pd.Timedelta(days=int(best_choice["window_days"]))).date().isoformat()
                        )
                    })
                    dl_for_job = JobLoader(
                        stock_data_dir=progressive_data_loader.stock_data_dir,
                        sequence_length=int(best_choice["sequence_length"]),
                        horizons=progressive_data_loader.horizons,
                        use_fundamentals=progressive_data_loader.use_fundamentals,
                        use_technical_indicators=progressive_data_loader.use_technical_indicators,
                        indicator_params=_indicator_profile(best_choice["profile"]),
                    )
                    job_state["scout"] = {"candidates": scout_report, "chosen": best_choice}
                    if not request.model_types or len(request.model_types) <= 1:
                        request = request.copy(update={"model_types": ["cnn", "lstm", "transformer"]})
                else:
                    job_state["scout"] = {
                        "candidates": scout_report,
                        "chosen": None,
                        "note": "No viable candidate met minimum predictions; proceeding with original plan",
                    }
        except Exception as scout_exc:
            logger.warning("Auto-scout skipped due to error: %s", scout_exc)

        from app.ml.progressive.backtester import ProgressiveBacktester as Backtester

        backtester = Backtester(
            data_loader=dl_for_job,
            trainer=progressive_trainer,
            predictor=progressive_predictor,
            progress_callback=_progress_cb,
            cancel_checker=lambda: bool(job_state.get("cancelled")),
        )

        results = backtester.run_backtest(
            symbol=request.symbol,
            train_start_date=request.train_start_date,
            train_end_date=request.train_end_date,
            test_period_days=request.test_period_days,
            max_iterations=request.max_iterations,
            target_accuracy=request.target_accuracy,
            auto_stop=request.auto_stop,
            model_types=request.model_types,
        )

        status_result = results.get("status") if isinstance(results, dict) else None
        if status_result == "cancelled":
            job_state.update({
                "status": "cancelled",
                "progress": job_state.get("progress", 0),
                "current_step": "⏹ Backtest cancelled",
                "eta_seconds": 0,
                "end_time": datetime.now(timezone.utc).isoformat(),
                "result": results,
            })
            return

        champion_info: Dict[str, Any] | None = None
        try:
            best_iteration = int(results.get("best_iteration")) if isinstance(results, dict) else None
            job_model_dir = getattr(backtester, "job_model_dir", None)
            if best_iteration and job_model_dir:
                iter_dir = Path(job_model_dir) / f"iter_{best_iteration:02d}"
                if iter_dir.exists():
                    champions_root = Path("app/ml/models/champions") / request.symbol
                    target_dir = champions_root / f"{results.get('job_id', job_id)}"
                    target_dir.mkdir(parents=True, exist_ok=True)
                    for artifact in iter_dir.glob("*.pth"):
                        shutil.copy2(artifact, target_dir / artifact.name)
                    meta = {
                        "symbol": request.symbol,
                        "job_id": results.get("job_id", job_id),
                        "best_iteration": best_iteration,
                        "summary": {
                            key: results.get(key)
                            for key in ["best_accuracy", "best_loss", "total_iterations", "total_time"]
                        },
                        "train_end_date": request.train_end_date,
                        "test_period_days": request.test_period_days,
                        "model_types": request.model_types,
                    }
                    with open(target_dir / "champion_meta.json", "w", encoding="utf-8") as meta_file:
                        json.dump(meta, meta_file, indent=2)
                    champion_info = {"dir": str(target_dir), "meta": meta}
            if champion_info:
                job_state["champion"] = champion_info
        except Exception as snapshot_exc:
            logger.warning("Champion snapshot failed: %s", snapshot_exc)

        try:
            if champion_info and PROGRESSIVE_ML_AVAILABLE:
                from app.ml.progressive.backtester import ProgressiveBacktester as EvalBacktester
                from app.ml.progressive.data_loader import ProgressiveDataLoader as EvalLoader
                import pandas as pd_eval
                from pathlib import Path as PathEval

                evaluator = EvalBacktester(
                    data_loader=progressive_data_loader,
                    trainer=progressive_trainer,
                    predictor=progressive_predictor,
                    progress_callback=None,
                    cancel_checker=None,
                )
                evaluator.job_model_dir = PathEval(champion_info["dir"])
                loader = EvalLoader(
                    stock_data_dir=progressive_data_loader.stock_data_dir,
                    sequence_length=progressive_data_loader.sequence_length,
                    horizons=progressive_data_loader.horizons,
                    use_fundamentals=progressive_data_loader.use_fundamentals,
                    use_technical_indicators=progressive_data_loader.use_technical_indicators,
                    indicator_params=getattr(progressive_data_loader, "indicator_params", None),
                )
                full_df = loader.load_stock_data(request.symbol)
                if full_df is not None and len(full_df.index) > 0:
                    last_date = full_df.index.max().date().isoformat()
                    train_end = champion_info["meta"].get("train_end_date")
                    start_date = (
                        (pd_eval.to_datetime(train_end) + pd_eval.Timedelta(days=1)).date().isoformat()
                        if train_end
                        else None
                    )
                    if start_date and pd_eval.to_datetime(start_date) <= pd_eval.to_datetime(last_date):
                        forward_eval = evaluator.evaluate_iteration(
                            symbol=request.symbol,
                            test_start_date=start_date,
                            test_end_date=last_date,
                            iteration_num=int(champion_info["meta"].get("best_iteration") or 0),
                            full_df=full_df,
                        )
                        job_state["forward"] = {
                            "forward_start": start_date,
                            "forward_end": last_date,
                            "metrics": forward_eval,
                        }
        except Exception as forward_exc:
            logger.warning("Auto forward test skipped: %s", forward_exc)

        try:
            if champion_info and PROGRESSIVE_ML_AVAILABLE:
                from app.ml.progressive.predictor import ProgressivePredictor as ChampionPredictor
                from app.ml.progressive.data_loader import ProgressiveDataLoader as PredictLoader
                from pathlib import Path as PathPredict
                import torch
                import pandas as pd_predict

                indicator_params = getattr(progressive_data_loader, "indicator_params", None)
                sequence_len = getattr(progressive_data_loader, "sequence_length", 60)
                try:
                    scout_ctx = job_state.get("scout", {})
                    chosen = scout_ctx.get("chosen") if isinstance(scout_ctx, dict) else None
                    if chosen and isinstance(chosen, dict):
                        sequence_len = int(chosen.get("sequence_length") or sequence_len)
                        profile = chosen.get("profile")
                        if profile:
                            indicator_params = _indicator_profile(str(profile))
                except Exception:
                    pass

                try:
                    checkpoints = list(PathPredict(champion_info["dir"]).glob("*.pth"))
                    if checkpoints:
                        metadata = torch.load(checkpoints[0], map_location="cpu")
                        seq_val = metadata.get("sequence_length")
                        if isinstance(seq_val, int) and seq_val > 0:
                            sequence_len = seq_val
                except Exception as ck_exc:
                    logger.debug("Could not read sequence_length from checkpoint: %s", ck_exc)

                pred_loader = PredictLoader(
                    stock_data_dir=progressive_data_loader.stock_data_dir,
                    sequence_length=int(sequence_len),
                    horizons=progressive_data_loader.horizons,
                    use_fundamentals=progressive_data_loader.use_fundamentals,
                    use_technical_indicators=progressive_data_loader.use_technical_indicators,
                    indicator_params=indicator_params,
                )
                predictor = ChampionPredictor(data_loader=pred_loader, model_dir=str(champion_info["dir"]))
                try:
                    predictions = predictor.predict_ensemble(symbol=request.symbol, mode="progressive")
                    try:
                        ind_path = data_manager.stock_data_dir / request.symbol / f"{request.symbol}_indicators.csv"
                        price_path = data_manager.stock_data_dir / request.symbol / f"{request.symbol}_price.csv"
                        close_price = float(predictions.get("current_price", 0.0))
                        atr_pct = None
                        if ind_path.exists():
                            ind_df = pd_predict.read_csv(ind_path, index_col=0)
                            if "ATR_14" in ind_df.columns and close_price > 0:
                                atr_val = float(pd_predict.to_numeric(ind_df["ATR_14"], errors="coerce").dropna().iloc[-1])
                                atr_pct = max(0.001, min(0.2, atr_val / close_price))
                        if atr_pct is None and price_path.exists():
                            price_df = pd_predict.read_csv(price_path, index_col=0)
                            price_df["Close"] = pd_predict.to_numeric(price_df["Close"], errors="coerce")
                            price_df = price_df.dropna(subset=["Close"])
                            returns = price_df["Close"].pct_change().dropna()
                            vol = float(returns.rolling(14).std().dropna().iloc[-1]) if len(returns) > 14 else float(returns.std())
                            atr_pct = max(0.001, min(0.2, vol * 1.5))
                        rr = 2.0
                        risk_pct = max(0.005, min(0.2, atr_pct or 0.01))
                        reward_pct = max(0.01, min(0.4, risk_pct * rr))
                        for horizon, payload in (predictions.get("predictions") or {}).items():
                            change_pct = float(payload.get("price_change_pct", 0.0))
                            stop_loss = close_price * (1 - risk_pct)
                            take_profit = close_price * (1 + reward_pct)
                            if change_pct < 0:
                                stop_loss = close_price * (1 + risk_pct)
                                take_profit = close_price * (1 - reward_pct)
                            payload["risk"] = {
                                "stop_loss": round(stop_loss, 4),
                                "take_profit": round(take_profit, 4),
                                "stop_loss_pct": -risk_pct if change_pct >= 0 else risk_pct,
                                "take_profit_pct": reward_pct if change_pct >= 0 else -reward_pct,
                                "basis": "ATR_14" if atr_pct is not None else "volatility",
                                "rr": rr,
                            }
                    except Exception as risk_exc:
                        logger.debug("Risk enrich (current preds) skipped: %s", risk_exc)

                    compact = {
                        "symbol": predictions.get("symbol"),
                        "current_price": predictions.get("current_price"),
                        "generated_at": predictions.get("generated_at"),
                        "predictions": {},
                    }
                    for horizon in ["1d", "7d", "30d"]:
                        if horizon in predictions.get("predictions", {}):
                            entry = predictions["predictions"][horizon]
                            try:
                                cap_map = {"1d": 0.10, "7d": 0.20, "30d": 0.40}
                                capped_flag = abs(float(entry.get("price_change_pct", 0.0))) >= (
                                    cap_map.get(horizon, 1.0) - 1e-6
                                )
                            except Exception:
                                capped_flag = False
                            compact["predictions"][horizon] = {
                                "target_price": entry.get("target_price"),
                                "price_change_pct": entry.get("price_change_pct"),
                                "direction": entry.get("direction"),
                                "direction_prob": entry.get("direction_prob"),
                                "confidence": entry.get("confidence"),
                                "signal": entry.get("signal"),
                                "risk": entry.get("risk"),
                                "capped": capped_flag,
                            }
                    job_state["current_predictions"] = compact
                except Exception as pred_exc:
                    logger.warning("Champion current predictions failed: %s", pred_exc)
        except Exception as predictions_exc:
            logger.debug("Attach current predictions skipped: %s", predictions_exc)

        job_state.update({
            "status": "completed",
            "progress": 100,
            "current_step": "✅ Backtest completed successfully!",
            "eta_seconds": 0,
            "end_time": datetime.now(timezone.utc).isoformat(),
            "result": results,
        })

        duration = time.time() - start_time
        logger.info("✅ Backtest job %s completed for %s (%.1fs)", job_id, request.symbol, duration)

    except Exception as exc:
        logger.error("❌ Backtest job %s failed: %s", job_id, exc)
        job_state.update({
            "status": "failed",
            "progress": 0,
            "current_step": f"❌ Error: {exc}",
            "end_time": datetime.now(timezone.utc).isoformat(),
            "error": str(exc),
            "eta_seconds": 0,
        })


async def schedule_backtest_job(job_id: str, request: BaseModel) -> None:
    """Schedule ``run_backtest_job`` on the current event loop."""
    asyncio.create_task(run_backtest_job(job_id, request))
