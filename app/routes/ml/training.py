"""
ML Training Router
Handles ML model training and backtesting operations
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from datetime import datetime
from typing import Dict, Any, List
import logging
import asyncio
import uuid

from app.core.config import (
    logger,
    PROGRESSIVE_ML_AVAILABLE,
    progressive_trainer,
    progressive_predictor,
    data_manager
)

router = APIRouter()

# Global training jobs tracking
training_jobs = {}

@router.post("/train/{symbol}")
async def train_ml_model(symbol: str, days_back: int = 365):
    """Train ML models for a specific symbol"""
    try:
        if not PROGRESSIVE_ML_AVAILABLE or not progressive_trainer:
            raise HTTPException(status_code=503, detail="Progressive ML trainer not available")

        training_result = progressive_trainer.train_progressive_models(symbol, model_types=["lstm", "cnn", "transformer"])
        return {
            "status": "success",
            "symbol": symbol,
            "training_result": training_result,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Failed to train models for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to train models for {symbol}")

@router.post("/train/progressive/{symbol}")
async def start_progressive_training(
    background_tasks: BackgroundTasks,
    symbol: str,
    model_types: str = "lstm",
    mode: str = "progressive"
):
    """Start progressive training for a stock symbol (async with progress tracking)"""
    try:
        if not PROGRESSIVE_ML_AVAILABLE or not progressive_trainer:
            raise HTTPException(status_code=503, detail="Progressive ML trainer not available")

        # Parse model_types from comma-separated string to list
        model_types_list = [mt.strip() for mt in model_types.split(',') if mt.strip()]
        if not model_types_list:
            model_types_list = ["lstm"]

        logger.info(f"🚀 Starting progressive training for {symbol}: {model_types_list}, mode={mode}")

        # Ensure data is ready before starting (price/indicators/sentiment/fundamentals)
        training_jobs.clear()  # keep one job at a time clarity
        ensure_summary = await asyncio.to_thread(data_manager.ensure_symbol_data, symbol)
        logger.info(f"📦 Ensure data summary for {symbol}: {ensure_summary.to_dict()}")

        # Generate unique job ID
        job_id = f"train_{symbol}_{uuid.uuid4().hex[:8]}"

        # Initialize job tracking
        training_jobs[job_id] = {
            "job_id": job_id,
            "symbol": symbol,
            "model_types": model_types_list,
            "mode": mode,
            "status": "starting",
            "progress": 0,
            "current_step": "Initializing...",
            "eta_seconds": None,
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "result": None,
            "error": None
        }

        # Start training in background
        background_tasks.add_task(
            run_training_job,
            job_id=job_id,
            symbol=symbol,
            model_types=model_types_list,
            mode=mode
        )

        return {
            "status": "training_started",
            "job_id": job_id,
            "symbol": symbol,
            "model_types": model_types_list,
            "mode": mode,
            "message": "Training started in background. Use job_id to track progress.",
            "timestamp": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to start progressive training: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to start progressive training: {str(e)}")

@router.get("/train/status/{job_id}")
async def get_training_job_status(job_id: str):
    """Get status of specific training job"""
    try:
        if job_id in training_jobs:
            # Return job data directly (frontend expects status, progress, etc. at root level)
            job_data = training_jobs[job_id].copy()
            job_data["timestamp"] = datetime.now().isoformat()
            return job_data

        # No job in memory and no files found
        raise HTTPException(status_code=404, detail=f"Training job {job_id} not found")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get training status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get training status")

@router.get("/train/status")
async def get_all_training_status():
    """Get status of all training jobs"""
    try:
        if not PROGRESSIVE_ML_AVAILABLE or not progressive_trainer:
            raise HTTPException(status_code=503, detail="Progressive ML trainer not available")

        # Return all jobs
        active_jobs = [job for job in training_jobs.values() if job["status"] in ["starting", "running"]]

        return {
            "status": "success",
            "trainer_available": True,
            "is_training": len(active_jobs) > 0,
            "active_jobs": active_jobs,
            "total_jobs": len(training_jobs),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Failed to get training status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get training status")

async def run_training_job(job_id: str, symbol: str, model_types: List[str], mode: str):
    """Background task to run training with progress tracking"""
    import time
    import asyncio
    import sys
    import subprocess
    import threading
    import uuid
    from datetime import datetime

    # Explicitly disable Torch Dynamo before any training begins
    try:
        import torch  # noqa: F401
        try:
            import torch._dynamo as _dynamo  # type: ignore
            try:
                _dynamo.reset()
            except Exception:
                pass
            try:
                _dynamo.disable()
                logger.info("🛡️ Torch Dynamo disabled in training job")
            except Exception:
                logger.debug("Torch Dynamo disable not available in training job")
        except Exception as dynamo_e:
            logger.debug(f"Torch Dynamo module not present: {dynamo_e}")
    except Exception:
        pass

    try:
        training_jobs[job_id]["status"] = "running"
        training_jobs[job_id]["progress"] = 10
        training_jobs[job_id]["current_step"] = f"Ensuring data for {symbol}..."

        start_time = time.time()

        # Ensure data presence before training
        try:
            ensure = await asyncio.to_thread(data_manager.ensure_symbol_data, symbol)
            training_jobs[job_id]["current_step"] = "Loading training data..."
            training_jobs[job_id]["progress"] = 15
        except Exception as e:
            training_jobs[job_id].update({
                "status": "failed",
                "current_step": f"❌ Ensure data failed: {e}",
                "error": str(e),
            })
            return

        # Start a background task to update progress periodically
        async def simulate_progress():
            # Estimate: ~30 seconds per model per horizon (3 horizons = 90 sec per model)
            estimated_duration = len(model_types) * 90  # seconds
            update_interval = 3  # Update every 3 seconds

            for progress in range(15, 90, 5):  # 15% to 85%
                await asyncio.sleep(update_interval)

                if training_jobs[job_id]["status"] != "running":
                    break

                elapsed = time.time() - start_time
                remaining = max(0, estimated_duration - elapsed)

                # Calculate which model we're on
                model_progress = int((progress - 15) / 75 * len(model_types))
                current_model = model_types[min(model_progress, len(model_types) - 1)]

                # Determine horizon based on progress within model
                horizons = ['1d', '7d', '30d']
                horizon_idx = int((progress % 25) / 8)  # Cycles through horizons
                current_horizon = horizons[min(horizon_idx, 2)]

                training_jobs[job_id].update({
                    "progress": progress,
                    "current_step": f"Training {current_model.upper()} model ({current_horizon} horizon)...",
                    "eta_seconds": int(remaining)
                })

        # Start progress simulation
        progress_task = asyncio.create_task(simulate_progress())

        # Update to training phase
        training_jobs[job_id]["current_step"] = f"Training {len(model_types)} model(s) on 3 horizons..."
        training_jobs[job_id]["progress"] = 15

        # Run actual training (blocking)
        await asyncio.sleep(0.1)  # Small delay to ensure progress updates start

        if mode == "progressive":
            result = progressive_trainer.train_progressive_models(
                symbol=symbol,
                model_types=model_types
            )
        else:
            result = progressive_trainer.train_unified_models(
                symbol=symbol,
                model_types=model_types
            )

        # Cancel progress simulation
        progress_task.cancel()

        # Complete
        training_jobs[job_id].update({
            "status": "completed",
            "progress": 100,
            "current_step": "✅ Training completed successfully!",
            "end_time": datetime.now().isoformat(),
            "result": result,
            "eta_seconds": 0
        })

        logger.info(f"✅ Training job {job_id} completed successfully for {symbol}")

    except Exception as e:
        logger.error(f"❌ Training job {job_id} failed: {e}")
        training_jobs[job_id].update({
            "status": "failed",
            "progress": 0,
            "current_step": f"❌ Error: {str(e)}",
            "end_time": datetime.now().isoformat(),
            "error": str(e),
            "eta_seconds": 0
        })