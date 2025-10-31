from fastapi import APIRouter, HTTPException
from typing import Optional, Dict, Any
import logging
import threading
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import uuid

from app.core.config import logger

router = APIRouter()

# PPO Training jobs storage
PPO_TRAIN_JOBS: Dict[str, Dict[str, Any]] = {}

def _run_ppo_job(job_id: str, args: list[str], cwd: str) -> None:
    """Run PPO training as a subprocess and track status/logs in PPO_TRAIN_JOBS."""
    PPO_TRAIN_JOBS[job_id].update({
        "status": "running",
        "started_at": datetime.utcnow().isoformat() + "Z",
        "logs": [],
        "model_path": None,
    })
    try:
        proc = subprocess.Popen(args, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        PPO_TRAIN_JOBS[job_id]["pid"] = proc.pid
        # stream logs
        if proc.stdout is not None:
            for line in proc.stdout:
                line = line.rstrip("\n")
                logs = PPO_TRAIN_JOBS[job_id].get("logs", [])
                logs.append(line)
                # keep only last 200 lines
                PPO_TRAIN_JOBS[job_id]["logs"] = logs[-200:]
                if "Saved PPO model to:" in line:
                    # parse path
                    try:
                        path = line.split("Saved PPO model to:", 1)[1].strip()
                        PPO_TRAIN_JOBS[job_id]["model_path"] = path
                    except Exception:
                        pass
        ret = proc.wait()
        PPO_TRAIN_JOBS[job_id]["ended_at"] = datetime.utcnow().isoformat() + "Z"
        if ret == 0:
            PPO_TRAIN_JOBS[job_id]["status"] = "completed"
        else:
            PPO_TRAIN_JOBS[job_id]["status"] = "failed"
            PPO_TRAIN_JOBS[job_id]["returncode"] = ret
    except Exception as e:
        PPO_TRAIN_JOBS[job_id]["status"] = "failed"
        PPO_TRAIN_JOBS[job_id]["error"] = str(e)
        PPO_TRAIN_JOBS[job_id]["ended_at"] = datetime.utcnow().isoformat() + "Z"

@router.post("/rl/ppo/train")
async def rl_ppo_train(symbol: str, timesteps: int = 100000, window: int = 60,
                       start_date: Optional[str] = None, end_date: Optional[str] = None,
                       seed: int = 42):
    """Start PPO training in background using SB3; returns a job_id."""
    try:
        # Build command
        repo_root = Path(__file__).resolve().parents[2]  # Go up to NewsFetcher root
        script = repo_root / "rl" / "training" / "train_ppo.py"
        if not script.exists():
            raise FileNotFoundError(f"train_ppo.py not found at {script}")
        cmd = [sys.executable or "python", "-u", str(script), "--symbol", symbol, "--timesteps", str(int(timesteps)),
               "--window", str(int(window)), "--seed", str(int(seed))]
        if start_date:
            cmd.extend(["--start", start_date])
        if end_date:
            cmd.extend(["--end", end_date])
        job_id = uuid.uuid4().hex[:12]
        PPO_TRAIN_JOBS[job_id] = {"status": "pending", "cmd": cmd}
        t = threading.Thread(target=_run_ppo_job, args=(job_id, cmd, str(repo_root)), daemon=True)
        t.start()
        return {"status": "started", "job_id": job_id}
    except Exception as e:
        logger.error(f"Failed to start PPO training: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start PPO training: {str(e)}")

@router.get("/rl/ppo/plan")
async def rl_ppo_plan(symbol: str, window: Optional[int] = None) -> Dict[str, Any]:
    """Plan sensible training dates, window size, and timesteps from local data.

    Heuristics:
    - Start at max(first available date, 2020-01-01) if enough data, else earliest date
    - End at last available date
    - Window defaults to 60 if >=60 training days, else min( max(10, floor(days/3)), 60 )
    - Timesteps ~= clamp(days * 200, 50k..1,000k)
    """
    try:
        from rl.data_adapters.local_stock_data import LocalStockData
        import pandas as pd
        adapter = LocalStockData()
        if not adapter.has_symbol(symbol):
            raise HTTPException(status_code=404, detail=f"Local data for {symbol} not found")
        bundle = adapter.load_symbol(symbol)
        dfp = bundle.get("price")
        if dfp is None or dfp.empty:
            raise HTTPException(status_code=400, detail=f"No price data for {symbol}")
        # Ensure DateTimeIndex
        if not isinstance(dfp.index, pd.DatetimeIndex):
            try:
                dfp.index = pd.to_datetime(dfp.index)
            except Exception:
                pass
        dfp = dfp.sort_index()
        first_dt = dfp.index[0]
        last_dt = dfp.index[-1]
        # Prefer start >= 2020-01-01 when possible
        pref_start = pd.Timestamp(year=2020, month=1, day=1, tz=None)
        start_dt = pref_start if pref_start >= first_dt else first_dt
        # If the span from preferred start is too short (< 120 days), fallback to earliest
        if (last_dt - start_dt).days < 120:
            start_dt = first_dt
        # Compute training days (calendar index length in slice)
        df_train = dfp[(dfp.index >= start_dt) & (dfp.index <= last_dt)]
        days = int(len(df_train))
        if days < 2:
            # Not enough data to train
            raise HTTPException(status_code=400, detail=f"Insufficient training range for {symbol} (days={days})")
        # Choose window
        if window is None:
            win = 60 if days >= 60 else int(max(10, min(60, days // 3)))
        else:
            win = int(max(10, min(window, 300)))
        # Estimate timesteps
        est_steps = int(days * 200)
        timesteps = int(max(50_000, min(1_000_000, est_steps)))
        plan = {
            "symbol": symbol.upper(),
            "train_start_date": start_dt.date().isoformat(),
            "train_end_date": last_dt.date().isoformat(),
            "window": int(win),
            "timesteps": int(timesteps),
            "training_days": days,
            "first_data_date": first_dt.date().isoformat(),
            "last_data_date": last_dt.date().isoformat(),
        }
        return {"status": "planned", "plan": plan}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to plan PPO training: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to plan PPO training: {str(e)}")

@router.get("/rl/ppo/train/status")
async def rl_ppo_train_status_all():
    """Return status of all PPO jobs (summary)."""
    out = {}
    for jid, info in PPO_TRAIN_JOBS.items():
        out[jid] = {
            "status": info.get("status"),
            "started_at": info.get("started_at"),
            "ended_at": info.get("ended_at"),
            "model_path": info.get("model_path"),
        }
    return out

@router.get("/rl/ppo/train/status/{job_id}")
async def rl_ppo_train_status(job_id: str):
    """Return detailed status of a specific PPO job, including last logs."""
    info = PPO_TRAIN_JOBS.get(job_id)
    if not info:
        raise HTTPException(status_code=404, detail="job_id not found")
    return {
        "status": info.get("status"),
        "started_at": info.get("started_at"),
        "ended_at": info.get("ended_at"),
        "model_path": info.get("model_path"),
        "logs": info.get("logs", []),
        "pid": info.get("pid"),
        "returncode": info.get("returncode"),
        "error": info.get("error"),
    }