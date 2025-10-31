from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path
import uuid
import threading
import subprocess
import sys
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

# Global for PPO training jobs
PPO_TRAIN_JOBS: Dict[str, Dict[str, Any]] = {}

# Helper functions for data management
def _get_next_daily_run():
    """Calculate next daily run time (17:10)"""
    try:
        now = datetime.now()
        next_run = now.replace(hour=17, minute=10, second=0, microsecond=0)
        if next_run <= now:
            next_run += timedelta(days=1)
        return next_run
    except:
        return None

def _get_next_weekly_run():
    """Calculate next weekly run time (Sunday 2:00 AM)"""
    try:
        now = datetime.now()
        days_until_sunday = (6 - now.weekday()) % 7
        if days_until_sunday == 0 and now.hour >= 2:
            days_until_sunday = 7
        next_run = now + timedelta(days=days_until_sunday)
        next_run = next_run.replace(hour=2, minute=0, second=0, microsecond=0)
        return next_run
    except:
        return None

async def run_daily_scan():
    """Background task to run daily stock scan"""
    try:
        import subprocess
        import sys

        logger.info("🚀 Starting daily stock scan...")

        # Run the daily scan script (canonical under app/data)
        result = subprocess.run([
            sys.executable, "daily_scan.py"
        ], capture_output=True, text=True, cwd="app/data")

        if result.returncode == 0:
            logger.info("✅ Daily scan completed successfully")
        else:
            logger.error(f"❌ Daily scan failed: {result.stderr}")
            raise Exception(f"Daily scan failed: {result.stderr}")

    except Exception as e:
        logger.error(f"Error in daily scan background task: {e}")

async def run_weekly_fundamentals():
    """Background task to run weekly fundamentals update"""
    try:
        import subprocess
        import sys

        logger.info("🚀 Starting weekly fundamentals update...")

        # Run the weekly fundamentals script via canonical downloader in app/data
        result = subprocess.run([
            sys.executable, "download_fundamentals.py"
        ], capture_output=True, text=True, cwd="app/data")

        if result.returncode == 0:
            logger.info("✅ Weekly fundamentals update completed successfully")
        else:
            logger.error(f"❌ Weekly fundamentals update failed: {result.stderr}")
            raise Exception(f"Weekly fundamentals update failed: {result.stderr}")

    except Exception as e:
        logger.error(f"Error in weekly fundamentals background task: {e}")

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

@router.get("/data-management", response_class=HTMLResponse)
async def data_management_page():
    """Data Management Dashboard - serve data_management.html"""
    from pathlib import Path

    dashboard_path = Path(__file__).parent.parent / "templates" / "data_management.html"

    try:
        with open(dashboard_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        return HTMLResponse(
            content="<h1>Data Management dashboard not found.</h1>",
            status_code=500
        )

@router.get("/api/data-management/status")
async def get_data_management_status():
    """Get current status of data download jobs"""
    try:
        from datetime import datetime
        import subprocess
        import json
        from pathlib import Path

        # Check if scheduler jobs exist and get their status
        status = {
            "daily_downloads": 0,
            "weekly_updates": 0,
            "error_count": 0,
            "last_run": None,
            "jobs": []
        }

        # Check log files for recent activity
        logs_dir = Path("logs")
        if logs_dir.exists():
            # Count daily scans from today
            today_daily_logs = list(logs_dir.glob(f"daily_scan_{datetime.now().strftime('%Y%m%d')}*.log"))
            status["daily_downloads"] = len(today_daily_logs)

            # Count weekly fundamentals from this week
            this_week = datetime.now().strftime('%Y%m')
            weekly_logs = list(logs_dir.glob(f"weekly_fundamentals_{this_week}*.log"))
            status["weekly_updates"] = len(weekly_logs)

            # Find most recent log
            all_logs = list(logs_dir.glob("*.log"))
            if all_logs:
                latest_log = max(all_logs, key=lambda x: x.stat().st_mtime)
                status["last_run"] = datetime.fromtimestamp(latest_log.stat().st_mtime).isoformat()

        # Live-only: do not return mocked job statuses. If a real scheduler is integrated,
        # this endpoint should reflect its state. For now, return only log-derived info
        # and an empty jobs list with a note.
        status["jobs"] = []
        status["note"] = "Scheduler integration not available; returning logs-derived status only."

        return JSONResponse(content=status)

    except Exception as e:
        logger.error(f"Error getting data management status: {e}")
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )

@router.post("/api/data-management/run-job/{job_type}")
async def run_data_job(job_type: str, background_tasks: BackgroundTasks):
    """Run a specific data download job"""
    try:
        if job_type == "daily":
            # Run daily stock data scan
            background_tasks.add_task(run_daily_scan)
            return JSONResponse(
                content={
                    "success": True,
                    "message": "Daily scan started",
                    "job_id": f"daily_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                }
            )
        elif job_type == "weekly":
            # Run weekly fundamentals update
            background_tasks.add_task(run_weekly_fundamentals)
            return JSONResponse(
                content={
                    "success": True,
                    "message": "Weekly fundamentals update started",
                    "job_id": f"weekly_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                }
            )
        else:
            return JSONResponse(
                content={"success": False, "error": f"Unknown job type: {job_type}"},
                status_code=400
            )

    except Exception as e:
        logger.error(f"Error running job {job_type}: {e}")
        return JSONResponse(
            content={"success": False, "error": str(e)},
            status_code=500
        )

@router.get("/api/data-management/logs/{job_type}")
async def get_job_logs(job_type: str):
    """Get logs for a specific job type"""
    try:
        from pathlib import Path

        logs_dir = Path("logs")
        if not logs_dir.exists():
            return JSONResponse(content={"logs": []})

        # Get logs based on job type
        if job_type == "daily":
            log_files = list(logs_dir.glob("daily_scan_*.log"))
        elif job_type == "weekly":
            log_files = list(logs_dir.glob("weekly_fundamentals_*.log"))
        else:
            log_files = list(logs_dir.glob("*.log"))

        # Get the most recent log file
        if not log_files:
            return JSONResponse(content={"logs": ["No logs found for this job type."]})

        latest_log = max(log_files, key=lambda x: x.stat().st_mtime)

        # Read log content
        with open(latest_log, 'r', encoding='utf-8') as f:
            log_content = f.read()

        return JSONResponse(content={
            "logs": log_content.split('\n'),
            "file": str(latest_log),
            "timestamp": datetime.fromtimestamp(latest_log.stat().st_mtime).isoformat()
        })

    except Exception as e:
        logger.error(f"Error reading logs for {job_type}: {e}")
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )

@router.get("/api/system/health")
async def system_health():
    """System health check endpoint"""
    try:
        from app.core.config import SCHEDULER_AVAILABLE

        health_status = {
            "database": True,  # Always true for now (SQLite)
            "api": True,       # If we're responding, API is working
            "scheduler": SCHEDULER_AVAILABLE,
            "storage": True,   # Check if we can write to logs directory
            "timestamp": datetime.now().isoformat()
        }

        # Check storage (try to write a temp file)
        try:
            from pathlib import Path
            logs_dir = Path("logs")
            logs_dir.mkdir(exist_ok=True)
            test_file = logs_dir / "health_check.tmp"
            test_file.write_text("test")
            test_file.unlink()
        except:
            health_status["storage"] = False

        return JSONResponse(content=health_status)

    except Exception as e:
        return JSONResponse(
            content={
                "database": False,
                "api": False,
                "scheduler": False,
                "storage": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            },
            status_code=500
        )

# ============================================================
# Dashboard Endpoint
# ============================================================
@router.get("/", response_class=HTMLResponse)
@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """Main dashboard - serve existing dashboard.html from templates"""
    from pathlib import Path

    # Try to read the dashboard.html file from templates
    dashboard_path = Path(__file__).parent.parent / "templates" / "dashboard.html"

    try:
        with open(dashboard_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        # Fallback - return error message
        return HTMLResponse(
            content="<h1>Dashboard file not found. Please ensure dashboard.html exists in app/templates/</h1>",
            status_code=500
        )

# ============================================================
# Progressive ML Guide (HTML)
# ============================================================
@router.get("/docs/progressive-ml", response_class=HTMLResponse)
async def progressive_ml_guide():
    """Serve the Progressive ML Guide page from templates/docs."""
    from pathlib import Path

    guide_path = Path(__file__).parent.parent / "templates" / "docs" / "progressive_ml_guide.html"
    try:
        with open(guide_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        return HTMLResponse(
            content="<h1>Guide file not found. Ensure app/templates/docs/progressive_ml_guide.html exists.</h1>",
            status_code=500
        )

# ============================================================
# RL Dashboard and Minimal API
# ============================================================
@router.get("/rl", response_class=HTMLResponse)
async def rl_dashboard_page():
    """Serve minimal RL dashboard page"""
    from pathlib import Path

    page_path = Path(__file__).parent.parent / "templates" / "rl_dashboard.html"
    try:
        with open(page_path, "r", encoding="utf-8") as f:
            html = f.read()
        return HTMLResponse(content=html)
    except FileNotFoundError:
        return HTMLResponse(content="<h3>RL Dashboard</h3><p>Page not found.</p>", status_code=404)

@router.get("/api/rl/status")
async def rl_status():
    """Minimal RL status placeholder (safe)."""
    return {
        "status": "idle",
        "positions": [],
        "pnl": 0.0,
        "decisions": []
    }

@router.get("/api/rl/simulate")
async def rl_simulate(symbol: str, days: int = 250, window: int = 60, policy: str = "follow_trend",
                      start_date: Optional[str] = None, end_date: Optional[str] = None):
    """Run a lightweight simulation on local stock_data for a symbol and return series for plotting."""
    try:
        from rl.simulation import run_simulation
        # If explicit dates are provided, prefer them over days
        sd = start_date if start_date else None
        ed = end_date if end_date else None
        use_days = None if (sd and ed) else max(30, int(days))
        result = run_simulation(
            symbol=symbol,
            days=use_days,
            window=max(10, int(window)),
            policy=policy,
            start_date=sd,
            end_date=ed,
        )
        return {"status": "success", "data": result}
    except FileNotFoundError as e:
        return {"status": "error", "message": str(e)}
    except ValueError as e:
        # Surface validation issues (e.g., insufficient data/window) to the UI
        return {"status": "error", "message": str(e)}
    except Exception as e:
        logger.error(f"RL simulate failed: {e}")
        return {"status": "error", "message": f"Simulation failed: {str(e)}"}

@router.get("/api/rl/simulate/plan")
async def rl_simulate_plan(symbol: str) -> Dict[str, Any]:
    """Plan sensible dates/days and window for Quick Simulation from local data.

    Heuristics:
    - Use last N=250 trading rows if available; else use all
    - start_date = index[-N], end_date = last index
    - window = min(60, max(10, floor(N/4)))
    - days = N
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
        if not isinstance(dfp.index, pd.DatetimeIndex):
            try:
                dfp.index = pd.to_datetime(dfp.index)
            except Exception:
                pass
        dfp = dfp.sort_index()
        total = len(dfp)
        if total < 2:
            raise HTTPException(status_code=400, detail=f"Insufficient data for {symbol}")
        N = 250 if total >= 250 else total
        start_dt = dfp.index[-N]
        end_dt = dfp.index[-1]
        window = int(min(60, max(10, N // 4)))
        plan = {
            "symbol": symbol.upper(),
            "start_date": start_dt.date().isoformat(),
            "end_date": end_dt.date().isoformat(),
            "window": window,
            "days": int(N),
            "first_data_date": dfp.index[0].date().isoformat(),
            "last_data_date": end_dt.date().isoformat(),
            "total_rows": int(total)
        }
        return {"status": "planned", "plan": plan}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to plan Quick Simulation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to plan Quick Simulation: {str(e)}")

# ============================================================
# RL PPO Training (SB3) — background job runner
# ============================================================

@router.post("/__deprecated/api/rl/ppo/train")
async def rl_ppo_train(symbol: str, timesteps: int = 100000, window: int = 60,
                       start_date: Optional[str] = None, end_date: Optional[str] = None,
                       seed: int = 42):
    """Start PPO training in background using SB3; returns a job_id."""
    try:
        # Build command
        repo_root = Path(__file__).resolve().parents[2]
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

@router.get("/__deprecated/api/rl/ppo/plan")
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

@router.get("/__deprecated/api/rl/ppo/train/status")
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

@router.get("/__deprecated/api/rl/ppo/train/status/{job_id}")
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
