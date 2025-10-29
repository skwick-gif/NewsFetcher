import sys
from pathlib import Path
from datetime import datetime
import subprocess
import threading


# Project root (…/)
project_root = Path(__file__).resolve().parent.parent.parent

# Shared job state
job_logs: dict[str, list[dict]] = {}
running_jobs: dict[str, dict] = {}


def log_job(job_id: str, level: str, msg: str) -> None:
    entry = {"timestamp": datetime.now().isoformat(), "level": level, "message": msg}
    job_logs.setdefault(job_id, []).append(entry)
    if len(job_logs[job_id]) > 400:
        job_logs[job_id] = job_logs[job_id][-400:]


def run_script_in_background(job_type: str, script_name: str, job_id: str) -> bool:
    """Run a Python script in the background and capture output."""
    job_logs[job_id] = []
    running_jobs[job_id] = {"status": "running", "start_time": datetime.now().isoformat(), "job_type": job_type}

    def _target():
        try:
            process = subprocess.Popen(
                [sys.executable, script_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=0,
                universal_newlines=True,
                cwd=str(project_root),
            )
            for line in process.stdout:
                line = (line or "").strip()
                if not line:
                    continue
                job_logs[job_id].append({"timestamp": datetime.now().isoformat(), "level": "INFO", "message": line})
                if len(job_logs[job_id]) > 200:
                    job_logs[job_id] = job_logs[job_id][-200:]
            process.wait()
            if process.returncode == 0:
                running_jobs[job_id]["status"] = "completed"
                job_logs[job_id].append({"timestamp": datetime.now().isoformat(), "level": "SUCCESS", "message": "✅ Job completed successfully!"})
            else:
                running_jobs[job_id]["status"] = "failed"
                job_logs[job_id].append({"timestamp": datetime.now().isoformat(), "level": "ERROR", "message": f"❌ Job failed with exit code {process.returncode}"})
        except Exception as e:
            running_jobs[job_id]["status"] = "error"
            job_logs[job_id].append({"timestamp": datetime.now().isoformat(), "level": "ERROR", "message": f"❌ Error: {str(e)}"})

    threading.Thread(target=_target, daemon=True).start()
    return True


def run_command_in_background(job_type: str, cmd_args: list[str], job_id: str) -> bool:
    """Run an arbitrary command (list of args) in the background and capture output."""
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
                cwd=str(project_root),
            )
            out_dir = None
            for line in process.stdout:
                line = (line or "").strip()
                if not line:
                    continue
                # Detect known output path hints
                if 'Portfolio evaluation report written to:' in line:
                    try:
                        out_dir = line.split('Portfolio evaluation report written to:')[-1].strip()
                    except Exception:
                        pass
                job_logs[job_id].append({"timestamp": datetime.now().isoformat(), "level": "INFO", "message": line})
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

    threading.Thread(target=_target, daemon=True).start()
    return True
