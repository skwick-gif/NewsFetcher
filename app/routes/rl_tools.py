import sys
from pathlib import Path
from datetime import datetime
import threading
from flask import Blueprint, request, jsonify

from app.config.runtime import get_backend_base_url
from app.services.jobs import job_logs, running_jobs, log_job, run_command_in_background

import requests


rl_tools_bp = Blueprint('rl_tools', __name__)

# Project root
project_root = Path(__file__).resolve().parent.parent.parent


def _try_fetch_articles(start_iso: str, end_iso: str, symbols: list[str] | None) -> list[dict]:
    items: list[dict] = []
    try:
        params = {"start": start_iso, "end": end_iso}
        if symbols:
            params["symbols"] = ",".join(symbols)
        base = get_backend_base_url().rstrip('/')
        r = requests.get(f"{base}/api/articles/query", params=params, timeout=20)
        if r.status_code == 200:
            j = r.json()
            if isinstance(j, dict) and "items" in j and isinstance(j["items"], list):
                return j["items"]
            if isinstance(j, list):
                return j
    except Exception:
        pass
    try:
        base = get_backend_base_url().rstrip('/')
        r = requests.get(f"{base}/api/articles/by-date", params={"start": start_iso, "end": end_iso}, timeout=20)
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
            base = get_backend_base_url().rstrip('/')
            r = requests.get(f"{base}/api/articles/recent", params={"limit": 10000}, timeout=20)
            if r.status_code == 200:
                j = r.json()
                if isinstance(j, dict) and "items" in j and isinstance(j["items"], list):
                    items = j["items"]
                elif isinstance(j, list):
                    items = j
        except Exception:
            pass
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
    import pandas as pd
    import numpy as np
    rows: list[dict] = []
    if not articles:
        return rows
    norm = []
    for it in articles:
        ts = it.get("published") or it.get("published_at") or it.get("Date") or it.get("timestamp")
        try:
            dt = pd.to_datetime(ts).normalize()
        except Exception:
            continue
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
    def has_kw(txt: str, kws: list[str]) -> bool:
        return any(k in txt for k in kws)
    df["fda_hit"] = df["title_text"].apply(lambda t: bool(has_kw(t, ["fda", "drug", "trial"])) )
    df["china_hit"] = df["title_text"].apply(lambda t: bool(has_kw(t, ["china", "beijing", "cpc"])) )
    df["geopol_hit"] = df["title_text"].apply(lambda t: bool(has_kw(t, ["war", "geopolit", "sanction", "tariff"])) )
    agg = df.groupby(["Date", "Symbol"]).agg(
        news_count=("Symbol", "count"),
        sentiment_avg=("sentiment", "mean"),
        avg_score=("score", "mean"),
        fda_count=("fda_hit", "sum"),
        china_count=("china_hit", "sum"),
        geopolitics_count=("geopol_hit", "sum"),
    ).reset_index()
    agg["llm_relevant_count"] = 0
    if "avg_score" in agg.columns:
        agg["llm_relevant_count"] = (agg["avg_score"].fillna(0.0) > 0.0).astype(int)
    agg["Date"] = pd.to_datetime(agg["Date"]).dt.tz_localize(None) + pd.Timedelta(days=1)
    today = pd.Timestamp.now().normalize()
    agg = agg[agg["Date"] <= today]
    for c in ["avg_score", "sentiment_avg"]:
        if c in agg.columns:
            agg[c] = agg[c].fillna(0.0)
    cols = ["Date","Symbol","news_count","llm_relevant_count","avg_score","fda_count","china_count","geopolitics_count","sentiment_avg"]
    agg = agg[cols].sort_values(["Symbol","Date"]).reset_index(drop=True)
    rows = agg.to_dict(orient="records")
    return rows


def _export_news_features_job(symbols: list[str], start: str, end: str, refresh: bool, job_id: str):
    log_job(job_id, 'INFO', f"Exporting news features for {','.join(symbols)} {start}->{end} (T-1)")
    try:
        articles = _try_fetch_articles(start, end, symbols)
        log_job(job_id, 'INFO', f"Fetched {len(articles)} articles from backend")
    except Exception as e:
        running_jobs[job_id]["status"] = "failed"
        log_job(job_id, 'ERROR', f"Article fetch failed: {e}")
        return
    try:
        rows = _compute_news_features_rows(articles, symbols)
        log_job(job_id, 'INFO', f"Aggregated into {len(rows)} (Date,Symbol) rows (after T-1 shift)")
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
            key = df_new[['Date','Symbol']].drop_duplicates()
            merged = df_old.merge(key.assign(_rm=1), on=['Date','Symbol'], how='left')
            df_old = merged[merged['_rm'].isna()].drop(columns=['_rm'])
        df_all = pd.concat([df_old, df_new], ignore_index=True)
        df_all = df_all.dropna(subset=['Date','Symbol']).drop_duplicates(subset=['Date','Symbol'], keep='last').sort_values(['Symbol','Date'])
        df_all.to_csv(nf_path, index=False)
        log_job(job_id, 'SUCCESS', f"Wrote {len(df_all)} total rows to {nf_path}")
        running_jobs[job_id]["status"] = "completed"
    except Exception as e:
        running_jobs[job_id]["status"] = "failed"
        log_job(job_id, 'ERROR', f"Export failed: {e}")


@rl_tools_bp.route('/api/rl/news/export', methods=['POST'])
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
    threading.Thread(target=_export_news_features_job, args=(symbols, start, end, refresh, job_id), daemon=True).start()
    return jsonify({'status':'started','job_id': job_id}), 202


@rl_tools_bp.route('/api/rl/news/export/status/<job_id>')
def api_rl_news_export_status(job_id):
    info = running_jobs.get(job_id)
    logs = job_logs.get(job_id, [])
    if not info:
        return jsonify({'status':'error','detail':'unknown job_id'}), 404
    return jsonify({'status': info.get('status'), 'job_id': job_id, 'logs': [l.get('message') for l in logs]})


@rl_tools_bp.route('/api/rl/portfolio/evaluate', methods=['POST'])
def api_rl_portfolio_evaluate():
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


@rl_tools_bp.route('/api/rl/portfolio/evaluate/status/<job_id>')
def api_rl_portfolio_evaluate_status(job_id):
    info = running_jobs.get(job_id)
    logs = job_logs.get(job_id, [])
    if not info:
        return jsonify({"status": "error", "detail": "unknown job_id"}), 404
    resp = {"status": info.get("status"), "job_id": job_id, "logs": [l.get("message") for l in logs]}
    out_dir = info.get("out_dir")
    if out_dir:
        resp["out_dir"] = out_dir
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


@rl_tools_bp.route('/api/rl/portfolio/walkforward', methods=['POST'])
def api_rl_portfolio_walkforward():
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


@rl_tools_bp.route('/api/rl/portfolio/walkforward/status/<job_id>')
def api_rl_portfolio_walkforward_status(job_id):
    info = running_jobs.get(job_id)
    logs = job_logs.get(job_id, [])
    if not info:
        return jsonify({"status": "error", "detail": "unknown job_id"}), 404
    resp = {"status": info.get("status"), "job_id": job_id, "logs": [l.get("message") for l in logs]}
    out_dir = info.get("out_dir")
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
