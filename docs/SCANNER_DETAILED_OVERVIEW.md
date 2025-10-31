# Scanner Module – Detailed Architecture & Flow

_Last updated: 2025-10-31_

This document exhaustively describes the **Scanner** tab end-to-end: frontend UI behaviour, backend endpoints, background workers, data structures, and storage artefacts. Use it as the canonical reference before modifying the scanner.

---

## 1. High-Level User Flow

1. User opens `/scanner` in the Flask UI (served via `app/server.py`).
2. UI loads `app/templates/scanner/scanner.html` and bootstraps the JS dashboard.
3. The page:
   - Fetches filter status/results.
   - Polls training queue state.
   - Lets user trigger **Filter**, **Train** (single/all), and **Daily Scan** in ML/Technical modes.
4. UI actions proxy through Flask to the FastAPI backend (`app/main.py`) using the `app.routes.scanner` blueprint.
5. FastAPI router `app/api/routers/scanner.py` handles filtering, training queue, and scan execution.
6. Scan strategies live under `app/strategies/` and read from `stock_data/` + ML models in `app/ml/models/`.

---

## 2. Frontend (Flask Template + JS)

### 2.1 Template Entry Point
- **File:** `app/templates/scanner/scanner.html`
- Rendered by FastAPI (GET `/scanner`) in `app/main.py`.
- Presents four main cards:
  1. **Filter Status** (`Run Filter`, `Refresh`, `Train All`).
  2. **Training Queue** summary.
  3. **Scan Status** (`Mode` dropdown, `Run Daily Scan`, status poll controls).
  4. **Top Ranked (Latest)** results table.

### 2.2 JavaScript Responsibilities
- Immediately binds buttons and starts polling.
- **fetchJSON:** common helper to call `/api/scanner/...` routes (proxied via Flask, served by FastAPI backend).
- **Filter:**
  - `Run Filter` posts to `/api/scanner/filter/run`.
  - Periodic polling (`pollFilter`) hits `/api/scanner/filter/status`.
  - On completion, `loadFilter()` pulls `/api/scanner/filter/results` and renders the table.
- **Training:**
  - `Train` button posts `/api/scanner/train/start?symbol=...`.
  - `Train All` posts `/api/scanner/train/start-all`.
  - `pollTrain` fetches `/api/scanner/train/status` to display queue, update rows, and refresh badges per symbol (`pollSymbol`).
- **Scanning:**
  - `Run Daily Scan` posts to `/api/scanner/run?mode=ml|technical`.
  - `pollStatus` retrieves `/api/scanner/status` (overall state).
  - When status flips to `completed`, `loadTop()` fetches `/api/scanner/top` to render the ranked table (all results sorted; no artificial 50-row cap).
- **Auto-refresh intervals:** filter & train every 5 seconds, status every 6 seconds. Additional fast polling while filter job is running (1000ms).

### 2.3 Proxying Through Flask
- File: `app/routes/scanner.py`
- Each UI request (e.g., `/api/scanner/run`) is proxied to the FastAPI backend via `proxy_to_backend` (preserves method, query string, optional JSON body).
- Result: UI continues to use familiar `/api/…` paths while the heavy logic lives in the modular FastAPI app.

---

## 3. Backend (FastAPI Router)

### 3.1 Router Registration
- File: `app/api/routers/scanner.py`
- Included in FastAPI app (`app/main.py`) with tag `Scanner`.
- Exposes endpoints under `/api/scanner/...`.

### 3.2 Shared State & Helpers
- **State Class:** `ScannerState` tracks filter results, training queue, and last scan output.
  - Key fields: `filter_state`, `filter_items`, `train_queue`, `train_status`, `top` (latest scan results).
- **Utilities:**
  - `_iter_local_symbols` (now in `app/strategies/metrics.py`) enumerates directories under `stock_data/`.
  - `_compute_local_metrics` (also in `metrics.py`) loads price/fundamental data and calculates metrics used by scans.
  - `_model_126d_exists` checks `app/ml/models/transformer_{SYMBOL}_126d_best.pth`.
  - `_has_minimum_history` ensures enough data to train a 126-day model.
  - `_get_progressive_trainer()` lazy-initialises `ProgressiveTrainer` (singleton). Configured with 60 epochs, patience 10, etc.
  - `_get_progressive_predictor()` lazy-initialises `ProgressivePredictor` for inference.

### 3.3 Filter Workflow
- **Endpoint:** `POST /api/scanner/filter/run`
  - Launches `_run_filter_job` in a background thread (so async loop remains responsive).
  - Default thresholds: price ≥ $3, average dollar volume ≥ $1M, market cap ≥ $300M.
- **Job Logic (`_run_filter_job`):**
  - Iterates *all* local symbols (stock_data directories).
  - `_load_basic_symbol_snapshot` (inline helper) loads recent price snapshot & fundamentals.
  - If criteria pass: append to `_scanner_state.filter_items` with `symbol`, price, ADV, market cap, `has_model` flag.
  - Updates progress metrics and logs every 1000 symbols.
  - Upon completion, `_scanner_state.filter_state = 'completed'`.
- **Status Endpoint:** `GET /api/scanner/filter/status`
  - Returns counts, progress, passes, trained count. Also refreshes `has_model` flags from filesystem when idle.
- **Results Endpoint:** `GET /api/scanner/filter/results`
  - Returns the `filter_items` array (converted to JSON-safe types) and updates `trained_count`.

### 3.4 Training Queue
- **Entry:** `POST /api/scanner/train/start?symbol=…`
  - Validates symbol.
  - Ensures trainer is available.
  - Calls `_enqueue_training([symbol], force=True)` (force ensures retraining allowed even if model exists).
  - Returns queue status, skip rationale, etc.
- **Bulk:** `POST /api/scanner/train/start-all`
  - Collects untrained filter symbols.
  - Enqueues via `_enqueue_training(symbols)` (default `force=False` so existing checkpoints skip).
- **Queue Mechanics:**
  - `_enqueue_training` (with lock):
    - Resets training tracking if queue empty.
    - Skips duplicates, records reasons (`already-trained`, `insufficient-history`).
    - For `force=True`, even if model exists it re-queues (and clears `has_model` in filter list).
    - Adds to `_scanner_state.train_queue`, set status to `queued`, increments totals.
    - Starts worker task `_drain_training_queue()` if not running.
  - `_drain_training_queue`: sequentially `await _train_symbol_job(symbol, trainer)`. Updates success/failure states, `train_completed`, `train_failed_symbols`.
  - `_train_symbol_job`: runs `trainer.train_progressive_models(symbol, TRAIN_MODEL_TYPES)` in a thread. After success ensures checkpoint exists, flips filter item to `has_model=True`, logs `Epoch …` each iteration, calculates training duration.
- **Status Endpoint:** `GET /api/scanner/train/status`
  - Exposes queue length, completed count, failures, current symbol, and per-symbol status map for UI table.
- **Per Symbol Check:** `GET /api/scanner/train/symbol/{symbol}/status`
  - Reports `status` and `has_model` (synchronises with filesystem).

### 3.5 Scan Execution
- **Entry:** `POST /api/scanner/run?mode=ml|technical` (optional `limit`, default unlimited).
  - Spawns `_run_scan(mode, limit)` as an async task (or background task if FastAPI BackgroundTasks provided).
  - `_scanner_state.filter_state` temporarily set to `'running'` for UI feedback.
- **Workflow:**
  1. Build candidate list:
     - If filter has results: use only those symbols.
     - For ML mode: narrow further to symbols with `has_model=True`.
     - For Technical: use all filtered symbols.
     - If no filter data exists: fall back to full universe and log this.
  2. Dispatch to correct strategy via `asyncio.to_thread` to avoid blocking event loop.
  3. On success: `result['data']['stocks']` stored in `_scanner_state.top` (entire list sorted by strategy logic).
  4. Set `_scanner_state.filter_state = 'completed'` (used by UI to auto-refresh top table).
- **Status Endpoint:** `GET /api/scanner/status`
  - Derived from `ScannerState.to_status()` (idle/ready, scan completion).
- **Top Endpoint:** `GET /api/scanner/top?limit=`
  - Returns cached `top` list (optionally truncated by query param). Includes metadata (`total`, `returned`, `date`).
  - If `top` empty, falls back to `_get_hot_stocks_internal` (which itself scans the full universe once, used as warm data).

---

## 4. Strategy Modules

### 4.1 Shared Metrics Helper (`app/strategies/metrics.py`)
- `_iter_local_symbols(max_symbols=None)` enumerates uppercase directories in `stock_data/`.
- `_compute_local_metrics(symbol)`:
  - Loads `stock_data/{SYM}/{SYM}_price.csv`.
  - Computes recent price change, % change, average volume, dollar volume, simple momentum, expected return heuristic.
  - Attempts to load fundamentals from `{SYM}_advanced.json` for market cap, sector, industry.
  - Checks for trained model: existence of `transformer_{SYM}_126d_best.pth`, optionally queries `_get_progressive_predictor()` and runs `predict_ensemble()` to fill `ml_score`.
  - Invokes `_compute_convergence_score` (from same module) to calculate MACD convergence indicators (ADX, histogram, volume dryness, etc.).
  - Returns consolidated metrics dict used by both ML and technical scans.
- `_compute_convergence_score(symbol, df)` calculates 100-bar MACD/ADX-based scoring with flexible thresholds; returns score, filters, reason.

### 4.2 ML Scan (`app/strategies/ml_scan.py`)
- `scan_ml(limit=None, symbols=None)`:
  - Normalises symbol list (if provided) or scans full universe.
  - For each symbol, collects metrics (`_compute_local_metrics`).
  - Sorts by `expected_return` descending.
  - Returns **all** rows (`limit` optional) along with metadata: `total_scanned`, `total`, `returned`, human-readable criteria string.

### 4.3 Technical Scan (`app/strategies/technical_scan.py`)
- `scan_technical(limit=None, min_score=60.0, symbols=None)`:
  - Similar normalisation logic as ML.
  - Filters to metrics with `technical_score ≥ min_score` and `convergence_data.meets_criteria` true.
  - Sorts by `technical_score` descending.
  - Outputs all matches with metadata (`total`, `returned`, `total_scanned`, `criteria`).

---

## 5. Data & Storage Locations

| Artefact                       | Location / Format                                            | Notes |
|--------------------------------|---------------------------------------------------------------|-------|
| Price history CSV              | `stock_data/{SYM}/{SYM}_price.csv`                            | Required columns: `Open, High, Low, Close, Volume` sorted by date. |
| Fundamentals JSON              | `stock_data/{SYM}/{SYM}_advanced.json`                        | Must include `marketCap`, optionally `sector/industry`. |
| ML Model Checkpoints           | `app/ml/models/transformer_{SYM}_126d_best.pth`               | Produced by training queue. Only 126d models tracked in UI. |
| Training history CSV           | `app/ml/models/{model_name}_{horizon}_history.csv`            | Written per trained model by `ProgressiveTrainer`. |
| Logs                           | Console / configured logging handlers                         | Training logs show per-epoch progress, warnings, errors. |
| Filter snapshot                | In-memory (`_scanner_state.filter_items`)                     | Exposed via `/filter/results`. Reset when new filter run starts. |
| Scan results                   | In-memory (`_scanner_state.top`)                              | Overwritten on each scan run. `/top` returns current snapshot. |
| Training queue state           | In-memory (`_scanner_state` fields)                           | Exposed via `/train/status` and per-symbol status endpoint. |

---

## 6. Background Execution & Concurrency

| Component                     | Execution Context                        | Implementation Detail |
|-------------------------------|-------------------------------------------|-----------------------|
| Filter Job                    | Dedicated Python thread (`threading.Thread`) | Allows blocking disk IO without blocking FastAPI loop. |
| Training Runner               | Async worker task with `asyncio` + `to_thread` | Training per symbol executed via `asyncio.to_thread` inside `_train_symbol_job`. |
| ML/Technical Scans            | `asyncio.to_thread` within `_run_scan`     | Prevents strategy execution from blocking event loop. |
| Hot Stocks fallback scan      | Synchronous but triggered only when `/top` empty | Runs once per call; sorts entire universe (calls `_compute_local_metrics`). |

---

## 7. Endpoint Summary

| Route                                     | Method | Description |
|-------------------------------------------|--------|-------------|
| `/api/scanner/status`                     | GET    | High-level scanner state (`state: idle/ready`, etc.). |
| `/api/scanner/top`                        | GET    | Latest scan results (complete list, sorted). Optional `limit`. |
| `/api/scanner/run?mode=ml|technical`      | POST   | Start ML or technical scan (async). Optional `limit`. |
| `/api/scanner/filter/run`                 | POST   | Start filtering job (thread). |
| `/api/scanner/filter/status`              | GET    | Current filter progress, counts, trained tally. |
| `/api/scanner/filter/results`             | GET    | Filtered items (JSON-safe). |
| `/api/scanner/train/start?symbol=SYM`     | POST   | Queue training for one symbol (`force=True`). |
| `/api/scanner/train/start-all`            | POST   | Queue all filtered symbols lacking a 126d model. |
| `/api/scanner/train/status`               | GET    | Training queue overview. |
| `/api/scanner/train/symbol/{SYM}/status`  | GET    | Status + checkpoint existence for specific symbol. |
| `/api/scanner/hot-stocks`                 | GET    | (Aux) Fallback to compute top potentials from entire universe. |

---

## 8. Key Interactions

1. **UI → Filter:** `POST /filter/run` → thread executes `_run_filter_job` → `_scanner_state.filter_items` filled → UI polls `/filter/status` and `/filter/results`.
2. **UI → Train:** `POST /train/start` → `_enqueue_training` (force) → `_drain_training_queue` → `ProgressiveTrainer` saves models → `has_model` updates → UI polls `/train/status`.
3. **UI → Scan (ML):** `POST /run?mode=ml` → `_run_scan` builds candidates with `has_model=True` → `scan_ml` uses `_compute_local_metrics` and ML predictions → `_scanner_state.top` updated → UI auto-refreshes `/top`.
4. **UI → Scan (Technical):** Similar but candidate set = all filtered symbols; uses technical convergence score.
5. **Fallback `/top`:** If `_scanner_state.top` empty (fresh load), `_get_hot_stocks_internal` ensures UI still shows data by scanning all symbols once.

---

## 9. Notes & Maintenance Tips

- Always re-run filter before relying on training counts—`filter_items` hold the symbol universe for scans and queue calculations.
- Training queue uses `force=True` only for manual single-symbol requests; bulk training respects existing checkpoints to save time.
- If `stock_data` or `app/ml/models` directories change outside the scanner, fetch new filter results to refresh `has_model` flags.
- Long-running scans/training should be monitored in log output (`Epoch …`, warnings). Crash logs surface in whichever process runs FastAPI (`uvicorn`).
- For debugging UI, the browser console will show proxied endpoint failures (e.g., `ERR_CONNECTION_RESET`) when backend is down—restart FastAPI.
- The document reflects current behaviour (ML scan returns full scoreboard). If reintroducing limits, ensure `/top` and UI align.

---

## 10. Related Modules to Review

- `app/ml/progressive/trainer.py` – training loop, logging, model saving.
- `app/ml/progressive/predictor.py` – `Predictor.predict_ensemble` used for ML scores.
- `app/strategies/metrics.py` – single source of truth for local metrics & MACD convergence.
- `app/templates/scanner/scanner.html` – UI layout & JS actions.
- `app/routes/scanner.py` – Flask proxy shim.
- `app/main.py` – FastAPI app composition.

---

_End of document._
