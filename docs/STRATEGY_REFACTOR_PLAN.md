# Strategy Lab Refactor & Indicator Integration Plan

> **Status tracker legend**: `[ ]` not started · `[~]` in progress · `[x]` complete

---

## 1. Objectives

- [ ] Improve Strategy Lab UI usability: logical grouping, collapsible/accordion layout, default symbol change to `SPY`, stop auto-scrolling on rerender.
- [ ] Dynamically tailor input controls per strategy to avoid overwhelming users with irrelevant fields.
- [ ] Enable optional indicators to *modify* backtest logic (not just chart overlays) while keeping existing strategies intact.
- [ ] Establish a maintainable backend structure (strategies/filters/indicators) that supports future expansion and testing.
- [ ] Ensure metrics (e.g., win rate, total return, drawdown) remain accurate when new logic toggles are active.
- [ ] Prepare three baseline strategy templates now so we can rename/tune them later without blocking UI work.

---

## 2. Backend Workstream

### 2.1 Architecture & Folder Structure

- [ ] Create modular layout:
  - `app/strategies/` — each strategy class/module (e.g., `macd_cross.py`, `macd_pre_cross.py`).
  - `app/indicators/` — shared indicator computations (EMA, RSI, ATR, etc.).
  - `app/filters/` — reusable trade filters (e.g., RSI filter, low-volume filter).
  - `app/backtest/runner.py` — orchestrates execution, applies strategy + chosen filters.
  - `app/backtest/schema.py` — pydantic/dataclass definitions for request/response payloads.

- [ ] Seed three baseline strategy modules (rename later) so the pipeline is ready:
  - `app/strategies/trend_follow_basic.py`
  - `app/strategies/mean_reversion_basic.py`
  - `app/strategies/breakout_basic.py`

- [ ] Introduce registry/loader so API can assemble a strategy from config + selected filters.

### 2.2 API Adjustments (`/api/strategy/backtest`)

- [ ] Accept dynamic payload with fields like:
  ```json
  {
    "strategy_id": "macd_cross",
    "params": { ... },
    "filters": [
      {"id": "rsi", "params": {"period": 14, "lower": 30, "upper": 70}},
      {"id": "atr_stop", "params": {...}}
    ],
    "chart_overlays": {"show_rsi": true, ...}
  }
  ```

- [ ] Validate incoming payload considering strategy-specific schema.
- [ ] Pass filter configs into runner and ensure they update trade decisions (enter/exit/skip).

### 2.3 Strategy/Filter Implementation

- [ ] Base strategy class defines interface: `prepare()`, `generate_signals()`, `apply_filters()`, `simulate_trades()`.
- [ ] Filters expose `should_enter(context)` / `should_exit(context)` to hook into strategy execution.
- [ ] Indicators module computes arrays once per backtest and caches results for both charts and filters.
- [ ] Ensure filters can request indicator arrays (e.g., RSI) without recomputing.

### 2.4 Metrics & Reporting

- [ ] Centralize metric calculation in `app/backtest/metrics.py`.
- [ ] When filters alter trades, metrics auto-update because they consume final equity/trade list.
- [ ] Extend metrics to include filter-specific stats if desired (e.g., number of trades skipped by RSI).

### 2.5 Testing & Validation

- [ ] Unit tests per strategy + filter combo.
- [ ] Snapshot tests for API responses (baseline JSON files) to detect regressions.
- [ ] Backtest smoke tests covering default strategy and one filter-enabled scenario.

### 2.6 Current Backend Touchpoints

- `app/api/routers/scanner.py` remains the FastAPI entry point for Strategy Lab and scanner backtests. It will load the new registry modules once the skeletons above exist, keeping external clients unchanged while we refactor internals.

### 2.7 Baseline Strategy Templates

- `trend_follow_basic`: moving-average cross with configurable fast/slow windows and optional ATR trailing stop.
- `mean_reversion_basic`: RSI + Bollinger mean reversion with parameters for lookback, threshold bands, and max concurrent positions.
- `breakout_basic`: volatility squeeze + range breakout using Donchian/ATR levels, includes risk per trade and confirmation toggle.
- Each template ships with a minimal `StrategyConfig` schema and docstring so later renaming or extension is low-effort.

---

## 3. UI Workstream

### 3.1 Layout Refresh

- [ ] Replace single long form with grouped cards:
  1. **Symbol & Dates**
  2. **Strategy Parameters** (dynamic)
  3. **Risk & Execution**
  4. **Visual Overlays / Chart Styling** (static)

- [ ] Implement accordion/expander per card so user can collapse sections.
- [ ] Move indicator toggles that affect logic into the Strategy Parameters card; overlay-only controls stay in Visuals.
- [ ] Default symbol to `SPY` and remove auto-scroll after render.

### 3.2 Dynamic Fields per Strategy

- [ ] Fetch strategy schema (via new `/api/strategy/meta` or embedded JS object).
- [ ] Render fields based on schema; hide irrelevant controls instead of disabling.
- [ ] Display helper text/tooltips describing conditions.
- [ ] Provide quick preset buttons that auto-select strategy and populate defaults.

### 3.3 Indicator Integration Controls

- [ ] Add checkbox/list for “Active Filters” (e.g., RSI filter). Selecting one reveals its parameters.
- [ ] Ensure toggling a filter triggers a single rerun (debounced Apply button).
- [ ] Update status line to reflect filters (e.g., `Filters: RSI(14/30-70)`).

### 3.4 Playback & Status Updates

- [ ] Set playback starting pointer to the selected `end` date if provided (falling back to latest data available).
- [ ] When result data lacks the exact end date, show warning but display closest available date.
- [ ] Maintain existing timeline trimming so chart range matches runtime.

### 3.5 Accessibility & Styling

- [ ] Ensure keyboard navigation across accordions and toggles.
- [ ] Use consistent button sizes and spacing; add section headers with icons.
- [ ] Provide light/dark compatible palette per existing theme.

---

## 4. Data Flow Overview

1. **User selects strategy + filters + overlays**.
2. UI composes payload (`strategy_id`, `params`, `filters`, `date_range`).
3. Backend runner:
   - loads strategy module
   - computes required indicators
   - applies filters
   - simulates trades
   - returns trades, equity curve, indicator arrays, metrics, metadata (e.g., warmup bars).
4. UI renders chart + metrics and updates playback timeline.

---

## 5. Checklist & Timeline (example)

| Status | Task | Owner | Notes |
|--------|------|-------|-------|
| [ ] | Confirm modular folder layout & create skeleton | Backend |  
| [ ] | Implement indicator cache + filter hooks | Backend |  
| [ ] | Update `/api/strategy/backtest` schema | Backend |  
| [ ] | Build new Strategy schema endpoint | Backend |  
| [ ] | Refactor Strategy Lab template into cards | Frontend |  
| [ ] | Implement dynamic field renderer | Frontend |  
| [ ] | Adjust playback to honor end date | Frontend |  
| [ ] | Update default symbol to `SPY` | Frontend |  
| [ ] | Remove auto-scroll after render | Frontend |  
| [ ] | Create unit/integration tests | Backend |  
| [ ] | Document new filter options | Docs |  
| [ ] | Scaffold three baseline strategy files (trend, mean reversion, breakout) | Backend | Placeholder names, rename later |

*(Add more rows as tasks emerge.)*

---

## 6. Documentation & Communication

- [ ] Update `docs/SCANNER_DETAILED_OVERVIEW.md` equivalent for Strategy Lab (new doc).
- [ ] Provide how-to for adding a new indicator filter (step-by-step for developers).
- [ ] Update README/UI guides with screenshots post-refresh.

---

## 7. Open Questions for Alignment

- Which indicators should affect logic in v1 (RSI, ATR, Volume)?
- Should filters be mutually exclusive or can multiple combine (AND/OR logic)?
- Do we need backward compatibility mode for saved presets/URLs?
- Will strategy defaults come from YAML/DB or remain hardcoded? (Impacts dynamic schema delivery.)

---

## 8. Next Steps

1. Review this plan and finalize scope (choose initial indicator & strategy targets).
2. Lock folder structure + create skeleton modules (including the three new baseline strategies).
3. Implement backend schema/runner changes.
4. Revamp UI layout & dynamic rendering.
5. Integrate filters end-to-end and validate metrics.
6. Document the changes + release notes.
