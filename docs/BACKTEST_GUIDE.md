# Progressive Backtesting Guide

This document explains how the Progressive Backtester works, how results are computed and displayed, how checkpoints are selected for predictions, common reasons accuracy may be zero in iterations, and how to stop a running job.

## What the backtester does

- Expanding-window training: it trains models on an initial train period [train_start_date..train_end_date], then evaluates on a test window ahead (test_period_days). After each iteration, it expands the train end to the end of the prior test window and repeats.
- Real predictions only: for each test day, it rebuilds a per-day data loader limited up to that date and asks the predictor to forecast the next day (1d) using the models trained in that iteration.
- Live progress: the server pushes iteration progress and ETA; the UI polls and shows a progress bar with status text.

## Metrics in the results table

- # (Iteration): ordinal number of the expanding-window iteration
- Train Until: last day included in training for that iteration
- Accuracy: direction accuracy over the test window (fraction of correct up/down signs) for 1-day horizon
- Loss: validation loss from the train phase (lower is better)
- Time: training time for that iteration (seconds)
- Status: marks the best iteration by highest accuracy

A chart shows Accuracy (%) and Loss across iterations to visualize improvement.

## Checkpoint selection for predictions

- During training, the trainer saves PyTorch checkpoints per model and horizon with the pattern: `{model_type}_{SYMBOL}_{HORIZON}_best.pth` (e.g., `lstm_INTC_1d_best.pth`). The file is overwritten each time training finds a new best.
- The predictor loads checkpoints for the requested symbol from its configured `model_dir` and uses best-by-file naming per horizon.
- In backtests, to avoid overwriting production checkpoints, each job writes into a job-specific folder: `app/ml/models/backtests/{job_id}`. The backtester temporarily points the predictor at that folder when evaluating.
- Ensemble weighting uses best observed validation loss from the training history CSVs if available; otherwise it falls back to configured base weights.

## Why accuracy can be 0 after the first iteration

A few common causes:
- No test data available in subsequent iterations (e.g., the chosen window pushes the test period beyond the available dates). The backtester returns accuracy 0.0 for that iteration and includes `note` or `error` fields when detectable.
- No predictions generated in the test window, often due to insufficient context length (sequence_length) for some test dates, or model load failures. The evaluation returns `predictions_made: 0` and accuracy 0.0.
- Newly trained checkpoints were not used by the predictor due to stale caches or incorrect model directory. The backtester now points the predictor to the job-specific `model_dir` and clears caches per iteration to force reload.

If you see persistent zeros, try:
- Shorter test_period_days or an earlier train_end_date, ensuring enough historical data exists
- Checking `stock_data/{SYMBOL}/{SYMBOL}_price.csv` date range
- Lowering sequence_length (default 60) in training settings and retraining

## Stopping a running backtest

- The Stop button requests cancellation via the server. The job will finish the current small step, then stop. The UI will show "Backtest cancelled" and hide the progress panel.
- Cancellation is cooperative; if you cancel during heavy compute, a brief delay is normal before it stops.

## Indicator parameters (customization)

- Current version uses default indicator windows (e.g., SMA/EMA [5,10,20,50], RSI period 14, MACD 12/26/9, etc.).
- Exposing these as editable settings in the UI and threading them into the `ProgressiveDataLoader` is planned. This will include validation and a saved presets mechanism.

## Next steps / tips

- Keep train periods realistic: at least `sequence_length + max(horizon) + ~30` days.
- Start with one or two model types (e.g., lstm, transformer) to reduce runtime.
- Use the chart to spot improvements; aim for a stable or rising accuracy and falling loss.
- For production predictions, the predictor uses the symbol’s latest "best" checkpoints in `app/ml/models`.
