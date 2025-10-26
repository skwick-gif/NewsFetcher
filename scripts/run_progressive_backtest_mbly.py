import sys
import json
from pathlib import Path
import time
import pandas as pd

# Ensure project root on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.data.data_manager import DataManager
from app.ml.progressive.data_loader import ProgressiveDataLoader
from app.ml.progressive.trainer import ProgressiveTrainer
from app.ml.progressive.predictor import ProgressivePredictor
from app.ml.progressive.backtester import ProgressiveBacktester


def main():
    symbol = "MBLY"
    print(f"=== Progressive ML backtest (regular) for {symbol} ===", flush=True)

    # Ensure data exists
    dm = DataManager()
    ensure = dm.ensure_symbol_data(symbol)
    print("Ensure summary:", json.dumps(ensure.to_dict(), indent=2), flush=True)

    # Data loader default
    loader = ProgressiveDataLoader(stock_data_dir=str(dm.stock_data_dir))

    # Derive a reasonable training window and backtest plan
    df = loader.load_stock_data(symbol)
    if df is None or len(df.index) < 250:
        print("Not enough data for backtest; need at least ~250 rows.")
        return

    df = df.sort_index()
    last_date = pd.to_datetime(df.index.max())

    # Backtest plan: 10-day windows, 3 iterations (adjust automatically if not enough days)
    test_period_days = 10
    max_iterations = 3

    # Compute a train_end_date leaving room for the planned iterations
    buffer_days = test_period_days * max_iterations + 1
    train_end_date = (last_date - pd.Timedelta(days=buffer_days)).date().isoformat()

    # 360-day training window if possible
    train_start_date = (pd.to_datetime(train_end_date) - pd.Timedelta(days=360)).date().isoformat()

    print(f"Backtest plan: train {train_start_date} -> {train_end_date}, test {test_period_days}d, up to {max_iterations} iterations", flush=True)

    # Trainer and Predictor
    trainer = ProgressiveTrainer(data_loader=loader)
    predictor = ProgressivePredictor(data_loader=loader, model_dir=str(ROOT / "app" / "ml" / "models" / "progressive"))

    # Backtester config and run
    backtester = ProgressiveBacktester(
        data_loader=loader,
        trainer=trainer,
        predictor=predictor,
        config={
            "save_all_models": False,
            "save_results": True,
            "results_dir": str(ROOT / "app" / "ml" / "models" / "backtest_results"),
        },
    )

    t0 = time.time()
    results = backtester.run_backtest(
        symbol=symbol,
        train_start_date=train_start_date,
        train_end_date=train_end_date,
        test_period_days=test_period_days,
        max_iterations=max_iterations,
        target_accuracy=0.85,
        auto_stop=True,
        model_types=["cnn", "lstm", "transformer"],
    )
    t1 = time.time()

    keys = ["status", "total_iterations", "best_iteration", "best_accuracy", "total_time", "job_id"]
    summary = {k: results.get(k) for k in keys if isinstance(results, dict) and k in results}
    print("Backtest summary:", json.dumps(summary, indent=2))
    print(f"Completed in {t1 - t0:.1f}s")


if __name__ == "__main__":
    main()
