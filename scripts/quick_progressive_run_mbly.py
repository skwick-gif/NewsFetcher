import sys
import json
import time
from pathlib import Path

# Ensure project root on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.data.data_manager import DataManager
from app.ml.progressive.data_loader import ProgressiveDataLoader
from app.ml.progressive.trainer import ProgressiveTrainer
from app.ml.progressive.predictor import ProgressivePredictor
from app.ml.progressive.backtester import ProgressiveBacktester
import pandas as pd


def main():
    symbol = "MBLY"
    print(f"=== Progressive ML quick run for {symbol} ===", flush=True)

    # Ensure data exists
    dm = DataManager()
    summary = dm.ensure_symbol_data(symbol)
    print("Ensure summary:", json.dumps(summary.to_dict(), indent=2), flush=True)

    # Data loader
    loader = ProgressiveDataLoader(stock_data_dir=str(dm.stock_data_dir))

    # Quick training config (short epochs to keep runtime reasonable)
    save_dir = ROOT / "app" / "ml" / "models" / "quick_mbly"
    save_dir.mkdir(parents=True, exist_ok=True)
    trainer = ProgressiveTrainer(
        data_loader=loader,
        training_config={
            "epochs": 3,
            "batch_size": 64,
            "validation_split": 0.2,
            "early_stopping_patience": 2,
            "verbose": 1,
        },
        save_dir=str(save_dir),
    )

    # Train a single light model type for speed
    t0 = time.time()
    train_res = trainer.train_progressive_models(symbol=symbol, model_types=["cnn"])
    t1 = time.time()

    print("Training done in {:.1f}s".format(t1 - t0), flush=True)
    # Summarize
    try:
        horizon_keys = list(next(iter(train_res.values())).keys())
    except Exception:
        horizon_keys = []
    print("Training horizons:", horizon_keys, flush=True)

    # Backtest (single iteration, short test window)
    df = loader.load_stock_data(symbol)
    if df is None or len(df.index) < 200:
        print("Not enough data for backtest; exiting.")
        return

    last_date = df.index.max()
    # Choose end date ~20 calendar days before last to leave test room
    train_end_date = (pd.to_datetime(last_date) - pd.Timedelta(days=20)).date().isoformat()
    # Choose a ~360 day training window if possible
    train_start_date = (pd.to_datetime(train_end_date) - pd.Timedelta(days=360)).date().isoformat()

    predictor = ProgressivePredictor(data_loader=loader, model_dir=str(save_dir))

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

    print(f"Running backtest: train {train_start_date} -> {train_end_date}, test 5d, 1 iteration", flush=True)
    bt_res = backtester.run_backtest(
        symbol=symbol,
        train_start_date=train_start_date,
        train_end_date=train_end_date,
        test_period_days=5,
        max_iterations=1,
        target_accuracy=0.9,
        auto_stop=True,
        model_types=["cnn"],
    )

    # Print concise result
    keys = ["status", "total_iterations", "best_iteration", "best_accuracy", "total_time"]
    summary = {k: bt_res.get(k) for k in keys if k in bt_res}
    print("Backtest summary:", json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
