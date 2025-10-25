import os
import sys
import json
import argparse
from datetime import datetime, timedelta

import pandas as pd

# Ensure we can import from app
sys.path.append(os.path.abspath("."))

from app.ml.progressive.data_loader import ProgressiveDataLoader
from app.ml.progressive.trainer import ProgressiveTrainer
from app.ml.progressive.predictor import ProgressivePredictor
from app.ml.progressive.backtester import ProgressiveBacktester


def ensure_gpu_ready():
    import torch
    print(f"torch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} | CUDA {torch.version.cuda}")
    else:
        print("WARNING: CUDA not available; training will run on CPU")


def get_date_bounds(symbol: str, stock_dir: str = "stock_data"):
    price_path = os.path.join(stock_dir, symbol, f"{symbol}_price.csv")
    df = pd.read_csv(price_path, index_col=0)
    df.index = pd.to_datetime(df.index, errors='coerce')
    return df.index.min().date(), df.index.max().date()


def run_for_model(symbol: str, model_type: str, cutoff_days: int = 30, epochs: int = 10, test_days: int = 30):
    print("\n" + "=" * 80)
    print(f"Running backtest for {symbol} | model={model_type}")
    print("=" * 80)

    # Dates
    min_date, max_date = get_date_bounds(symbol)
    today = datetime.utcnow().date()
    cutoff_date = today - timedelta(days=cutoff_days)
    # Clamp cutoff not to exceed max available - 1 day
    if cutoff_date > max_date:
        cutoff_date = max_date
    train_start = str(min_date)
    train_end = str(cutoff_date)
    # test_days passed as argument

    # Data loader
    dl = ProgressiveDataLoader(
        stock_data_dir="stock_data",
        sequence_length=60,
        horizons=[1, 7, 30],
        use_fundamentals=True,
        use_technical_indicators=True,
    )

    # Trainer (light epochs for a quick run; early stopping enabled by default)
    trainer = ProgressiveTrainer(
        data_loader=dl,
        training_config={
            'epochs': epochs,
            'batch_size': 64,
            'validation_split': 0.2,
            'early_stopping_patience': 6,
            'reduce_lr_patience': 4,
            'reduce_lr_factor': 0.5,
        }
    )

    predictor = ProgressivePredictor(data_loader=dl)

    backtester = ProgressiveBacktester(
        data_loader=dl,
        trainer=trainer,
        predictor=predictor,
        config={'save_all_models': False, 'save_results': True}
    )

    # Single-iteration backtest for this model type
    summary = backtester.run_backtest(
        symbol=symbol,
        train_start_date=train_start,
        train_end_date=train_end,
        test_period_days=test_days,
        max_iterations=1,
        target_accuracy=0.85,
        auto_stop=True,
        model_types=[model_type],
    )

    print("Backtest summary:")
    print(json.dumps(summary, indent=2, default=str))

    # Per-model prediction now from the job-specific checkpoint directory (if available)
    try:
        job_dir = getattr(backtester, 'job_model_dir', None)
        if job_dir is not None:
            predictor.model_dir = job_dir
        # Clear cache and load only this model type
        predictor.loaded_models.pop(symbol, None)
        predictor.load_models(symbol, [model_type])
        pred = predictor.predict_ensemble(symbol)
    except Exception as e:
        print(f"WARNING: Prediction after backtest failed: {e}")
        pred = {
            'symbol': symbol,
            'predictions': {},
            'note': f'Prediction unavailable: {e}'
        }
    print("\nCurrent predictions (ensemble contains only this model type):")
    print(json.dumps(pred, indent=2, default=str))

    return {
        'model_type': model_type,
        'backtest': summary,
        'prediction': pred,
    }


def main():
    parser = argparse.ArgumentParser(description="Run progressive backtest/training for a symbol.")
    parser.add_argument("symbol", nargs="?", default="INTC", help="Ticker symbol (default: INTC)")
    parser.add_argument("--models", nargs="+", default=["lstm", "transformer", "cnn"], choices=["lstm", "transformer", "cnn"], help="Model types to train")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs per model (default: 10)")
    parser.add_argument("--cutoff-days", type=int, default=30, help="Days before today to end training window (default: 30)")
    parser.add_argument("--test-days", type=int, default=30, help="Test period days after training (default: 30)")
    parser.add_argument("--skip-ensemble", action="store_true", help="Skip the ensemble evaluation phase")
    args = parser.parse_args()

    ensure_gpu_ready()
    symbol = args.symbol.upper()
    results = []
    for m in args.models:
        results.append(run_for_model(symbol, m, cutoff_days=args.cutoff_days, epochs=args.epochs, test_days=args.test_days))

    # After training individual models, run an ENSEMBLE evaluation over the same window
    try:
        if args.skip_ensemble:
            raise RuntimeError("Ensemble evaluation skipped by flag")
        print("\n" + "=" * 80)
        print(f"Running ensemble evaluation for {symbol}")
        print("=" * 80)

        # Dates
        min_date, max_date = get_date_bounds(symbol)
        today = datetime.utcnow().date()
        cutoff_date = today - timedelta(days=args.cutoff_days)
        if cutoff_date > max_date:
            cutoff_date = max_date
        train_end = str(cutoff_date)
        test_start = (cutoff_date + timedelta(days=1)).strftime('%Y-%m-%d')
        test_end = (cutoff_date + timedelta(days=args.test_days+1)).strftime('%Y-%m-%d')

        dl = ProgressiveDataLoader(
            stock_data_dir="stock_data",
            sequence_length=60,
            horizons=[1, 7, 30],
            use_fundamentals=True,
            use_technical_indicators=True,
        )
        trainer = ProgressiveTrainer(data_loader=dl)
        predictor = ProgressivePredictor(data_loader=dl)
        backtester = ProgressiveBacktester(data_loader=dl, trainer=trainer, predictor=predictor)

        # Load full df for evaluation
        full_min, full_max = get_date_bounds(symbol)
        # Evaluate ensemble on the next month after cutoff
        # Ensure predictor uses all available model types (clear any previous filtering)
        predictor.loaded_models.pop(symbol, None)
        full_loader = ProgressiveDataLoader(stock_data_dir="stock_data", sequence_length=60, horizons=[1,7,30])
        full_df = full_loader.load_stock_data(symbol)
        ensemble_eval = backtester.evaluate_iteration(
            symbol=symbol,
            test_start_date=test_start,
            test_end_date=test_end,
            iteration_num=1,
            full_df=full_df
        )

        # Current ensemble prediction
        predictor.loaded_models.pop(symbol, None)
        ensemble_pred = predictor.predict_ensemble(symbol)

        print("Ensemble evaluation summary:")
        print(json.dumps(ensemble_eval, indent=2, default=str))
        print("\nCurrent predictions (full ensemble):")
        print(json.dumps(ensemble_pred, indent=2, default=str))

        results.append({
            'model_type': 'ensemble_all',
            'backtest': {
                'train_until': train_end,
                'test_period': f"{test_start} to {test_end}",
                **ensemble_eval
            },
            'prediction': ensemble_pred,
        })
    except Exception as e:
        print(f"WARNING: Ensemble evaluation skipped/failed: {e}")

    # Save combined results to models folder
    out_dir = os.path.join("app", "ml", "models")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"run_{symbol}_summary_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved combined summary to: {out_path}")


if __name__ == "__main__":
    main()
