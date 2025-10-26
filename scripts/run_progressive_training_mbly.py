import sys
import json
from pathlib import Path
import time

# Ensure project root on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.data.data_manager import DataManager
from app.ml.progressive.data_loader import ProgressiveDataLoader
from app.ml.progressive.trainer import ProgressiveTrainer


def main():
    symbol = "MBLY"
    print(f"=== Progressive ML training (regular) for {symbol} ===", flush=True)

    # Ensure data exists
    dm = DataManager()
    ensure = dm.ensure_symbol_data(symbol)
    print("Ensure summary:", json.dumps(ensure.to_dict(), indent=2), flush=True)

    # Data loader with defaults
    loader = ProgressiveDataLoader(stock_data_dir=str(dm.stock_data_dir))

    # Regular training config (moderate epochs, all models)
    trainer = ProgressiveTrainer(
        data_loader=loader,
        training_config={
            "epochs": 10,
            "batch_size": 64,
            "validation_split": 0.2,
            "early_stopping_patience": 3,
            "reduce_lr_patience": 2,
            "reduce_lr_factor": 0.5,
            "verbose": 1,
        },
        save_dir=str(ROOT / "app" / "ml" / "models" / "progressive")
    )

    t0 = time.time()
    result = trainer.train_progressive_models(symbol=symbol, model_types=["cnn", "lstm", "transformer"])
    t1 = time.time()

    horizons = list(result.get("cnn", {}).keys()) if isinstance(result, dict) else []
    print(f"Training completed in {t1 - t0:.1f}s")
    print("Model types: cnn, lstm, transformer")
    print("Horizons:", horizons)
    print("Models saved under:", trainer.save_dir)


if __name__ == "__main__":
    main()
