"""
Train ML Models on 1000 Selected Stocks
Trains LSTM, Transformer, CNN, and Ensemble models
Run this weekly (Sunday 04:00) for model updates
"""
import sys
from pathlib import Path
import logging
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Change to project directory for relative paths
import os
os.chdir(project_root)

# Setup logging
log_file = f"logs/weekly_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
Path("logs").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

def main():
    logging.info("=" * 70)
    logging.info("WEEKLY ML MODEL TRAINING - 1000 STOCKS")
    logging.info("=" * 70)
    logging.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Import training module
        from app.ml import train_model
        
        logging.info("\n📥 Loading prepared data from app/ml/data/...")
        logging.info("Expected: ~1,000 stocks with ~1.4M training samples")
        logging.info("This will take 2-3 hours on CPU\n")
        
        # Run the training
        train_model.main()
        
        logging.info("\n" + "=" * 70)
        logging.info("✅ WEEKLY TRAINING COMPLETED SUCCESSFULLY")
        logging.info("=" * 70)
        logging.info(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logging.info("\nModels saved:")
        logging.info("  - app/ml/models/lstm_model.keras")
        logging.info("  - app/ml/models/transformer_model.keras")
        logging.info("  - app/ml/models/cnn_model.keras")
        logging.info("  - app/ml/models/ensemble_model.keras")
        logging.info("\n🎯 Ready for daily scanning!")
        
    except Exception as e:
        logging.error(f"\n❌ TRAINING FAILED: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
