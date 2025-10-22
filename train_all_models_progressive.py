"""
Train Progressive ML Models on All Stocks
Trains Transformer and CNN models (LSTM already trained)
Skips stocks that already have the model trained
Run overnight for full training
"""
import sys
from pathlib import Path
import logging
from datetime import datetime
import time

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Don't change CWD - let the classes use absolute paths
# The Trainer and Predictor now use absolute paths by default

# Setup logging
log_file = f"logs/progressive_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
Path("logs").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Set UTF-8 encoding for console output on Windows
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

logger = logging.getLogger(__name__)

def get_all_stock_symbols():
    """Get all stock symbols from stock_data directory"""
    # Use absolute path
    project_root = Path(__file__).parent
    stock_data_dir = project_root / "stock_data"
    symbols = []
    
    for stock_dir in stock_data_dir.iterdir():
        if stock_dir.is_dir():
            symbol = stock_dir.name
            price_file = stock_dir / f"{symbol}_price.csv"
            if price_file.exists():
                symbols.append(symbol)
    
    logger.info(f"Found {len(symbols)} stocks with price data")
    return sorted(symbols)

def check_model_exists(symbol, model_type, mode="progressive"):
    """Check if a model already exists for a symbol"""
    # Calculate absolute path
    project_root = Path(__file__).parent
    model_dir = project_root / "app" / "ml" / "models"
    
    # Check for model files in different formats:
    # 1. Flat file from callbacks: {model_type}_{symbol}_{horizon}_best.h5
    # 2. Directory structure: {model_type}_{symbol}/{horizon}.h5
    # 3. Legacy .keras format
    
    # Check flat files (most common from training)
    for horizon in ['1d', '7d', '30d']:
        flat_file = model_dir / f"{model_type}_{symbol}_{horizon}_best.h5"
        if flat_file.exists():
            return True
    
    # Check directory structure
    model_subdir = model_dir / f"{model_type}_{symbol}"
    if model_subdir.exists() and model_subdir.is_dir():
        for horizon in ['1d', '7d', '30d']:
            dir_file = model_subdir / f"{horizon}.h5"
            if dir_file.exists():
                return True
    
    # Check legacy .keras format
    for horizon in ['1d', '7d', '30d']:
        keras_file = model_dir / f"{symbol}_{model_type}_{horizon}_{mode}.keras"
        if keras_file.exists():
            return True
    
    return False

def train_stock(symbol, model_types, progressive_trainer, mode="progressive"):
    """Train models for a single stock"""
    try:
        logger.info(f"\n{'='*70}")
        logger.info(f"🚀 Training {symbol} - Models: {', '.join(model_types)}")
        logger.info(f"{'='*70}")
        
        start_time = time.time()
        
        # Train based on mode
        if mode == "progressive":
            result = progressive_trainer.train_progressive_models(
                symbol=symbol,
                model_types=model_types
            )
        else:
            result = progressive_trainer.train_unified_models(
                symbol=symbol,
                model_types=model_types
            )
        
        elapsed = time.time() - start_time
        logger.info(f"✅ {symbol} completed in {elapsed/60:.1f} minutes")
        
        return True, result
        
    except Exception as e:
        logger.error(f"❌ {symbol} failed: {e}")
        return False, str(e)

def main():
    logger.info("=" * 70)
    logger.info("PROGRESSIVE ML TRAINING - ALL STOCKS")
    logger.info("=" * 70)
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Working directory: {os.getcwd()}")
    
    try:
        # Import Progressive ML system
        logger.info("\n📦 Loading Progressive ML system...")
        from app.ml.progressive.data_loader import ProgressiveDataLoader
        from app.ml.progressive.trainer import ProgressiveTrainer
        
        progressive_data_loader = ProgressiveDataLoader()
        progressive_trainer = ProgressiveTrainer(progressive_data_loader)
        logger.info("✅ Progressive ML system loaded")
        
        # Get all stocks
        logger.info("\n📊 Scanning stock_data directory...")
        all_symbols = get_all_stock_symbols()
        
        if not all_symbols:
            logger.error("❌ No stocks found!")
            sys.exit(1)
        
        # Models to train (LSTM already done, so we do Transformer and CNN)
        available_models = ['transformer', 'cnn']
        
        logger.info(f"\n🎯 Training Plan:")
        logger.info(f"   Total stocks: {len(all_symbols)}")
        logger.info(f"   Models: {', '.join(available_models)}")
        logger.info(f"   Mode: Progressive (1d → 7d → 30d)")
        
        # Statistics
        stats = {
            'total': len(all_symbols),
            'trained': 0,
            'skipped': 0,
            'failed': 0,
            'by_model': {model: {'trained': 0, 'skipped': 0} for model in available_models}
        }
        
        start_time = time.time()
        
        # Train each stock
        for idx, symbol in enumerate(all_symbols, 1):
            logger.info(f"\n{'='*70}")
            logger.info(f"📈 Progress: {idx}/{len(all_symbols)} ({idx/len(all_symbols)*100:.1f}%)")
            logger.info(f"Stock: {symbol}")
            logger.info(f"{'='*70}")
            
            # Check which models need training
            models_to_train = []
            for model_type in available_models:
                if check_model_exists(symbol, model_type, "progressive"):
                    logger.info(f"⏭️  Skipping {model_type} - already trained")
                    stats['by_model'][model_type]['skipped'] += 1
                else:
                    models_to_train.append(model_type)
                    logger.info(f"🎯 Will train {model_type}")
            
            # Skip if all models already trained
            if not models_to_train:
                logger.info(f"✅ {symbol} - All models already trained, skipping")
                stats['skipped'] += 1
                continue
            
            # Train the needed models
            success, result = train_stock(symbol, models_to_train, progressive_trainer, "progressive")
            
            if success:
                stats['trained'] += 1
                for model_type in models_to_train:
                    stats['by_model'][model_type]['trained'] += 1
            else:
                stats['failed'] += 1
            
            # Progress report every 10 stocks
            if idx % 10 == 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / idx
                remaining = (len(all_symbols) - idx) * avg_time
                
                logger.info(f"\n{'='*70}")
                logger.info(f"📊 PROGRESS REPORT")
                logger.info(f"{'='*70}")
                logger.info(f"Completed: {idx}/{len(all_symbols)} ({idx/len(all_symbols)*100:.1f}%)")
                logger.info(f"Trained: {stats['trained']}, Skipped: {stats['skipped']}, Failed: {stats['failed']}")
                logger.info(f"Elapsed: {elapsed/3600:.1f} hours")
                logger.info(f"ETA: {remaining/3600:.1f} hours remaining")
                logger.info(f"{'='*70}\n")
        
        # Final report
        total_time = time.time() - start_time
        
        logger.info("\n" + "=" * 70)
        logger.info("✅ PROGRESSIVE TRAINING COMPLETED")
        logger.info("=" * 70)
        logger.info(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Total time: {total_time/3600:.2f} hours")
        logger.info(f"\n📊 Final Statistics:")
        logger.info(f"   Total stocks: {stats['total']}")
        logger.info(f"   Successfully trained: {stats['trained']}")
        logger.info(f"   Skipped (already done): {stats['skipped']}")
        logger.info(f"   Failed: {stats['failed']}")
        
        logger.info(f"\n🎯 By Model:")
        for model_type in available_models:
            trained = stats['by_model'][model_type]['trained']
            skipped = stats['by_model'][model_type]['skipped']
            logger.info(f"   {model_type.upper():15} - Trained: {trained:4}, Skipped: {skipped:4}")
        
        logger.info(f"\n💾 Models saved in: app/ml/models/")
        logger.info(f"📝 Full log: {log_file}")
        logger.info("=" * 70)
        
    except Exception as e:
        logger.error(f"\n❌ TRAINING FAILED: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
