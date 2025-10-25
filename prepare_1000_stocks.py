"""
Data Preparation for Selected 1000 Stocks
Prepares training data for the best quality stocks
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.ml.data_preparation_local import DataPreparation
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_selected_symbols(symbols_file="app/ml/data/selected_symbols.txt"):
    """Load the list of selected symbols"""
    try:
        with open(symbols_file, 'r') as f:
            symbols = [line.strip() for line in f if line.strip()]
        logging.info(f"Loaded {len(symbols)} selected symbols")
        return symbols
    except FileNotFoundError:
        logging.error(f"File not found: {symbols_file}")
        logging.info("Run select_top_stocks.py first to generate the symbols list")
        return []

def main():
    logging.info("=" * 70)
    logging.info("PREPARING DATA FOR 1000 SELECTED STOCKS")
    logging.info("=" * 70)
    
    # Load selected symbols
    symbols = load_selected_symbols()
    
    if not symbols:
        logging.error("No symbols to process. Exiting.")
        return
    
    logging.info(f"Starting data preparation for {len(symbols)} stocks...")
    logging.info("This will take approximately 30-60 minutes...")
    
    # Run data preparation
    prep = DataPreparation(symbols=symbols, period='3y')
    results = prep.prepare_all_symbols()
    
    # Print summary
    print(f"\n{'='*70}")
    print("DATA PREPARATION COMPLETED")
    print(f"{'='*70}")
    print(f"✅ Successful: {len(results['symbols'])}/{len(symbols)}")
    print(f"❌ Failed: {len(results['failed'])}")
    print(f"📊 Total samples: {results['total_samples']:,}")
    print(f"📈 Features: {results['feature_count']}")
    print(f"📁 Output directory: app/ml/data/")
    
    if results['failed']:
        print(f"\n⚠️  Some stocks failed ({len(results['failed'])} total)")
        print(f"   Check logs for details")
    
    print(f"{'='*70}")
    print("\n✅ Ready for training! Run: py train_on_1000_stocks.py")

if __name__ == "__main__":
    main()
