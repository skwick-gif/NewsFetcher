"""
Standalone Data Preparation Script
===================================
הכנת נתונים ללא תלות בסרבר
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.ml.data_preparation import DataPreparation
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def main():
    """הרצת הכנת הנתונים"""
    
    print("=" * 80)
    print("📊 MarketPulse ML Data Preparation")
    print("=" * 80)
    print()
    
    # List of major stocks to prepare
    symbols = [
        'AAPL',   # Apple
        'MSFT',   # Microsoft
        'GOOGL',  # Google
        'AMZN',   # Amazon
        'TSLA',   # Tesla
        'NVDA',   # Nvidia
        'META',   # Meta
        'NFLX',   # Netflix
        'JPM',    # JPMorgan
        'V',      # Visa
    ]
    
    print(f"📋 Preparing data for {len(symbols)} symbols:")
    for i, symbol in enumerate(symbols, 1):
        print(f"   {i}. {symbol}")
    print()
    
    # Create data preparation instance
    prep = DataPreparation(
        symbols=symbols,
        period='3y',           # 3 years of data (faster than 5y)
        sequence_length=60,    # 60 days lookback
        prediction_horizon=1   # Predict 1 day ahead
    )
    
    # Prepare all data
    results = prep.prepare_all_symbols()
    
    print()
    print("=" * 80)
    print("✅ DATA PREPARATION COMPLETE!")
    print("=" * 80)
    print()
    print(f"📊 Summary:")
    print(f"   ✓ Symbols processed: {len(results)}")
    print(f"   ✓ Total training samples: {sum(len(X) for X, y in results.values())}")
    print(f"   ✓ Features per sample: {results[list(results.keys())[0]][0].shape[2] if results else 0}")
    print(f"   ✓ Sequence length: 60 days")
    print()
    
    # Show details per symbol
    print("📈 Per-symbol breakdown:")
    for symbol, (X, y) in results.items():
        print(f"   {symbol}: {len(X):,} samples")
    
    print()
    print("💾 Files saved:")
    data_dir = Path("app/ml/data")
    print(f"   ✓ Location: {data_dir.absolute()}")
    print(f"   ✓ Feature files: {len(list(data_dir.glob('*_features.csv')))} CSV files")
    print(f"   ✓ Scaler: scaler.pkl")
    print(f"   ✓ Summary: preparation_summary.json")
    print()
    
    print("🎯 Next steps:")
    print("   1. Review the prepared data in app/ml/data/")
    print("   2. Run the training script to train the models")
    print("   3. Integrate trained models with Hot Stocks Scanner")
    print()
    print("=" * 80)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Error during data preparation: {e}", exc_info=True)
        sys.exit(1)
