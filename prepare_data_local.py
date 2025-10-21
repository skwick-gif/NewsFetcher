"""
Data Preparation Script - Local Files Version
==============================================
Uses local CSV/JSON files from stock_data/ directory

Usage:
    python prepare_data_local.py
"""

import sys
sys.path.insert(0, '.')

from app.ml.data_preparation_local import DataPreparation
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ml/data/preparation.log'),
        logging.StreamHandler()
    ]
)

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🚀 ML DATA PREPARATION - LOCAL FILES VERSION")
    print("="*70 + "\n")
    
    # Initial 10 stocks for proof of concept
    symbols = [
        'AAPL',   # Apple - Tech
        'MSFT',   # Microsoft - Tech
        'GOOGL',  # Google - Tech
        'AMZN',   # Amazon - E-commerce/Tech
        'TSLA',   # Tesla - Auto/Tech
        'NVDA',   # Nvidia - Semiconductors
        'META',   # Meta - Social Media
        'NFLX',   # Netflix - Entertainment
        'JPM',    # JPMorgan - Finance
        'V'       # Visa - Payments
    ]
    
    print(f"📊 Preparing data for {len(symbols)} symbols")
    print(f"   Symbols: {', '.join(symbols)}")
    print(f"\n🔹 Data source: LOCAL FILES (stock_data/ directory)")
    print(f"🔹 Fallback: yfinance API")
    print(f"🔹 Period: 3 years")
    print(f"🔹 Sequence length: 60 days")
    print(f"🔹 Fundamentals: ENABLED (P/E, EPS, Market Cap, ROE)")
    print("\n" + "-"*70 + "\n")
    
    # Create data preparation instance
    prep = DataPreparation(
        symbols=symbols,
        period='3y',  # 3 years of data
        sequence_length=60,  # 60 days lookback
        prediction_horizon=1,  # Predict 1 day ahead
        stock_data_dir='stock_data',  # Local data directory
        use_local=True,  # Try local files first
        use_fundamentals=True  # Include fundamental data
    )
    
    # Prepare all symbols
    results = prep.prepare_all_symbols()
    
    # Print summary
    print("\n" + "="*70)
    print("📈 PREPARATION SUMMARY")
    print("="*70)
    print(f"✅ Successful: {len(results['symbols'])}")
    print(f"   {', '.join(results['symbols'])}")
    print(f"\n❌ Failed: {len(results['failed'])}")
    if results['failed']:
        print(f"   {', '.join(results['failed'])}")
    print(f"\n📊 Total samples: {results['total_samples']:,}")
    print(f"📊 Features per sample: {results['feature_count']}")
    print(f"📊 Sequence length: {results['sequence_length']} days")
    print(f"\n💾 Output: ml/data/")
    print("="*70 + "\n")
    
    if results['symbols']:
        print("✅ DATA READY FOR TRAINING!")
        print("\nNext steps:")
        print("1. Review ml/data/preparation_summary.json")
        print("2. Check ml/data/{SYMBOL}_features.csv files")
        print("3. Run training script: python app/ml/train_model.py")
    else:
        print("❌ NO DATA PREPARED - check errors above")
