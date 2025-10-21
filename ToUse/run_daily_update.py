"""
Daily Stock Data Update Automation
Schedule: Daily at 5:00 PM (after market close)
Purpose: Download latest stock prices for all tickers
"""
import sys
import os
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Setup logging
log_dir = project_root / "logs"
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"daily_update_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

import logging
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
    logging.info("DAILY STOCK DATA UPDATE - START")
    logging.info("=" * 70)
    
    try:
        # Import the stocks module
        from ToUseForData import stocks
        
        # Get all tickers
        logging.info("Loading tickers...")
        all_tickers = stocks.get_all_tickers()
        logging.info(f"Found {len(all_tickers)} tickers to update")
        
        # Update prices for all tickers
        logging.info("Starting price updates...")
        stocks.update_price_data(all_tickers)
        
        logging.info("=" * 70)
        logging.info("DAILY STOCK DATA UPDATE - COMPLETED SUCCESSFULLY")
        logging.info("=" * 70)
        
    except Exception as e:
        logging.error(f"DAILY UPDATE FAILED: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
