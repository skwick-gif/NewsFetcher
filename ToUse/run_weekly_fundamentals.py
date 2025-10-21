"""
Weekly Stock Fundamentals Update Automation
Schedule: Weekly on Sunday at 2:00 AM
Purpose: Scrape comprehensive fundamental data for all tickers
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
log_file = log_dir / f"weekly_fundamentals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

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
    logging.info("WEEKLY FUNDAMENTALS UPDATE - START")
    logging.info("=" * 70)
    
    try:
        # Import the stocks module
        from ToUseForData import stocks
        
        # Get all tickers
        logging.info("Loading tickers...")
        all_tickers = stocks.get_all_tickers()
        logging.info(f"Found {len(all_tickers)} tickers to scrape")
        
        # Scrape fundamentals for all tickers
        logging.info("Starting fundamental data scraping...")
        logging.info("This will take several hours - scraping Yahoo, Finviz, Macrotrends...")
        stocks.scrape_all_data(all_tickers)
        
        logging.info("=" * 70)
        logging.info("WEEKLY FUNDAMENTALS UPDATE - COMPLETED SUCCESSFULLY")
        logging.info("=" * 70)
        
    except Exception as e:
        logging.error(f"WEEKLY UPDATE FAILED: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
