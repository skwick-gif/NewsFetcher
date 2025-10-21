import sys
import json
from stocks import all_tickers, update_price_data, Config

DATA_FOLDER = Config.DATA_FOLDER
START_DATE = Config.START_DATE
TODO_FILE = Config.TODO_FILE
COMPLETED_FILE = Config.COMPLETED_FILE

if __name__ == "__main__":
    # טען את רשימת הטיקרים
    tickers = all_tickers
    print(f"Starting price data download for {len(tickers)} tickers...")
    for ticker in tickers:
        try:
            update_price_data(ticker, START_DATE, DATA_FOLDER)
        except Exception as e:
            print(f"Error downloading price data for {ticker}: {e}")
    print("Price data download completed.")
