#!/usr/bin/env python3
"""
Data Integration Script for RL + Transformer Stock Trading System

Combines multiple data sources into unified datasets for ML training:
- Stock prices and technical indicators
- News sentiment scores
- Economic indicators
- VIX data for market volatility

Creates training-ready datasets with proper temporal alignment.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import glob

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataIntegrator:
    def __init__(self, base_dir='app/DATA'):
        self.base_dir = base_dir
        self.stock_data_dir = os.path.join(base_dir, 'stock_data')
        self.economic_dir = os.path.join(base_dir, 'economic')
        self.output_dir = os.path.join(base_dir, 'integrated')
        os.makedirs(self.output_dir, exist_ok=True)

    def load_stock_data(self, ticker):
        """Load all stock-related data for a ticker"""
        ticker_dir = os.path.join(self.stock_data_dir, ticker.upper())
        if not os.path.exists(ticker_dir):
            logger.warning(f"No data directory found for {ticker}")
            return {}

        data = {}

        # Load price data
        price_file = os.path.join(ticker_dir, 'prices.csv')
        if os.path.exists(price_file):
            data['prices'] = pd.read_csv(price_file)
            data['prices']['Date'] = pd.to_datetime(data['prices']['Date'])
            logger.info(f"Loaded {len(data['prices'])} price records for {ticker}")

        # Load technical indicators
        indicators_file = os.path.join(ticker_dir, 'indicators.csv')
        if os.path.exists(indicators_file):
            data['indicators'] = pd.read_csv(indicators_file)
            data['indicators']['Date'] = pd.to_datetime(data['indicators']['Date'])
            logger.info(f"Loaded {len(data['indicators'])} indicator records for {ticker}")

        # Load sentiment data
        sentiment_file = os.path.join(ticker_dir, 'sentiment.csv')
        if os.path.exists(sentiment_file):
            data['sentiment'] = pd.read_csv(sentiment_file)
            data['sentiment']['date'] = pd.to_datetime(data['sentiment']['date'])
            logger.info(f"Loaded {len(data['sentiment'])} sentiment records for {ticker}")

        return data

    def load_economic_data(self):
        """Load all economic indicators"""
        economic_data = {}

        if not os.path.exists(self.economic_dir):
            logger.warning("Economic data directory not found")
            return economic_data

        # Load each economic indicator
        for csv_file in glob.glob(os.path.join(self.economic_dir, '*.csv')):
            series_id = os.path.basename(csv_file).replace('.csv', '')
            df = pd.read_csv(csv_file)
            df['date'] = pd.to_datetime(df['date'])
            economic_data[series_id] = df
            logger.info(f"Loaded {len(df)} records for {series_id}")

        return economic_data

    def load_vix_data(self):
        """Load VIX data if available"""
        vix_file = os.path.join(self.stock_data_dir, 'VIX', 'prices.csv')
        if os.path.exists(vix_file):
            vix_df = pd.read_csv(vix_file)
            vix_df['Date'] = pd.to_datetime(vix_df['Date'])
            logger.info(f"Loaded {len(vix_df)} VIX records")
            return vix_df
        else:
            logger.warning("VIX data not found")
            return None

    def merge_stock_data(self, ticker_data):
        """Merge price, indicators, and sentiment data for a stock"""
        if 'prices' not in ticker_data:
            return None

        # Start with price data
        merged = ticker_data['prices'].copy()

        # Merge technical indicators
        if 'indicators' in ticker_data:
            # Convert indicators Date to datetime if needed
            ticker_data['indicators']['Date'] = pd.to_datetime(ticker_data['indicators']['Date'])
            merged = pd.merge(merged, ticker_data['indicators'], on='Date', how='left')

        # Merge sentiment data (daily frequency)
        if 'sentiment' in ticker_data:
            # Group sentiment by date and calculate daily averages
            sentiment_daily = ticker_data['sentiment'].groupby('date').agg({
                'sentiment_score': 'mean',
                'article_count': 'sum',
                'avg_confidence': 'mean'
            }).reset_index()
            sentiment_daily.rename(columns={'date': 'Date'}, inplace=True)

            merged = pd.merge(merged, sentiment_daily, on='Date', how='left')

        # Fill missing sentiment values with 0 (neutral)
        sentiment_cols = ['sentiment_score', 'article_count', 'avg_confidence']
        for col in sentiment_cols:
            if col in merged.columns:
                merged[col] = merged[col].fillna(0)

        # Sort by date
        merged = merged.sort_values('Date').reset_index(drop=True)

        return merged

    def add_economic_indicators(self, stock_df, economic_data):
        """Add economic indicators to stock data"""
        if stock_df is None or not economic_data:
            return stock_df

        # Create a copy to avoid modifying original
        df = stock_df.copy()

        # Add each economic indicator
        for series_id, econ_df in economic_data.items():
            # For quarterly data (GDP), forward fill to daily
            if series_id == 'GDP':
                # Resample to daily by forward filling
                econ_daily = econ_df.set_index('date').resample('D').ffill().reset_index()
            else:
                # For monthly data, forward fill to daily
                econ_daily = econ_df.set_index('date').resample('D').ffill().reset_index()

            # Rename value column to series_id
            econ_daily = econ_daily.rename(columns={'value': series_id})

            # Merge with stock data
            df = pd.merge(df, econ_daily[['date', series_id]], left_on='Date', right_on='date', how='left')
            df = df.drop('date', axis=1)

        # Forward fill any missing economic data
        econ_cols = list(economic_data.keys())
        df[econ_cols] = df[econ_cols].fillna(method='ffill')

        return df

    def add_vix_data(self, stock_df, vix_df):
        """Add VIX data for market volatility context"""
        if stock_df is None or vix_df is None:
            return stock_df

        df = stock_df.copy()

        # Merge VIX data
        vix_merged = pd.merge(df[['Date']], vix_df, on='Date', how='left')
        vix_merged = vix_merged.rename(columns={'Close': 'VIX'})

        # Add VIX to main dataframe
        df = pd.merge(df, vix_merged[['Date', 'VIX']], on='Date', how='left')

        # Forward fill VIX values
        df['VIX'] = df['VIX'].fillna(method='ffill')

        return df

    def create_target_variables(self, df, prediction_horizon=5):
        """Create target variables for ML training"""
        if df is None or len(df) < prediction_horizon:
            return df

        df = df.copy()

        # Future returns (percentage change)
        df[f'future_return_{prediction_horizon}d'] = (
            df['Close'].shift(-prediction_horizon) / df['Close'] - 1
        ) * 100

        # Binary classification target (up/down)
        df[f'target_up_{prediction_horizon}d'] = (
            df[f'future_return_{prediction_horizon}d'] > 0
        ).astype(int)

        # Multi-class target (strong up, weak up, neutral, weak down, strong down)
        returns = df[f'future_return_{prediction_horizon}d']
        df[f'target_multi_{prediction_horizon}d'] = pd.cut(
            returns,
            bins=[-np.inf, -2, -0.5, 0.5, 2, np.inf],
            labels=[0, 1, 2, 3, 4]  # 0=strong down, 4=strong up
        ).astype(float)

        return df

    def create_training_dataset(self, ticker, prediction_horizon=5, include_economic=True, include_vix=True):
        """Create complete training dataset for a ticker"""
        logger.info(f"Creating training dataset for {ticker}...")

        # Load all data sources
        ticker_data = self.load_stock_data(ticker)
        if not ticker_data:
            return None

        economic_data = self.load_economic_data() if include_economic else {}
        vix_data = self.load_vix_data() if include_vix else None

        # Merge stock data
        df = self.merge_stock_data(ticker_data)
        if df is None:
            return None

        # Add economic indicators
        if include_economic:
            df = self.add_economic_indicators(df, economic_data)

        # Add VIX data
        if include_vix:
            df = self.add_vix_data(df, vix_data)

        # Create target variables
        df = self.create_target_variables(df, prediction_horizon)

        # Remove rows with NaN targets
        df = df.dropna(subset=[f'future_return_{prediction_horizon}d'])

        # Fill remaining NaN values with 0 (for technical indicators)
        df = df.fillna(0)

        logger.info(f"Created dataset with {len(df)} samples and {len(df.columns)} features")

        return df

    def save_training_dataset(self, df, ticker, filename=None):
        """Save training dataset to CSV"""
        if df is None:
            return False

        if filename is None:
            filename = f"{ticker.lower()}_training_data.csv"

        filepath = os.path.join(self.output_dir, filename)

        try:
            df.to_csv(filepath, index=False)
            logger.info(f"Saved training dataset to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to save dataset: {e}")
            return False

    def create_multi_ticker_dataset(self, tickers, prediction_horizon=5):
        """Create combined dataset from multiple tickers"""
        logger.info(f"Creating multi-ticker dataset for {len(tickers)} stocks...")

        all_data = []
        for ticker in tickers:
            df = self.create_training_dataset(ticker, prediction_horizon)
            if df is not None:
                df['ticker'] = ticker  # Add ticker identifier
                all_data.append(df)

        if not all_data:
            logger.error("No valid data found for any ticker")
            return None

        # Combine all tickers
        combined_df = pd.concat(all_data, ignore_index=True)

        # Sort by date and ticker
        combined_df = combined_df.sort_values(['Date', 'ticker']).reset_index(drop=True)

        logger.info(f"Created multi-ticker dataset with {len(combined_df)} total samples")

        return combined_df

def main():
    """Main function to create integrated datasets"""
    integrator = DataIntegrator()

    # Test tickers (from previous testing)
    test_tickers = ['AAPL', 'TSLA', 'NVDA']

    # Create individual datasets
    for ticker in test_tickers:
        df = integrator.create_training_dataset(ticker, prediction_horizon=5)
        if df is not None:
            integrator.save_training_dataset(df, ticker)

    # Create combined dataset
    combined_df = integrator.create_multi_ticker_dataset(test_tickers, prediction_horizon=5)
    if combined_df is not None:
        integrator.save_training_dataset(combined_df, 'combined', 'multi_ticker_training_data.csv')

    logger.info("Data integration complete!")

if __name__ == "__main__":
    main()