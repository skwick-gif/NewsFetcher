#!/usr/bin/env python3
"""
Economic Data Collector for RL + Transformer Stock Trading System

Fetches key economic indicators from FRED API to provide market context.
These indicators help the RL agent understand broader market conditions.

Key Indicators:
- GDP (Gross Domestic Product)
- UNRATE (Unemployment Rate)
- FEDFUNDS (Federal Funds Rate)
- CPIAUCSL (Consumer Price Index)
- UMCSENT (Consumer Sentiment)
- RETAIL (Retail Sales)
- INDPRO (Industrial Production)
- HOUST (Housing Starts)

Data is saved as CSV files in app/DATA/economic/ directory.
"""

import os
import sys
import yaml
import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EconomicDataCollector:
    def __init__(self, config_path='app/DATA/config.yaml'):
        self.config_path = config_path
        self.config = self.load_config()
        self.api_key = self.config.get('api_keys', {}).get('fred', '')
        self.base_url = 'https://api.stlouisfed.org/fred/series/observations'

        # Create data directory
        self.data_dir = 'app/DATA/economic'
        os.makedirs(self.data_dir, exist_ok=True)

        # Economic indicators configuration
        self.indicators = {
            'GDP': {
                'name': 'Gross Domestic Product',
                'frequency': 'Quarterly',
                'units': 'Billions of Dollars'
            },
            'UNRATE': {
                'name': 'Unemployment Rate',
                'frequency': 'Monthly',
                'units': 'Percent'
            },
            'FEDFUNDS': {
                'name': 'Federal Funds Rate',
                'frequency': 'Monthly',
                'units': 'Percent'
            },
            'CPIAUCSL': {
                'name': 'Consumer Price Index',
                'frequency': 'Monthly',
                'units': 'Index 1982-1984=100'
            },
            'UMCSENT': {
                'name': 'Consumer Sentiment',
                'frequency': 'Monthly',
                'units': 'Index 1966:Q1=100'
            },
            'RETAIL': {
                'name': 'Retail Sales',
                'frequency': 'Monthly',
                'units': 'Millions of Dollars'
            },
            'INDPRO': {
                'name': 'Industrial Production Index',
                'frequency': 'Monthly',
                'units': 'Index 2017=100'
            },
            'HOUST': {
                'name': 'Housing Starts',
                'frequency': 'Monthly',
                'units': 'Thousands of Units'
            },
            'VIXCLS': {
                'name': 'CBOE Volatility Index (VIX)',
                'frequency': 'Daily',
                'units': 'Index'
            }
        }

    def load_config(self):
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {}

    def fetch_indicator_data(self, series_id, start_date=None, end_date=None):
        """
        Fetch data for a specific economic indicator from FRED API

        Args:
            series_id (str): FRED series ID (e.g., 'GDP', 'UNRATE')
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format

        Returns:
            pd.DataFrame: DataFrame with date and value columns
        """
        if not self.api_key or self.api_key == 'YOUR_FREE_FRED_API_KEY_HERE':
            logger.warning(f"No FRED API key found for series {series_id}")
            return pd.DataFrame()

        params = {
            'series_id': series_id,
            'api_key': self.api_key,
            'file_type': 'json',
            'observation_start': start_date or '2000-01-01',
            'observation_end': end_date or datetime.now().strftime('%Y-%m-%d')
        }

        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()
            observations = data.get('observations', [])

            # Convert to DataFrame
            df = pd.DataFrame(observations)
            if df.empty:
                logger.warning(f"No data returned for {series_id}")
                return pd.DataFrame()

            # Clean and format data
            df['date'] = pd.to_datetime(df['date'])
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            df = df[['date', 'value']].dropna()

            logger.info(f"Fetched {len(df)} observations for {series_id}")
            return df

        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed for {series_id}: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error processing data for {series_id}: {e}")
            return pd.DataFrame()

    def save_indicator_data(self, series_id, df):
        """Save indicator data to CSV file"""
        if df.empty:
            return False

        filename = f"{series_id}.csv"
        filepath = os.path.join(self.data_dir, filename)

        try:
            df.to_csv(filepath, index=False)
            logger.info(f"Saved {len(df)} records to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to save {series_id}: {e}")
            return False

    def load_existing_data(self, series_id):
        """Load existing data for an indicator if it exists"""
        filepath = os.path.join(self.data_dir, f"{series_id}.csv")
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath)
                df['date'] = pd.to_datetime(df['date'])
                logger.info(f"Loaded existing data for {series_id}: {len(df)} records")
                return df
            except Exception as e:
                logger.warning(f"Failed to load existing data for {series_id}: {e}")
                return pd.DataFrame()
        return pd.DataFrame()

    def update_indicator_data(self, series_id, force_refresh=False):
        """
        Update data for a specific indicator with incremental updates

        Args:
            series_id (str): FRED series ID
            force_refresh (bool): If True, re-download all data

        Returns:
            bool: True if successful
        """
        logger.info(f"Updating data for {series_id}...")

        if force_refresh:
            # Download all data
            df_new = self.fetch_indicator_data(series_id)
            if df_new.empty:
                return False
            return self.save_indicator_data(series_id, df_new)

        # Incremental update
        df_existing = self.load_existing_data(series_id)

        if df_existing.empty:
            # No existing data, fetch all
            df_new = self.fetch_indicator_data(series_id)
            if df_new.empty:
                return False
            return self.save_indicator_data(series_id, df_new)

        # Find the latest date in existing data
        last_date = df_existing['date'].max()
        start_date = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')

        # Fetch only new data
        df_new = self.fetch_indicator_data(series_id, start_date=start_date)
        if df_new.empty:
            logger.info(f"No new data for {series_id}")
            return True

        # Combine existing and new data
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined = df_combined.drop_duplicates(subset=['date']).sort_values('date')

        return self.save_indicator_data(series_id, df_combined)

    def collect_all_indicators(self, force_refresh=False):
        """Collect data for all configured economic indicators"""
        logger.info("Starting economic data collection...")

        results = {}
        for series_id in self.indicators.keys():
            try:
                success = self.update_indicator_data(series_id, force_refresh)
                results[series_id] = success

                # Rate limiting - FRED API allows 120 calls per minute
                time.sleep(0.5)

            except Exception as e:
                logger.error(f"Failed to collect {series_id}: {e}")
                results[series_id] = False

        # Summary
        successful = sum(results.values())
        total = len(results)
        logger.info(f"Economic data collection complete: {successful}/{total} indicators updated")

        return results

    def get_latest_values(self):
        """Get the latest values for all indicators"""
        latest_data = {}

        for series_id in self.indicators.keys():
            df = self.load_existing_data(series_id)
            if not df.empty:
                latest_row = df.iloc[-1]
                latest_data[series_id] = {
                    'date': latest_row['date'].strftime('%Y-%m-%d'),
                    'value': latest_row['value'],
                    'name': self.indicators[series_id]['name'],
                    'units': self.indicators[series_id]['units']
                }

        return latest_data

def main():
    """Main function to run economic data collection"""
    collector = EconomicDataCollector()

    if not collector.api_key or collector.api_key == 'YOUR_FREE_FRED_API_KEY_HERE':
        logger.error("FRED API key not configured. Please add your API key to config.yaml")
        logger.info("Get a free API key from: https://fred.stlouisfed.org/docs/api/api_key.html")
        return

    # Collect all economic data
    results = collector.collect_all_indicators()

    # Show latest values
    latest = collector.get_latest_values()
    if latest:
        logger.info("Latest economic indicators:")
        for series_id, data in latest.items():
            logger.info(f"  {series_id}: {data['value']} ({data['date']}) - {data['name']}")

if __name__ == "__main__":
    main()