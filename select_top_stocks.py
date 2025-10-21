"""
Stock Selection for ML Training - Hybrid Strategy
Selects 500-1000 high-quality stocks from 10,825 available stocks
"""
import pandas as pd
from pathlib import Path
import logging
import json
from datetime import datetime, timedelta

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class StockSelector:
    def __init__(self, stock_data_dir="stock_data", min_rows=1000, min_volume=500_000, min_market_cap=1_000_000_000):
        self.stock_data_dir = Path(stock_data_dir)
        self.min_rows = min_rows  # Minimum 1000 days of data (~4 years)
        self.min_volume = min_volume  # 500K average volume
        self.min_market_cap = min_market_cap  # $1B market cap
        
    def check_data_quality(self, symbol):
        """Check if stock has sufficient quality data"""
        try:
            csv_path = self.stock_data_dir / symbol / f"{symbol}_price.csv"
            if not csv_path.exists():
                return None, "No CSV file"
            
            df = pd.read_csv(csv_path)
            
            # Check row count
            if len(df) < self.min_rows:
                return None, f"Insufficient data: {len(df)} rows"
            
            # Check required columns
            required_cols = ['Date', 'Close', 'Volume']
            if not all(col in df.columns for col in required_cols):
                return None, "Missing columns"
            
            # Calculate average volume
            avg_volume = df['Volume'].mean()
            if avg_volume < self.min_volume:
                return None, f"Low volume: {avg_volume:.0f}"
            
            # Check for recent data (within last 30 days)
            df['Date'] = pd.to_datetime(df['Date'])
            latest_date = df['Date'].max()
            days_ago = (datetime.now() - latest_date).days
            if days_ago > 30:
                return None, f"Stale data: {days_ago} days old"
            
            # Check for completeness (no large gaps)
            date_diff = df['Date'].diff().dt.days
            max_gap = date_diff.max()
            if max_gap > 30:
                return None, f"Data gap: {max_gap} days"
            
            return {
                'symbol': symbol,
                'rows': len(df),
                'avg_volume': avg_volume,
                'first_date': df['Date'].min().strftime('%Y-%m-%d'),
                'last_date': df['Date'].max().strftime('%Y-%m-%d'),
                'avg_close': df['Close'].mean(),
                'max_gap': max_gap
            }, None
            
        except Exception as e:
            return None, f"Error: {str(e)}"
    
    def load_fundamentals(self, symbol):
        """Load fundamental data if available"""
        try:
            json_path = self.stock_data_dir / symbol / f"{symbol}_advanced.json"
            if not json_path.exists():
                return {}
            
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # Extract key metrics
            fundamentals = {}
            if 'market_cap' in data:
                try:
                    # Parse market cap (can be like "1.2T", "500M", etc.)
                    mc_str = str(data['market_cap']).upper()
                    mc_value = 0
                    if 'T' in mc_str:
                        mc_value = float(mc_str.replace('T', '').strip()) * 1_000_000_000_000
                    elif 'B' in mc_str:
                        mc_value = float(mc_str.replace('B', '').strip()) * 1_000_000_000
                    elif 'M' in mc_str:
                        mc_value = float(mc_str.replace('M', '').strip()) * 1_000_000
                    fundamentals['market_cap'] = mc_value
                except:
                    pass
            
            # Other useful metrics
            for key in ['pe_ratio', 'eps', 'dividend_yield', 'sector', 'industry']:
                if key in data:
                    fundamentals[key] = data[key]
            
            return fundamentals
            
        except Exception as e:
            logging.debug(f"Could not load fundamentals for {symbol}: {e}")
            return {}
    
    def select_stocks(self, target_count=500, max_per_sector=100):
        """Select stocks based on quality criteria"""
        logging.info(f"Scanning {len(list(self.stock_data_dir.iterdir()))} stock folders...")
        
        quality_stocks = []
        failed_count = 0
        
        # Scan all stocks
        for stock_folder in self.stock_data_dir.iterdir():
            if not stock_folder.is_dir():
                continue
            
            symbol = stock_folder.name
            quality_info, error = self.check_data_quality(symbol)
            
            if quality_info:
                # Add fundamentals
                fundamentals = self.load_fundamentals(symbol)
                quality_info.update(fundamentals)
                
                # Filter by market cap if available
                if 'market_cap' in quality_info:
                    if quality_info['market_cap'] < self.min_market_cap:
                        failed_count += 1
                        continue
                
                quality_stocks.append(quality_info)
            else:
                failed_count += 1
                if failed_count % 100 == 0:
                    logging.info(f"Scanned: {len(quality_stocks)} passed, {failed_count} failed")
        
        logging.info(f"✅ Found {len(quality_stocks)} quality stocks")
        logging.info(f"❌ Filtered out {failed_count} stocks")
        
        # Convert to DataFrame for sorting
        df = pd.DataFrame(quality_stocks)
        
        # Sort by: 1) Market Cap (if available), 2) Volume, 3) Data completeness
        if 'market_cap' in df.columns:
            df = df.sort_values(['market_cap', 'avg_volume'], ascending=[False, False])
        else:
            df = df.sort_values('avg_volume', ascending=False)
        
        # Sector diversification if available
        if 'sector' in df.columns and max_per_sector:
            selected = []
            sector_counts = {}
            
            for _, row in df.iterrows():
                sector = row.get('sector', 'Unknown')
                if sector_counts.get(sector, 0) < max_per_sector:
                    selected.append(row)
                    sector_counts[sector] = sector_counts.get(sector, 0) + 1
                    
                    if len(selected) >= target_count:
                        break
            
            df_selected = pd.DataFrame(selected)
        else:
            df_selected = df.head(target_count)
        
        return df_selected
    
    def save_selection(self, df, output_path="selected_stocks.json"):
        """Save selected stocks to JSON"""
        output_path = Path(output_path)
        
        # Save full details
        df.to_json(output_path, orient='records', indent=2)
        logging.info(f"💾 Saved selection to: {output_path}")
        
        # Save symbols list
        symbols_path = output_path.with_name("selected_symbols.txt")
        with open(symbols_path, 'w') as f:
            for symbol in df['symbol']:
                f.write(f"{symbol}\n")
        logging.info(f"💾 Saved symbols list to: {symbols_path}")
        
        # Print summary by sector
        if 'sector' in df.columns:
            logging.info("\n📊 Sector Distribution:")
            sector_counts = df['sector'].value_counts()
            for sector, count in sector_counts.items():
                logging.info(f"   {sector}: {count}")


def main():
    logging.info("=" * 70)
    logging.info("STOCK SELECTION FOR ML TRAINING")
    logging.info("=" * 70)
    
    selector = StockSelector(
        stock_data_dir="stock_data",
        min_rows=1000,  # ~4 years of data
        min_volume=500_000,  # 500K average volume
        min_market_cap=1_000_000_000  # $1B market cap
    )
    
    # Select 500-1000 stocks
    df = selector.select_stocks(target_count=1000, max_per_sector=150)
    
    logging.info(f"\n✅ Selected {len(df)} stocks for ML training")
    logging.info(f"   Average volume: {df['avg_volume'].mean():,.0f}")
    if 'market_cap' in df.columns:
        logging.info(f"   Average market cap: ${df['market_cap'].mean():,.0f}")
    logging.info(f"   Average data rows: {df['rows'].mean():.0f}")
    
    # Save
    selector.save_selection(df, "app/ml/data/selected_stocks.json")
    
    logging.info("=" * 70)
    logging.info("STOCK SELECTION COMPLETED")
    logging.info("=" * 70)


if __name__ == "__main__":
    main()
