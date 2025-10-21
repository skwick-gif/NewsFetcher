"""
ML Data Preparation - Local Files Version
==========================================
Uses local CSV files from stock_data/ directory instead of yfinance
Includes fundamental data from JSON files (P/E, EPS, Market Cap, ROE)
Fallback to yfinance if local files missing
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import pickle
from sklearn.preprocessing import MinMaxScaler
import json

logger = logging.getLogger(__name__)

class TechnicalIndicators:
    """חישוב אינדיקטורים טכניים"""
    
    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index - מדד חוזק יחסי"""
        delta = prices.diff()
        
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def calculate_macd(prices: pd.Series, 
                       fast: int = 12, 
                       slow: int = 26, 
                       signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Moving Average Convergence Divergence"""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    def calculate_bollinger_bands(prices: pd.Series, 
                                   period: int = 20, 
                                   std_dev: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands - רצועות בולינגר"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        return upper_band, sma, lower_band
    
    @staticmethod
    def calculate_sma(prices: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average - ממוצע נע פשוט"""
        return prices.rolling(window=period).mean()
    
    @staticmethod
    def calculate_ema(prices: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average - ממוצע נע מעריכי"""
        return prices.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def calculate_stochastic(high: pd.Series, 
                            low: pd.Series, 
                            close: pd.Series, 
                            period: int = 14) -> Tuple[pd.Series, pd.Series]:
        """Stochastic Oscillator"""
        lowest_low = low.rolling(window=period).min()
        highest_high = high.rolling(window=period).max()
        
        k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d = k.rolling(window=3).mean()
        
        return k, d
    
    @staticmethod
    def calculate_atr(high: pd.Series, 
                     low: pd.Series, 
                     close: pd.Series, 
                     period: int = 14) -> pd.Series:
        """Average True Range - תנודתיות ממוצעת"""
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr

class DataPreparation:
    """הכנת נתונים מלאה למודלי ML - גרסה עם קבצים מקומיים"""
    
    def __init__(self, 
                 symbols: List[str],
                 period: str = "5y",
                 sequence_length: int = 60,
                 prediction_horizon: int = 1,
                 stock_data_dir: str = "stock_data",
                 use_local: bool = True,
                 use_fundamentals: bool = True):
        """
        Args:
            symbols: רשימת מניות (לדוגמה: ['AAPL', 'MSFT', 'GOOGL'])
            period: תקופה ('1y', '2y', '5y', 'max')
            sequence_length: אורך sequence לאימון (60 = 60 ימים אחורה)
            prediction_horizon: כמה ימים קדימה לחזות (1 = יום אחד)
            stock_data_dir: ספריית נתונים מקומית (ברירת מחדל: stock_data)
            use_local: האם לנסות לטעון קבצים מקומיים לפני yfinance
            use_fundamentals: האם לכלול נתוני יסודות מ-JSON
        """
        self.symbols = symbols
        self.period = period
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.stock_data_dir = Path(stock_data_dir)
        self.use_local = use_local
        self.use_fundamentals = use_fundamentals
        
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.data_cache = {}
        self.features_cache = {}
        self.fundamentals_cache = {}
        
        # Create data directory
        self.data_dir = Path("ml/data")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"✅ DataPreparation initialized for {len(symbols)} symbols")
        logger.info(f"   Local files: {use_local}, Fundamentals: {use_fundamentals}")
    
    def load_local_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """טעינת נתונים מקובץ CSV מקומי"""
        csv_path = self.stock_data_dir / symbol / f"{symbol}_price.csv"
        
        if not csv_path.exists():
            logger.debug(f"Local file not found for {symbol}: {csv_path}")
            return None
        
        try:
            # Read CSV with Date column
            df = pd.read_csv(csv_path, parse_dates=['Date'], index_col='Date')
            
            # Validate required columns
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.warning(f"Missing columns for {symbol}: {missing_cols}")
                return None
            
            # Remove rows with all NaN values
            df = df.dropna(how='all')
            
            # Filter by period if needed (convert period like '3y' to years)
            if self.period and self.period != 'max':
                years = int(self.period.replace('y', ''))
                cutoff_date = datetime.now() - pd.DateOffset(years=years)
                df = df[df.index >= cutoff_date]
            
            if df.empty:
                logger.warning(f"No data after filtering for {symbol}")
                return None
            
            logger.info(f"✅ Loaded {len(df)} rows from LOCAL CSV for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error reading local file for {symbol}: {e}")
            return None
    
    def load_fundamentals(self, symbol: str) -> Dict[str, float]:
        """טעינת נתוני יסודות מקובץ JSON"""
        json_path = self.stock_data_dir / symbol / f"{symbol}_advanced.json"
        
        if not json_path.exists():
            logger.debug(f"Fundamentals file not found for {symbol}")
            return {}
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract numeric fundamentals
            fundamentals = {}
            
            # P/E Ratio
            if "P/E" in data or "PE Ratio (TTM)" in data:
                try:
                    pe_str = data.get("P/E") or data.get("PE Ratio (TTM)", "")
                    fundamentals['PE_Ratio'] = float(str(pe_str).replace(',', ''))
                except:
                    fundamentals['PE_Ratio'] = np.nan
            else:
                fundamentals['PE_Ratio'] = np.nan
            
            # EPS
            if "EPS (ttm)" in data or "EPS" in data:
                try:
                    eps_str = data.get("EPS (ttm)") or data.get("EPS", "")
                    fundamentals['EPS'] = float(str(eps_str).replace(',', ''))
                except:
                    fundamentals['EPS'] = np.nan
            else:
                fundamentals['EPS'] = np.nan
            
            # Market Cap (convert to billions)
            if "Market Cap" in data:
                try:
                    mc = str(data["Market Cap"])
                    if 'T' in mc:  # Trillion
                        fundamentals['Market_Cap'] = float(mc.replace('T', '').replace(',', '')) * 1000
                    elif 'B' in mc:  # Billion
                        fundamentals['Market_Cap'] = float(mc.replace('B', '').replace(',', ''))
                    elif 'M' in mc:  # Million
                        fundamentals['Market_Cap'] = float(mc.replace('M', '').replace(',', '')) / 1000
                    else:
                        fundamentals['Market_Cap'] = float(mc.replace(',', ''))
                except:
                    fundamentals['Market_Cap'] = np.nan
            else:
                fundamentals['Market_Cap'] = np.nan
            
            # ROE
            if "ROE" in data or "Return on Equity" in data:
                try:
                    roe_str = data.get("ROE") or data.get("Return on Equity", "")
                    fundamentals['ROE'] = float(str(roe_str).replace('%', '').replace(',', ''))
                except:
                    fundamentals['ROE'] = np.nan
            else:
                fundamentals['ROE'] = np.nan
            
            logger.debug(f"Loaded fundamentals for {symbol}: {fundamentals}")
            return fundamentals
            
        except Exception as e:
            logger.error(f"Error loading fundamentals for {symbol}: {e}")
            return {}
    
    def download_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        הורדת/טעינת נתונים היסטוריים
        נסיון ראשון: קבצים מקומיים
        Fallback: yfinance
        """
        data = None
        
        # Try loading from local files first
        if self.use_local:
            data = self.load_local_data(symbol)
            
            if data is not None:
                # Load fundamentals if requested
                if self.use_fundamentals:
                    fundamentals = self.load_fundamentals(symbol)
                    self.fundamentals_cache[symbol] = fundamentals
                
                # Save to cache
                self.data_cache[symbol] = data
                return data
        
        # Fallback to yfinance if local load failed
        logger.info(f"📥 Downloading data from yfinance for {symbol}...")
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=self.period)
            
            if data.empty:
                logger.warning(f"⚠️ No data available from yfinance for {symbol}")
                return None
            
            logger.info(f"✅ Downloaded {len(data)} days from yfinance for {symbol}")
            
            # Save to cache
            self.data_cache[symbol] = data
            
            return data
            
        except Exception as e:
            logger.error(f"❌ Error downloading {symbol}: {e}")
            return None
    
    def calculate_all_indicators(self, data: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        """חישוב כל האינדיקטורים הטכניים + יסודות"""
        try:
            df = data.copy()
            
            # === TECHNICAL INDICATORS ===
            
            # RSI
            df['rsi'] = TechnicalIndicators.calculate_rsi(df['Close'])
            
            # MACD
            macd, signal, hist = TechnicalIndicators.calculate_macd(df['Close'])
            df['macd'] = macd
            df['macd_signal'] = signal
            df['macd_hist'] = hist
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = TechnicalIndicators.calculate_bollinger_bands(df['Close'])
            df['bb_upper'] = bb_upper
            df['bb_middle'] = bb_middle
            df['bb_lower'] = bb_lower
            
            # Moving Averages
            df['sma_20'] = TechnicalIndicators.calculate_sma(df['Close'], 20)
            df['sma_50'] = TechnicalIndicators.calculate_sma(df['Close'], 50)
            df['sma_200'] = TechnicalIndicators.calculate_sma(df['Close'], 200)
            df['ema_12'] = TechnicalIndicators.calculate_ema(df['Close'], 12)
            df['ema_26'] = TechnicalIndicators.calculate_ema(df['Close'], 26)
            
            # Stochastic
            stoch_k, stoch_d = TechnicalIndicators.calculate_stochastic(
                df['High'], df['Low'], df['Close']
            )
            df['stoch_k'] = stoch_k
            df['stoch_d'] = stoch_d
            
            # ATR (volatility)
            df['atr'] = TechnicalIndicators.calculate_atr(
                df['High'], df['Low'], df['Close']
            )
            
            # Volume indicators
            df['volume_sma_20'] = df['Volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['Volume'] / df['volume_sma_20']
            
            # Price changes
            df['price_change'] = df['Close'].pct_change()
            df['price_change_5d'] = df['Close'].pct_change(periods=5)
            
            # === FUNDAMENTAL INDICATORS (if available) ===
            
            if self.use_fundamentals and symbol and symbol in self.fundamentals_cache:
                fundamentals = self.fundamentals_cache[symbol]
                
                # Add as constant columns (same value for all rows)
                for key, value in fundamentals.items():
                    df[key] = value
                
                logger.debug(f"Added {len(fundamentals)} fundamental features for {symbol}")
            
            # Drop rows with NaN (from indicators that need history)
            initial_rows = len(df)
            df = df.dropna()
            dropped_rows = initial_rows - len(df)
            
            if dropped_rows > 0:
                logger.debug(f"Dropped {dropped_rows} rows with NaN values")
            
            logger.info(f"✅ Calculated indicators for {len(df)} rows")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error calculating indicators: {e}")
            return data
    
    def add_sentiment_data(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """הוספת נתוני סנטימנט (mock data לבינתיים)"""
        # TODO: integrate with actual news sentiment
        data['sentiment_score'] = np.random.uniform(-1, 1, len(data))
        data['news_volume'] = np.random.randint(0, 100, len(data))
        
        return data
    
    def create_sequences(self, 
                        data: pd.DataFrame,
                        feature_columns: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        יצירת sequences לאימון
        
        Returns:
            X: (n_samples, sequence_length, n_features)
            y: (n_samples,) - תשואה עתידית
        """
        try:
            # Normalize features
            scaled_data = self.scaler.fit_transform(data[feature_columns])
            
            X, y = [], []
            
            for i in range(self.sequence_length, len(scaled_data) - self.prediction_horizon):
                # X: last 60 days of features
                X.append(scaled_data[i - self.sequence_length:i])
                
                # y: future return (prediction_horizon days ahead)
                current_price = data['Close'].iloc[i]
                future_price = data['Close'].iloc[i + self.prediction_horizon]
                future_return = (future_price - current_price) / current_price
                
                y.append(future_return)
            
            X = np.array(X)
            y = np.array(y)
            
            logger.info(f"✅ Created sequences: X shape {X.shape}, y shape {y.shape}")
            
            return X, y
            
        except Exception as e:
            logger.error(f"❌ Error creating sequences: {e}")
            return np.array([]), np.array([])
    
    def prepare_symbol(self, symbol: str) -> Optional[Dict]:
        """הכנת נתונים למניה בודדת"""
        logger.info(f"\n{'='*60}")
        logger.info(f"📊 Processing {symbol}")
        logger.info(f"{'='*60}")
        
        # Download/load data
        data = self.download_data(symbol)
        if data is None:
            return None
        
        # Calculate indicators (including fundamentals)
        data_with_indicators = self.calculate_all_indicators(data, symbol)
        
        # Add sentiment
        data_with_sentiment = self.add_sentiment_data(data_with_indicators, symbol)
        
        # Feature columns (everything except the target)
        feature_columns = [col for col in data_with_sentiment.columns 
                          if col not in ['Close']]  # Close is used for target calculation
        
        logger.info(f"📈 Using {len(feature_columns)} features: {feature_columns[:10]}...")
        
        # Create sequences
        X, y = self.create_sequences(data_with_sentiment, feature_columns)
        
        if len(X) == 0:
            logger.warning(f"⚠️ No sequences created for {symbol}")
            return None
        
        # Save processed data
        output_file = self.data_dir / f"{symbol}_features.csv"
        data_with_sentiment.to_csv(output_file)
        logger.info(f"💾 Saved features to {output_file}")
        
        result = {
            'symbol': symbol,
            'data': data_with_sentiment,
            'X': X,
            'y': y,
            'n_samples': len(X),
            'n_features': X.shape[2] if len(X.shape) > 2 else 0,
            'feature_columns': feature_columns,
            'date_range': f"{data.index[0].date()} to {data.index[-1].date()}"
        }
        
        logger.info(f"✅ {symbol}: {result['n_samples']} samples, {result['n_features']} features")
        
        return result
    
    def prepare_all_symbols(self) -> Dict:
        """הכנת נתונים לכל המניות"""
        results = {
            'symbols': [],
            'failed': [],
            'total_samples': 0,
            'feature_count': 0,
            'sequence_length': self.sequence_length
        }
        
        logger.info(f"\n{'#'*70}")
        logger.info(f"🚀 STARTING DATA PREPARATION FOR {len(self.symbols)} SYMBOLS")
        logger.info(f"{'#'*70}\n")
        
        for i, symbol in enumerate(self.symbols, 1):
            logger.info(f"[{i}/{len(self.symbols)}] Processing {symbol}...")
            
            result = self.prepare_symbol(symbol)
            
            if result:
                results['symbols'].append(symbol)
                results['total_samples'] += result['n_samples']
                results['feature_count'] = result['n_features']
            else:
                results['failed'].append(symbol)
                logger.warning(f"❌ Failed to process {symbol}")
        
        # Save scaler
        scaler_file = self.data_dir / "scaler.pkl"
        with open(scaler_file, 'wb') as f:
            pickle.dump(self.scaler, f)
        logger.info(f"💾 Saved scaler to {scaler_file}")
        
        # Save summary
        summary_file = self.data_dir / "preparation_summary.json"
        with open(summary_file, 'w') as f:
            summary = {
                'symbols': results['symbols'],
                'failed': results['failed'],
                'total_samples': results['total_samples'],
                'feature_count': results['feature_count'],
                'sequence_length': results['sequence_length'],
                'use_local': self.use_local,
                'use_fundamentals': self.use_fundamentals
            }
            json.dump(summary, f, indent=2)
        logger.info(f"💾 Saved summary to {summary_file}")
        
        logger.info(f"\n{'#'*70}")
        logger.info(f"✅ DATA PREPARATION COMPLETE!")
        logger.info(f"   Successful: {len(results['symbols'])}")
        logger.info(f"   Failed: {len(results['failed'])}")
        logger.info(f"   Total samples: {results['total_samples']}")
        logger.info(f"   Features: {results['feature_count']}")
        logger.info(f"{'#'*70}\n")
        
        return results


if __name__ == "__main__":
    # Define stocks to prepare
    stocks = ['MSFT', 'AAPL', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX', 'JPM', 'V']
    
    # Run data preparation
    prep = DataPreparation(symbols=stocks)
    results = prep.prepare_all_symbols()
    
    # Print summary
    print(f"\n{'='*70}")
    print("DATA PREPARATION COMPLETED")
    print(f"{'='*70}")
    print(f"✅ Successful: {len(results['successful'])}/{len(results['successful']) + len(results['failed'])}")
    print(f"❌ Failed: {len(results['failed'])}")
    print(f"📊 Total samples: {results['total_samples']}")
    print(f"📈 Features: {results['feature_count']}")
    if results['failed']:
        print(f"\n⚠️  Failed stocks: {', '.join(results['failed'])}")
    print(f"{'='*70}")
