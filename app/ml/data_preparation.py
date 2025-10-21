"""
ML Data Preparation - הכנת נתונים לאימון המודלים
====================================================

מטרות:
1. הורדת נתוני מחירים היסטוריים (3-5 שנים)
2. חישוב אינדיקטורים טכניים (RSI, MACD, Bollinger Bands, SMA, EMA)
3. הוספת נתוני סנטימנט מחדשות
4. נרמול נתונים (0-1 scale)
5. יצירת sequences לאימון
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
        highest_high = high.rolling(window=period).min()
        
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
    """הכנת נתונים מלאה למודלי ML"""
    
    def __init__(self, 
                 symbols: List[str],
                 period: str = "5y",
                 sequence_length: int = 60,
                 prediction_horizon: int = 1):
        """
        Args:
            symbols: רשימת מניות (לדוגמה: ['AAPL', 'MSFT', 'GOOGL'])
            period: תקופה להורדה ('1y', '2y', '5y', 'max')
            sequence_length: אורך sequence לאימון (60 = 60 ימים אחורה)
            prediction_horizon: כמה ימים קדימה לחזות (1 = יום אחד)
        """
        self.symbols = symbols
        self.period = period
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.data_cache = {}
        self.features_cache = {}
        
        # Create data directory
        self.data_dir = Path("ml/data")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"✅ DataPreparation initialized for {len(symbols)} symbols")
    
    def download_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """הורדת נתונים היסטוריים מ-Yahoo Finance"""
        try:
            logger.info(f"📥 Downloading data for {symbol}...")
            
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=self.period)
            
            if data.empty:
                logger.warning(f"⚠️ No data available for {symbol}")
                return None
            
            logger.info(f"✅ Downloaded {len(data)} days for {symbol}")
            
            # Save to cache
            self.data_cache[symbol] = data
            
            return data
            
        except Exception as e:
            logger.error(f"❌ Error downloading {symbol}: {e}")
            return None
    
    def calculate_all_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """חישוב כל האינדיקטורים הטכניים"""
        try:
            df = data.copy()
            
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
            
            # Price change features
            df['price_change'] = df['Close'].pct_change()
            df['volume_change'] = df['Volume'].pct_change()
            
            # Momentum
            df['momentum'] = df['Close'] - df['Close'].shift(10)
            
            # Volatility (20-day rolling)
            df['volatility'] = df['Close'].rolling(window=20).std()
            
            logger.info(f"✅ Calculated {len(df.columns)} features")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error calculating indicators: {e}")
            return data
    
    def add_sentiment_data(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """הוספת נתוני סנטימנט (כרגע mock - נשלב עם מערכת החדשות שלנו)"""
        try:
            df = data.copy()
            
            # Mock sentiment data (בעתיד נחבר למערכת החדשות האמיתית)
            # ניצור sentiment score רנדומלי עם bias לכיוון המגמה
            np.random.seed(42)
            
            # Sentiment score בין -1 ל-1
            df['sentiment'] = np.random.normal(0, 0.3, len(df))
            
            # Bias sentiment לפי מגמת המחיר
            price_trend = df['Close'].pct_change(5).fillna(0)
            df['sentiment'] = df['sentiment'] + (price_trend * 0.5)
            df['sentiment'] = df['sentiment'].clip(-1, 1)
            
            # News volume (כמות חדשות - mock)
            df['news_volume'] = np.random.poisson(5, len(df))
            
            logger.info(f"✅ Added sentiment data for {symbol}")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error adding sentiment: {e}")
            return data
    
    def prepare_features(self, symbol: str) -> Optional[pd.DataFrame]:
        """הכנה מלאה של כל הפיצ'רים למניה"""
        try:
            # Download data
            data = self.download_data(symbol)
            if data is None:
                return None
            
            # Calculate technical indicators
            data = self.calculate_all_indicators(data)
            
            # Add sentiment
            data = self.add_sentiment_data(data, symbol)
            
            # Drop NaN values (נוצרים בגלל חישובי rolling)
            data = data.dropna()
            
            logger.info(f"✅ Prepared {len(data)} samples for {symbol}")
            
            # Save to cache
            self.features_cache[symbol] = data
            
            # Save to file
            self.save_features(symbol, data)
            
            return data
            
        except Exception as e:
            logger.error(f"❌ Error preparing features for {symbol}: {e}")
            return None
    
    def create_sequences(self, 
                        data: pd.DataFrame, 
                        feature_columns: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        יצירת sequences לאימון
        
        Returns:
            X: (samples, sequence_length, features) - נתוני קלט
            y: (samples,) - מחיר המטרה (המחיר הבא)
        """
        try:
            # Select only the feature columns we want
            feature_data = data[feature_columns].values
            
            # Normalize data
            scaled_data = self.scaler.fit_transform(feature_data)
            
            X = []
            y = []
            
            # Create sequences
            for i in range(self.sequence_length, len(scaled_data) - self.prediction_horizon):
                # Input: last `sequence_length` days
                X.append(scaled_data[i - self.sequence_length:i])
                
                # Target: close price after `prediction_horizon` days
                # נשתמש בעמודה הראשונה (Close price)
                y.append(scaled_data[i + self.prediction_horizon - 1, 0])
            
            X = np.array(X)
            y = np.array(y)
            
            logger.info(f"✅ Created {len(X)} sequences")
            logger.info(f"   X shape: {X.shape}")
            logger.info(f"   y shape: {y.shape}")
            
            return X, y
            
        except Exception as e:
            logger.error(f"❌ Error creating sequences: {e}")
            return np.array([]), np.array([])
    
    def get_feature_columns(self, data: pd.DataFrame) -> List[str]:
        """בחירת העמודות שישמשו כפיצ'רים"""
        # כל העמודות החשובות לאימון
        feature_cols = [
            'Close',           # מחיר סגירה (המטרה העיקרית)
            'Open',            # מחיר פתיחה
            'High',            # שיא
            'Low',             # שפל
            'Volume',          # נפח
            'rsi',             # RSI
            'macd',            # MACD
            'macd_signal',     # MACD Signal
            'macd_hist',       # MACD Histogram
            'bb_upper',        # Bollinger Upper
            'bb_middle',       # Bollinger Middle
            'bb_lower',        # Bollinger Lower
            'sma_20',          # SMA 20
            'sma_50',          # SMA 50
            'ema_12',          # EMA 12
            'ema_26',          # EMA 26
            'stoch_k',         # Stochastic K
            'stoch_d',         # Stochastic D
            'atr',             # ATR (volatility)
            'price_change',    # שינוי מחיר
            'volume_change',   # שינוי נפח
            'momentum',        # מומנטום
            'volatility',      # תנודתיות
            'sentiment',       # סנטימנט
            'news_volume'      # כמות חדשות
        ]
        
        # Return only columns that exist in the data
        available_cols = [col for col in feature_cols if col in data.columns]
        
        logger.info(f"✅ Selected {len(available_cols)} feature columns")
        
        return available_cols
    
    def save_features(self, symbol: str, data: pd.DataFrame):
        """שמירת הפיצ'רים לקובץ"""
        try:
            filepath = self.data_dir / f"{symbol}_features.csv"
            data.to_csv(filepath)
            logger.info(f"💾 Saved features to {filepath}")
            
        except Exception as e:
            logger.error(f"❌ Error saving features: {e}")
    
    def save_scaler(self):
        """שמירת ה-scaler לשימוש עתידי"""
        try:
            filepath = self.data_dir / "scaler.pkl"
            with open(filepath, 'wb') as f:
                pickle.dump(self.scaler, f)
            logger.info(f"💾 Saved scaler to {filepath}")
            
        except Exception as e:
            logger.error(f"❌ Error saving scaler: {e}")
    
    def prepare_all_symbols(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """הכנת נתונים לכל המניות"""
        results = {}
        
        logger.info(f"🚀 Starting data preparation for {len(self.symbols)} symbols...")
        
        for symbol in self.symbols:
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing {symbol}...")
            logger.info(f"{'='*60}")
            
            # Prepare features
            data = self.prepare_features(symbol)
            if data is None:
                continue
            
            # Get feature columns
            feature_cols = self.get_feature_columns(data)
            
            # Create sequences
            X, y = self.create_sequences(data, feature_cols)
            
            if len(X) > 0:
                results[symbol] = (X, y)
                logger.info(f"✅ {symbol}: {len(X)} training samples ready")
            else:
                logger.warning(f"⚠️ {symbol}: No sequences created")
        
        # Save scaler
        self.save_scaler()
        
        # Save summary
        self.save_preparation_summary(results)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"🎉 Data preparation complete!")
        logger.info(f"   Prepared {len(results)} symbols")
        logger.info(f"   Total samples: {sum(len(X) for X, y in results.values())}")
        logger.info(f"{'='*60}\n")
        
        return results
    
    def save_preparation_summary(self, results: Dict):
        """שמירת סיכום ההכנה"""
        try:
            summary = {
                'timestamp': datetime.now().isoformat(),
                'symbols': list(results.keys()),
                'period': self.period,
                'sequence_length': self.sequence_length,
                'prediction_horizon': self.prediction_horizon,
                'samples_per_symbol': {
                    symbol: len(X) for symbol, (X, y) in results.items()
                },
                'total_samples': sum(len(X) for X, y in results.values()),
                'feature_count': results[list(results.keys())[0]][0].shape[2] if results else 0
            }
            
            filepath = self.data_dir / "preparation_summary.json"
            with open(filepath, 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"💾 Saved summary to {filepath}")
            
        except Exception as e:
            logger.error(f"❌ Error saving summary: {e}")

# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # List of symbols to prepare
    symbols = [
        'AAPL',   # Apple
        'MSFT',   # Microsoft
        'GOOGL',  # Google
        'AMZN',   # Amazon
        'TSLA',   # Tesla
        'NVDA',   # Nvidia
        'META',   # Meta
        'NFLX',   # Netflix
    ]
    
    # Create data preparation instance
    prep = DataPreparation(
        symbols=symbols,
        period='5y',           # 5 years of data
        sequence_length=60,    # 60 days lookback
        prediction_horizon=1   # Predict 1 day ahead
    )
    
    # Prepare all data
    results = prep.prepare_all_symbols()
    
    print("\n✅ Data preparation complete!")
    print(f"Ready to train models on {len(results)} symbols")
