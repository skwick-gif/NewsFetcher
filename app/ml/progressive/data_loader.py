"""
Progressive Data Loader for Multi-Horizon Stock Predictions
===========================================================
Loads and processes stock data for Progressive ML Training System

Features:
- Loads CSV (price data) + JSON (fundamental data)
- Technical indicators calculation
- Multi-horizon target creation (1, 7, 30 days)
- Progressive and Unified training data preparation
- Comprehensive feature engineering
- Data validation and quality checks
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Technical indicators
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    logging.warning("⚠️ TA-Lib not available. Using manual calculations.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProgressiveDataLoader:
    """
    Advanced data loader for Progressive ML Training System
    
    Supports both Progressive (1→7→30 days) and Unified (all horizons) training modes
    """
    
    def __init__(self, 
                 stock_data_dir: str = "stock_data",
                 sequence_length: int = 60,
                 horizons: List[int] = [1, 7, 30],
                 use_fundamentals: bool = True,
                 use_technical_indicators: bool = True,
                 validation_split: float = 0.2,
                 train_start_date: Optional[str] = None,
                 train_end_date: Optional[str] = None,
                 test_period_days: int = 14,
                 indicator_params: Optional[Dict[str, Union[int, float, List[int], str]]] = None):
        """
        Initialize Progressive Data Loader
        
        Args:
            stock_data_dir: Directory containing stock data folders
            sequence_length: Number of days to look back for predictions
            horizons: Prediction horizons in days [1, 7, 30]
            use_fundamentals: Include fundamental data from JSON files
            use_technical_indicators: Calculate technical indicators
            validation_split: Fraction of data for validation
            train_start_date: Optional start date for training data (YYYY-MM-DD)
            train_end_date: Optional end date for training data (YYYY-MM-DD)
            test_period_days: Number of days for testing period (default: 14)
        """
        # Handle relative path - if running from app/ dir, go up one level
        stock_path = Path(stock_data_dir)
        if not stock_path.is_absolute():
            # Check if running from app/ subdirectory
            if Path.cwd().name == 'app' and not stock_path.exists():
                stock_path = Path('..') / stock_data_dir
        
        self.stock_data_dir = stock_path
        self.sequence_length = sequence_length
        self.horizons = sorted(horizons)  # [1, 7, 30]
        self.use_fundamentals = use_fundamentals
        self.use_technical_indicators = use_technical_indicators
        self.validation_split = validation_split
        # Indicator parameters with defaults and safe parsing
        self.indicator_params = self._normalize_indicator_params(indicator_params or {})
        
        # Backtesting parameters
        self.train_start_date = train_start_date
        self.train_end_date = train_end_date
        self.test_period_days = test_period_days
        
        # Feature cache for efficiency
        self._feature_cache = {}
        self._fundamental_cache = {}
        
        logger.info(f"🚀 Progressive DataLoader initialized")
        logger.info(f"   📊 Horizons: {self.horizons} days")
        logger.info(f"   📈 Sequence length: {self.sequence_length} days")
        logger.info(f"   📋 Fundamentals: {'✅' if use_fundamentals else '❌'}")
        logger.info(f"   🔧 Technical indicators: {'✅' if use_technical_indicators else '❌'}")
        if self.use_technical_indicators:
            ip = self.indicator_params
            logger.info(f"   🧩 Indicator params: RSI={ip['rsi_period']}, MACD={ip['macd_fast']}/{ip['macd_slow']}/{ip['macd_signal']}, "
                        f"SMA={ip['sma_periods']}, EMA={ip['ema_periods']}, BB={ip['bb_period']}±{ip['bb_std']}")
        if train_start_date or train_end_date:
            logger.info(f"   📅 Date range: {train_start_date or 'START'} → {train_end_date or 'END'}")
            logger.info(f"   🧪 Test period: {test_period_days} days")
    
    def load_stock_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load CSV price data for a symbol"""
        try:
            csv_path = self.stock_data_dir / symbol / f"{symbol}_price.csv"
            
            if not csv_path.exists():
                logger.warning(f"⚠️ Price data not found: {csv_path}")
                return None
            
            # Load CSV with proper date parsing
            df = pd.read_csv(csv_path, index_col=0)
            
            # Parse dates with flexible format - convert to UTC first then remove timezone
            df.index = pd.to_datetime(df.index, format='mixed', errors='coerce', utc=True).tz_localize(None)
            
            # Ensure columns are in correct order and named properly
            expected_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in df.columns for col in expected_columns):
                logger.error(f"❌ Missing required columns in {symbol} data. Expected: {expected_columns}")
                return None
            
            # Sort by date
            df = df.sort_index()
            
            # Filter by date range if specified
            if self.train_start_date:
                df = df[df.index >= self.train_start_date]
                logger.info(f"   📅 Filtered from {self.train_start_date}")
            
            if self.train_end_date:
                df = df[df.index <= self.train_end_date]
                logger.info(f"   📅 Filtered until {self.train_end_date}")
            
            # Basic data validation
            if len(df) < self.sequence_length + max(self.horizons) + 10:
                logger.warning(f"⚠️ Insufficient data for {symbol}: {len(df)} rows")
                return None
            
            # Remove any NaN values
            initial_len = len(df)
            df = df.dropna()
            if len(df) < initial_len:
                logger.info(f"📝 Cleaned {initial_len - len(df)} NaN rows from {symbol}")
            
            logger.info(f"✅ Loaded {symbol}: {len(df)} days ({df.index[0].date()} to {df.index[-1].date()})")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error loading {symbol} price data: {e}")
            return None
    
    def load_technical_indicators(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load pre-calculated technical indicators for a symbol"""
        try:
            indicators_path = self.stock_data_dir / symbol / f"{symbol}_indicators.csv"
            
            if not indicators_path.exists():
                logger.warning(f"⚠️ Technical indicators not found: {indicators_path}")
                return None
            
            # Load CSV with proper date parsing
            df = pd.read_csv(indicators_path, index_col=0)
            
            # Parse dates with flexible format - convert to UTC first then remove timezone
            df.index = pd.to_datetime(df.index, format='mixed', errors='coerce', utc=True).tz_localize(None)
            
            # Sort by date
            df = df.sort_index()
            
            # Filter by date range if specified
            if self.train_start_date:
                df = df[df.index >= self.train_start_date]
            
            if self.train_end_date:
                df = df[df.index <= self.train_end_date]
            
            logger.info(f"✅ Loaded {symbol} indicators: {len(df)} days, {len(df.columns)} indicators")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error loading {symbol} indicators: {e}")
            return None
    
    def load_sentiment_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load sentiment analysis data for a symbol"""
        try:
            sentiment_path = self.stock_data_dir / symbol / f"{symbol}_sentiment.csv"
            
            if not sentiment_path.exists():
                logger.warning(f"⚠️ Sentiment data not found: {sentiment_path}")
                return None
            
            # Load CSV with proper date parsing
            df = pd.read_csv(sentiment_path, index_col=0, parse_dates=True)
            
            # Sort by date
            df = df.sort_index()
            
            # Filter by date range if specified
            if self.train_start_date:
                df = df[df.index >= self.train_start_date]
            
            if self.train_end_date:
                df = df[df.index <= self.train_end_date]
            
            logger.info(f"✅ Loaded {symbol} sentiment: {len(df)} days, {len(df.columns)} sentiment features")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error loading {symbol} sentiment: {e}")
            return None
    
    def load_fundamental_data(self, symbol: str) -> Dict[str, float]:
        """Load fundamental data including economic indicators from advanced JSON"""
        try:
            # Check cache first
            if symbol in self._fundamental_cache:
                return self._fundamental_cache[symbol]
            
            advanced_path = self.stock_data_dir / symbol / f"{symbol}_advanced.json"
            
            if not advanced_path.exists():
                logger.warning(f"⚠️ Advanced data not found: {advanced_path}")
                return {}
            
            # Load JSON data
            with open(advanced_path, 'r') as f:
                data = json.load(f)
            
            fundamentals = {}
            
            # Extract fundamental metrics
            if 'fundamentals' in data:
                fund_data = data['fundamentals']
                
                # Financial ratios and metrics
                financial_keys = [
                    'market_cap', 'pe_ratio', 'pb_ratio', 'debt_to_equity', 
                    'return_on_equity', 'return_on_assets', 'gross_margin',
                    'operating_margin', 'net_margin', 'revenue_growth',
                    'earnings_growth', 'dividend_yield', 'beta'
                ]
                
                for key in financial_keys:
                    if key in fund_data and fund_data[key] is not None:
                        parsed_value = self._parse_financial_value(str(fund_data[key]))
                        if parsed_value is not None:
                            fundamentals[key] = parsed_value
                
                # Company info
                if 'sector' in fund_data:
                    fundamentals['sector'] = hash(fund_data['sector']) % 1000  # Convert to numeric
                if 'industry' in fund_data:
                    fundamentals['industry'] = hash(fund_data['industry']) % 1000  # Convert to numeric
            
            # Extract economic indicators
            if 'Economic_Data' in data:
                econ_data = data['Economic_Data']
                
                # Interest rates
                if 'FEDFUNDS' in econ_data:
                    fundamentals['fed_funds_rate'] = float(econ_data['FEDFUNDS'])
                if 'treasury_10y' in econ_data:
                    fundamentals['treasury_10y'] = float(econ_data['treasury_10y'])
                if 'treasury_2y' in econ_data:
                    fundamentals['treasury_2y'] = float(econ_data['treasury_2y'])
                
                # Economic indicators with proper key mapping
                economic_mappings = {
                    'GDP': 'gdp_growth',
                    'UNRATE': 'unemployment_rate', 
                    'CPIAUCSL': 'cpi',
                    'UMCSENT': 'consumer_confidence',
                    'RSAFS': 'retail_sales',
                    'INDPRO': 'industrial_production',
                    'HOUST': 'housing_starts',
                    'VIX': 'vix'
                }
                
                for json_key, internal_key in economic_mappings.items():
                    if json_key in econ_data and econ_data[json_key] is not None:
                        parsed_value = self._parse_financial_value(str(econ_data[json_key]))
                        if parsed_value is not None:
                            fundamentals[f'econ_{internal_key}'] = parsed_value
            
            # Cache the results
            self._fundamental_cache[symbol] = fundamentals
            
            logger.info(f"✅ Loaded {symbol} fundamentals: {len(fundamentals)} features")
            return fundamentals
            
        except Exception as e:
            logger.error(f"❌ Error loading {symbol} fundamental data: {e}")
            return {}
    
    def _parse_financial_value(self, value_str: str) -> Optional[float]:
        """Parse financial values like '37.29', '3639.90B', '0.75%' """
        if not isinstance(value_str, str):
            return None
        
        try:
            # Remove common suffixes and convert
            value_str = value_str.strip()
            
            # Handle percentages
            if value_str.endswith('%'):
                return float(value_str[:-1]) / 100.0
            
            # Handle billions/millions
            if value_str.endswith('B'):
                return float(value_str[:-1]) * 1e9
            elif value_str.endswith('M'):
                return float(value_str[:-1]) * 1e6
            elif value_str.endswith('K'):
                return float(value_str[:-1]) * 1e3
            
            # Handle regular numbers
            return float(value_str)
            
        except (ValueError, AttributeError):
            return None
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical indicators"""
        try:
            df = df.copy()
            ip = self.indicator_params
            
            # Basic price features
            df['returns'] = df['Close'].pct_change()
            df['log_returns'] = np.log(df['Close']).diff()
            df['volatility'] = df['returns'].rolling(20).std()
            df['volume_ma'] = df['Volume'].rolling(20).mean()
            df['price_volume_trend'] = (df['Close'].pct_change() * df['Volume']).rolling(10).mean()
            
            # Moving averages
            for period in ip['sma_periods']:
                df[f'sma_{period}'] = df['Close'].rolling(period).mean()
            for period in ip['ema_periods']:
                df[f'ema_{period}'] = df['Close'].ewm(span=period).mean()
            
            # Price relative to MAs
            df['price_above_sma20'] = (df['Close'] > df['sma_20']).astype(int)
            df['price_above_sma50'] = (df['Close'] > df['sma_50']).astype(int)
            
            # Bollinger Bands
            bb_period = int(ip['bb_period'])
            bb_std = float(ip['bb_std'])
            df['bb_middle'] = df['Close'].rolling(bb_period).mean()
            bb_std_val = df['Close'].rolling(bb_period).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std_val * bb_std)
            df['bb_lower'] = df['bb_middle'] - (bb_std_val * bb_std)
            df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            if TALIB_AVAILABLE:
                # Advanced indicators with TA-Lib
                df['rsi'] = talib.RSI(df['Close'].values, timeperiod=int(ip['rsi_period']))
                df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(
                    df['Close'].values,
                    fastperiod=int(ip['macd_fast']),
                    slowperiod=int(ip['macd_slow']),
                    signalperiod=int(ip['macd_signal'])
                )
                df['atr'] = talib.ATR(df['High'].values, df['Low'].values, df['Close'].values)
                df['adx'] = talib.ADX(df['High'].values, df['Low'].values, df['Close'].values)
                df['cci'] = talib.CCI(df['High'].values, df['Low'].values, df['Close'].values)
                df['williams_r'] = talib.WILLR(df['High'].values, df['Low'].values, df['Close'].values)
                df['stoch_k'], df['stoch_d'] = talib.STOCH(df['High'].values, df['Low'].values, df['Close'].values)
            else:
                # Manual calculations
                df['rsi'] = self._calculate_rsi(df['Close'], period=int(ip['rsi_period']))
                df['macd'], df['macd_signal'] = self._calculate_macd(
                    df['Close'],
                    fast=int(ip['macd_fast']),
                    slow=int(ip['macd_slow']),
                    signal=int(ip['macd_signal'])
                )
                df['atr'] = self._calculate_atr(df['High'], df['Low'], df['Close'])
            
            # Volume indicators
            df['volume_sma_ratio'] = df['Volume'] / df['volume_ma']
            df['volume_price_trend'] = (df['Volume'] * df['Close']).rolling(10).mean()
            
            # Support/Resistance levels
            df['high_20'] = df['High'].rolling(20).max()
            df['low_20'] = df['Low'].rolling(20).min()
            df['resistance_distance'] = (df['high_20'] - df['Close']) / df['Close']
            df['support_distance'] = (df['Close'] - df['low_20']) / df['Close']
            
            # Trend strength
            df['trend_strength'] = (df['Close'] - df['Close'].shift(20)) / df['Close'].shift(20)
            
            # Drop rows with NaN values created by indicators
            initial_len = len(df)
            df = df.dropna()
            logger.info(f"📊 Technical indicators calculated. Rows: {initial_len} → {len(df)}")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error calculating technical indicators: {e}")
            return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Manual RSI calculation"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        """Manual MACD calculation with configurable periods"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=signal).mean()
        return macd, signal
    
    def _calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Manual ATR calculation"""
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(period).mean()
        return atr

    def _normalize_indicator_params(self, params: Dict[str, Union[int, float, List[int], str]]) -> Dict[str, Union[int, float, List[int]]]:
        """Fill defaults and convert CSV strings to lists for indicator params"""
        def _parse_int_list(v, default):
            try:
                if isinstance(v, list):
                    return [int(x) for x in v if int(x) > 0]
                if isinstance(v, str):
                    return [int(x.strip()) for x in v.split(',') if x.strip().isdigit()]
                if isinstance(v, int):
                    return [int(v)]
            except Exception:
                pass
            return default

        out = {
            'rsi_period': int(params.get('rsi_period', 14) or 14),
            'macd_fast': int(params.get('macd_fast', 12) or 12),
            'macd_slow': int(params.get('macd_slow', 26) or 26),
            'macd_signal': int(params.get('macd_signal', 9) or 9),
            'sma_periods': _parse_int_list(params.get('sma_periods'), [5, 10, 20, 50]),
            'ema_periods': _parse_int_list(params.get('ema_periods'), [5, 10, 20, 50]),
            'bb_period': int(params.get('bb_period', 20) or 20),
            'bb_std': float(params.get('bb_std', 2) or 2.0)
        }
        # Sanity clamps
        out['rsi_period'] = max(2, min(200, out['rsi_period']))
        out['macd_fast'] = max(2, min(200, out['macd_fast']))
        out['macd_slow'] = max(out['macd_fast'] + 1, min(400, out['macd_slow']))
        out['macd_signal'] = max(2, min(200, out['macd_signal']))
        out['bb_period'] = max(5, min(200, out['bb_period']))
        out['bb_std'] = max(0.5, min(5.0, out['bb_std']))
        if not out['sma_periods']:
            out['sma_periods'] = [5, 10, 20, 50]
        if not out['ema_periods']:
            out['ema_periods'] = [5, 10, 20, 50]
        return out
    
    def create_multi_horizon_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create target variables for all prediction horizons"""
        df = df.copy()
        
        for horizon in self.horizons:
            # Future price
            df[f'target_price_{horizon}d'] = df['Close'].shift(-horizon)
            
            # Price change (absolute)
            df[f'target_change_{horizon}d'] = df[f'target_price_{horizon}d'] - df['Close']
            
            # Price change (percentage)
            df[f'target_return_{horizon}d'] = (df[f'target_price_{horizon}d'] / df['Close']) - 1
            
            # Direction (binary: 1 for up, 0 for down)
            df[f'target_direction_{horizon}d'] = (df[f'target_return_{horizon}d'] > 0).astype(int)
            
            # Magnitude buckets (for classification)
            df[f'target_magnitude_{horizon}d'] = pd.cut(
                df[f'target_return_{horizon}d'], 
                bins=[-np.inf, -0.05, -0.02, 0.02, 0.05, np.inf],
                labels=[0, 1, 2, 3, 4]  # Large Down, Small Down, Flat, Small Up, Large Up
            ).astype(float)
        
        logger.info(f"🎯 Created targets for horizons: {self.horizons} days")
        return df
    
    def prepare_features(self, symbol: str, for_prediction: bool = False) -> Optional[pd.DataFrame]:
        """Prepare complete feature set for a symbol
        
        Args:
            symbol: Stock symbol
            for_prediction: If True, keeps the last rows for prediction (doesn't remove target rows)
        """
        try:
            # Load price data
            df = self.load_stock_data(symbol)
            if df is None:
                return None
            
            # Load pre-calculated technical indicators
            if self.use_technical_indicators:
                indicators_df = self.load_technical_indicators(symbol)
                if indicators_df is not None:
                    # Remove overlapping columns to avoid conflicts
                    overlapping_cols = set(df.columns) & set(indicators_df.columns)
                    if overlapping_cols:
                        indicators_df = indicators_df.drop(columns=list(overlapping_cols))
                        logger.info(f"📊 Removed {len(overlapping_cols)} overlapping columns from indicators")
                    
                    # Merge indicators with price data
                    df = df.join(indicators_df, how='left')
                    logger.info(f"📊 Merged {len(indicators_df.columns)} technical indicators")
                else:
                    logger.warning(f"⚠️ Using calculated indicators for {symbol}")
                    df = self.calculate_technical_indicators(df)
            
            # Load sentiment data
            sentiment_df = self.load_sentiment_data(symbol)
            if sentiment_df is not None:
                # Remove overlapping columns to avoid conflicts
                overlapping_cols = set(df.columns) & set(sentiment_df.columns)
                if overlapping_cols:
                    sentiment_df = sentiment_df.drop(columns=list(overlapping_cols))
                    logger.info(f"📊 Removed {len(overlapping_cols)} overlapping columns from sentiment")
                
                # Merge sentiment with price data (left join to keep all price data)
                df = df.join(sentiment_df, how='left')
                
                # Forward fill sentiment data to handle sparse dates
                sentiment_cols_in_df = [col for col in df.columns if col in sentiment_df.columns]
                if sentiment_cols_in_df:
                    df[sentiment_cols_in_df] = df[sentiment_cols_in_df].fillna(method='ffill')
                    # Fill any remaining NaN at the beginning with 0
                    df[sentiment_cols_in_df] = df[sentiment_cols_in_df].fillna(0)
                
                logger.info(f"📊 Merged {len(sentiment_df.columns)} sentiment features (forward-filled)")
            
            # Add fundamental features (broadcast to all rows)
            if self.use_fundamentals:
                fundamentals = self.load_fundamental_data(symbol)
                for feature_name, value in fundamentals.items():
                    df[f'fund_{feature_name}'] = value
            
            # Create multi-horizon targets
            df = self.create_multi_horizon_targets(df)
            
            # Remove rows that don't have targets (last N rows) - only for training
            if not for_prediction:
                max_horizon = max(self.horizons)
                df = df.iloc[:-max_horizon]  # Remove last max_horizon rows
                logger.info(f"📊 Removed last {max_horizon} rows for training (no targets available)")
            else:
                logger.info(f"📊 Keeping all rows for prediction mode")
            
            # Final cleanup - only remove rows where essential data is missing
            # Keep rows even if sentiment data is missing (filled with 0 or ffill)
            essential_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            initial_len = len(df)
            df = df.dropna(subset=essential_cols)
            if len(df) < initial_len:
                logger.info(f"📝 Removed {initial_len - len(df)} rows with missing essential data")
            
            # Fill NaN values in technical indicators with forward fill, then backward fill
            indicator_cols = [col for col in df.columns if col not in essential_cols and not col.startswith(('target_', 'fund_'))]
            if indicator_cols:
                df[indicator_cols] = df[indicator_cols].fillna(method='ffill')
                df[indicator_cols] = df[indicator_cols].fillna(method='bfill')
                # Fill any remaining NaN (at the beginning) with 0
                df[indicator_cols] = df[indicator_cols].fillna(0)
            
            # Convert all columns to numeric, coercing errors to NaN
            numeric_cols = [col for col in df.columns if not col.startswith('target_')]
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Fill any NaN values created by coercion with 0
            df[numeric_cols] = df[numeric_cols].fillna(0)
            
            # Ensure all data types are float64 for consistency
            df[numeric_cols] = df[numeric_cols].astype('float64')
            
            logger.info(f"✅ Features prepared for {symbol}: {len(df)} samples, {len(df.columns)} features")
            logger.info(f"📊 Data types: {df.dtypes.value_counts().to_dict()}")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error preparing features for {symbol}: {e}")
            return None
    
    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get list of feature columns (excluding targets)"""
        target_prefixes = ['target_']
        feature_cols = [col for col in df.columns 
                       if not any(col.startswith(prefix) for prefix in target_prefixes)]
        return feature_cols
    
    def create_sequences(self, df: pd.DataFrame, mode: str = "unified") -> Dict:
        """
        Create training sequences for Progressive or Unified training
        
        Args:
            df: DataFrame with features and targets
            mode: "progressive" or "unified"
            
        Returns:
            Dictionary with training data for each horizon or unified data
        """
        try:
            feature_cols = self.get_feature_columns(df)
            
            if mode == "progressive":
                # Progressive: separate datasets for each horizon
                datasets = {}
                
                for horizon in self.horizons:
                    target_col = f'target_return_{horizon}d'
                    direction_col = f'target_direction_{horizon}d'
                    
                    if target_col not in df.columns:
                        logger.warning(f"⚠️ Target column {target_col} not found")
                        continue
                    
                    # Remove rows with NaN targets
                    valid_data = df.dropna(subset=[target_col, direction_col])
                    
                    if len(valid_data) < self.sequence_length + 10:
                        logger.warning(f"⚠️ Insufficient valid data for {horizon}d horizon")
                        continue
                    
                    # Create sequences
                    X, y_reg, y_clf = self._create_sequences_for_horizon(
                        valid_data[feature_cols].values,
                        valid_data[target_col].values,
                        valid_data[direction_col].values
                    )
                    
                    datasets[f'{horizon}d'] = {
                        'X': X,
                        'y_regression': y_reg,  # Price change %
                        'y_classification': y_clf,  # Direction (up/down)
                        'samples': len(X),
                        'features': len(feature_cols)
                    }
                    
                    logger.info(f"📊 Progressive {horizon}d: {len(X)} sequences, {len(feature_cols)} features")
                
                return datasets
                
            else:  # unified mode
                # Unified: single dataset with multi-target outputs
                
                # Get all target columns
                target_cols = []
                direction_cols = []
                for horizon in self.horizons:
                    target_col = f'target_return_{horizon}d'
                    direction_col = f'target_direction_{horizon}d'
                    if target_col in df.columns and direction_col in df.columns:
                        target_cols.append(target_col)
                        direction_cols.append(direction_col)
                
                # Remove rows with any NaN targets
                all_target_cols = target_cols + direction_cols
                valid_data = df.dropna(subset=all_target_cols)
                
                if len(valid_data) < self.sequence_length + 10:
                    logger.error("❌ Insufficient valid data for unified training")
                    return {}
                
                # Create sequences with multi-target outputs
                X, y_reg, y_clf = self._create_unified_sequences(
                    valid_data[feature_cols].values,
                    valid_data[target_cols].values,
                    valid_data[direction_cols].values
                )
                
                dataset = {
                    'X': X,
                    'y_regression': y_reg,  # Shape: (samples, num_horizons)
                    'y_classification': y_clf,  # Shape: (samples, num_horizons)  
                    'samples': len(X),
                    'features': len(feature_cols),
                    'horizons': self.horizons,
                    'target_names': {
                        'regression': target_cols,
                        'classification': direction_cols
                    }
                }
                
                logger.info(f"📊 Unified: {len(X)} sequences, {len(feature_cols)} features, {len(self.horizons)} horizons")
                return {'unified': dataset}
                
        except Exception as e:
            logger.error(f"❌ Error creating sequences: {e}")
            return {}
    
    def _create_sequences_for_horizon(self, features: np.ndarray, targets_reg: np.ndarray, targets_clf: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create sequences for a single horizon"""
        X, y_reg, y_clf = [], [], []
        
        for i in range(self.sequence_length, len(features)):
            # Feature sequence (last sequence_length days)
            X.append(features[i-self.sequence_length:i])
            
            # Targets for current day
            y_reg.append(targets_reg[i])
            y_clf.append(targets_clf[i])
        
        return np.array(X), np.array(y_reg), np.array(y_clf)
    
    def _create_unified_sequences(self, features: np.ndarray, targets_reg: np.ndarray, targets_clf: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create sequences for unified multi-horizon training"""
        X, y_reg, y_clf = [], [], []
        
        for i in range(self.sequence_length, len(features)):
            # Feature sequence (last sequence_length days)
            X.append(features[i-self.sequence_length:i])
            
            # Multi-horizon targets for current day
            y_reg.append(targets_reg[i])  # All horizons
            y_clf.append(targets_clf[i])  # All horizons
        
        return np.array(X), np.array(y_reg), np.array(y_clf)
    
    def split_data(self, dataset: Dict, validation_split: float = None) -> Dict:
        """Split data into train/validation sets"""
        if validation_split is None:
            validation_split = self.validation_split
        
        split_data = {}
        
        for key, data in dataset.items():
            if 'X' not in data:
                continue
                
            X = data['X']
            y_reg = data['y_regression']
            y_clf = data['y_classification']
            
            # Calculate split point
            split_idx = int(len(X) * (1 - validation_split))
            
            split_data[key] = {
                'X_train': X[:split_idx],
                'X_val': X[split_idx:],
                'y_reg_train': y_reg[:split_idx],
                'y_reg_val': y_reg[split_idx:],
                'y_clf_train': y_clf[:split_idx],
                'y_clf_val': y_clf[split_idx:],
                'train_samples': split_idx,
                'val_samples': len(X) - split_idx,
                'features': data['features'],
                'horizons': data.get('horizons', [key.replace('d', '')]),
                'target_names': data.get('target_names', {
                    'regression': [f'target_return_{key}'],
                    'classification': [f'target_direction_{key}']
                })
            }
            
            logger.info(f"📊 {key}: Train={split_idx}, Val={len(X) - split_idx}")
        
        return split_data
    
    def split_by_date(self, df: pd.DataFrame, split_date: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split DataFrame by date instead of random split
        
        Args:
            df: DataFrame with DateTimeIndex
            split_date: Date string (YYYY-MM-DD) to split on
            
        Returns:
            Tuple of (train_df, test_df)
        """
        try:
            split_date_parsed = pd.to_datetime(split_date)
            
            # Validate date is within DataFrame range
            if split_date_parsed < df.index.min():
                logger.warning(f"⚠️ Split date {split_date} is before data start {df.index.min().date()}")
                return df, pd.DataFrame()
            
            if split_date_parsed > df.index.max():
                logger.warning(f"⚠️ Split date {split_date} is after data end {df.index.max().date()}")
                return df, pd.DataFrame()
            
            # Split data
            train_df = df[df.index < split_date_parsed]
            test_df = df[df.index >= split_date_parsed]
            
            logger.info(f"📊 Split by date {split_date}:")
            logger.info(f"   Train: {len(train_df)} samples ({train_df.index[0].date()} to {train_df.index[-1].date()})")
            logger.info(f"   Test:  {len(test_df)} samples ({test_df.index[0].date()} to {test_df.index[-1].date()})")
            
            return train_df, test_df
            
        except Exception as e:
            logger.error(f"❌ Error splitting by date: {e}")
            # Fall back to regular split
            split_idx = int(len(df) * 0.8)
            return df[:split_idx], df[split_idx:]
    
    def validate_date_range(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> bool:
        """
        Validate that date range is logical
        
        Args:
            start_date: Start date string (YYYY-MM-DD)
            end_date: End date string (YYYY-MM-DD)
            
        Returns:
            True if valid, False otherwise
        """
        try:
            if start_date and end_date:
                start = pd.to_datetime(start_date)
                end = pd.to_datetime(end_date)
                
                if start >= end:
                    logger.error(f"❌ Start date {start_date} must be before end date {end_date}")
                    return False
                
                if end > pd.Timestamp.now():
                    logger.warning(f"⚠️ End date {end_date} is in the future")
                    return False
                
                # Check minimum data requirements
                days_diff = (end - start).days
                min_days = self.sequence_length + max(self.horizons) + 30
                
                if days_diff < min_days:
                    logger.error(f"❌ Date range too short. Need at least {min_days} days, got {days_diff}")
                    return False
            
            logger.info(f"✅ Date range validation passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Date validation error: {e}")
            return False


def test_progressive_dataloader():
    """Test the Progressive Data Loader"""
    print("🧪 Testing Progressive Data Loader...")
    
    # Initialize loader
    loader = ProgressiveDataLoader(
        stock_data_dir="stock_data",
        sequence_length=60,
        horizons=[1, 7, 30],
        use_fundamentals=True,
        use_technical_indicators=True
    )
    
    # Test with AAPL
    print("\n📊 Testing with AAPL...")
    features_df = loader.prepare_features("AAPL")
    
    if features_df is not None:
        print(f"✅ Features prepared: {len(features_df)} samples, {len(features_df.columns)} features")
        print(f"📈 Date range: {features_df.index[0].date()} to {features_df.index[-1].date()}")
        
        # Test Progressive mode
        print("\n🔄 Testing Progressive mode...")
        progressive_data = loader.create_sequences(features_df, mode="progressive")
        
        for horizon, data in progressive_data.items():
            print(f"   {horizon}: {data['samples']} sequences, {data['features']} features")
        
        # Test Unified mode
        print("\n🔗 Testing Unified mode...")
        unified_data = loader.create_sequences(features_df, mode="unified")
        
        if 'unified' in unified_data:
            data = unified_data['unified']
            print(f"   Unified: {data['samples']} sequences, {data['features']} features, {len(data['horizons'])} horizons")
        
        # Test train/validation split
        print("\n📊 Testing train/validation split...")
        split_progressive = loader.split_data(progressive_data)
        
        for horizon, data in split_progressive.items():
            print(f"   {horizon}: Train={data['train_samples']}, Val={data['val_samples']}")
        
        print("\n✅ Progressive Data Loader test completed successfully!")
        return True
    else:
        print("❌ Failed to prepare features for AAPL")
        return False


if __name__ == "__main__":
    test_progressive_dataloader()