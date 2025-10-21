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
                 validation_split: float = 0.2):
        """
        Initialize Progressive Data Loader
        
        Args:
            stock_data_dir: Directory containing stock data folders
            sequence_length: Number of days to look back for predictions
            horizons: Prediction horizons in days [1, 7, 30]
            use_fundamentals: Include fundamental data from JSON files
            use_technical_indicators: Calculate technical indicators
            validation_split: Fraction of data for validation
        """
        self.stock_data_dir = Path(stock_data_dir)
        self.sequence_length = sequence_length
        self.horizons = sorted(horizons)  # [1, 7, 30]
        self.use_fundamentals = use_fundamentals
        self.use_technical_indicators = use_technical_indicators
        self.validation_split = validation_split
        
        # Feature cache for efficiency
        self._feature_cache = {}
        self._fundamental_cache = {}
        
        logger.info(f"🚀 Progressive DataLoader initialized")
        logger.info(f"   📊 Horizons: {self.horizons} days")
        logger.info(f"   📈 Sequence length: {self.sequence_length} days")
        logger.info(f"   📋 Fundamentals: {'✅' if use_fundamentals else '❌'}")
        logger.info(f"   🔧 Technical indicators: {'✅' if use_technical_indicators else '❌'}")
    
    def load_stock_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load CSV price data for a symbol"""
        try:
            csv_path = self.stock_data_dir / symbol / f"{symbol}_price.csv"
            
            if not csv_path.exists():
                logger.warning(f"⚠️ Price data not found: {csv_path}")
                return None
            
            # Load CSV with proper date parsing
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            
            # Ensure columns are in correct order and named properly
            expected_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in df.columns for col in expected_columns):
                logger.error(f"❌ Missing required columns in {symbol} data. Expected: {expected_columns}")
                return None
            
            # Sort by date
            df = df.sort_index()
            
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
    
    def load_fundamental_data(self, symbol: str) -> Dict:
        """Load JSON fundamental data for a symbol"""
        try:
            json_path = self.stock_data_dir / symbol / f"{symbol}_advanced.json"
            
            if not json_path.exists():
                logger.warning(f"⚠️ Fundamental data not found: {json_path}")
                return {}
            
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # Convert percentage strings to floats
            fundamental_features = {}
            
            # Key fundamental ratios
            key_metrics = {
                'P/E': 'pe_ratio',
                'EPS (ttm)': 'eps_ttm', 
                'P/S': 'ps_ratio',
                'P/B': 'pb_ratio',
                'PEG': 'peg_ratio',
                'ROE': 'roe',
                'ROI': 'roi',
                'Debt/Eq': 'debt_to_equity',
                'Market Cap': 'market_cap',
                'Forward P/E': 'forward_pe'
            }
            
            for json_key, feature_name in key_metrics.items():
                if json_key in data:
                    value = self._parse_financial_value(data[json_key])
                    if value is not None:
                        fundamental_features[feature_name] = value
            
            logger.info(f"✅ Loaded {symbol} fundamentals: {len(fundamental_features)} features")
            return fundamental_features
            
        except Exception as e:
            logger.warning(f"⚠️ Error loading {symbol} fundamentals: {e}")
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
            
            # Basic price features
            df['returns'] = df['Close'].pct_change()
            df['log_returns'] = np.log(df['Close']).diff()
            df['volatility'] = df['returns'].rolling(20).std()
            df['volume_ma'] = df['Volume'].rolling(20).mean()
            df['price_volume_trend'] = (df['Close'].pct_change() * df['Volume']).rolling(10).mean()
            
            # Moving averages
            for period in [5, 10, 20, 50]:
                df[f'sma_{period}'] = df['Close'].rolling(period).mean()
                df[f'ema_{period}'] = df['Close'].ewm(span=period).mean()
            
            # Price relative to MAs
            df['price_above_sma20'] = (df['Close'] > df['sma_20']).astype(int)
            df['price_above_sma50'] = (df['Close'] > df['sma_50']).astype(int)
            
            # Bollinger Bands
            bb_period = 20
            bb_std = 2
            df['bb_middle'] = df['Close'].rolling(bb_period).mean()
            bb_std_val = df['Close'].rolling(bb_period).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std_val * bb_std)
            df['bb_lower'] = df['bb_middle'] - (bb_std_val * bb_std)
            df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            if TALIB_AVAILABLE:
                # Advanced indicators with TA-Lib
                df['rsi'] = talib.RSI(df['Close'].values, timeperiod=14)
                df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(df['Close'].values)
                df['atr'] = talib.ATR(df['High'].values, df['Low'].values, df['Close'].values)
                df['adx'] = talib.ADX(df['High'].values, df['Low'].values, df['Close'].values)
                df['cci'] = talib.CCI(df['High'].values, df['Low'].values, df['Close'].values)
                df['williams_r'] = talib.WILLR(df['High'].values, df['Low'].values, df['Close'].values)
                df['stoch_k'], df['stoch_d'] = talib.STOCH(df['High'].values, df['Low'].values, df['Close'].values)
            else:
                # Manual calculations
                df['rsi'] = self._calculate_rsi(df['Close'])
                df['macd'], df['macd_signal'] = self._calculate_macd(df['Close'])
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
    
    def _calculate_macd(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Manual MACD calculation"""
        ema12 = prices.ewm(span=12).mean()
        ema26 = prices.ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
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
    
    def prepare_features(self, symbol: str) -> Optional[pd.DataFrame]:
        """Prepare complete feature set for a symbol"""
        try:
            # Load price data
            df = self.load_stock_data(symbol)
            if df is None:
                return None
            
            # Calculate technical indicators
            if self.use_technical_indicators:
                df = self.calculate_technical_indicators(df)
            
            # Add fundamental features (broadcast to all rows)
            if self.use_fundamentals:
                fundamentals = self.load_fundamental_data(symbol)
                for feature_name, value in fundamentals.items():
                    df[f'fund_{feature_name}'] = value
            
            # Create multi-horizon targets
            df = self.create_multi_horizon_targets(df)
            
            # Remove rows that don't have targets (last N rows)
            max_horizon = max(self.horizons)
            df = df.iloc[:-max_horizon]  # Remove last max_horizon rows
            
            logger.info(f"✅ Features prepared for {symbol}: {len(df)} samples, {len(df.columns)} features")
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