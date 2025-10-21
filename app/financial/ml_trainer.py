"""
Advanced ML Training Module for MarketPulse AI
Real ML model training, persistence, and historical data analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
import asyncio
import json
import pickle
import os
from pathlib import Path

# ML Libraries
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    from sklearn.pipeline import Pipeline
    import joblib
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

# TensorFlow for Neural Networks
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, optimizers, callbacks
    from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# Database persistence
try:
    import sqlite3
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False

logger = logging.getLogger(__name__)

class MLModelTrainer:
    """Advanced ML model trainer with persistence and historical analysis"""
    
    def __init__(self, data_path: str = "data/models"):
        self.data_path = Path(data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)
        
        self.models = {}
        self.scalers = {}
        self.performance_metrics = {}
        
        # Model configurations
        self.model_configs = {
            'random_forest': {
                'type': RandomForestRegressor,
                'params': {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'random_state': 42
                }
            },
            'gradient_boost': {
                'type': GradientBoostingRegressor,
                'params': {
                    'n_estimators': 100,
                    'learning_rate': 0.1,
                    'max_depth': 6,
                    'random_state': 42
                }
            },
            'ridge_regression': {
                'type': Ridge,
                'params': {
                    'alpha': 1.0,
                    'random_state': 42
                }
            }
        }
        
        if ML_AVAILABLE:
            logger.info("✅ Advanced ML trainer initialized")
        else:
            logger.warning("⚠️ ML libraries not available")
    
    async def collect_historical_data(self, symbol: str, days: int = 365) -> pd.DataFrame:
        """Collect and prepare historical data for training"""
        try:
            from app.financial.market_data import FinancialDataProvider
            
            financial_provider = FinancialDataProvider()
            
            # Get historical price data
            price_data = await financial_provider.get_stock_data(symbol, days=days)
            
            if not price_data:
                raise ValueError(f"No data available for {symbol}")
            
            # Convert to DataFrame
            df = pd.DataFrame(price_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Add technical indicators
            df = self._add_technical_features(df)
            
            # Add time-based features
            df = self._add_time_features(df)
            
            # Add target variables
            df = self._add_targets(df)
            
            logger.info(f"✅ Collected {len(df)} historical records for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error collecting historical data for {symbol}: {e}")
            return pd.DataFrame()
    
    def _add_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical analysis features"""
        # Price-based features
        df['price_change_1d'] = df['close'].pct_change()
        df['price_change_3d'] = df['close'].pct_change(periods=3)
        df['price_change_7d'] = df['close'].pct_change(periods=7)
        df['price_change_14d'] = df['close'].pct_change(periods=14)
        df['price_change_30d'] = df['close'].pct_change(periods=30)
        
        # Moving averages
        df['sma_5'] = df['close'].rolling(window=5).mean()
        df['sma_10'] = df['close'].rolling(window=10).mean()
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        
        # Exponential moving averages
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Volatility
        df['volatility_5d'] = df['close'].rolling(window=5).std()
        df['volatility_20d'] = df['close'].rolling(window=20).std()
        
        # Volume features
        if 'volume' in df.columns:
            df['volume_sma_10'] = df['volume'].rolling(window=10).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma_10']
        
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        df['quarter'] = df['timestamp'].dt.quarter
        
        # Market session indicators
        df['is_market_open'] = (df['hour'] >= 9) & (df['hour'] <= 16)
        df['is_weekend'] = df['day_of_week'].isin([5, 6])
        
        return df
    
    def _add_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add target variables for prediction"""
        # Future prices (1h, 1d, 3d, 7d ahead)
        df['target_price_1h'] = df['close'].shift(-1)  # Next hour
        df['target_price_1d'] = df['close'].shift(-24)  # Next day
        df['target_price_3d'] = df['close'].shift(-72)  # 3 days
        df['target_price_7d'] = df['close'].shift(-168)  # 7 days
        
        # Price direction (binary classification)
        df['target_direction_1h'] = (df['target_price_1h'] > df['close']).astype(int)
        df['target_direction_1d'] = (df['target_price_1d'] > df['close']).astype(int)
        
        # Price change percentage
        df['target_change_1h'] = (df['target_price_1h'] - df['close']) / df['close']
        df['target_change_1d'] = (df['target_price_1d'] - df['close']) / df['close']
        
        # Volatility targets
        df['target_volatility_1d'] = df['close'].rolling(window=24).std().shift(-24)
        
        return df
    
    async def train_price_prediction_model(self, symbol: str, horizon: str = '1d') -> Dict:
        """Train price prediction model"""
        try:
            # Collect data
            df = await self.collect_historical_data(symbol, days=730)  # 2 years
            
            if len(df) < 100:
                raise ValueError("Insufficient data for training")
            
            # Prepare features
            feature_columns = [
                'price_change_1d', 'price_change_3d', 'price_change_7d', 'price_change_14d',
                'sma_5', 'sma_10', 'sma_20', 'macd', 'rsi', 'bb_position',
                'volatility_5d', 'volatility_20d', 'hour', 'day_of_week', 'month'
            ]
            
            # Add volume features if available
            if 'volume_ratio' in df.columns:
                feature_columns.append('volume_ratio')
            
            target_column = f'target_change_{horizon}'
            
            # Clean data
            feature_data = df[feature_columns + [target_column]].dropna()
            
            if len(feature_data) < 50:
                raise ValueError("Insufficient clean data after feature engineering")
            
            X = feature_data[feature_columns].values
            y = feature_data[target_column].values
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=False
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train models
            results = {}
            
            for model_name, config in self.model_configs.items():
                model = config['type'](**config['params'])
                
                # Train model
                model.fit(X_train_scaled, y_train)
                
                # Predictions
                train_pred = model.predict(X_train_scaled)
                test_pred = model.predict(X_test_scaled)
                
                # Metrics
                train_mse = mean_squared_error(y_train, train_pred)
                test_mse = mean_squared_error(y_test, test_pred)
                train_r2 = r2_score(y_train, train_pred)
                test_r2 = r2_score(y_test, test_pred)
                
                results[model_name] = {
                    'train_mse': train_mse,
                    'test_mse': test_mse,
                    'train_r2': train_r2,
                    'test_r2': test_r2,
                    'model': model,
                    'scaler': scaler
                }
                
                logger.info(f"✅ {model_name} trained - Test R²: {test_r2:.4f}")
            
            # Select best model
            best_model_name = max(results.keys(), key=lambda k: results[k]['test_r2'])
            best_result = results[best_model_name]
            
            # Save best model
            model_filename = self.data_path / f"{symbol}_{horizon}_price_model.joblib"
            scaler_filename = self.data_path / f"{symbol}_{horizon}_scaler.joblib"
            
            joblib.dump(best_result['model'], model_filename)
            joblib.dump(best_result['scaler'], scaler_filename)
            
            # Store in memory
            self.models[f"{symbol}_{horizon}"] = best_result['model']
            self.scalers[f"{symbol}_{horizon}"] = best_result['scaler']
            
            training_result = {
                'symbol': symbol,
                'horizon': horizon,
                'best_model': best_model_name,
                'test_r2': best_result['test_r2'],
                'test_mse': best_result['test_mse'],
                'feature_columns': feature_columns,
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'all_results': {k: {
                    'test_r2': v['test_r2'],
                    'test_mse': v['test_mse']
                } for k, v in results.items()},
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"✅ Price prediction model trained for {symbol} ({horizon}) - R²: {best_result['test_r2']:.4f}")
            return training_result
            
        except Exception as e:
            logger.error(f"Error training price prediction model: {e}")
            return {'error': str(e)}
    
    async def predict_with_trained_model(self, symbol: str, features: Dict, horizon: str = '1d') -> Dict:
        """Make prediction using trained model"""
        try:
            model_key = f"{symbol}_{horizon}"
            
            if model_key not in self.models:
                # Try to load from disk
                model_filename = self.data_path / f"{symbol}_{horizon}_price_model.joblib"
                scaler_filename = self.data_path / f"{symbol}_{horizon}_scaler.joblib"
                
                if model_filename.exists() and scaler_filename.exists():
                    self.models[model_key] = joblib.load(model_filename)
                    self.scalers[model_key] = joblib.load(scaler_filename)
                else:
                    return {'error': f'No trained model found for {symbol} ({horizon})'}
            
            model = self.models[model_key]
            scaler = self.scalers[model_key]
            
            # Prepare features
            feature_array = np.array([list(features.values())]).reshape(1, -1)
            
            # Scale features
            feature_scaled = scaler.transform(feature_array)
            
            # Make prediction
            prediction = model.predict(feature_scaled)[0]
            
            # Get model confidence (for tree-based models)
            if hasattr(model, 'predict_proba'):
                confidence = 0.8  # Default for regression
            else:
                confidence = 0.7
            
            return {
                'prediction': float(prediction),
                'confidence': confidence,
                'model_type': type(model).__name__,
                'horizon': horizon,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return {'error': str(e)}
    
    async def train_neural_network(self, symbol: str, sequence_length: int = 60) -> Dict:
        """Train LSTM neural network for time series prediction"""
        if not TF_AVAILABLE:
            return {'error': 'TensorFlow not available'}
        
        try:
            # Collect data
            df = await self.collect_historical_data(symbol, days=1000)  # More data for neural networks
            
            if len(df) < sequence_length * 3:
                raise ValueError("Insufficient data for neural network training")
            
            # Prepare sequences
            prices = df['close'].values
            scaler = MinMaxScaler()
            prices_scaled = scaler.fit_transform(prices.reshape(-1, 1))
            
            # Create sequences
            X, y = [], []
            for i in range(sequence_length, len(prices_scaled)):
                X.append(prices_scaled[i-sequence_length:i, 0])
                y.append(prices_scaled[i, 0])
            
            X, y = np.array(X), np.array(y)
            X = X.reshape((X.shape[0], X.shape[1], 1))
            
            # Split data
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Build LSTM model
            model = models.Sequential([
                layers.LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)),
                layers.Dropout(0.2),
                layers.LSTM(50, return_sequences=False),
                layers.Dropout(0.2),
                layers.Dense(25),
                layers.Dense(1)
            ])
            
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            
            # Callbacks
            early_stopping = callbacks.EarlyStopping(
                monitor='val_loss', patience=10, restore_best_weights=True
            )
            
            # Train model
            history = model.fit(
                X_train, y_train,
                batch_size=32,
                epochs=50,
                validation_data=(X_test, y_test),
                callbacks=[early_stopping],
                verbose=0
            )
            
            # Evaluate
            train_loss = model.evaluate(X_train, y_train, verbose=0)
            test_loss = model.evaluate(X_test, y_test, verbose=0)
            
            # Save model
            model_path = self.data_path / f"{symbol}_lstm_model.h5"
            scaler_path = self.data_path / f"{symbol}_lstm_scaler.joblib"
            
            model.save(model_path)
            joblib.dump(scaler, scaler_path)
            
            result = {
                'symbol': symbol,
                'model_type': 'LSTM',
                'sequence_length': sequence_length,
                'train_loss': float(train_loss[0]),
                'test_loss': float(test_loss[0]),
                'train_mae': float(train_loss[1]),
                'test_mae': float(test_loss[1]),
                'epochs_trained': len(history.history['loss']),
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"✅ LSTM model trained for {symbol} - Test Loss: {test_loss[0]:.6f}")
            return result
            
        except Exception as e:
            logger.error(f"Error training neural network: {e}")
            return {'error': str(e)}
    
    async def get_model_performance_summary(self) -> Dict:
        """Get performance summary of all trained models"""
        try:
            summary = {
                'models_available': len(self.models),
                'ml_library_status': ML_AVAILABLE,
                'tensorflow_status': TF_AVAILABLE,
                'models': []
            }
            
            # Get model files
            model_files = list(self.data_path.glob("*_model.joblib")) + list(self.data_path.glob("*_model.h5"))
            
            for model_file in model_files:
                parts = model_file.stem.split('_')
                if len(parts) >= 3:
                    symbol = parts[0]
                    model_type = '_'.join(parts[1:-1])
                    
                    summary['models'].append({
                        'symbol': symbol,
                        'type': model_type,
                        'file': model_file.name,
                        'size_mb': model_file.stat().st_size / (1024 * 1024),
                        'modified': datetime.fromtimestamp(model_file.stat().st_mtime).isoformat()
                    })
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting model performance summary: {e}")
            return {'error': str(e)}

class HistoricalDataManager:
    """Manages historical data storage and retrieval"""
    
    def __init__(self, db_path: str = "data/historical.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        if DB_AVAILABLE:
            self._init_database()
        
    def _init_database(self):
        """Initialize SQLite database for historical data"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS price_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, timestamp)
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    horizon TEXT NOT NULL,
                    prediction REAL NOT NULL,
                    confidence REAL,
                    actual_value REAL,
                    timestamp DATETIME NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_price_symbol_time ON price_data(symbol, timestamp);
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_predictions_symbol_time ON model_predictions(symbol, timestamp);
            ''')
            
            conn.commit()
            conn.close()
            
            logger.info("✅ Historical data database initialized")
            
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
    
    async def store_price_data(self, symbol: str, data: List[Dict]):
        """Store historical price data"""
        if not DB_AVAILABLE:
            return False
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for record in data:
                cursor.execute('''
                    INSERT OR REPLACE INTO price_data 
                    (symbol, timestamp, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    symbol,
                    record['timestamp'],
                    record.get('open'),
                    record.get('high'),
                    record.get('low'),
                    record['close'],
                    record.get('volume')
                ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"✅ Stored {len(data)} price records for {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing price data: {e}")
            return False
    
    async def get_stored_data(self, symbol: str, days: int = 30) -> List[Dict]:
        """Retrieve stored historical data"""
        if not DB_AVAILABLE:
            return []
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT timestamp, open, high, low, close, volume
                FROM price_data
                WHERE symbol = ? AND timestamp >= datetime('now', '-{} days')
                ORDER BY timestamp
            '''.format(days), (symbol,))
            
            rows = cursor.fetchall()
            conn.close()
            
            return [
                {
                    'timestamp': row[0],
                    'open': row[1],
                    'high': row[2],
                    'low': row[3],
                    'close': row[4],
                    'volume': row[5]
                }
                for row in rows
            ]
            
        except Exception as e:
            logger.error(f"Error retrieving stored data: {e}")
            return []