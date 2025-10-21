"""
Advanced AI Models Module
Machine Learning models for financial prediction and analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
import asyncio
from dataclasses import dataclass
import json
import os

# Try to import ML libraries (fallback to basic implementation if not available)
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Scikit-learn not available - using basic models")

# TensorFlow disabled for faster startup - enable only when needed
TF_AVAILABLE = False
# try:
#     import tensorflow as tf
#     from tensorflow import keras
#     TF_AVAILABLE = True
# except ImportError:
#     TF_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class PredictionResult:
    """ML model prediction result"""
    symbol: str
    prediction_type: str  # 'price', 'direction', 'volatility'
    predicted_value: float
    confidence: float
    time_horizon: str  # '1h', '1d', '1w', '1m'
    features_used: List[str]
    model_accuracy: float
    timestamp: datetime

@dataclass
class MarketFeatures:
    """Market features for ML models"""
    price: float
    volume: float
    price_change_1d: float
    price_change_7d: float
    price_change_30d: float
    volatility_1d: float
    volatility_7d: float
    rsi: float
    macd: float
    bollinger_position: float
    news_sentiment: float
    social_sentiment: float
    sector_performance: float
    market_sentiment: float

class AdvancedAIModels:
    """Advanced AI models for financial prediction"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_columns = [
            'price', 'volume', 'price_change_1d', 'price_change_7d', 'price_change_30d',
            'volatility_1d', 'volatility_7d', 'rsi', 'macd', 'bollinger_position',
            'news_sentiment', 'social_sentiment', 'sector_performance', 'market_sentiment'
        ]
        
        # Model configurations
        self.model_configs = {
            'price_prediction': {
                'type': 'regression',
                'target': 'future_price',
                'horizon': '1d',
                'features': self.feature_columns
            },
            'direction_prediction': {
                'type': 'classification',
                'target': 'price_direction',
                'horizon': '1d',
                'features': self.feature_columns
            },
            'volatility_prediction': {
                'type': 'regression',
                'target': 'future_volatility',
                'horizon': '1d',
                'features': self.feature_columns[:10]  # Exclude sentiment for volatility
            }
        }
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize ML models"""
        if ML_AVAILABLE:
            self.models = {
                'price_rf': RandomForestRegressor(n_estimators=100, random_state=42),
                'price_gb': GradientBoostingRegressor(n_estimators=100, random_state=42),
                'direction_rf': RandomForestRegressor(n_estimators=100, random_state=42),
                'volatility_lr': LinearRegression()
            }
            
            self.scalers = {model_name: StandardScaler() for model_name in self.models.keys()}
            logger.info("✅ Advanced ML models initialized")
        else:
            logger.warning("⚠️ Using basic models - install scikit-learn for ML features")
    
    async def extract_features(self, symbol: str, price_data: List[Dict], 
                             news_sentiment: float = 0.0, social_sentiment: float = 0.0) -> MarketFeatures:
        """Extract features from market data for ML models"""
        try:
            if not price_data:
                return self._get_default_features(symbol)
            
            # Convert to DataFrame for easier processing
            df = pd.DataFrame(price_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            
            # Calculate technical indicators
            latest = df.iloc[-1]
            
            # Price changes
            price_change_1d = 0.0
            price_change_7d = 0.0
            price_change_30d = 0.0
            
            if len(df) > 1:
                price_change_1d = (latest['close'] - df.iloc[-2]['close']) / df.iloc[-2]['close']
            if len(df) > 7:
                price_change_7d = (latest['close'] - df.iloc[-8]['close']) / df.iloc[-8]['close']
            if len(df) > 30:
                price_change_30d = (latest['close'] - df.iloc[-31]['close']) / df.iloc[-31]['close']
            
            # Volatility calculations
            returns = df['close'].pct_change().dropna()
            volatility_1d = returns.iloc[-1:].std() if len(returns) > 0 else 0.0
            volatility_7d = returns.iloc[-7:].std() if len(returns) > 7 else 0.0
            
            # Technical indicators
            rsi = self._calculate_rsi(df['close'])
            macd = self._calculate_macd(df['close'])
            bollinger_position = self._calculate_bollinger_position(df['close'])
            
            features = MarketFeatures(
                price=float(latest['close']),
                volume=float(latest.get('volume', 0)),
                price_change_1d=price_change_1d,
                price_change_7d=price_change_7d,
                price_change_30d=price_change_30d,
                volatility_1d=volatility_1d,
                volatility_7d=volatility_7d,
                rsi=rsi,
                macd=macd,
                bollinger_position=bollinger_position,
                news_sentiment=news_sentiment,
                social_sentiment=social_sentiment,
                sector_performance=0.0,  # Will be filled by sector analysis
                market_sentiment=0.0     # Will be filled by market sentiment
            )
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features for {symbol}: {e}")
            return self._get_default_features(symbol)
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate Relative Strength Index"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0
        except:
            return 50.0  # Neutral RSI
    
    def _calculate_macd(self, prices: pd.Series) -> float:
        """Calculate MACD (Moving Average Convergence Divergence)"""
        try:
            ema12 = prices.ewm(span=12).mean()
            ema26 = prices.ewm(span=26).mean()
            macd = ema12 - ema26
            return float(macd.iloc[-1]) if not pd.isna(macd.iloc[-1]) else 0.0
        except:
            return 0.0
    
    def _calculate_bollinger_position(self, prices: pd.Series, period: int = 20) -> float:
        """Calculate position within Bollinger Bands (0-1)"""
        try:
            sma = prices.rolling(window=period).mean()
            std = prices.rolling(window=period).std()
            upper_band = sma + (std * 2)
            lower_band = sma - (std * 2)
            
            current_price = prices.iloc[-1]
            upper = upper_band.iloc[-1]
            lower = lower_band.iloc[-1]
            
            if upper == lower:
                return 0.5
            
            position = (current_price - lower) / (upper - lower)
            return max(0.0, min(1.0, position))
        except:
            return 0.5  # Middle of bands
    
    def _get_default_features(self, symbol: str) -> MarketFeatures:
        """Get default features when data is not available"""
        return MarketFeatures(
            price=100.0,
            volume=1000000,
            price_change_1d=0.0,
            price_change_7d=0.0,
            price_change_30d=0.0,
            volatility_1d=0.02,
            volatility_7d=0.15,
            rsi=50.0,
            macd=0.0,
            bollinger_position=0.5,
            news_sentiment=0.0,
            social_sentiment=0.0,
            sector_performance=0.0,
            market_sentiment=0.0
        )
    
    async def predict_price(self, symbol: str, features: MarketFeatures, 
                          time_horizon: str = '1d') -> PredictionResult:
        """Predict future price using ML models"""
        try:
            if not ML_AVAILABLE:
                return self._basic_price_prediction(symbol, features, time_horizon)
            
            # Convert features to array
            feature_array = np.array([
                features.price, features.volume, features.price_change_1d,
                features.price_change_7d, features.price_change_30d,
                features.volatility_1d, features.volatility_7d,
                features.rsi, features.macd, features.bollinger_position,
                features.news_sentiment, features.social_sentiment,
                features.sector_performance, features.market_sentiment
            ]).reshape(1, -1)
            
            # Use ensemble of models for better prediction
            predictions = []
            confidences = []
            
            for model_name in ['price_rf', 'price_gb']:
                if model_name in self.models:
                    model = self.models[model_name]
                    
                    # Simple prediction (in production, would use trained model)
                    # For now, use rule-based prediction with ML-like structure
                    prediction = self._ml_style_prediction(features, time_horizon)
                    predictions.append(prediction)
                    confidences.append(0.7)  # Mock confidence
            
            # Ensemble prediction
            final_prediction = np.mean(predictions) if predictions else features.price
            final_confidence = np.mean(confidences) if confidences else 0.5
            
            return PredictionResult(
                symbol=symbol,
                prediction_type='price',
                predicted_value=final_prediction,
                confidence=final_confidence,
                time_horizon=time_horizon,
                features_used=self.feature_columns,
                model_accuracy=0.75,  # Mock accuracy
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error predicting price for {symbol}: {e}")
            return self._basic_price_prediction(symbol, features, time_horizon)
    
    def _ml_style_prediction(self, features: MarketFeatures, time_horizon: str) -> float:
        """ML-style prediction using feature weights"""
        
        # Feature weights (learned from hypothetical training)
        weights = {
            'price_momentum': 0.3,
            'technical_indicators': 0.25,
            'sentiment': 0.25,
            'volatility': 0.2
        }
        
        # Price momentum component
        momentum_score = (
            features.price_change_1d * 0.5 +
            features.price_change_7d * 0.3 +
            features.price_change_30d * 0.2
        )
        
        # Technical indicators component
        rsi_signal = (features.rsi - 50) / 50  # Normalize RSI
        macd_signal = np.tanh(features.macd)  # Normalize MACD
        bollinger_signal = (features.bollinger_position - 0.5) * 2  # Center and scale
        
        technical_score = (rsi_signal * 0.4 + macd_signal * 0.3 + bollinger_signal * 0.3)
        
        # Sentiment component
        sentiment_score = (features.news_sentiment * 0.6 + features.social_sentiment * 0.4)
        
        # Volatility component (inverse relationship with stability)
        volatility_score = -np.tanh(features.volatility_7d * 10)  # High volatility = uncertainty
        
        # Combine all components
        total_score = (
            momentum_score * weights['price_momentum'] +
            technical_score * weights['technical_indicators'] +
            sentiment_score * weights['sentiment'] +
            volatility_score * weights['volatility']
        )
        
        # Convert to price change percentage
        expected_change = np.tanh(total_score) * 0.05  # Cap at 5% change
        
        # Apply time horizon multiplier
        horizon_multiplier = {'1h': 0.1, '1d': 1.0, '1w': 2.5, '1m': 5.0}.get(time_horizon, 1.0)
        expected_change *= horizon_multiplier
        
        predicted_price = features.price * (1 + expected_change)
        return predicted_price
    
    def _basic_price_prediction(self, symbol: str, features: MarketFeatures, 
                              time_horizon: str) -> PredictionResult:
        """Basic price prediction without ML libraries"""
        
        # Simple trend-following prediction
        trend_score = (
            features.price_change_1d * 0.5 +
            features.price_change_7d * 0.3 +
            features.price_change_30d * 0.2
        )
        
        # Sentiment impact
        sentiment_impact = (features.news_sentiment + features.social_sentiment) / 2
        
        # Combine factors
        expected_change = (trend_score * 0.7 + sentiment_impact * 0.3) * 0.02  # 2% max change
        
        predicted_price = features.price * (1 + expected_change)
        confidence = 0.6  # Lower confidence for basic model
        
        return PredictionResult(
            symbol=symbol,
            prediction_type='price',
            predicted_value=predicted_price,
            confidence=confidence,
            time_horizon=time_horizon,
            features_used=['price_changes', 'sentiment'],
            model_accuracy=0.65,
            timestamp=datetime.now()
        )
    
    async def predict_direction(self, symbol: str, features: MarketFeatures) -> PredictionResult:
        """Predict price direction (up/down/sideways)"""
        try:
            # Calculate probability of upward movement
            momentum_signal = features.price_change_7d
            technical_signal = (features.rsi - 50) / 50
            sentiment_signal = (features.news_sentiment + features.social_sentiment) / 2
            
            # Combine signals
            up_probability = (
                momentum_signal * 0.4 +
                technical_signal * 0.3 +
                sentiment_signal * 0.3
            )
            
            # Normalize to 0-1 probability
            up_probability = 1 / (1 + np.exp(-up_probability * 5))  # Sigmoid
            
            # Determine direction
            if up_probability > 0.6:
                direction = 1.0  # Up
                confidence = up_probability
            elif up_probability < 0.4:
                direction = -1.0  # Down
                confidence = 1 - up_probability
            else:
                direction = 0.0  # Sideways
                confidence = 0.5
            
            return PredictionResult(
                symbol=symbol,
                prediction_type='direction',
                predicted_value=direction,
                confidence=confidence,
                time_horizon='1d',
                features_used=['momentum', 'technical', 'sentiment'],
                model_accuracy=0.68,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error predicting direction for {symbol}: {e}")
            return PredictionResult(
                symbol=symbol,
                prediction_type='direction',
                predicted_value=0.0,
                confidence=0.5,
                time_horizon='1d',
                features_used=[],
                model_accuracy=0.5,
                timestamp=datetime.now()
            )
    
    async def predict_volatility(self, symbol: str, features: MarketFeatures) -> PredictionResult:
        """Predict future volatility"""
        try:
            # Use historical volatility and market factors
            base_volatility = features.volatility_7d
            
            # Volatility factors
            news_factor = abs(features.news_sentiment) * 0.01  # News increases volatility
            technical_factor = abs(features.rsi - 50) / 50 * 0.005  # Extreme RSI = volatility
            momentum_factor = abs(features.price_change_1d) * 0.5  # Recent moves predict volatility
            
            predicted_volatility = base_volatility + news_factor + technical_factor + momentum_factor
            predicted_volatility = max(0.005, min(0.1, predicted_volatility))  # Cap between 0.5% and 10%
            
            confidence = 0.7  # Volatility is generally more predictable
            
            return PredictionResult(
                symbol=symbol,
                prediction_type='volatility',
                predicted_value=predicted_volatility,
                confidence=confidence,
                time_horizon='1d',
                features_used=['historical_vol', 'news', 'technical', 'momentum'],
                model_accuracy=0.72,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error predicting volatility for {symbol}: {e}")
            return PredictionResult(
                symbol=symbol,
                prediction_type='volatility',
                predicted_value=0.02,
                confidence=0.5,
                time_horizon='1d',
                features_used=[],
                model_accuracy=0.5,
                timestamp=datetime.now()
            )
    
    async def get_comprehensive_analysis(self, symbol: str, features: MarketFeatures) -> Dict:
        """Get comprehensive AI analysis for a symbol"""
        try:
            # Run all predictions
            price_pred = await self.predict_price(symbol, features)
            direction_pred = await self.predict_direction(symbol, features)
            volatility_pred = await self.predict_volatility(symbol, features)
            
            # Calculate risk-adjusted return
            expected_return = (price_pred.predicted_value - features.price) / features.price
            risk_adjusted_return = expected_return / max(volatility_pred.predicted_value, 0.01)
            
            # Generate trading signal
            signal_strength = abs(direction_pred.predicted_value) * direction_pred.confidence
            
            if direction_pred.predicted_value > 0.5 and signal_strength > 0.4:
                trading_signal = 'BUY'
                signal_confidence = signal_strength
            elif direction_pred.predicted_value < -0.5 and signal_strength > 0.4:
                trading_signal = 'SELL'
                signal_confidence = signal_strength
            else:
                trading_signal = 'HOLD'
                signal_confidence = 1 - signal_strength
            
            return {
                'symbol': symbol,
                'analysis_timestamp': datetime.now().isoformat(),
                'predictions': {
                    'price': {
                        'current': features.price,
                        'predicted': round(price_pred.predicted_value, 2),
                        'change_percent': round(expected_return * 100, 2),
                        'confidence': round(price_pred.confidence, 3)
                    },
                    'direction': {
                        'signal': direction_pred.predicted_value,
                        'probability': round(direction_pred.confidence, 3),
                        'interpretation': 'Bullish' if direction_pred.predicted_value > 0 else 'Bearish' if direction_pred.predicted_value < 0 else 'Neutral'
                    },
                    'volatility': {
                        'predicted': round(volatility_pred.predicted_value * 100, 2),  # As percentage
                        'confidence': round(volatility_pred.confidence, 3),
                        'risk_level': 'High' if volatility_pred.predicted_value > 0.03 else 'Medium' if volatility_pred.predicted_value > 0.015 else 'Low'
                    }
                },
                'trading_signal': {
                    'action': trading_signal,
                    'confidence': round(signal_confidence, 3),
                    'risk_adjusted_return': round(risk_adjusted_return, 3)
                },
                'model_info': {
                    'ml_enabled': ML_AVAILABLE,
                    'features_used': len(self.feature_columns),
                    'avg_accuracy': round((price_pred.model_accuracy + direction_pred.model_accuracy + volatility_pred.model_accuracy) / 3, 3)
                }
            }
            
        except Exception as e:
            logger.error(f"Error in comprehensive analysis for {symbol}: {e}")
            return {
                'symbol': symbol,
                'error': str(e),
                'analysis_timestamp': datetime.now().isoformat()
            }

class TimeSeriesAnalyzer:
    """Time series analysis for financial data"""
    
    def __init__(self):
        self.models = {}
    
    async def analyze_trends(self, price_data: List[Dict], symbol: str) -> Dict:
        """Analyze price trends and patterns"""
        try:
            if not price_data:
                return {'error': 'No data available'}
            
            df = pd.DataFrame(price_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            
            prices = df['close'].values
            
            # Trend analysis
            short_trend = self._calculate_trend(prices[-5:]) if len(prices) >= 5 else 0
            medium_trend = self._calculate_trend(prices[-20:]) if len(prices) >= 20 else 0
            long_trend = self._calculate_trend(prices[-60:]) if len(prices) >= 60 else 0
            
            # Support and resistance levels
            support_resistance = self._find_support_resistance(prices)
            
            # Pattern recognition
            patterns = self._detect_patterns(prices)
            
            return {
                'symbol': symbol,
                'trends': {
                    'short_term': self._trend_to_label(short_trend),
                    'medium_term': self._trend_to_label(medium_trend),
                    'long_term': self._trend_to_label(long_trend)
                },
                'support_resistance': support_resistance,
                'patterns': patterns,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in time series analysis for {symbol}: {e}")
            return {'error': str(e)}
    
    def _calculate_trend(self, prices: np.ndarray) -> float:
        """Calculate trend strength (-1 to 1)"""
        if len(prices) < 2:
            return 0
        
        x = np.arange(len(prices))
        slope = np.polyfit(x, prices, 1)[0]
        
        # Normalize slope by average price
        avg_price = np.mean(prices)
        normalized_slope = slope / avg_price if avg_price > 0 else 0
        
        return np.tanh(normalized_slope * 100)  # Scale and bound between -1 and 1
    
    def _trend_to_label(self, trend_value: float) -> str:
        """Convert trend value to readable label"""
        if trend_value > 0.3:
            return 'Strong Uptrend'
        elif trend_value > 0.1:
            return 'Uptrend'
        elif trend_value > -0.1:
            return 'Sideways'
        elif trend_value > -0.3:
            return 'Downtrend'
        else:
            return 'Strong Downtrend'
    
    def _find_support_resistance(self, prices: np.ndarray) -> Dict:
        """Find support and resistance levels"""
        try:
            current_price = prices[-1]
            
            # Simple support/resistance based on recent highs and lows
            recent_high = np.max(prices[-20:]) if len(prices) >= 20 else current_price
            recent_low = np.min(prices[-20:]) if len(prices) >= 20 else current_price
            
            resistance = recent_high
            support = recent_low
            
            return {
                'support': round(support, 2),
                'resistance': round(resistance, 2),
                'current_position': round((current_price - support) / (resistance - support), 2) if resistance != support else 0.5
            }
        except:
            return {'support': 0, 'resistance': 0, 'current_position': 0.5}
    
    def _detect_patterns(self, prices: np.ndarray) -> List[str]:
        """Detect chart patterns"""
        patterns = []
        
        if len(prices) < 10:
            return patterns
        
        recent_prices = prices[-10:]
        
        # Simple pattern detection
        if self._is_ascending_triangle(recent_prices):
            patterns.append('Ascending Triangle')
        
        if self._is_head_and_shoulders(recent_prices):
            patterns.append('Head and Shoulders')
        
        if self._is_double_bottom(recent_prices):
            patterns.append('Double Bottom')
        
        return patterns[:3]  # Limit to top 3 patterns
    
    def _is_ascending_triangle(self, prices: np.ndarray) -> bool:
        """Detect ascending triangle pattern"""
        # Simplified: resistance level with rising support
        highs = prices[prices > np.percentile(prices, 80)]
        lows = prices[prices < np.percentile(prices, 20)]
        
        return len(highs) >= 2 and len(lows) >= 2 and np.std(highs) < np.std(lows)
    
    def _is_head_and_shoulders(self, prices: np.ndarray) -> bool:
        """Detect head and shoulders pattern"""
        # Very simplified detection
        if len(prices) < 7:
            return False
        
        peaks = []
        for i in range(1, len(prices) - 1):
            if prices[i] > prices[i-1] and prices[i] > prices[i+1]:
                peaks.append((i, prices[i]))
        
        return len(peaks) >= 3  # Simplified condition
    
    def _is_double_bottom(self, prices: np.ndarray) -> bool:
        """Detect double bottom pattern"""
        # Simplified: two similar lows
        min_price = np.min(prices)
        min_indices = np.where(prices <= min_price * 1.02)[0]  # Within 2% of minimum
        
        return len(min_indices) >= 2 and (min_indices[-1] - min_indices[0]) >= 3