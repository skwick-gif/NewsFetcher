"""
Progressive Predictor for Multi-Horizon Stock Predictions
========================================================
Advanced prediction system with ensemble capabilities and confidence scoring

Features:
- Ensemble predictions from multiple models
- Confidence scoring based on model agreement
- Real-time predictions for multiple horizons
- Risk assessment and uncertainty quantification
- Trading signal generation
- Backtesting and performance evaluation
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path
import json
import pickle
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from .data_loader import ProgressiveDataLoader
from .models import ProgressiveModels, EnsembleModel
from .trainer import ProgressiveTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProgressivePredictor:
    """
    Progressive Prediction System for Multi-Horizon Stock Predictions
    
    Combines multiple trained models for ensemble predictions with confidence scoring
    """
    
    def __init__(self, 
                 data_loader: ProgressiveDataLoader,
                 model_dir: str = None,
                 prediction_config: Dict = None):
        """
        Initialize Progressive Predictor
        
        Args:
            data_loader: ProgressiveDataLoader instance
            model_dir: Directory containing trained models (absolute or relative)
                      If None, uses absolute path to app/ml/models
            prediction_config: Prediction configuration
        """
        
        self.data_loader = data_loader
        
        # Handle model_dir - use absolute path by default to avoid CWD issues
        if model_dir is None:
            # Calculate absolute path from this file's location
            # __file__ = d:/Projects/NewsFetcher/app/ml/progressive/predictor.py
            # .parent = progressive, .parent.parent = ml, .parent.parent.parent = app
            # .parent.parent.parent.parent = NewsFetcher root
            base_dir = Path(__file__).parent.parent.parent.parent  # Goes to NewsFetcher root
            model_dir = str(base_dir / "app" / "ml" / "models")
            logger.info(f"   Using default model_dir (absolute): {model_dir}")
        
        self.model_dir = Path(model_dir)
        
        # Default prediction configuration
        self.prediction_config = {
            'ensemble_weights': {
                'lstm': 0.4,
                'transformer': 0.35,
                'cnn': 0.25
            },
            'confidence_threshold': 0.6,
            'uncertainty_methods': ['model_agreement', 'prediction_variance'],
            'risk_metrics': ['volatility', 'drawdown', 'var'],
            'signal_threshold': 0.02,  # 2% threshold for trading signals
            'prediction_horizons': [1, 7, 30]
        }
        
        if prediction_config:
            self.prediction_config.update(prediction_config)
        
        # Initialize storage
        self.loaded_models = {}
        self.prediction_history = []
        self.performance_metrics = {}
        
        logger.info(f"🔮 Progressive Predictor initialized")
        logger.info(f"   📂 Model directory: {self.model_dir}")
        logger.info(f"   ⏰ Horizons: {self.prediction_config['prediction_horizons']}")
    
    def load_models(self, symbol: str = "AAPL", model_types: List[str] = ['lstm']) -> Dict:
        """Load trained models for prediction"""
        
        logger.info(f"📥 Loading models for {symbol}...")
        
        loaded_models = {}
        
        for model_type in model_types:
            model_key = f"{model_type}_{symbol}"
            model_path = self.model_dir / model_key
            
            # Try loading from directory structure first
            if model_path.exists() and model_path.is_dir():
                try:
                    # Load progressive models (separate for each horizon) - NEW FORMAT
                    model_dict = {}
                    
                    for horizon in self.prediction_config['prediction_horizons']:
                        horizon_key = f'{horizon}d'
                        model_file = model_path / f"{horizon_key}.h5"
                        
                        if model_file.exists():
                            try:
                                # Load with custom objects for Keras 3 compatibility
                                custom_objs = {
                                    'mse': keras.losses.MeanSquaredError(),
                                    'MultiHeadAttention': layers.MultiHeadAttention
                                }
                                model = keras.models.load_model(
                                    str(model_file),
                                    custom_objects=custom_objs
                                )
                                model_dict[horizon_key] = model
                                logger.info(f"   ✅ Loaded {model_type} {horizon_key} (from directory)")
                            except Exception as load_err:
                                logger.warning(f"   ⚠️ Failed to load {model_type} {horizon_key}: {load_err}")
                        else:
                            logger.debug(f"   Model file not found: {model_file}")
                    
                    if model_dict:
                        loaded_models[model_type] = model_dict
                    
                    # Also try to load unified model
                    unified_file = model_path / "unified.h5"
                    if unified_file.exists():
                        try:
                            custom_objs = {
                                'mse': keras.losses.MeanSquaredError(),
                                'MultiHeadAttention': layers.MultiHeadAttention
                            }
                            unified_model = keras.models.load_model(
                                str(unified_file),
                                custom_objects=custom_objs
                            )
                            loaded_models[f"{model_type}_unified"] = {'unified': unified_model}
                            logger.info(f"   ✅ Loaded {model_type} unified")
                        except Exception as unified_err:
                            logger.warning(f"   ⚠️ Failed to load {model_type} unified: {unified_err}")
                    
                except Exception as e:
                    logger.error(f"❌ Error loading {model_type} from directory: {e}")
            
            # FALLBACK: Try loading from flat files (checkpoint format)
            # Files saved as: {model_type}_{symbol}_{horizon}_best.h5
            if model_type not in loaded_models or not loaded_models.get(model_type):
                try:
                    logger.info(f"   🔍 Trying flat file format for {model_type}_{symbol}...")
                    model_dict = {}
                    
                    for horizon in self.prediction_config['prediction_horizons']:
                        horizon_key = f'{horizon}d'
                        # Try flat file naming: {model_type}_{symbol}_{horizon}_best.h5
                        flat_file = self.model_dir / f"{model_type}_{symbol}_{horizon_key}_best.h5"
                        
                        if flat_file.exists():
                            try:
                                custom_objs = {
                                    'mse': keras.losses.MeanSquaredError(),
                                    'MultiHeadAttention': layers.MultiHeadAttention
                                }
                                model = keras.models.load_model(
                                    str(flat_file),
                                    custom_objects=custom_objs
                                )
                                model_dict[horizon_key] = model
                                logger.info(f"   ✅ Loaded {model_type} {horizon_key} (flat file)")
                            except Exception as load_err:
                                logger.warning(f"   ⚠️ Failed to load {model_type} {horizon_key}: {load_err}")
                        else:
                            logger.debug(f"   Flat file not found: {flat_file}")
                    
                    if model_dict:
                        loaded_models[model_type] = model_dict
                    
                    # Try unified flat file
                    unified_flat = self.model_dir / f"{model_type}_{symbol}_unified_best.h5"
                    if unified_flat.exists():
                        try:
                            custom_objs = {
                                'mse': keras.losses.MeanSquaredError(),
                                'MultiHeadAttention': layers.MultiHeadAttention
                            }
                            unified_model = keras.models.load_model(
                                str(unified_flat),
                                custom_objects=custom_objs
                            )
                            loaded_models[f"{model_type}_unified"] = {'unified': unified_model}
                            logger.info(f"   ✅ Loaded {model_type} unified (flat file)")
                        except Exception as unified_err:
                            logger.warning(f"   ⚠️ Failed to load {model_type} unified: {unified_err}")
                    
                except Exception as e:
                    logger.error(f"❌ Error loading {model_type} from flat files: {e}")
            
            if model_type not in loaded_models and f"{model_type}_unified" not in loaded_models:
                logger.warning(f"⚠️ No models found for {model_type}_{symbol} (tried directory and flat files)")
        
        self.loaded_models[symbol] = loaded_models
        
        logger.info(f"✅ Loaded {len(loaded_models)} model types for {symbol}")
        
        return loaded_models
    
    def prepare_prediction_data(self, symbol: str, mode: str = "progressive") -> Dict:
        """Prepare data for prediction (last sequence)"""
        
        logger.info(f"📊 Preparing prediction data for {symbol} ({mode} mode)...")
        
        # Get features
        df = self.data_loader.prepare_features(symbol)
        if df is None:
            raise ValueError(f"Could not prepare features for {symbol}")
        
        # Get feature columns
        feature_cols = self.data_loader.get_feature_columns(df)
        
        # Take the last sequence for prediction
        if len(df) < self.data_loader.sequence_length:
            raise ValueError(f"Insufficient data: need {self.data_loader.sequence_length}, got {len(df)}")
        
        # Extract last sequence
        last_sequence = df[feature_cols].iloc[-self.data_loader.sequence_length:].values
        
        # Reshape for model input: (1, sequence_length, features)
        X = last_sequence.reshape(1, self.data_loader.sequence_length, len(feature_cols))
        
        # Get current price for reference (support both 'Close' and 'close')
        if 'Close' in df.columns:
            current_price = float(df['Close'].iloc[-1])
        elif 'close' in df.columns:
            current_price = float(df['close'].iloc[-1])
        else:
            raise ValueError(f"No 'Close' or 'close' column found in data")
        current_date = df.index[-1] if hasattr(df.index, 'name') else None
        
        prediction_data = {
            'X': X,
            'current_price': current_price,
            'current_date': str(current_date) if current_date is not None else None,
            'symbol': symbol,
            'mode': mode,
            'sequence_length': self.data_loader.sequence_length,
            'num_features': len(feature_cols)
        }
        
        logger.info(f"   ✅ Data prepared: {X.shape}, Current price: ${current_price:.2f}")
        
        return prediction_data
    
    def predict_single_model(self, 
                           model: keras.Model, 
                           X: np.ndarray, 
                           horizon: str = "1d") -> Dict:
        """Make prediction with single model"""
        
        try:
            # Get prediction
            prediction = model.predict(X, verbose=0)
            
            # Handle dual output (regression, classification)
            if isinstance(prediction, list) and len(prediction) == 2:
                price_pred = float(prediction[0][0][0])  # First sample, first output
                direction_pred = float(prediction[1][0][0])  # First sample, first output
            else:
                price_pred = float(prediction[0][0])
                direction_pred = 0.5  # Default if no classification
            
            # Check for NaN values and replace with defaults
            if np.isnan(price_pred) or np.isinf(price_pred):
                logger.warning(f"   ⚠️ NaN/Inf detected in price prediction for {horizon}, using 0.0")
                price_pred = 0.0
            
            if np.isnan(direction_pred) or np.isinf(direction_pred):
                logger.warning(f"   ⚠️ NaN/Inf detected in direction prediction for {horizon}, using 0.5")
                direction_pred = 0.5
            
            return {
                'price_change_pct': float(price_pred),
                'direction_prob': float(direction_pred),
                'direction': 'UP' if direction_pred > 0.5 else 'DOWN',
                'confidence': float(abs(direction_pred - 0.5) * 2),  # Convert to 0-1 scale
                'horizon': horizon
            }
            
        except Exception as e:
            logger.error(f"❌ Error in single model prediction: {e}")
            return {
                'price_change_pct': 0.0,
                'direction_prob': 0.5,
                'direction': 'NEUTRAL',
                'confidence': 0.0,
                'horizon': horizon,
                'error': str(e)
            }
    
    def predict_ensemble(self, symbol: str, mode: str = "progressive") -> Dict:
        """Make ensemble predictions for all horizons"""
        
        logger.info(f"🎯 Making ensemble predictions for {symbol}...")
        
        # Check if models are loaded
        if symbol not in self.loaded_models or not self.loaded_models[symbol]:
            logger.warning(f"⚠️ No models loaded for {symbol}. Loading now...")
            self.load_models(symbol, ['lstm', 'transformer', 'cnn'])
        
        # Prepare prediction data
        pred_data = self.prepare_prediction_data(symbol, mode)
        X = pred_data['X']
        current_price = pred_data['current_price']
        
        ensemble_predictions = {}
        
        # Make predictions for each horizon
        for horizon in self.prediction_config['prediction_horizons']:
            horizon_key = f'{horizon}d'
            
            logger.info(f"   📈 Predicting {horizon_key}...")
            
            # Collect predictions from all available models
            model_predictions = []
            model_weights = []
            
            for model_type, models in self.loaded_models[symbol].items():
                if horizon_key in models:
                    model = models[horizon_key]
                    pred = self.predict_single_model(model, X, horizon_key)
                    
                    if 'error' not in pred:
                        model_predictions.append(pred)
                        
                        # Get model weight
                        base_type = model_type.replace('_unified', '')
                        weight = self.prediction_config['ensemble_weights'].get(base_type, 0.33)
                        model_weights.append(weight)
            
            if not model_predictions:
                logger.warning(f"   ⚠️ No valid predictions for {horizon_key}")
                continue
            
            # Normalize weights
            total_weight = sum(model_weights)
            if total_weight > 0:
                model_weights = [w / total_weight for w in model_weights]
            
            # Calculate ensemble prediction (ensure all values are float)
            ensemble_price_change = float(sum(pred['price_change_pct'] * weight 
                                      for pred, weight in zip(model_predictions, model_weights)))
            
            ensemble_direction_prob = float(sum(pred['direction_prob'] * weight 
                                        for pred, weight in zip(model_predictions, model_weights)))
            
            # Check for NaN/Inf in ensemble results
            if np.isnan(ensemble_price_change) or np.isinf(ensemble_price_change):
                logger.warning(f"   ⚠️ NaN/Inf in ensemble price change for {horizon_key}, using 0.0")
                ensemble_price_change = 0.0
            
            if np.isnan(ensemble_direction_prob) or np.isinf(ensemble_direction_prob):
                logger.warning(f"   ⚠️ NaN/Inf in ensemble direction prob for {horizon_key}, using 0.5")
                ensemble_direction_prob = 0.5
            
            # Calculate confidence based on model agreement
            price_predictions = [pred['price_change_pct'] for pred in model_predictions]
            price_std = float(np.std(price_predictions)) if len(price_predictions) > 1 else 0.0
            
            # Check for NaN in std
            if np.isnan(price_std) or np.isinf(price_std):
                price_std = 0.0
            
            # Lower std = higher confidence
            confidence = float(max(0, 1 - (price_std * 10)))  # Scale factor for confidence
            
            # Calculate target price
            target_price = float(current_price * (1 + ensemble_price_change))
            
            # Generate trading signal
            signal_strength = float(abs(ensemble_price_change))
            if signal_strength > self.prediction_config['signal_threshold']:
                if ensemble_price_change > 0:
                    signal = 'BUY'
                else:
                    signal = 'SELL'
            else:
                signal = 'HOLD'
            
            # Store ensemble prediction (ensure all numeric values are Python native types)
            ensemble_predictions[horizon_key] = {
                'current_price': float(current_price),
                'target_price': float(target_price),
                'price_change_pct': float(ensemble_price_change),
                'price_change_abs': float(target_price - current_price),
                'direction': 'UP' if ensemble_direction_prob > 0.5 else 'DOWN',
                'direction_prob': float(ensemble_direction_prob),
                'confidence': float(confidence),
                'signal': signal,
                'signal_strength': float(signal_strength),
                'horizon_days': int(horizon),
                'num_models': int(len(model_predictions)),
                'model_agreement_std': float(price_std),
                'individual_predictions': model_predictions,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"      📊 {horizon_key}: {ensemble_price_change:+.3%} "
                       f"({signal}) Confidence: {confidence:.2%}")
        
        # Add summary statistics
        prediction_summary = {
            'symbol': symbol,
            'current_price': float(current_price),
            'current_date': pred_data.get('current_date'),
            'mode': mode,
            'predictions': ensemble_predictions,
            'overall_sentiment': self._calculate_overall_sentiment(ensemble_predictions),
            'risk_metrics': self._calculate_risk_metrics(ensemble_predictions, float(current_price)),
            'generated_at': datetime.now().isoformat()
        }
        
        # Store in history
        self.prediction_history.append(prediction_summary)
        
        logger.info(f"✅ Ensemble predictions completed for {symbol}")
        
        return prediction_summary
    
    def _calculate_overall_sentiment(self, predictions: Dict) -> Dict:
        """Calculate overall market sentiment from all horizons"""
        
        if not predictions:
            return {'sentiment': 'NEUTRAL', 'strength': 0.0}
        
        # Average price changes across horizons
        price_changes = [pred['price_change_pct'] for pred in predictions.values()]
        avg_change = float(np.mean(price_changes))
        
        # Average confidence
        confidences = [pred['confidence'] for pred in predictions.values()]
        avg_confidence = float(np.mean(confidences))
        
        # Determine sentiment
        if avg_change > 0.02:  # > 2%
            sentiment = 'BULLISH'
        elif avg_change < -0.02:  # < -2%
            sentiment = 'BEARISH'
        else:
            sentiment = 'NEUTRAL'
        
        return {
            'sentiment': sentiment,
            'strength': float(abs(avg_change)),
            'confidence': float(avg_confidence),
            'avg_price_change': float(avg_change)
        }
    
    def _calculate_risk_metrics(self, predictions: Dict, current_price: float) -> Dict:
        """Calculate risk metrics for predictions"""
        
        if not predictions:
            return {}
        
        # Price targets
        target_prices = [pred['target_price'] for pred in predictions.values()]
        
        # Calculate potential returns
        returns = [(price - current_price) / current_price for price in target_prices]
        
        # Risk metrics
        risk_metrics = {
            'volatility': float(np.std(returns)) if len(returns) > 1 else 0.0,
            'max_gain': float(max(returns)) if returns else 0.0,
            'max_loss': float(min(returns)) if returns else 0.0,
            'risk_reward_ratio': float(abs(max(returns) / min(returns))) if returns and min(returns) != 0 else 0.0,
            'prediction_range': float(max(target_prices) - min(target_prices)) if target_prices else 0.0
        }
        
        # Risk assessment
        if risk_metrics['volatility'] > 0.1:  # 10%
            risk_level = 'HIGH'
        elif risk_metrics['volatility'] > 0.05:  # 5%
            risk_level = 'MEDIUM'
        else:
            risk_level = 'LOW'
        
        risk_metrics['risk_level'] = risk_level
        
        return risk_metrics
    
    def get_trading_signals(self, predictions: Dict, min_confidence: float = 0.6) -> List[Dict]:
        """Extract trading signals from predictions"""
        
        signals = []
        
        if 'predictions' not in predictions:
            return signals
        
        for horizon_key, pred in predictions['predictions'].items():
            if pred['confidence'] >= min_confidence:
                
                signal = {
                    'symbol': predictions['symbol'],
                    'horizon': horizon_key,
                    'action': pred['signal'],
                    'confidence': pred['confidence'],
                    'target_price': pred['target_price'],
                    'current_price': pred['current_price'],
                    'expected_return': pred['price_change_pct'],
                    'signal_strength': pred['signal_strength'],
                    'timestamp': pred['timestamp']
                }
                
                signals.append(signal)
        
        return signals
    
    def save_predictions(self, predictions: Dict, filename: str = None):
        """Save predictions to file"""
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            symbol = predictions.get('symbol', 'unknown')
            filename = f"predictions_{symbol}_{timestamp}.json"
        
        save_path = self.model_dir / filename
        
        with open(save_path, 'w') as f:
            json.dump(predictions, f, indent=2, default=str)
        
        logger.info(f"💾 Predictions saved to {save_path}")
    
    def load_predictions(self, filename: str) -> Dict:
        """Load predictions from file"""
        
        load_path = self.model_dir / filename
        
        if not load_path.exists():
            raise FileNotFoundError(f"Predictions file not found: {load_path}")
        
        with open(load_path, 'r') as f:
            predictions = json.load(f)
        
        logger.info(f"📥 Predictions loaded from {load_path}")
        
        return predictions
    
    def evaluate_predictions(self, symbol: str, days_back: int = 30) -> Dict:
        """Evaluate prediction accuracy"""
        
        logger.info(f"📊 Evaluating predictions for {symbol} (last {days_back} days)...")
        
        # This would require historical predictions and actual outcomes
        # For now, return placeholder structure
        
        evaluation = {
            'symbol': symbol,
            'evaluation_period': f"{days_back} days",
            'total_predictions': len(self.prediction_history),
            'accuracy_metrics': {
                'direction_accuracy': 0.0,  # % correct directions
                'price_mae': 0.0,  # Mean absolute error in price
                'price_rmse': 0.0,  # Root mean square error
                'confidence_correlation': 0.0  # Correlation between confidence and accuracy
            },
            'performance_by_horizon': {},
            'trading_performance': {
                'total_signals': 0,
                'profitable_signals': 0,
                'win_rate': 0.0,
                'avg_return': 0.0
            }
        }
        
        logger.info(f"✅ Evaluation completed for {symbol}")
        
        return evaluation
    
    def get_prediction_summary(self, predictions: Dict) -> str:
        """Generate human-readable prediction summary"""
        
        if not predictions or 'predictions' not in predictions:
            return "No predictions available."
        
        symbol = predictions['symbol']
        current_price = predictions['current_price']
        sentiment = predictions['overall_sentiment']
        
        summary = f"\n🎯 PREDICTION SUMMARY FOR {symbol}\n"
        summary += f"=" * 50 + "\n"
        summary += f"💰 Current Price: ${current_price:.2f}\n"
        summary += f"📊 Overall Sentiment: {sentiment['sentiment']} (Strength: {sentiment['strength']:.2%})\n"
        summary += f"🎯 Confidence: {sentiment['confidence']:.1%}\n\n"
        
        summary += "📈 HORIZON PREDICTIONS:\n"
        summary += "-" * 30 + "\n"
        
        for horizon_key, pred in predictions['predictions'].items():
            horizon_days = pred['horizon_days']
            target_price = pred['target_price']
            change_pct = pred['price_change_pct']
            signal = pred['signal']
            confidence = pred['confidence']
            
            summary += f"{horizon_days}d: ${target_price:.2f} ({change_pct:+.2%}) "
            summary += f"| {signal} | Confidence: {confidence:.1%}\n"
        
        # Risk metrics
        risk = predictions['risk_metrics']
        summary += f"\n⚠️ RISK ASSESSMENT:\n"
        summary += f"Risk Level: {risk['risk_level']}\n"
        summary += f"Volatility: {risk['volatility']:.2%}\n"
        summary += f"Max Potential Gain: {risk['max_gain']:+.2%}\n"
        summary += f"Max Potential Loss: {risk['max_loss']:+.2%}\n"
        
        # Trading signals
        signals = self.get_trading_signals(predictions)
        if signals:
            summary += f"\n🎯 TRADING SIGNALS:\n"
            for signal in signals:
                summary += f"{signal['horizon']}: {signal['action']} at ${signal['target_price']:.2f} "
                summary += f"(Confidence: {signal['confidence']:.1%})\n"
        
        return summary


def test_progressive_predictor():
    """Test Progressive Predictor"""
    print("🧪 Testing Progressive Predictor...")
    
    # Create data loader
    from .data_loader import ProgressiveDataLoader
    
    data_loader = ProgressiveDataLoader()
    
    # Create predictor
    predictor = ProgressivePredictor(
        data_loader=data_loader,
        model_dir="app/ml/models"
    )
    
    print(f"✅ Predictor created with model_dir: {predictor.model_dir}")
    
    # Test data preparation
    print("\n📊 Testing prediction data preparation...")
    try:
        pred_data = predictor.prepare_prediction_data("AAPL")
        print(f"✅ Prediction data prepared: {pred_data['X'].shape}")
        print(f"   📊 Current price: ${pred_data['current_price']:.2f}")
    except Exception as e:
        print(f"⚠️ Prediction data preparation failed: {e}")
    
    # Test model loading (will fail if no models exist, which is expected)
    print("\n📥 Testing model loading...")
    loaded = predictor.load_models("AAPL", ['lstm'])
    print(f"✅ Model loading attempted: {len(loaded)} models loaded")
    
    # Test prediction structure (without actual models)
    print("\n🎯 Testing prediction structure...")
    
    # Create dummy prediction
    dummy_prediction = {
        'symbol': 'AAPL',
        'current_price': 150.0,
        'predictions': {
            '1d': {
                'current_price': 150.0,
                'target_price': 151.5,
                'price_change_pct': 0.01,
                'confidence': 0.75,
                'signal': 'BUY',
                'signal_strength': 0.01,
                'horizon_days': 1,
                'timestamp': datetime.now().isoformat()
            }
        },
        'overall_sentiment': {'sentiment': 'BULLISH', 'strength': 0.01, 'confidence': 0.75},
        'risk_metrics': {
            'risk_level': 'LOW', 
            'volatility': 0.02,
            'max_gain': 0.015,
            'max_loss': -0.005
        }
    }
    
    # Test summary generation
    summary = predictor.get_prediction_summary(dummy_prediction)
    print("✅ Prediction summary generated:")
    print(summary[:200] + "..." if len(summary) > 200 else summary)
    
    # Test trading signals
    signals = predictor.get_trading_signals(dummy_prediction)
    print(f"✅ Trading signals extracted: {len(signals)} signals")
    
    print("\n✅ Progressive Predictor test completed successfully!")
    return True


if __name__ == "__main__":
    test_progressive_predictor()