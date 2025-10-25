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

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path
import json
import pickle
from datetime import datetime, timedelta
from fastapi import HTTPException
import warnings
warnings.filterwarnings('ignore')

# Configure logger early
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# PyTorch-only path (TensorFlow/Keras removed)
logger.info("🧠 Using PyTorch models only (TensorFlow/Keras support removed)")

from .data_loader import ProgressiveDataLoader
try:
    from .models import ProgressiveModels, EnsembleModel, LSTMModel, TransformerModel, CNNModel
    PYTORCH_MODELS_AVAILABLE = True
except ImportError:
    PYTORCH_MODELS_AVAILABLE = False
    logger.warning("⚠️ PyTorch models not available - limited functionality")
from .trainer import ProgressiveTrainer

 # logger already configured above


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
                'cnn': 0.1
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
        
        # Optional confidence calibration (a*conf + b), temperature reserved for future
        self._conf_cal = {'a': 1.0, 'b': 0.0, 'temperature': 1.0}

        # Try to load calibration from champion meta if available
        try:
            meta_candidates = []
            # model_dir/champion_meta.json
            meta_candidates.append(self.model_dir / 'champion_meta.json')
            # parent dir
            meta_candidates.append(self.model_dir.parent / 'champion_meta.json')
            for meta_path in meta_candidates:
                if meta_path.exists():
                    with open(meta_path, 'r', encoding='utf-8') as f:
                        meta = json.load(f)
                        cal = meta.get('confidence_calibration') or meta.get('calibration')
                        if isinstance(cal, dict):
                            a = float(cal.get('a', 1.0))
                            b = float(cal.get('b', 0.0))
                            t = float(cal.get('temperature', 1.0))
                            self._conf_cal = {'a': a, 'b': b, 'temperature': t}
                            logger.info(f"   🎛️ Loaded confidence calibration from {meta_path.name}: a={a}, b={b}, T={t}")
                            break
        except Exception as _e:
            logger.debug(f"Calibration meta load skipped: {_e}")

        logger.info(f"🔮 Progressive Predictor initialized")
        logger.info(f"   📂 Model directory: {self.model_dir}")
        logger.info(f"   ⏰ Horizons: {self.prediction_config['prediction_horizons']}")
    
    def load_models(self, symbol: str = "AAPL", model_types: List[str] = ['lstm']) -> Dict:
        """Load trained models for prediction"""
        
        logger.info(f"📥 Loading models for {symbol}...")
        
        loaded_models = {}
        
        for model_type in model_types:
            model_key = f"{model_type}_{symbol}"
            # Only support PyTorch checkpoint files: {model_type}_{symbol}_{horizon}_best.pth
            try:
                logger.info(f"   🔍 Looking for PyTorch checkpoints for {model_type}_{symbol}...")
                model_dict = {}

                for horizon in self.prediction_config['prediction_horizons']:
                    horizon_key = f'{horizon}d'
                    pytorch_file = self.model_dir / f"{model_type}_{symbol}_{horizon_key}_best.pth"
                    if pytorch_file.exists():
                        try:
                            model = self._load_pytorch_model(pytorch_file, model_type)
                            model_dict[horizon_key] = model
                            logger.info(f"   ✅ Loaded {model_type} {horizon_key} (PyTorch)")
                        except Exception as load_err:
                            logger.warning(f"   ⚠️ Failed to load PyTorch {model_type} {horizon_key}: {load_err}")

                if model_dict:
                    loaded_models[model_type] = model_dict
                else:
                    logger.warning(f"⚠️ No PyTorch checkpoints found for {model_type}_{symbol}")
            except Exception as e:
                logger.error(f"❌ Error loading {model_type} checkpoints: {e}")
        
        self.loaded_models[symbol] = loaded_models
        
        logger.info(f"✅ Loaded {len(loaded_models)} model types for {symbol}")
        
        return loaded_models
    
    def _load_pytorch_model(self, model_path: Path, model_type: str):
        """Load PyTorch model from checkpoint"""
        if not PYTORCH_MODELS_AVAILABLE:
            raise ImportError(f"PyTorch models not available for {model_type}")
        
        try:
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # Get model class and parameters
            model_class_name = checkpoint.get('model_class', '')
            model_params = checkpoint.get('model_params', {})
            horizons = checkpoint.get('horizons', [1, 7, 30])
            mode = checkpoint.get('mode', 'progressive')
            # Prefer top-level persisted IO shapes; fallback to params; then conservative defaults
            saved_input_size = checkpoint.get('input_size', None)
            saved_sequence_length = checkpoint.get('sequence_length', None)
            input_size = saved_input_size if isinstance(saved_input_size, int) and saved_input_size > 0 else model_params.get('input_size', 35)
            sequence_length = saved_sequence_length if isinstance(saved_sequence_length, int) and saved_sequence_length > 0 else model_params.get('sequence_length', 60)
            
            # Create model based on type
            if model_type == 'lstm' and model_class_name == 'LSTMModel':
                # Use saved parameters, they're more accurate than defaults
                model = LSTMModel(
                    input_size=input_size,
                    sequence_length=sequence_length,
                    horizons=horizons,
                    mode=mode,
                    model_params=model_params
                )
            elif model_type == 'transformer' and model_class_name == 'TransformerModel':
                model = TransformerModel(
                    input_size=input_size,
                    sequence_length=sequence_length,
                    horizons=horizons,
                    mode=mode,
                    model_params=model_params
                )
            elif model_type == 'cnn' and model_class_name == 'CNNModel':
                model = CNNModel(
                    input_size=input_size,
                    sequence_length=sequence_length,
                    horizons=horizons,
                    mode=mode,
                    model_params=model_params
                )
            else:
                raise ValueError(f"Unsupported PyTorch model type: {model_type} with class {model_class_name}")
            
            # Load model state
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to load PyTorch model {model_path}: {e}")
            raise
    
    def prepare_prediction_data(self, symbol: str, mode: str = "progressive") -> Dict:
        """Prepare data for prediction (last sequence)"""
        
        logger.info(f"📊 Preparing prediction data for {symbol} ({mode} mode)...")
        
        # Get features - use prediction mode to keep the latest data
        df = self.data_loader.prepare_features(symbol, for_prediction=True)
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
                           model, 
                           X: np.ndarray, 
                           horizon: str = "1d") -> Dict:
        """Make prediction with single PyTorch model"""
        
        try:
            # Check if it's a PyTorch model
            if nn is not None and isinstance(model, nn.Module):
                # PyTorch model
                model.eval()
                with torch.no_grad():
                    # Convert to tensor and move to same device as model
                    X_tensor = torch.FloatTensor(X)
                    
                    # Get model device
                    model_device = next(model.parameters()).device
                    X_tensor = X_tensor.to(model_device)
                    
                    # Get prediction - need to extract horizon number from horizon string
                    horizon_num = int(horizon.replace('d', ''))
                    prediction = model(X_tensor, horizon=horizon_num)
                    
                    # Handle different output formats
                    if isinstance(prediction, tuple) and len(prediction) == 2:
                        # Dual output (regression, classification)
                        price_pred = float(prediction[0][0][0].item())
                        direction_pred = float(prediction[1][0][0].item())
                    elif isinstance(prediction, torch.Tensor):
                        if prediction.dim() == 3 and prediction.size(-1) >= 2:
                            # Multi-output format
                            price_pred = float(prediction[0][0][0].item())
                            direction_pred = float(prediction[0][0][1].item() if prediction.size(-1) > 1 else 0.5)
                        else:
                            # Single output
                            price_pred = float(prediction[0][0].item())
                            direction_pred = 0.5
                    else:
                        price_pred = float(prediction)
                        direction_pred = 0.5
            else:
                raise TypeError("Only PyTorch models are supported")
            
            # Check for NaN values and replace with defaults
            if np.isnan(price_pred) or np.isinf(price_pred):
                logger.warning(f"   ⚠️ NaN/Inf detected in price prediction for {horizon}, using 0.0")
                price_pred = 0.0
            # Extra safety: clamp predicted percentage return to [-1, 1]
            try:
                price_pred = float(max(-1.0, min(1.0, price_pred)))
            except Exception:
                pass
            
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
            error_msg = str(e)
            logger.error(f"❌ Error in single model prediction: {error_msg}")
            
            # Check if it's a feature mismatch error
            if "incompatible" in error_msg.lower() and "shape" in error_msg.lower():
                # Extract expected vs found shapes
                if "expected shape=" in error_msg and "found shape=" in error_msg:
                    raise Exception(f"Feature mismatch for {horizon}: The model was trained with different features. Please retrain the model with current data.")
                else:
                    raise Exception(f"Model input shape mismatch for {horizon}: {error_msg}")
            else:
                # Re-raise the original error
                raise Exception(f"Model prediction failed for {horizon}: {error_msg}")
    
    def _read_best_val_loss(self, symbol: str, model_type: str, horizon_key: str) -> Optional[float]:
        """Read best validation loss from saved training history CSV for weighting.
        Returns None if not available or invalid.
        """
        try:
            history_path = self.model_dir / f"{model_type}_{symbol}_{horizon_key}_history.csv"
            if not history_path.exists():
                return None
            df_hist = pd.read_csv(history_path)
            if 'val_loss' not in df_hist.columns or df_hist['val_loss'].empty:
                return None
            vals = pd.to_numeric(df_hist['val_loss'], errors='coerce').dropna()
            if vals.empty:
                return None
            best_val = float(vals.min())
            if np.isnan(best_val) or np.isinf(best_val):
                return None
            return best_val
        except Exception as e:
            logger.warning(f"   ⚠️ Failed reading {model_type} {horizon_key} history for weights: {e}")
            return None

    def _compute_model_weight(self, symbol: str, model_type: str, horizon_key: str) -> float:
        """Compute performance-based ensemble weight via inverse best val_loss.
        Falls back to configured base weight.
        """
        base_type = model_type.replace('_unified', '')
        base_weight = float(self.prediction_config['ensemble_weights'].get(base_type, 0.33))
        best_val = self._read_best_val_loss(symbol, model_type, horizon_key)
        if best_val is None:
            return base_weight
        eps = 1e-8
        try:
            inv = 1.0 / max(best_val, eps)
            # Bound pre-normalization weight to avoid dominance; 0.25x..2x of base
            weight = base_weight * inv
            low, high = base_weight * 0.25, base_weight * 2.0
            return float(min(max(weight, low), high))
        except Exception:
            return base_weight
    
    def predict_ensemble(self, symbol: str, mode: str = "progressive") -> Dict:
        """Make ensemble predictions for all horizons"""
        
        logger.info(f"🎯 Making ensemble predictions for {symbol}...")
        
        # Prepare prediction data first
        pred_data = self.prepare_prediction_data(symbol, mode)
        X = pred_data['X']
        current_price = pred_data['current_price']
        
        # Check if models are loaded
        if symbol not in self.loaded_models or not self.loaded_models[symbol]:
            logger.info(f"📥 Loading models for {symbol}...")
            loaded = self.load_models(symbol, ['cnn', 'transformer', 'lstm'])
            logger.info(f"📥 Load models returned: {len(loaded)} model types")
        
        # Count total available models
        total_models = 0
        if symbol in self.loaded_models and self.loaded_models[symbol]:
            for model_type, models in self.loaded_models[symbol].items():
                if models:
                    total_models += len(models)
        
        # If still no models loaded, try alternative model types
        if total_models == 0:
            logger.warning(f"⚠️ No models loaded for {symbol}. Trying alternative model types...")
            # Try loading any available models for this symbol
            import os
            model_files = [f for f in os.listdir(self.model_dir) if f.startswith(f"cnn_{symbol}_") or f.startswith(f"transformer_{symbol}_") or f.startswith(f"lstm_{symbol}_")]
            logger.info(f"🔍 Found {len(model_files)} model files for {symbol}")
            
            if model_files:
                # Try loading with specific available model types
                available_types = set()
                for f in model_files:
                    if f.startswith(f"cnn_{symbol}_"):
                        available_types.add('cnn')
                    elif f.startswith(f"transformer_{symbol}_"):
                        available_types.add('transformer')
                    elif f.startswith(f"lstm_{symbol}_"):
                        available_types.add('lstm')
                
                logger.info(f"🎯 Available model types for {symbol}: {list(available_types)}")
                loaded = self.load_models(symbol, list(available_types))
                
                # Recount models
                total_models = 0
                if symbol in self.loaded_models and self.loaded_models[symbol]:
                    for model_type, models in self.loaded_models[symbol].items():
                        if models:
                            total_models += len(models)
        
        # Only fall back to error if absolutely no models are available
        if total_models == 0:
            logger.error(f"❌ No models could be loaded for {symbol}. Available model files: {len([f for f in os.listdir(self.model_dir) if symbol in f])}")
            available_symbols = set()
            for f in os.listdir(self.model_dir):
                if '_' in f and f.endswith(('.h5', '.pth')):
                    parts = f.split('_')
                    if len(parts) >= 2:
                        available_symbols.add(parts[1])
            raise HTTPException(status_code=404, detail=f"No trained models available for {symbol}. Available symbols: {sorted(list(available_symbols))[:10]}")
        
        logger.info(f"✅ Using {total_models} trained models for {symbol} predictions")
        
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
                        # Performance-based weight (fallback to base)
                        weight = self._compute_model_weight(symbol, model_type, horizon_key)
                        model_weights.append(float(weight))
            
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
            # Clamp ensemble to a reasonable band per horizon
            # 1d: 10%, 7d: 20%, 30d: 40% (hard safety caps)
            horizon_caps = {
                '1d': 0.10,
                '7d': 0.20,
                '30d': 0.40
            }
            max_abs_return = horizon_caps.get(horizon_key, 0.5)
            if np.isnan(ensemble_price_change) or np.isinf(ensemble_price_change):
                ensemble_price_change = 0.0
            else:
                ensemble_price_change = float(np.clip(ensemble_price_change, -max_abs_return, max_abs_return))
            
            ensemble_direction_prob = float(sum(pred['direction_prob'] * weight 
                                        for pred, weight in zip(model_predictions, model_weights)))
            
            # Check for NaN/Inf in ensemble results
            if np.isnan(ensemble_price_change) or np.isinf(ensemble_price_change):
                logger.warning(f"   ⚠️ NaN/Inf in ensemble price change for {horizon_key}, using 0.0")
                ensemble_price_change = 0.0
            
            if np.isnan(ensemble_direction_prob) or np.isinf(ensemble_direction_prob):
                logger.warning(f"   ⚠️ NaN/Inf in ensemble direction prob for {horizon_key}, using 0.5")
                ensemble_direction_prob = 0.5
            
            # Calculate confidence based on model agreement and classifier certainty
            price_predictions = [pred['price_change_pct'] for pred in model_predictions]
            price_std = float(np.std(price_predictions)) if len(price_predictions) > 1 else 0.0
            
            # Check for NaN in std
            if np.isnan(price_std) or np.isinf(price_std):
                price_std = 0.0
            
            # Lower std => higher agreement confidence (clamped 0..1)
            agreement_conf = float(max(0.0, min(1.0, 1.0 - (price_std * 10.0))))
            # Probabilistic confidence from ensemble direction probability (distance from 0.5)
            prob_conf = float(max(0.0, min(1.0, abs(ensemble_direction_prob - 0.5) * 2.0)))
            # Combine (equal weights)
            confidence_raw = float(max(0.0, min(1.0, 0.5 * agreement_conf + 0.5 * prob_conf)))
            # Apply optional linear calibration (identity if no meta)
            try:
                a = float(self._conf_cal.get('a', 1.0))
                b = float(self._conf_cal.get('b', 0.0))
                confidence = float(min(1.0, max(0.0, a * confidence_raw + b)))
            except Exception:
                confidence = confidence_raw
            
            # Calculate target price
            target_price = float(current_price * (1 + ensemble_price_change))
            
            # Determine consistent direction
            reg_direction = 'UP' if ensemble_price_change > 0 else ('DOWN' if ensemble_price_change < 0 else 'NEUTRAL')
            thr = float(self.prediction_config.get('confidence_threshold', 0.6))
            final_direction = reg_direction
            if ensemble_direction_prob >= thr:
                final_direction = 'UP'
            elif ensemble_direction_prob <= (1.0 - thr):
                final_direction = 'DOWN'

            # Generate trading signal from regression magnitude
            signal_strength = float(abs(ensemble_price_change))
            if signal_strength > self.prediction_config['signal_threshold']:
                if ensemble_price_change > 0:
                    signal = 'BUY'
                else:
                    signal = 'SELL'
            else:
                signal = 'HOLD'
            # Prevent contradictions between signal and direction
            if (signal == 'BUY' and final_direction == 'DOWN') or (signal == 'SELL' and final_direction == 'UP'):
                signal = 'HOLD'
            
            # Store ensemble prediction (ensure all numeric values are Python native types)
            ensemble_predictions[horizon_key] = {
                'current_price': float(current_price),
                'target_price': float(target_price),
                'price_change_pct': float(ensemble_price_change),
                'price_change_abs': float(target_price - current_price),
                'direction': final_direction if final_direction != 'NEUTRAL' else ('UP' if ensemble_direction_prob > 0.5 else 'DOWN'),
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
        
        # If no predictions were made (all horizons failed), raise error
        if not ensemble_predictions:
            logger.error(f"❌ No valid predictions made for {symbol} from {total_models} models")
            raise HTTPException(status_code=500, detail=f"Failed to generate predictions for {symbol}")
        
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
        # Live-only policy: no placeholder evaluation. Implement or remove.
        raise NotImplementedError("Prediction evaluation requires historical datasets; placeholders are not allowed under live-only policy.")
    
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
    
    # Skip creating any dummy predictions under live-only policy
    print("\n⏭️ Skipping dummy prediction structure test (live-only policy)")
    
    print("\n✅ Progressive Predictor test completed (no mock data used)")
    return True


if __name__ == "__main__":
    test_progressive_predictor()