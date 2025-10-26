"""
Progressive Trainer for Multi-Horizon Stock Predictions
=======================================================
Advanced training orchestration for Progressive ML Training System

Features:
- Progressive training workflow (1→7→30 days)
- Unified training for all horizons simultaneously
- Model checkpointing and resumption with PyTorch
- Advanced callbacks and monitoring
- Training history tracking and visualization
- Early stopping and learning rate scheduling
- K-fold cross-validation support
- CUDA GPU acceleration
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from pathlib import Path
import json
import pickle
# Matplotlib is optional; avoid hard import errors in headless/minimal environments
try:
    import matplotlib
    # Use a non-interactive backend to be safe on servers/CI
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore
    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("📉 Matplotlib not available; plotting disabled.")
from datetime import datetime
import time
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, classification_report

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"🚀 Trainer using device: {device}")

# Hard-disable Torch Dynamo/compile in case the environment enables it implicitly
try:
    import torch._dynamo as _dynamo  # type: ignore
    try:
        _dynamo.reset()
    except Exception:
        pass
    try:
        _dynamo.disable()
        logger.info("🛡️ Torch Dynamo disabled in trainer")
    except Exception:
        logger.debug("Torch Dynamo disable not available")
except Exception:
    # Safe to ignore if not available
    pass

# Optional imports
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    logger.warning("⚠️ Seaborn not available. Some visualizations may be limited.")

from .data_loader import ProgressiveDataLoader
from .models import LSTMModel, TransformerModel, CNNModel, EnsembleModel, ProgressiveModels


class ProgressiveTrainer:
    """
    Progressive Training System for Multi-Horizon Stock Predictions
    
    Supports both Progressive training (sequential) and Unified training (simultaneous)
    """
    
    def __init__(self, 
                 data_loader: ProgressiveDataLoader,
                 model_config: Dict = None,
                 training_config: Dict = None,
                 save_dir: str = None):
        """
        Initialize Progressive Trainer
        
        Args:
            data_loader: ProgressiveDataLoader instance
            model_config: Model configuration parameters
            training_config: Training parameters and callbacks
            save_dir: Directory to save models and checkpoints (absolute or relative)
                     If None, uses absolute path to app/ml/models
        """
        
        self.data_loader = data_loader
        
        # Handle save_dir - use absolute path by default to avoid CWD issues
        if save_dir is None:
            # Calculate absolute path from this file's location
            # __file__ = d:/Projects/NewsFetcher/app/ml/progressive/trainer.py
            # .parent = progressive, .parent.parent = ml, .parent.parent.parent = app
            # .parent.parent.parent.parent = NewsFetcher root
            base_dir = Path(__file__).parent.parent.parent.parent  # Goes to NewsFetcher root
            save_dir = str(base_dir / "app" / "ml" / "models")
            logger.info(f"   Using default save_dir (absolute): {save_dir}")
        
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = device
        
        # Default model configuration
        self.model_config = {
            'lstm_params': {
                'lstm_units': [128, 64, 32],
                'dropout_rate': 0.3,
                'learning_rate': 0.001,
                'num_layers': 3
            },
            'transformer_params': {
                'num_heads': 8,
                'ff_dim': 256,
                'num_transformer_blocks': 2,
                'embed_dim': 128,
                'dropout_rate': 0.1,
                'learning_rate': 0.0001
            },
            'cnn_params': {
                'filters': [64, 128, 64],
                'kernel_sizes': [3, 3, 3],
                'pool_sizes': [2, 2, 2],
                'dropout_rate': 0.35,
                'learning_rate': 0.0003
            }
        }
        
        if model_config:
            self.model_config.update(model_config)
        
        # Default training configuration
        self.training_config = {
            'epochs': 100,
            'batch_size': 32,
            'validation_split': 0.2,
            'early_stopping_patience': 15,
            'reduce_lr_patience': 10,
            'reduce_lr_factor': 0.5,
            'min_lr': 1e-7,
            'save_best_only': True,
            'monitor': 'val_loss',
            'mode': 'min',
            'verbose': 1,
            'loss_weights': {
                'regression': 1.0,
                'classification': 0.5
            }
        }
        
        if training_config:
            self.training_config.update(training_config)
        
        # Initialize models
        self.models = {}
        self.training_history = {}
        self.ensemble_model = None
        
        logger.info(f"🏃‍♂️ Progressive Trainer initialized")
        logger.info(f"   💾 Save directory: {self.save_dir}")
        logger.info(f"   📊 Data loader: {self.data_loader.horizons} horizons")
    
    def prepare_progressive_data(self, symbol: str = "AAPL") -> Dict:
        """Prepare data for progressive training"""
        
        logger.info(f"📊 Preparing progressive data for {symbol}...")
        
        # Prepare features
        df = self.data_loader.prepare_features(symbol)
        if df is None:
            raise ValueError(f"Could not prepare features for {symbol}")
        
        # Create sequences in progressive mode
        data = self.data_loader.create_sequences(df, mode='progressive')
        
        # Extract training data for each horizon
        progressive_data = {}
        
        for horizon in self.data_loader.horizons:
            horizon_key = f'{horizon}d'
            
            if horizon_key in data:
                X = data[horizon_key]['X'] 
                y_reg = data[horizon_key]['y_regression']
                y_clf = data[horizon_key]['y_classification']
                
                # Split train/validation
                val_split = self.training_config['validation_split']
                split_idx = int(len(X) * (1 - val_split))
                
                progressive_data[horizon_key] = {
                    'X_train': X[:split_idx],
                    'X_val': X[split_idx:],
                    'y_reg_train': y_reg[:split_idx],
                    'y_reg_val': y_reg[split_idx:],
                    'y_clf_train': y_clf[:split_idx],
                    'y_clf_val': y_clf[split_idx:]
                }
                
                logger.info(f"   ✅ {horizon_key}: Train={split_idx}, Val={len(X)-split_idx}")
        
        return progressive_data
    
    def prepare_unified_data(self, symbol: str = "AAPL") -> Dict:
        """Prepare data for unified training"""
        
        logger.info(f"📊 Preparing unified data for {symbol}...")
        
        # Prepare features
        df = self.data_loader.prepare_features(symbol)
        if df is None:
            raise ValueError(f"Could not prepare features for {symbol}")
        
        # Create sequences in unified mode
        data = self.data_loader.create_sequences(df, mode='unified')
        
        if 'unified' not in data:
            raise ValueError("No unified data available")
        
        X = data['unified']['X']
        y_reg = data['unified']['y_regression']  # Shape: (samples, num_horizons)
        y_clf = data['unified']['y_classification']  # Shape: (samples, num_horizons)
        
        # Split train/validation
        val_split = self.training_config['validation_split']
        split_idx = int(len(X) * (1 - val_split))
        
        # Prepare targets for each horizon
        y_train_dict = {}
        y_val_dict = {}
        
        for i, horizon in enumerate(self.data_loader.horizons):
            # Regression targets - extract column for this horizon
            y_train_dict[f'price_pred_{horizon}d'] = y_reg[:split_idx, i]
            y_val_dict[f'price_pred_{horizon}d'] = y_reg[split_idx:, i]
            
            # Classification targets - extract column for this horizon
            y_train_dict[f'direction_pred_{horizon}d'] = y_clf[:split_idx, i]
            y_val_dict[f'direction_pred_{horizon}d'] = y_clf[split_idx:, i]
        
        unified_data = {
            'X_train': X[:split_idx],
            'X_val': X[split_idx:],
            'y_train': y_train_dict,
            'y_val': y_val_dict
        }
        
        logger.info(f"   ✅ Unified: Train={split_idx}, Val={len(X)-split_idx}")
        
        return unified_data
    
    def create_model(self, model_type: str, input_size: int, sequence_length: int, 
                     horizons: List[int], mode: str = "progressive") -> nn.Module:
        """Create PyTorch model"""
        # Normalize aliases (e.g., 'cnn_lstm' -> 'cnn')
        normalized = self._normalize_model_type(model_type)
        model_type = normalized
        
        if model_type == "lstm":
            return LSTMModel(
                input_size=input_size,
                sequence_length=sequence_length,
                horizons=horizons,
                mode=mode,
                model_params=self.model_config['lstm_params']
            )
        elif model_type == "transformer":
            return TransformerModel(
                input_size=input_size,
                sequence_length=sequence_length,
                horizons=horizons,
                mode=mode,
                model_params=self.model_config['transformer_params']
            )
        elif model_type == "cnn":
            return CNNModel(
                input_size=input_size,
                sequence_length=sequence_length,
                horizons=horizons,
                mode=mode,
                model_params=self.model_config['cnn_params']
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def _normalize_model_type(self, model_type: str) -> str:
        """Normalize UI aliases and typos to supported model keys"""
        mt = (model_type or '').strip().lower()
        alias_map = {
            'cnn_lstm': 'cnn',
            'lstm_cnn': 'cnn',
            'gru': 'lstm',
            'transformers': 'transformer',
            'convolution': 'cnn',
            'rnn': 'lstm'
        }
        return alias_map.get(mt, mt)
    
    def save_model(self, model: nn.Module, model_name: str, horizon: str = None):
        """Save PyTorch model"""
        checkpoint_path = self.save_dir / f"{model_name}{'_'+horizon if horizon else ''}_best.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_class': model.__class__.__name__,
            'model_params': model.params if hasattr(model, 'params') else {},
            'horizons': model.horizons if hasattr(model, 'horizons') else [],
            'mode': model.mode if hasattr(model, 'mode') else 'progressive',
            # Persist critical IO shapes to ensure reliable reload
            'input_size': getattr(model, 'input_size', None),
            'sequence_length': getattr(model, 'sequence_length', None)
        }, checkpoint_path)
        logger.info(f"💾 Model saved: {checkpoint_path}")
        
    def load_model(self, model_name: str, horizon: str = None) -> nn.Module:
        """Load PyTorch model"""
        checkpoint_path = self.save_dir / f"{model_name}{'_'+horizon if horizon else ''}_best.pth"
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Model not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        model_class_name = checkpoint.get('model_class')
        model_params = checkpoint.get('model_params', {})
        horizons = checkpoint.get('horizons', [1])
        mode = checkpoint.get('mode', 'progressive')
        input_size = checkpoint.get('input_size')
        sequence_length = checkpoint.get('sequence_length')

        # Map class name to actual class
        class_map = {
            'LSTMModel': LSTMModel,
            'TransformerModel': TransformerModel,
            'CNNModel': CNNModel,
        }
        if model_class_name not in class_map:
            raise ValueError(f"Unknown saved model class: {model_class_name}")

        ModelCls = class_map[model_class_name]
        model = ModelCls(
            input_size=input_size,
            sequence_length=sequence_length,
            horizons=horizons,
            mode=mode,
            model_params=model_params
        ).to(self.device)

        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model

    def train_progressive_models(self, symbol: str = "AAPL", model_types: List[str] = ['lstm']) -> Dict:
        """Train separate models per horizon in progressive mode."""
        logger.info(f"🚀 Starting Progressive Training for {symbol}")
        # Normalize all model types first
        model_types = [self._normalize_model_type(mt) for mt in model_types]
        logger.info(f"📋 Model types: {model_types}")

        # Prepare data per horizon
        data = self.prepare_progressive_data(symbol)
        if not data:
            raise ValueError("No progressive data available for training")

        # Infer input sizes from any horizon
        any_key = next(iter(data.keys()))
        X_train_sample = data[any_key]['X_train']
        sequence_length, input_size = X_train_sample.shape[1], X_train_sample.shape[2]

        results: Dict[str, Dict[str, Any]] = {}
        for model_type in model_types:
            logger.info(f"\n🧠 Training {model_type.upper()} models (progressive)...")
            model_results: Dict[str, Any] = {}

            for horizon_key, d in data.items():
                horizon = int(horizon_key.replace('d', ''))

                # Convert to tensors
                X_train = torch.FloatTensor(d['X_train']).to(self.device)
                X_val = torch.FloatTensor(d['X_val']).to(self.device)
                y_train = torch.FloatTensor(d['y_reg_train']).to(self.device)
                y_val = torch.FloatTensor(d['y_reg_val']).to(self.device)
                y_clf_train = torch.FloatTensor(d['y_clf_train']).to(self.device)
                y_clf_val = torch.FloatTensor(d['y_clf_val']).to(self.device)

                # Create model
                model = self.create_model(
                    model_type=model_type,
                    input_size=input_size,
                    sequence_length=sequence_length,
                    horizons=[horizon],
                    mode="progressive"
                )

                # Train model
                history = self._train_single_model(
                    model=model,
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    model_name=f"{model_type}_{symbol}",
                    horizon=horizon_key,
                    y_clf_train=y_clf_train,
                    y_clf_val=y_clf_val
                )

                model_results[horizon_key] = {
                    'history': history,
                    'final_loss': history['val_loss'][-1] if history.get('val_loss') else float('inf')
                }

                logger.info(f"   ✅ {model_type} {horizon_key} completed")

            results[model_type] = model_results

        logger.info(f"✅ Progressive training completed for {symbol}")
        return results
    
    def train_unified_models(self, symbol: str = "AAPL", model_types: List[str] = ['lstm']) -> Dict:
        """
        Train unified models (all horizons simultaneously)
        
        Args:
            symbol: Stock symbol to train on
            model_types: List of model types ['lstm', 'transformer', 'cnn']
        
        Returns:
            Dictionary with training results
        """
        
        logger.info(f"🚀 Starting Unified Training for {symbol}")
        # Normalize all model types first
        model_types = [self._normalize_model_type(mt) for mt in model_types]
        logger.info(f"📋 Model types: {model_types}")
        
        # Prepare data
        data = self.prepare_unified_data(symbol)
        
        # Get input dimensions
        X_train = data['X_train']
        sequence_length, input_size = X_train.shape[1], X_train.shape[2]
        
        results = {}
        
        for model_type in model_types:
            logger.info(f"\n🧠 Training unified {model_type.upper()} model...")
            
            # Convert to tensors
            X_train_tensor = torch.FloatTensor(X_train).to(self.device)
            X_val_tensor = torch.FloatTensor(data['X_val']).to(self.device)
            
            # For unified mode, we need all horizon targets
            y_train_tensor = torch.FloatTensor(np.column_stack([
                data['y_train'][f'price_pred_{h}d'] for h in self.data_loader.horizons
            ])).to(self.device)
            
            y_val_tensor = torch.FloatTensor(np.column_stack([
                data['y_val'][f'price_pred_{h}d'] for h in self.data_loader.horizons
            ])).to(self.device)
            
            # Create unified model
            model = self.create_model(
                model_type=model_type,
                input_size=input_size,
                sequence_length=sequence_length,
                horizons=self.data_loader.horizons,
                mode="unified"
            )
            
            # Train model
            history = self._train_single_model(
                model=model,
                X_train=X_train_tensor,
                y_train=y_train_tensor,
                X_val=X_val_tensor,
                y_val=y_val_tensor,
                model_name=f"{model_type}_{symbol}",
                horizon="unified"
            )
            
            results[model_type] = {
                'unified': {
                    'history': history,
                    'final_loss': history['val_loss'][-1] if history['val_loss'] else float('inf')
                }
            }
            
            logger.info(f"   ✅ Unified {model_type} completed")
        
        logger.info(f"✅ Unified training completed for {symbol}")
        return results
    
    def _train_single_model(self, model: nn.Module, X_train: torch.Tensor, y_train: torch.Tensor,
                           X_val: torch.Tensor, y_val: torch.Tensor, model_name: str, horizon: str,
                           y_clf_train: Optional[torch.Tensor] = None, y_clf_val: Optional[torch.Tensor] = None) -> Dict:
        """Train a single PyTorch model"""
        
        # Setup optimizer and loss function
        optimizer = model.get_optimizer()
        criterion_reg = model.get_loss_fn()
        criterion_clf = nn.BCELoss()
        
        # Training parameters
        epochs = self.training_config['epochs']
        batch_size = self.training_config['batch_size']
        
        # Clamp regression targets to a sane range (percentage returns) to stabilize training
        try:
            y_train = y_train.clamp(min=-1.0, max=1.0)
            y_val = y_val.clamp(min=-1.0, max=1.0)
        except Exception:
            # If shapes are unexpected, proceed without clamp
            pass

        # Create data loaders (include classification targets if provided)
        if y_clf_train is not None and y_clf_val is not None:
            train_dataset = TensorDataset(X_train, y_train, y_clf_train)
            val_dataset = TensorDataset(X_val, y_val, y_clf_val)
        else:
            train_dataset = TensorDataset(X_train, y_train)
            val_dataset = TensorDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Training history (per-epoch aggregates only to ensure equal lengths)
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_reg_loss_epoch': [],
            'train_clf_loss_epoch': [],
            'val_reg_loss_epoch': [],
            'val_clf_loss_epoch': []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        logger.info(f"   📊 Training on {len(X_train)} samples, validating on {len(X_val)} samples")
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            # Accumulate per-batch components to report per-epoch means
            train_reg_sum, train_clf_sum, train_batches = 0.0, 0.0, 0
            
            for batch in train_loader:
                if y_clf_train is not None and y_clf_val is not None:
                    batch_X, batch_y, batch_y_clf = batch
                else:
                    batch_X, batch_y = batch
                optimizer.zero_grad()
                
                if model.mode == "progressive":
                    horizon_num = int(horizon.replace('d', ''))
                    outputs = model(batch_X, horizon=horizon_num)
                else:
                    outputs = model(batch_X)

                # Compute losses
                reg_loss = None
                clf_loss = None
                if isinstance(outputs, tuple) and len(outputs) == 2 and y_clf_train is not None:
                    reg_out, clf_out = outputs
                    reg_loss = criterion_reg(reg_out.squeeze(), batch_y)
                    # use the batch's classification targets, not a slice from the start
                    clf_loss = criterion_clf(clf_out.squeeze(), batch_y_clf)
                    loss = (self.training_config['loss_weights']['regression'] * reg_loss +
                            self.training_config['loss_weights']['classification'] * clf_loss)
                else:
                    reg_loss = criterion_reg(outputs.squeeze(), batch_y)
                    loss = reg_loss
                loss.backward()
                optimizer.step()
                
                train_loss += float(loss.item())
                train_reg_sum += float(reg_loss.item()) if reg_loss is not None else 0.0
                train_clf_sum += float(clf_loss.item()) if clf_loss is not None else 0.0
                train_batches += 1
            
            train_loss /= len(train_loader)
            
            # Aggregate per-epoch train components
            if train_batches > 0:
                history['train_reg_loss_epoch'].append(train_reg_sum / train_batches)
                history['train_clf_loss_epoch'].append(train_clf_sum / train_batches)

            # Validation phase
            model.eval()
            val_loss = 0.0
            val_reg_sum, val_clf_sum, val_batches = 0.0, 0.0, 0
            
            with torch.no_grad():
                for i, batch in enumerate(val_loader):
                    if y_clf_train is not None and y_clf_val is not None:
                        batch_X, batch_y, batch_y_clf = batch
                    else:
                        batch_X, batch_y = batch
                    if model.mode == "progressive":
                        horizon_num = int(horizon.replace('d', ''))
                        outputs = model(batch_X, horizon=horizon_num)
                    else:
                        outputs = model(batch_X)

                    if isinstance(outputs, tuple) and len(outputs) == 2 and y_clf_val is not None:
                        reg_out, clf_out = outputs
                        reg_l = criterion_reg(reg_out.squeeze(), batch_y)
                        # use the batch's classification targets directly
                        clf_l = criterion_clf(clf_out.squeeze(), batch_y_clf)
                        loss = (self.training_config['loss_weights']['regression'] * reg_l +
                                self.training_config['loss_weights']['classification'] * clf_l)
                        val_reg_sum += float(reg_l.item())
                        val_clf_sum += float(clf_l.item())
                    else:
                        reg_l = criterion_reg(outputs.squeeze(), batch_y)
                        loss = reg_l
                        val_reg_sum += float(reg_l.item())
                        val_clf_sum += 0.0

                    val_loss += float(loss.item())
                    val_batches += 1
            
            val_loss /= len(val_loader)
            
            # Update history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            if val_batches > 0:
                history['val_reg_loss_epoch'].append(val_reg_sum / val_batches)
                history['val_clf_loss_epoch'].append(val_clf_sum / val_batches)
            
            # Check for improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                self.save_model(model, model_name, horizon)
            else:
                patience_counter += 1
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                logger.info(f"      Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Early stopping
            if patience_counter >= self.training_config['early_stopping_patience']:
                logger.info(f"      Early stopping at epoch {epoch+1}")
                break
        
        # Save training history (ensure equal-length columns)
        history_path = self.save_dir / f"{model_name}{'_'+horizon if horizon else ''}_history.csv"
        pd.DataFrame(history).to_csv(history_path, index=False)

        logger.info(f"      Final validation loss: {best_val_loss:.4f}")

        return history
    
    def _make_json_serializable(self, obj):
        """Convert numpy arrays and other non-serializable objects to JSON-compatible format"""
        
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif hasattr(obj, 'history') and isinstance(obj.history, dict):  # Keras History object
            return obj.history
        elif str(type(obj)).endswith("callbacks.History'>"):  # Direct History object
            return obj.history
        else:
            # Try to convert to string for any other non-serializable objects
            try:
                json.dumps(obj)
                return obj
            except (TypeError, ValueError):
                return str(type(obj).__name__)
    
    def plot_training_history(self, symbol: str = "AAPL", save_plots: bool = True):
        """Plot training history"""
        if not MATPLOTLIB_AVAILABLE:
            logger.info("Plotting skipped: Matplotlib not available")
            return
        
        logger.info(f"📊 Plotting training history for {symbol}...")
        
        if not self.training_history:
            logger.warning("No training history available")
            return
        
        for training_type, results in self.training_history.items():
            if symbol in training_type:
                
                for model_name, model_results in results.items():
                    
                    if isinstance(model_results, dict) and 'history' in model_results:
                        # Unified model
                        self._plot_single_history(model_results['history'], f"{model_name}_{training_type}")
                    else:
                        # Progressive models
                        for horizon, horizon_results in model_results.items():
                            if 'history' in horizon_results:
                                self._plot_single_history(
                                    horizon_results['history'], 
                                    f"{model_name}_{horizon}_{training_type}"
                                )
        
        if save_plots:
            plt.close('all')
        
        logger.info("✅ Training history plots completed")
    
    def _plot_single_history(self, history: Dict, title: str):
        """Plot single model training history"""
        if not MATPLOTLIB_AVAILABLE:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Training History: {title}', fontsize=16)
        
        # Loss
        axes[0, 0].plot(history['loss'], label='Training Loss')
        if 'val_loss' in history:
            axes[0, 0].plot(history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Find regression metrics
        reg_metrics = [k for k in history.keys() if 'price_pred' in k and 'mae' in k]
        if reg_metrics:
            metric_key = reg_metrics[0]
            axes[0, 1].plot(history[metric_key], label='Training MAE')
            val_key = f'val_{metric_key}'
            if val_key in history:
                axes[0, 1].plot(history[val_key], label='Validation MAE')
            axes[0, 1].set_title('Regression MAE')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('MAE')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        # Find classification metrics
        clf_metrics = [k for k in history.keys() if 'direction_pred' in k and 'accuracy' in k]
        if clf_metrics:
            metric_key = clf_metrics[0]
            axes[1, 0].plot(history[metric_key], label='Training Accuracy')
            val_key = f'val_{metric_key}'
            if val_key in history:
                axes[1, 0].plot(history[val_key], label='Validation Accuracy')
            axes[1, 0].set_title('Classification Accuracy')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Accuracy')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Learning rate (if available)
        if 'lr' in history:
            axes[1, 1].plot(history['lr'])
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_yscale('log')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.save_dir / f"{title}_history.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()


def test_progressive_trainer():
    """Test Progressive Trainer"""
    print("🧪 Testing Progressive Trainer...")
    
    # Create data loader
    from .data_loader import ProgressiveDataLoader
    
    data_loader = ProgressiveDataLoader()
    
    # Create trainer
    trainer = ProgressiveTrainer(
        data_loader=data_loader,
        training_config={
            'epochs': 5,  # Short test
            'batch_size': 16,
            'validation_split': 0.2,
            'verbose': 1
        }
    )
    
    print(f"✅ Trainer created with save_dir: {trainer.save_dir}")
    
    # Test data preparation
    print("\n📊 Testing data preparation...")
    progressive_data = trainer.prepare_progressive_data("AAPL")
    print(f"✅ Progressive data prepared: {list(progressive_data.keys())}")
    
    unified_data = trainer.prepare_unified_data("AAPL")
    print(f"✅ Unified data prepared: X_train shape = {unified_data['X_train'].shape}")
    
    print("\n✅ Progressive Trainer test completed successfully!")
    return True


if __name__ == "__main__":
    test_progressive_trainer()