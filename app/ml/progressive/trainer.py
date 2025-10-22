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
import matplotlib.pyplot as plt
from datetime import datetime
import time
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, classification_report

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"🚀 Trainer using device: {device}")

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
                'dropout_rate': 0.3,
                'learning_rate': 0.001
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
            'verbose': 1
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
    
    def save_model(self, model: nn.Module, model_name: str, horizon: str = None):
        """Save PyTorch model"""
        checkpoint_path = self.save_dir / f"{model_name}{'_'+horizon if horizon else ''}_best.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_class': model.__class__.__name__,
            'model_params': model.params if hasattr(model, 'params') else {},
            'horizons': model.horizons if hasattr(model, 'horizons') else [],
            'mode': model.mode if hasattr(model, 'mode') else 'progressive'
        }, checkpoint_path)
        logger.info(f"💾 Model saved: {checkpoint_path}")
        
    def load_model(self, model_name: str, horizon: str = None) -> nn.Module:
        """Load PyTorch model"""
        checkpoint_path = self.save_dir / f"{model_name}{'_'+horizon if horizon else ''}_best.pth"
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Model not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Create model based on saved class name
        model_class = checkpoint['model_class']
        if model_class == 'LSTMModel':
            model = LSTMModel(
                input_size=checkpoint.get('input_size', 20),
                sequence_length=checkpoint.get('sequence_length', 60),
                horizons=checkpoint.get('horizons', [1, 7, 30]),
                mode=checkpoint.get('mode', 'progressive'),
                model_params=checkpoint.get('model_params', {})
            )
        elif model_class == 'TransformerModel':
            model = TransformerModel(
                input_size=checkpoint.get('input_size', 20),
                sequence_length=checkpoint.get('sequence_length', 60),
                horizons=checkpoint.get('horizons', [1, 7, 30]),
                mode=checkpoint.get('mode', 'progressive'),
                model_params=checkpoint.get('model_params', {})
            )
        elif model_class == 'CNNModel':
            model = CNNModel(
                input_size=checkpoint.get('input_size', 20),
                sequence_length=checkpoint.get('sequence_length', 60),
                horizons=checkpoint.get('horizons', [1, 7, 30]),
                mode=checkpoint.get('mode', 'progressive'),
                model_params=checkpoint.get('model_params', {})
            )
        else:
            raise ValueError(f"Unknown model class: {model_class}")
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        logger.info(f"📥 Model loaded: {checkpoint_path}")
        return model
    
    def early_stopping_check(self, current_loss: float, best_loss: float, patience_counter: int) -> Tuple[bool, int]:
        """Check if training should stop early"""
        if current_loss < best_loss:
            return False, 0  # Continue training, reset patience
        else:
            patience_counter += 1
            if patience_counter >= self.training_config['early_stopping_patience']:
                return True, patience_counter  # Stop training
            return False, patience_counter  # Continue training
    
    def train_progressive_models(self, symbol: str = "AAPL", model_types: List[str] = ['lstm']) -> Dict:
        """
        Train models in progressive mode (1d → 7d → 30d)
        
        Args:
            symbol: Stock symbol to train on
            model_types: List of model types ['lstm', 'transformer', 'cnn']
        
        Returns:
            Dictionary with training results
        """
        
        logger.info(f"🚀 Starting Progressive Training for {symbol}")
        logger.info(f"📋 Model types: {model_types}")
        
        # Prepare data
        data = self.prepare_progressive_data(symbol)
        
        # Get input dimensions from first horizon
        sample_X = data['1d']['X_train']
        sequence_length, input_size = sample_X.shape[1], sample_X.shape[2]
        
        results = {}
        
        for model_type in model_types:
            logger.info(f"\n🧠 Training {model_type.upper()} models...")
            
            model_results = {}
            
            # Train each horizon progressively
            for horizon in self.data_loader.horizons:
                horizon_key = f'{horizon}d'
                
                logger.info(f"   📈 Training {model_type} for {horizon_key}...")
                
                # Get data for this horizon
                X_train = torch.FloatTensor(data[horizon_key]['X_train']).to(self.device)
                X_val = torch.FloatTensor(data[horizon_key]['X_val']).to(self.device)
                y_train = torch.FloatTensor(data[horizon_key]['y_reg_train']).to(self.device)
                y_val = torch.FloatTensor(data[horizon_key]['y_reg_val']).to(self.device)
                
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
                    horizon=horizon_key
                )
                
                model_results[horizon_key] = {
                    'history': history,
                    'final_loss': history['val_loss'][-1] if history['val_loss'] else float('inf')
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
                           X_val: torch.Tensor, y_val: torch.Tensor, model_name: str, horizon: str) -> Dict:
        """Train a single PyTorch model"""
        
        # Setup optimizer and loss function
        optimizer = model.get_optimizer()
        criterion = model.get_loss_fn()
        
        # Training parameters
        epochs = self.training_config['epochs']
        batch_size = self.training_config['batch_size']
        
        # Create data loaders
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        logger.info(f"   📊 Training on {len(X_train)} samples, validating on {len(X_val)} samples")
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                
                if model.mode == "progressive":
                    horizon_num = int(horizon.replace('d', ''))
                    outputs = model(batch_X, horizon=horizon_num)
                else:
                    outputs = model(batch_X)
                
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    if model.mode == "progressive":
                        horizon_num = int(horizon.replace('d', ''))
                        outputs = model(batch_X, horizon=horizon_num)
                    else:
                        outputs = model(batch_X)
                    
                    loss = criterion(outputs.squeeze(), batch_y)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            
            # Update history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            
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
        
        # Save training history
        history_path = self.save_dir / f"{model_name}{'_'+horizon if horizon else ''}_history.csv"
        pd.DataFrame(history).to_csv(history_path, index=False)
        
        logger.info(f"      Final validation loss: {best_val_loss:.4f}")
        
        return history
        
        # Prepare data
        data = self.prepare_unified_data(symbol)
        input_shape = data['X_train'].shape[1:]
        
        results = {}
        
        for model_type in model_types:
            logger.info(f"\n🧠 Training {model_type.upper()} unified model...")
            
            # Create model
            if model_type == 'lstm':
                model_creator = ProgressiveModels.create_lstm(
                    input_shape,
                    self.data_loader.horizons,
                    'unified',
                    **self.model_config['lstm_params']
                )
            elif model_type == 'transformer':
                model_creator = ProgressiveModels.create_transformer(
                    input_shape,
                    self.data_loader.horizons,
                    'unified',
                    **self.model_config['transformer_params']
                )
            elif model_type == 'cnn':
                model_creator = ProgressiveModels.create_cnn(
                    input_shape,
                    self.data_loader.horizons,
                    'unified',
                    **self.model_config['cnn_params']
                )
            else:
                logger.warning(f"⚠️ Unknown model type: {model_type}")
                continue
            
            # Get unified model
            model = model_creator.get_model()
            
            # Create callbacks
            callbacks = self.create_callbacks(f"{model_type}_{symbol}_unified")
            
            # Train model
            start_time = time.time()
            
            history = model.fit(
                data['X_train'],
                data['y_train'],
                validation_data=(data['X_val'], data['y_val']),
                epochs=self.training_config['epochs'],
                batch_size=self.training_config['batch_size'],
                callbacks=callbacks,
                verbose=self.training_config['verbose']
            )
            
            training_time = time.time() - start_time
            
            # Evaluate model
            val_metrics = model.evaluate(data['X_val'], data['y_val'], verbose=0)
            
            # Store results
            results[model_type] = {
                'model': model,
                'history': history.history,
                'training_time': training_time,
                'val_metrics': val_metrics,
                'best_epoch': len(history.history['loss'])
            }
            
            self.models[f"{model_type}_unified"] = model_creator
            
            logger.info(f"   ✅ {model_type} unified completed in {training_time:.1f}s")
            logger.info(f"   📊 Val Loss: {val_metrics[0]:.4f}")
        
        # Store training history
        self.training_history[f'unified_{symbol}'] = results
        
        logger.info(f"✅ Unified training completed for {symbol}")
        
        return results
    
    def create_ensemble(self, symbol: str = "AAPL", mode: str = "progressive") -> EnsembleModel:
        """Create ensemble model from trained individual models"""
        
        logger.info(f"🎯 Creating ensemble model ({mode} mode)...")
        
        # Get data shape
        if mode == "progressive":
            data = self.prepare_progressive_data(symbol)
            input_shape = data['1d']['X_train'].shape[1:]
        else:
            data = self.prepare_unified_data(symbol)
            input_shape = data['X_train'].shape[1:]
        
        # Create ensemble
        self.ensemble_model = ProgressiveModels.create_ensemble(
            input_shape,
            self.data_loader.horizons,
            mode
        )
        
        logger.info(f"✅ Ensemble model created")
        
        return self.ensemble_model
    
    def evaluate_models(self, symbol: str = "AAPL", mode: str = "progressive") -> Dict:
        """Evaluate trained models"""
        
        logger.info(f"📊 Evaluating models ({mode} mode)...")
        
        # Prepare test data
        if mode == "progressive":
            data = self.prepare_progressive_data(symbol)
        else:
            data = self.prepare_unified_data(symbol)
        
        evaluation_results = {}
        
        # Evaluate individual models
        for model_name, model_creator in self.models.items():
            if mode in model_name or (mode == "progressive" and "unified" not in model_name):
                logger.info(f"   📈 Evaluating {model_name}...")
                
                if mode == "progressive":
                    model_eval = {}
                    for horizon in self.data_loader.horizons:
                        horizon_key = f'{horizon}d'
                        model = model_creator.get_model(horizon)
                        
                        X_val = data[horizon_key]['X_val']
                        y_reg_val = data[horizon_key]['y_reg_val']
                        y_clf_val = data[horizon_key]['y_clf_val']
                        
                        # Get predictions
                        predictions = model.predict(X_val, verbose=0)
                        pred_reg, pred_clf = predictions[0], predictions[1]
                        
                        # Calculate metrics
                        mse = mean_squared_error(y_reg_val, pred_reg)
                        mae = mean_absolute_error(y_reg_val, pred_reg)
                        accuracy = accuracy_score(y_clf_val, (pred_clf > 0.5).astype(int))
                        
                        model_eval[horizon_key] = {
                            'mse': mse,
                            'mae': mae,
                            'rmse': np.sqrt(mse),
                            'accuracy': accuracy
                        }
                    
                    evaluation_results[model_name] = model_eval
                    
                else:  # unified
                    model = model_creator.get_model()
                    predictions = model.predict(data['X_val'], verbose=0)
                    
                    # Handle multiple outputs
                    model_eval = {}
                    for i, horizon in enumerate(self.data_loader.horizons):
                        horizon_key = f'{horizon}d'
                        
                        # Extract predictions for this horizon
                        pred_reg = predictions[i]
                        pred_clf = predictions[i + len(self.data_loader.horizons)]
                        
                        y_reg_val = data['y_val'][f'price_pred_{horizon}d']
                        y_clf_val = data['y_val'][f'direction_pred_{horizon}d']
                        
                        # Calculate metrics
                        mse = mean_squared_error(y_reg_val, pred_reg)
                        mae = mean_absolute_error(y_reg_val, pred_reg)
                        accuracy = accuracy_score(y_clf_val, (pred_clf > 0.5).astype(int))
                        
                        model_eval[horizon_key] = {
                            'mse': mse,
                            'mae': mae,
                            'rmse': np.sqrt(mse),
                            'accuracy': accuracy
                        }
                    
                    evaluation_results[model_name] = model_eval
        
        logger.info(f"✅ Model evaluation completed")
        
        return evaluation_results
    
    def save_models(self, symbol: str = "AAPL"):
        """Save all trained models"""
        
        logger.info(f"💾 Saving models for {symbol}...")
        
        for model_name, model_creator in self.models.items():
            model_dir = self.save_dir / f"{model_name}_{symbol}"
            model_dir.mkdir(exist_ok=True)
            
            if hasattr(model_creator, 'models'):
                for horizon_name, model in model_creator.models.items():
                    save_path = model_dir / f"{horizon_name}.h5"
                    model.save(str(save_path))
                    logger.info(f"   💾 Saved {model_name}_{horizon_name}")
        
        # Save training history
        history_path = self.save_dir / f"training_history_{symbol}.json"
        with open(history_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_history = {}
            for key, value in self.training_history.items():
                serializable_history[key] = self._make_json_serializable(value)
            
            json.dump(serializable_history, f, indent=2)
        
        logger.info(f"✅ Models saved to {self.save_dir}")
    
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
    
    # Test callback creation
    print("\n⚙️ Testing callbacks...")
    callbacks = trainer.create_callbacks("test_model", "1d")
    print(f"✅ Created {len(callbacks)} callbacks")
    
    print("\n✅ Progressive Trainer test completed successfully!")
    return True


if __name__ == "__main__":
    test_progressive_trainer()