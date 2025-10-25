"""
Progressive ML Models for Multi-Horizon Stock Predictions
=========================================================
Advanced neural network models for Progressive Training System

Models Included:
- LSTMModel: Recurrent model for sequential patterns
- TransformerModel: Attention-based model for long-term dependencies  
- CNNModel: Convolutional model for feature extraction
- EnsembleModel: Combined predictions from multiple models
- UnifiedModel: Single model handling all horizons simultaneously

Features:
- Progressive training support (1→7→30 days)
- Unified training support (all horizons together)
- Both regression (price prediction) and classification (direction prediction)
- Ensemble predictions with confidence scoring
- Model checkpointing and resumption
- PyTorch CUDA GPU acceleration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path
import json
import pickle
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set PyTorch device and GPU configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"🚀 Using device: {device}")

if torch.cuda.is_available():
    logger.info(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"   CUDA Version: {torch.version.cuda}")
    logger.info(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    logger.info("ℹ️ Running on CPU (no GPU detected)")


class LSTMModel(nn.Module):
    """PyTorch LSTM model for sequential pattern recognition"""
    
    def __init__(self, 
                 input_size: int,
                 sequence_length: int,
                 horizons: List[int] = [1, 7, 30],
                 mode: str = "progressive",
                 model_params: Dict = None):
        """
        Initialize LSTM Model
        
        Args:
            input_size: Number of input features
            sequence_length: Length of input sequences
            horizons: Prediction horizons [1, 7, 30]
            mode: "progressive" or "unified"
            model_params: Model hyperparameters
        """
        super(LSTMModel, self).__init__()
        
        self.input_size = input_size
        self.sequence_length = sequence_length
        self.horizons = horizons
        self.mode = mode
        
        # Default parameters
        self.params = {
            'lstm_units': [128, 64, 32],
            'dropout_rate': 0.3,
            'learning_rate': 0.001,
            'num_layers': 3
        }
        
        if model_params:
            self.params.update(model_params)
        
        self.device = device
        self._build_layers()
        
        # Move model to device
        self.to(self.device)
        
        logger.info(f"🧠 LSTM Model initialized ({mode} mode)")
        logger.info(f"   📊 Input size: {input_size}, Sequence length: {sequence_length}")
        logger.info(f"   ⏰ Horizons: {horizons}")
        logger.info(f"   🎯 Device: {self.device}")
    
    def _build_layers(self):
        """Build LSTM layers"""
        lstm_units = self.params['lstm_units']
        dropout_rate = self.params['dropout_rate']
        num_layers = self.params['num_layers']
        
        # LSTM layers
        self.lstm_layers = nn.ModuleList()
        
        # First LSTM layer
        self.lstm_layers.append(
            nn.LSTM(
                input_size=self.input_size,
                hidden_size=lstm_units[0],
                num_layers=1,
                batch_first=True,
                dropout=0 if num_layers == 1 else dropout_rate
            )
        )
        
        # Additional LSTM layers
        for i in range(1, len(lstm_units)):
            self.lstm_layers.append(
                nn.LSTM(
                    input_size=lstm_units[i-1],
                    hidden_size=lstm_units[i],
                    num_layers=1,
                    batch_first=True,
                    dropout=0 if i == len(lstm_units)-1 else dropout_rate
                )
            )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)
        
        # Output layers based on mode
        if self.mode == "progressive":
            # Separate heads per horizon (regression + classification)
            self.regression_heads = nn.ModuleDict()
            self.classification_heads = nn.ModuleDict()
            for horizon in self.horizons:
                self.regression_heads[f'{horizon}d'] = nn.Sequential(
                    nn.Linear(lstm_units[-1], 64),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(64, 1)
                )
                self.classification_heads[f'{horizon}d'] = nn.Sequential(
                    nn.Linear(lstm_units[-1], 64),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(64, 1),
                    nn.Sigmoid()
                )
        else:
            # Unified output for all horizons
            self.output_layer = nn.Sequential(
                nn.Linear(lstm_units[-1], 64),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(64, len(self.horizons))  # Output for all horizons
            )
    
    def forward(self, x, horizon=None):
        """Forward pass"""
        # x shape: (batch_size, sequence_length, input_size)
        # Ensure input is on the same device as the model
        x = x.to(self.device)
        
        # Pass through LSTM layers
        hidden = x
        for lstm in self.lstm_layers:
            hidden, _ = lstm(hidden)
            hidden = self.dropout(hidden)
        
        # Take the last timestep
        hidden = hidden[:, -1, :]  # (batch_size, hidden_size)
        
        if self.mode == "progressive":
            if horizon is None:
                raise ValueError("horizon must be specified for progressive mode")
            reg = self.regression_heads[f'{horizon}d'](hidden)
            # Bound regression output to a reasonable range to avoid exploding prices
            reg = torch.tanh(reg)
            clf = self.classification_heads[f'{horizon}d'](hidden)
            return reg, clf
        else:
            return self.output_layer(hidden)


    
    def get_optimizer(self):
        """Get PyTorch optimizer"""
        return torch.optim.Adam(self.parameters(), lr=self.params['learning_rate'])
    
    def get_loss_fn(self):
        """Get loss function"""
        return nn.MSELoss()


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism for Transformer"""
    
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size()
        
        # Linear transformations
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        
        return self.out(attended)


class TransformerBlock(nn.Module):
    """Transformer block with self-attention and feed-forward"""
    
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Self-attention with residual connection
        attn_out = self.attention(x)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward with residual connection
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))
        
        return x


class TransformerModel(nn.Module):
    """Transformer model with attention mechanism"""
    
    def __init__(self, 
                 input_size: int,
                 sequence_length: int,
                 horizons: List[int] = [1, 7, 30],
                 mode: str = "progressive",
                 model_params: Dict = None):
        """Initialize Transformer Model"""
        super(TransformerModel, self).__init__()
        
        self.input_size = input_size
        self.sequence_length = sequence_length
        self.horizons = horizons
        self.mode = mode
        
        # Default parameters
        self.params = {
            'num_heads': 8,
            'ff_dim': 256,
            'num_transformer_blocks': 2,
            'embed_dim': 128,
            'dropout_rate': 0.1,
            'learning_rate': 0.0001
        }
        
        if model_params:
            self.params.update(model_params)
        
        self.device = device
        self._build_layers()
        
        # Move model to device
        self.to(self.device)
        
        logger.info(f"🎯 Transformer Model initialized ({mode} mode)")
        logger.info(f"   📊 Input size: {input_size}, Sequence length: {sequence_length}")
        logger.info(f"   ⏰ Horizons: {horizons}")
        logger.info(f"   🎯 Device: {self.device}")
    
    def _build_layers(self):
        """Build Transformer layers"""
        embed_dim = self.params['embed_dim']
        num_heads = self.params['num_heads']
        ff_dim = self.params['ff_dim']
        num_blocks = self.params['num_transformer_blocks']
        dropout_rate = self.params['dropout_rate']
        
        # Input projection
        self.input_projection = nn.Linear(self.input_size, embed_dim)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, self.sequence_length, embed_dim))
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout_rate)
            for _ in range(num_blocks)
        ])
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Output layers based on mode
        if self.mode == "progressive":
            # Separate heads per horizon (regression + classification)
            self.regression_heads = nn.ModuleDict()
            self.classification_heads = nn.ModuleDict()
            for horizon in self.horizons:
                self.regression_heads[f'{horizon}d'] = nn.Sequential(
                    nn.Linear(embed_dim, 64),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(64, 1)
                )
                self.classification_heads[f'{horizon}d'] = nn.Sequential(
                    nn.Linear(embed_dim, 64),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(64, 1),
                    nn.Sigmoid()
                )
        else:
            # Unified output for all horizons
            self.output_layer = nn.Sequential(
                nn.Linear(embed_dim, 64),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(64, len(self.horizons))
            )
    
    def forward(self, x, horizon=None):
        """Forward pass"""
        # x shape: (batch_size, sequence_length, input_size)
        # Ensure input is on the same device as the model
        x = x.to(self.device)
        
        # Input projection
        x = self.input_projection(x)
        
        # Add positional encoding
        x = x + self.pos_encoding
        
        # Pass through transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        
        # Global average pooling over sequence dimension
        x = x.transpose(1, 2)  # (batch_size, embed_dim, sequence_length)
        x = self.global_pool(x).squeeze(-1)  # (batch_size, embed_dim)
        
        # Output
        if self.mode == "progressive":
            if horizon is None:
                raise ValueError("horizon must be specified for progressive mode")
            reg = self.regression_heads[f'{horizon}d'](x)
            # Bound regression output to a reasonable range to avoid exploding prices
            reg = torch.tanh(reg)
            clf = self.classification_heads[f'{horizon}d'](x)
            return reg, clf
        else:
            return self.output_layer(x)
    
    def get_optimizer(self):
        """Get PyTorch optimizer"""
        return torch.optim.Adam(self.parameters(), lr=self.params['learning_rate'])
    
    def get_loss_fn(self):
        """Get loss function"""
        return nn.MSELoss()


class CNNModel(nn.Module):
    """CNN model for feature extraction"""
    
    def __init__(self, 
                 input_size: int,
                 sequence_length: int,
                 horizons: List[int] = [1, 7, 30],
                 mode: str = "progressive",
                 model_params: Dict = None):
        """Initialize CNN Model"""
        super(CNNModel, self).__init__()
        
        self.input_size = input_size
        self.sequence_length = sequence_length
        self.horizons = horizons
        self.mode = mode
        
        # Default parameters
        self.params = {
            'filters': [64, 128, 64],
            'kernel_sizes': [3, 3, 3],
            'dropout_rate': 0.3,
            'learning_rate': 0.001
        }
        
        if model_params:
            self.params.update(model_params)
        
        self.device = device
        self._build_layers()
        
        # Move model to device
        self.to(self.device)
        
        logger.info(f"🔍 CNN Model initialized ({mode} mode)")
        logger.info(f"   📊 Input size: {input_size}, Sequence length: {sequence_length}")
        logger.info(f"   ⏰ Horizons: {horizons}")
        logger.info(f"   🎯 Device: {self.device}")
    
    def _build_layers(self):
        """Build CNN layers"""
        filters = self.params['filters']
        kernel_sizes = self.params['kernel_sizes']
        dropout_rate = self.params['dropout_rate']
        
        # Conv1D layers
        self.conv_layers = nn.ModuleList()
        
        # First conv layer
        self.conv_layers.append(
            nn.Sequential(
                nn.Conv1d(self.input_size, filters[0], kernel_size=kernel_sizes[0], padding=1),
                nn.BatchNorm1d(filters[0]),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            )
        )
        
        # Additional conv layers
        for i in range(1, len(filters)):
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv1d(filters[i-1], filters[i], kernel_size=kernel_sizes[i], padding=1),
                    nn.BatchNorm1d(filters[i]),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate)
                )
            )
        
        # Global max pooling
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        
        # Output layers based on mode
        if self.mode == "progressive":
            # Separate heads per horizon (regression + classification)
            self.regression_heads = nn.ModuleDict()
            self.classification_heads = nn.ModuleDict()
            for horizon in self.horizons:
                self.regression_heads[f'{horizon}d'] = nn.Sequential(
                    nn.Linear(filters[-1], 64),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(64, 1)
                )
                self.classification_heads[f'{horizon}d'] = nn.Sequential(
                    nn.Linear(filters[-1], 64),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(64, 1),
                    nn.Sigmoid()
                )
        else:
            # Unified output for all horizons
            self.output_layer = nn.Sequential(
                nn.Linear(filters[-1], 64),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(64, len(self.horizons))
            )
    
    def forward(self, x, horizon=None):
        """Forward pass"""
        # x shape: (batch_size, sequence_length, input_size)
        # Ensure input is on the same device as the model
        x = x.to(self.device)
        # Conv1D expects: (batch_size, input_size, sequence_length)
        x = x.transpose(1, 2)
        
        # Pass through conv layers
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        
        # Global max pooling
        x = self.global_pool(x).squeeze(-1)  # (batch_size, filters[-1])
        
        # Output
        if self.mode == "progressive":
            if horizon is None:
                raise ValueError("horizon must be specified for progressive mode")
            reg = self.regression_heads[f'{horizon}d'](x)
            # Bound regression output to a reasonable range to avoid exploding prices
            reg = torch.tanh(reg)
            clf = self.classification_heads[f'{horizon}d'](x)
            return reg, clf
        else:
            return self.output_layer(x)
    
    def get_optimizer(self):
        """Get PyTorch optimizer"""
        # Slight weight decay to improve generalization
        return torch.optim.Adam(self.parameters(), lr=self.params['learning_rate'], weight_decay=1e-5)
    
    def get_loss_fn(self):
        """Get loss function"""
        return nn.MSELoss()
# Legacy wrapper classes for backward compatibility
class ProgressiveModels:
    """Legacy wrapper for backward compatibility"""
    
    @staticmethod
    def create_lstm(input_shape, horizons, mode, **params):
        """Create LSTM model (legacy)"""
        sequence_length, input_size = input_shape
        return LSTMModel(
            input_size=input_size,
            sequence_length=sequence_length,
            horizons=horizons,
            mode=mode,
            model_params=params
        )
    
    @staticmethod
    def create_transformer(input_shape, horizons, mode, **params):
        """Create Transformer model (legacy)"""
        sequence_length, input_size = input_shape
        return TransformerModel(
            input_size=input_size,
            sequence_length=sequence_length,
            horizons=horizons,
            mode=mode,
            model_params=params
        )
    
    @staticmethod
    def create_cnn(input_shape, horizons, mode, **params):
        """Create CNN model (legacy)"""
        sequence_length, input_size = input_shape
        return CNNModel(
            input_size=input_size,
            sequence_length=sequence_length,
            horizons=horizons,
            mode=mode,
            model_params=params
        )


class EnsembleModel:
    """Ensemble model combining multiple PyTorch models"""
    
    def __init__(self, models: Dict[str, nn.Module], weights: Dict[str, float] = None):
        """
        Initialize ensemble model
        
        Args:
            models: Dictionary of model_name -> model
            weights: Dictionary of model_name -> weight (defaults to equal weights)
        """
        self.models = models
        self.device = device
        
        # Default equal weights
        if weights is None:
            num_models = len(models)
            self.weights = {name: 1.0/num_models for name in models.keys()}
        else:
            self.weights = weights
        
        # Move all models to device
        for model in self.models.values():
            model.to(self.device)
        
        logger.info(f"🎭 Ensemble model initialized with {len(models)} models")
    
    def predict(self, x: torch.Tensor, horizon: int = None) -> torch.Tensor:
        """Make ensemble prediction"""
        predictions = []
        
        for model_name, model in self.models.items():
            model.eval()
            with torch.no_grad():
                if hasattr(model, 'mode') and model.mode == "progressive":
                    pred = model(x, horizon=horizon)
                else:
                    pred = model(x)
                
                # Weight the prediction
                weighted_pred = pred * self.weights[model_name]
                predictions.append(weighted_pred)
        
        # Average weighted predictions
        ensemble_pred = torch.stack(predictions).sum(dim=0)
        return ensemble_pred
    
    def to(self, device):
        """Move ensemble to device"""
        self.device = device
        for model in self.models.values():
            model.to(device)
        return self


# Alias for backward compatibility
UnifiedModel = ProgressiveModels