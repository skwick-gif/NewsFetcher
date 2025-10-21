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
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path
import json
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set TensorFlow GPU configuration
try:
    # Try to configure GPU
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        logger.info(f"✅ GPU configured: {physical_devices[0]}")
    else:
        logger.info("ℹ️ Running on CPU (no GPU detected)")
except Exception as e:
    logger.warning(f"⚠️ GPU configuration warning: {e}")


class LSTMModel:
    """LSTM model for sequential pattern recognition"""
    
    def __init__(self, 
                 input_shape: Tuple[int, int],
                 horizons: List[int] = [1, 7, 30],
                 mode: str = "progressive",
                 model_params: Dict = None):
        """
        Initialize LSTM Model
        
        Args:
            input_shape: (sequence_length, num_features)
            horizons: Prediction horizons [1, 7, 30]
            mode: "progressive" or "unified"
            model_params: Model hyperparameters
        """
        self.input_shape = input_shape
        self.horizons = horizons
        self.mode = mode
        
        # Default parameters
        self.params = {
            'lstm_units': [128, 64, 32],
            'dropout_rate': 0.3,
            'l2_regularization': 0.001,
            'learning_rate': 0.001,
            'activation': 'tanh'
        }
        
        if model_params:
            self.params.update(model_params)
        
        self.models = {}
        self._build_models()
        
        logger.info(f"🧠 LSTM Model initialized ({mode} mode)")
        logger.info(f"   📊 Input shape: {input_shape}")
        logger.info(f"   ⏰ Horizons: {horizons}")
    
    def _build_models(self):
        """Build LSTM models based on mode"""
        if self.mode == "progressive":
            # Separate model for each horizon
            for horizon in self.horizons:
                self.models[f'{horizon}d'] = self._build_single_horizon_model(horizon)
        else:
            # Single unified model for all horizons
            self.models['unified'] = self._build_unified_model()
    
    def _build_single_horizon_model(self, horizon: int) -> keras.Model:
        """Build LSTM model for single horizon"""
        
        # Input layer
        inputs = keras.Input(shape=self.input_shape, name=f'input_{horizon}d')
        
        # LSTM layers with dropout
        x = inputs
        for i, units in enumerate(self.params['lstm_units']):
            return_sequences = i < len(self.params['lstm_units']) - 1
            x = layers.LSTM(
                units,
                return_sequences=return_sequences,
                dropout=self.params['dropout_rate'],
                recurrent_dropout=self.params['dropout_rate'],
                kernel_regularizer=keras.regularizers.l2(self.params['l2_regularization']),
                name=f'lstm_{i+1}_{horizon}d'
            )(x)
        
        # Additional dense layers
        x = layers.Dense(64, activation='relu', name=f'dense1_{horizon}d')(x)
        x = layers.Dropout(self.params['dropout_rate'])(x)
        x = layers.Dense(32, activation='relu', name=f'dense2_{horizon}d')(x)
        x = layers.Dropout(self.params['dropout_rate'])(x)
        
        # Dual outputs: regression and classification
        regression_output = layers.Dense(1, activation='linear', name=f'price_pred_{horizon}d')(x)
        classification_output = layers.Dense(1, activation='sigmoid', name=f'direction_pred_{horizon}d')(x)
        
        model = keras.Model(
            inputs=inputs,
            outputs=[regression_output, classification_output],
            name=f'LSTM_{horizon}d'
        )
        
        # Compile with dual loss
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.params['learning_rate']),
            loss={
                f'price_pred_{horizon}d': 'mse',
                f'direction_pred_{horizon}d': 'binary_crossentropy'
            },
            loss_weights={
                f'price_pred_{horizon}d': 1.0,
                f'direction_pred_{horizon}d': 2.0  # Higher weight for direction
            },
            metrics={
                f'price_pred_{horizon}d': ['mae'],
                f'direction_pred_{horizon}d': ['accuracy']
            }
        )
        
        return model
    
    def _build_unified_model(self) -> keras.Model:
        """Build unified LSTM model for all horizons"""
        
        # Input layer
        inputs = keras.Input(shape=self.input_shape, name='unified_input')
        
        # Shared LSTM layers
        x = inputs
        for i, units in enumerate(self.params['lstm_units']):
            return_sequences = i < len(self.params['lstm_units']) - 1
            x = layers.LSTM(
                units,
                return_sequences=return_sequences,
                dropout=self.params['dropout_rate'],
                recurrent_dropout=self.params['dropout_rate'],
                kernel_regularizer=keras.regularizers.l2(self.params['l2_regularization']),
                name=f'shared_lstm_{i+1}'
            )(x)
        
        # Shared dense layer
        shared_dense = layers.Dense(64, activation='relu', name='shared_dense')(x)
        shared_dropout = layers.Dropout(self.params['dropout_rate'])(shared_dense)
        
        # Separate heads for each horizon
        regression_outputs = []
        classification_outputs = []
        
        for horizon in self.horizons:
            # Horizon-specific dense layers
            horizon_dense = layers.Dense(32, activation='relu', name=f'dense_{horizon}d')(shared_dropout)
            horizon_dropout = layers.Dropout(self.params['dropout_rate'])(horizon_dense)
            
            # Outputs for this horizon
            reg_out = layers.Dense(1, activation='linear', name=f'price_pred_{horizon}d')(horizon_dropout)
            clf_out = layers.Dense(1, activation='sigmoid', name=f'direction_pred_{horizon}d')(horizon_dropout)
            
            regression_outputs.append(reg_out)
            classification_outputs.append(clf_out)
        
        all_outputs = regression_outputs + classification_outputs
        
        model = keras.Model(
            inputs=inputs,
            outputs=all_outputs,
            name='LSTM_Unified'
        )
        
        # Compile with multiple losses
        loss_dict = {}
        loss_weights_dict = {}
        metrics_dict = {}
        
        for i, horizon in enumerate(self.horizons):
            # Regression
            loss_dict[f'price_pred_{horizon}d'] = 'mse'
            loss_weights_dict[f'price_pred_{horizon}d'] = 1.0
            metrics_dict[f'price_pred_{horizon}d'] = ['mae']
            
            # Classification
            loss_dict[f'direction_pred_{horizon}d'] = 'binary_crossentropy'
            loss_weights_dict[f'direction_pred_{horizon}d'] = 2.0
            metrics_dict[f'direction_pred_{horizon}d'] = ['accuracy']
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.params['learning_rate']),
            loss=loss_dict,
            loss_weights=loss_weights_dict,
            metrics=metrics_dict
        )
        
        return model
    
    def get_model(self, horizon: Union[str, int] = None) -> keras.Model:
        """Get specific model"""
        if self.mode == "progressive":
            if horizon is None:
                raise ValueError("Horizon must be specified for progressive mode")
            key = f'{horizon}d' if isinstance(horizon, int) else horizon
            return self.models.get(key)
        else:
            return self.models['unified']
    
    def summary(self):
        """Print model summaries"""
        for name, model in self.models.items():
            print(f"\n📊 {name.upper()} MODEL SUMMARY:")
            print("=" * 60)
            model.summary()


class TransformerModel:
    """Transformer model with attention mechanism"""
    
    def __init__(self, 
                 input_shape: Tuple[int, int],
                 horizons: List[int] = [1, 7, 30],
                 mode: str = "progressive",
                 model_params: Dict = None):
        """Initialize Transformer Model"""
        
        self.input_shape = input_shape
        self.horizons = horizons
        self.mode = mode
        
        # Default parameters
        self.params = {
            'num_heads': 8,
            'ff_dim': 256,
            'num_transformer_blocks': 2,
            'mlp_units': [128, 64],
            'dropout_rate': 0.1,
            'learning_rate': 0.0001
        }
        
        if model_params:
            self.params.update(model_params)
        
        self.models = {}
        self._build_models()
        
        logger.info(f"🎭 Transformer Model initialized ({mode} mode)")
    
    def _build_models(self):
        """Build Transformer models"""
        if self.mode == "progressive":
            for horizon in self.horizons:
                self.models[f'{horizon}d'] = self._build_single_horizon_model(horizon)
        else:
            self.models['unified'] = self._build_unified_model()
    
    def _transformer_encoder(self, inputs, head_size, num_heads, ff_dim, dropout=0):
        """Transformer encoder block"""
        # Multi-head self-attention
        attention_layer = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=head_size, dropout=dropout
        )
        attention_output = attention_layer(inputs, inputs)
        attention_output = layers.Dropout(dropout)(attention_output)
        attention_output = layers.LayerNormalization(epsilon=1e-6)(inputs + attention_output)
        
        # Feed-forward network
        ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(inputs.shape[-1]),
        ])
        ffn_output = ffn(attention_output)
        ffn_output = layers.Dropout(dropout)(ffn_output)
        return layers.LayerNormalization(epsilon=1e-6)(attention_output + ffn_output)
    
    def _build_single_horizon_model(self, horizon: int) -> keras.Model:
        """Build Transformer model for single horizon"""
        
        inputs = keras.Input(shape=self.input_shape)
        
        # Positional encoding (simple version)
        positions = tf.range(start=0, limit=self.input_shape[0], delta=1)
        positions = tf.cast(positions, tf.float32)
        position_embedding = layers.Embedding(
            input_dim=self.input_shape[0], output_dim=self.input_shape[1]
        )(positions)
        
        x = inputs + position_embedding
        
        # Transformer blocks
        head_size = self.input_shape[1] // self.params['num_heads']
        for _ in range(self.params['num_transformer_blocks']):
            x = self._transformer_encoder(
                x, head_size, self.params['num_heads'], 
                self.params['ff_dim'], self.params['dropout_rate']
            )
        
        # Global average pooling
        x = layers.GlobalAveragePooling1D()(x)
        
        # MLP layers
        for units in self.params['mlp_units']:
            x = layers.Dense(units, activation="relu")(x)
            x = layers.Dropout(self.params['dropout_rate'])(x)
        
        # Dual outputs
        regression_output = layers.Dense(1, activation='linear', name=f'price_pred_{horizon}d')(x)
        classification_output = layers.Dense(1, activation='sigmoid', name=f'direction_pred_{horizon}d')(x)
        
        model = keras.Model(inputs=inputs, outputs=[regression_output, classification_output])
        
        # Compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.params['learning_rate']),
            loss=['mse', 'binary_crossentropy'],
            loss_weights=[1.0, 2.0],
            metrics=[['mae'], ['accuracy']]
        )
        
        return model
    
    def _build_unified_model(self) -> keras.Model:
        """Build unified Transformer model"""
        # Similar to LSTM but with Transformer layers
        # Implementation similar to unified LSTM but with transformer_encoder calls
        
        inputs = keras.Input(shape=self.input_shape)
        
        # Positional encoding
        positions = tf.range(start=0, limit=self.input_shape[0], delta=1)
        position_embedding = layers.Embedding(
            input_dim=self.input_shape[0], output_dim=self.input_shape[1]
        )(positions)
        
        x = inputs + position_embedding
        
        # Transformer blocks
        head_size = self.input_shape[1] // self.params['num_heads']
        for _ in range(self.params['num_transformer_blocks']):
            x = self._transformer_encoder(
                x, head_size, self.params['num_heads'],
                self.params['ff_dim'], self.params['dropout_rate']
            )
        
        x = layers.GlobalAveragePooling1D()(x)
        shared_dense = layers.Dense(128, activation="relu")(x)
        
        # Multi-horizon outputs
        regression_outputs = []
        classification_outputs = []
        
        for horizon in self.horizons:
            horizon_dense = layers.Dense(64, activation="relu")(shared_dense)
            horizon_dropout = layers.Dropout(self.params['dropout_rate'])(horizon_dense)
            
            reg_out = layers.Dense(1, activation='linear', name=f'price_pred_{horizon}d')(horizon_dropout)
            clf_out = layers.Dense(1, activation='sigmoid', name=f'direction_pred_{horizon}d')(horizon_dropout)
            
            regression_outputs.append(reg_out)
            classification_outputs.append(clf_out)
        
        model = keras.Model(inputs=inputs, outputs=regression_outputs + classification_outputs)
        
        # Compile with multi-task losses
        loss_dict = {}
        loss_weights_dict = {}
        metrics_dict = {}
        
        for horizon in self.horizons:
            loss_dict[f'price_pred_{horizon}d'] = 'mse'
            loss_weights_dict[f'price_pred_{horizon}d'] = 1.0
            metrics_dict[f'price_pred_{horizon}d'] = ['mae']
            
            loss_dict[f'direction_pred_{horizon}d'] = 'binary_crossentropy'
            loss_weights_dict[f'direction_pred_{horizon}d'] = 2.0
            metrics_dict[f'direction_pred_{horizon}d'] = ['accuracy']
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.params['learning_rate']),
            loss=loss_dict,
            loss_weights=loss_weights_dict,
            metrics=metrics_dict
        )
        
        return model
    
    def get_model(self, horizon: Union[str, int] = None) -> keras.Model:
        """Get specific model"""
        if self.mode == "progressive":
            key = f'{horizon}d' if isinstance(horizon, int) else horizon
            return self.models.get(key)
        else:
            return self.models['unified']


class CNNModel:
    """CNN model for feature extraction from time series"""
    
    def __init__(self, 
                 input_shape: Tuple[int, int],
                 horizons: List[int] = [1, 7, 30],
                 mode: str = "progressive",
                 model_params: Dict = None):
        """Initialize CNN Model"""
        
        self.input_shape = input_shape
        self.horizons = horizons
        self.mode = mode
        
        # Default parameters
        self.params = {
            'filters': [64, 128, 64],
            'kernel_sizes': [3, 3, 3],
            'pool_sizes': [2, 2, 2],
            'dense_units': [128, 64],
            'dropout_rate': 0.3,
            'learning_rate': 0.001
        }
        
        if model_params:
            self.params.update(model_params)
        
        self.models = {}
        self._build_models()
        
        logger.info(f"🔍 CNN Model initialized ({mode} mode)")
    
    def _build_models(self):
        """Build CNN models"""
        if self.mode == "progressive":
            for horizon in self.horizons:
                self.models[f'{horizon}d'] = self._build_single_horizon_model(horizon)
        else:
            self.models['unified'] = self._build_unified_model()
    
    def _build_single_horizon_model(self, horizon: int) -> keras.Model:
        """Build CNN model for single horizon"""
        
        inputs = keras.Input(shape=self.input_shape)
        
        # Reshape for Conv1D if needed
        x = layers.Reshape((self.input_shape[0], self.input_shape[1]))(inputs)
        
        # Convolutional layers
        for i, (filters, kernel_size, pool_size) in enumerate(zip(
            self.params['filters'], 
            self.params['kernel_sizes'], 
            self.params['pool_sizes']
        )):
            x = layers.Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                activation='relu',
                padding='same',
                name=f'conv1d_{i+1}_{horizon}d'
            )(x)
            x = layers.MaxPooling1D(pool_size=pool_size)(x)
            x = layers.Dropout(self.params['dropout_rate'])(x)
        
        # Flatten and dense layers
        x = layers.GlobalMaxPooling1D()(x)
        
        for units in self.params['dense_units']:
            x = layers.Dense(units, activation='relu')(x)
            x = layers.Dropout(self.params['dropout_rate'])(x)
        
        # Dual outputs
        regression_output = layers.Dense(1, activation='linear', name=f'price_pred_{horizon}d')(x)
        classification_output = layers.Dense(1, activation='sigmoid', name=f'direction_pred_{horizon}d')(x)
        
        model = keras.Model(inputs=inputs, outputs=[regression_output, classification_output])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.params['learning_rate']),
            loss=['mse', 'binary_crossentropy'],
            loss_weights=[1.0, 2.0],
            metrics=[['mae'], ['accuracy']]
        )
        
        return model
    
    def _build_unified_model(self) -> keras.Model:
        """Build unified CNN model"""
        inputs = keras.Input(shape=self.input_shape)
        
        x = layers.Reshape((self.input_shape[0], self.input_shape[1]))(inputs)
        
        # Shared convolutional layers
        for i, (filters, kernel_size, pool_size) in enumerate(zip(
            self.params['filters'], 
            self.params['kernel_sizes'], 
            self.params['pool_sizes']
        )):
            x = layers.Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', padding='same')(x)
            x = layers.MaxPooling1D(pool_size=pool_size)(x)
            x = layers.Dropout(self.params['dropout_rate'])(x)
        
        x = layers.GlobalMaxPooling1D()(x)
        shared_dense = layers.Dense(128, activation='relu')(x)
        
        # Multi-horizon outputs
        regression_outputs = []
        classification_outputs = []
        
        for horizon in self.horizons:
            horizon_dense = layers.Dense(64, activation='relu')(shared_dense)
            horizon_dropout = layers.Dropout(self.params['dropout_rate'])(horizon_dense)
            
            reg_out = layers.Dense(1, activation='linear', name=f'price_pred_{horizon}d')(horizon_dropout)
            clf_out = layers.Dense(1, activation='sigmoid', name=f'direction_pred_{horizon}d')(horizon_dropout)
            
            regression_outputs.append(reg_out)
            classification_outputs.append(clf_out)
        
        model = keras.Model(inputs=inputs, outputs=regression_outputs + classification_outputs)
        
        # Multi-task compilation
        loss_dict = {}
        loss_weights_dict = {}
        metrics_dict = {}
        
        for horizon in self.horizons:
            loss_dict[f'price_pred_{horizon}d'] = 'mse'
            loss_weights_dict[f'price_pred_{horizon}d'] = 1.0
            metrics_dict[f'price_pred_{horizon}d'] = ['mae']
            
            loss_dict[f'direction_pred_{horizon}d'] = 'binary_crossentropy'
            loss_weights_dict[f'direction_pred_{horizon}d'] = 2.0
            metrics_dict[f'direction_pred_{horizon}d'] = ['accuracy']
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.params['learning_rate']),
            loss=loss_dict,
            loss_weights=loss_weights_dict,
            metrics=metrics_dict
        )
        
        return model
    
    def get_model(self, horizon: Union[str, int] = None) -> keras.Model:
        """Get specific model"""
        if self.mode == "progressive":
            key = f'{horizon}d' if isinstance(horizon, int) else horizon
            return self.models.get(key)
        else:
            return self.models['unified']


class EnsembleModel:
    """Ensemble model combining LSTM, Transformer, and CNN"""
    
    def __init__(self, 
                 input_shape: Tuple[int, int],
                 horizons: List[int] = [1, 7, 30],
                 mode: str = "progressive"):
        """Initialize Ensemble Model"""
        
        self.input_shape = input_shape
        self.horizons = horizons
        self.mode = mode
        
        # Initialize individual models
        self.lstm_model = LSTMModel(input_shape, horizons, mode)
        self.transformer_model = TransformerModel(input_shape, horizons, mode)
        self.cnn_model = CNNModel(input_shape, horizons, mode)
        
        # Ensemble weights (can be learned or fixed)
        self.ensemble_weights = {
            'lstm': 0.4,
            'transformer': 0.35,
            'cnn': 0.25
        }
        
        logger.info(f"🎯 Ensemble Model initialized ({mode} mode)")
        logger.info(f"   ⚖️ Weights: LSTM={self.ensemble_weights['lstm']}, "
                   f"Transformer={self.ensemble_weights['transformer']}, "
                   f"CNN={self.ensemble_weights['cnn']}")
    
    def predict_ensemble(self, X: np.ndarray, horizon: Union[str, int] = None) -> Dict:
        """Make ensemble predictions"""
        
        # Get predictions from individual models
        lstm_pred = self.lstm_model.get_model(horizon).predict(X, verbose=0)
        transformer_pred = self.transformer_model.get_model(horizon).predict(X, verbose=0)
        cnn_pred = self.cnn_model.get_model(horizon).predict(X, verbose=0)
        
        # Handle different output formats
        if isinstance(lstm_pred, list):
            # Dual output (regression, classification)
            lstm_reg, lstm_clf = lstm_pred[0], lstm_pred[1]
            transformer_reg, transformer_clf = transformer_pred[0], transformer_pred[1]
            cnn_reg, cnn_clf = cnn_pred[0], cnn_pred[1]
        else:
            # Single output
            lstm_reg = lstm_pred
            transformer_reg = transformer_pred
            cnn_reg = cnn_pred
            lstm_clf = transformer_clf = cnn_clf = None
        
        # Ensemble regression predictions
        ensemble_reg = (
            self.ensemble_weights['lstm'] * lstm_reg +
            self.ensemble_weights['transformer'] * transformer_reg +
            self.ensemble_weights['cnn'] * cnn_reg
        )
        
        # Ensemble classification predictions (if available)
        ensemble_clf = None
        if lstm_clf is not None:
            ensemble_clf = (
                self.ensemble_weights['lstm'] * lstm_clf +
                self.ensemble_weights['transformer'] * transformer_clf +
                self.ensemble_weights['cnn'] * cnn_clf
            )
        
        # Calculate confidence based on prediction agreement
        predictions_reg = [lstm_reg.flatten(), transformer_reg.flatten(), cnn_reg.flatten()]
        reg_std = np.std(predictions_reg, axis=0)
        confidence = np.maximum(0, 1 - (reg_std * 5))  # Higher agreement = higher confidence
        
        return {
            'ensemble_regression': ensemble_reg,
            'ensemble_classification': ensemble_clf,
            'confidence': confidence,
            'individual_predictions': {
                'lstm': {'regression': lstm_reg, 'classification': lstm_clf},
                'transformer': {'regression': transformer_reg, 'classification': transformer_clf},
                'cnn': {'regression': cnn_reg, 'classification': cnn_clf}
            }
        }
    
    def get_individual_model(self, model_type: str, horizon: Union[str, int] = None):
        """Get individual model"""
        if model_type == 'lstm':
            return self.lstm_model.get_model(horizon)
        elif model_type == 'transformer':
            return self.transformer_model.get_model(horizon)
        elif model_type == 'cnn':
            return self.cnn_model.get_model(horizon)
        else:
            raise ValueError(f"Unknown model type: {model_type}")


# Convenience class for easy model creation
class ProgressiveModels:
    """Factory class for creating Progressive ML models"""
    
    @staticmethod
    def create_lstm(input_shape, horizons=[1, 7, 30], mode="progressive", **kwargs):
        """Create LSTM model"""
        return LSTMModel(input_shape, horizons, mode, kwargs)
    
    @staticmethod
    def create_transformer(input_shape, horizons=[1, 7, 30], mode="progressive", **kwargs):
        """Create Transformer model"""
        return TransformerModel(input_shape, horizons, mode, kwargs)
    
    @staticmethod
    def create_cnn(input_shape, horizons=[1, 7, 30], mode="progressive", **kwargs):
        """Create CNN model"""
        return CNNModel(input_shape, horizons, mode, kwargs)
    
    @staticmethod
    def create_ensemble(input_shape, horizons=[1, 7, 30], mode="progressive"):
        """Create Ensemble model"""
        return EnsembleModel(input_shape, horizons, mode)


# Alias for backward compatibility
UnifiedModel = ProgressiveModels


def test_progressive_models():
    """Test Progressive Models"""
    print("🧪 Testing Progressive Models...")
    
    # Test data shape (like our AAPL data)
    input_shape = (60, 44)  # 60 days, 44 features
    horizons = [1, 7, 30]
    
    print(f"\n📊 Input shape: {input_shape}")
    print(f"⏰ Horizons: {horizons}")
    
    # Test Progressive mode
    print("\n🔄 Testing Progressive Mode...")
    
    lstm_prog = ProgressiveModels.create_lstm(input_shape, horizons, "progressive")
    print(f"✅ LSTM Progressive: {len(lstm_prog.models)} models")
    
    transformer_prog = ProgressiveModels.create_transformer(input_shape, horizons, "progressive") 
    print(f"✅ Transformer Progressive: {len(transformer_prog.models)} models")
    
    cnn_prog = ProgressiveModels.create_cnn(input_shape, horizons, "progressive")
    print(f"✅ CNN Progressive: {len(cnn_prog.models)} models")
    
    # Test Unified mode
    print("\n🔗 Testing Unified Mode...")
    
    lstm_unified = ProgressiveModels.create_lstm(input_shape, horizons, "unified")
    print(f"✅ LSTM Unified: {len(lstm_unified.models)} models")
    
    transformer_unified = ProgressiveModels.create_transformer(input_shape, horizons, "unified")
    print(f"✅ Transformer Unified: {len(transformer_unified.models)} models")
    
    cnn_unified = ProgressiveModels.create_cnn(input_shape, horizons, "unified")
    print(f"✅ CNN Unified: {len(cnn_unified.models)} models")
    
    # Test Ensemble
    print("\n🎯 Testing Ensemble...")
    ensemble = ProgressiveModels.create_ensemble(input_shape, horizons, "progressive")
    print(f"✅ Ensemble created with 3 base models")
    
    # Test model retrieval
    print("\n📋 Testing Model Retrieval...")
    lstm_1d = lstm_prog.get_model(1)  # 1-day model
    lstm_unified_model = lstm_unified.get_model()  # Unified model
    
    print(f"✅ LSTM 1d model: {lstm_1d.name if lstm_1d else 'None'}")
    print(f"✅ LSTM Unified model: {lstm_unified_model.name if lstm_unified_model else 'None'}")
    
    # Show model structures
    if lstm_1d:
        print(f"\n📊 LSTM 1d Parameters: {lstm_1d.count_params():,}")
    if lstm_unified_model:
        print(f"📊 LSTM Unified Parameters: {lstm_unified_model.count_params():,}")
    
    print("\n✅ Progressive Models test completed successfully!")
    return True


if __name__ == "__main__":
    test_progressive_models()