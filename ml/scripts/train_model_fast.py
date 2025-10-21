"""
ML Model Training - FAST VERSION with Checkpoints
=================================================

אימון 3 מודלים מזורז:
1. LSTM (40%) - תלות זמנית ארוכת טווח  
2. Transformer (35%) - attention mechanisms
3. CNN (25%) - זיהוי patterns

תכונות מהירות:
- Batch size גדול (128 במקום 32)
- ModelCheckpoint - המשך מנקודת עצירה
- Mixed Precision - זירוז חישובים
- Early stopping חכם יותר
- פחות epochs (50 במקום 100)

Ensemble: weighted average של התחזיות
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import json
from pathlib import Path
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# GPU Configuration + Mixed Precision
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        logger.info(f"GPU available: {physical_devices[0]}")
        # Enable mixed precision
        policy = keras.mixed_precision.Policy('mixed_float16')
        keras.mixed_precision.set_global_policy(policy)
        logger.info("Mixed precision enabled")
    except:
        logger.info("GPU configuration failed, using CPU")
else:
    logger.info("No GPU found, using CPU")


def load_prepared_data(data_dir='app/ml/data'):
    """Load all prepared data from CSV files"""
    data_dir = Path(data_dir)
    
    # Load summary
    with open(data_dir / 'preparation_summary.json', 'r') as f:
        summary = json.load(f)
    
    symbols = summary['symbols']
    sequence_length = summary['sequence_length']
    
    logger.info(f"Loading data for {len(symbols)} symbols...")
    
    # Load scaler
    with open(data_dir / 'scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    all_X = []
    all_y = []
    
    expected_features = scaler.feature_names_in_ if hasattr(scaler, 'feature_names_in_') else None
    
    for symbol in symbols:
        csv_path = data_dir / f"{symbol}_features.csv"
        if not csv_path.exists():
            continue
            
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        
        # Get feature columns (exclude Close which is used for target)
        feature_cols = [col for col in df.columns if col != 'Close']
        
        if expected_features is not None:
            for col in expected_features:
                if col not in df.columns:
                    df[col] = 0.0
            feature_cols = [col for col in expected_features if col in df.columns]
        
        # Scale features
        scaled_features = scaler.transform(df[feature_cols])
        
        # Create sequences
        for i in range(sequence_length, len(scaled_features) - 1):
            X_seq = scaled_features[i - sequence_length:i]
            
            # Calculate future return as target
            current_price = df['Close'].iloc[i]
            future_price = df['Close'].iloc[i + 1]
            future_return = (future_price - current_price) / current_price
            
            all_X.append(X_seq)
            all_y.append(future_return)
    
    X = np.array(all_X)
    y = np.array(all_y)
    
    logger.info(f"Loaded data: X shape {X.shape}, y shape {y.shape}")
    
    return X, y, scaler, symbols


class FastLSTMModel:
    """Fast LSTM Model with checkpoints"""
    
    def __init__(self, sequence_length, n_features, model_dir="app/ml/models"):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.history = None
        self.checkpoint_path = self.model_dir / "lstm_checkpoint.keras"
    
    def build_model(self):
        """Build LSTM architecture"""
        model = keras.Sequential([
            layers.LSTM(64, return_sequences=True, 
                       input_shape=(self.sequence_length, self.n_features)),
            layers.Dropout(0.3),
            
            layers.LSTM(64, return_sequences=True),
            layers.Dropout(0.3),
            
            layers.LSTM(32, return_sequences=False),
            layers.Dropout(0.2),
            
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            
            # Mixed precision requires float32 output
            layers.Dense(1, dtype='float32')
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model
        logger.info(f"Fast LSTM Model built: {model.count_params()} parameters")
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=128):
        """Train LSTM model with checkpoints"""
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,  # Reduced patience
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=4,  # Reduced patience
                min_lr=0.00001,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=str(self.checkpoint_path),
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            )
        ]
        
        # Check if checkpoint exists
        if self.checkpoint_path.exists():
            logger.info(f"🔄 Loading checkpoint from {self.checkpoint_path}")
            self.model = keras.models.load_model(str(self.checkpoint_path))
        
        logger.info("Training Fast LSTM model...")
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,  # Increased batch size
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X, verbose=0)


class FastTransformerModel:
    """Fast Transformer Model with checkpoints"""
    
    def __init__(self, sequence_length, n_features, model_dir="app/ml/models"):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.history = None
        self.checkpoint_path = self.model_dir / "transformer_checkpoint.keras"
    
    def transformer_encoder(self, inputs, head_size, num_heads, ff_dim, dropout=0.2):
        """Transformer encoder block"""
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=head_size,
            dropout=dropout
        )(inputs, inputs)
        
        attention_output = layers.Dropout(dropout)(attention_output)
        attention_output = layers.LayerNormalization(epsilon=1e-6)(
            inputs + attention_output
        )
        
        ff_output = layers.Conv1D(filters=ff_dim, kernel_size=1, activation='relu')(
            attention_output
        )
        ff_output = layers.Dropout(dropout)(ff_output)
        ff_output = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(ff_output)
        ff_output = layers.Dropout(dropout)(ff_output)
        
        output = layers.LayerNormalization(epsilon=1e-6)(
            attention_output + ff_output
        )
        
        return output
    
    def build_model(self):
        """Build Fast Transformer architecture"""
        inputs = keras.Input(shape=(self.sequence_length, self.n_features))
        
        x = inputs
        
        # Smaller transformer blocks for speed
        x = self.transformer_encoder(x, head_size=32, num_heads=4, ff_dim=64)
        x = self.transformer_encoder(x, head_size=32, num_heads=4, ff_dim=64)
        
        x = layers.GlobalAveragePooling1D()(x)
        
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        
        # Mixed precision requires float32 output
        outputs = layers.Dense(1, dtype='float32')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model
        logger.info(f"Fast Transformer Model built: {model.count_params()} parameters")
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=128):
        """Train Transformer model with checkpoints"""
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=4,
                min_lr=0.00001,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=str(self.checkpoint_path),
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            )
        ]
        
        # Check if checkpoint exists
        if self.checkpoint_path.exists():
            logger.info(f"🔄 Loading checkpoint from {self.checkpoint_path}")
            self.model = keras.models.load_model(str(self.checkpoint_path))
        
        logger.info("Training Fast Transformer model...")
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X, verbose=0)


class FastCNNModel:
    """Fast CNN Model with checkpoints"""
    
    def __init__(self, sequence_length, n_features, model_dir="app/ml/models"):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.history = None
        self.checkpoint_path = self.model_dir / "cnn_checkpoint.keras"
    
    def build_model(self):
        """Build Fast CNN architecture"""
        model = keras.Sequential([
            layers.Conv1D(filters=32, kernel_size=3, activation='relu',
                         input_shape=(self.sequence_length, self.n_features)),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.2),
            
            layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.2),
            
            layers.Conv1D(filters=128, kernel_size=3, activation='relu'),
            layers.BatchNormalization(),
            layers.GlobalMaxPooling1D(),
            
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            
            # Mixed precision requires float32 output
            layers.Dense(1, dtype='float32')
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model
        logger.info(f"Fast CNN Model built: {model.count_params()} parameters")
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=128):
        """Train CNN model with checkpoints"""
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=4,
                min_lr=0.00001,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=str(self.checkpoint_path),
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            )
        ]
        
        # Check if checkpoint exists
        if self.checkpoint_path.exists():
            logger.info(f"🔄 Loading checkpoint from {self.checkpoint_path}")
            self.model = keras.models.load_model(str(self.checkpoint_path))
        
        logger.info("Training Fast CNN model...")
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X, verbose=0)


def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model performance"""
    predictions = model.predict(X_test).flatten()
    
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    
    direction_correct = np.sum((predictions > 0) == (y_test > 0))
    direction_accuracy = direction_correct / len(y_test)
    
    metrics = {
        'model': model_name,
        'mse': float(mse),
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': float(r2),
        'direction_accuracy': float(direction_accuracy)
    }
    
    logger.info(f"\n{'='*60}")
    logger.info(f"{model_name} Evaluation:")
    logger.info(f"  MSE: {mse:.6f}")
    logger.info(f"  MAE: {mae:.6f}")
    logger.info(f"  RMSE: {rmse:.6f}")
    logger.info(f"  R²: {r2:.4f}")
    logger.info(f"  Direction Accuracy: {direction_accuracy:.2%}")
    logger.info(f"{'='*60}\n")
    
    return metrics


def main():
    """Main training pipeline - FAST VERSION"""
    logger.info("\n" + "="*70)
    logger.info("🚀 STARTING FAST ML MODEL TRAINING")
    logger.info("="*70 + "\n")
    
    models_dir = Path('app/ml/models')
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info("📥 Loading prepared data...")
    X, y, scaler, symbols = load_prepared_data()
    
    # Split data
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42, shuffle=True
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.111, random_state=42, shuffle=True
    )
    
    logger.info(f"📊 Data split:")
    logger.info(f"   Train: {X_train.shape[0]} samples ({X_train.shape[0]/len(X):.1%})")
    logger.info(f"   Val:   {X_val.shape[0]} samples ({X_val.shape[0]/len(X):.1%})")
    logger.info(f"   Test:  {X_test.shape[0]} samples ({X_test.shape[0]/len(X):.1%})")
    
    sequence_length = X.shape[1]
    n_features = X.shape[2]
    
    logger.info(f"\n📐 Input shape: ({sequence_length}, {n_features})")
    logger.info(f"   Sequence length: {sequence_length} days")
    logger.info(f"   Features: {n_features}")
    
    # Training parameters - FAST VERSION
    epochs = 50  # Reduced from 100
    batch_size = 128  # Increased from 32
    
    # ========== Fast LSTM Model ==========
    logger.info("\n" + "-"*70)
    logger.info("🔵 FAST LSTM MODEL")
    logger.info("-"*70)
    
    lstm_model = FastLSTMModel(sequence_length, n_features)
    lstm_model.build_model()
    lstm_history = lstm_model.train(X_train, y_train, X_val, y_val, epochs, batch_size)
    lstm_metrics = evaluate_model(lstm_model, X_test, y_test, "Fast LSTM")
    
    # Save LSTM model
    lstm_model.model.save(models_dir / 'fast_lstm_model.keras')
    logger.info(f"💾 Fast LSTM model saved to {models_dir / 'fast_lstm_model.keras'}")
    
    # ========== Fast Transformer Model ==========
    logger.info("\n" + "-"*70)
    logger.info("🟢 FAST TRANSFORMER MODEL")
    logger.info("-"*70)
    
    transformer_model = FastTransformerModel(sequence_length, n_features)
    transformer_model.build_model()
    transformer_history = transformer_model.train(X_train, y_train, X_val, y_val, epochs, batch_size)
    transformer_metrics = evaluate_model(transformer_model, X_test, y_test, "Fast Transformer")
    
    # Save Transformer model
    transformer_model.model.save(models_dir / 'fast_transformer_model.keras')
    logger.info(f"💾 Fast Transformer model saved to {models_dir / 'fast_transformer_model.keras'}")
    
    # ========== Fast CNN Model ==========
    logger.info("\n" + "-"*70)
    logger.info("🟡 FAST CNN MODEL")
    logger.info("-"*70)
    
    cnn_model = FastCNNModel(sequence_length, n_features)
    cnn_model.build_model()
    cnn_history = cnn_model.train(X_train, y_train, X_val, y_val, epochs, batch_size)
    cnn_metrics = evaluate_model(cnn_model, X_test, y_test, "Fast CNN")
    
    # Save CNN model
    cnn_model.model.save(models_dir / 'fast_cnn_model.keras')
    logger.info(f"💾 Fast CNN model saved to {models_dir / 'fast_cnn_model.keras'}")
    
    # ========== Save Results ==========
    results = {
        'training_date': datetime.now().isoformat(),
        'symbols': symbols,
        'training_mode': 'fast',
        'epochs': epochs,
        'batch_size': batch_size,
        'data_shape': {
            'total_samples': int(len(X)),
            'train_samples': int(len(X_train)),
            'val_samples': int(len(X_val)),
            'test_samples': int(len(X_test)),
            'sequence_length': int(sequence_length),
            'n_features': int(n_features)
        },
        'models': {
            'fast_lstm': lstm_metrics,
            'fast_transformer': transformer_metrics,
            'fast_cnn': cnn_metrics
        }
    }
    
    with open(models_dir / 'fast_training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"💾 Fast training results saved to {models_dir / 'fast_training_results.json'}")
    
    # ========== Summary ==========
    logger.info("\n" + "="*70)
    logger.info("✅ FAST TRAINING COMPLETE!")
    logger.info("="*70)
    logger.info(f"\n📊 Final Results:")
    logger.info(f"   Fast LSTM:        Direction Accuracy = {lstm_metrics['direction_accuracy']:.2%}")
    logger.info(f"   Fast Transformer: Direction Accuracy = {transformer_metrics['direction_accuracy']:.2%}")
    logger.info(f"   Fast CNN:         Direction Accuracy = {cnn_metrics['direction_accuracy']:.2%}")
    logger.info(f"\n💾 Fast models saved to: {models_dir}")
    logger.info(f"   - fast_lstm_model.keras")
    logger.info(f"   - fast_transformer_model.keras")
    logger.info(f"   - fast_cnn_model.keras")
    logger.info(f"   - fast_training_results.json")
    logger.info(f"\n🔄 Checkpoints available for resuming training:")
    logger.info(f"   - lstm_checkpoint.keras")
    logger.info(f"   - transformer_checkpoint.keras")
    logger.info(f"   - cnn_checkpoint.keras")
    logger.info("="*70 + "\n")


if __name__ == "__main__":
    main()