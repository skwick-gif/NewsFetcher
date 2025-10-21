"""
ML Model Training - LSTM + Transformer + CNN Ensemble
======================================================

אימון 3 מודלים:
1. LSTM (40%) - תלות זמנית ארוכת טווח
2. Transformer (35%) - attention mechanisms
3. CNN (25%) - זיהוי patterns

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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# GPU Configuration with mixed precision
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        # Enable mixed precision for faster training
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        logger.info(f"GPU available: {physical_devices[0]} with mixed precision enabled")
    except:
        logger.info("GPU configuration failed, using CPU")
else:
    logger.info("No GPU found, using CPU")


class LSTMModel:
    """LSTM Model for time series prediction"""
    
    def __init__(self, sequence_length, n_features):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model = None
        self.history = None
    
    def build_model(self):
        """Build LSTM architecture"""
        model = keras.Sequential([
            # Layer 1: LSTM with return sequences
            layers.LSTM(50, return_sequences=True, 
                       input_shape=(self.sequence_length, self.n_features)),
            layers.Dropout(0.2),
            
            # Layer 2: LSTM with return sequences
            layers.LSTM(50, return_sequences=True),
            layers.Dropout(0.2),
            
            # Layer 3: LSTM without return sequences
            layers.LSTM(50, return_sequences=False),
            layers.Dropout(0.2),
            
            # Dense layers
            layers.Dense(25, activation='relu'),
            layers.Dropout(0.2),
            
            # Output layer - predicting future return
            layers.Dense(1)
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model
        logger.info(f"LSTM Model built: {model.count_params()} parameters")
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, 
              checkpoint_dir='ml/models/checkpoints', initial_epoch=0):
        """Train LSTM model"""
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,  # Reduced from 15 for faster training
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,  # Reduced from 5 for faster convergence
                min_lr=0.00001,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=str(checkpoint_dir / 'lstm_checkpoint_{epoch:03d}.keras'),
                monitor='val_loss',
                save_best_only=False,
                save_freq=5,  # Save every 5 epochs instead of every epoch
                verbose=1
            ),
            ModelCheckpoint(
                filepath=str(checkpoint_dir / 'lstm_best_model.keras'),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        logger.info("Training LSTM model...")
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
            initial_epoch=initial_epoch
        )
        
        return self.history
    
    def load_checkpoint(self, checkpoint_path):
        """Load model from checkpoint"""
        if Path(checkpoint_path).exists():
            self.model = keras.models.load_model(checkpoint_path)
            logger.info(f"LSTM model loaded from {checkpoint_path}")
            return True
        return False
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X, verbose=0)


class TransformerModel:
    """Transformer Model with multi-head attention"""
    
    def __init__(self, sequence_length, n_features):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model = None
        self.history = None
    
    def transformer_encoder(self, inputs, head_size, num_heads, ff_dim, dropout=0.2):
        """Transformer encoder block"""
        # Multi-head attention
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=head_size,
            dropout=dropout
        )(inputs, inputs)
        
        attention_output = layers.Dropout(dropout)(attention_output)
        attention_output = layers.LayerNormalization(epsilon=1e-6)(
            inputs + attention_output
        )
        
        # Feed-forward network
        ff_output = layers.Conv1D(filters=ff_dim, kernel_size=1, activation='relu')(
            attention_output
        )
        ff_output = layers.Dropout(dropout)(ff_output)
        ff_output = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(ff_output)
        ff_output = layers.Dropout(dropout)(ff_output)
        
        # Residual connection and normalization
        output = layers.LayerNormalization(epsilon=1e-6)(
            attention_output + ff_output
        )
        
        return output
    
    def build_model(self):
        """Build Transformer architecture"""
        inputs = keras.Input(shape=(self.sequence_length, self.n_features))
        
        # Positional encoding (simple version)
        x = inputs
        
        # Transformer blocks
        x = self.transformer_encoder(x, head_size=64, num_heads=8, ff_dim=128)
        x = self.transformer_encoder(x, head_size=64, num_heads=8, ff_dim=128)
        
        # Global pooling
        x = layers.GlobalAveragePooling1D()(x)
        
        # Dense layers
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        
        # Output
        outputs = layers.Dense(1)(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model
        logger.info(f"Transformer Model built: {model.count_params()} parameters")
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32,
              checkpoint_dir='ml/models/checkpoints', initial_epoch=0):
        """Train Transformer model"""
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,  # Reduced for faster training
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,  # Reduced for faster convergence
                min_lr=0.00001,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=str(checkpoint_dir / 'transformer_checkpoint_{epoch:03d}.keras'),
                monitor='val_loss',
                save_best_only=False,
                save_freq=5,  # Save every 5 epochs
                verbose=1
            ),
            ModelCheckpoint(
                filepath=str(checkpoint_dir / 'transformer_best_model.keras'),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        logger.info("Training Transformer model...")
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
            initial_epoch=initial_epoch
        )
        
        return self.history
    
    def load_checkpoint(self, checkpoint_path):
        """Load model from checkpoint"""
        if Path(checkpoint_path).exists():
            self.model = keras.models.load_model(checkpoint_path)
            logger.info(f"Transformer model loaded from {checkpoint_path}")
            return True
        return False
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X, verbose=0)


class CNNModel:
    """CNN Model for pattern recognition"""
    
    def __init__(self, sequence_length, n_features):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model = None
        self.history = None
    
    def build_model(self):
        """Build CNN architecture"""
        model = keras.Sequential([
            # Conv block 1
            layers.Conv1D(filters=64, kernel_size=3, activation='relu',
                         input_shape=(self.sequence_length, self.n_features)),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.2),
            
            # Conv block 2
            layers.Conv1D(filters=128, kernel_size=3, activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.2),
            
            # Conv block 3
            layers.Conv1D(filters=256, kernel_size=3, activation='relu'),
            layers.BatchNormalization(),
            layers.GlobalMaxPooling1D(),
            
            # Dense layers
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            
            # Output
            layers.Dense(1)
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model
        logger.info(f"CNN Model built: {model.count_params()} parameters")
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32,
              checkpoint_dir='ml/models/checkpoints', initial_epoch=0):
        """Train CNN model"""
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,  # Reduced for faster training
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,  # Reduced for faster convergence
                min_lr=0.00001,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=str(checkpoint_dir / 'cnn_checkpoint_{epoch:03d}.keras'),
                monitor='val_loss',
                save_best_only=False,
                save_freq=5,  # Save every 5 epochs
                verbose=1
            ),
            ModelCheckpoint(
                filepath=str(checkpoint_dir / 'cnn_best_model.keras'),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        logger.info("Training CNN model...")
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
            initial_epoch=initial_epoch
        )
        
        return self.history
    
    def load_checkpoint(self, checkpoint_path):
        """Load model from checkpoint"""
        if Path(checkpoint_path).exists():
            self.model = keras.models.load_model(checkpoint_path)
            logger.info(f"CNN model loaded from {checkpoint_path}")
            return True
        return False
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X, verbose=0)


class EnsembleModel:
    """Ensemble of LSTM, Transformer, and CNN"""
    
    def __init__(self, lstm_weight=0.4, transformer_weight=0.35, cnn_weight=0.25):
        self.lstm_model = None
        self.transformer_model = None
        self.cnn_model = None
        
        self.lstm_weight = lstm_weight
        self.transformer_weight = transformer_weight
        self.cnn_weight = cnn_weight
        
        # Verify weights sum to 1
        total = lstm_weight + transformer_weight + cnn_weight
        assert abs(total - 1.0) < 0.001, f"Weights must sum to 1.0, got {total}"
    
    def set_models(self, lstm_model, transformer_model, cnn_model):
        """Set the trained models"""
        self.lstm_model = lstm_model
        self.transformer_model = transformer_model
        self.cnn_model = cnn_model
    
    def predict(self, X):
        """Make ensemble predictions"""
        lstm_pred = self.lstm_model.predict(X)
        transformer_pred = self.transformer_model.predict(X)
        cnn_pred = self.cnn_model.predict(X)
        
        # Weighted average
        ensemble_pred = (
            self.lstm_weight * lstm_pred +
            self.transformer_weight * transformer_pred +
            self.cnn_weight * cnn_pred
        )
        
        return ensemble_pred


def find_latest_checkpoint(checkpoint_dir, model_name):
    """Find the latest checkpoint for a model"""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None, 0
    
    # Find all checkpoint files for this model
    pattern = f"{model_name}_checkpoint_*.keras"
    checkpoints = list(checkpoint_dir.glob(pattern))
    
    if not checkpoints:
        return None, 0
    
    # Extract epoch numbers and find the latest
    latest_epoch = 0
    latest_file = None
    
    for checkpoint in checkpoints:
        try:
            # Extract epoch number from filename
            epoch_str = checkpoint.stem.split('_')[-1]
            epoch = int(epoch_str)
            if epoch > latest_epoch:
                latest_epoch = epoch
                latest_file = checkpoint
        except ValueError:
            continue
    
    return latest_file, latest_epoch


def save_training_state(checkpoint_dir, model_name, epoch, batch_size, total_epochs):
    """Save training state metadata"""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    state = {
        'model_name': model_name,
        'current_epoch': epoch,
        'batch_size': batch_size,
        'total_epochs': total_epochs,
        'last_update': datetime.now().isoformat(),
        'status': 'training'
    }
    
    state_file = checkpoint_dir / f"{model_name}_training_state.json"
    with open(state_file, 'w') as f:
        json.dump(state, f, indent=2)


def load_training_state(checkpoint_dir, model_name):
    """Load training state metadata"""
    checkpoint_dir = Path(checkpoint_dir)
    state_file = checkpoint_dir / f"{model_name}_training_state.json"
    
    if not state_file.exists():
        return None
    
    with open(state_file, 'r') as f:
        return json.load(f)


def load_prepared_data(data_dir='ml/data'):
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
    
    # We need to recreate X and y from the CSV files
    # Since we saved features but not the sequences
    # We'll load the features and recreate sequences
    
    all_X = []
    all_y = []
    
    # Get the expected features from the scaler
    expected_features = scaler.feature_names_in_ if hasattr(scaler, 'feature_names_in_') else None
    
    for symbol in symbols:
        csv_path = data_dir / f"{symbol}_features.csv"
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        
        # Get feature columns (exclude Close which is used for target)
        feature_cols = [col for col in df.columns if col != 'Close']
        
        # If we have expected features, align the dataframe
        if expected_features is not None:
            # Add missing columns with 0
            for col in expected_features:
                if col not in df.columns:
                    df[col] = 0.0
            
            # Reorder columns to match expected order
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


def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model performance"""
    predictions = model.predict(X_test).flatten()
    
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    
    # Direction accuracy (did we predict up/down correctly?)
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


def plot_training_history(histories, model_names, output_dir='ml/models'):
    """Plot training history for all models"""
    output_dir = Path(output_dir)
    
    fig, axes = plt.subplots(len(histories), 2, figsize=(15, 5 * len(histories)))
    
    if len(histories) == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (history, name) in enumerate(zip(histories, model_names)):
        # Loss plot
        axes[idx, 0].plot(history.history['loss'], label='Train Loss')
        axes[idx, 0].plot(history.history['val_loss'], label='Val Loss')
        axes[idx, 0].set_title(f'{name} - Loss')
        axes[idx, 0].set_xlabel('Epoch')
        axes[idx, 0].set_ylabel('Loss')
        axes[idx, 0].legend()
        axes[idx, 0].grid(True)
        
        # MAE plot
        axes[idx, 1].plot(history.history['mae'], label='Train MAE')
        axes[idx, 1].plot(history.history['val_mae'], label='Val MAE')
        axes[idx, 1].set_title(f'{name} - MAE')
        axes[idx, 1].set_xlabel('Epoch')
        axes[idx, 1].set_ylabel('MAE')
        axes[idx, 1].legend()
        axes[idx, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_history.png', dpi=150)
    logger.info(f"Training history plot saved to {output_dir / 'training_history.png'}")
    plt.close()


def main(batch_size=32, resume_training=True, max_epochs=100):
    """Main training pipeline"""
    logger.info("\n" + "="*70)
    logger.info("🚀 STARTING ML MODEL TRAINING")
    logger.info("="*70 + "\n")
    
    # Create models directory
    models_dir = Path('ml/models')
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Create checkpoints directory
    checkpoint_dir = models_dir / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"💾 Checkpoints will be saved to: {checkpoint_dir}")
    logger.info(f"📦 Batch size: {batch_size}")
    logger.info(f"🔄 Resume training: {resume_training}")
    
    # Load data
    logger.info("📥 Loading prepared data...")
    X, y, scaler, symbols = load_prepared_data()
    
    # Split data: 80% train, 10% validation, 10% test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42, shuffle=True
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.111, random_state=42, shuffle=True
    )  # 0.111 of 90% = 10% of total
    
    logger.info(f"📊 Data split:")
    logger.info(f"   Train: {X_train.shape[0]} samples ({X_train.shape[0]/len(X):.1%})")
    logger.info(f"   Val:   {X_val.shape[0]} samples ({X_val.shape[0]/len(X):.1%})")
    logger.info(f"   Test:  {X_test.shape[0]} samples ({X_test.shape[0]/len(X):.1%})")
    
    sequence_length = X.shape[1]
    n_features = X.shape[2]
    
    logger.info(f"\n📐 Input shape: ({sequence_length}, {n_features})")
    logger.info(f"   Sequence length: {sequence_length} days")
    logger.info(f"   Features: {n_features}")
    
    # Initialize results storage
    results = {
        'training_date': datetime.now().isoformat(),
        'batch_size': batch_size,
        'max_epochs': max_epochs,
        'symbols': symbols,
        'data_shape': {
            'total_samples': int(len(X)),
            'train_samples': int(len(X_train)),
            'val_samples': int(len(X_val)),
            'test_samples': int(len(X_test)),
            'sequence_length': int(sequence_length),
            'n_features': int(n_features)
        },
        'models': {},
        'ensemble_weights': {
            'lstm': 0.4,
            'transformer': 0.35,
            'cnn': 0.25
        }
    }
    
    # ========== LSTM Model ==========
    logger.info("\n" + "-"*70)
    logger.info("🔵 LSTM MODEL")
    logger.info("-"*70)
    
    lstm_model = LSTMModel(sequence_length, n_features)
    lstm_initial_epoch = 0
    
    # Check for existing checkpoint
    if resume_training:
        latest_checkpoint, latest_epoch = find_latest_checkpoint(checkpoint_dir, 'lstm')
        if latest_checkpoint and lstm_model.load_checkpoint(latest_checkpoint):
            lstm_initial_epoch = latest_epoch + 1
            logger.info(f"🔄 Resuming LSTM training from epoch {lstm_initial_epoch}")
        else:
            lstm_model.build_model()
            logger.info("🆕 Starting LSTM training from scratch")
    else:
        lstm_model.build_model()
        logger.info("🆕 Starting LSTM training from scratch")
    
    if lstm_initial_epoch < max_epochs:
        lstm_history = lstm_model.train(
            X_train, y_train, X_val, y_val, 
            epochs=max_epochs, batch_size=batch_size,
            checkpoint_dir=checkpoint_dir, initial_epoch=lstm_initial_epoch
        )
        lstm_metrics = evaluate_model(lstm_model, X_test, y_test, "LSTM")
        
        # Save final model
        lstm_model.model.save(models_dir / 'lstm_model.keras')
        logger.info(f"💾 LSTM model saved to {models_dir / 'lstm_model.keras'}")
        
        # Save training state
        save_training_state(checkpoint_dir, 'lstm', max_epochs, batch_size, max_epochs)
        
        results['models']['lstm'] = lstm_metrics
    else:
        logger.info("✅ LSTM training already completed")
        lstm_metrics = evaluate_model(lstm_model, X_test, y_test, "LSTM")
        results['models']['lstm'] = lstm_metrics
    
    # ========== Transformer Model ==========
    logger.info("\n" + "-"*70)
    logger.info("🟢 TRANSFORMER MODEL")
    logger.info("-"*70)
    
    transformer_model = TransformerModel(sequence_length, n_features)
    transformer_initial_epoch = 0
    
    # Check for existing checkpoint
    if resume_training:
        latest_checkpoint, latest_epoch = find_latest_checkpoint(checkpoint_dir, 'transformer')
        if latest_checkpoint and transformer_model.load_checkpoint(latest_checkpoint):
            transformer_initial_epoch = latest_epoch + 1
            logger.info(f"🔄 Resuming Transformer training from epoch {transformer_initial_epoch}")
        else:
            transformer_model.build_model()
            logger.info("🆕 Starting Transformer training from scratch")
    else:
        transformer_model.build_model()
        logger.info("🆕 Starting Transformer training from scratch")
    
    if transformer_initial_epoch < max_epochs:
        transformer_history = transformer_model.train(
            X_train, y_train, X_val, y_val,
            epochs=max_epochs, batch_size=batch_size,
            checkpoint_dir=checkpoint_dir, initial_epoch=transformer_initial_epoch
        )
        transformer_metrics = evaluate_model(transformer_model, X_test, y_test, "Transformer")
        
        # Save final model
        transformer_model.model.save(models_dir / 'transformer_model.keras')
        logger.info(f"💾 Transformer model saved to {models_dir / 'transformer_model.keras'}")
        
        # Save training state
        save_training_state(checkpoint_dir, 'transformer', max_epochs, batch_size, max_epochs)
        
        results['models']['transformer'] = transformer_metrics
    else:
        logger.info("✅ Transformer training already completed")
        transformer_metrics = evaluate_model(transformer_model, X_test, y_test, "Transformer")
        results['models']['transformer'] = transformer_metrics
    
    # ========== CNN Model ==========
    logger.info("\n" + "-"*70)
    logger.info("🟡 CNN MODEL")
    logger.info("-"*70)
    
    cnn_model = CNNModel(sequence_length, n_features)
    cnn_initial_epoch = 0
    
    # Check for existing checkpoint
    if resume_training:
        latest_checkpoint, latest_epoch = find_latest_checkpoint(checkpoint_dir, 'cnn')
        if latest_checkpoint and cnn_model.load_checkpoint(latest_checkpoint):
            cnn_initial_epoch = latest_epoch + 1
            logger.info(f"🔄 Resuming CNN training from epoch {cnn_initial_epoch}")
        else:
            cnn_model.build_model()
            logger.info("🆕 Starting CNN training from scratch")
    else:
        cnn_model.build_model()
        logger.info("🆕 Starting CNN training from scratch")
    
    if cnn_initial_epoch < max_epochs:
        cnn_history = cnn_model.train(
            X_train, y_train, X_val, y_val,
            epochs=max_epochs, batch_size=batch_size,
            checkpoint_dir=checkpoint_dir, initial_epoch=cnn_initial_epoch
        )
        cnn_metrics = evaluate_model(cnn_model, X_test, y_test, "CNN")
        
        # Save final model
        cnn_model.model.save(models_dir / 'cnn_model.keras')
        logger.info(f"💾 CNN model saved to {models_dir / 'cnn_model.keras'}")
        
        # Save training state
        save_training_state(checkpoint_dir, 'cnn', max_epochs, batch_size, max_epochs)
        
        results['models']['cnn'] = cnn_metrics
    else:
        logger.info("✅ CNN training already completed")
        cnn_metrics = evaluate_model(cnn_model, X_test, y_test, "CNN")
        results['models']['cnn'] = cnn_metrics
    
    # ========== Ensemble Model ==========
    logger.info("\n" + "-"*70)
    logger.info("🟣 ENSEMBLE MODEL")
    logger.info("-"*70)
    
    ensemble = EnsembleModel(lstm_weight=0.4, transformer_weight=0.35, cnn_weight=0.25)
    ensemble.set_models(lstm_model, transformer_model, cnn_model)
    ensemble_metrics = evaluate_model(ensemble, X_test, y_test, "Ensemble")
    results['models']['ensemble'] = ensemble_metrics
    
    # ========== Save Results ==========
    with open(models_dir / 'training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"💾 Training results saved to {models_dir / 'training_results.json'}")
    
    # Plot training history if we have any
    histories = []
    model_names = []
    
    if 'lstm_history' in locals():
        histories.append(lstm_history)
        model_names.append('LSTM')
    if 'transformer_history' in locals():
        histories.append(transformer_history)
        model_names.append('Transformer')
    if 'cnn_history' in locals():
        histories.append(cnn_history)
        model_names.append('CNN')
    
    if histories:
        plot_training_history(histories, model_names, models_dir)
    
    # ========== Summary ==========
    logger.info("\n" + "="*70)
    logger.info("✅ TRAINING COMPLETE!")
    logger.info("="*70)
    logger.info(f"\n📊 Final Results:")
    if 'lstm' in results['models']:
        logger.info(f"   LSTM:        Direction Accuracy = {results['models']['lstm']['direction_accuracy']:.2%}")
    if 'transformer' in results['models']:
        logger.info(f"   Transformer: Direction Accuracy = {results['models']['transformer']['direction_accuracy']:.2%}")
    if 'cnn' in results['models']:
        logger.info(f"   CNN:         Direction Accuracy = {results['models']['cnn']['direction_accuracy']:.2%}")
    if 'ensemble' in results['models']:
        logger.info(f"   Ensemble:    Direction Accuracy = {results['models']['ensemble']['direction_accuracy']:.2%}")
    
    logger.info(f"\n💾 Models saved to: {models_dir}")
    logger.info(f"   - checkpoints/ (training checkpoints)")
    logger.info(f"   - lstm_model.keras")
    logger.info(f"   - transformer_model.keras")
    logger.info(f"   - cnn_model.keras")
    logger.info(f"   - training_results.json")
    if histories:
        logger.info(f"   - training_history.png")
    logger.info("\n🎯 Next steps:")
    logger.info("   1. Review training_results.json for detailed metrics")
    if histories:
        logger.info("   2. Check training_history.png for convergence")
    logger.info("   3. Integrate models with scanner for predictions")
    logger.info("="*70 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train ML models with checkpoint support')
    parser.add_argument('--batch-size', type=int, default=32, 
                       help='Batch size for training (default: 32)')
    parser.add_argument('--no-resume', action='store_true', 
                       help='Start training from scratch (ignore checkpoints)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Maximum number of epochs (default: 100)')
    
    args = parser.parse_args()
    
    main(
        batch_size=args.batch_size,
        resume_training=not args.no_resume,
        max_epochs=args.epochs
    )
