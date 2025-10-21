"""
Daily Stock Scanner - Uses Trained ML Models
Scans all 10,825 stocks daily and saves predictions
Run this daily at 17:10 (after price updates)
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import json

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Setup logging
log_file = f"logs/daily_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
Path("logs").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

class DailyScanner:
    def __init__(self, models_dir="app/ml/models", stock_data_dir="stock_data"):
        self.models_dir = Path(models_dir)
        self.stock_data_dir = Path(stock_data_dir)
        self.models_loaded = False
        
    def load_models(self):
        """Load trained ML models"""
        try:
            import tensorflow as tf
            import pickle
            
            logging.info("Loading trained models...")
            
            # Load models
            self.lstm_model = tf.keras.models.load_model(self.models_dir / "lstm_model.keras")
            self.transformer_model = tf.keras.models.load_model(self.models_dir / "transformer_model.keras")
            self.cnn_model = tf.keras.models.load_model(self.models_dir / "cnn_model.keras")
            
            # Load scaler
            scaler_path = Path("app/ml/data/scaler.pkl")
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            
            self.models_loaded = True
            logging.info("✅ Models loaded successfully")
            
        except FileNotFoundError as e:
            logging.error(f"❌ Model files not found: {e}")
            logging.error("Run train_on_1000_stocks.py first to train the models")
            raise
        except Exception as e:
            logging.error(f"❌ Error loading models: {e}")
            raise
    
    def prepare_features(self, symbol):
        """Prepare features for a single stock"""
        try:
            from app.ml.data_preparation_local import DataPreparation
            
            # Create temp preparation instance  
            prep = DataPreparation(
                symbols=[symbol], 
                period='3y',
                stock_data_dir=str(self.stock_data_dir),
                use_local=True,
                use_fundamentals=True
            )
            
            # Load and prepare data
            data = prep.load_local_data(symbol)
            if data is None or len(data) < 100:
                return None, "Insufficient data"
            
            # Calculate indicators
            data = prep.calculate_all_indicators(data, symbol)
            if data is None or len(data) < 60:
                return None, "Failed to calculate indicators"
            
            # Get feature columns (exclude Close which is used for target)
            feature_cols = [col for col in data.columns if col != 'Close']
            
            # Get last 60 days of features
            features = data[feature_cols].iloc[-60:].values
            
            # Check if we have enough data
            if len(features) < 60:
                return None, f"Only {len(features)} days available"
            
            # Normalize
            features_scaled = self.scaler.transform(features)
            
            # Reshape for model input (1, 60, n_features)
            X = features_scaled.reshape(1, 60, -1)
            
            return X, None
            
        except Exception as e:
            return None, f"Error: {str(e)}"
    
    def predict(self, X):
        """Make ensemble prediction"""
        try:
            # Get predictions from each model
            lstm_pred = self.lstm_model.predict(X, verbose=0)[0][0]
            transformer_pred = self.transformer_model.predict(X, verbose=0)[0][0]
            cnn_pred = self.cnn_model.predict(X, verbose=0)[0][0]
            
            # Ensemble (weighted average)
            ensemble_pred = (0.40 * lstm_pred + 
                           0.35 * transformer_pred + 
                           0.25 * cnn_pred)
            
            # Calculate confidence (inverse of std deviation)
            predictions = [lstm_pred, transformer_pred, cnn_pred]
            std_dev = np.std(predictions)
            confidence = max(0, 1 - (std_dev * 10))  # Higher agreement = higher confidence
            
            return {
                'ensemble': float(ensemble_pred),
                'lstm': float(lstm_pred),
                'transformer': float(transformer_pred),
                'cnn': float(cnn_pred),
                'confidence': float(confidence)
            }
            
        except Exception as e:
            logging.error(f"Prediction error: {e}")
            return None
    
    def scan_all_stocks(self):
        """Scan all available stocks"""
        logging.info("=" * 70)
        logging.info("DAILY STOCK SCAN")
        logging.info("=" * 70)
        logging.info(f"Scanning stocks in: {self.stock_data_dir}")
        
        results = []
        scanned = 0
        successful = 0
        failed = 0
        
        # Get all stock folders
        stock_folders = sorted([f for f in self.stock_data_dir.iterdir() if f.is_dir()])
        total_stocks = len(stock_folders)
        
        logging.info(f"Found {total_stocks} stocks to scan...")
        
        for stock_folder in stock_folders:
            symbol = stock_folder.name
            scanned += 1
            
            if scanned % 100 == 0:
                logging.info(f"Progress: {scanned}/{total_stocks} ({successful} successful, {failed} failed)")
            
            # Prepare features
            X, error = self.prepare_features(symbol)
            if X is None:
                failed += 1
                continue
            
            # Make prediction
            prediction = self.predict(X)
            if prediction is None:
                failed += 1
                continue
            
            # Store result
            results.append({
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                **prediction
            })
            successful += 1
        
        logging.info(f"\n✅ Scan completed: {successful}/{total_stocks} stocks processed")
        logging.info(f"❌ Failed: {failed}")
        
        return results
    
    def save_predictions(self, results):
        """Save predictions to database"""
        try:
            from app.storage.db import get_db_session
            from app.storage.models import StockPrediction
            
            session = get_db_session()
            
            logging.info(f"\n💾 Saving {len(results)} predictions to database...")
            
            for result in results:
                prediction = StockPrediction(
                    symbol=result['symbol'],
                    prediction_date=datetime.now().date(),
                    predicted_price=result['ensemble'],
                    confidence_score=result['confidence'],
                    lstm_prediction=result['lstm'],
                    transformer_prediction=result['transformer'],
                    cnn_prediction=result['cnn'],
                    model_version='v1.0'
                )
                session.add(prediction)
            
            session.commit()
            session.close()
            
            logging.info("✅ Predictions saved to stock_predictions table")
            
        except Exception as e:
            logging.error(f"❌ Error saving predictions: {e}")
            
            # Fallback: save to JSON
            output_file = f"predictions_{datetime.now().strftime('%Y%m%d')}.json"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            logging.info(f"💾 Predictions saved to {output_file}")
    
    def get_top_opportunities(self, results, top_n=50):
        """Get top opportunities based on prediction and confidence"""
        # Sort by prediction * confidence
        sorted_results = sorted(
            results,
            key=lambda x: x['ensemble'] * x['confidence'],
            reverse=True
        )
        
        logging.info("\n🎯 TOP OPPORTUNITIES:")
        logging.info("=" * 70)
        
        for i, result in enumerate(sorted_results[:top_n], 1):
            logging.info(
                f"{i:2d}. {result['symbol']:6s} | "
                f"Prediction: {result['ensemble']:+.2%} | "
                f"Confidence: {result['confidence']:.2f} | "
                f"Score: {result['ensemble'] * result['confidence']:.4f}"
            )
        
        return sorted_results[:top_n]


def main():
    logging.info("Starting Daily Stock Scanner...")
    
    try:
        scanner = DailyScanner()
        
        # Load models
        scanner.load_models()
        
        # Scan all stocks
        results = scanner.scan_all_stocks()
        
        # Save predictions
        scanner.save_predictions(results)
        
        # Show top opportunities
        scanner.get_top_opportunities(results, top_n=50)
        
        logging.info("\n" + "=" * 70)
        logging.info("✅ DAILY SCAN COMPLETED")
        logging.info("=" * 70)
        
    except Exception as e:
        logging.error(f"\n❌ SCAN FAILED: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
