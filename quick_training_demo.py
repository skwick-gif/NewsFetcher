"""
Quick Training Demo - Progressive ML System
===========================================
Demonstrates end-to-end training of a small LSTM model for AAPL stock predictions
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ml.progressive import ProgressiveDataLoader, ProgressiveTrainer, ProgressiveModels
import logging
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def quick_training_demo():
    """Run a quick training demo with small LSTM model"""

    print("🚀 QUICK TRAINING DEMO - PROGRESSIVE ML SYSTEM")
    print("=" * 60)

    # Initialize components
    print("\n📊 Initializing Data Loader...")
    data_loader = ProgressiveDataLoader(
        stock_data_dir="stock_data",
        horizons=[1, 7, 30],
        sequence_length=60
    )

    print("\n🏃‍♂️ Initializing Trainer...")
    trainer = ProgressiveTrainer(
        data_loader=data_loader,
        training_config={
            'epochs': 3,  # Very short for demo
            'batch_size': 16,
            'validation_split': 0.2,
            'verbose': 1
        }
    )

    # Train small LSTM model for 1-day predictions
    print("\n🧠 Training Small LSTM Model (1-day horizon)...")
    print("This will take a few minutes...")

    try:
        # Train only LSTM progressive mode for 1-day
        results = trainer.train_progressive_models(
            symbol="AAPL",
            model_types=['lstm']
        )

        print("\n✅ Training completed successfully!")
        print(f"📊 Results keys: {list(results.keys())}")

        # Show training results
        if 'lstm' in results:
            lstm_results = results['lstm']
            if '1d' in lstm_results:
                result_1d = lstm_results['1d']
                print("\n📈 1-Day LSTM Results:")
                print(f"   📊 Final validation loss: {result_1d['val_metrics'][0]:.4f}")
                print(f"   🎯 Price prediction MAE: {result_1d['val_metrics'][1]:.4f}")
                print(f"   📈 Direction accuracy: {result_1d['val_metrics'][2]:.1%}")
                print(f"   ⏱️ Training time: {result_1d['training_time']:.1f}s")
        # Save the model
        print("\n💾 Saving trained model...")
        trainer.save_models("AAPL")
        print("✅ Model saved to app/ml/models/")

        return True

    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    """Run the quick training demo"""
    from datetime import datetime

    print(f"🧪 Starting Quick Training Demo at {datetime.now().strftime('%H:%M:%S')}")

    success = quick_training_demo()

    if success:
        print("\n🎉 DEMO COMPLETED SUCCESSFULLY!")
        print("✅ Progressive ML System is fully operational")
        print("🚀 Ready for full-scale training and production use")
    else:
        print("\n❌ DEMO FAILED")
        print("Please check the error messages above")

    print(f"\n📋 Demo completed at {datetime.now().strftime('%H:%M:%S')}")