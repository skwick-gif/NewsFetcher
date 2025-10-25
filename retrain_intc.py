#!/usr/bin/env python3
"""
Retrain INTC models with correct feature count
"""

import sys
import os
import logging

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

# Set up environment
os.environ['PYTHONPATH'] = os.path.abspath('.')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from app.ml.progressive.trainer import ProgressiveTrainer
    from app.ml.progressive.data_loader import ProgressiveDataLoader
    
    print("🚀 Starting INTC model retraining with 35 features...")
    
    # Initialize components
    data_loader = ProgressiveDataLoader()
    trainer = ProgressiveTrainer(data_loader=data_loader)
    
    print("📊 Training transformer models for INTC...")
    
    # Train models for INTC
    result = trainer.train_progressive_models(
        symbol='INTC', 
        model_types=['transformer']
    )
    
    print("✅ Training completed!")
    print(f"📊 Result: {result}")
    
except Exception as e:
    print(f"❌ Error during training: {e}")
    import traceback
    traceback.print_exc()