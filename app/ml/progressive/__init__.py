"""
Progressive ML Training System - __init__.py
============================================
Initializes the Progressive Training System for multi-horizon stock predictions

This system supports:
- Progressive training (1 day → 7 days → 30 days)
- Unified training (all horizons simultaneously)
- Multiple model architectures (LSTM, Transformer, CNN)
- Ensemble predictions with confidence scoring
"""

__version__ = "1.0.0"
__author__ = "MarketPulse AI Team"

import logging
logger = logging.getLogger(__name__)

from .data_loader import ProgressiveDataLoader
try:
    from .models import ProgressiveModels, UnifiedModel, LSTMModel, TransformerModel, CNNModel, EnsembleModel
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False
    logger.warning("⚠️ PyTorch models not available")
from .trainer import ProgressiveTrainer
from .predictor import ProgressivePredictor

__all__ = [
    "ProgressiveDataLoader",
    "ProgressiveModels", 
    "UnifiedModel",
    "LSTMModel",
    "TransformerModel", 
    "CNNModel",
    "EnsembleModel",
    "ProgressiveTrainer",
    "ProgressivePredictor"
]