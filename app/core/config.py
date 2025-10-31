"""
MarketPulse Core Configuration
Centralized imports and global component initialization
"""
import logging
import os
import sys
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Set, Optional
import asyncio

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks, Query, Depends, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from contextlib import asynccontextmanager
import json
import requests
import numpy as np
from pathlib import Path
import yaml
from pathlib import Path
import subprocess
import threading

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set up logging FIRST
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Safety: disable torch dynamo/compile in this process to avoid environment-induced issues
# Some environments auto-enable Dynamo which may import torch._C._dynamo.eval_frame.skip_code
# Our code does not use torch.compile; explicitly disable to be safe.
os.environ.setdefault("PYTORCH_ENABLE_DYNAMO", "0")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

# Import our components
# Scheduler functionality
try:
    from app.scheduler_bg.scheduler import get_scheduler, MarketPulseScheduler
    SCHEDULER_AVAILABLE = True
    logger.info("✅ Real scheduler loaded successfully")
except ImportError as e:
    SCHEDULER_AVAILABLE = False
    logger.warning(f"⚠️ Real scheduler not available: {e}")

# Core components imports
try:
    from app.smart.keywords_engine import FinancialKeywordsEngine
except ImportError:
    print("⚠️ Keywords engine not available")
from app.ingest.rss_loader import FinancialDataLoader
from app.integrations.ibkr_client import ibkr

# Financial modules imports
try:
    from app.financial.market_data import FinancialDataProvider
    from app.financial.market_data_clean import FinancialDataProvider as CleanFinancialDataProvider
    from app.financial.news_impact import NewsImpactAnalyzer
    from app.financial.social_sentiment import SocialMediaAnalyzer
    from app.financial.social_sentiment_enhanced import RealSocialMediaAnalyzer
    from app.financial.ai_models import AdvancedAIModels, TimeSeriesAnalyzer
    # from app.financial.neural_networks import EnsembleNeuralNetwork  # DISABLED - OLD TENSORFLOW
    # from app.financial.ml_trainer import MLModelTrainer  # DISABLED - TensorFlow compatibility issues
    ML_TRAINER_AVAILABLE = False
    from app.financial.websocket_manager import WebSocketManager, MarketDataStreamer

    FINANCIAL_MODULES_AVAILABLE = True
    logger.info("✅ Financial modules loaded successfully")
except ImportError as e:
    FINANCIAL_MODULES_AVAILABLE = False
    ML_TRAINER_AVAILABLE = False
    logger.warning(f"⚠️ Financial modules not available: {e}")
    logger.info("📊 Running in demo mode with limited functionality")

# Progressive ML imports
try:
    from app.ml.progressive.data_loader import ProgressiveDataLoader
    from app.ml.progressive.trainer import ProgressiveTrainer
    from app.ml.progressive.predictor import ProgressivePredictor
    from app.ml.progressive.models import ProgressiveModels
    from app.data.data_manager import DataManager

    PROGRESSIVE_ML_AVAILABLE = True
    logger.info("✅ Progressive ML system loaded successfully")
except (ImportError, KeyboardInterrupt, Exception) as e:
    PROGRESSIVE_ML_AVAILABLE = False
    logger.warning(f"⚠️ Progressive ML system not available: {e}")

# News sentiment provider (wraps real providers: NewsAPI, Yahoo, Alpha Vantage, Bing)
try:
    from app.financial.news_sentiment_provider import NewsSentimentProvider
    NEWS_SENTIMENT_AVAILABLE = True
except ImportError as e:
    NEWS_SENTIMENT_AVAILABLE = False
    logger.warning(f"⚠️ News sentiment provider unavailable: {e}")

# Initialize templates
try:
    templates = Jinja2Templates(directory="templates")
    TEMPLATES_AVAILABLE = True
except Exception as e:
    logger.warning(f"Templates not available: {e}")
    TEMPLATES_AVAILABLE = False

# Global components initialization
financial_provider = None
news_impact_analyzer = None
social_analyzer = None
ai_models = None
ml_trainer = None
websocket_manager = None
market_streamer = None
keyword_analyzer = None
real_data_loader = None
news_sentiment_provider = None

# Progressive ML instances
progressive_data_loader = None
progressive_trainer = None
progressive_predictor = None
data_manager = None

# Initialize financial components if available
if FINANCIAL_MODULES_AVAILABLE:
    try:
        # Initialize providers
        financial_provider = FinancialDataProvider()
        news_impact_analyzer = NewsImpactAnalyzer()
        social_analyzer = RealSocialMediaAnalyzer()
        ai_models = AdvancedAIModels()
        if ML_TRAINER_AVAILABLE:
            ml_trainer = MLModelTrainer()
        else:
            ml_trainer = None
        websocket_manager = WebSocketManager()
        market_streamer = MarketDataStreamer(websocket_manager)

        logger.info("✅ Financial components initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize financial components: {e}")

# Initialize RSS and keyword systems
try:
    # Use correct config path
    config_path = os.path.join(os.path.dirname(__file__), "config", "data_sources.yaml")
    if os.path.exists(config_path):
        real_data_loader = FinancialDataLoader(config_path)

        # Load config for keyword analyzer
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        keyword_analyzer = FinancialKeywordsEngine()

        logger.info("✅ RSS and keyword systems initialized successfully")
    else:
        logger.warning(f"Config file not found: {config_path}")
except Exception as e:
    logger.error(f"Failed to initialize RSS/keyword systems: {e}")

# Initialize Progressive ML system
if PROGRESSIVE_ML_AVAILABLE:
    try:
        progressive_data_loader = ProgressiveDataLoader()
        progressive_trainer = ProgressiveTrainer(progressive_data_loader)
        progressive_predictor = ProgressivePredictor(progressive_data_loader)
        data_manager = DataManager()
        logger.info("✅ Progressive ML system initialized successfully")
    except (KeyboardInterrupt, Exception) as e:
        logger.error(f"Failed to initialize Progressive ML system: {e}")
        PROGRESSIVE_ML_AVAILABLE = False

# Initialize News Sentiment Provider
if NEWS_SENTIMENT_AVAILABLE:
    try:
        # 14 days back, cache TTL 10 minutes
        news_sentiment_provider = NewsSentimentProvider(days_back=14, ttl_seconds=600)
        logger.info("✅ News Sentiment provider initialized")
    except Exception as e:
        logger.error(f"Failed to initialize News Sentiment provider: {e}")
        NEWS_SENTIMENT_AVAILABLE = False