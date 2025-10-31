"""
MarketPulse Core Lifespan Management
Application startup and shutdown lifecycle
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket
from typing import Set, Dict, Any
import json
from app.core.config import (
    logger, SCHEDULER_AVAILABLE, FINANCIAL_MODULES_AVAILABLE,
    get_scheduler, market_streamer
)

# ============================================================
# WebSocket Connection Manager
# ============================================================
class ConnectionManager:
    """Manage WebSocket connections for real-time alerts"""

    def __init__(self):
        self.active_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        """Accept new WebSocket connection"""
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"✅ WebSocket connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection"""
        self.active_connections.discard(websocket)
        logger.info(f"❌ WebSocket disconnected. Total: {len(self.active_connections)}")

    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast message to all connected clients"""
        if not self.active_connections:
            return

        # Convert to JSON
        json_message = json.dumps(message)

        # Send to all connections
        disconnected = set()
        for connection in self.active_connections:
            try:
                await connection.send_text(json_message)
            except Exception as e:
                logger.error(f"Failed to send message: {e}")
                disconnected.add(connection)

        # Clean up disconnected
        for conn in disconnected:
            self.disconnect(conn)

        logger.debug(f"📡 Broadcast to {len(self.active_connections)} clients")

# Global connection manager
manager = ConnectionManager()

# ============================================================
# Application Lifespan
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown"""
    logger.info("=" * 80)
    logger.info("🚀 Starting MarketPulse Financial Intelligence Platform")
    logger.info("=" * 80)

    # Initialize scheduler
    scheduler = None
    if SCHEDULER_AVAILABLE:
        try:
            scheduler = get_scheduler()
            # Set WebSocket broadcast callback
            scheduler.websocket_broadcast_callback = manager.broadcast
            logger.info("✅ Real scheduler initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize real scheduler: {e}")
            scheduler = None
    else:
        logger.warning("⚠️ Scheduler not available - running without background tasks")

    # Initialize financial streaming if available
    if FINANCIAL_MODULES_AVAILABLE and market_streamer:
        try:
            # Market data streaming will be handled by WebSocket connections
            logger.info("✅ Market data streamer ready (WebSocket-based)")
        except Exception as e:
            logger.error(f"Failed to initialize market streaming: {e}")

    # Start background scheduler if available
    if scheduler:
        try:
            scheduler.start()
            logger.info("✅ Background scheduler started")
        except Exception as e:
            logger.error(f"Failed to start scheduler: {e}")

    logger.info("✅ All systems operational!")
    logger.info("=" * 80)

    yield

    # Shutdown
    logger.info("=" * 80)
    logger.info("👋 Shutting down MarketPulse...")
    if scheduler:
        try:
            scheduler.stop()
            logger.info("✅ Scheduler stopped")
        except Exception as e:
            logger.error(f"Error stopping scheduler: {e}")
    logger.info("=" * 80)