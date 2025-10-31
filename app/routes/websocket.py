from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.core.config import market_streamer
from app.core.lifespan import manager
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

@router.websocket("/ws/alerts")
async def websocket_alerts(websocket: WebSocket):
    """
    WebSocket endpoint for real-time alerts

    Clients connect here to receive:
    - Critical alerts (WhatsApp-level)
    - Important alerts
    - Watch alerts
    - Real-time market updates
    """
    await manager.connect(websocket)

    try:
        # Send welcome message
        from datetime import datetime, timezone
        await websocket.send_json({
            "type": "connected",
            "message": "✅ Connected to MarketPulse alerts",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

        # Keep connection alive and handle incoming messages
        while True:
            # Receive messages from client (e.g., preferences, ping)
            data = await websocket.receive_text()

            # Echo back (or handle commands)
            try:
                import json
                client_msg = json.loads(data)

                if client_msg.get("type") == "ping":
                    await websocket.send_json({
                        "type": "pong",
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })

            except json.JSONDecodeError:
                pass

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("Client disconnected normally")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

@router.websocket("/ws/market/{symbol}")
async def websocket_market_data(websocket: WebSocket, symbol: str):
    """WebSocket endpoint for real-time market data streaming"""
    await websocket.accept()
    logger.info(f"🔌 Market WebSocket connected for {symbol}")

    try:
        if market_streamer:
            # Add this WebSocket to market streaming
            await market_streamer.add_subscription(websocket, symbol)

        # Keep connection alive and handle messages
        while True:
            try:
                # Wait for client messages (ping/pong, etc.)
                message = await websocket.receive_text()

                # Echo back or handle specific commands
                if message == "ping":
                    await websocket.send_text("pong")

            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"WebSocket error for {symbol}: {e}")
                break

    except WebSocketDisconnect:
        logger.info(f"🔌 Market WebSocket disconnected for {symbol}")
    except Exception as e:
        logger.error(f"Market WebSocket error for {symbol}: {e}")
    finally:
        if market_streamer:
            await market_streamer.remove_subscription(websocket, symbol)