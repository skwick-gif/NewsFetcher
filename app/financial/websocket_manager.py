"""
WebSocket Manager for Real-time Market Data Streaming
Provides live updates to connected clients
"""

import asyncio
import json
import logging
from typing import Dict, List, Set, Any, Optional
from datetime import datetime
import websockets
from websockets.server import WebSocketServerProtocol
from fastapi import WebSocket, WebSocketDisconnect
import uuid

logger = logging.getLogger(__name__)

class WebSocketManager:
    """Manages WebSocket connections for real-time updates"""
    
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.symbol_subscriptions: Dict[str, Set[WebSocket]] = {}
        self.connection_metadata: Dict[WebSocket, Dict] = {}
        self.message_queues: Dict[WebSocket, asyncio.Queue] = {}
        
    async def connect(self, websocket: WebSocket):
        """Accept new WebSocket connection"""
        try:
            await websocket.accept()
            self.active_connections.add(websocket)
            
            # Generate connection ID
            connection_id = str(uuid.uuid4())
            
            # Store connection metadata
            self.connection_metadata[websocket] = {
                'id': connection_id,
                'connected_at': datetime.now(),
                'subscriptions': set(),
                'message_count': 0
            }
            
            # Create message queue for this connection
            self.message_queues[websocket] = asyncio.Queue()
            
            # Send welcome message
            await self.send_personal_message({
                'type': 'connection_established',
                'connection_id': connection_id,
                'timestamp': datetime.now().isoformat(),
                'message': 'Connected to MarketPulse AI WebSocket'
            }, websocket)
            
            logger.info(f"✅ WebSocket connection established: {connection_id}")
            
        except Exception as e:
            logger.error(f"Error establishing WebSocket connection: {e}")
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
    
    async def disconnect(self, websocket: WebSocket):
        """Handle WebSocket disconnection"""
        try:
            # Remove from active connections
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
            
            # Remove from symbol subscriptions
            for symbol_connections in self.symbol_subscriptions.values():
                symbol_connections.discard(websocket)
            
            # Clean up metadata
            connection_info = self.connection_metadata.pop(websocket, {})
            
            # Clean up message queue
            if websocket in self.message_queues:
                del self.message_queues[websocket]
            
            logger.info(f"🔌 WebSocket disconnected: {connection_info.get('id', 'unknown')}")
            
        except Exception as e:
            logger.error(f"Error handling WebSocket disconnection: {e}")
    
    async def send_personal_message(self, message: Dict[str, Any], websocket: WebSocket):
        """Send message to specific WebSocket connection"""
        try:
            if websocket in self.active_connections:
                await websocket.send_text(json.dumps(message))
                
                # Update message count
                if websocket in self.connection_metadata:
                    self.connection_metadata[websocket]['message_count'] += 1
                    
        except WebSocketDisconnect:
            await self.disconnect(websocket)
        except Exception as e:
            logger.error(f"Error sending personal message: {e}")
            await self.disconnect(websocket)
    
    async def broadcast_message(self, message: Dict[str, Any]):
        """Broadcast message to all connected clients"""
        if not self.active_connections:
            return
        
        message['timestamp'] = datetime.now().isoformat()
        disconnected = set()
        
        for websocket in self.active_connections.copy():
            try:
                await websocket.send_text(json.dumps(message))
                
                # Update message count
                if websocket in self.connection_metadata:
                    self.connection_metadata[websocket]['message_count'] += 1
                    
            except WebSocketDisconnect:
                disconnected.add(websocket)
            except Exception as e:
                logger.error(f"Error broadcasting to connection: {e}")
                disconnected.add(websocket)
        
        # Clean up disconnected websockets
        for websocket in disconnected:
            await self.disconnect(websocket)
        
        logger.debug(f"📡 Broadcasted message to {len(self.active_connections)} connections")
    
    async def subscribe_to_symbol(self, websocket: WebSocket, symbol: str):
        """Subscribe WebSocket to specific symbol updates"""
        try:
            if symbol not in self.symbol_subscriptions:
                self.symbol_subscriptions[symbol] = set()
            
            self.symbol_subscriptions[symbol].add(websocket)
            
            # Update connection metadata
            if websocket in self.connection_metadata:
                self.connection_metadata[websocket]['subscriptions'].add(symbol)
            
            await self.send_personal_message({
                'type': 'subscription_confirmed',
                'symbol': symbol,
                'message': f'Subscribed to {symbol} updates'
            }, websocket)
            
            logger.info(f"📊 WebSocket subscribed to {symbol}")
            
        except Exception as e:
            logger.error(f"Error subscribing to symbol {symbol}: {e}")
    
    async def unsubscribe_from_symbol(self, websocket: WebSocket, symbol: str):
        """Unsubscribe WebSocket from symbol updates"""
        try:
            if symbol in self.symbol_subscriptions:
                self.symbol_subscriptions[symbol].discard(websocket)
                
                # Clean up empty subscriptions
                if not self.symbol_subscriptions[symbol]:
                    del self.symbol_subscriptions[symbol]
            
            # Update connection metadata
            if websocket in self.connection_metadata:
                self.connection_metadata[websocket]['subscriptions'].discard(symbol)
            
            await self.send_personal_message({
                'type': 'unsubscription_confirmed',
                'symbol': symbol,
                'message': f'Unsubscribed from {symbol} updates'
            }, websocket)
            
            logger.info(f"📊 WebSocket unsubscribed from {symbol}")
            
        except Exception as e:
            logger.error(f"Error unsubscribing from symbol {symbol}: {e}")
    
    async def send_symbol_update(self, symbol: str, data: Dict[str, Any]):
        """Send update to all subscribers of a specific symbol"""
        if symbol not in self.symbol_subscriptions:
            return
        
        message = {
            'type': 'symbol_update',
            'symbol': symbol,
            'data': data,
            'timestamp': datetime.now().isoformat()
        }
        
        disconnected = set()
        subscribers = self.symbol_subscriptions[symbol].copy()
        
        for websocket in subscribers:
            try:
                await websocket.send_text(json.dumps(message))
                
                # Update message count
                if websocket in self.connection_metadata:
                    self.connection_metadata[websocket]['message_count'] += 1
                    
            except WebSocketDisconnect:
                disconnected.add(websocket)
            except Exception as e:
                logger.error(f"Error sending symbol update: {e}")
                disconnected.add(websocket)
        
        # Clean up disconnected websockets
        for websocket in disconnected:
            await self.disconnect(websocket)
        
        logger.debug(f"📈 Sent {symbol} update to {len(subscribers) - len(disconnected)} subscribers")
    
    async def send_ai_analysis_update(self, symbol: str, analysis: Dict[str, Any]):
        """Send AI analysis update to symbol subscribers"""
        if symbol not in self.symbol_subscriptions:
            return
        
        message = {
            'type': 'ai_analysis_update',
            'symbol': symbol,
            'analysis': analysis,
            'timestamp': datetime.now().isoformat()
        }
        
        disconnected = set()
        subscribers = self.symbol_subscriptions[symbol].copy()
        
        for websocket in subscribers:
            try:
                await websocket.send_text(json.dumps(message))
                
                if websocket in self.connection_metadata:
                    self.connection_metadata[websocket]['message_count'] += 1
                    
            except WebSocketDisconnect:
                disconnected.add(websocket)
            except Exception as e:
                logger.error(f"Error sending AI analysis update: {e}")
                disconnected.add(websocket)
        
        # Clean up disconnected websockets
        for websocket in disconnected:
            await self.disconnect(websocket)
        
        logger.debug(f"🤖 Sent AI analysis for {symbol} to {len(subscribers) - len(disconnected)} subscribers")
    
    async def handle_client_message(self, websocket: WebSocket, message: str):
        """Handle incoming messages from WebSocket clients"""
        try:
            data = json.loads(message)
            message_type = data.get('type')
            
            if message_type == 'subscribe':
                symbol = data.get('symbol', '').upper()
                if symbol:
                    await self.subscribe_to_symbol(websocket, symbol)
                else:
                    await self.send_personal_message({
                        'type': 'error',
                        'message': 'Symbol is required for subscription'
                    }, websocket)
            
            elif message_type == 'unsubscribe':
                symbol = data.get('symbol', '').upper()
                if symbol:
                    await self.unsubscribe_from_symbol(websocket, symbol)
                else:
                    await self.send_personal_message({
                        'type': 'error',
                        'message': 'Symbol is required for unsubscription'
                    }, websocket)
            
            elif message_type == 'ping':
                await self.send_personal_message({
                    'type': 'pong',
                    'timestamp': datetime.now().isoformat()
                }, websocket)
            
            elif message_type == 'get_subscriptions':
                metadata = self.connection_metadata.get(websocket, {})
                await self.send_personal_message({
                    'type': 'subscriptions_list',
                    'subscriptions': list(metadata.get('subscriptions', set())),
                    'connection_id': metadata.get('id'),
                    'connected_since': metadata.get('connected_at', datetime.now()).isoformat(),
                    'messages_received': metadata.get('message_count', 0)
                }, websocket)
            
            else:
                await self.send_personal_message({
                    'type': 'error',
                    'message': f'Unknown message type: {message_type}'
                }, websocket)
                
        except json.JSONDecodeError:
            await self.send_personal_message({
                'type': 'error',
                'message': 'Invalid JSON format'
            }, websocket)
        except Exception as e:
            logger.error(f"Error handling client message: {e}")
            await self.send_personal_message({
                'type': 'error',
                'message': 'Internal server error'
            }, websocket)
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get WebSocket connection statistics"""
        total_subscriptions = sum(len(subs) for subs in self.symbol_subscriptions.values())
        
        return {
            'active_connections': len(self.active_connections),
            'unique_symbol_subscriptions': len(self.symbol_subscriptions),
            'total_subscriptions': total_subscriptions,
            'symbols_tracked': list(self.symbol_subscriptions.keys()),
            'average_subscriptions_per_connection': (
                total_subscriptions / len(self.active_connections)
                if self.active_connections else 0
            ),
            'connection_details': [
                {
                    'id': metadata.get('id'),
                    'connected_at': metadata.get('connected_at', datetime.now()).isoformat(),
                    'subscriptions': list(metadata.get('subscriptions', set())),
                    'message_count': metadata.get('message_count', 0)
                }
                for metadata in self.connection_metadata.values()
            ]
        }

class MarketDataStreamer:
    """Real-time market data streaming service"""
    
    def __init__(self, websocket_manager: WebSocketManager):
        self.websocket_manager = websocket_manager
        self.is_streaming = False
        self.stream_tasks: Dict[str, asyncio.Task] = {}
        
    async def start_symbol_stream(self, symbol: str, interval: int = 10):
        """Start streaming data for a specific symbol"""
        if symbol in self.stream_tasks:
            logger.info(f"📊 Stream already active for {symbol}")
            return
        
        task = asyncio.create_task(self._stream_symbol_data(symbol, interval))
        self.stream_tasks[symbol] = task
        
        logger.info(f"▶️ Started streaming {symbol} (interval: {interval}s)")
    
    async def stop_symbol_stream(self, symbol: str):
        """Stop streaming data for a specific symbol"""
        if symbol in self.stream_tasks:
            self.stream_tasks[symbol].cancel()
            del self.stream_tasks[symbol]
            logger.info(f"⏹️ Stopped streaming {symbol}")
    
    async def _stream_symbol_data(self, symbol: str, interval: int):
        """Stream data for a symbol at regular intervals"""
        try:
            from app.financial.market_data import FinancialDataProvider
            from app.financial.ai_models import AdvancedAIModels
            
            financial_provider = FinancialDataProvider()
            ai_models = AdvancedAIModels()
            
            while True:
                try:
                    # Get current market data
                    price_data = await financial_provider.get_live_price(symbol)
                    
                    if price_data:
                        # Send price update
                        await self.websocket_manager.send_symbol_update(symbol, {
                            'type': 'price_data',
                            'price': price_data['price'],
                            'change': price_data.get('change', 0),
                            'change_percent': price_data.get('change_percent', 0),
                            'volume': price_data.get('volume', 0),
                            'market_open': price_data.get('market_open', True)
                        })
                        
                        # Get AI analysis (less frequently)
                        if hash(symbol) % 6 == 0:  # Every 6th update
                            try:
                                # Get recent data for analysis
                                recent_data = await financial_provider.get_stock_data(symbol, days=7)
                                
                                if recent_data:
                                    features = await ai_models.extract_features(symbol, recent_data)
                                    analysis = await ai_models.get_comprehensive_analysis(symbol, features)
                                    
                                    await self.websocket_manager.send_ai_analysis_update(symbol, analysis)
                                    
                            except Exception as e:
                                logger.warning(f"Error getting AI analysis for {symbol}: {e}")
                    
                    await asyncio.sleep(interval)
                    
                except Exception as e:
                    logger.error(f"Error in symbol stream for {symbol}: {e}")
                    await asyncio.sleep(interval * 2)  # Wait longer on error
                    
        except asyncio.CancelledError:
            logger.info(f"📊 Stream cancelled for {symbol}")
        except Exception as e:
            logger.error(f"Fatal error in symbol stream for {symbol}: {e}")
    
    async def start_market_overview_stream(self, interval: int = 30):
        """Start streaming market overview data"""
        task = asyncio.create_task(self._stream_market_overview(interval))
        self.stream_tasks['market_overview'] = task
        logger.info(f"▶️ Started market overview stream (interval: {interval}s)")
    
    async def _stream_market_overview(self, interval: int):
        """Stream market overview at regular intervals"""
        try:
            while True:
                try:
                    # Get market indices
                    from app.financial.market_data import FinancialDataProvider
                    financial_provider = FinancialDataProvider()
                    
                    indices = await financial_provider.get_market_indices()
                    
                    # Broadcast market overview
                    await self.websocket_manager.broadcast_message({
                        'type': 'market_overview',
                        'indices': indices,
                        'timestamp': datetime.now().isoformat()
                    })
                    
                    await asyncio.sleep(interval)
                    
                except Exception as e:
                    logger.error(f"Error in market overview stream: {e}")
                    await asyncio.sleep(interval * 2)
                    
        except asyncio.CancelledError:
            logger.info("📊 Market overview stream cancelled")
        except Exception as e:
            logger.error(f"Fatal error in market overview stream: {e}")
    
    def get_stream_status(self) -> Dict[str, Any]:
        """Get status of all active streams"""
        return {
            'active_streams': len(self.stream_tasks),
            'streaming_symbols': [symbol for symbol in self.stream_tasks.keys() if symbol != 'market_overview'],
            'market_overview_active': 'market_overview' in self.stream_tasks,
            'total_connections': len(self.websocket_manager.active_connections)
        }

# Global WebSocket manager instance
websocket_manager = WebSocketManager()
market_streamer = MarketDataStreamer(websocket_manager)