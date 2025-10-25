/**
 * WebSocket Client Module
 * Handles real-time updates via WebSocket connections
 */

class WebSocketClient {
    constructor() {
        this.alertsSocket = null;
        this.marketSocket = null;
    this.aiAnalysisSocket = null; // AI analysis WS disabled until server endpoint exists
        this.isConnected = false;
        this.currentSymbol = 'AAPL';
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 3000; // 3 seconds
    }

    /**
     * Initialize WebSocket client
     */
    init() {
        // WebSockets are optional, don't connect automatically
        console.log('✅ WebSocket Client initialized (not connected)');
    }

    /**
     * Connect to WebSocket servers
     */
    connect(symbol = 'AAPL') {
        if (this.isConnected) {
            console.log('⚠️ Already connected to WebSocket');
            return;
        }

        this.currentSymbol = symbol;
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const host = window.location.host;

        try {
            // Alerts WebSocket
            this.alertsSocket = new WebSocket(`${protocol}//${host}/ws/alerts`);
            this.setupAlertsSocketHandlers();

            // Market Data WebSocket
            this.marketSocket = new WebSocket(`${protocol}//${host}/ws/market/${this.currentSymbol}`);
            this.setupMarketSocketHandlers();

            // AI Analysis WebSocket (disabled - endpoint not implemented server-side)
            // this.aiAnalysisSocket = new WebSocket(`${protocol}//${host}/ws/ai-analysis/${this.currentSymbol}`);
            // this.setupAIAnalysisSocketHandlers();

            this.isConnected = true;
            this.reconnectAttempts = 0;

            console.log('✅ WebSocket connections established');
            this.showNotification('Real-time streaming enabled! 🔴', 'success');
        } catch (error) {
            console.error('❌ Error connecting to WebSocket:', error);
            this.showNotification('Failed to enable real-time updates', 'error');
        }
    }

    /**
     * Setup alerts WebSocket event handlers
     */
    setupAlertsSocketHandlers() {
        if (!this.alertsSocket) return;

        this.alertsSocket.onopen = () => {
            console.log('✅ Alerts WebSocket connected');
        };

        this.alertsSocket.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                this.handleAlertsMessage(data);
            } catch (error) {
                console.error('Error parsing alerts message:', error);
            }
        };

        this.alertsSocket.onclose = () => {
            console.log('⚠️ Alerts WebSocket disconnected');
            this.handleDisconnect();
        };

        this.alertsSocket.onerror = (error) => {
            console.error('❌ Alerts WebSocket error:', error);
        };
    }

    /**
     * Setup market data WebSocket event handlers
     */
    setupMarketSocketHandlers() {
        if (!this.marketSocket) return;

        this.marketSocket.onopen = () => {
            console.log('✅ Market WebSocket connected');
        };

        this.marketSocket.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                this.handleMarketMessage(data);
            } catch (error) {
                console.error('Error parsing market message:', error);
            }
        };

        this.marketSocket.onclose = () => {
            console.log('⚠️ Market WebSocket disconnected');
        };

        this.marketSocket.onerror = (error) => {
            console.error('❌ Market WebSocket error:', error);
        };
    }

    /**
     * Setup AI analysis WebSocket event handlers
     */
    setupAIAnalysisSocketHandlers() {
        if (!this.aiAnalysisSocket) return;

        this.aiAnalysisSocket.onopen = () => {
            console.log('✅ AI Analysis WebSocket connected');
        };

        this.aiAnalysisSocket.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                this.handleAIAnalysisMessage(data);
            } catch (error) {
                console.error('Error parsing AI analysis message:', error);
            }
        };

        this.aiAnalysisSocket.onclose = () => {
            console.log('⚠️ AI Analysis WebSocket disconnected');
        };

        this.aiAnalysisSocket.onerror = (error) => {
            console.error('❌ AI Analysis WebSocket error:', error);
        };
    }

    /**
     * Handle alerts WebSocket message
     */
    handleAlertsMessage(data) {
        console.log('📢 New alert received:', data);

        if (data.type === 'new_alert') {
            // Refresh alerts display if AlertsManager is available
            if (window.alertsManager) {
                window.alertsManager.loadAlerts();
            }

            // Show notification
            this.showNotification(`New Alert: ${data.title || 'Check alerts tab'}`, 'warning');
        }
    }

    /**
     * Handle market data WebSocket message
     */
    handleMarketMessage(data) {
        console.log('📊 Market update received:', data);

        if (data.symbol === this.currentSymbol && data.market_data) {
            // Update market sentiment if available
            if (data.market_data.sentiment) {
                const sentimentElement = document.getElementById('market-sentiment-value');
                if (sentimentElement) {
                    sentimentElement.textContent = data.market_data.sentiment || 'Neutral';
                }
            }

            // Update last updated timestamp (harmonized with dashboard id="last-update")
            const lastUpdated = document.getElementById('last-update');
            if (lastUpdated) {
                lastUpdated.textContent = `Last updated: ${new Date().toLocaleTimeString()} (Live)`;
                lastUpdated.style.color = '#4CAF50';
            }

            // Show brief notification
            this.showNotification(`📈 ${data.symbol} data updated`, 'info', 2000);
        }
    }

    /**
     * Handle AI analysis WebSocket message
     */
    handleAIAnalysisMessage(data) {
        console.log('🤖 AI analysis update received:', data);

        if (data.predictions) {
            // Update prediction displays
            if (data.predictions.price && document.getElementById('price-prediction')) {
                document.getElementById('price-prediction').textContent = 
                    `$${data.predictions.price.toFixed(2)}`;
            }

            if (data.predictions.direction && document.getElementById('direction-prediction')) {
                document.getElementById('direction-prediction').textContent = data.predictions.direction;
            }

            // Show notification
            this.showNotification('AI predictions updated', 'info', 2000);
        }
    }

    /**
     * Handle WebSocket disconnection and attempt reconnect
     */
    handleDisconnect() {
        this.isConnected = false;

        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            console.log(`Attempting to reconnect (${this.reconnectAttempts}/${this.maxReconnectAttempts})...`);

            setTimeout(() => {
                this.connect(this.currentSymbol);
            }, this.reconnectDelay);
        } else {
            console.error('Max reconnection attempts reached');
            this.showNotification('Real-time connection lost', 'error');
        }
    }

    /**
     * Disconnect from all WebSocket servers
     */
    disconnect() {
        if (this.alertsSocket) {
            this.alertsSocket.close();
            this.alertsSocket = null;
        }

        if (this.marketSocket) {
            this.marketSocket.close();
            this.marketSocket = null;
        }

        // AI Analysis WebSocket is disabled; nothing to close

        this.isConnected = false;
        this.reconnectAttempts = 0;

        console.log('✅ WebSocket connections closed');
        this.showNotification('Real-time streaming disabled', 'info');
    }

    /**
     * Toggle WebSocket connection on/off
     */
    toggle() {
        if (this.isConnected) {
            this.disconnect();
            return false;
        } else {
            this.connect(this.currentSymbol);
            return true;
        }
    }

    /**
     * Change the symbol for market and AI analysis WebSockets
     */
    changeSymbol(symbol) {
        if (this.currentSymbol === symbol) return;

        this.currentSymbol = symbol;

        // Reconnect market and AI sockets if connected
        if (this.isConnected) {
            // Close existing sockets
            if (this.marketSocket) {
                this.marketSocket.close();
            }
            // AI Analysis WebSocket disabled; no reconnect for this socket

            // Reconnect with new symbol
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const host = window.location.host;

            this.marketSocket = new WebSocket(`${protocol}//${host}/ws/market/${symbol}`);
            this.setupMarketSocketHandlers();

            // AI Analysis WebSocket disabled; skip

            console.log(`✅ WebSocket connections updated for ${symbol}`);
        }
    }

    /**
     * Show notification to user
     */
    showNotification(message, type = 'info', duration = 3000) {
        // Try to find existing notification container
        let notification = document.getElementById('ws-notification');

        if (!notification) {
            notification = document.createElement('div');
            notification.id = 'ws-notification';
            notification.style.cssText = `
                position: fixed;
                bottom: 20px;
                right: 20px;
                padding: 12px 20px;
                border-radius: 8px;
                font-size: 0.9em;
                z-index: 10000;
                animation: fadeIn 0.3s;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
            `;
            document.body.appendChild(notification);
        }

        // Set color based on type
        const colors = {
            success: 'rgba(76, 175, 80, 0.9)',
            error: 'rgba(244, 67, 54, 0.9)',
            warning: 'rgba(255, 193, 7, 0.9)',
            info: 'rgba(33, 150, 243, 0.9)'
        };

        notification.style.background = colors[type] || colors.info;
        notification.style.color = 'white';
        notification.textContent = message;
        notification.style.display = 'block';

        // Auto-hide after duration
        setTimeout(() => {
            if (notification && notification.parentNode) {
                notification.style.display = 'none';
            }
        }, duration);
    }

    /**
     * Get connection status
     */
    getStatus() {
        return {
            connected: this.isConnected,
            alerts: this.alertsSocket?.readyState === WebSocket.OPEN,
            market: this.marketSocket?.readyState === WebSocket.OPEN,
            aiAnalysis: false,
            currentSymbol: this.currentSymbol,
            reconnectAttempts: this.reconnectAttempts
        };
    }
}

// Export for use in main.js
window.WebSocketClient = WebSocketClient;