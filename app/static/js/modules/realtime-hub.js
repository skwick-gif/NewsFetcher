'use strict';

(function bootstrapRealtimeHub(global) {
    if (!global) {
        return;
    }
    const signalR = global.signalR;
    if (!signalR || typeof signalR.HubConnectionBuilder !== 'function') {
        console.warn('⚠️ SignalR client library not detected; real-time hub disabled');
        return;
    }

    class RealtimeHubClient {
        constructor(options = {}) {
            const reconnectDelays = Array.isArray(options.reconnectDelays)
                ? options.reconnectDelays
                : [0, 2000, 5000, 10000, 30000];

            this._options = {
                url: options.url || '/hubs/marketdata',
                reconnectDelays,
            };

            this._connection = null;
            this._connectPromise = null;
            this._subscriptionLock = Promise.resolve();

            this._quoteHandlers = new Set();
            this._candleHandlers = new Set();
            this._statusHandlers = new Set();

            this._desiredSymbol = null;
            this._desiredTimeframes = new Set();
            this._activeSymbol = null;
            this._activeTimeframes = new Set();
        }

        onQuote(handler) {
            if (typeof handler !== 'function') {
                return () => undefined;
            }
            this._quoteHandlers.add(handler);
            return () => this._quoteHandlers.delete(handler);
        }

        onCandle(handler) {
            if (typeof handler !== 'function') {
                return () => undefined;
            }
            this._candleHandlers.add(handler);
            return () => this._candleHandlers.delete(handler);
        }

        onStatus(handler) {
            if (typeof handler !== 'function') {
                return () => undefined;
            }
            this._statusHandlers.add(handler);
            return () => this._statusHandlers.delete(handler);
        }

        async connect() {
            if (!this._connection) {
                this._buildConnection();
            }

            if (!this._connection) {
                throw new Error('SignalR connection unavailable');
            }

            if (this._connection.state === 'Connected') {
                return;
            }

            if (this._connectPromise) {
                return this._connectPromise;
            }

            this._connectPromise = this._startConnection();
            try {
                await this._connectPromise;
            } finally {
                this._connectPromise = null;
            }
        }

        async setActive(symbol, timeframes = []) {
            const normalizedSymbol = (symbol || '').trim().toUpperCase();
            if (!normalizedSymbol) {
                return;
            }

            const normalizedFrames = new Set(
                (Array.isArray(timeframes) ? timeframes : [timeframes])
                    .map((tf) => (tf || '').toString().trim().toLowerCase())
                    .filter(Boolean)
            );

            if (normalizedFrames.size === 0) {
                normalizedFrames.add('1m');
            }

            this._desiredSymbol = normalizedSymbol;
            this._desiredTimeframes = normalizedFrames;

            await this._applyDesiredSubscriptions();
        }

        async disconnect() {
            this._desiredSymbol = null;
            this._desiredTimeframes.clear();

            await this._applyDesiredSubscriptions();

            if (this._connection && this._connection.state !== 'Disconnected') {
                try {
                    await this._connection.stop();
                } catch (err) {
                    console.warn('⚠️ Error stopping SignalR connection', err);
                }
            }

            this._emitStatus('disconnected');
        }

        _buildConnection() {
            try {
                this._connection = new signalR.HubConnectionBuilder()
                    .withUrl(this._options.url)
                    .withAutomaticReconnect(this._options.reconnectDelays)
                    .build();
            } catch (err) {
                console.error('❌ Failed to build SignalR connection', err);
                this._connection = null;
                throw err;
            }

            this._connection.serverTimeoutInMilliseconds = 60000;
            this._connection.keepAliveIntervalInMilliseconds = 15000;

            this._connection.on('StockQuote', (quote) => this._emitQuote(quote));
            this._connection.on('CandleUpdate', (candle) => this._emitCandle(candle));

            this._connection.onreconnecting((error) => {
                this._emitStatus('reconnecting', error);
            });

            this._connection.onreconnected(() => {
                this._emitStatus('connected');
                this._activeSymbol = null;
                this._activeTimeframes.clear();
                this._applyDesiredSubscriptions().catch((err) => {
                    console.warn('⚠️ Failed to resubscribe after reconnect', err);
                });
            });

            this._connection.onclose((error) => {
                this._emitStatus('disconnected', error);
            });
        }

        async _startConnection() {
            if (!this._connection) {
                return;
            }
            this._emitStatus('connecting');
            try {
                await this._connection.start();
                this._emitStatus('connected');
            } catch (err) {
                this._emitStatus('error', err);
                throw err;
            }
        }

        async _applyDesiredSubscriptions() {
            this._subscriptionLock = this._subscriptionLock
                .catch(() => undefined)
                .then(() => this._applyDesiredSubscriptionsInternal());
            return this._subscriptionLock;
        }

        async _applyDesiredSubscriptionsInternal() {
            if (!this._desiredSymbol) {
                await this._unsubscribeAll();
                return;
            }

            await this.connect();

            if (!this._connection || this._connection.state !== 'Connected') {
                return;
            }

            if (this._activeSymbol && this._activeSymbol !== this._desiredSymbol) {
                await this._unsubscribeAll();
            }

            if (!this._activeSymbol) {
                try {
                    await this._connection.invoke('SubscribeToSymbol', this._desiredSymbol);
                    this._activeSymbol = this._desiredSymbol;
                } catch (err) {
                    console.warn('⚠️ Failed to subscribe to symbol', this._desiredSymbol, err);
                    throw err;
                }
            }

            for (const tf of this._desiredTimeframes) {
                if (this._activeTimeframes.has(tf)) {
                    continue;
                }
                try {
                    await this._connection.invoke('SubscribeToBars', this._activeSymbol, tf);
                    this._activeTimeframes.add(tf);
                } catch (err) {
                    console.warn('⚠️ Failed to subscribe to timeframe', tf, err);
                }
            }

            for (const tf of Array.from(this._activeTimeframes)) {
                if (this._desiredTimeframes.has(tf)) {
                    continue;
                }
                try {
                    await this._connection.invoke('UnsubscribeFromBars', this._activeSymbol, tf);
                } catch (err) {
                    console.warn('⚠️ Failed to unsubscribe timeframe', tf, err);
                }
                this._activeTimeframes.delete(tf);
            }
        }

        async _unsubscribeAll() {
            if (!this._connection || this._connection.state !== 'Connected' || !this._activeSymbol) {
                this._activeSymbol = null;
                this._activeTimeframes.clear();
                return;
            }

            for (const tf of Array.from(this._activeTimeframes)) {
                try {
                    await this._connection.invoke('UnsubscribeFromBars', this._activeSymbol, tf);
                } catch (err) {
                    console.warn('⚠️ Error unsubscribing timeframe', tf, err);
                }
            }
            this._activeTimeframes.clear();

            try {
                await this._connection.invoke('UnsubscribeFromSymbol', this._activeSymbol);
            } catch (err) {
                console.warn('⚠️ Error unsubscribing symbol', this._activeSymbol, err);
            }

            this._activeSymbol = null;
        }

        _emitQuote(quote) {
            for (const handler of this._quoteHandlers) {
                try {
                    handler(quote);
                } catch (err) {
                    console.error('❌ Quote handler error', err);
                }
            }
        }

        _emitCandle(candle) {
            for (const handler of this._candleHandlers) {
                try {
                    handler(candle);
                } catch (err) {
                    console.error('❌ Candle handler error', err);
                }
            }
        }

        _emitStatus(state, error) {
            const payload = error
                ? { state, error, message: error?.message }
                : { state };
            for (const handler of this._statusHandlers) {
                try {
                    handler(payload);
                } catch (err) {
                    console.error('❌ Status handler error', err);
                }
            }
        }
    }

    global.RealtimeHubClient = RealtimeHubClient;
})(typeof window !== 'undefined' ? window : undefined);
