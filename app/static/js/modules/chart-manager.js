/**
 * Chart Manager Module
 * Handles all price chart functionality including Chart.js initialization,
 * stock selection, timeframe changes, and real-time data loading.
 */

const TIMEFRAME_CONFIG = {
    // Intraday presets – include outside RTH by default per request
    '1m': { timeframe: '1m', include_outside_rth: true },
    '5m': { timeframe: '5m', include_outside_rth: true },
    '15m': { timeframe: '15m', include_outside_rth: true },
    '30m': { timeframe: '30m', include_outside_rth: true },
    '1h': { timeframe: '1h', include_outside_rth: true },
    '2h': { timeframe: '2h', include_outside_rth: true },
    '4h': { timeframe: '4h', include_outside_rth: true },
    // Legacy broader ranges (still supported)
    '1D': { timeframe: '1D', include_outside_rth: true },
    '1W': { timeframe: '1W', include_outside_rth: true },
    '1M': { timeframe: '1M', include_outside_rth: true },
    '3M': { timeframe: '3M', include_outside_rth: true },
    '6M': { timeframe: '6M', include_outside_rth: true },
    '1Y': { timeframe: '1Y', include_outside_rth: true },
};

const AUTO_REFRESH_SECONDS = {
    '1m': 12,
    '5m': 30,
    '15m': 60,
    '30m': 120,
    '1h': 180,
    '2h': 300,
    '4h': 600,
    // Fallbacks for legacy keys
    '1D': 60,
    '1W': 300,
    '1M': 600,
    '3M': 900,
    '6M': 1200,
    '1Y': 1800,
};

const REALTIME_RENDER_DEBOUNCE_MS = 250;

class ChartManager {
    constructor() {
        this.chart = null;
        this.currentSymbol = 'AAPL';  // Default symbol
        this.currentTimeframe = '5m';  // Default timeframe per request
        this.currentType = 'line';     // 'line' or 'candlestick'
        this._autoTimer = null;
        this.latestBars = [];
        this.realtimeClient = null;
        this._realtimeActive = false;
        this._pendingRealtimeRender = false;
        this._latestQuote = null;
        this._realtimeStatus = 'off';
        this._realtimeSubscriptions = { quote: null, candle: null, status: null };
    }

    /**
     * Initialize the price chart with Chart.js
     */
    async init() {
        const ctx = document.getElementById('priceChart');
        if (!ctx) {
            console.warn('⚠️ Price chart canvas not found');
            return;
        }

        this._createChart(ctx.getContext('2d'));

        // Setup event listeners
        this.setupStockSelector();
        this.setupTimeframeButtons();
        this.setupChartTypeToggle();

        this._initRealtime();

        // Load initial chart data
        await this.loadChartData(this.currentSymbol, this.currentTimeframe);
        this._scheduleAutoRefresh();

        console.log('✅ Chart Manager initialized');
    }

    /**
     * Load chart data from API
     * @param {string} symbol - Stock symbol (e.g., 'AAPL')
     * @param {string} timeframe - Time period ('1D', '1W', '1M', '3M')
     */
    async loadChartData(symbol, timeframe) {
        if (!this.chart) {
            console.warn('⚠️ Chart not initialized yet');
            return;
        }

        await this._ensureRealtimeSubscription(symbol, timeframe);

        // Build a sequence of parameter attempts to maximize compatibility with the bridge
        const makeAttempts = () => {
            const attempts = [];
            const cfg = TIMEFRAME_CONFIG[timeframe] || {};
            const base = { symbol, timeframe: cfg.timeframe || timeframe };

            // Attempt 1: Use explicit config (intraday preferences) if present
            const p1 = new URLSearchParams({ symbol });
            p1.set('timeframe', base.timeframe);
            // Always include outside RTH to cover pre/post market as requested
            p1.set('include_outside_rth', 'true');
            if (cfg.bar_size) p1.set('bar_size', cfg.bar_size);
            if (cfg.duration) p1.set('duration', cfg.duration);
            if (cfg.what_to_show) p1.set('what_to_show', cfg.what_to_show);
            if ([...p1.keys()].length > 1) attempts.push(p1);

            // Attempt 2: Let backend resolve defaults (only timeframe)
            const p2 = new URLSearchParams({ symbol });
            p2.set('timeframe', base.timeframe);
            p2.set('include_outside_rth', 'true');
            attempts.push(p2);

            // Attempt 3: timeframe + bar_count (hint number of bars)
            const p3 = new URLSearchParams({ symbol });
            p3.set('timeframe', base.timeframe);
            p3.set('include_outside_rth', 'true');
            // choose a sensible default bar_count depending on timeframe
            let bars = 100;
            if (base.timeframe === '1m') bars = 390; // all session minutes
            else if (base.timeframe === '5m') bars = 78;
            else if (base.timeframe === '15m') bars = 26;
            else if (base.timeframe === '30m') bars = 13;
            else if (base.timeframe === '1h') bars = 30;
            else if (base.timeframe === '2h') bars = 15;
            else if (base.timeframe === '4h') bars = 8;
            else if (base.timeframe === '1D') bars = 78; // legacy approx
            else if (base.timeframe === '1W') bars = 52; // weeks
            else if (base.timeframe === '1M') bars = 24; // months/weeks approximation
            p3.set('bar_count', String(bars));
            attempts.push(p3);

            return attempts;
        };

        const attempts = makeAttempts();

        const tryFetch = async (params) => {
            const res = await fetch(`/api/ibkr/market/historical?${params.toString()}`);
            let payload = null;
            let text = '';
            try { payload = await res.json(); }
            catch { try { text = await res.text(); } catch { text = ''; } }
            if (!res.ok) {
                const detail = (payload && (payload.detail || payload.error || payload.message)) || text || `HTTP ${res.status}`;
                throw new Error(detail);
            }
            return payload || {};
        };

        let lastError = null;
        for (const params of attempts) {
            try {
                const result = await tryFetch(params);
                if (result && result.success && Array.isArray(result.bars) && result.bars.length > 0) {
                    const normalizedBars = result.bars
                        .map((bar) => this._normalizeHistoricalBar(bar))
                        .filter(Boolean);
                    this.latestBars = normalizedBars;
                    const chartData = this._prepareChartSeries(timeframe, normalizedBars);
                    this._refreshChartWithData(symbol, timeframe, chartData);
                    console.log(`✅ Loaded ${symbol} chart data (${result.count ?? result.bars.length} bars)`);
                    return;
                }
                // If structure unexpected, try next attempt
                lastError = new Error('unexpected payload structure');
            } catch (err) {
                lastError = err;
                continue;
            }
        }

        console.error('❌ Error loading chart data:', lastError);
        try {
            // Optional: show a small inline notice if available
            const notice = document.getElementById('chart-error-notice');
            if (notice) {
                notice.textContent = `Failed to load chart data: ${lastError?.message || lastError}`;
                notice.style.display = 'block';
            }
        } catch (_) {}
    }

    /**
     * Setup stock selector dropdown event listener
     */
    setupStockSelector() {
        const selector = document.getElementById('stock-selector');
        if (selector) {
            selector.addEventListener('change', (e) => {
                const newSymbol = e.target.value;
                console.log(`📊 Stock changed to: ${newSymbol}`);
                this.currentSymbol = newSymbol;
                this.loadChartData(this.currentSymbol, this.currentTimeframe);
            });
            console.log('✅ Stock selector event listener attached');
        } else {
            console.warn('⚠️ Stock selector element not found');
        }
    }

    /**
     * Setup timeframe buttons event listeners
     */
    setupTimeframeButtons() {
        const buttons = document.querySelectorAll('.chart-timeframe');
        
        if (buttons.length === 0) {
            console.warn('⚠️ No timeframe buttons found');
            return;
        }

        buttons.forEach(button => {
            button.addEventListener('click', (e) => {
                const newTimeframe = e.target.dataset.timeframe;
                
                if (!newTimeframe) {
                    console.warn('⚠️ Timeframe not specified in button');
                    return;
                }

                console.log(`📅 Timeframe changed to: ${newTimeframe}`);

                // Update active button state
                buttons.forEach(btn => btn.classList.remove('active'));
                e.target.classList.add('active');

                // Update chart with new timeframe
                this.currentTimeframe = newTimeframe;
                this.loadChartData(this.currentSymbol, this.currentTimeframe);
                this._scheduleAutoRefresh();
            });
        });

        console.log(`✅ Timeframe button listeners attached (${buttons.length} buttons)`);
    }

    /**
     * Change the displayed stock
     * @param {string} symbol - Stock symbol to display
     */
    changeStock(symbol) {
        this.currentSymbol = symbol;
        
        // Update dropdown if it exists
        const selector = document.getElementById('stock-selector');
        if (selector) {
            selector.value = symbol;
        }
        
        this.loadChartData(symbol, this.currentTimeframe);
        this._scheduleAutoRefresh();
    }

    /**
     * Change the timeframe
     * @param {string} timeframe - Timeframe to display
     */
    changeTimeframe(timeframe) {
        this.currentTimeframe = timeframe;
        
        // Update active button
        document.querySelectorAll('.chart-timeframe').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.timeframe === timeframe);
        });
        
        this.loadChartData(this.currentSymbol, timeframe);
        this._scheduleAutoRefresh();
    }

    /**
     * Refresh current chart data
     */
    refresh() {
        this.loadChartData(this.currentSymbol, this.currentTimeframe);
    }

    _scheduleAutoRefresh() {
        // Clear previous timer
        if (this._autoTimer) {
            clearInterval(this._autoTimer);
            this._autoTimer = null;
        }
        const tf = this.currentTimeframe;
        const baseSeconds = AUTO_REFRESH_SECONDS[tf] || 60;
        const seconds = this._realtimeActive ? Math.max(baseSeconds, 300) : baseSeconds;
        // Only refresh when page is visible to be gentle on pacing
        const tick = () => {
            if (document.visibilityState === 'visible') {
                this.loadChartData(this.currentSymbol, this.currentTimeframe);
            }
        };
        this._autoTimer = setInterval(tick, seconds * 1000);
        // Also refresh once after a short delay to show quick update when not in live mode
        if (!this._realtimeActive) {
            setTimeout(() => tick(), Math.min(5000, seconds * 1000));
        }
    }

    // ----- Helpers -----
    _parseNumeric(value) {
        if (value === null || value === undefined) {
            return null;
        }
        if (typeof value === 'number') {
            return Number.isFinite(value) ? value : null;
        }
        if (typeof value === 'string') {
            const cleaned = value.replace(/[^0-9.\-]/g, '');
            if (!cleaned) {
                return null;
            }
            const parsed = Number(cleaned);
            return Number.isFinite(parsed) ? parsed : null;
        }
        return null;
    }

    _prepareChartSeries(timeframe, bars) {
        const safeBars = Array.isArray(bars) ? bars : [];
        const resolveClose = (bar) => {
            if (!bar) {
                return null;
            }
            const candidates = [bar.close, bar.last, bar.price, bar.c];
            for (const candidate of candidates) {
                const parsed = this._parseNumeric(candidate);
                if (parsed !== null) {
                    return parsed;
                }
            }
            return null;
        };

        const labels = safeBars.map((bar) => {
            const raw = bar.time || bar.timestamp || bar.date || bar.t;
            if (!raw) {
                return '';
            }
            const dt = new Date(raw);
            if (Number.isNaN(dt.getTime())) {
                return String(raw);
            }
            if (timeframe === '1D') {
                return dt.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' });
            }
            if (timeframe === '1W') {
                return dt.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
            }
            return dt.toLocaleDateString();
        });

        const ohlc = safeBars.map((bar, index) => {
            const raw = bar.time || bar.timestamp || bar.date || bar.t || labels[index] || index;
            const parsedDate = new Date(raw);
            const timeValue = Number.isNaN(parsedDate.getTime()) ? raw : parsedDate;
            return {
                t: timeValue,
                o: this._parseNumeric(bar.open ?? bar.o),
                h: this._parseNumeric(bar.high ?? bar.h),
                l: this._parseNumeric(bar.low ?? bar.l),
                c: resolveClose(bar),
                v: this._parseNumeric(bar.volume ?? bar.vol ?? bar.v),
            };
        });

        const prices = ohlc.map(entry => entry.c);
        const volumes = ohlc.map(entry => entry.v);

        const firstClose = resolveClose(safeBars[0]);
        const lastClose = resolveClose(safeBars[safeBars.length - 1]);
        const isPositive = firstClose === null || lastClose === null ? true : lastClose >= firstClose;

        return {
            labels,
            prices,
            ohlc,
            volumes,
            isPositive,
        };
    }

    _normalizeHistoricalBar(bar) {
        if (!bar) {
            return null;
        }
        const open = this._parseNumeric(bar.open ?? bar.o);
        const high = this._parseNumeric(bar.high ?? bar.h);
        const low = this._parseNumeric(bar.low ?? bar.l);
        const close = this._parseNumeric(bar.close ?? bar.last ?? bar.price ?? bar.c);
        const volume = this._parseNumeric(bar.volume ?? bar.vol ?? bar.v) ?? 0;
        const time = this._coerceIsoTime(bar.time ?? bar.timestamp ?? bar.date ?? bar.t);
        if (!time) {
            return null;
        }
        return { time, open, high, low, close, volume };
    }

    _coerceIsoTime(raw) {
        if (raw === null || raw === undefined) {
            return null;
        }
        if (raw instanceof Date && !Number.isNaN(raw.getTime())) {
            return raw.toISOString();
        }
        const fromNumber = typeof raw === 'number' ? new Date(raw) : null;
        if (fromNumber && !Number.isNaN(fromNumber.getTime())) {
            return fromNumber.toISOString();
        }
        const fromString = new Date(raw);
        if (!Number.isNaN(fromString.getTime())) {
            return fromString.toISOString();
        }
        return typeof raw === 'string' ? raw : null;
    }

    async _ensureRealtimeSubscription(symbol, timeframe) {
        if (!this.realtimeClient || !this._realtimeActive) {
            return;
        }
        if (!symbol || !timeframe) {
            return;
        }
        try {
            await this.realtimeClient.setActive(symbol, [timeframe]);
        } catch (err) {
            console.warn('⚠️ Failed to update real-time subscription', err);
        }
    }

    _initRealtime() {
        const btn = document.getElementById('realtime-btn');
        const hasSignalR = typeof window !== 'undefined'
            && window.signalR
            && typeof window.signalR.HubConnectionBuilder === 'function'
            && typeof window.RealtimeHubClient === 'function';

        if (!hasSignalR) {
            if (btn) {
                btn.disabled = true;
                btn.textContent = '⚪ Real-time unavailable';
            }
            this._updateRealtimeBadge('unsupported');
            return;
        }

        this.realtimeClient = new window.RealtimeHubClient();
        this._realtimeSubscriptions.quote = this.realtimeClient.onQuote((quote) => this._handleRealtimeQuote(quote));
        this._realtimeSubscriptions.candle = this.realtimeClient.onCandle((candle) => this._handleRealtimeCandle(candle));
        this._realtimeSubscriptions.status = this.realtimeClient.onStatus((status) => this._handleRealtimeStatus(status));

        if (btn) {
            btn.addEventListener('click', async () => {
                if (this._realtimeActive) {
                    await this._disableRealtime();
                } else {
                    try {
                        await this._enableRealtime();
                    } catch (err) {
                        console.error('❌ Real-time enable failed', err);
                    }
                }
            });
        }

        // Auto-start live updates by default
        this._enableRealtime().catch((err) => {
            console.error('❌ Real-time connection failed', err);
            this._updateRealtimeBadge('error', err?.message);
        });
    }

    async _enableRealtime() {
        if (!this.realtimeClient) {
            return;
        }
        if (this._realtimeActive) {
            return;
        }

        this._updateRealtimeBadge('connecting');
        try {
            await this.realtimeClient.connect();
            await this.realtimeClient.setActive(this.currentSymbol, [this.currentTimeframe]);
            this._realtimeActive = true;
            this._updateRealtimeBadge('connected');
            this._scheduleAutoRefresh();
        } catch (err) {
            this._updateRealtimeBadge('error', err?.message);
            throw err;
        }
    }

    async _disableRealtime() {
        if (!this.realtimeClient) {
            return;
        }

        this._realtimeActive = false;
        try {
            await this.realtimeClient.disconnect();
        } catch (err) {
            console.warn('⚠️ Error disconnecting real-time stream', err);
        }
        this._updateRealtimeBadge('off');
        this._scheduleAutoRefresh();
    }

    _handleRealtimeStatus(status) {
        if (!status) {
            return;
        }

        switch (status.state) {
            case 'connected':
                if (this._realtimeActive) {
                    this._updateRealtimeBadge('connected');
                }
                break;
            case 'connecting':
                this._updateRealtimeBadge('connecting');
                break;
            case 'reconnecting':
                this._updateRealtimeBadge('reconnecting');
                break;
            case 'disconnected':
                this._updateRealtimeBadge(this._realtimeActive ? 'reconnecting' : 'off');
                break;
            case 'error':
                this._updateRealtimeBadge('error', status.message || status.error);
                break;
            default:
                break;
        }
    }

    _handleRealtimeQuote(quote) {
        if (!quote) {
            return;
        }
        const symbol = (quote.Symbol || quote.symbol || '').toUpperCase();
        if (!symbol || symbol !== this.currentSymbol.toUpperCase()) {
            return;
        }
        this._latestQuote = quote;

        const lastUpdate = document.getElementById('last-update');
        if (lastUpdate) {
            lastUpdate.textContent = `Live ${new Date().toLocaleTimeString()}`;
            lastUpdate.style.color = '#10b981';
        }
    }

    _handleRealtimeCandle(candle) {
        if (!candle) {
            return;
        }
        const symbol = (candle.Symbol || candle.symbol || '').toUpperCase();
        const timeframe = (candle.Timeframe || candle.timeframe || '').toLowerCase();
        if (!symbol || symbol !== this.currentSymbol.toUpperCase()) {
            return;
        }
        if (!timeframe || timeframe !== this.currentTimeframe.toLowerCase()) {
            return;
        }

        const normalized = {
            time: this._coerceIsoTime(candle.Time || candle.time),
            open: this._parseNumeric(candle.Open ?? candle.open),
            high: this._parseNumeric(candle.High ?? candle.high),
            low: this._parseNumeric(candle.Low ?? candle.low),
            close: this._parseNumeric(candle.Close ?? candle.close),
            volume: this._parseNumeric(candle.Volume ?? candle.volume) ?? 0,
            partial: Boolean(candle.Partial ?? candle.partial)
        };

        if (!normalized.time) {
            return;
        }

        if (!Array.isArray(this.latestBars)) {
            this.latestBars = [];
        }

        const ts = this._barTimestamp(normalized);
        if (ts === null) {
            return;
        }

        let updated = false;
        for (let i = this.latestBars.length - 1; i >= 0; i -= 1) {
            const existingTs = this._barTimestamp(this.latestBars[i]);
            if (existingTs === null) {
                continue;
            }
            if (existingTs === ts) {
                const current = this.latestBars[i];
                this.latestBars[i] = {
                    ...current,
                    ...normalized,
                };
                updated = true;
                break;
            }
            if (existingTs < ts) {
                break;
            }
        }

        if (!updated) {
            this.latestBars.push(normalized);
        }

        this.latestBars.sort((a, b) => {
            const left = this._barTimestamp(a) ?? 0;
            const right = this._barTimestamp(b) ?? 0;
            return left - right;
        });

        if (this.latestBars.length > 600) {
            this.latestBars = this.latestBars.slice(this.latestBars.length - 600);
        }

        this._scheduleRealtimeRender();
    }

    _scheduleRealtimeRender() {
        if (this._pendingRealtimeRender) {
            return;
        }
        this._pendingRealtimeRender = true;
        setTimeout(() => {
            this._pendingRealtimeRender = false;
            this._renderRealtimeSnapshot();
        }, REALTIME_RENDER_DEBOUNCE_MS);
    }

    _renderRealtimeSnapshot() {
        if (!this.chart || !Array.isArray(this.latestBars) || !this._realtimeActive) {
            return;
        }
        const chartData = this._prepareChartSeries(this.currentTimeframe, this.latestBars);
        this._refreshChartWithData(this.currentSymbol, this.currentTimeframe, chartData);
    }

    _updateRealtimeBadge(state, detail) {
        this._realtimeStatus = state;
        const btn = document.getElementById('realtime-btn');
        const badge = document.getElementById('ws-status-badge');
        if (badge) {
            badge.removeAttribute('title');
        }

        if (!btn || !badge) {
            return;
        }

        btn.classList.remove('on');
        btn.disabled = false;

        switch (state) {
            case 'connected':
                btn.classList.add('on');
                btn.textContent = '🟢 Disable Real-time';
                badge.className = 'ws-badge ws-on';
                badge.textContent = 'WS: ON';
                break;
            case 'connecting':
                btn.disabled = true;
                btn.textContent = '🟡 Connecting...';
                badge.className = 'ws-badge ws-warn';
                badge.textContent = 'WS: CONNECTING';
                break;
            case 'reconnecting':
                btn.disabled = true;
                btn.textContent = '🟡 Reconnecting...';
                badge.className = 'ws-badge ws-warn';
                badge.textContent = 'WS: RECONNECTING';
                break;
            case 'error':
                btn.textContent = '⚠️ Retry Real-time';
                badge.className = 'ws-badge ws-warn';
                badge.textContent = 'WS: ERROR';
                if (detail) {
                    badge.title = detail;
                }
                break;
            case 'unsupported':
                btn.disabled = true;
                btn.textContent = '⚪ Real-time unavailable';
                badge.className = 'ws-badge ws-off';
                badge.textContent = 'WS: OFF';
                break;
            case 'off':
            default:
                btn.textContent = '⚪ Enable Real-time';
                badge.className = 'ws-badge ws-off';
                badge.textContent = 'WS: OFF';
                break;
        }
    }

    _barTimestamp(bar) {
        if (!bar) {
            return null;
        }
        const raw = bar.time ?? bar.timestamp ?? bar.date ?? bar.t;
        if (!raw) {
            return null;
        }
        const dt = new Date(raw);
        if (!Number.isNaN(dt.getTime())) {
            return dt.getTime();
        }
        if (typeof raw === 'number') {
            return raw;
        }
        return null;
    }

    _createChart(ctx) {
        // Destroy existing if any
        if (this.chart) { this.chart.destroy(); }
        if (this.currentType === 'candlestick' && Chart?.registry?.getChart('candlestick')) {
            // Financial plugin chart
            this.chart = new Chart(ctx, {
                type: 'candlestick',
                data: {
                    datasets: [{
                        label: 'Loading...',
                        data: [],
                        borderColor: '#e5e7eb',
                        color: { up: '#10b981', down: '#ef4444', unchanged: '#9ca3af' },
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { labels: { color: '#e2e8f0' } } },
                    scales: {
                        x: {
                            ticks: { color: '#a0aec0' },
                            grid: { color: 'rgba(160,174,192,0.2)' }
                        },
                        y: {
                            ticks: { color: '#a0aec0' },
                            grid: { color: 'rgba(160,174,192,0.2)' }
                        }
                    }
                }
            });
        } else {
            // Line chart
            this.chart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Loading...',
                        data: [],
                        borderColor: '#10b981',
                        backgroundColor: 'rgba(16, 185, 129, 0.1)',
                        borderWidth: 2,
                        fill: true,
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { labels: { color: '#e2e8f0' } },
                        tooltip: {
                            mode: 'index', intersect: false,
                            backgroundColor: 'rgba(0, 0, 0, 0.8)',
                            titleColor: '#fff', bodyColor: '#fff',
                            borderColor: '#10b981', borderWidth: 1
                        }
                    },
                    scales: {
                        x: { ticks: { color: '#a0aec0' }, grid: { color: 'rgba(160,174,192,0.2)' } },
                        y: { ticks: { color: '#a0aec0', callback: v => '$' + Number(v).toFixed(2) }, grid: { color: 'rgba(160,174,192,0.2)' } }
                    },
                    interaction: { mode: 'nearest', axis: 'x', intersect: false }
                }
            });
        }
    }

    _refreshChartWithData(symbol, timeframe, data) {
        if (this.currentType === 'candlestick' && this.chart.config.type === 'candlestick') {
            this.chart.data.labels = Array.isArray(data.labels) ? data.labels : [];
            const ds = this.chart.data.datasets[0];
            ds.label = `${symbol} - ${timeframe}`;
            ds.data = (data.ohlc || [])
                .filter(entry => entry && entry.c !== null && entry.o !== null && entry.h !== null && entry.l !== null)
                .map(d => ({ x: d.t, o: d.o, h: d.h, l: d.l, c: d.c }));
            this.chart.update();
        } else {
            const labels = Array.isArray(data.labels) ? data.labels : [];
            this.chart.data.labels = labels;
            const ds = this.chart.data.datasets[0];
            ds.label = `${symbol} - ${timeframe}`;
            ds.data = (data.prices || []).map(val => (val === null ? null : Number(val)));
            if (data.isPositive) {
                ds.borderColor = '#10b981';
                ds.backgroundColor = 'rgba(16, 185, 129, 0.1)';
            } else {
                ds.borderColor = '#ef4444';
                ds.backgroundColor = 'rgba(239, 68, 68, 0.1)';
            }
            this.chart.update();
        }
    }

    setupChartTypeToggle() {
        const line = document.getElementById('chart-type-line');
        const candle = document.getElementById('chart-type-candle');
        const canvas = document.getElementById('priceChart');
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        const onChange = () => {
            const newType = candle && candle.checked ? 'candlestick' : 'line';
            if (newType !== this.currentType) {
                this.currentType = newType;
                this._createChart(ctx);
                // reload to populate
                this.loadChartData(this.currentSymbol, this.currentTimeframe);
            }
        };
        if (line) line.addEventListener('change', onChange);
        if (candle) candle.addEventListener('change', onChange);
    }
}

// Export for use in main.js
window.ChartManager = ChartManager;