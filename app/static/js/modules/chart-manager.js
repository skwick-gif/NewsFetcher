/**
 * Chart Manager Module
 * Handles all price chart functionality including Chart.js initialization,
 * stock selection, timeframe changes, and real-time data loading.
 */

const TIMEFRAME_CONFIG = {
    '1D': { timeframe: '1D', bar_size: '5 mins', duration: '1 D', include_outside_rth: false },
    '1W': { timeframe: '1W', bar_size: '30 mins', duration: '5 D', include_outside_rth: false },
    '1M': { timeframe: '1M', bar_size: '1 day', duration: '1 M' },
    '3M': { timeframe: '3M', bar_size: '1 day', duration: '3 M' },
    '6M': { timeframe: '6M', bar_size: '1 day', duration: '6 M' },
    '1Y': { timeframe: '1Y', bar_size: '1 day', duration: '12 M' },
};

class ChartManager {
    constructor() {
        this.chart = null;
        this.currentSymbol = 'AAPL';  // Default symbol
        this.currentTimeframe = '1D';  // Default timeframe
        this.currentType = 'line';     // 'line' or 'candlestick'
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

        // Load initial chart data
        await this.loadChartData(this.currentSymbol, this.currentTimeframe);

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

        const params = new URLSearchParams({ symbol });
        const cfg = TIMEFRAME_CONFIG[timeframe] || {};
        if (cfg.timeframe) {
            params.set('timeframe', cfg.timeframe);
        } else {
            params.set('timeframe', timeframe);
        }
        if (cfg.bar_size) {
            params.set('bar_size', cfg.bar_size);
        }
        if (cfg.duration) {
            params.set('duration', cfg.duration);
        }
        if (cfg.bar_count) {
            params.set('bar_count', String(cfg.bar_count));
        }
        if (typeof cfg.include_outside_rth === 'boolean') {
            params.set('include_outside_rth', String(cfg.include_outside_rth));
        }
        if (cfg.what_to_show) {
            params.set('what_to_show', cfg.what_to_show);
        }

        try {
            const response = await fetch(`/api/ibkr/market/historical?${params.toString()}`);
            let payload = null;
            let fallbackText = '';
            try {
                payload = await response.json();
            } catch (jsonError) {
                try {
                    fallbackText = await response.text();
                } catch (textError) {
                    fallbackText = '';
                }
            }
            if (!response.ok) {
                const detail = (payload && (payload.detail || payload.error || payload.message)) || fallbackText || `HTTP ${response.status}`;
                throw new Error(detail);
            }
            const result = payload || {};

            if (result.success && Array.isArray(result.bars)) {
                const chartData = this._prepareChartSeries(timeframe, result.bars);
                this.latestBars = result.bars;
                this._refreshChartWithData(symbol, timeframe, chartData);
                console.log(`✅ Loaded ${symbol} chart data (${result.count ?? result.bars.length} bars)`);
            } else {
                console.error('❌ Failed to load chart data:', result);
            }
        } catch (error) {
            console.error('❌ Error loading chart data:', error);
        }
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
    }

    /**
     * Refresh current chart data
     */
    refresh() {
        this.loadChartData(this.currentSymbol, this.currentTimeframe);
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