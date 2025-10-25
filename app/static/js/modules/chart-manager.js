/**
 * Chart Manager Module
 * Handles all price chart functionality including Chart.js initialization,
 * stock selection, timeframe changes, and real-time data loading.
 */

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

        try {
            const response = await fetch(`/api/financial/historical/${symbol}?timeframe=${timeframe}`);
            const result = await response.json();

            if (result.status === 'success' && result.data) {
                const data = result.data;
                this._refreshChartWithData(symbol, timeframe, data);
                console.log(`✅ Loaded ${symbol} chart data (${data.data_points} points)`);
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
            // Expect data.ohlc: [{t, o,h,l,c}]
            const ds = this.chart.data.datasets[0];
            ds.label = `${symbol} - ${timeframe}`;
            ds.data = (data.ohlc || []).map(d => ({ x: d.t, o: d.o, h: d.h, l: d.l, c: d.c }));
            this.chart.update();
        } else {
            // Line mode uses labels + prices
            this.chart.data.labels = data.labels;
            const ds = this.chart.data.datasets[0];
            ds.label = `${symbol} - ${timeframe}`;
            ds.data = data.prices;
            if (data.is_positive) {
                ds.borderColor = '#10b981'; ds.backgroundColor = 'rgba(16, 185, 129, 0.1)';
            } else {
                ds.borderColor = '#ef4444'; ds.backgroundColor = 'rgba(239, 68, 68, 0.1)';
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