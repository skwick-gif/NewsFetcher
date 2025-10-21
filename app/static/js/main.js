/**
 * Main Application Orchestrator
 * Initializes all modules and manages application lifecycle
 */

// Global module instances
let chartManager;
let marketData;
let alertsManager;
let mlScanner;
let settingsManager;
let wsClient;

// Periodic update intervals
let statusInterval;
let statsInterval;
let alertsInterval;
let marketIndicesInterval;
let marketSentimentInterval;
let tradingInsightsInterval;

/**
 * Initialize all application modules
 */
async function initializeApp() {
    console.log('🚀 Initializing Tariff Radar Dashboard...');

    try {
        // Initialize settings first (other modules may depend on it)
        settingsManager = new SettingsManager();
        settingsManager.init();

        // Initialize chart manager
        chartManager = new ChartManager();
        await chartManager.init();

        // Initialize market data manager
        marketData = new MarketDataManager();
        await marketData.updateMarketIndices();
        await marketData.updateMarketSentiment();
        await marketData.updateTradingInsights();

        // Initialize alerts manager
        alertsManager = new AlertsManager();
        await alertsManager.init();

        // Initialize ML scanner
        mlScanner = new MLScanner();
        await mlScanner.init();

        // Initialize WebSocket client (not connected by default)
        wsClient = new WebSocketClient();
        wsClient.init();

        // Make instances globally available
        window.chartManager = chartManager;
        window.marketData = marketData;
        window.alertsManager = alertsManager;
        window.mlScanner = mlScanner;
        window.settingsManager = settingsManager;
        window.wsClient = wsClient;

        // Setup periodic updates
        setupPeriodicUpdates();

        // Setup event listeners
        setupEventListeners();

        console.log('✅ Dashboard initialized successfully!');
    } catch (error) {
        console.error('❌ Failed to initialize dashboard:', error);
    }
}

/**
 * Setup periodic update intervals
 */
function setupPeriodicUpdates() {
    const settings = settingsManager.getSettings();

    if (!settings.autoRefresh) {
        console.log('Auto-refresh disabled in settings');
        return;
    }

    // Status updates every 30 seconds
    statusInterval = setInterval(async () => {
        if (mlScanner && typeof mlScanner.loadMLStatus === 'function') {
            await mlScanner.loadMLStatus();
        }
    }, 30000);

    // Stats updates every 60 seconds
    statsInterval = setInterval(async () => {
        if (alertsManager && typeof alertsManager.loadStats === 'function') {
            await alertsManager.loadStats();
        }
    }, 60000);

    // Alerts updates every 30 seconds
    alertsInterval = setInterval(async () => {
        if (alertsManager && typeof alertsManager.loadAlerts === 'function') {
            await alertsManager.loadAlerts();
        }
    }, 30000);

    // Market indices every 10 seconds
    marketIndicesInterval = setInterval(async () => {
        await marketData.updateMarketIndices();
    }, 10000);

    // Market sentiment every 15 seconds
    marketSentimentInterval = setInterval(async () => {
        await marketData.updateMarketSentiment();
    }, 15000);

    // Trading insights every 20 seconds
    tradingInsightsInterval = setInterval(async () => {
        await marketData.updateTradingInsights();
    }, 20000);

    console.log('✅ Periodic updates configured');
}

/**
 * Stop all periodic updates
 */
function stopPeriodicUpdates() {
    clearInterval(statusInterval);
    clearInterval(statsInterval);
    clearInterval(alertsInterval);
    clearInterval(marketIndicesInterval);
    clearInterval(marketSentimentInterval);
    clearInterval(tradingInsightsInterval);

    console.log('⏸️ Periodic updates stopped');
}

/**
 * Setup event listeners for UI interactions
 */
function setupEventListeners() {
    // Tab switching - FIXED: use .tab instead of .tab-btn
    document.querySelectorAll('.tab').forEach(btn => {
        btn.addEventListener('click', function() {
            console.log('Tab clicked:', this.dataset.tab);
            showTab(this.dataset.tab);
        });
    });

    // Refresh buttons
    const refreshBtn = document.getElementById('refresh-btn');
    if (refreshBtn) {
        refreshBtn.addEventListener('click', async () => {
            console.log('🔄 Manual refresh triggered');
            await refreshAll();
        });
    }
    
    // Refresh Market Intelligence button
    const refreshMarketBtn = document.getElementById('refresh-market-btn');
    if (refreshMarketBtn) {
        refreshMarketBtn.addEventListener('click', async () => {
            console.log('🔄 Refreshing market intelligence');
            if (marketData) {
                await marketData.updateMarketIndices();
                await marketData.updateMarketSentiment();
            }
        });
    }

    // Save settings button
    const saveSettingsBtn = document.getElementById('save-settings-btn');
    if (saveSettingsBtn) {
        saveSettingsBtn.addEventListener('click', () => {
            settingsManager.saveSettings();
        });
    }

    // Reset settings button
    const resetSettingsBtn = document.getElementById('reset-settings-btn');
    if (resetSettingsBtn) {
        resetSettingsBtn.addEventListener('click', () => {
            settingsManager.resetSettings();
        });
    }

    // Article filter buttons
    document.querySelectorAll('.filter-btn').forEach(btn => {
        btn.addEventListener('click', function() {
            alertsManager.filterArticles(this.dataset.filter, this);
        });
    });

    // Analyze stock button (AI tab)
    const analyzeBtn = document.getElementById('analyze-stock-btn');
    if (analyzeBtn) {
        analyzeBtn.addEventListener('click', async () => {
            await mlScanner.analyzeStock();
        });
    }

    // Stock input - Enter key triggers analysis
    const stockInput = document.getElementById('ai-symbol-input');
    if (stockInput) {
        stockInput.addEventListener('keypress', async (e) => {
            if (e.key === 'Enter') {
                await mlScanner.analyzeStock();
            }
        });
    }

    // Realtime toggle button
    const realtimeBtn = document.getElementById('realtime-btn');
    if (realtimeBtn) {
        realtimeBtn.addEventListener('click', () => {
            toggleRealtime();
        });
    }

    console.log('✅ Event listeners setup');
}

/**
 * Show specific tab and hide others
 */
function showTab(tabName) {
    console.log('🔄 Switching to tab:', tabName);
    
    // Hide all tab content
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.remove('active');
    });

    // Remove active class from all tab buttons - FIXED: use .tab instead of .tab-btn
    document.querySelectorAll('.tab').forEach(btn => {
        btn.classList.remove('active');
    });

    // Show selected tab content
    const selectedContent = document.getElementById(tabName);
    if (selectedContent) {
        selectedContent.classList.add('active');
        console.log('✅ Tab content shown:', tabName);
    } else {
        console.error('❌ Tab content not found:', tabName);
    }

    // Add active class to selected tab button
    const selectedBtn = document.querySelector(`[data-tab="${tabName}"]`);
    if (selectedBtn) {
        selectedBtn.classList.add('active');
        console.log('✅ Tab button activated');
    } else {
        console.error('❌ Tab button not found for:', tabName);
    }

    console.log(`📑 Switched to tab: ${tabName}`);
}

/**
 * Refresh all data from all modules
 */
async function refreshAll() {
    try {
        const refreshTasks = [];
        
        if (chartManager && typeof chartManager.refresh === 'function') {
            refreshTasks.push(chartManager.refresh());
        }
        
        if (marketData && typeof marketData.refresh === 'function') {
            refreshTasks.push(marketData.refresh());
        }
        
        if (alertsManager && typeof alertsManager.refresh === 'function') {
            refreshTasks.push(alertsManager.refresh());
        }
        
        if (mlScanner && typeof mlScanner.refresh === 'function') {
            refreshTasks.push(mlScanner.refresh());
        }

        await Promise.all(refreshTasks);

        console.log('✅ All data refreshed');
        showRefreshMessage();
    } catch (error) {
        console.error('❌ Error refreshing data:', error);
    }
}

/**
 * Show refresh confirmation message
 */
function showRefreshMessage() {
    const refreshBtn = document.getElementById('refresh-btn');
    if (refreshBtn) {
        const originalText = refreshBtn.textContent;
        refreshBtn.textContent = '✅ Refreshed!';
        refreshBtn.style.background = '#4CAF50';

        setTimeout(() => {
            refreshBtn.textContent = originalText;
            refreshBtn.style.background = '';
        }, 2000);
    }
}

/**
 * Toggle real-time WebSocket connections
 */
function toggleRealtime() {
    const isConnected = wsClient.toggle();
    const button = document.getElementById('realtime-btn');

    if (!button) return;

    if (isConnected) {
        button.textContent = '🔴 Live Streaming ON';
        button.classList.remove('btn-outline-success');
        button.classList.add('btn-success');
    } else {
        button.textContent = '⚪ Enable Real-time';
        button.classList.remove('btn-success');
        button.classList.add('btn-outline-success');
    }
}

/**
 * Handle visibility change (pause updates when tab is hidden)
 */
document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
        console.log('⏸️ Tab hidden, pausing updates');
        stopPeriodicUpdates();
    } else {
        console.log('▶️ Tab visible, resuming updates');
        setupPeriodicUpdates();
        refreshAll(); // Refresh data when tab becomes visible
    }
});

/**
 * Cleanup before page unload
 */
window.addEventListener('beforeunload', () => {
    stopPeriodicUpdates();
    if (wsClient) {
        wsClient.disconnect();
    }
    console.log('👋 Dashboard cleanup complete');
});

/**
 * Initialize when DOM is ready
 */
document.addEventListener('DOMContentLoaded', () => {
    initializeApp();
});

/**
 * Handle settings file import
 */
function handleSettingsImport(input) {
    const file = input.files[0];
    if (file && settingsManager) {
        settingsManager.importSettings(file);
    }
    // Clear the input so the same file can be imported again if needed
    input.value = '';
}

/**
 * Analyze selected stock with AI
 */
async function analyzeStock(symbol) {
    if (!symbol) {
        // Reset to default state
        document.getElementById('selected-stock-analysis').innerHTML = 
            '<p style="text-align: center; opacity: 0.6;">Select a stock above for detailed AI analysis</p>';
        resetPredictions();
        resetRiskAssessment();
        return;
    }

    try {
        // Update stock analysis section
        document.getElementById('selected-stock-analysis').innerHTML = `
            <div class="loading-spinner" style="text-align: center;">
                <p>🔄 Analyzing ${symbol}...</p>
            </div>
        `;

        // Simulate AI analysis with realistic data
        const analysisData = await simulateStockAnalysis(symbol);
        
        // Update stock analysis
        document.getElementById('selected-stock-analysis').innerHTML = `
            <div class="stock-analysis-result">
                <h5 style="color: #3b82f6; margin-bottom: 10px;">${symbol} Analysis</h5>
                <div class="analysis-metrics">
                    <div class="analysis-metric">
                        <span class="metric-label">Current Price:</span>
                        <span class="metric-value">$${analysisData.currentPrice}</span>
                    </div>
                    <div class="analysis-metric">
                        <span class="metric-label">AI Score:</span>
                        <span class="metric-value ${analysisData.aiScore >= 7 ? 'bullish' : analysisData.aiScore >= 4 ? 'neutral' : 'bearish'}">
                            ${analysisData.aiScore}/10
                        </span>
                    </div>
                    <div class="analysis-metric">
                        <span class="metric-label">Recommendation:</span>
                        <span class="metric-value ${analysisData.recommendation.toLowerCase()}">${analysisData.recommendation}</span>
                    </div>
                </div>
                <div class="analysis-summary">
                    <p style="font-size: 0.85em; opacity: 0.8; margin-top: 10px;">
                        ${analysisData.summary}
                    </p>
                </div>
            </div>
        `;

        // Update neural network predictions
        updateNeuralPredictions(analysisData.predictions);
        
        // Update time series analysis
        updateTimeSeriesAnalysis(analysisData.timeSeries);
        
        // Update risk assessment
        updateRiskAssessment(analysisData.risk);

    } catch (error) {
        console.error('Error analyzing stock:', error);
        document.getElementById('selected-stock-analysis').innerHTML = 
            '<p style="color: #ef4444; text-align: center;">Error analyzing stock. Please try again.</p>';
    }
}

/**
 * Simulate stock analysis with realistic data
 */
async function simulateStockAnalysis(symbol) {
    // Simulate API call delay
    await new Promise(resolve => setTimeout(resolve, 1500));
    
    // Mock data based on symbol
    const stockData = {
        'AAPL': {
            currentPrice: '174.25',
            aiScore: 8.2,
            recommendation: 'BUY',
            summary: 'Strong fundamentals with positive momentum in services revenue. AI models indicate continued growth potential.',
            predictions: { short: 'Bullish', medium: 'Bullish', long: 'Neutral' },
            timeSeries: { volatility: 0.24, trendStrength: 0.78, support: 168.50, resistance: 180.00 },
            risk: { level: 0.3, factors: ['Market saturation concerns', 'Regulatory pressures'] }
        },
        'TSLA': {
            currentPrice: '248.42',
            aiScore: 6.8,
            recommendation: 'HOLD',
            summary: 'Mixed signals from production data and market sentiment. High volatility expected to continue.',
            predictions: { short: 'Neutral', medium: 'Bullish', long: 'Bullish' },
            timeSeries: { volatility: 0.45, trendStrength: 0.62, support: 230.00, resistance: 260.00 },
            risk: { level: 0.7, factors: ['High volatility', 'Regulatory changes', 'Competition'] }
        },
        'GOOGL': {
            currentPrice: '139.67',
            aiScore: 7.5,
            recommendation: 'BUY',
            summary: 'Strong cloud growth and AI initiatives positioning for long-term success. Search revenue remains stable.',
            predictions: { short: 'Bullish', medium: 'Bullish', long: 'Bullish' },
            timeSeries: { volatility: 0.28, trendStrength: 0.72, support: 135.00, resistance: 145.00 },
            risk: { level: 0.4, factors: ['Antitrust concerns', 'Competition in AI'] }
        }
    };

    return stockData[symbol] || {
        currentPrice: '150.00',
        aiScore: 5.0,
        recommendation: 'HOLD',
        summary: 'Limited data available for comprehensive analysis. Monitor for more information.',
        predictions: { short: 'Neutral', medium: 'Neutral', long: 'Neutral' },
        timeSeries: { volatility: 0.30, trendStrength: 0.50, support: 140.00, resistance: 160.00 },
        risk: { level: 0.5, factors: ['Limited data', 'Market uncertainty'] }
    };
}

/**
 * Update neural network predictions
 */
function updateNeuralPredictions(predictions) {
    const container = document.getElementById('neural-predictions');
    container.innerHTML = `
        <div class="prediction-item">
            <span class="prediction-label">Short-term (1 day):</span>
            <span class="prediction-value ${predictions.short.toLowerCase()}">${predictions.short}</span>
        </div>
        <div class="prediction-item">
            <span class="prediction-label">Medium-term (1 week):</span>
            <span class="prediction-value ${predictions.medium.toLowerCase()}">${predictions.medium}</span>
        </div>
        <div class="prediction-item">
            <span class="prediction-label">Long-term (1 month):</span>
            <span class="prediction-value ${predictions.long.toLowerCase()}">${predictions.long}</span>
        </div>
    `;
}

/**
 * Update time series analysis
 */
function updateTimeSeriesAnalysis(timeSeries) {
    const container = document.getElementById('timeseries-results');
    container.innerHTML = `
        <div class="timeseries-metric">
            <span class="metric-name">Volatility Index:</span>
            <span class="metric-value ${timeSeries.volatility > 0.4 ? 'bearish' : timeSeries.volatility > 0.3 ? 'neutral' : 'bullish'}">
                ${(timeSeries.volatility * 100).toFixed(1)}%
            </span>
        </div>
        <div class="timeseries-metric">
            <span class="metric-name">Trend Strength:</span>
            <span class="metric-value ${timeSeries.trendStrength > 0.7 ? 'bullish' : timeSeries.trendStrength > 0.4 ? 'neutral' : 'bearish'}">
                ${(timeSeries.trendStrength * 100).toFixed(1)}%
            </span>
        </div>
        <div class="timeseries-metric">
            <span class="metric-name">Support Level:</span>
            <span class="metric-value bullish">$${timeSeries.support.toFixed(2)}</span>
        </div>
        <div class="timeseries-metric">
            <span class="metric-name">Resistance Level:</span>
            <span class="metric-value bearish">$${timeSeries.resistance.toFixed(2)}</span>
        </div>
    `;
}

/**
 * Update risk assessment
 */
function updateRiskAssessment(risk) {
    // Update risk needle position
    const needle = document.querySelector('.risk-needle');
    const rotation = (risk.level * 180) - 90; // -90 to 90 degrees
    needle.style.transform = `translateX(-50%) rotate(${rotation}deg)`;
    
    // Update risk factors
    const factorsContainer = document.getElementById('risk-factors');
    const riskLevel = risk.level < 0.33 ? 'Low' : risk.level < 0.66 ? 'Medium' : 'High';
    const riskColor = risk.level < 0.33 ? '#10b981' : risk.level < 0.66 ? '#f59e0b' : '#ef4444';
    
    factorsContainer.innerHTML = `
        <div style="text-align: center; margin-bottom: 10px;">
            <span style="color: ${riskColor}; font-weight: 600;">${riskLevel} Risk</span>
            <span style="opacity: 0.7;">(${(risk.level * 100).toFixed(0)}%)</span>
        </div>
        <div style="font-size: 0.8em;">
            <strong>Key Factors:</strong>
            <ul style="margin: 5px 0; padding-left: 20px;">
                ${risk.factors.map(factor => `<li>${factor}</li>`).join('')}
            </ul>
        </div>
    `;
}

/**
 * Reset predictions to default state
 */
function resetPredictions() {
    document.getElementById('neural-predictions').innerHTML = `
        <div class="prediction-item">
            <span class="prediction-label">Short-term (1 day):</span>
            <span class="prediction-value neutral">Analysis pending</span>
        </div>
        <div class="prediction-item">
            <span class="prediction-label">Medium-term (1 week):</span>
            <span class="prediction-value neutral">Analysis pending</span>
        </div>
        <div class="prediction-item">
            <span class="prediction-label">Long-term (1 month):</span>
            <span class="prediction-value neutral">Analysis pending</span>
        </div>
    `;
}

/**
 * Reset risk assessment to default state
 */
function resetRiskAssessment() {
    const needle = document.querySelector('.risk-needle');
    needle.style.transform = 'translateX(-50%) rotate(0deg)';
    
    document.getElementById('risk-factors').innerHTML = 
        '<p style="text-align: center; opacity: 0.6;">Select a stock for risk analysis</p>';
    
    document.getElementById('timeseries-results').innerHTML = `
        <div class="timeseries-metric">
            <span class="metric-name">Volatility Index:</span>
            <span class="metric-value">--</span>
        </div>
        <div class="timeseries-metric">
            <span class="metric-name">Trend Strength:</span>
            <span class="metric-value">--</span>
        </div>
        <div class="timeseries-metric">
            <span class="metric-name">Support Level:</span>
            <span class="metric-value">--</span>
        </div>
        <div class="timeseries-metric">
            <span class="metric-name">Resistance Level:</span>
            <span class="metric-value">--</span>
        </div>
    `;
}

// Export functions for console access
window.showTab = showTab;
window.refreshAll = refreshAll;
window.toggleRealtime = toggleRealtime;
window.stopPeriodicUpdates = stopPeriodicUpdates;
window.setupPeriodicUpdates = setupPeriodicUpdates;
window.handleSettingsImport = handleSettingsImport;
window.analyzeStock = analyzeStock;