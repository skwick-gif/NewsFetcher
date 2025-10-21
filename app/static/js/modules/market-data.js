/**
 * Market Data Manager Module
 * Handles market indices updates, market sentiment, and trading insights
 */

class MarketDataManager {
    constructor() {
        this.updateInterval = null;
    }

    /**
     * Initialize market data updates
     */
    async init() {
        // Load initial data
        await this.updateMarketIndices();
        await this.updateMarketSentiment();
        await this.updateTradingInsights();

        console.log('✅ Market Data Manager initialized');
    }

    /**
     * Update market indices (S&P500, Nasdaq, Dow Jones, etc.)
     */
    async updateMarketIndices() {
        try {
            const response = await fetch('/api/financial/market-indices');
            const data = await response.json();

            console.log('📊 API Response:', data);

            if (data.status === 'success' && data.data) {
                console.log('📈 Indices data:', data.data);
                
                // Map API keys to HTML IDs
                const indexNameMap = {
                    'sp500': 'SP500',
                    'nasdaq': 'NASDAQ',
                    'dow': 'DOW',
                    'russell': 'RUSSELL2000',
                    'vix': 'VIX'
                };
                
                Object.entries(data.data).forEach(([indexName, indexData]) => {
                    console.log(`🔍 Processing ${indexName}:`, indexData);
                    
                    // Convert API key to HTML ID format
                    const htmlId = indexNameMap[indexName] || indexName.toUpperCase();
                    console.log(`🔄 Mapping ${indexName} → ${htmlId}`);
                    
                    const valueElement = document.getElementById(`${htmlId}-value`);
                    const changeElement = document.getElementById(`${htmlId}-change`);

                    console.log(`🎯 Elements found: value=${!!valueElement}, change=${!!changeElement}`);

                    if (valueElement && indexData.price) {
                        valueElement.textContent = indexData.price.toFixed(2);
                        console.log(`✏️ Set ${htmlId} value to:`, indexData.price.toFixed(2));
                    }

                    if (changeElement && indexData.change_percent !== undefined) {
                        const isPositive = indexData.is_positive;
                        changeElement.className = isPositive ? 
                            'index-change positive' : 
                            'index-change negative';
                        changeElement.textContent = `${isPositive ? '+' : ''}${indexData.change_percent.toFixed(2)}%`;
                        console.log(`✏️ Set ${htmlId} change to:`, changeElement.textContent);
                    }
                });

                // Update last updated time
                const lastUpdateElement = document.getElementById('last-update');
                if (lastUpdateElement) {
                    const now = new Date();
                    lastUpdateElement.textContent = now.toLocaleTimeString();
                }

                console.log('✅ Market indices updated');
            } else {
                console.warn('⚠️ Invalid data structure:', data);
            }
        } catch (error) {
            console.error('❌ Error updating market indices:', error);
        }
    }

    /**
     * Update market sentiment gauge
     */
    async updateMarketSentiment() {
        try {
            const response = await fetch('/api/financial/market-sentiment');
            const data = await response.json();

            if (data.status === 'success' && data.data) {
                const sentiment = data.data;

                // Update sentiment bar
                const sentimentFill = document.getElementById('sentiment-fill');
                const sentimentLabel = document.getElementById('sentiment-label');

                if (sentimentFill) {
                    sentimentFill.style.width = `${sentiment.score}%`;

                    // Color based on sentiment
                    if (sentiment.score >= 60) {
                        sentimentFill.style.backgroundColor = '#10b981'; // Bullish - Green
                    } else if (sentiment.score <= 40) {
                        sentimentFill.style.backgroundColor = '#ef4444'; // Bearish - Red
                    } else {
                        sentimentFill.style.backgroundColor = '#f59e0b'; // Neutral - Orange
                    }
                }

                if (sentimentLabel) {
                    sentimentLabel.textContent = `${sentiment.score.toFixed(1)}% - ${sentiment.label} ${sentiment.emoji}`;
                }

                console.log(`✅ Market sentiment updated: ${sentiment.label}`);
            }
        } catch (error) {
            console.error('❌ Error updating market sentiment:', error);
        }
    }

    /**
     * Create Top Movers container if it doesn't exist
     */
    createTopMoversContainer() {
        // Find a good place to insert it - look for existing sections in overview tab
        const overviewTab = document.getElementById('overview-tab');
        
        if (overviewTab) {
            const section = document.createElement('div');
            section.className = 'section';
            section.innerHTML = `
                <h3>Top Market Movers 📈</h3>
                <div id="top-movers" style="display: grid; gap: 10px;">
                    <p style="text-align: center; opacity: 0.6;">Loading top market movers...</p>
                </div>
            `;
            
            // Insert at the end of overview tab
            overviewTab.appendChild(section);
            console.log('✅ Top Movers container created dynamically');
            return document.getElementById('top-movers');
        }
        
        console.warn('⚠️ Could not find overview tab for Top Movers container');
        return null;
    }

    /**
     * Update trading insights (top movers, etc.)
     */
    async updateTradingInsights() {
        try {
            const response = await fetch('/api/financial/top-stocks');
            const data = await response.json();

            if (data.status === 'success' && data.data) {
                const stocks = data.data.slice(0, 5); // Top 5
                let container = document.getElementById('top-movers');

                // Create container if it doesn't exist
                if (!container) {
                    container = this.createTopMoversContainer();
                }

                if (container) {
                    container.innerHTML = stocks.map(stock => `
                        <div class="stock-ticker">
                            <div class="stock-symbol">${stock.symbol}</div>
                            <div class="stock-price">$${stock.price.toFixed(2)}</div>
                            <div class="${stock.is_positive ? 'change-positive' : 'change-negative'}">
                                ${stock.is_positive ? '+' : ''}${stock.change_percent.toFixed(2)}%
                            </div>
                        </div>
                    `).join('');

                    console.log(`✅ Trading insights updated (${stocks.length} stocks)`);
                }
            }
        } catch (error) {
            console.error('❌ Error updating trading insights:', error);
        }
    }

    /**
     * Update Market Intelligence Dashboard
     */
    async updateMarketIntelligence() {
        try {
            const response = await fetch('/api/ai/market-intelligence');
            const data = await response.json();

            if (data.status === 'success' && data.data) {
                const intelligence = data.data;

                // Update Market Sentiment Display
                const sentimentDisplay = document.getElementById('market-sentiment-display');
                if (sentimentDisplay && intelligence.market_sentiment) {
                    const sentiment = intelligence.market_sentiment;
                    sentimentDisplay.innerHTML = `
                        <strong style="font-size: 1.1em;">${sentiment.interpretation}</strong>
                        <div style="font-size: 0.8em; opacity: 0.7;">
                            ${sentiment.bullish_stocks}/${sentiment.total_analyzed} bullish stocks
                        </div>
                    `;
                    sentimentDisplay.className = sentiment.interpretation === 'Bullish' ? 
                        'loading-text' : '';
                    sentimentDisplay.style.color = sentiment.interpretation === 'Bullish' ? '#10b981' : 
                        sentiment.interpretation === 'Bearish' ? '#ef4444' : '#f59e0b';
                }

                // Update Overall Market Display
                const overallDisplay = document.getElementById('overall-market-display');
                if (overallDisplay && intelligence.overview) {
                    overallDisplay.innerHTML = `
                        <strong>${intelligence.overview.symbols_analyzed} stocks analyzed</strong>
                        <div style="font-size: 0.8em; opacity: 0.7;">
                            Using ${intelligence.overview.ai_models_used.length} AI models
                        </div>
                    `;
                    overallDisplay.className = '';
                }

                // Update Risk Assessment Display
                const riskDisplay = document.getElementById('market-risk-display');
                const riskInterpretation = document.getElementById('risk-interpretation');
                if (riskDisplay && intelligence.risk_assessment) {
                    const risk = intelligence.risk_assessment;
                    riskDisplay.innerHTML = `
                        <strong style="font-size: 1.1em;">${risk.overall_risk} Risk</strong>
                    `;
                    riskDisplay.className = '';
                    riskDisplay.style.color = risk.overall_risk === 'High' ? '#ef4444' : 
                        risk.overall_risk === 'Medium' ? '#f59e0b' : '#10b981';

                    if (riskInterpretation) {
                        riskInterpretation.textContent = `${risk.high_risk_stocks}/${intelligence.market_sentiment.total_analyzed} stocks flagged as high risk (${risk.risk_percentage}%)`;
                    }
                }

                console.log('✅ Market intelligence updated');
            }
        } catch (error) {
            console.error('❌ Error updating market intelligence:', error);
        }
    }

    /**
     * Fetch real geopolitical risks from backend and render
     */
    async updateGeopoliticalRisks() {
        try {
            const response = await fetch('/api/financial/geopolitical-risks');
            const data = await response.json();

            if (data.status === 'success' && data.data) {
                const eventsEl = document.getElementById('geopolitical-events');
                const summaryEl = document.getElementById('geopolitical-summary');

                if (eventsEl) {
                    // Support both old format (events) and new format (factors)
                    const events = data.data.events || [];
                    const factors = data.data.factors || [];
                    
                    if (events.length > 0) {
                        // Old format
                        eventsEl.innerHTML = events.map(ev => `
                            <div style="margin-bottom:6px;">
                                <strong>${ev.region || ev.title}</strong>
                                <div style="font-size:0.85em; opacity:0.8;">${ev.summary || ev.description}</div>
                            </div>
                        `).join('');
                    } else if (factors.length > 0) {
                        // New format with factors
                        eventsEl.innerHTML = factors.map(factor => `
                            <div style="margin-bottom:6px;">
                                <strong>${factor.factor}</strong>
                                <div style="font-size:0.85em; opacity:0.8;">Severity: ${factor.severity}/10</div>
                                <div style="font-size:0.8em; opacity:0.7;">${factor.source}</div>
                            </div>
                        `).join('');
                    } else {
                        eventsEl.textContent = 'No significant geopolitical risks detected.';
                    }
                }

                if (summaryEl) {
                    // Support both formats
                    const assessment = data.data.overall_assessment || 
                                     (data.data.risk_level ? `Risk Level: ${data.data.risk_level} (Score: ${data.data.risk_score})` : null);
                    
                    if (assessment) {
                        summaryEl.textContent = assessment;
                    } else if (data.data.affected_sectors) {
                        summaryEl.textContent = `Affected sectors: ${data.data.affected_sectors.join(', ')}`;
                    }
                }

                console.log('✅ Geopolitical risks updated');
            }
        } catch (error) {
            console.error('❌ Error updating geopolitical risks:', error);
        }
    }

    /**
     * Start periodic updates
     * @param {number} intervalMs - Update interval in milliseconds
     */
    startPeriodicUpdates(intervalMs = 10000) {
        if (this.updateInterval) {
            this.stopPeriodicUpdates();
        }

        this.updateInterval = setInterval(async () => {
            await this.updateMarketIndices();
            await this.updateMarketSentiment();
            await this.updateTradingInsights();
            await this.updateGeopoliticalRisks();
        }, intervalMs);

        console.log(`✅ Periodic updates started (every ${intervalMs/1000}s)`);
    }

    /**
     * Stop periodic updates
     */
    stopPeriodicUpdates() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
            this.updateInterval = null;
            console.log('⏹️ Periodic updates stopped');
        }
    }

    /**
     * Manually refresh all market data
     */
    async refresh() {
        await this.updateMarketIndices();
        await this.updateMarketSentiment();
        await this.updateTradingInsights();
        await this.updateMarketIntelligence();
        console.log('🔄 Market data refreshed');
    }
}

// Export for use in main.js
window.MarketDataManager = MarketDataManager;