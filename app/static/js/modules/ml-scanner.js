/**
 * ML Scanner Module
 * Handles ML model status, hot stocks, market intelligence, and stock analysis
 */

class MLScanner {
    constructor() {
        this.mlAvailable = false;
        this.tfAvailable = false;
    }

    /**
     * Initialize ML scanner
     */
    async init() {
        await this.loadMLStatus();
        await this.loadHotStocks();
        await this.loadMarketIntelligence();

        console.log('✅ ML Scanner initialized');
    }

    /**
     * Load and display ML model status
     */
    async loadMLStatus() {
        try {
            const response = await fetch('/api/ai/status');
            const result = await response.json();

            if (result.status === 'success') {
                const data = result.data;
                
                // Update AI Models Status grid
                const statusGrid = document.getElementById('ai-models-status');
                if (statusGrid && data) {
                    let html = '';
                    Object.entries(data).forEach(([key, model]) => {
                        html += `
                            <div class="status-item">
                                <div class="status-icon">✅</div>
                                <div class="status-label">${model.name}</div>
                                <div class="status-name">${model.status}</div>
                            </div>
                        `;
                    });
                    statusGrid.innerHTML = html;
                }
                
                console.log('✅ AI Status loaded:', data);
            } else {
                console.error('AI status error:', result);
                this.updateMLStatusError();
            }
        } catch (error) {
            console.error('❌ Error loading ML status:', error);
            this.updateMLStatusError();
        }
    }

    /**
     * Update ML status elements with error state
     */
    updateMLStatusError() {
        const mlStatus = document.getElementById('ml-status');
        const nnStatus = document.getElementById('nn-status');
        if (mlStatus) mlStatus.textContent = '❌ Error';
        if (nnStatus) nnStatus.textContent = '❌ Error';
    }

    /**
     * Create Hot Alerts container if it doesn't exist
     */
    createHotAlertsContainer() {
        // Look for AI Intelligence grid to add hot alerts card
        const aiGrid = document.querySelector('.ai-grid');
        
        if (aiGrid) {
            const hotAlertsCard = document.createElement('div');
            hotAlertsCard.className = 'ai-card';
            hotAlertsCard.innerHTML = `
                <h4>AI Hot Stocks 🔥</h4>
                <div id="hot-alerts" style="margin: 8px 0;">
                    <p style="text-align: center; opacity: 0.6;">Scanning for hot opportunities...</p>
                </div>
            `;
            
            aiGrid.appendChild(hotAlertsCard);
            console.log('✅ Hot Alerts container created dynamically');
            return document.getElementById('hot-alerts');
        }
        
        console.warn('⚠️ Could not find AI grid for Hot Alerts container');
        return null;
    }

    /**
     * Load and display hot stocks (top ML predictions)
     */
    async loadHotStocks() {
        try {
            const response = await fetch('/api/scanner/hot-stocks?limit=10');
            const result = await response.json();

            let container = document.getElementById('hot-alerts');
            
            // Create container if it doesn't exist
            if (!container) {
                container = this.createHotAlertsContainer();
            }
            
            if (!container) return;

            if (result.status === 'success' && result.data.hot_stocks && result.data.hot_stocks.length > 0) {
                const hotStocks = result.data.hot_stocks;
                
                // Create grid layout with 3 stocks per row
                const hotStocksHtml = hotStocks.map(stock => {
                    const isPositive = stock.expected_return > 0;
                    const changeClass = isPositive ? 'change-positive' : 'change-negative';
                    const emoji = this.getRecommendationEmoji(stock.recommendation);
                    
                    return `
                        <div class="hot-stock-card">
                            <div class="stock-header">
                                <span class="stock-symbol">${stock.symbol}</span>
                                <div class="stock-price">$${stock.current_price?.toFixed(2) || 'N/A'}</div>
                            </div>
                            <div class="stock-return ${changeClass}">
                                ${isPositive ? '+' : ''}${stock.expected_return?.toFixed(2)}%
                            </div>
                            <div class="stock-details">
                                <div>${emoji} ${stock.recommendation}</div>
                                <div style="font-size: 0.8em; opacity: 0.8;">
                                    Score: ${stock.ml_score?.toFixed(1)} | ${(stock.confidence * 100)?.toFixed(0)}%
                                </div>
                            </div>
                        </div>
                    `;
                }).join('');
                
                const html = `
                    <div class="hot-stocks-grid">
                        ${hotStocksHtml}
                    </div>
                `;
                
                container.innerHTML = html;
                console.log(`✅ Loaded ${hotStocks.length} hot stocks`);
            } else {
                container.innerHTML = `
                    <div style="text-align: center; padding: 20px; opacity: 0.6;">
                        <p>No high-potential opportunities detected at this time.</p>
                        <p style="font-size: 0.85em; margin-top: 5px;">Continuously scanning markets...</p>
                    </div>
                `;
            }
        } catch (error) {
            console.error('❌ Error loading hot stocks:', error);
            const container = document.getElementById('hot-alerts');
            if (container) {
                container.innerHTML = `
                    <div style="text-align: center; padding: 20px; color: #ef4444;">
                        <p>⚠️ Error loading hot stocks</p>
                        <p style="font-size: 0.85em; margin-top: 5px;">${error.message}</p>
                    </div>
                `;
            }
        }
    }

    /**
     * Get emoji for recommendation type
     */
    getRecommendationEmoji(recommendation) {
        switch (recommendation) {
            case 'STRONG BUY': return '🚀';
            case 'BUY': return '📈';
            case 'STRONG SELL': return '📉';
            case 'SELL': return '⬇️';
            default: return '➡️';
        }
    }

    /**
     * Ensure AI Analysis tab elements exist
     */
    ensureAIAnalysisElements() {
        if (!document.getElementById('comprehensive-results')) {
            this.createAIAnalysisSection();
        }
    }

    /**
     * Create AI Analysis section in AI tab
     */
    createAIAnalysisSection() {
        const aiTab = document.getElementById('ai-tab');
        
        if (aiTab) {
            // Clear existing content and add comprehensive structure
            aiTab.innerHTML = `
                <div class="section">
                    <h3>Comprehensive AI Analysis 🤖</h3>
                    <div id="comprehensive-results" style="margin-bottom: 20px;">
                        <p style="text-align: center; opacity: 0.6;">Select a stock symbol to see comprehensive AI analysis</p>
                    </div>
                    
                    <h3>Neural Network Predictions 🧠</h3>
                    <div id="neural-network-results" style="margin-bottom: 20px;">
                        <p style="text-align: center; opacity: 0.6;">Neural network predictions will appear here</p>
                    </div>
                    
                    <h3>Time Series Analysis 📊</h3>
                    <div id="time-series-results">
                        <p style="text-align: center; opacity: 0.6;">Time series analysis will appear here</p>
                    </div>
                </div>
            `;
            
            console.log('✅ AI Analysis section created dynamically');
        }
    }

    /**
     * Ensure ML Dashboard elements exist
     */
    ensureMLDashboardElements() {
        // Check if ML dashboard section exists, if not create it
        if (!document.getElementById('market-sentiment-value')) {
            this.createMLDashboardSection();
        }
    }

    /**
     * Create ML Dashboard section with all required elements
     */
    createMLDashboardSection() {
        const overviewTab = document.getElementById('overview-tab');
        
        if (overviewTab) {
            const section = document.createElement('div');
            section.className = 'section';
            section.innerHTML = `
                <h3>ML Intelligence Dashboard 🧠</h3>
                <div class="metrics-grid">
                    <div class="metric-item">
                        <div class="metric-value" id="market-sentiment-value">Loading...</div>
                        <div class="metric-label" id="market-sentiment-label">Market Sentiment</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-value" id="risk-level-value">Loading...</div>
                        <div class="metric-label" id="risk-level-label">Risk Level</div>
                    </div>
                </div>
                <div id="ai-recommendations" style="margin-top: 15px;">
                    <h4 style="color: #ffd700; margin-bottom: 10px;">AI Recommendations</h4>
                    <p style="text-align: center; opacity: 0.6;">Analyzing market conditions...</p>
                </div>
            `;
            
            overviewTab.appendChild(section);
            console.log('✅ ML Dashboard section created dynamically');
        }
    }

    /**
     * Load and display market intelligence
     */
    async loadMarketIntelligence() {
        try {
            // Ensure ML dashboard elements exist
            this.ensureMLDashboardElements();
            
            const response = await fetch('/api/ai/market-intelligence');
            const result = await response.json();

            if (result.status === 'success') {
                const data = result.data;

                // Update market sentiment
                const sentimentValue = document.getElementById('market-sentiment-value');
                const sentimentLabel = document.getElementById('market-sentiment-label');
                if (sentimentValue) {
                    sentimentValue.textContent = data.market_sentiment?.interpretation || 'N/A';
                }
                if (sentimentLabel) {
                    sentimentLabel.textContent = `${data.market_sentiment?.overall_score || 0} score`;
                }

                // Update risk assessment
                const riskValue = document.getElementById('risk-level-value');
                const riskLabel = document.getElementById('risk-level-label');
                if (riskValue) {
                    riskValue.textContent = data.risk_assessment?.overall_risk || 'N/A';
                }
                if (riskLabel) {
                    riskLabel.textContent = `${data.risk_assessment?.risk_percentage || 0}% high risk`;
                }

                // Update AI recommendations
                const recommendationsContainer = document.getElementById('ai-recommendations');
                if (recommendationsContainer && data.recommendations) {
                    const html = data.recommendations.map(rec => `
                        <div class="alert-card ${rec.action.toLowerCase()}">
                            <div class="alert-title">📊 ${rec.symbol}</div>
                            <div class="alert-meta">
                                Action: ${rec.action} | 
                                Confidence: ${(rec.confidence * 100).toFixed(1)}%
                            </div>
                            <div class="alert-description">${rec.reasoning}</div>
                        </div>
                    `).join('');
                    recommendationsContainer.innerHTML = html || '<p>No recommendations available</p>';
                }

                console.log('✅ Market intelligence loaded');
            } else {
                console.error('Market intelligence error:', result);
                this.updateMarketIntelligenceError();
            }
        } catch (error) {
            console.error('❌ Error loading market intelligence:', error);
            this.updateMarketIntelligenceError();
        }
    }

    /**
     * Update market intelligence elements with error state
     */
    updateMarketIntelligenceError() {
        const sentimentValue = document.getElementById('market-sentiment-value');
        const riskValue = document.getElementById('risk-level-value');
        if (sentimentValue) sentimentValue.textContent = 'Error';
        if (riskValue) riskValue.textContent = 'Error';
    }

    /**
     * Analyze a specific stock using AI
     */
    async analyzeStock() {
        const inputElement = document.getElementById('ai-symbol-input');
        const resultsElement = document.getElementById('ai-analysis-results');
        
        if (!inputElement) {
            console.error('❌ Stock symbol input not found');
            return;
        }

        const symbol = inputElement.value.toUpperCase().trim();
        if (!symbol) {
            alert('Please enter a stock symbol (e.g., AAPL)');
            return;
        }

        console.log(`🔍 Analyzing stock: ${symbol}`);

        // Show loading state
        if (resultsElement) {
            resultsElement.innerHTML = '<p style="text-align: center; opacity: 0.6; padding: 20px;">🔄 Analyzing ' + symbol + '...</p>';
        }

        // Load analysis from API
        try {
            const response = await fetch(`/api/ai/comprehensive-analysis/${symbol}`);
            const result = await response.json();

            if (result.status === 'success' && resultsElement) {
                const data = result.data;
                
                // Display results
                resultsElement.innerHTML = `
                    <div style="background: rgba(255,255,255,0.05); padding: 20px; border-radius: 8px; margin-top: 15px;">
                        <h4 style="color: #ffd700; margin-bottom: 15px;">Analysis Results for ${data.symbol}</h4>
                        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;">
                            <div style="background: rgba(59, 130, 246, 0.1); padding: 15px; border-radius: 8px;">
                                <div style="font-size: 1.5em; font-weight: bold; color: #3b82f6;">$${data.price?.toFixed(2) || 'N/A'}</div>
                                <div style="font-size: 0.85em; opacity: 0.8;">Current Price</div>
                                <div style="font-size: 0.8em; color: ${data.change_percent > 0 ? '#10b981' : '#ef4444'}; margin-top: 5px;">
                                    ${data.change_percent > 0 ? '+' : ''}${data.change_percent?.toFixed(2) || 0}%
                                </div>
                            </div>
                            <div style="background: rgba(16, 185, 129, 0.1); padding: 15px; border-radius: 8px;">
                                <div style="font-size: 1.5em; font-weight: bold; color: #10b981;">${data.recommendation || 'HOLD'}</div>
                                <div style="font-size: 0.85em; opacity: 0.8;">Recommendation</div>
                                <div style="font-size: 0.8em; margin-top: 5px;">
                                    Confidence: ${((data.confidence || 0) * 100).toFixed(0)}%
                                </div>
                            </div>
                            <div style="background: rgba(139, 92, 246, 0.1); padding: 15px; border-radius: 8px;">
                                <div style="font-size: 1.5em; font-weight: bold; color: #8b5cf6;">${data.sentiment_label || 'Neutral'}</div>
                                <div style="font-size: 0.85em; opacity: 0.8;">Sentiment</div>
                                <div style="font-size: 0.8em; margin-top: 5px;">
                                    Score: ${(data.sentiment_score * 100).toFixed(0)}%
                                </div>
                            </div>
                        </div>
                    </div>
                `;
                
                console.log('✅ Analysis displayed successfully');
            } else {
                throw new Error('Failed to get analysis');
            }
        } catch (error) {
            console.error('❌ Error analyzing stock:', error);
            if (resultsElement) {
                resultsElement.innerHTML = `<p style="text-align: center; color: #ef4444; padding: 20px;">❌ Error analyzing ${symbol}. Please try again.</p>`;
            }
        }
    }

    /**
     * Load comprehensive analysis for a stock
     */
    async loadComprehensiveAnalysis(symbol) {
        try {
            // Ensure AI Analysis elements exist
            this.ensureAIAnalysisElements();
            
            const response = await fetch(`/api/ai/comprehensive-analysis/${symbol}`);
            const result = await response.json();

            const resultsElement = document.getElementById('comprehensive-results');
            if (!resultsElement) return;

            if (result.status === 'success') {
                const data = result.data;
                const html = `
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;">
                        <div class="metric-card">
                            <div class="metric-value">$${data.predictions?.price?.predicted || 'N/A'}</div>
                            <div class="metric-label">Price Prediction</div>
                            <div style="font-size: 0.8em; color: #ccc;">
                                ${data.predictions?.price?.change_percent || 0}% change
                            </div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">${data.predictions?.direction?.interpretation || 'Neutral'}</div>
                            <div class="metric-label">Direction</div>
                            <div style="font-size: 0.8em; color: #ccc;">
                                ${((data.predictions?.direction?.probability || 0) * 100).toFixed(1)}% confidence
                            </div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">${data.trading_signal?.action || 'HOLD'}</div>
                            <div class="metric-label">Trading Signal</div>
                            <div style="font-size: 0.8em; color: #ccc;">
                                ${((data.trading_signal?.confidence || 0) * 100).toFixed(1)}% confidence
                            </div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">${data.predictions?.volatility?.risk_level || 'Medium'}</div>
                            <div class="metric-label">Risk Level</div>
                            <div style="font-size: 0.8em; color: #ccc;">
                                ${data.predictions?.volatility?.predicted || 0}% volatility
                            </div>
                        </div>
                    </div>
                `;
                resultsElement.innerHTML = html;
                console.log(`✅ Comprehensive analysis loaded for ${symbol}`);
            } else {
                resultsElement.innerHTML = `<p style="color: #ff6b6b;">Error loading analysis: ${result.message || 'Unknown error'}</p>`;
            }
        } catch (error) {
            console.error('Error loading comprehensive analysis:', error);
            const resultsElement = document.getElementById('comprehensive-results');
            if (resultsElement) {
                resultsElement.innerHTML = `<p style="color: #ff6b6b;">Error: ${error.message}</p>`;
            }
        }
    }

    /**
     * Load neural network analysis for a stock
     */
    async loadNeuralNetworkAnalysis(symbol) {
        try {
            // Ensure AI Analysis elements exist
            this.ensureAIAnalysisElements();
            
            const response = await fetch(`/api/ai/neural-network-prediction/${symbol}`);
            const result = await response.json();

            const resultsElement = document.getElementById('neural-network-results');
            if (!resultsElement) return;

            if (result.status === 'success') {
                const data = result.data;
                const html = `
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 15px;">
                        <div class="metric-card">
                            <div class="metric-value">$${data.ensemble_prediction?.prediction || 'N/A'}</div>
                            <div class="metric-label">Ensemble Prediction</div>
                            <div style="font-size: 0.8em; color: #ccc;">
                                ${((data.ensemble_prediction?.confidence || 0) * 100).toFixed(1)}% confidence
                            </div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">${data.individual_models?.lstm?.prediction || 'N/A'}</div>
                            <div class="metric-label">LSTM Model</div>
                            <div style="font-size: 0.8em; color: #ccc;">
                                Weight: ${((data.individual_models?.lstm?.weight || 0) * 100).toFixed(0)}%
                            </div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">${data.individual_models?.transformer?.prediction || 'N/A'}</div>
                            <div class="metric-label">Transformer</div>
                            <div style="font-size: 0.8em; color: #ccc;">
                                Weight: ${((data.individual_models?.transformer?.weight || 0) * 100).toFixed(0)}%
                            </div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">${data.individual_models?.cnn?.prediction || 'N/A'}</div>
                            <div class="metric-label">CNN Model</div>
                            <div style="font-size: 0.8em; color: #ccc;">
                                Weight: ${((data.individual_models?.cnn?.weight || 0) * 100).toFixed(0)}%
                            </div>
                        </div>
                    </div>
                    <div style="margin-top: 15px;">
                        <p><strong>Model Agreement:</strong> ${((data.model_agreement?.agreement_score || 0) * 100).toFixed(1)}%</p>
                        <p><strong>Prediction Range:</strong> $${data.ensemble_prediction?.prediction_range?.[0] || 'N/A'} - $${data.ensemble_prediction?.prediction_range?.[1] || 'N/A'}</p>
                    </div>
                `;
                resultsElement.innerHTML = html;
                console.log(`✅ Neural network analysis loaded for ${symbol}`);
            } else {
                resultsElement.innerHTML = `<p style="color: #ff6b6b;">Error loading analysis: ${result.message || 'Unknown error'}</p>`;
            }
        } catch (error) {
            console.error('Error loading neural network analysis:', error);
            const resultsElement = document.getElementById('neural-network-results');
            if (resultsElement) {
                resultsElement.innerHTML = `<p style="color: #ff6b6b;">Error: ${error.message}</p>`;
            }
        }
    }

    /**
     * Load time series analysis for a stock
     */
    async loadTimeSeriesAnalysis(symbol) {
        try {
            // Ensure AI Analysis elements exist
            this.ensureAIAnalysisElements();
            
            const response = await fetch(`/api/ai/time-series-analysis/${symbol}`);
            const result = await response.json();

            const resultsElement = document.getElementById('time-series-results');
            if (!resultsElement) return;

            if (result.status === 'success') {
                const data = result.data;
                const html = `
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px;">
                        <div class="metric-card">
                            <div class="metric-value">${data.trends?.short_term || 'N/A'}</div>
                            <div class="metric-label">Short Term</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">${data.trends?.medium_term || 'N/A'}</div>
                            <div class="metric-label">Medium Term</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">${data.trends?.long_term || 'N/A'}</div>
                            <div class="metric-label">Long Term</div>
                        </div>
                    </div>
                    <div style="margin-top: 15px;">
                        <h5>Support & Resistance:</h5>
                        <p><strong>Support:</strong> $${data.support_resistance?.support || 'N/A'}</p>
                        <p><strong>Resistance:</strong> $${data.support_resistance?.resistance || 'N/A'}</p>
                        <p><strong>Current:</strong> $${data.support_resistance?.current || 'N/A'}</p>
                    </div>
                `;
                resultsElement.innerHTML = html;
                console.log(`✅ Time series analysis loaded for ${symbol}`);
            } else {
                resultsElement.innerHTML = `<p style="color: #ff6b6b;">Error loading analysis: ${result.message || 'Unknown error'}</p>`;
            }
        } catch (error) {
            console.error('Error loading time series analysis:', error);
            const resultsElement = document.getElementById('time-series-results');
            if (resultsElement) {
                resultsElement.innerHTML = `<p style="color: #ff6b6b;">Error: ${error.message}</p>`;
            }
        }
    }

    /**
     * Refresh all ML data
     */
    async refresh() {
        await this.loadMLStatus();
        await this.loadHotStocks();
        await this.loadMarketIntelligence();
        console.log('🔄 ML Scanner data refreshed');
    }
}

// Export for use in main.js
window.MLScanner = MLScanner;