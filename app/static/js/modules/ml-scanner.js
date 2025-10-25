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

            // Backend returns shape: { status: "operational", components: { ... }, performance: {...} }
            const hasComponents = result && typeof result === 'object' && result.components && typeof result.components === 'object';

            if (hasComponents) {
                const components = result.components;

                // Update AI Models Status grid
                const statusGrid = document.getElementById('ai-models-status');
                if (statusGrid) {
                    const prettyName = (key) => ({
                        ml_trainer: 'ML Trainer',
                        news_analyzer: 'News Analyzer',
                        social_analyzer: 'Social Analyzer',
                        ai_models: 'AI Models'
                    }[key] || key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()));

                    let html = '';
                    Object.entries(components).forEach(([key, comp]) => {
                        const available = !!comp.available || comp.status === 'ready' || comp.status === 'active' || comp.status === 'Operational';
                        const icon = available ? '✅' : '❌';
                        const label = prettyName(key);
                        const status = comp.status || (available ? 'ready' : 'unavailable');

                        html += `
                            <div class="status-item">
                                <div class="status-icon">${icon}</div>
                                <div class="status-label">${label}</div>
                                <div class="status-name">${status}</div>
                            </div>
                        `;
                    });
                    statusGrid.innerHTML = html;
                }

                // Optional: reflect availability flags
                this.mlAvailable = Object.values(components).some(c => (c?.available === true) || c?.status === 'ready');
                this.tfAvailable = false; // TensorFlow disabled in backend

                console.log('✅ AI Status loaded (components):', components);
            } else {
                console.error('AI status error or unexpected shape:', result);
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
                    const isPositive = (stock.expected_return || 0) > 0;
                    const changeClass = isPositive ? 'change-positive' : 'change-negative';
                    const emoji = this.getRecommendationEmoji(stock.recommendation);
                    const confPct = typeof stock.confidence === 'number' ? `${(stock.confidence * 100).toFixed(0)}%` : '--';
                    const sl = stock?.risk_7d?.stop_loss;
                    const tp = stock?.risk_7d?.take_profit;
                    const rr = stock?.risk_7d?.rr;
                    const basis = stock?.risk_7d?.basis;
                    const riskLine = (sl && tp) ? `<div class="sltp-line" title="Basis: ${basis || '—'} | RR ${rr || '—'}">SL/TP → $${Number(sl).toFixed(2)} / $${Number(tp).toFixed(2)}</div>` : '';
                    
                    return `
                        <div class="hot-stock-card">
                            <div class="stock-header">
                                <span class="stock-symbol">${stock.symbol}</span>
                                <div class="stock-price">$${(stock.current_price ?? 0).toFixed(2)}</div>
                            </div>
                            <div class="stock-return ${changeClass}">
                                ${isPositive ? '+' : ''}${(stock.expected_return ?? 0).toFixed(2)}%
                            </div>
                            <div class="stock-details">
                                <div>${emoji} ${stock.recommendation || 'HOLD'} • ${confPct}</div>
                                ${riskLine}
                                <div style="font-size: 0.8em; opacity: 0.8;">
                                    Score: ${stock.ml_score?.toFixed(1) ?? '--'}
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
        const inputElement = document.getElementById('ai-stock-input');
        if (!inputElement) {
            console.error('❌ Stock symbol input not found (#ai-stock-input)');
            return;
        }

        const symbol = inputElement.value.toUpperCase().trim();
        if (!symbol) {
            alert('Please enter a stock symbol (e.g., AAPL)');
            return;
        }

        console.log(`🔍 Analyzing stock: ${symbol}`);

        // Show loading state in all AI sections
        const selectedInfo = document.getElementById('selected-stock-analysis');
        const compEl = document.getElementById('ai-comprehensive-results');
        const techEl = document.getElementById('ai-technical-results');
        const riskEl = document.getElementById('ai-risk-factors');
        if (selectedInfo) selectedInfo.innerHTML = `<p style="text-align: center;">🤖 Analyzing ${symbol}...</p>`;
        if (compEl) compEl.innerHTML = `<p style="text-align: center;">🔄 Running AI analysis...</p>`;
        if (techEl) techEl.innerHTML = `<p style="text-align: center;">📊 Processing technical data...</p>`;
        if (riskEl) riskEl.innerHTML = `<p style="text-align: center;">⚠️ Calculating risk factors...</p>`;

        try {
            const response = await fetch(`/api/ai/comprehensive-analysis/${symbol}`);
            const data = await response.json();

            if (data.status === 'success') {
                const analysis = data.analysis || {};
                const prediction = data.prediction || {};
                const meta = data.ai_metadata || {};

                // Stock info + metadata
                if (selectedInfo) {
                    selectedInfo.innerHTML = `
                        <div style="background: rgba(59, 130, 246, 0.1); border-radius: 6px; padding: 10px;">
                            <strong style="color: #3b82f6;">${symbol}</strong><br>
                            <span style="color: #10b981;">Current Price: $${data.current_price ?? 'N/A'}</span><br>
                            <small>🤖 AI Model: ${meta.model || 'N/A'}</small><br>
                            <small>📊 Sources: ${meta.search_results_count || 0} | Cost: $${(meta.cost?.total_cost || 0).toFixed?.(4) || 0}</small><br>
                            <small>📝 Response: ${meta.raw_response_length || 0} chars | ${new Date().toLocaleTimeString()}</small>
                            ${meta.raw_response_preview ? `
                            <details style="margin-top: 8px;">
                                <summary style="cursor: pointer; color: #3b82f6; font-size: 0.8em;">📄 View AI Response Preview</summary>
                                <pre style="white-space: pre-wrap; margin-top: 8px; background: rgba(0,0,0,0.3); padding: 8px; border-radius: 4px; font-size: 0.75em; max-height: 200px; overflow-y: auto;">${meta.raw_response_preview}</pre>
                            </details>` : ''}
                        </div>
                    `;
                }

                // Comprehensive analysis block
                if (compEl) {
                    const dir = (prediction.direction || 'neutral').toLowerCase();
                    const dirColor = dir === 'up' ? '#10b981' : dir === 'down' ? '#ef4444' : '#f59e0b';
                    const confPct = typeof prediction.confidence === 'number' ? `${(prediction.confidence * 100).toFixed(1)}%` : '--';
                    compEl.innerHTML = `
                        <div class="prediction-item">
                            <span class="prediction-label">Overall Prediction:</span>
                            <span class="prediction-value" style="color: ${dirColor};">${(prediction.direction || 'NEUTRAL').toString().toUpperCase()}</span>
                        </div>
                        <div class="prediction-item">
                            <span class="prediction-label">Confidence Level:</span>
                            <span class="prediction-value">${confPct}</span>
                        </div>
                        <div class="prediction-item">
                            <span class="prediction-label">Target Price:</span>
                            <span class="prediction-value">$${prediction.target_price ?? '--'}</span>
                        </div>
                    `;
                }

                // Technical analysis block
                if (techEl) {
                    const tech = analysis.technical || {};
                    const trend = (tech.trend || 'neutral').toLowerCase();
                    const trendColor = trend === 'bullish' ? '#10b981' : trend === 'bearish' ? '#ef4444' : '#f59e0b';
                    techEl.innerHTML = `
                        <div class="timeseries-metric">
                            <span class="metric-name">Trend Direction:</span>
                            <span class="metric-value" style="color: ${trendColor};">${tech.trend || 'neutral'}</span>
                        </div>
                        <div class="timeseries-metric">
                            <span class="metric-name">Support Level:</span>
                            <span class="metric-value">$${tech.support_level ?? '--'}</span>
                        </div>
                        <div class="timeseries-metric">
                            <span class="metric-name">Resistance Level:</span>
                            <span class="metric-value">$${tech.resistance_level ?? '--'}</span>
                        </div>
                        <div class="timeseries-metric">
                            <span class="metric-name">RSI/MACD Signal:</span>
                            <span class="metric-value">${tech.macd_signal || '--'}</span>
                        </div>
                    `;
                }

                // Risk assessment
                if (riskEl) {
                    const risk = analysis.risk_assessment || {};
                    const riskScore = typeof risk.risk_score === 'number' ? (risk.risk_score * 100).toFixed(0) : '--';
                    riskEl.innerHTML = `
                        <div style="font-size: 0.9em;">
                            <strong>Risk Level:</strong> ${risk.volatility || 'N/A'}<br>
                            <strong>Beta:</strong> ${risk.beta ?? 'N/A'}<br>
                            <strong>Liquidity:</strong> ${risk.liquidity || 'N/A'}<br>
                            <strong>Overall Score:</strong> ${riskScore}%
                        </div>
                    `;
                }

                console.log('✅ Analysis displayed successfully');
            } else {
                if (selectedInfo) selectedInfo.innerHTML = '<p style="color: #ef4444; text-align: center;">❌ Analysis failed</p>';
                if (compEl) compEl.innerHTML = '<p style="color: #ef4444; text-align: center;">❌ Failed to get AI analysis</p>';
                if (techEl) techEl.innerHTML = '<p style="color: #ef4444; text-align: center;">❌ Technical analysis unavailable</p>';
                if (riskEl) riskEl.innerHTML = '<p style="color: #ef4444; text-align: center;">❌ Risk analysis failed</p>';
            }
        } catch (error) {
            console.error('❌ Error analyzing stock:', error);
            if (selectedInfo) selectedInfo.innerHTML = `<p style="color: #ef4444; text-align: center;">❌ Error: ${error.message}</p>`;
            if (compEl) compEl.innerHTML = '<p style="color: #ef4444; text-align: center;">❌ Connection error</p>';
            if (techEl) techEl.innerHTML = '<p style="color: #ef4444; text-align: center;">❌ Network error</p>';
            if (riskEl) riskEl.innerHTML = '<p style="color: #ef4444; text-align: center;">❌ Service unavailable</p>';
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
            if (!response.ok) {
                throw new Error(`API error ${response.status}`);
            }
            const result = await response.json();

            const resultsElement = document.getElementById('comprehensive-results');
            if (!resultsElement) return;

            if (result.status === 'success') {
                const pred = result.prediction || {};
                const direction = pred.direction || 'Neutral';
                const directionConf = typeof pred.confidence === 'number' ? `${(pred.confidence * 100).toFixed(1)}%` : 'N/A';
                const targetPrice = typeof pred.target_price === 'number' ? `$${pred.target_price.toFixed(2)}` : 'N/A';
                const riskLevel = (result.analysis && result.analysis.volatility && result.analysis.volatility.risk_level) || 'N/A';
                const volatility = (result.analysis && result.analysis.volatility && typeof result.analysis.volatility.predicted === 'number') ? `${result.analysis.volatility.predicted}%` : 'N/A';

                const html = `
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;">
                        <div class="metric-card">
                            <div class="metric-value">${targetPrice}</div>
                            <div class="metric-label">Target Price</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">${direction}</div>
                            <div class="metric-label">Direction</div>
                            <div style="font-size: 0.8em; color: #ccc;">${directionConf} confidence</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">${riskLevel}</div>
                            <div class="metric-label">Risk Level</div>
                            <div style="font-size: 0.8em; color: #ccc;">Volatility: ${volatility}</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">$${(result.current_price ?? 0).toFixed(2)}</div>
                            <div class="metric-label">Current Price</div>
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