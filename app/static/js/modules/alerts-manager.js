/**
 * Alerts Manager Module
 * Handles active alerts, articles loading, filtering, and display
 */

class AlertsManager {
    constructor() {
        this.currentFilter = 'all';
        this.allArticles = [];
    }

    /**
     * Initialize alerts manager
     */
    async init() {
        await this.loadStats();
        await this.loadAlerts();
        await this.loadArticles();

        console.log('✅ Alerts Manager initialized');
    }

    /**
     * Load and display statistics
     */
    async loadStats() {
        try {
            const response = await fetch('/api/stats');
            const data = await response.json();

            // Support both legacy {metrics: {...}} and new {status:'success', data:{...}}
            const metrics = data.metrics || (data.data ? {
                articles_total: data.data.articles?.total_processed,
                articles_last_24h: data.data.articles?.today,
                articles_high_score: data.data.alerts?.active,
                articles_medium_score: data.data.alerts?.resolved,
                sources_monitored: data.data.monitoring?.feeds_monitored
            } : null);

            if (metrics) {
                this.updateStatElement('total-articles', metrics.articles_total ?? 0);
                this.updateStatElement('articles-24h', metrics.articles_last_24h ?? 0);
                this.updateStatElement('high-priority-count', metrics.articles_high_score ?? 0);
                this.updateStatElement('medium-priority-count', metrics.articles_medium_score ?? 0);
                this.updateStatElement('sources-count', metrics.sources_monitored ?? 0);

                console.log('✅ Stats updated');
            }
        } catch (error) {
            console.error('❌ Failed to load stats:', error);
            this.updateStatElement('total-articles', 'ERROR');
            this.updateStatElement('articles-24h', 'ERROR');
            this.updateStatElement('high-priority-count', 'ERROR');
            this.updateStatElement('medium-priority-count', 'ERROR');
            this.updateStatElement('sources-count', 'ERROR');
        }
    }

    /**
     * Helper to update stat elements
     */
    updateStatElement(id, value) {
        const element = document.getElementById(id);
        if (element) {
            element.textContent = value;
        }
    }

    /**
     * Create Alerts container in Articles tab
     */
    createAlertsContainer() {
        const articlesTab = document.getElementById('articles-tab');
        
        if (articlesTab) {
            // Check if section already exists
            let section = articlesTab.querySelector('.section');
            if (!section) {
                section = document.createElement('div');
                section.className = 'section';
                articlesTab.appendChild(section);
            }
            
            // Add alerts section
            const alertsSection = document.createElement('div');
            alertsSection.innerHTML = `
                <h3>Active Alerts 🚨</h3>
                <div id="alerts-container" style="margin-bottom: 20px;">
                    <p style="text-align: center; opacity: 0.6;">Loading active alerts...</p>
                </div>
            `;
            
            section.appendChild(alertsSection);
            console.log('✅ Alerts container created dynamically');
            return document.getElementById('alerts-container');
        }
        
        console.warn('⚠️ Could not find articles tab for alerts container');
        return null;
    }

    /**
     * Create Articles container in Articles tab
     */
    createArticlesContainer() {
        const articlesTab = document.getElementById('articles-tab');
        
        if (articlesTab) {
            // Check if section already exists
            let section = articlesTab.querySelector('.section');
            if (!section) {
                section = document.createElement('div');
                section.className = 'section';
                articlesTab.appendChild(section);
            }
            
            // Add articles section
            const articlesSection = document.createElement('div');
            articlesSection.innerHTML = `
                <h3>Recent Articles 📄</h3>
                <div id="articles-container">
                    <p style="text-align: center; opacity: 0.6;">Loading recent articles...</p>
                </div>
            `;
            
            section.appendChild(articlesSection);
            console.log('✅ Articles container created dynamically');
            return document.getElementById('articles-container');
        }
        
        console.warn('⚠️ Could not find articles tab for articles container');
        return null;
    }

    /**
     * Load and display active alerts
     */
    async loadAlerts() {
        try {
            const response = await fetch('/api/alerts/active');
            const data = await response.json();

            let container = document.getElementById('alerts-container');
            
            // Create container if it doesn't exist
            if (!container) {
                container = this.createAlertsContainer();
            }
            
            if (!container) return;

            if (data.error || !data.alerts || data.alerts.length === 0) {
                container.innerHTML = '<p style="color: #a0aec0;">No active alerts at the moment.</p>';
                return;
            }

            const html = data.alerts.map(alert => {
                const severityClass = alert.severity || 'low';
                const timestamp = new Date(alert.timestamp).toLocaleString();
                
                return `
                    <div class="alert-card ${severityClass}">
                        <div class="alert-title">
                            <span class="alert-badge alert-${severityClass}">${severityClass.toUpperCase()}</span>
                            ${alert.title}
                        </div>
                        <div class="alert-meta">
                            📅 ${timestamp} | 📊 Score: ${alert.score ? alert.score.toFixed(2) : 'N/A'}
                        </div>
                        ${alert.description ? `<div class="alert-description">${alert.description}</div>` : ''}
                        ${alert.url ? `<a href="${alert.url}" target="_blank" class="alert-link">View Details →</a>` : ''}
                    </div>
                `;
            }).join('');

            container.innerHTML = html;
            console.log(`✅ Loaded ${data.alerts.length} alerts`);
        } catch (error) {
            console.error('❌ Failed to load alerts:', error);
        }
    }

    /**
     * Load articles with filtering
     */
    async loadArticles() {
        try {
            const response = await fetch('/api/articles');
            const data = await response.json();

            let container = document.getElementById('articles-container');
            
            // Create container if it doesn't exist
            if (!container) {
                container = this.createArticlesContainer();
            }
            
            if (!container) return;

            if (data.error || !data.articles) {
                container.innerHTML = `<p style="color: #ff6b6b;">Error loading articles: ${data.error || 'Unknown error'}</p>`;
                return;
            }

            if (data.articles.length === 0) {
                container.innerHTML = '<p style="color: #a0aec0;">No articles found. Check back soon!</p>';
                return;
            }

            this.allArticles = data.articles;
            this.displayArticles(this.allArticles);

            console.log(`✅ Loaded ${data.articles.length} articles`);
        } catch (error) {
            console.error('❌ Failed to load articles:', error);
            const container = document.getElementById('articles-container');
            if (container) {
                container.innerHTML = `<p style="color: #ff6b6b;">Failed to load articles: ${error.message}</p>`;
            }
        }
    }

    /**
     * Filter articles by priority
     */
    filterArticles(filterType, buttonElement) {
        this.currentFilter = filterType;

        // Update active button state
        document.querySelectorAll('.filter-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        if (buttonElement) {
            buttonElement.classList.add('active');
        }

        // Filter articles
        let filtered = this.allArticles;

        if (filterType === 'high') {
            filtered = this.allArticles.filter(a => a.score >= 0.7);
        } else if (filterType === 'medium') {
            filtered = this.allArticles.filter(a => a.score >= 0.4 && a.score < 0.7);
        } else if (filterType === 'low') {
            filtered = this.allArticles.filter(a => a.score < 0.4);
        }

        this.displayArticles(filtered);

        console.log(`🔍 Filtered to ${filterType}: ${filtered.length} articles`);
    }

    /**
     * Display articles in the container
     */
    displayArticles(articles) {
        const container = document.getElementById('articles-container');
        if (!container) return;

        if (articles.length === 0) {
            container.innerHTML = `<p style="color: #a0aec0;">No articles match the current filter (${this.currentFilter}).</p>`;
            return;
        }

        let html = `<p style="margin-bottom: 15px; color: #ffd700;">Showing ${articles.length} articles (${this.currentFilter}):</p>`;

        articles.forEach(article => {
            const score = article.score || 0;
            const scoreColor = score >= 0.7 ? '#4CAF50' : score >= 0.4 ? '#ffd700' : '#ff6b6b';
            const timestamp = new Date(article.published || article.timestamp).toLocaleString();

            // AI status indicator
            let aiStatusColor = '#999';
            let aiStatusText = '🤖⛔ AI: Skipped';
            
            if (article.llm_checked) {
                if (article.llm_relevant) {
                    aiStatusColor = '#4CAF50';
                    aiStatusText = '🤖✅ AI: Relevant';
                } else {
                    aiStatusColor = '#ff6b6b';
                    aiStatusText = '🤖❌ AI: Not Relevant';
                }
            } else if (score >= 0.4) {
                aiStatusColor = '#FF9800';
                aiStatusText = '🤖⚠️ AI: Processing...';
            }

            html += `
                <div class="article-card">
                    <h4>${article.title}</h4>
                    <p class="article-meta">
                        Score: <span style="color: ${scoreColor}">${score.toFixed(2)}</span> | 
                        📅 ${timestamp}
                    </p>
                    <p class="article-meta" style="color: ${aiStatusColor}; font-weight: bold;">
                        ${aiStatusText}
                    </p>
                    ${article.llm_summary ? `
                        <p style="margin: 10px 0; padding: 10px; background: rgba(255,215,0,0.1); border-left: 3px solid #ffd700; border-radius: 4px;">
                            <strong>🤖 AI Summary:</strong> ${article.llm_summary}
                        </p>
                    ` : ''}
                    ${article.llm_tags && article.llm_tags.length > 0 ? `
                        <p style="margin: 5px 0;">
                            <strong>🏷️ AI Tags:</strong> 
                            ${article.llm_tags.map(tag => 
                                `<span style="background: rgba(255,215,0,0.2); padding: 2px 6px; border-radius: 3px; margin-right: 5px;">${tag}</span>`
                            ).join('')}
                        </p>
                    ` : ''}
                    <p>${(article.content || 'No content available').substring(0, 200)}...</p>
                    <a href="${article.url}" target="_blank" class="article-link">Read full article →</a>
                </div>
            `;
        });

        container.innerHTML = html;
    }

    /**
     * Refresh all alerts data
     */
    async refresh() {
        await this.loadStats();
        await this.loadAlerts();
        await this.loadArticles();
        console.log('🔄 Alerts data refreshed');
    }
}

// Export for use in main.js
window.AlertsManager = AlertsManager;