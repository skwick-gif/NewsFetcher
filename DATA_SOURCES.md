# 📊 MarketPulse Data Sources Strategy

## Overview
MarketPulse monitors comprehensive financial intelligence across multiple domains, similar to how TARIFF RADAR monitored trade policy news.

## 🎯 Data Source Categories

### 1. **Market Data** (Real-time)
- ✅ **Yahoo Finance** - Primary stock prices, indices, market data
- ✅ **Alpha Vantage API** - Backup stock data, technical indicators
- 📊 Coverage: All major US stocks, ETFs, indices
- 🔄 Update frequency: Real-time / 1-minute intervals

### 2. **Financial News** (Breaking news)
- 📰 **Reuters Finance** - Breaking financial news
- 📰 **Bloomberg** - Markets and companies
- 📰 **Wall Street Journal** - Market analysis
- 📰 **Financial Times** - International business
- 📰 **CNBC** - Real-time market news
- 📰 **MarketWatch** - Market pulse and trends
- 📰 **Seeking Alpha** - Investment research
- 🔄 Update frequency: RSS feeds checked every 5 minutes

### 3. **Regulatory & Government** (Official filings)
- 🏛️ **SEC EDGAR** - Company filings (10-K, 10-Q, 8-K, insider trades)
  - Coverage: All publicly traded US companies
  - Data: Earnings reports, financial statements, material events
- 🏛️ **FDA Newsroom** - Drug approvals, medical device clearances
  - Coverage: Pharmaceutical and biotech companies
  - Data: New drug approvals, clinical trial results, safety alerts
- 🏛️ **USPTO** - Patent applications and grants
  - Coverage: Technology companies, pharmaceutical innovations
  - Data: New patents, patent disputes, IP protection
- 🏛️ **FTC** - Antitrust, mergers, regulatory actions
  - Coverage: Large cap companies, M&A activity
- 🔄 Update frequency: Daily checks for new filings

### 4. **Social Media Sentiment** (Public opinion)
- ✅ **Twitter/X API** - Real-time social sentiment
  - Coverage: Trending stocks, breaking news reactions
  - Analysis: Sentiment scoring, volume tracking
- ✅ **Reddit API** - Community discussions
  - Subreddits: r/wallstreetbets, r/stocks, r/investing, r/options
  - Analysis: Discussion volume, sentiment trends
- 🔄 Update frequency: Every 15 minutes (API rate limits)

### 5. **Geopolitical Events** (Market movers)
- 🌍 **Reuters World** - Global political events
- 🌍 **BBC Business** - International business news
- 🌍 **Associated Press** - Breaking geopolitical news
- 📋 Keywords monitored:
  - Trade: tariffs, sanctions, embargoes, trade deals
  - Energy: oil prices, OPEC, energy policy
  - Policy: central bank decisions, interest rates, fiscal policy
  - Conflict: wars, political instability, elections
  - Currency: forex movements, currency crises
- 🔄 Update frequency: Every 10 minutes

### 6. **Economic Indicators** (Macro data)
- 📈 **FRED (Federal Reserve)** - Economic data
  - GDP, inflation, unemployment, interest rates
- 📈 **Bureau of Labor Statistics** - Employment data
- 📈 **Census Bureau** - Economic indicators
- 🔄 Update frequency: Daily (indicators publish monthly/quarterly)

### 7. **Earnings & Events** (Corporate calendar)
- 📅 **Earnings Calendar** - Upcoming earnings announcements
- 📅 **IPO Calendar** - New public offerings
- 📅 **Economic Calendar** - Fed meetings, data releases
- 🔄 Update frequency: Weekly updates

## 🤖 AI Analysis Layer

### **Perplexity AI** (llama-3.1-sonar-small-128k-online)
- ✅ Model: Same as TARIFF RADAR
- 🎯 Purpose: Intelligent analysis of news, events, and market data
- 📊 Capabilities:
  - News sentiment analysis
  - Market event impact assessment
  - Stock-specific insights (real-time search)
  - Sector trend analysis
  - Risk identification

## 📊 Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     DATA COLLECTION                          │
├─────────────────────────────────────────────────────────────┤
│  Market Data    Financial News    Regulatory    Social       │
│  (Yahoo/AV)     (RSS Feeds)       (SEC/FDA)     (Twitter)    │
└────────┬────────────────┬──────────────┬──────────┬─────────┘
         │                │              │          │
         ▼                ▼              ▼          ▼
┌─────────────────────────────────────────────────────────────┐
│                    DATA PROCESSING                           │
├─────────────────────────────────────────────────────────────┤
│  • Deduplication           • Text normalization              │
│  • Sentiment scoring       • Entity extraction               │
│  • Relevance filtering     • Keyword extraction              │
└────────────────────────────┬───────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                   AI ANALYSIS LAYER                          │
├─────────────────────────────────────────────────────────────┤
│  Perplexity AI: Intelligent news & event analysis           │
│  • Impact assessment  • Trend identification                 │
│  • Risk analysis      • Trading insights                     │
└────────────────────────────┬───────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                    STORAGE & INDEXING                        │
├─────────────────────────────────────────────────────────────┤
│  SQLite: Historical data, events, filings                    │
│  Vector DB: Semantic search, similar events                  │
│  Cache: Real-time data, hot stocks                           │
└────────────────────────────┬───────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                  ML PREDICTION LAYER                         │
├─────────────────────────────────────────────────────────────┤
│  TensorFlow Models:                                          │
│  • LSTM: Time series prediction                              │
│  • Transformer: Market sentiment analysis                    │
│  • CNN: Pattern recognition                                  │
│  • Ensemble: Combined predictions                            │
└────────────────────────────┬───────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                    DELIVERY LAYER                            │
├─────────────────────────────────────────────────────────────┤
│  • WebSocket: Real-time updates                              │
│  • REST API: Historical queries                              │
│  • Alerts: Telegram/Email notifications                      │
│  • Dashboard: Interactive visualization                      │
└─────────────────────────────────────────────────────────────┘
```

## 🎯 Monitoring Strategy

### **High Priority** (Check every 5-15 minutes)
1. Market data (prices, volumes)
2. Breaking financial news (Reuters, Bloomberg, CNBC)
3. Social sentiment (Twitter trending, Reddit spikes)
4. Geopolitical events (major news wires)

### **Medium Priority** (Check hourly)
1. Company news (press releases, announcements)
2. Analyst ratings and price targets
3. Earnings calendar updates
4. Economic calendar

### **Low Priority** (Check daily)
1. SEC filings (10-K, 10-Q)
2. FDA approvals
3. Patent filings
4. Long-term economic indicators

## 🔍 Specific Use Cases

### 1. **Pharmaceutical Stocks** (e.g., PFE, JNJ, MRNA)
- ✅ FDA approval news (drug approvals)
- ✅ Clinical trial results
- ✅ Patent expiration tracking
- ✅ Regulatory investigations

### 2. **Technology Stocks** (e.g., AAPL, MSFT, GOOGL)
- ✅ Patent filings (new innovations)
- ✅ Regulatory scrutiny (antitrust)
- ✅ Product launches
- ✅ Earnings announcements

### 3. **Financial Stocks** (e.g., JPM, GS, BAC)
- ✅ Fed policy changes (interest rates)
- ✅ Regulatory changes (banking rules)
- ✅ Economic indicators (GDP, unemployment)
- ✅ Earnings reports

### 4. **Energy Stocks** (e.g., XOM, CVX, BP)
- ✅ Geopolitical events (OPEC, sanctions)
- ✅ Oil price movements
- ✅ Environmental regulations
- ✅ Production reports

### 5. **Consumer Stocks** (e.g., WMT, AMZN, COST)
- ✅ Economic indicators (retail sales, consumer confidence)
- ✅ Earnings reports
- ✅ E-commerce trends
- ✅ Supply chain news

## 📈 Data Quality Metrics

| Source Category | Coverage | Reliability | Latency | Cost |
|-----------------|----------|-------------|---------|------|
| Market Data | 100% | ⭐⭐⭐⭐⭐ | Real-time | Free |
| Financial News | 95% | ⭐⭐⭐⭐⭐ | 1-5 min | Free |
| Social Media | 80% | ⭐⭐⭐⭐ | Real-time | API limits |
| SEC Filings | 100% | ⭐⭐⭐⭐⭐ | Same day | Free |
| FDA Approvals | 100% | ⭐⭐⭐⭐⭐ | Same day | Free |
| Geopolitical | 90% | ⭐⭐⭐⭐ | 5-15 min | Free |
| Patents | 60% | ⭐⭐⭐ | Daily | Limited |

## 🚀 Next Steps

1. ✅ **Perplexity Integration** - Fixed API format (llama-3.1-sonar-small-128k-online)
2. ⏳ **RSS Feed Monitoring** - Deploy financial_sources.py
3. ⏳ **SEC Filing Parser** - Auto-extract key metrics from 10-K/10-Q
4. ⏳ **FDA Alert System** - Real-time drug approval notifications
5. ⏳ **Geopolitical Impact Scoring** - Perplexity analysis of events
6. ⏳ **Patent Tracking** - USPTO API integration
7. ⏳ **Earnings Surprise Detection** - Compare actual vs. expected
8. ⏳ **Insider Trading Tracker** - Monitor Form 4 filings

## 💡 Comparison to TARIFF RADAR

| Feature | TARIFF RADAR | MarketPulse |
|---------|--------------|-------------|
| **Primary Focus** | Trade policy news | Financial intelligence |
| **News Sources** | Political/trade outlets | Financial news wires |
| **AI Analysis** | Perplexity (trade impact) | Perplexity (market impact) |
| **Data Sources** | RSS feeds, government sites | Market data + news + filings |
| **Alert System** | Trade policy changes | Price moves + news events |
| **User Base** | Trade policy analysts | Traders & investors |
| **ML Models** | News classification | Price prediction + sentiment |

## 🎯 Success Metrics

- **Coverage**: Monitor 100+ financial news sources
- **Latency**: < 5 minutes for breaking news
- **Accuracy**: 85%+ sentiment classification
- **Volume**: Process 10,000+ articles/day
- **Relevance**: 90%+ of alerts actionable

---

**Status**: 
- ✅ Perplexity AI fixed (same model as TARIFF RADAR)
- ✅ Financial sources module created
- ⏳ RSS monitoring deployment pending
- ⏳ Integration with main server pending
