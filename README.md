# 📊 MarketPulse - AI-Powered Financial Intelligence Platform

> Real-time market monitoring with smart alerts and AI analysis

## 🎯 What is MarketPulse?

MarketPulse is an intelligent financial monitoring system that:
- 📰 **Monitors** 10+ financial news sources 24/7
- 🔍 **Detects** market-moving events using keywords
- 🤖 **Analyzes** news impact with Perplexity AI
- 🚨 **Alerts** you via WhatsApp, Telegram, and Dashboard
- 📈 **Predicts** stock movements with ML models

---

## 📚 Documentation Index

### **Start Here:**
1. **[SUMMARY.md](./SUMMARY.md)** - Quick overview (5 min read)
2. **[COMPARISON.md](./COMPARISON.md)** - Why it doesn't work yet vs TARIFF RADAR
3. **[SYSTEM_OVERVIEW.md](./SYSTEM_OVERVIEW.md)** - Visual architecture guide

### **For Development:**
4. **[ROADMAP.md](./ROADMAP.md)** - Full development plan (8 weeks)
5. **[keywords.yaml](./keywords.yaml)** - Complete keyword dictionary
6. **[config.yaml](./app/config.yaml)** - System configuration (600+ lines)

### **For Customization:**
7. **[PROMPTS_GUIDE.md](./PROMPTS_GUIDE.md)** - How to edit AI prompts
8. **[DATA_SOURCES.md](./DATA_SOURCES.md)** - All data sources explained
9. **[PERPLEXITY_UPDATE.md](./PERPLEXITY_UPDATE.md)** - Perplexity model changes

---

## ⚡ Quick Start

### **Current Status: Prototype Phase**

✅ What works:
- Perplexity AI integration (model: `sonar`)
- Yahoo Finance + Alpha Vantage APIs
- Twitter + Reddit sentiment
- TensorFlow ML infrastructure
- Editable prompts system

❌ What doesn't work yet:
- No automatic monitoring (runs manually)
- No real-time alerts
- No RSS feed scraping
- No keyword detection engine
- No WhatsApp/Telegram integration

### **To Test Current Features:**

```bash
# 1. Clone and setup
cd MarketPulse
pip install -r requirements.txt

# 2. Configure API keys (edit .env)
PERPLEXITY_API_KEY=your_key
ALPHA_VANTAGE_API_KEY=your_key
TWITTER_BEARER_TOKEN=your_token
REDDIT_CLIENT_ID=your_id
REDDIT_CLIENT_SECRET=your_secret

# 3. Test Perplexity
python test_perplexity_direct.py

# 4. Test market data
python app/financial/market_data_clean.py

# 5. Run server (dashboard only, no alerts)
python app/main_production_enhanced.py
```

---

## 🔍 What We Monitor

### **📰 Financial News** (Real-time)
- Reuters Business
- Bloomberg Markets
- WSJ Markets
- CNBC Breaking News
- MarketWatch
- Seeking Alpha
- Yahoo Finance

### **🏛️ Regulatory Filings**
- SEC EDGAR (10-K, 10-Q, 8-K, Form 4)
- FDA Drug Approvals
- USPTO Patents
- FTC Antitrust

### **💬 Social Sentiment**
- Twitter $SYMBOL mentions
- Reddit (r/wallstreetbets, r/stocks)
- StockTwits

### **🌍 Geopolitical Events**
- Trade wars & tariffs
- Central bank decisions
- Political developments
- Oil & commodity prices

---

## 🔑 Keyword System

### **Bullish Signals** (+1 to +3 points)
```
+3: "FDA approval", "record earnings", "beat expectations"
+2: "analyst upgrade", "strong demand", "exceeds forecast"
+1: "growth", "expansion", "improved"
```

### **Bearish Signals** (-1 to -3 points)
```
-3: "bankruptcy", "fraud", "investigation", "product recall"
-2: "miss expectations", "downgrade", "lowered guidance"
-1: "decline", "weakness", "concern"
```

### **Alert Thresholds**
```
Score ≥ 2.5 → 🔴 CRITICAL (WhatsApp + Telegram + Dashboard)
Score ≥ 1.5 → 🟡 IMPORTANT (Telegram + Dashboard)
Score ≥ 0.8 → 🟢 WATCH (Dashboard only)
```

**Full keyword dictionary:** [keywords.yaml](./keywords.yaml)

---

## 🚨 Alert System (Planned)

### **Multi-Channel Notifications**

```
┌─────────────────────────────────────┐
│ 🔴 CRITICAL ALERT                   │
├─────────────────────────────────────┤
│ AAPL | +5.2% 📈                     │
│ FDA approved breakthrough drug      │
│                                     │
│ AI Analysis:                        │
│ • Sentiment: Very Bullish (0.92)   │
│ • Recommendation: STRONG BUY        │
│ • Target: $280 | Stop: $250        │
│                                     │
│ Sent via:                           │
│ ✅ WhatsApp                         │
│ ✅ Telegram                         │
│ ✅ Dashboard                        │
└─────────────────────────────────────┘
```

---

## 📅 Development Timeline

### **Phase 1: Foundation** (Week 1-2)
- [ ] Setup background jobs (Celery/APScheduler)
- [ ] Build RSS feed scrapers
- [ ] Create keyword matching engine
- [ ] Design alerts database schema

### **Phase 2: Alerts** (Week 3-4)
- [ ] Alert decision engine
- [ ] Telegram integration
- [ ] WhatsApp integration (Twilio)
- [ ] Dashboard WebSocket
- [ ] Alert settings UI

### **Phase 3: Intelligence** (Week 5-6)
- [ ] Perplexity real-time analysis
- [ ] Smart filtering (reduce noise)
- [ ] ML pattern recognition
- [ ] User feedback system

### **Phase 4: Polish** (Week 7-8)
- [ ] Dashboard enhancements
- [ ] Alert customization
- [ ] Performance optimization
- [ ] Testing & documentation

**Total:** 8 weeks to production-ready

**Fast track:** 3 weeks for basic working system

---

## 💰 Operating Costs

| Service | Cost/Month | Notes |
|---------|-----------|-------|
| Yahoo Finance | $0 | Free tier |
| Alpha Vantage | $0-50 | 500 calls/day free |
| Twitter API | $100 | Basic tier |
| Reddit API | $0 | Free unlimited |
| Perplexity AI | ~$50 | Pay per use (~$0.005/query) |
| Twilio WhatsApp | ~$10 | $0.005/message |
| Telegram Bot | $0 | Free unlimited |
| VPS Server | $10-20 | 2GB RAM |
| **Total** | **~$170-180** | Full featured |

**Minimum viable:** ~$60/month (Perplexity + Server only)

---

## 🛠️ Tech Stack

### **Backend**
- FastAPI (REST API + WebSocket)
- Python 3.13
- SQLite (can upgrade to PostgreSQL)

### **AI/ML**
- Perplexity AI (sonar model)
- TensorFlow 2.20.0
- Scikit-learn 1.7.2
- LSTM, Transformer, CNN models

### **Data Sources**
- Yahoo Finance (primary)
- Alpha Vantage (backup)
- Twitter API (sentiment)
- Reddit API (community sentiment)
- RSS feeds (news aggregation)

### **Notifications**
- Twilio (WhatsApp)
- Telegram Bot API
- WebSocket (dashboard)
- SMTP (email)

### **Automation**
- APScheduler / Celery
- Background jobs
- Scheduled tasks

---

## 🎓 Learning Resources

### **Understanding the System**
1. Read [SUMMARY.md](./SUMMARY.md) first
2. Check [COMPARISON.md](./COMPARISON.md) to see what's missing
3. Review [SYSTEM_OVERVIEW.md](./SYSTEM_OVERVIEW.md) for architecture

### **Customizing Prompts**
1. Read [PROMPTS_GUIDE.md](./PROMPTS_GUIDE.md)
2. Edit [config.yaml](./app/config.yaml)
3. Test with `python app/financial/perplexity_analyzer.py`

### **Adding Keywords**
1. Study [keywords.yaml](./keywords.yaml) structure
2. Add your keywords following the format
3. Adjust scoring rules as needed

### **Understanding Data Sources**
1. Read [DATA_SOURCES.md](./DATA_SOURCES.md)
2. See what websites we monitor
3. Learn why each source matters

---

## 🔧 Configuration

### **Environment Variables** (.env)
```bash
# AI Services
PERPLEXITY_API_KEY=your_key_here

# Market Data
ALPHA_VANTAGE_API_KEY=your_key_here

# Social Media
TWITTER_BEARER_TOKEN=your_token_here
REDDIT_CLIENT_ID=your_id_here
REDDIT_CLIENT_SECRET=your_secret_here

# Notifications
TELEGRAM_BOT_TOKEN=your_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
TWILIO_ACCOUNT_SID=your_sid_here
TWILIO_AUTH_TOKEN=your_token_here

# Security
SECRET_KEY=change_this_in_production
```

### **System Configuration** (config.yaml)
```yaml
llm:
  provider: "perplexity"
  model: "sonar"
  temperature: 0.2
  max_tokens: 1000

prompts:
  stock_analysis:
    system: "You are a financial analyst..."
    user_template: "Analyze {symbol}..."

keywords:
  very_bullish:
    score: +3
    keywords: ["FDA approval", "record earnings"]
```

**Full config:** [config.yaml](./app/config.yaml) (600+ lines)

---

## 📖 API Documentation

### **Current Endpoints** (Limited functionality)

```bash
# Market Data
GET /api/stock/{symbol}        # Get stock price
GET /api/market/indices         # Get market indices

# Analysis (manual trigger)
POST /api/analyze/{symbol}      # Get AI analysis

# Health
GET /health                     # System health check
```

**Note:** Alert endpoints don't exist yet!

---

## 🤝 Contributing

This is currently a prototype. Main tasks needed:

1. **Background Jobs** - Setup automatic monitoring
2. **RSS Scrapers** - Build news fetchers
3. **Keyword Engine** - Implement scoring system
4. **Alert System** - Multi-channel notifications
5. **Dashboard** - Real-time WebSocket alerts

See [ROADMAP.md](./ROADMAP.md) for details.

---

## 📝 Version History

### **v0.1.0** (Oct 19, 2025) - Prototype
- ✅ Perplexity AI integrated (sonar model)
- ✅ Market data APIs working
- ✅ Social sentiment APIs configured
- ✅ ML infrastructure setup
- ✅ Configuration system with editable prompts
- ✅ Comprehensive documentation
- ❌ No automation yet
- ❌ No alerts yet

### **v0.2.0** (Planned)
- [ ] Background jobs running
- [ ] RSS feed scraping active
- [ ] Keyword detection working
- [ ] Basic Telegram alerts

### **v1.0.0** (Target: 8 weeks)
- [ ] Full automation
- [ ] Multi-channel alerts
- [ ] Dashboard real-time updates
- [ ] WhatsApp integration
- [ ] Smart filtering
- [ ] Production ready

---

## 📞 Support & Questions

### **Documentation Issues?**
- Check if your question is in [SUMMARY.md](./SUMMARY.md)
- Review [PROMPTS_GUIDE.md](./PROMPTS_GUIDE.md) for customization
- See [COMPARISON.md](./COMPARISON.md) to understand the gap

### **Want to Contribute?**
- Read [ROADMAP.md](./ROADMAP.md)
- Pick a task from Phase 1-4
- Follow TARIFF RADAR structure (see [COMPARISON.md](./COMPARISON.md))

---

## 📄 License

MIT License - Free to use and modify

---

## 🎯 Project Status

**Current:** 🟡 Prototype Phase  
**Goal:** 🟢 Production System (8 weeks)

**What works:** APIs, AI, ML infrastructure  
**What's needed:** Automation, Alerts, Real-time monitoring

**Start here:** [SUMMARY.md](./SUMMARY.md) → [ROADMAP.md](./ROADMAP.md) → Build!

---

*Last Updated: October 19, 2025*  
*Version: 0.1.0 (Prototype)*

A comprehensive automated monitoring system for US-China trade news, tariffs, and economic developments with AI-powered analysis and multi-channel notifications.

## 🎯 Overview

Tariff Radar combines deterministic data ingestion with smart AI analysis to monitor global trade news sources, identify relevant articles about US-China trade relationships, and deliver timely notifications through multiple channels (WeChat Work, Email, Telegram).

### Key Features

- **Multi-Source Ingestion**: Automated RSS feed monitoring from major financial news sources
- **Smart Content Analysis**: Multilingual keyword filtering + semantic embeddings + ML classification + LLM triage
- **Duplicate Detection**: Advanced similarity detection to avoid redundant alerts
- **Multi-Channel Notifications**: WeCom (Enterprise WeChat), Email, and Telegram integration
- **Web Dashboard**: Real-time monitoring interface with article management
- **Scalable Architecture**: Docker Compose deployment with Celery task queues
- **דשבורד**: ממשק ניהול וצפייה ב-FastAPI

## Architecture

```
[Sources] → [Ingestion] → ["Smart" Layer] → [Storage] → [Alerting]
```

### תהליך עבודה:
1. **איסוף נתונים**: RSSHub, scrapers, APIs
2. **נירמול**: ניקוי HTML, זיהוי שפה, דה-דופליקציה
3. **ניתוח**: מילות מפתח → embeddings → classifier → LLM triage
4. **התראות**: WeCom + Email לפי רמת חשיבות
5. **ארכיון**: PostgreSQL + Qdrant vectors

## Quick Start

```bash
# Clone and setup
cd tariff-radar
cp .env.example .env
# Edit .env with your credentials

# Start services
docker-compose up -d

# Access dashboard
http://localhost:8000
```

## Configuration

ערוך את `config.yaml` להגדרת מקורות, ספים, והתראות.

## Requirements

- Docker & Docker Compose
- Python 3.11+
- WeCom corporate account (optional)
- OpenAI/Anthropic/Perplexity API key (for LLM triage)

## Project Structure

```
tariff-radar/
  ├─ app/
  │  ├─ ingest/         # Data collection & normalization
  │  ├─ smart/          # AI analysis layer
  │  ├─ notify/         # Alert handlers
  │  ├─ storage/        # DB models & vector store
  │  └─ sched/          # Task scheduling
  ├─ docker-compose.yml
  ├─ config.yaml
  └─ .env
```

## License

MIT