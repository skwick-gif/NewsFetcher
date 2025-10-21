# 🔄 MarketPulse vs TARIFF RADAR - Comparison

## 🎯 למה TARIFF RADAR עובד ו-MarketPulse לא?

### **TARIFF RADAR** ✅ (Working System)

```
┌─────────────────────────────────────────────┐
│         WHAT MAKES IT WORK?                 │
├─────────────────────────────────────────────┤
│                                             │
│  ✅ Background Jobs (Celery)                │
│     └─ Runs every 5-15 minutes              │
│                                             │
│  ✅ RSS Feed Scrapers                       │
│     ├─ Reuters                              │
│     ├─ NY Times                             │
│     ├─ Guardian                             │
│     └─ NDRC (China)                         │
│                                             │
│  ✅ Keyword Matching                        │
│     ├─ Primary: 关税, tariff                │
│     ├─ Secondary: 贸易战, trade war          │
│     └─ Scoring: keyword_score + semantic    │
│                                             │
│  ✅ AI Triage (Perplexity)                  │
│     └─ Analyzes if article is relevant      │
│                                             │
│  ✅ Alert System                            │
│     ├─ WeChat Work                          │
│     ├─ Telegram                             │
│     └─ Email                                │
│                                             │
│  ✅ Database Storage                        │
│     └─ PostgreSQL with vector search        │
│                                             │
└─────────────────────────────────────────────┘
```

### **MarketPulse** ❌ (Not Working Yet)

```
┌─────────────────────────────────────────────┐
│         WHAT'S MISSING?                     │
├─────────────────────────────────────────────┤
│                                             │
│  ❌ NO Background Jobs                      │
│     └─ Everything is manual                 │
│                                             │
│  ❌ NO RSS Feed Scrapers                    │
│     └─ APIs configured but not used         │
│                                             │
│  ❌ NO Keyword Matching                     │
│     └─ Don't know what to look for          │
│                                             │
│  ❌ Perplexity Not Integrated               │
│     └─ Works standalone, not in pipeline    │
│                                             │
│  ❌ NO Alert System                         │
│     └─ No WhatsApp, no Telegram, no dash    │
│                                             │
│  ✅ Database Schema Exists                  │
│     └─ But not used for alerts              │
│                                             │
└─────────────────────────────────────────────┘
```

---

## 📊 Side-by-Side Comparison

| Feature | TARIFF RADAR | MarketPulse | Gap |
|---------|--------------|-------------|-----|
| **Background Jobs** | ✅ Celery | ❌ None | Need Celery/APScheduler |
| **RSS Scraping** | ✅ 6 sources | ❌ 0 sources | Build scrapers |
| **Keyword System** | ✅ ~50 keywords | ❌ 0 keywords | Define keywords |
| **AI Analysis** | ✅ Integrated | ⚠️ Standalone | Connect to pipeline |
| **Alerts** | ✅ WeChat/Telegram | ❌ None | Build alert system |
| **Dashboard** | ✅ Real-time | ⚠️ Static | Add WebSocket |
| **Database** | ✅ PostgreSQL | ✅ SQLite | Good enough |
| **Vector Search** | ✅ Qdrant | ⚠️ Not used | Optional |

---

## 🔍 Deep Dive: How TARIFF RADAR Works

### **1. Background Job Configuration**

**File:** `tariff-radar/sched/tasks.py`
```python
from celery import Celery
from celery.schedules import crontab

app = Celery('tariff_radar')

# Main ingestion task - runs every 5 minutes
@app.task
def ingest_news():
    """Fetch and process news from all sources"""
    
    # 1. Fetch RSS feeds
    articles = fetch_rss_feeds()
    
    # 2. Extract keywords
    for article in articles:
        keywords = extract_keywords(article.text)
        article.keyword_score = calculate_score(keywords)
    
    # 3. Filter by threshold
    relevant = [a for a in articles if a.keyword_score > 1.0]
    
    # 4. AI triage for high-score articles
    for article in relevant:
        if article.keyword_score > 2.0:
            ai_analysis = perplexity_analyze(article)
            article.ai_relevant = ai_analysis['relevant']
    
    # 5. Send alerts
    for article in relevant:
        if should_alert(article):
            send_alerts(article)
    
    # 6. Save to database
    save_to_db(relevant)

# Schedule
app.conf.beat_schedule = {
    'ingest-news': {
        'task': 'tasks.ingest_news',
        'schedule': 300.0,  # every 5 minutes
    },
}
```

### **2. Keyword Matching**

**File:** `tariff-radar/app/smart/keywords.py`
```python
class KeywordMatcher:
    def __init__(self):
        self.keywords = {
            'primary': {
                'chinese': ['关税', '加征关税', '关税清单'],
                'english': ['tariff', '301 tariffs', 'customs duties']
            },
            'secondary': {
                'chinese': ['反制措施', '贸易战', '301关税'],
                'english': ['retaliation', 'trade war', 'countermeasure']
            }
        }
    
    def calculate_score(self, text):
        score = 0
        
        # Primary keywords: +2 points each
        for keyword in self.keywords['primary']:
            if keyword in text.lower():
                score += 2
        
        # Secondary keywords: +1 point each
        for keyword in self.keywords['secondary']:
            if keyword in text.lower():
                score += 1
        
        return score
```

### **3. Alert System**

**File:** `tariff-radar/app/notify/wecom.py`
```python
def send_wecom_alert(article):
    """Send alert via WeChat Work"""
    
    message = f"""
    📢 贸易政策新闻提醒
    
    标题: {article.title}
    来源: {article.source}
    时间: {article.published}
    
    关键词: {', '.join(article.keywords)}
    相关度: {article.keyword_score}/5
    AI评分: {article.ai_confidence}
    
    摘要:
    {article.summary}
    
    [查看详情]({article.url})
    """
    
    requests.post(
        f"https://qyapi.weixin.qq.com/cgi-bin/message/send",
        json={
            "touser": "@all",
            "msgtype": "markdown",
            "markdown": {"content": message}
        }
    )
```

---

## 🎯 What MarketPulse Needs to Copy

### **1. Background Jobs Setup**

**Create:** `MarketPulse/app/sched/tasks.py`
```python
from apscheduler.schedulers.background import BackgroundScheduler

scheduler = BackgroundScheduler()

@scheduler.scheduled_job('interval', minutes=5)
def fetch_financial_news():
    """Fetch news every 5 minutes"""
    
    # 1. Scrape RSS feeds
    articles = scrape_all_feeds()
    
    # 2. Extract keywords & calculate scores
    for article in articles:
        article.score = analyze_keywords(article)
    
    # 3. Filter high-impact
    alerts = [a for a in articles if a.score >= 2.5]
    
    # 4. Send notifications
    for alert in alerts:
        notify_users(alert)

scheduler.start()
```

### **2. RSS Feed Scrapers**

**Create:** `MarketPulse/app/ingest/rss_scraper.py`
```python
import feedparser

RSS_FEEDS = {
    'reuters': 'https://www.reuters.com/finance/rss',
    'bloomberg': 'https://www.bloomberg.com/feed',
    'wsj': 'https://feeds.a.dj.com/rss/RSSMarketsMain.xml',
    'cnbc': 'https://www.cnbc.com/id/100003114/device/rss',
}

def scrape_all_feeds():
    articles = []
    
    for source, url in RSS_FEEDS.items():
        feed = feedparser.parse(url)
        
        for entry in feed.entries:
            article = {
                'title': entry.title,
                'text': entry.summary,
                'source': source,
                'url': entry.link,
                'published': entry.published,
            }
            articles.append(article)
    
    return articles
```

### **3. Keyword Analysis**

**Create:** `MarketPulse/app/smart/keywords_engine.py`
```python
import yaml

class KeywordAnalyzer:
    def __init__(self, keywords_file='keywords.yaml'):
        with open(keywords_file) as f:
            self.config = yaml.safe_load(f)
    
    def analyze(self, article):
        score = 0
        matched_keywords = []
        
        text = article['title'] + ' ' + article['text']
        text_lower = text.lower()
        
        # Check bullish keywords
        for keyword in self.config['keywords']['very_bullish']['keywords']:
            if keyword in text_lower:
                score += 3
                matched_keywords.append((keyword, +3))
        
        for keyword in self.config['keywords']['bullish']['keywords']:
            if keyword in text_lower:
                score += 2
                matched_keywords.append((keyword, +2))
        
        # Check bearish keywords
        for keyword in self.config['keywords']['very_bearish']['keywords']:
            if keyword in text_lower:
                score -= 3
                matched_keywords.append((keyword, -3))
        
        for keyword in self.config['keywords']['bearish']['keywords']:
            if keyword in text_lower:
                score -= 2
                matched_keywords.append((keyword, -2))
        
        # Apply source weight
        source_weight = self.config['scoring_rules']['source_weight'].get(
            article['source'], 1.0
        )
        final_score = score * source_weight
        
        return {
            'score': final_score,
            'keywords': matched_keywords,
            'sentiment': self._get_sentiment(final_score),
            'priority': self._get_priority(abs(final_score))
        }
    
    def _get_sentiment(self, score):
        if score >= 2.5: return 'Very Bullish'
        elif score >= 1.5: return 'Bullish'
        elif score > -1.5: return 'Neutral'
        elif score > -2.5: return 'Bearish'
        else: return 'Very Bearish'
    
    def _get_priority(self, score):
        if score >= 2.5: return 'critical'
        elif score >= 1.5: return 'important'
        elif score >= 0.8: return 'watch'
        else: return 'ignore'
```

### **4. Alert Dispatcher**

**Create:** `MarketPulse/app/notify/dispatcher.py`
```python
from telegram import Bot
from twilio.rest import Client

class AlertDispatcher:
    def __init__(self):
        self.telegram_bot = Bot(token=os.getenv('TELEGRAM_BOT_TOKEN'))
        self.twilio_client = Client(
            os.getenv('TWILIO_ACCOUNT_SID'),
            os.getenv('TWILIO_AUTH_TOKEN')
        )
    
    def dispatch(self, alert):
        priority = alert['priority']
        
        if priority == 'critical':
            # Send to all channels
            self.send_whatsapp(alert)
            self.send_telegram(alert)
            self.send_dashboard(alert)
        
        elif priority == 'important':
            # Skip WhatsApp
            self.send_telegram(alert)
            self.send_dashboard(alert)
        
        elif priority == 'watch':
            # Dashboard only
            self.send_dashboard(alert)
    
    def send_whatsapp(self, alert):
        message = f"""
🚨 MARKET ALERT

{alert['symbol']} | {alert['change']}%

{alert['reason']}

AI: {alert['recommendation']} ({alert['confidence']})
        """
        
        self.twilio_client.messages.create(
            from_='whatsapp:+14155238886',
            to='whatsapp:+972XXXXXXXXX',
            body=message
        )
    
    def send_telegram(self, alert):
        message = f"""
🔔 *Alert: {alert['symbol']}*

📈 Price: {alert['change']}%
📰 {alert['reason']}
🎯 {alert['recommendation']}
🤖 Confidence: {alert['confidence']}
        """
        
        self.telegram_bot.send_message(
            chat_id=os.getenv('TELEGRAM_CHAT_ID'),
            text=message,
            parse_mode='Markdown'
        )
    
    def send_dashboard(self, alert):
        # Push via WebSocket
        websocket_broadcast(alert)
```

---

## 📋 File Structure Comparison

### **TARIFF RADAR** (Working)
```
tariff-radar/
├── app/
│   ├── sched/
│   │   ├── __init__.py
│   │   └── tasks.py           ✅ Background jobs
│   ├── ingest/
│   │   ├── rsshub_loader.py   ✅ RSS scraping
│   │   └── dedup.py           ✅ Deduplication
│   ├── smart/
│   │   ├── keywords.py        ✅ Keyword matching
│   │   ├── embedder.py        ✅ Semantic analysis
│   │   └── triage_agent.py    ✅ AI analysis
│   ├── notify/
│   │   ├── wecom.py           ✅ WeChat alerts
│   │   ├── telegram.py        ✅ Telegram alerts
│   │   └── emailer.py         ✅ Email alerts
│   └── storage/
│       ├── db.py              ✅ PostgreSQL
│       └── models.py          ✅ Data models
├── config.yaml                ✅ Keywords defined
├── docker-compose.yml         ✅ Full stack
└── requirements.txt           ✅ All dependencies
```

### **MarketPulse** (Needs Work)
```
MarketPulse/
├── app/
│   ├── sched/
│   │   └── tasks.py           ❌ Empty/not working
│   ├── ingest/
│   │   └── financial_sources.py  ⚠️ Created but not running
│   ├── financial/
│   │   ├── perplexity_analyzer.py  ✅ Works standalone
│   │   ├── market_data_clean.py    ✅ Real data
│   │   └── social_sentiment.py     ✅ APIs configured
│   ├── notify/
│   │   └── ???                ❌ Doesn't exist!
│   └── storage/
│       └── db.py              ⚠️ Exists but no alerts table
├── config.yaml                ✅ Just created
├── keywords.yaml              ✅ Just created
└── requirements.txt           ✅ Has dependencies
```

---

## 🚀 Action Plan: Copy TARIFF RADAR Structure

### **Step 1: Setup Background Jobs** (Day 1)
```bash
# Install scheduler
pip install apscheduler celery redis

# Create tasks.py like TARIFF RADAR
cp tariff-radar/app/sched/tasks.py MarketPulse/app/sched/tasks.py

# Modify for financial markets
# - Change keywords from tariffs to stocks
# - Change sources from Reuters China to Reuters Finance
```

### **Step 2: Copy RSS Scraper** (Day 2)
```bash
# Copy structure
cp tariff-radar/app/ingest/rsshub_loader.py MarketPulse/app/ingest/rss_scraper.py

# Update URLs
# - Old: MOFCOM, GACC, People's Daily
# - New: Reuters Finance, Bloomberg, WSJ, CNBC
```

### **Step 3: Copy Keyword Engine** (Day 3)
```bash
# Copy logic
cp tariff-radar/app/smart/keywords.py MarketPulse/app/smart/keywords_engine.py

# Update keywords
# - Old: 关税, 加征关税, 贸易战
# - New: FDA approval, earnings beat, downgrade
```

### **Step 4: Copy Alert System** (Day 4)
```bash
# Copy notification infrastructure
cp -r tariff-radar/app/notify/ MarketPulse/app/notify/

# Update:
# - wecom.py → whatsapp.py (use Twilio)
# - Keep telegram.py (works the same)
# - Keep emailer.py (works the same)
```

### **Step 5: Integrate Everything** (Day 5-7)
```python
# Main task loop (like TARIFF RADAR)
@scheduler.scheduled_job('interval', minutes=5)
def monitor_markets():
    # 1. Fetch news
    articles = scrape_financial_news()
    
    # 2. Analyze keywords
    for article in articles:
        article.analysis = keyword_analyzer.analyze(article)
    
    # 3. Filter alerts
    alerts = [a for a in articles if a.analysis['priority'] != 'ignore']
    
    # 4. AI analysis for critical alerts
    for alert in alerts:
        if alert.analysis['priority'] == 'critical':
            ai_result = perplexity.analyze_market_event(alert)
            alert.ai_analysis = ai_result
    
    # 5. Send notifications
    for alert in alerts:
        dispatcher.dispatch(alert)
    
    # 6. Save to database
    save_alerts(alerts)
```

---

## 💡 Key Lessons from TARIFF RADAR

### **What Made It Successful:**

1. **Clear Focus**
   - TARIFF: US-China trade relations
   - MarketPulse: Stock market movements
   - **→ Need specific keywords and sources**

2. **Automated Pipeline**
   - TARIFF: Runs every 5 minutes automatically
   - MarketPulse: Nothing runs automatically
   - **→ Need background jobs**

3. **Smart Filtering**
   - TARIFF: keyword_score > 1.0 → investigate
   - TARIFF: keyword_score > 2.0 → AI analyze
   - **→ Need scoring thresholds**

4. **Multi-Channel Alerts**
   - TARIFF: WeChat (instant) + Telegram (backup) + Email (digest)
   - MarketPulse: None
   - **→ Need WhatsApp + Telegram + Dashboard**

5. **Clear Data Model**
   - TARIFF: Articles have scores, keywords, AI analysis
   - MarketPulse: No alert data model
   - **→ Need alerts table**

---

## 🎯 Summary: The Gap

| Aspect | TARIFF RADAR | MarketPulse | What to Do |
|--------|--------------|-------------|------------|
| **Automation** | ✅ Celery beats | ❌ Manual | Copy tasks.py structure |
| **Data Collection** | ✅ 6 RSS feeds | ❌ 0 active | Copy rss scraper |
| **Intelligence** | ✅ Keywords + AI | ⚠️ AI only | Add keyword engine |
| **Alerts** | ✅ 3 channels | ❌ 0 channels | Copy notify/ folder |
| **Storage** | ✅ Full schema | ⚠️ Basic | Add alerts table |

---

## 📝 Bottom Line

**TARIFF RADAR works because:**
- ✅ It runs automatically (Celery)
- ✅ It knows what to look for (keywords)
- ✅ It tells you when it finds something (alerts)

**MarketPulse doesn't work because:**
- ❌ Nothing runs automatically
- ❌ No keywords defined
- ❌ No alert system

**Solution:**
**Copy the TARIFF RADAR structure and adapt it for financial markets!** 🚀

The code is already there - just need to:
1. Change keywords (tariffs → stocks)
2. Change sources (MOFCOM → Reuters Finance)
3. Change alerts (WeChat → WhatsApp)

**Time to copy:** 1 week  
**Time to build from scratch:** 4-6 weeks

**→ Just copy it! 💡**
