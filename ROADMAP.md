# 🎯 MarketPulse - ניתוח מצב נוכחי ותוכנית שדרוג

## 📊 מה יש לנו היום? (סטטוס נוכחי)

### ✅ **מה עובד:**
1. **Perplexity AI** - מודל `sonar` מעודכן ועובד
2. **Yahoo Finance** - נתוני מניות בזמן אמת
3. **Alpha Vantage** - גיבוי לנתונים פיננסיים
4. **Twitter API** - סנטימנט ממדיה חברתית
5. **Reddit API** - סנטימנט מקהילות משקיעים
6. **TensorFlow + Scikit-learn** - תשתית ML מותקנת

### ⚠️ **מה חסר / לא עובד:**

#### 1. **אין מערכת התראות אמיתית!**
```
❌ אין ניטור אוטומטי 24/7
❌ אין התראות Telegram/WhatsApp
❌ אין התראות בדשבורד בזמן אמת
❌ אין הגדרת alerts לפי מילות מפתח
```

#### 2. **אין ניטור חדשות אמיתי!**
```
❌ אין fetch אוטומטי מ-RSS feeds
❌ אין סריקת אתרים פיננסיים
❌ אין ניתוח FDA approvals בזמן אמת
❌ אין מעקב אחרי SEC filings
```

#### 3. **אין מילות מפתח מוגדרות!**
```
❌ לא ברור מה המילים החשובות
❌ אין scoring למילות מפתח
❌ אין התראות לפי keywords
```

#### 4. **אין אוטומציה!**
```
❌ צריך להריץ ידנית
❌ אין scheduler שעובד
❌ אין background jobs
```

---

## 🔍 לאילו אתרים אנחנו צריכים לפנות?

### 📰 **חדשות פיננסיות - Real-time**
| אתר | URL | מה מחפשים | עדכון |
|-----|-----|-----------|--------|
| **Reuters Business** | `feeds.reuters.com/reuters/businessNews` | מניות, M&A, רווחים | כל 5 דק' |
| **Bloomberg Markets** | `bloomberg.com/feed` | שוק, מאקרו, Fed | כל 5 דק' |
| **WSJ Markets** | `feeds.a.dj.com/rss/RSSMarketsMain.xml` | ניתוח עומק | כל 5 דק' |
| **CNBC** | `cnbc.com/id/100003114/device/rss` | Breaking news | כל 3 דק' |
| **MarketWatch** | `marketwatch.com/rss/marketpulse` | מגמות | כל 10 דק' |
| **Seeking Alpha** | `seekingalpha.com/feed.xml` | מחקרים | כל 15 דק' |
| **Yahoo Finance** | `finance.yahoo.com/rss` | כללי | כל 10 דק' |

### 🏛️ **גופים רגולטוריים - Official**
| גוף | URL | מה מחפשים | עדכון |
|-----|-----|-----------|--------|
| **SEC EDGAR** | `sec.gov/cgi-bin/browse-edgar` | 10-K, 10-Q, 8-K, Form 4 | כל שעה |
| **FDA Approvals** | `fda.gov/rss` | אישורי תרופות | פעמיים ביום |
| **USPTO Patents** | `uspto.gov` | פטנטים חדשים | יומי |
| **FTC** | `ftc.gov/news-events/rss` | אנטי-טראסט, M&A | יומי |

### 💬 **מדיה חברתית - Sentiment**
| פלטפורמה | מה מחפשים | כמה | תדירות |
|-----------|-----------|-----|----------|
| **Twitter/X** | $SYMBOL hashtags, טרנדים | 100 tweets/symbol | כל 15 דק' |
| **Reddit** | r/wallstreetbets, r/stocks | 50 posts | כל 15 דק' |
| **StockTwits** | Sentiment על מניות | חם | כל 10 דק' |

### 🌍 **גאופוליטיקה - Market Movers**
| אתר | מה מחפשים | למה חשוב |
|-----|-----------|----------|
| **Reuters World** | מלחמות, בחירות, סנקציות | השפעה על אנרגיה, defense |
| **BBC Business** | בנקים מרכזיים, ריביות | כיוון שוק |
| **Bloomberg Politics** | מדיניות, תקציב | מגזרים |

---

## 🔑 מילות מפתח - מה אנחנו מחפשים?

### 📈 **Bullish Keywords** (ציון +1 עד +3)
```yaml
very_bullish: ["+3 points"]
  - "record earnings"
  - "beat expectations"
  - "raised guidance"
  - "breakthrough"
  - "FDA approval"
  - "major deal"
  - "surges"
  - "all-time high"
  
bullish: ["+2 points"]
  - "positive outlook"
  - "growth accelerating"
  - "market share gains"
  - "strong demand"
  - "upgrade"
  - "buy rating"
  - "exceeds forecast"
  
moderately_bullish: ["+1 point"]
  - "increase"
  - "improved"
  - "expansion"
  - "partnership"
  - "investment"
```

### 📉 **Bearish Keywords** (ציון -1 עד -3)
```yaml
very_bearish: ["-3 points"]
  - "bankruptcy"
  - "fraud"
  - "investigation"
  - "scandal"
  - "massive losses"
  - "crashes"
  - "suspended"
  - "recall"
  
bearish: ["-2 points"]
  - "miss expectations"
  - "lowered guidance"
  - "downgrade"
  - "sell rating"
  - "loses market share"
  - "regulatory issues"
  - "lawsuit"
  
moderately_bearish: ["-1 point"]
  - "decline"
  - "weakness"
  - "concern"
  - "challenge"
  - "delay"
```

### 🎯 **High-Impact Events** (התראה מיידית!)
```yaml
immediate_alert:
  - "FDA approval"
  - "merger"
  - "acquisition"
  - "bankruptcy"
  - "earnings beat"
  - "earnings miss"
  - "product recall"
  - "CEO resign"
  - "war"
  - "rate hike"
  - "rate cut"
```

### 🏢 **Sector-Specific Keywords**
```yaml
tech:
  - "AI breakthrough"
  - "chip shortage"
  - "data breach"
  - "cloud growth"
  - "semiconductor"
  
pharma:
  - "clinical trial"
  - "FDA approval"
  - "patent expiry"
  - "generic competition"
  - "drug pricing"
  
finance:
  - "interest rate"
  - "credit rating"
  - "loan default"
  - "stress test"
  - "capital ratio"
  
energy:
  - "oil price"
  - "OPEC"
  - "production cut"
  - "pipeline"
  - "sanctions"
```

---

## 🚨 מערכת התראות - איך זה צריך לעבוד?

### 1️⃣ **התראות בזמן אמת (Real-time)**

#### **A. התראות בדשבורד**
```
┌─────────────────────────────────────────┐
│  🔴 LIVE ALERTS                          │
├─────────────────────────────────────────┤
│  🚨 AAPL +5.2% - FDA approval news      │
│  ⏰ 2 minutes ago                        │
│  💡 AI: Strong Buy (0.92 confidence)    │
│  📊 Price: $252 → $265                  │
│  [View Details] [Dismiss]               │
├─────────────────────────────────────────┤
│  ⚠️  TSLA -3.8% - Delivery miss         │
│  ⏰ 15 minutes ago                       │
│  💡 AI: Neutral (0.68 confidence)       │
│  📊 Price: $245 → $236                  │
│  [View Details] [Dismiss]               │
└─────────────────────────────────────────┘
```

**מה צריך:**
- WebSocket connection לדשבורד
- Pop-up notifications בדפדפן
- Sound alert (אופציונלי)
- Counter של alerts (🔔 5)

#### **B. התראות WhatsApp** (המלצה!)
```
📱 WhatsApp Message:
━━━━━━━━━━━━━━━━━━━
🚨 *MARKET ALERT*

*AAPL* | +5.2% 📈
Price: $252 → $265

*Reason:* FDA approved new drug
*Sentiment:* 🟢 Very Bullish
*AI Confidence:* 92%

*Recommendation:* STRONG BUY
━━━━━━━━━━━━━━━━━━━
MarketPulse | 14:32
```

**איך מיישמים:**
- Twilio WhatsApp API
- או WhatsApp Business API
- Template messages

#### **C. התראות Telegram** (כבר מוגדר!)
```
🤖 Telegram Bot:
━━━━━━━━━━━━━━━━━━━
🔔 *Alert: AAPL*

📈 Price: +5.2% ($265)
📰 News: FDA approval
🎯 Signal: STRONG BUY
🤖 AI Score: 0.92

*Action:*
• Entry: $265
• Target: $280
• Stop: $250

🔗 [Full Analysis]
━━━━━━━━━━━━━━━━━━━
```

#### **D. Email Alerts** (גיבוי)
```
Subject: 🚨 MarketPulse Alert: AAPL +5.2%

AAPL is up 5.2% following FDA approval news.

Current Price: $265 (+$13)
AI Recommendation: STRONG BUY
Confidence: 92%

Key Points:
• FDA approved breakthrough drug
• Twitter sentiment: 85% bullish
• Reddit mentions: +340%
• Analyst upgrades: 3 new

[View Full Dashboard]
```

---

## 🎯 אסטרטגיית התראות - מתי להתריע?

### **Level 1: Critical Alerts** 🔴 (מיידי!)
```yaml
triggers:
  price_move: ±5% תוך 15 דקות
  volume_spike: 5x מהממוצע
  breaking_news: 
    - FDA approval/rejection
    - M&A announcement
    - CEO resignation
    - Earnings surprise >10%
  sentiment_flip: שינוי מ-bullish ל-bearish או להיפך
  
notification:
  - WhatsApp: ✅ מיידי
  - Telegram: ✅ מיידי
  - Dashboard: ✅ Pop-up + Sound
  - Email: ✅ בוא
```

### **Level 2: Important Alerts** 🟡 (תוך 5 דקות)
```yaml
triggers:
  price_move: ±3% תוך שעה
  volume_spike: 3x מהממוצע
  news_impact: High
  analyst_rating: Upgrade/Downgrade
  social_buzz: מגמת מנטיון חזקה
  
notification:
  - WhatsApp: ⚠️ אם מופעל
  - Telegram: ✅
  - Dashboard: ✅ Badge
  - Email: ⚠️ אם מופעל
```

### **Level 3: Watch Alerts** 🟢 (סיכום כל שעה)
```yaml
triggers:
  price_move: ±2%
  news_mention: מניה מוזכרת בחדשות
  sector_trend: מגזר עולה/יורד
  technical_signal: RSI, MACD crossover
  
notification:
  - Dashboard: ✅ רשימת watch
  - Email: סיכום שעתי (אופציונלי)
```

---

## 🏗️ ארכיטקטורה חדשה - איך זה יעבוד?

### **Data Collection Layer** (איסוף נתונים)
```
┌─────────────────────────────────────────┐
│  RSS Feed Scrapers (Background Jobs)    │
│  ├─ Reuters      [every 5 min]          │
│  ├─ Bloomberg    [every 5 min]          │
│  ├─ WSJ          [every 5 min]          │
│  ├─ CNBC         [every 3 min]          │
│  ├─ MarketWatch  [every 10 min]         │
│  └─ Seeking Alpha [every 15 min]        │
├─────────────────────────────────────────┤
│  Social Media Monitors                   │
│  ├─ Twitter API  [every 15 min]         │
│  ├─ Reddit API   [every 15 min]         │
│  └─ StockTwits   [every 10 min]         │
├─────────────────────────────────────────┤
│  Regulatory Monitors                     │
│  ├─ SEC EDGAR    [hourly]               │
│  ├─ FDA          [2x daily]             │
│  └─ USPTO        [daily]                │
└─────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────┐
│  Keyword Matching Engine                 │
│  • Scan for bullish/bearish keywords    │
│  • Calculate sentiment score             │
│  • Identify high-impact events           │
│  • Extract stock symbols                 │
└─────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────┐
│  AI Analysis Layer (Perplexity)         │
│  • Analyze news impact                   │
│  • Predict price movement                │
│  • Generate recommendations              │
│  • Calculate confidence score            │
└─────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────┐
│  Alert Decision Engine                   │
│  • Match against user rules              │
│  • Calculate alert priority              │
│  • Deduplicate alerts                    │
│  • Format messages                       │
└─────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────┐
│  Notification Dispatcher                 │
│  ├─ WhatsApp     [critical only]        │
│  ├─ Telegram     [all levels]           │
│  ├─ Dashboard    [WebSocket push]       │
│  └─ Email        [digest]               │
└─────────────────────────────────────────┘
```

---

## 📋 תוכנית פיתוח - מה צריך לעשות?

### **Phase 1: Foundation** (שבוע 1-2) 🏗️

#### 1.1 **Background Jobs System**
```python
# Celery or APScheduler
- RSS feed scrapers (כל 5-15 דק')
- Market data updater (כל דקה)
- Social media monitor (כל 15 דק')
- SEC filing checker (כל שעה)
```

#### 1.2 **Keyword System**
```python
# keywords_engine.py
- Define keyword lists (bullish/bearish)
- Keyword scoring algorithm
- Symbol extraction from text
- Sentiment calculation
```

#### 1.3 **Database Schema**
```sql
CREATE TABLE alerts (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10),
    alert_type VARCHAR(50),
    priority VARCHAR(20), -- critical/important/watch
    title TEXT,
    description TEXT,
    sentiment_score FLOAT,
    ai_confidence FLOAT,
    price_change FLOAT,
    source VARCHAR(100),
    keywords TEXT[],
    created_at TIMESTAMP,
    dismissed BOOLEAN DEFAULT false
);

CREATE TABLE user_alert_rules (
    id SERIAL PRIMARY KEY,
    user_id INT,
    symbol VARCHAR(10),
    alert_on TEXT[], -- price_move, news, sentiment, etc
    min_price_change FLOAT,
    keywords TEXT[],
    notification_channels TEXT[], -- whatsapp, telegram, email
    enabled BOOLEAN DEFAULT true
);
```

---

### **Phase 2: Alert System** (שבוע 3-4) 🚨

#### 2.1 **Alert Engine**
```python
class AlertEngine:
    def process_news(self, article):
        # Extract keywords
        keywords = self.extract_keywords(article)
        
        # Calculate scores
        sentiment = self.calculate_sentiment(keywords)
        priority = self.determine_priority(sentiment, keywords)
        
        # Check user rules
        affected_users = self.match_user_rules(article.symbol, keywords)
        
        # Generate alert
        if priority >= threshold:
            self.create_alert(article, sentiment, priority)
            self.notify_users(affected_users)
```

#### 2.2 **WhatsApp Integration**
```python
# Using Twilio
from twilio.rest import Client

def send_whatsapp_alert(symbol, price_change, reason):
    message = f"""
    🚨 ALERT: {symbol}
    📈 Change: {price_change}%
    📰 {reason}
    """
    client.messages.create(
        from_='whatsapp:+14155238886',
        to='whatsapp:+972XXXXXXXXX',
        body=message
    )
```

#### 2.3 **Dashboard WebSocket**
```javascript
// Real-time alerts in dashboard
const ws = new WebSocket('ws://localhost:8000/alerts');

ws.onmessage = (event) => {
    const alert = JSON.parse(event.data);
    showAlertPopup(alert);
    playSound();
    updateBadge();
};
```

---

### **Phase 3: Intelligence** (שבוע 5-6) 🧠

#### 3.1 **Smart Filtering**
```python
# לא כל ציוץ = התראה!
def should_alert(signal):
    # Check if significant
    if signal.price_change < 3% and signal.volume < 2x:
        return False
    
    # Check if already alerted
    if recently_alerted(signal.symbol, minutes=30):
        return False
    
    # Check AI confidence
    if signal.ai_confidence < 0.70:
        return False
    
    return True
```

#### 3.2 **Perplexity Integration**
```python
# Real analysis for high-impact events
async def analyze_breaking_news(article):
    if article.priority == 'critical':
        ai_analysis = await perplexity.analyze_market_event(
            event_description=article.text
        )
        
        return {
            'impact': ai_analysis['severity'],
            'affected_sectors': ai_analysis['affected_sectors'],
            'recommendation': ai_analysis['recommendation'],
            'confidence': ai_analysis['confidence']
        }
```

#### 3.3 **Pattern Recognition**
```python
# ML model for recurring patterns
def train_alert_model():
    # Features:
    # - Time of day
    # - News source
    # - Keyword combinations
    # - Historical price reaction
    
    # Target:
    # - Was this alert useful? (user feedback)
    
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
```

---

### **Phase 4: Dashboard Enhancement** (שבוע 7-8) 📊

#### 4.1 **Alert Center UI**
```html
<!-- Dashboard section -->
<div class="alert-center">
    <div class="alert-header">
        <h2>🔔 Alerts <span class="badge">5</span></h2>
        <button>Settings</button>
    </div>
    
    <div class="alert-filters">
        <button class="active">All</button>
        <button>Critical 🔴</button>
        <button>Important 🟡</button>
        <button>Watch 🟢</button>
    </div>
    
    <div class="alert-list">
        <!-- Real-time alerts here -->
    </div>
</div>
```

#### 4.2 **Alert Settings Panel**
```html
<div class="alert-settings">
    <h3>Configure Alerts</h3>
    
    <!-- Per symbol -->
    <div class="symbol-alerts">
        <input placeholder="Symbol (e.g., AAPL)">
        
        <label>
            <input type="checkbox"> Price moves ±3%
        </label>
        <label>
            <input type="checkbox"> Breaking news
        </label>
        <label>
            <input type="checkbox"> Social media buzz
        </label>
        
        <select name="notification">
            <option>Dashboard only</option>
            <option>Dashboard + Telegram</option>
            <option>Dashboard + WhatsApp</option>
            <option>All channels</option>
        </select>
    </div>
    
    <!-- Keywords -->
    <div class="keyword-alerts">
        <h4>Custom Keywords</h4>
        <input placeholder="e.g., FDA approval, merger">
        <span class="hint">Get notified when these appear</span>
    </div>
</div>
```

---

## 💰 עלויות ו-API Limits

### **APIs שנצטרך:**
| API | חינמי? | גבול | עלות חודשית |
|-----|--------|------|-------------|
| Yahoo Finance | ✅ | ∞ | $0 |
| Alpha Vantage | ✅ | 500/day | $0 (או $50/חודש לרמה גבוהה) |
| Twitter | ⚠️ | 50 req/15min | $100/חודש (basic) |
| Reddit | ✅ | 60 req/min | $0 |
| Perplexity | 💰 | Pay-per-use | ~$0.005/query (~$50/חודש) |
| Twilio WhatsApp | 💰 | Pay-per-message | $0.005/הודעה (~$10/חודש) |
| Telegram Bot | ✅ | ∞ | $0 |

**סה"כ צפוי:** ~$150-200/חודש לפעולה רצינית

---

## 🎯 סיכום - מה חסר לנו?

### ❌ **לא עובד כרגע:**
1. **אין ניטור אוטומטי** - הכל ידני
2. **אין התראות** - לא WhatsApp, לא Telegram, לא דשבורד
3. **אין סריקת חדשות** - לא באמת קוראים RSS feeds
4. **אין מילות מפתח** - לא יודעים מה לחפש
5. **אין AI analysis בזמן אמת** - Perplexity לא מחובר לזרימה

### ✅ **מה צריך לבנות:**
1. **Background jobs** - Celery/APScheduler
2. **RSS feed scrapers** - לכל אתרי החדשות
3. **Keyword engine** - עם scoring ו-sentiment
4. **Alert system** - priority levels + filtering
5. **WhatsApp integration** - Twilio
6. **Dashboard WebSocket** - real-time alerts
7. **Alert settings UI** - למשתמש
8. **Smart filtering** - לא להציף במסר זבל

---

## 📅 Timeline מוצע

| שבוע | מה בונים | Output |
|------|----------|--------|
| **1-2** | Background jobs + Keywords | רוץ כל 5 דק', מזהה מילות מפתח |
| **3-4** | Alert system + WhatsApp | שולח התראות אמיתיות |
| **5-6** | AI integration + Filtering | Perplexity מנתח אירועים |
| **7-8** | Dashboard + Settings UI | משתמש מגדיר alerts |

**זמן כולל:** ~2 חודשים לפעולה מלאה 🚀

---

## 💡 המלצות

1. **התחל קטן** - תחילה Telegram (חינמי ופשוט)
2. **בדוק value** - לפני WhatsApp ($$$), ודא שהאלגוריתם עובד
3. **User feedback** - תן למשתמש לדרג alerts (useful/spam)
4. **ML over time** - למד מה alertים חשובים באמת
5. **הצף בהדרגה** - תחילה top 10 symbols, אחר כך הרחב

**זה פרויקט רציני - צריך 2-3 חודשי פיתוח מרוכז!** 💪
