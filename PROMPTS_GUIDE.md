# 📝 עריכת פרומפטים במערכת MarketPulse

## 🎯 מיקום הפרומפטים

כל הפרומפטים של המערכת נמצאים בקובץ אחד מרכזי:
```
MarketPulse/app/config.yaml
```

## 🔧 מבנה קובץ ההגדרות

הקובץ מחולק לחלקים:

### 1️⃣ **הגדרות AI/LLM** (שורות 7-16)
```yaml
llm:
  provider: "perplexity"      # ספק ה-AI
  model: "sonar"              # מודל Perplexity 2025
  temperature: 0.2            # רמת יצירתיות (0.0-1.0)
  max_tokens: 1000            # אורך מקסימלי של תשובה
  timeout: 30                 # זמן המתנה מקסימלי
```

**פרמטרים לשינוי:**
- `temperature`: נמוך יותר (0.1-0.3) = תשובות יותר עקביות ומדויקות
- `temperature`: גבוה יותר (0.5-0.9) = תשובות יותר יצירתיות ומגוונות
- `max_tokens`: קצר יותר (500) = תשובות תמציתיות
- `max_tokens`: ארוך יותר (2000) = תשובות מפורטות

### 2️⃣ **פרומפטים** (שורות 18-250)

כל פרומפט מורכב מ-2 חלקים:

#### **System Prompt** - הגדרת התפקיד של ה-AI
```yaml
system: |
  You are a professional financial analyst with expertise in...
```

#### **User Template** - התבנית לשאלה הספציפית
```yaml
user_template: |
  Analyze {symbol} stock and provide:
  1. Current Outlook...
  2. Recent News...
```

---

## 📋 סוגי הפרומפטים הזמינים

### 1. **Stock Analysis** (ניתוח מניה)
```yaml
prompts:
  stock_analysis:
    system: "You are a professional financial analyst..."
    user_template: "Analyze {symbol} stock..."
```

**משתנים זמינים:**
- `{symbol}` - סימול המניה (למשל: AAPL, MSFT)

**שימוש במערכת:**
- כאשר משתמש מבקש ניתוח מניה
- בדשבורד בלחיצה על "Analyze"
- ב-API endpoint: `/api/analyze/{symbol}`

**דוגמה לעריכה:**
אם רוצה תשובות יותר טכניות:
```yaml
user_template: |
  Provide TECHNICAL analysis for {symbol}:
  1. **Support/Resistance** - Key price levels
  2. **Indicators** - RSI, MACD, Moving Averages
  3. **Chart Patterns** - Triangles, Head&Shoulders, etc
  4. **Volume Analysis** - Accumulation/Distribution
  5. **Price Targets** - Short-term and long-term
  6. **Entry/Exit Strategy** - Specific prices with stop-loss
```

---

### 2. **Market Event Analysis** (ניתוח אירוע שוק)
```yaml
prompts:
  market_event:
    system: "You are a senior market strategist..."
    user_template: "Analyze this market event: {event_description}"
```

**משתנים זמינים:**
- `{event_description}` - תיאור האירוע

**שימוש במערכת:**
- כאשר יש חדשות גדולות (Fed, earnings, geopolitics)
- בהתראות אוטומטיות על אירועים חריגים

**דוגמה לעריכה:**
אם רוצה פוקוס על אסטרטגיית מסחר:
```yaml
user_template: |
  Event: {event_description}
  
  Give me TRADING PLAN:
  1. **Immediate Action** - What to do now (buy/sell/hedge)
  2. **Risk Management** - Stop losses and position sizes
  3. **Time Horizon** - Day trade, swing trade, or long-term
  4. **Specific Trades** - Exact symbols with entry prices
  5. **Profit Targets** - Where to take profits
  6. **Worst Case Scenario** - Exit plan if wrong
```

---

### 3. **News Sentiment** (ניתוח סנטימנט חדשות)
```yaml
prompts:
  news_sentiment:
    system: "You are a financial news analyst..."
    user_template: "Analyze these headlines about {symbol}: {headlines}"
```

**משתנים זמינים:**
- `{symbol}` - סימול המניה
- `{headlines}` - רשימת כותרות חדשות

**שימוש במערכת:**
- ניתוח אוטומטי של חדשות שמתקבלות מ-RSS feeds
- כל 5-15 דקות במהלך שעות המסחר

**דוגמה לעריכה:**
אם רוצה תשובה מובנית בעברית:
```yaml
user_template: |
  נתח את הכותרות הבאות על {symbol}:
  {headlines}
  
  תן לי:
  1. **סנטימנט כללי** - שורי/דובי/נייטרלי עם ציון 0-100
  2. **השפעה על המחיר** - צפוי לעלות/לרדת/להישאר X%
  3. **תובנות מפתח** - 3 נקודות חשובות מהחדשות
  4. **סיכונים** - מה עלול להשתבש
  5. **המלצת מסחר** - קנה/מכור/המתן עם נימוקים
  6. **רמת ביטחון** - נמוכה/בינונית/גבוהה
```

---

### 4. **Sector Analysis** (ניתוח מגזר)
```yaml
prompts:
  sector_analysis:
    system: "You are a sector specialist..."
    user_template: "Analyze the {sector} sector..."
```

**משתנים זמינים:**
- `{sector}` - שם המגזר (Technology, Healthcare, Finance, Energy...)

**דוגמה לעריכה:**
אם רוצה השוואה תחרותית:
```yaml
user_template: |
  מגזר: {sector}
  
  תן לי comparative analysis:
  1. **Top 3 Stocks** - הטובים ביותר במגזר + למה
  2. **Bottom 3 Stocks** - הגרועים ביותר + למה
  3. **Value vs Growth** - איזה גישה עובדת עכשיו במגזר
  4. **Catalysts Calendar** - אירועים חשובים בחודשיים הקרובים
  5. **Sector Rotation** - כסף נכנס למגזר או יוצא? מאיפה לאן?
  6. **Best Pick** - המניה הטובה ביותר במגזר כרגע
```

---

### 5. **Earnings Preview** (תחזית רווחים)
```yaml
prompts:
  earnings_preview:
    system: "You are an earnings analyst..."
    user_template: "Prepare earnings preview for {symbol} on {date}..."
```

**משתנים זמינים:**
- `{symbol}` - סימול המניה
- `{date}` - תאריך הדיווח

**דוגמה לעריכה:**
אם רוצה אסטרטגיית אופציות:
```yaml
user_template: |
  {symbol} מדווחת רווחים ב-{date}
  
  Options Strategy:
  1. **Expected Move** - תנועת מחיר צפויה (implied volatility)
  2. **Call Strategy** - אם צפוי beat - איזה strikes לקנות
  3. **Put Strategy** - אם צפוי miss - איזה strikes לקנות
  4. **Straddle/Strangle** - האם כדאי לשחק volatility
  5. **Risk/Reward** - מה הסיכון מקסימלי מול רווח פוטנציאלי
  6. **Exit Plan** - מתי לסגור לפני/אחרי הדיווח
```

---

### 6. **Risk Assessment** (הערכת סיכונים)
```yaml
prompts:
  risk_assessment:
    system: "You are a risk management specialist..."
    user_template: "Assess risks for {symbol} position..."
```

**דוגמה לעריכה:**
אם רוצה בדיקת תיק השקעות:
```yaml
user_template: |
  בדוק סיכונים עבור {symbol}:
  
  Portfolio Risk Analysis:
  1. **Position Size Risk** - האם הפוזיציה גדולה מדי? כמה % מהתיק?
  2. **Correlation Risk** - למניות אילו אחרות יש קורלציה גבוהה?
  3. **Sector Concentration** - כמה מהתיק במגזר הזה?
  4. **Downside Protection** - איפה לשים stop-loss?
  5. **Hedging Options** - איך לגדר את הפוזיציה (puts, inverse ETF)?
  6. **Maximum Loss** - מה התרחיש הגרוע ביותר בכסף?
```

---

### 7. **FDA Analysis** (ניתוח אישורי FDA)
```yaml
prompts:
  fda_analysis:
    system: "You are a biotech analyst..."
    user_template: "Analyze FDA approval: {drug_name} for {company_symbol}"
```

**משתנים זמינים:**
- `{drug_name}` - שם התרופה
- `{company_symbol}` - סימול החברה

**שימוש במערכת:**
- ניטור אוטומטי של אישורי FDA
- התראות בזמן אמת על אישורים חדשים

---

### 8. **Geopolitical Analysis** (ניתוח גאופוליטי)
```yaml
prompts:
  geopolitical_analysis:
    system: "You are a geopolitical analyst..."
    user_template: "Analyze geopolitical event: {event_description}"
```

**דוגמה לעריכה:**
אם רוצה פוקוס על סחורות:
```yaml
user_template: |
  אירוע גאופוליטי: {event_description}
  
  Commodities & Currency Impact:
  1. **Oil Impact** - מחיר נפט עולה/יורד? איזה חברות נהנות?
  2. **Gold/Safe Havens** - האם לקנות זהב, bonds, USD?
  3. **Currency Moves** - איזה מטבעות נחלשים/מתחזקים?
  4. **Emerging Markets** - השפעה על EM stocks/bonds?
  5. **Defense Stocks** - האם משבר = עליות בנשק/ביטחון?
  6. **Supply Chain** - אילו תעשיות יפגעו בשרשרת אספקה?
```

---

## 🛠️ איך לערוך פרומפט - מדריך צעד אחר צעד

### שלב 1: פתח את קובץ ההגדרות
```bash
code MarketPulse/app/config.yaml
```

### שלב 2: מצא את הפרומפט שרוצה לשנות
חפש לפי שם:
- `stock_analysis` - לניתוח מניות
- `market_event` - לאירועי שוק
- `news_sentiment` - לסנטימנט חדשות
- וכו'

### שלב 3: ערוך את ה-System Prompt (אופציונלי)
```yaml
system: |
  אתה אנליסט פיננסי מקצועי המתמחה ב...
  [כאן תוכל לשנות את התפקיד/המומחיות של ה-AI]
```

### שלב 4: ערוך את ה-User Template (המלצה)
```yaml
user_template: |
  נתח את {symbol} ותן לי:
  1. [מה שאתה רוצה לקבל]
  2. [עוד משהו]
  ...
```

### שלב 5: שמור את הקובץ
- `Ctrl+S` ב-VS Code
- המערכת תטען את הפרומפט החדש בהפעלה הבאה

### שלב 6: בדוק שהפרומפט עובד
```bash
cd MarketPulse
$env:PERPLEXITY_API_KEY="YOUR_KEY"
py app/financial/perplexity_analyzer.py
```

---

## 💡 טיפים לכתיבת פרומפטים טובים

### ✅ עשה:
1. **היה ספציפי** - "תן לי 3 מניות טכנולוגיה" במקום "תן לי מניות"
2. **השתמש במבנה** - רשימות ממוספרות, כדורים, כותרות
3. **הגדר פורמט** - "תשובה בעברית", "JSON format", "טבלה"
4. **בקש דוגמאות** - "כולל סימולי מניות ספציפיים"
5. **קבע גבולות** - "עד 3 פסקאות", "לא יותר מ-5 מניות"

### ❌ אל תעשה:
1. **שאלות מעורפלות** - "תן לי מידע" (מידע על מה?)
2. **יותר מדי בקשות** - אל תבקש 20 דברים בפרומפט אחד
3. **סתירות** - "היה שמרן אבל אגרסיבי" - תבחר אחד
4. **הנחות** - אל תניח שה-AI יודע את ההקשר שלך

---

## 🔄 דוגמאות לשינויים נפוצים

### רוצה תשובות בעברית?
```yaml
system: |
  אתה אנליסט פיננסי ישראלי המתמחה בשוק האמריקאי.
  תן תשובות בעברית ברורה וממוקדת.
  השתמש במינוחים מקצועיים בעברית.

user_template: |
  נתח את המניה {symbol} ותן תשובה מקצועית בעברית...
```

### רוצה תשובות קצרות וממוקדות?
```yaml
user_template: |
  {symbol} - תן לי תשובה קצרה (עד 3 משפטים):
  1. כיוון: עולה/יורדת/רוחבי
  2. המלצה: קנה/מכור/המתן
  3. נימוק: סיבה אחת מרכזית
```

### רוצה פורמט JSON מובנה?
```yaml
user_template: |
  Analyze {symbol} and return ONLY valid JSON:
  {{
    "direction": "up/down/sideways",
    "recommendation": "buy/sell/hold",
    "price_target": 123.45,
    "stop_loss": 100.00,
    "confidence": 0.85,
    "reasoning": "brief explanation"
  }}
  
  NO additional text outside JSON!
```

### רוצה השוואה בין מניות?
```yaml
user_template: |
  השווה {symbol1} מול {symbol2}:
  
  | קטגוריה | {symbol1} | {symbol2} | מי עדיף? |
  |----------|-----------|-----------|----------|
  | גדילה   |           |           |          |
  | רווחיות |           |           |          |
  | סיכון   |           |           |          |
  | מומנטום |           |           |          |
  
  המלצה סופית: [מי לקנות ולמה]
```

---

## 🧪 בדיקת פרומפטים

לאחר שינוי פרומפט, בדוק אותו:

### בדיקה מהירה:
```bash
cd MarketPulse
py test_perplexity_direct.py
```

### בדיקה מלאה:
```bash
py app/financial/perplexity_analyzer.py
```

### בדיקה דרך השרת:
```bash
# התחל שרת
py app/main_production_enhanced.py

# בדפדפן
http://localhost:8000/docs
# נסה endpoint: POST /api/analyze/AAPL
```

---

## 📊 ניטור ביצועי פרומפטים

המערכת שומרת logs של כל קריאה ל-AI:
```
MarketPulse/logs/marketpulse.log
```

חפש שורות:
```
✅ Got Perplexity insights for AAPL from 11 sources
❌ Failed to get insights for AAPL: timeout
```

בדוק:
- **זמן תגובה** - האם התשובה מהירה מספיק?
- **איכות** - האם התשובות מועילות?
- **שגיאות** - האם יש פרומפטים שנכשלים?

---

## 🚀 פרומפטים מתקדמים

### Chain of Thought (חשיבה שלב-אחר-שלב)
```yaml
user_template: |
  Let's think step by step about {symbol}:
  
  Step 1: Current Price and Trend
  [Analyze price action]
  
  Step 2: Fundamental Analysis
  [Check earnings, revenue, margins]
  
  Step 3: News and Sentiment
  [What are people saying?]
  
  Step 4: Technical Levels
  [Support, resistance, indicators]
  
  Step 5: Final Recommendation
  [Buy/Sell/Hold with confidence score]
```

### Few-Shot Learning (דוגמאות)
```yaml
user_template: |
  Analyze {symbol} like these examples:
  
  Example 1:
  AAPL - Bullish (0.85 confidence)
  - Strong iPhone sales
  - Services growth accelerating
  - Target: $260, Stop: $240
  
  Example 2:
  TSLA - Bearish (0.70 confidence)
  - Delivery miss
  - Competition increasing
  - Target: $180, Stop: $220
  
  Now analyze {symbol}:
```

---

## 📞 צור קשר / תמיכה

אם אתה לא בטוח איך לשנות משהו:
1. קרא את המדריך הזה שוב
2. בדוק את הדוגמאות
3. נסה שינוי קטן ובדוק
4. אל תפחד לשחק עם הפרומפטים!

---

**עדכון אחרון:** 19 אוקטובר 2025  
**גרסת Perplexity:** 2025 (מודל: `sonar`)  
**קובץ הגדרות:** `MarketPulse/app/config.yaml`
