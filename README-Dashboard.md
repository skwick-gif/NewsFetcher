# 📈 MarketPulse Dashboard

## Real-Time Financial Intelligence Platform

דשבורד מתקדם לניתוח שוק ההון בזמן אמת עם יכולות AI ואינטליגנציה פיננסית.

### ✨ תכונות עיקריות

#### 📊 **Market Overview**
- **מדדי שוק בזמן אמת**: S&P 500, NASDAQ, DOW JONES, VIX
- **מד סנטימנט השוק**: 75% Bullish עם מדד חזותי אינטראקטיבי
- **גרפי מחירים דינמיים**: Chart.js עם מסגרות זמן שונות (1D, 1W, 1M, 3M)

#### 🔥 **Hot Financial Alerts**
- התראות בזמן אמת על תנועות מחירים משמעותיות
- הודעות חדשות ועדכונים קריטיים
- מעקב אחר מניות חמות (AAPL, TSLA, NVDA, etc.)

#### ⚠️ **Risk Intelligence Dashboard**
- **מוניטור סיכונים גיאופוליטיים**: מעקב אחר השפעת מתיחויות סחר ופוליטיקה
- **ניתוח ביצועי סקטורים**: טכנולוגיה, פיננסים, בריאות, תעשייה
- **מדד רמת סיכון**: 7.2/10 HIGH עם פירוט השפעות

#### 🧠 **AI-Powered Trading Insights**
- **המלצות AI**: BUY/SELL/HOLD עם רמת ביטחון
- **מחיר יעד חכם**: ניבוי מחירי יעד מבוסס למידת מכונה
- **ניתוח סנטימנט**: עיבוד חדשות ורשתות חברתיות

#### 🤖 **AI Analysis Tab**
- **סורק מניות AI**: ניתוח אוטומטי של 10 מניות מובילות
- **סטטוס מודלי AI**: Machine Learning, Neural Networks, Time Series
- **מדדי ביצועים**: דיוק ניבוי 78%, דיוק כיוון 85%

### 🎨 **עיצוב ואופי חזותי**

#### **ערכת צבעים**
- **Background**: גרדיאנט כהה (#0d1421 → #1a2332 → #2d3748)
- **Green (#10b981)**: עליות חיוביות
- **Red (#ef4444)**: ירידות שליליות  
- **Blue (#3b82f6)**: מידע כללי
- **Gold (#ffd700)**: כותרות וטקסטים חשובים
- **Purple (#8b5cf6)**: תכונות AI

#### **אפקטים חזותיים**
- **Glass Morphism**: רקעים שקופים עם blur effects
- **Responsive Design**: תמיכה מלאה במובייל וטאבלט
- **Smooth Animations**: מעברים חלקים ואנימציות CSS

### 🏗️ **ארכיטקטורה טכנית**

```
MarketPulse/
├── app/
│   ├── templates/
│   │   └── dashboard.html      # דשבורד ראשי
│   ├── server.py              # Flask server
│   └── static/                # קבצים סטטיים (עתידי)
├── requirements-dashboard.txt  # תלויות Python
└── README.md                  # תיעוד זה
```

### 🚀 **הרצה מקומית**

#### **התקנה**
```bash
# התקנת תלויות
pip install -r requirements-dashboard.txt

# הרצת השרת
cd MarketPulse/app
python server.py
```

#### **גישה לדשבורד**
- **Dashboard**: http://localhost:5000
- **Health Check**: http://localhost:5000/health  
- **API Data**: http://localhost:5000/api/market-data

### 📱 **מבנה Navigation**

#### **4 Tabs עיקריים**
1. **Overview** 🏠
   - מדדי שוק + גרף מחירים
   - התראות חמות + סטטיסטיקות
   - דשבורד סיכונים + AI insights

2. **AI Analysis** 🤖
   - סורק מניות AI
   - סטטוס מודלים
   - מדדי ביצועים

3. **Articles** 📄
   - מאמרים ואנליזות (עתידי)

4. **Settings** ⚙️
   - הגדרות משתמש (עתידי)

### 🔧 **תכונות טכניות**

#### **JavaScript Modules**
```javascript
- market-data.js      // נתוני שוק
- chart-manager.js    // ניהול גרפים
- alerts-manager.js   // ניהול התראות
- ml-scanner.js       // סורק AI
- websocket-client.js // תקשורת בזמן אמת
// orchestration מתבצע inline בתוך dashboard.html (אין main.js)
// settings מנוהלות inline בתוך dashboard.html (אין settings-manager.js)
```

#### **Chart.js Integration**
- גרפי קווים אינטראקטיביים
- מסגרות זמן מרובות
- עיצוב מותאם לנושא כהה
- Responsive design

### 📊 **נתונים ו-APIs**

#### **Market Data Structure**
```json
{
  "indices": {
    "SP500": {"value": 4327.78, "change": 0.8},
    "NASDAQ": {"value": 13431.34, "change": 1.2},
    "DOW": {"value": 34098.10, "change": -0.3},
    "VIX": {"value": 18.42, "change": 2.1}
  },
  "sentiment": 75,
  "alerts": [...]
}
```

### 🎯 **מטרות העיצוב**

#### **חוויית משתמש**
- **אינטואיטיבי**: ניווט פשוט וברור
- **מהיר**: טעינה מהירה ואינטראקציה חלקה  
- **מידע עשיר**: מרביב נתונים בצורה מארגנת
- **חזותי**: גרפיקה מתקדמת ואפקטים

#### **פונקציונליות**
- **Real-time**: עדכונים בזמן אמת
- **AI-Powered**: המלצות חכמות  
- **Risk-Aware**: ניהול סיכונים מתקדם
- **Mobile-Ready**: תמיכה מלאה במובייל

### 🔮 **תכנון עתידי**

#### **שלב 1** ✅ (הושלם)
- דשבורד בסיסי עם כל התכונות הויזואליות
- מבנה HTML/CSS מתקדם
- JavaScript בסיסי לניווט

#### **שלב 2** 🔄 (בתכנון)
- חיבור לנתוני שוק אמיתיים
- WebSocket לעדכונים בזמן אמת  
- מודלי JavaScript נפרדים

#### **שלב 3** 📋 (עתידי)
- מערכת התראות מתקדמת
- AI models לניבוי מחירים
- ניהול פורטפוליו

### 💡 **השראה ועיצוב**

הדשבורד מעוצב בהשראת פלטפורמות trading מתקדמות כמו:
- Bloomberg Terminal
- TradingView  
- Robinhood Dashboard
- Interactive Brokers

עם דגש על:
- **נקייה וסדר**: מידע מאורגן ברור
- **צבעוניות חכמה**: ירוק/אדום לחיוביות/שליליות
- **אינטראקטיביות**: אלמנטים ניתנים לקליק
- **מודרניות**: עיצוב Material Design + Glass morphism

---

### 📧 **תמיכה וקשר**

לשאלות ועזרה נוספת, אנא פנו למפתח הפרויקט.

**MarketPulse Dashboard v1.0** 🚀
*Real-Time Financial Intelligence Platform*