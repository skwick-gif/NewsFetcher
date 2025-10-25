# תכנית עבודה - מערכת RL + Transformers למסחר מניות

## סקירה כללית
נבנה מערכת למידה חיזוקית (RL) המשתמשת ב-Transformers לחיזוי מחירי מניות, עם יכולת מסחר אוטומטי.

## נתונים נדרשים

### 1. נתוני מחירים היסטוריים (יש לנו)
- **מה יש**: OHLCV data עבור 5000+ מניות מ-Yahoo Finance
- **מיקום**: `stock_data/{TICKER}/{TICKER}_price.csv`
- **מקור**: Yahoo Finance (חינם)
- **תדירות**: יומית

### 2. מדדי שוק (חלקית יש לנו)
- **SPY (S&P 500 ETF)**: יש לנו ✓
- **QQQ (Nasdaq 100 ETF)**: יש לנו ✓
- **VIX (CBOE Volatility Index)**: חסר ❌
  - **מקור**: Yahoo Finance (`^VIX`)
  - **שימוש**: מדד תנודתיות השוק

### 3. אינדיקטורים טכניים (חסרים)
- **מה צריך**: RSI, MACD, Bollinger Bands, Moving Averages
- **מקור**: חישוב מ-נתוני המחירים הקיימים
- **ספריות**: pandas-ta או talib
- **תדירות**: יומית

### 4. נתוני יסוד (יש לנו חלקית)
- **מה יש**: P/E, EPS, ROE, מידע פיננסי מ-Yahoo, Finviz, Macrotrends
- **מיקום**: `stock_data/{TICKER}/{TICKER}_advanced.json`
- **מקור**: Yahoo Finance, Finviz, Macrotrends (חינם)

### 5. ניתוח סנטימנט (חסר)
- **מקור**: NewsAPI או Alpha Vantage
- **מה צריך**: ציוצי טוויטר, חדשות פיננסיות
- **שימוש**: מדד סנטימנט יומי

### 6. נתונים כלכליים (חסרים)
- **מקור**: FRED (Federal Reserve Economic Data) - חינם
- **מה צריך**: GDP, אבטלה, ריבית, CPI
- **שימוש**: השפעת נתונים מאקרו על השוק

## תכנית יישום מפורטת

### שלב 1: הוספת VIX
**קובץ לעריכה**: `stocks.py`
**מה להוסיף**:
```python
# הוסף לרשימת המניות המיוחדות
vix_ticker = "^VIX"
```

**פונקציה חדשה**:
```python
def update_vix_data():
    # הורד נתוני VIX מ-Yahoo Finance
    # שמור ב-stock_data/VIX/VIX_price.csv
```

### שלב 2: חישוב אינדיקטורים טכניים
**קובץ חדש**: `compute_indicators.py`
**תוכן**:
```python
import pandas as pd
import talib
from stocks import all_tickers, Config

def compute_technical_indicators(ticker):
    # קרא נתוני מחירים
    # חשב RSI(14), MACD, Bollinger Bands
    # שמור לקובץ נפרד
```

### שלב 3: הוספת סנטימנט
**קובץ לעריכה**: `stocks.py`
**מה להוסיף**:
```python
import requests

def get_news_sentiment(ticker):
    # NewsAPI integration
    # חשב ציון סנטימנט מ-כותרות חדשות
```

### שלב 4: הוספת נתונים כלכליים
**קובץ חדש**: `economic_data.py`
**מקור**: https://fred.stlouisfed.org/docs/api/fred/
**מה צריך**:
- GDP (GDPC1)
- Unemployment Rate (UNRATE)
- Federal Funds Rate (FEDFUNDS)
- CPI (CPIAUCSL)

### שלב 5: יצירת datasets משולבים
**קובץ חדש**: `create_datasets.py`
**פלט**: קבצי CSV עם:
- מחירים + אינדיקטורים + יסוד + סנטימנט + כלכלה
- לפי תאריך (לצורך RL training)

## מקורות נתונים חינמיים

### Yahoo Finance (יfinance)
- מניות, ETFs, מדדים
- נתונים היסטוריים
- חינם ללא API key

### FRED (Federal Reserve)
- נתונים כלכליים
- חינם, API פשוט
- https://fred.stlouisfed.org/docs/api/fred/

### NewsAPI
- חדשות פיננסיות
- חינם עד 100 בקשות/יום
- https://newsapi.org/

### Alpha Vantage (חלופה)
- סנטימנט וחדשות
- חינם עם מגבלות
- https://www.alphavantage.co/

## סדר עדיפויות

1. **VIX** - קל להוסיף, חיוני למדד תנודתיות
2. **אינדיקטורים טכניים** - חישוב מהנתונים הקיימים
3. **נתונים כלכליים** - השפעה חזקה על השוק
4. **סנטימנט** - מורכב יותר, אך מוסיף ערך
5. **RL implementation** - אחרי שכל הנתונים מוכנים

## הערות טכניות

- כל הנתונים יישמרו בפורמט CSV/JSON לתאימות
- עדכון יומי אוטומטי
- שימוש ב-pandas לניקוי ומיזוג נתונים
- PyTorch ל-RL, transformers למודלים הקיימים