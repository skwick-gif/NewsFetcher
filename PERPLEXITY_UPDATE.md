# 🔄 עדכוני Perplexity - אוקטובר 2025

## ⚠️ שינוי קריטי - שמות מודלים

Perplexity שינתה את שמות המודלים שלה באוקטובר 2025. המודל הישן כבר לא עובד!

### ❌ מודל ישן (לא עובד יותר):
```yaml
model: "llama-3.1-sonar-small-128k-online"
```
**שגיאה:** HTTP 400 - "Invalid model"

### ✅ מודל חדש (2025):
```yaml
model: "sonar"
```
**סטטוס:** עובד מצוין! ✨

---

## 📊 המודלים החדשים של Perplexity

| Model | Purpose | Speed | Cost | Best For |
|-------|---------|-------|------|----------|
| **sonar** | Fast search with grounding | ⚡⚡⚡ | $ | Quick factual queries, real-time data |
| **sonar-pro** | Advanced search | ⚡⚡ | $$ | Complex queries, follow-ups |
| **sonar-reasoning** | Problem-solving with CoT | ⚡ | $$$ | Multi-step analysis, deep thinking |
| **sonar-deep-research** | Comprehensive reports | 🐌 | $$$$ | In-depth research projects |

**בחרנו:** `sonar` - מהיר, זול, מושלם לניתוח פיננסי בזמן אמת

---

## 🔧 קבצים שעודכנו

### 1. ✅ MarketPulse (מערכת פיננסית חדשה)
```
✅ MarketPulse/app/config.yaml - מודל עודכן ל-"sonar"
✅ MarketPulse/app/financial/perplexity_analyzer.py - מודל עודכן
✅ MarketPulse/test_perplexity_direct.py - מודל עודכן
```

### 2. ✅ TARIFF RADAR (מערכת קיימת)
```
✅ tariff-radar/config.yaml - מודל עודכן ל-"sonar"
```

**שני הפרויקטים עכשיו משתמשים באותו מודל מעודכן!**

---

## 🧪 בדיקות שבוצעו

### Test #1: Direct API Call
```bash
py test_perplexity_direct.py
```
**תוצאה:** ✅ HTTP 200 - SUCCESS!
- קיבלנו ניתוח מלא של AAPL
- 11 מקורות (citations)
- 500 tokens תשובה
- זמן תגובה: ~2 שניות

### Test #2: Full Module
```bash
py app/financial/perplexity_analyzer.py
```
**תוצאה:** ✅ Perplexity client initialized
- Model: sonar
- API key: מזוהה
- Timeout: 30 שניות
- מוכן לשימוש ✨

---

## 📝 מה חדש במערכת?

### 1. **קובץ הגדרות מקיף** (`config.yaml`)
- 🎯 **פרומפטים ניתנים לעריכה** - כל הפרומפטים במקום אחד
- 🔧 **8 סוגי ניתוחים** - מניות, אירועים, FDA, גאופוליטיקה
- 🌍 **תמיכה בעברית** - ניתן להגדיר פרומפטים בעברית
- ⚙️ **פרמטרים מתקדמים** - temperature, max_tokens, timeouts

### 2. **מדריך עריכת פרומפטים** (`PROMPTS_GUIDE.md`)
- 📖 הסבר מפורט איך לערוך כל פרומפט
- 💡 דוגמאות לשינויים נפוצים
- ✅ Do's and Don'ts
- 🔄 מדריך בדיקה

### 3. **מקורות מידע מורחבים** (`DATA_SOURCES.md`)
- 📰 Financial news (Reuters, Bloomberg, WSJ, CNBC)
- 🏛️ Regulatory (SEC, FDA, USPTO, FTC)
- 🌍 Geopolitical (Reuters World, BBC)
- 💬 Social media (Twitter, Reddit)
- 📊 Market data (Yahoo, Alpha Vantage)

---

## 🚀 איך משתמשים בפרומפטים?

### דוגמה 1: ניתוח מניה פשוט
```python
from app.financial.perplexity_analyzer import PerplexityFinancialAnalyzer

analyzer = PerplexityFinancialAnalyzer()
insights = await analyzer.get_stock_insights("AAPL")

# התשובה תכלול:
# - analysis: הניתוח המלא מ-Perplexity
# - citations: מקורות המידע (עד 5)
# - timestamp: מתי בוצע הניתוח
# - model: איזה מודל שימש
```

### דוגמה 2: עריכת פרומפט
פתח: `MarketPulse/app/config.yaml`

מצא:
```yaml
prompts:
  stock_analysis:
    user_template: |
      Analyze {symbol} stock and provide...
```

שנה ל:
```yaml
prompts:
  stock_analysis:
    user_template: |
      נתח את המניה {symbol} ותן לי:
      1. כיוון המחיר - עולה/יורדת/רוחבי
      2. המלצה - קנה/מכור/המתן
      3. יעד מחיר ו-stop loss
      4. רמת ביטחון
```

שמור ותריץ שוב - הפרומפט עודכן!

---

## 🎯 המלצות שימוש

### לניתוח יומיומי מהיר:
```yaml
model: "sonar"           # מהיר וזול
temperature: 0.2         # עקבי
max_tokens: 500          # תמציתי
```

### למחקר מעמיק:
```yaml
model: "sonar-pro"       # יותר מתקדם
temperature: 0.3         # מעט יותר יצירתי
max_tokens: 2000         # מפורט
```

### לחשיבה מורכבת:
```yaml
model: "sonar-reasoning" # עם Chain of Thought
temperature: 0.2         # מדויק
max_tokens: 1500         # מקום לחשיבה
```

---

## ✅ סטטוס נוכחי

| Component | Status | Notes |
|-----------|--------|-------|
| Perplexity API | ✅ עובד | מודל `sonar` מעודכן |
| MarketPulse Config | ✅ מוכן | 8 פרומפטים זמינים |
| TARIFF RADAR | ✅ מעודכן | מודל `sonar` |
| Documentation | ✅ מלא | PROMPTS_GUIDE.md |
| Data Sources | ✅ מוגדר | DATA_SOURCES.md |

---

## 🔜 הצעדים הבאים

1. **עדכן פרומפטים לפי צורך שלך** - ערוך `config.yaml`
2. **בדוק שהכל עובד** - הרץ `test_perplexity_direct.py`
3. **התחל להשתמש במערכת** - `py app/main_production_enhanced.py`

---

## 📚 קישורים מועילים

- [Perplexity API Docs](https://docs.perplexity.ai/getting-started/models)
- [PROMPTS_GUIDE.md](./PROMPTS_GUIDE.md) - מדריך מלא לעריכת פרומפטים
- [DATA_SOURCES.md](./DATA_SOURCES.md) - מקורות המידע של המערכת

---

**תאריך עדכון:** 19 אוקטובר 2025  
**גרסה:** MarketPulse v1.0  
**סטטוס:** ✅ Production Ready
