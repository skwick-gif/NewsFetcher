# דו"ח שגיאות/פערים קריטיים בממשק + דרישות לתיקון (כולל IBKR)

תאריך: 2025-10-31

מסמך זה מרכז את כל הבעיות הקריטיות שמצאתי בלשוניות ML / RL / SCANNER / STRATEGY ואת מה שנדרש כדי שהן יעבדו בצורה מלאה. תחת כל בעיה מצוינים גם הקבצים שדורשים בדיקה/טיפול. בנוסף מצורפת תכולה להפעלת חיבור IBKR כחלק ממימוש Agent למסחר בלשונית ה‑RL.

---

## תכנית עבודה (צ'ק-ליסט) — לא מוחק שום מידע, רק מוסיף מעקב סטטוסים

אגדת סטטוסים: ✅ הושלם | 🟡 בתהליך | ❌ טרם בוצע

- [ ] ❌ סעיף 1 — ML Backtesting (Frontend)
  - [ ] לממש פונקציות JS חסרות ב־`app/static/js/progressive-ml.js` (startBacktesting, startAutoBacktesting, stopBacktesting, loadBacktestHistory, runChampionForwardTest)
  - [ ] לחווט לנתיבי Flask Proxy: `/api/ml/progressive/backtest*`, `/api/ml/progressive/champions/*`, `/api/ml/progressive/history`
  - [ ] לאשש חוזי נתונים וזמני timeout מול ה‑Backend (FastAPI)
  - [ ] להשלים UI: מצב התקדמות, היסטוריה, Champion, Forward Test

- [ ] ❌ סעיף 2 — תלות ב‑FastAPI Backend
  - [ ] לאשר שה־`FASTAPI_BACKEND` מוגדר נכון וה‑Backend פעיל
  - [ ] (אופציונלי) להוסיף בדיקת בריאות/אזהרה ב‑UI/Flask כש‑Backend לא זמין

- [ ] ❌ סעיף 3 — טעינת Blueprints בשקט
  - [ ] להוסיף לוג אזהרה על כשלי import/רישום ב־`app/server.py`
  - [ ] לאמת אילו Blueprints נטענו בפועל בעת עליית השרת
  - [ ] להסיר/לתקן רישום שגוי של `system.py` (FastAPI בלבד); ליצור Proxy Flask נפרד במידת הצורך

- [ ] ❌ סעיף 4 — Legacy Dashboard: /api/stats ו‑/api/alerts/active
  - [ ] להחליט אם תומכים בדשבורד הישן; אם כן —
    - [ ] ליצור `app/routes/system_proxy.py` עם פרוקסי ל‑/api/stats ו‑/api/alerts/active
    - [ ] לרשום את ה‑Blueprint החדש ב־`app/server.py` עם לוג תקין
    - [ ] לעדכן התבנית במידת הצורך

- [ ] ❌ סעיף 5 — SCANNER חוזי API
  - [ ] לאשש זרימת filter/train/scan/status/top ב‑Backend וההתאמה ל‑UI
  - [ ] לתאם פורמטי תשובה במידת הצורך

- [ ] ❌ סעיף 6 — STRATEGY ותלות בנתונים
  - [ ] להריץ Ensure/Ensure All ליצירת קבצי price נדרשים
  - [ ] לוודא backtest תקין עבור סימבולי דוגמה (e.g., AAPL)

- [ ] ❌ סעיף 7 — FINANCIAL 500s
  - [ ] לחקור 500 ב‑`/api/financial/geopolitical-risks` ו‑`/api/financial/market-indices` בצד ה‑Backend
  - [ ] (אופציונלי) להעלות timeout/logging בפרוקסי לצורכי דיבוג

- [ ] ❌ סעיף 8 — IBKR Agent (Paper → Live)
  - [ ] לבחור ארכיטקטורה: Python ib_insync (מועדף) או C# bridge
  - [ ] Backend (FastAPI): לממש מודול Broker (`/api/broker/ibkr/*`: connect, status, order, cancel, positions, account)
  - [ ] Flask: ליצור `app/routes/ibkr.py` כ‑Proxy ל‑Backend
  - [ ] UI (RL): להוסיף בלוק IBKR (שדות חיבור, Connect/Disconnect, Start/Stop, הזמנות/פוזיציות)
  - [ ] קונפיג: env/config.yaml; להתחיל מ‑Paper ולשלב אישורי פעולה כפולים ב‑Live

- [ ] ❌ סעיף 9 — Ensure/Data Ops
  - [ ] לשמר UI התקדמות; לאשר ריצה אינקרמנטלית
  - [ ] ניטור לוגים וזמני ריצה של הורדות

הערה: פרטי הרקע המלאים לכל סעיף (תיאור, סיבות, קבצים נוגעים) מופיעים בהמשך המסמך — התוכן המקורי נשמר במלואו.

## 1) ML — כפתורי Backtesting בלשונית RL אינם ממומשים בצד ה‑Frontend (שבירת UI)

- סימפטום: הכפתורים "Start", "Auto (Plan & Run)", "Stop", "Refresh" (History) ו‑"Run Forward Test" תחת "Advanced Backtesting Training" לא עושים כלום וזורקים שגיאת JS בקונסול.
- סיבה: בדף `app/templates/rl_dashboard.html` נקראות פונקציות שאינן קיימות (startBacktesting, startAutoBacktesting, stopBacktesting, loadBacktestHistory, runChampionForwardTest). הקובץ `app/static/js/progressive-ml.js` מספק רק: initProgressiveML, getProgressivePrediction, startProgressiveTraining.
- מה נדרש כדי שיעבוד:
  - להוסיף מימוש JS לפונקציות החסרות (קריאות ל‑/api/ml/progressive/backtest*, champions, history וכו'), כולל ניהול מצב (progress bar, history list, champion card, forward test).
  - לאשש את פורמט התשובות של ה‑Backend (FastAPI) בנתיבים `ml/progressive/*` כדי לעדכן UI בהתאם.
- קבצים לבדיקה/טיפול:
  - Frontend: `app/templates/rl_dashboard.html` (קריאות הפונקציות), `app/static/js/progressive-ml.js` (להוסיף מימוש פונקציות Backtest/History/Champion).
  - Proxy Flask: `app/routes/ml.py` (המסלולים קיימים; לוודא שממשק ה‑API וה‑timeouts תואמים למה שה‑UI מצפה — למשל backtest עשוי לדרוש timeout גבוה).
  - Backend (לידיעה): מודולי FastAPI ל‑`/api/ml/progressive/backtest*`, `/api/ml/progressive/champions/*` (מחוץ לרפו זה). 

---

## 2) תלות מלאה ב‑FastAPI Backend עבור רוב לשוניות RL/ML/SCANNER (כשלא רץ — הכל נשבר)

- סימפטום: פעולות רבות מחזירות 503/שגיאה אם ה‑Backend לא זמין או שגוי (FASTAPI_BACKEND).
- סיבה: מרבית המסלולים ב‑RL/ML/SCANNER הם Proxy ל‑Backend דרך `proxy_to_backend`.
- מה נדרש כדי שיעבוד:
  - לוודא שה‑Backend רץ וזמין ב‑URL הנכון. משתנה סביבה: `FASTAPI_BACKEND` (ברירת מחדל: `http://localhost:8000`).
  - להוסיף בדיקת בריאות/אזהרה ב‑UI או בסרבר Flask שמתריעה אם ה‑Backend לא זמין (אופציונלי אך מומלץ).
- קבצים לבדיקה/טיפול:
  - קונפיג: `app/config/runtime.py` (קריאת FASTAPI_BACKEND).
  - Proxy: `app/utils/proxy.py` (timeouts, handling).
  - Blueprints: `app/routes/rl.py`, `app/routes/ml.py`, `app/routes/scanner.py` (נתיבים פרוקסי).
  - Health (UI): `app/server.py` (`/health` מחזיר את ה‑backend base url; אפשר להרחיב בדיקת reachability — אופציונלי).

---

## 3) טעינת Blueprints ב‑Flask בשקט (try/except) — הסתרת כשלים ברישום ראוטים

- סימפטום: אם קובץ Blueprint לא קיים/שגוי, הרישום נכשל בשקט; ה‑UI יקבל 404 מבלי שהשרת יזרוק אזהרה בהפעלה.
- סיבה: `app/server.py` עוטף רישום Blueprints ב‑try/except ריק.
- מה נדרש כדי שיעבוד:
  - לאתר Blueprints חסרים/שגויים; לפחות לשלב לוג אזהרה בזמן רישום כושל (לראות בדיוק איזה Blueprint לא עלה).
  - דגש: `app/routes/system.py` הוא קובץ FastAPI (APIRouter) ולא Flask Blueprint — ולכן לא יכול להיטען ל‑Flask כפי שמנסה ה‑server.
- קבצים לבדיקה/טיפול:
  - `app/server.py` (להוסיף לוג אזהרה על כשלי import; לבדוק אילו Blueprints באמת נטענו).
  - `app/routes/system.py` (FastAPI בלבד — לשקול יצירת קובץ Flask Proxy נפרד ל‑Stats/Alerts אם צריך במסך legacy).

---

## 4) דשבורד Legacy ("/") — קריאות /api/stats ו‑/api/alerts/active לא פרוקסי ב‑Flask (404)

- סימפטום: במסך הדשבורד הישן נטענים /api/stats ו‑/api/alerts/active — לא קיימים ב‑Flask, ולכן 404.
- סיבה: ה‑Routes הללו קיימים ב‑FastAPI (מערכת), אך לא נוצר Proxy מקביל ב‑Flask.
- מה נדרש כדי שיעבוד:
  - ליצור נתיבי Proxy ב‑Flask עבור /api/stats ו‑/api/alerts/active, או להסתיר/להסיר שימוש בהם מה‑UI אם הדשבורד הישן לא נדרש.
- קבצים לבדיקה/טיפול:
  - UI: `templates/dashboard.html` (או `app/templates/dashboard.html` אם בשימוש).
  - Flask Proxy חדש: קובץ חדש למשל `app/routes/system_proxy.py` עם ראוטים ל‑/api/stats ו‑/api/alerts/active דרך `proxy_to_backend`.
  - `app/server.py` (רישום ה‑Blueprint החדש עם לוג).

---

## 5) SCANNER — ה‑UI תקין, אך תלוי ב‑Backend לבריאות filter/train/status/top

- מצב: דף `app/templates/scanner/scanner.html` קורא למסלולים קיימים ב‑Flask Proxy: 
  - Filter: `/api/scanner/filter/run|status|results` 
  - Train: `/api/scanner/train/*` 
  - Scan: `/api/scanner/run`, `/api/scanner/status`, `/api/scanner/top`
- מה נדרש כדי שיעבוד:
  - לוודא שה‑Backend אכן מספק את הזרימה המלאה (כולל ETA/status תקין) ושפורמט התשובה תואם לציפיית ה‑UI.
- קבצים לבדיקה/טיפול:
  - UI: `app/templates/scanner/scanner.html`
  - Proxy: `app/routes/scanner.py`
  - Backend (לידיעה): מסלולי scanner/ai. 

---

## 6) STRATEGY — עובד מקומית ב‑Flask, תלוי בנוכחות נתוני price CSV

- מצב: `app/routes/strategy.py` מחשב אינדיקטורים, מריץ אסטרטגיה ומחזיר סדרות/מדדים. ה‑UI (`app/templates/strategy/lab.html`) נשען על price CSV ב‑`stock_data/SYM/SYM_price.csv`.
- סימפטום אפשרי: "Backtest failed" אם אין נתונים.
- מה נדרש כדי שיעבוד:
  - להריץ Ensure (בלשונית RL) או לודא קבצי price קיימים עבור הסימבולים המבוקשים.
- קבצים לבדיקה/טיפול:
  - UI: `app/templates/strategy/lab.html`
  - לוגיקה: `app/routes/strategy.py`
  - נתונים: `stock_data/<SYM>/<SYM>_price.csv`

---

## 7) FINANCIAL — 500 מה‑Backend בחלק מהנתיבים

- סימפטום: לפי המצב שתועד, `/api/financial/geopolitical-risks` ו‑`/api/financial/market-indices` מחזירים 500.
- מצב קיים: ב‑Flask קיימים פרוקסי (בקובץ `app/routes/financial.py`), אך השגיאה מגיעה מה‑Backend.
- מה נדרש כדי שיעבוד:
  - לחקור בצד ה‑Backend את מקור ה‑500 (סכימות נתונים/אישורי גישה/זמינות שירותים חיצוניים). בצד Flask אין צורך שינוי אם ה‑Backend יתוקן.
- קבצים לבדיקה/טיפול:
  - Proxy: `app/routes/financial.py` (קיים; ייתכן צורך ב‑timeout/logging גבוה יותר לצורך דיבוג).
  - Backend (לידיעה): מודולי FastAPI ל‑financial.

---

## 8) IBKR — השמשת חיבור וסגירת מימוש Agent למסחר בלשונית RL

- מצב נוכחי:
  - ב‑`app/server.py` יש ניסיון לרשום `app.routes.ibkr` (Blueprint) — אך קובץ כזה לא נמצא ברפו. בגלל try/except שקט, זה לא נחשף בלוגים. קיימת תיקייה `IBKR/` ברוט (כולל `csharp_bridge/`), אך אין אינטגרציית Flask/FastAPI פעילה.
- מטרת יעד:
  - הפעלת Agent למסחר אוטונומי בלשונית RL (לצד Paper). 
  - ה‑Agent יבצע הנפקת הוראות ל‑IBKR (TWS/IB Gateway), שליפת פוזיציות/יתרות/מצב הזמנות, ומעקב סטטוס.
- חלופות ארכיטקטורה לבחירה (אחת):
  1) אינטגרציה ישירה ב‑FastAPI עם `ib_insync` (מועדף לפשטות Python‑only):
     - שירות Backend לניהול חיבור (connect/disconnect), פרמטרי סביבה (host/port/clientId), פעולות: place_order, cancel, positions, account, market_data.
     - חשיפת REST נקיות: `/api/broker/ibkr/connect|disconnect|status|order|cancel|positions|account`.
  2) שימוש בגשר קיים (C# bridge) תחת `IBKR/csharp_bridge` בתור Service חיצוני:
     - ה‑Backend ידבר עם הגשר ב‑HTTP/WebSocket/Named Pipe; דורש סטנדרטיזציה של פרוטוקול.
- מה נדרש כדי שיעבוד (תכולה מוצעת):
  1) Backend (FastAPI):
     - מודול Broker IBKR חדש (למשל `app/routes/broker_ibkr.py` ב‑FastAPI) עם ניהול חיבור, הזמנות, סטטוסים, ופוליסת ריטריים/Timeouts.
     - מחלקת Service מחוסנת (reconnect, heartbeat), לוגים ברורים, ומיפוי שגיאות IBKR לידידותיות UI.
  2) Flask Proxy (UI Server):
     - `app/routes/ibkr.py` — Blueprint חדש שיפרוקסי ל‑FastAPI (`/api/broker/ibkr/*`).
  3) UI (בלשונית RL):
     - כרטיס "Live (IBKR)" נוסף לצד Paper: שדות התחברות (Host/Port/ClientId), מצב חיבור, כפתורי Connect/Disconnect.
     - פעולות Agent: Start/Stop (סשן חי), תצוגת הזמנות פעילות/בוצע/שגוי, ופידבק על פוזיציות/יתרות בזמן אמת.
     - אבטחה: banner "Paper/Live" ברור, ו‑Confirmations כפולים במצב Live.
  4) קונפיג והפעלה:
     - משתני סביבה/קובץ `config.yaml` לעדכון פרטי TWS/IB Gateway.
     - המלצה: להתחיל ב‑Paper Trading Account ב‑TWS.
- קבצים לבדיקה/טיפול:
  - Flask: ליצור `app/routes/ibkr.py` (לא קיים כרגע) — Proxy ל‑Backend.
  - FastAPI Backend: מודולי Broker חדשים (מחוץ לרפו זה אם ה‑Backend מופרד). אם ה‑Backend כאן — ליצור `app/routes/broker_ibkr.py` (FastAPI) + Service.
  - UI: `app/templates/rl_dashboard.html` — הוספת בלוק UI ל‑IBKR (חיבור/סטטוס/פעולות), או דף משנה.
  - קונפיג: `app/config/config.yaml` (אם משתמשים), או env vars.
  - תשתית קיימת: `IBKR/` (לבדוק אם ה‑bridge בשימוש/לנטוש לטובת Python‑only).

---

## 9) ניהול נתונים (Ensure) — תלויות חיצוניות וזמני ריצה

- מצב: פעולת Ensure רצה ב‑Flask (thread), מושכת yfinance, כותבת price ו‑indicators, ומפיקה גם news_features בסיסי.
- רגישויות:
  - תלות ברשת/קצבי הורדה; CSVs קיימים/שדות עמודות (הקוד כבר מגן עם נורמליזציות).
  - רצוי לעדכן UI בהתקדמות (כבר קיים), ולאפשר ריצה אינקרמנטלית (Ensure All עושה זאת).
- קבצים רלוונטיים:
  - `app/server.py` (ensure endpoints, רקע, לוגים)
  - `stock_data/` (תוצרי הורדה)

---

## 10) סיכום פעולות קצר (לביצוע לאחר אישורך)

- ML Backtesting (Frontend): לממש פונקציות JS חסרות ולהשלים wiring למסלולי `/api/ml/progressive/backtest*` + "Champion/Forward test".
- FastAPI Backend: לוודא שכל מסלולי progressive ML שה‑UI מבקש קיימים ועומדים בחוזה הנתונים.
- Proxy ל‑Stats/Alerts (Legacy): להוסיף ראוטים ב‑Flask או להוריד שימוש ב‑UI הישן.
- Blueprints: להוסיף לוגים על כשלי רישום ולהסיר רישום שגוי של `system_bp` (שאינו Flask BP).
- IBKR: לבחור ארכיטקטורה (מומלץ ib_insync), להוסיף מודול Broker ב‑Backend + Proxy + UI ללשונית RL.
- קונפיג: לאשר `FASTAPI_BACKEND`; אופציונלית הוספת בדיקת בריאות/אזהרה ב‑UI.

---

## נספח: מיפוי קבצים לפי לשונית

- RL: `app/templates/rl_dashboard.html`, `app/static/js/progressive-ml.js`, `app/static/js/rl.js`, `app/routes/rl.py`, `app/routes/rl_tools.py`
- ML (progressive): `app/routes/ml.py` (Proxy), פונקציות חסרות ב‑JS (להוסיף ב‑`app/static/js/progressive-ml.js`)
- SCANNER: `app/templates/scanner/scanner.html`, `app/routes/scanner.py`
- STRATEGY: `app/templates/strategy/lab.html`, `app/routes/strategy.py`, נתונים ב‑`stock_data/`
- Financial: `app/routes/financial.py` (Proxy)
- System/Legacy: `templates/dashboard.html` (אם בשימוש), `app/server.py`, (Proxy חדש מוצע ל‑Stats/Alerts)
- Proxy תשתיתי: `app/utils/proxy.py`, קונפיג: `app/config/runtime.py`

---

אם תרצה — אחל מיידית בשדרוג ה‑Frontend ל‑Backtesting ML, או בהקמה מדורגת של IBKR Agent (Paper תחילה, ואז Live), לפי עדיפות שתגדיר.