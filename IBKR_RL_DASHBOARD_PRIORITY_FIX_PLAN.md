# IBKR + RL_DASHBOARD — תכנית תיקונים לפי דחיפות/קריטיות

תאריך: 2025-11-03

מטרת המסמך: לתעד, לפי סדר עדיפויות, את השינויים המומלצים עבור אינטגרציית IBKR והטאב RL_DASHBOARD — מה לשנות ובאיזה קובץ, ההשפעה על הפונקציונליות, וההשפעה על המשתמש.

הערה: לא מבצע כעת שינויי קוד בפועל. זהו מסמך תכנון ממוקד לביצוע.

---

## 1) אבטחה והרשאות על IBKR/RL endpoints (קריטי מאוד)

- מה לשנות + קבצים:
  - להגן באימות/הרשאות על כל הנתיבים הרגישים:
    - FastAPI: `app/api/routers/ibkr.py`, `app/api/routers/rl.py` — הוספת Depends(get_current_user) ובדיקות roles (למשל trader/admin) לכל: connect/status/account/portfolio/orders/updates.
    - Flask Proxy: `app/routes/ibkr.py`, `app/routes/rl.py` — לאכוף session/auth ולשקול CSRF לטפסי POST.
- השפעה על פונקציונליות:
  - חוסם שימוש לא מורשה בהתחברות ל‑TWS ושליחת הזמנות; מצמצם סיכון תפעולי.
- השפעה על המשתמש:
  - יידרש login; משתמש ללא הרשאות יקבל 401/403. ה‑UI צריך להפנות למסך התחברות/להציג הודעה.

---

## 2) Safety והגבלת קצב להזמנות (קריטי)

- מה לשנות + קבצים:
  - FastAPI: להוסיף Rate limiting לנתיבים: `/api/rl/live/orders`, כל `/api/ibkr/*order*` (לדוגמה שילוב SlowAPI/middleware).
  - `app/api/routers/rl.py` — אימות פרמטרים קפדני בעת יצירת `TradeRequest` (טווחים, ערכים חיוביים, order_type תקף).
- השפעה על פונקציונליות:
  - מונע spam/הזרמת הזמנות לא רצויה; מייצב את המערכת וה‑bridge.
- השפעה על המשתמש:
  - במקרי עומס יקבל 429 עם הנחיה; שימוש רגיל לא יושפע.

---

## 3) עמידות SignalR (reconnect/backoff/מנויים) (גבוה)

- מה לשנות + קבצים:
  - `app/integrations/ibkr_bridge.py`:
    - להוסיף auto‑reconnect עם backoff והחזרת `_subscriptions` לאחר reconnect.
    - לטפל באירועי onclose/onreconnected (signalrcore) ולרשום handlers מחדש.
    - להעביר `verify_ssl` ו‑`skip_negotiation` ל‑config (ב‑`app/config/settings.py`) במקום hard‑coded.
- השפעה על פונקציונליות:
  - זרימת quotes/order updates מתאוששת לאחר ניתוק; פחות "מצב תקוע".
- השפעה על המשתמש:
  - RL_DASHBOARD נשאר מחובר ויציב יותר; פחות ניתוקים מורגשים.

---

## 4) ניטור ובריאות קצה‑לקצה (Backend ⇄ Bridge ⇄ TWS) (גבוה)

- מה לשנות + קבצים:
  - ליצור מסלול בריאות מאוחד:
    - FastAPI: קובץ חדש `app/api/routers/health.py` או הרחבה ב‑`ibkr.py` עבור `/api/health/ibkr` שבודק: FastAPI חי, Python Bridge זמין, C# Bridge מגיב, `isConnected` ל‑TWS, ו‑hub מחובר.
  - Frontend: עדכון `app/templates/rl_dashboard.html` להצגת badge/אייקון health ופרטי כשל (אופציונלי אך מומלץ).
- השפעה על פונקציונליות:
  - דיאגנוסטיקה מהירה; פחות זמן חיפוש תקלות.
- השפעה על המשתמש:
  - רואה בבירור מי נפל (Backend/Bridge/TWS) ומתי חזר.

---

## 5) כיול טיים‑אאוטים ופעולות איטיות (גבוה)

- מה לשנות + קבצים:
  - Flask Proxy: `app/utils/proxy.py` — להגדיל timeouts לנתיבים כבדים (connect/history/portfolio) או להבדיל בין GET/POST "כבדים".
  - Bridge (Python): `app/integrations/ibkr_bridge.py` — לכייל `httpx.Timeout` דרך `IBKRConfig` ולהעלות ערכים במידת הצורך.
  - אופציונלי: להפוך פעולות ארוכות למשימות async עם polling.
- השפעה על פונקציונליות:
  - פחות 503/timeout בפעולות connect/market‑history.
- השפעה על המשתמש:
  - פחות הודעות כשל; ייתכן המתנה מעט ארוכה במקום כשל מיידי.

---

## 6) סטנדרטיזציית שגיאות ו‑UX עקבי (בינוני‑גבוה)

- מה לשנות + קבצים:
  - FastAPI: `app/api/routers/ibkr.py`, `app/api/routers/rl.py` — לאחד מבנה שגיאה סטנדרטי: `{status, code, detail, hint}`.
  - Frontend: `app/static/js/rl.js` — לעדכן טיפול בשגיאות לפי `code`/`detail` ולהציג טוסטים ברורים (e.g., bridge_down, not_connected_to_tws).
- השפעה על פונקציונליות:
  - עקביות ב‑API ובדיווח בעיות בין שכבות.
- השפעה על המשתמש:
  - הודעות ברורות ופעילות מוצעת לתיקון; פחות בלבול.

---

## 7) יישור תאימות נתיבי היסטוריה (בינוני)

- מה לשנות + קבצים:
  - צד C#: לאחד לנתיב `
/api/market/historical`.
  - Python: `app/integrations/ibkr_bridge.py` — לוג אזהרה אם נדרש fallback לנתיב ישן, ולהסיר את ה‑fallback לאחר עדכון ה‑bridge.
- השפעה על פונקציונליות:
  - מונע 404 ספורדיים; תאימות בין רכיבים.
- השפעה על המשתמש:
  - טעינה יציבה לגרפים/דוחות.

---

## 8) קשיחות קונפיג ואבטחה (בינוני)

- מה לשנות + קבצים:
  - `app/config/settings.py` — להוסיף flags כמו: `signalr_verify_ssl`, `signalr_skip_negotiation`, `allowed_origins`, `max_request_timeout` ולקרוא אותם מ‑env/config.
  - לשפר ברירות מחדל ל‑prod (לא `verify_ssl=False`).
- השפעה על פונקציונליות:
  - פחות הפתעות בין סביבות; הקשחת אבטחה ב‑prod.
- השפעה על המשתמש:
  - ללא שינוי מורגש; יציבות גבוהה יותר.

---

## 9) ביטול תלות ב‑CDN לספריות גרפים (בינוני‑נמוך)

- מה לשנות + קבצים:
  - להעתיק Chart.js ומודולים ל‑`app/static/vendor/...` ולעדכן רפרנסים ב‑`app/templates/rl_dashboard.html`.
- השפעה על פונקציונליות:
  - ה‑UI לא תלוי בחיבור אינטרנט לטעינת גרפים.
- השפעה על המשתמש:
  - הדשבורד נטען גם בסביבה מנותקת; אמינות טובה יותר.

---

## 10) שיפור הודעות Proxy (נמוך)

- מה לשנות + קבצים:
  - `app/utils/proxy.py` — להעשיר את הודעות השגיאה: ציון סוג כשל (DNS/timeout/refused) ושם השירות (FastAPI/C# bridge) בשדה `upstream`.
- השפעה על פונקציונליות:
  - דיבוג מהיר יותר של שרשרת הפרוקסי.
- השפעה על המשתמש:
  - הודעות ברורות יותר ב‑UI במקום "שגיאה כללית".

---

## נספח — מיפוי עיקרי של קבצים מעורבים

- Frontend UI:
  - `app/templates/rl_dashboard.html` — תצוגת RL + חיבור IBKR.
  - `app/static/js/rl.js` — סטטוס IBKR, Live Summary, UI toasts.
- Flask Proxy:
  - `app/routes/ibkr.py`, `app/routes/rl.py`, `app/utils/proxy.py` — ניתוב ל‑FastAPI.
- FastAPI:
  - `app/api/routers/ibkr.py`, `app/api/routers/rl.py` — Endpoints IBKR + RL (live, orders, preview, paper, auto‑tune, promotion).
- Bridge (Python):
  - `app/integrations/ibkr_bridge.py` — HTTP+SignalR ל‑C# Bridge.
- קונפיג:
  - `app/config/settings.py` — `IBKRConfig` ופרמטרים נוספים.
- Bridge (C#):
  - `IBKR/csharp_bridge/` — שירות REST+SignalR חיצוני ל‑TWS/IBKR.
