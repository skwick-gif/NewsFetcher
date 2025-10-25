# מודול RL (Reינפורסמנט לרנינג) — תיעוד מרכזי

מסמך זה מרכז את כל התכנון והביצוע לשילוב RL מעל Progressive ML הקיים, כולל ארכיטקטורה, מבנה תיקיות, תכנית עבודה, נתונים היסטוריים, UI, ושיקולי בטיחות. יעד סופי: סוכן שמבצע מסחר בזמן אמת, נתחיל ב‑Offline → Forward → Paper (IBKR) → Production.

## תוכן עניינים

- [מטרות וסקירה](#מטרות-וסקירה)
- [ארכיטקטורה](#ארכיטקטורה)
  - [Environment (Gym-like)](#environment-gym-like)
  - [Action/Observation/Reward/Episode](#actionobservationrewardepisode)
  - [שילוב Progressive ML](#שילוב-progressive-ml)
- [מבנה תיקיות וקבצים](#מבנה-תיקיות-וקבצים)
- [תכנית עבודה וסטטוסים](#תכנית-עבודה-וסטטוסים)
- [היסטוריית נתונים והעשרה חינמית](#היסטוריית-נתונים-והעשרה-חינמית)
 - [היסטוריית נתונים והעשרה חינמית](#היסטוריית-נתונים-והעשרה-חינמית)
 - [מוכנות נתונים ל‑RL (Spec מלא)](#מוכנות-נתונים-ל-rl-spec-מלא)
- [UI ולוחות](#ui-ולוחות)
- [בטיחות ו‑Guardrails](#בטיחות-ו-guardrails)
- [תלויות וספריות](#תלויות-וספריות)
- [How-To קצר (כשנהיה מוכנים)](#how-to-קצר-כשנהיה-מוכנים)

---

## מטרות וסקירה

- אימון סוכן RL על נתונים היסטוריים מקומיים מ־`stock_data/` (ללא תלות בברוקר).
- שימוש ב‑Progressive ML כסיגנלים עשירים בתצפית: BUY/SELL/HOLD, Confidence, Expected Return, SL/TP, VIX.
- הערכה מול Benchmarks: Buy&Hold ו”חוק Progressive פשוט”.
- Paper Trading (IBKR) כשיהיה חיבור: הזמנות, פוזיציות, PnL, ותיעוד החלטות.
- UI ייעודי קליל שלא מעמיס על הדשבורד.

היתרון: RL לומד “מתי וכמה לסמוך” על הסיגנלים בהתאם למשטרי שוק, עלויות, וסיכון — ומוסיף שכבת קבלת החלטות דינמית מעל ה‑Transformer/LSTM/CNN.

## ארכיטקטורה

### Environment (Gym-like)
- מחלקה בסגנון Gym/Gymnasium עם reset/step.
- מקור נתונים: `stock_data/<SYMBOL>/{_price,_indicators}.csv` + יציאות Progressive Predictor בזמן היסטורי.
- סימולציה של עלויות/סליפג' וניהול פורטפוליו.

### Action/Observation/Reward/Episode
- Observation (מצב):
  - חלון טכני אחרון (T×F) מאותם פיצ'רים המשמשים ל‑Progressive.
  - לכל אופק 1/7/30: signal, confidence (כולל calibration), expected_return, capped, SL/TP, ו‑VIX.
  - מצב פורטפוליו: פוזיציה, Cash, Equity, Unrealized PnL.
- Action (פעולה):
  - MVP: דיסקרטי {0, 0.5, 1} (שטוח / חצי לונג / לונג מלא). בהמשך רציף ושורט.
- Reward (תגמול):
  - r_t = ΔPnL − עלויות (עמלות/סליפג') − קנסות (drawdown/turnover).
- Episode:
  - חלון זמן לכל סמל (או אוסף סמלים) עם reset בתחילת החלון.

### שילוב Progressive ML
- Predictor מספק תכונות עשירות בתצפית ולא “כופה” פעולה.
- Warm‑Start אופציונלי: Imitation/Behavior Cloning של כלל Progressive בתחילת האימון, ואז RL משפר.
- שמירה על ה‑Champion והערכה Walk‑Forward, בדומה למערכת הקיימת.

## מבנה תיקיות וקבצים

```
rl/
  envs/
    market_env.py            # Environment למסחר (Gym-like) — reset/step, ניהול מצב
  data_adapters/
    local_stock_data.py      # קריאה מ-stock_data, מיזוג מחירים/אינדיקטורים, הכנה לתצפיות
  training/
    train_ppo.py             # מריץ אימון Offline (PPO) — יופעל עם SB3 כשנוסיף תלות
  live/
    paper_trader.py          # ריצה במצב נייר (IBKR בהמשך דרך שכבת Broker)
  config/
    rl_config.yaml           # חלון, פיצ'רים, עלויות, היפרים, נתיבים
  README_RL.md               # המסמך הזה
```

קבצים קיימים מחוץ ל‑rl:
- `app/templates/rl_dashboard.html` — דף RL קליל.
- `app/static/js/rl.js` — משיכת סטטוס.
- `app/main_realtime.py` — מסלולים: `GET /rl`, `GET /api/rl/status` (Placeholder בטוח).

אחריות לפי קובץ:
- market_env.py — לוגיקת סימולציה/פורטפוליו/עלויות/תגמול.
- local_stock_data.py — טעינת CSV, סידור חלונות, חיבור Progressive ל‑obs.
- train_ppo.py — אתחול Env(s), PPO, callbacks, שמירת checkpoints/מדדים.
- paper_trader.py — תזמור סוכן חי, הזמנות/מצבים/יומן החלטות (IBKR בהמשך).
- rl_config.yaml — פרמטרים ולא קוד, כדי שאפשר לכייל בלי לשנות מודולים.

## תכנית עבודה וסטטוסים

אגדה: ✅ הושלם | 🟡 בתהליך | 🔴 טרם התחיל

- ✅ שלד מודול RL, דף UI מינימלי, ו‑API סטטוס בטוח.
- 🔴 אפיון מלא Environment (obs/action/reward/episodes, עלויות, DD guards).
- 🔴 Data Adapter אמיתי ל‑`stock_data` + איחוד עם יציאות Progressive.
- 🔴 MVP אימון Offline (SB3 PPO) + Benchmarks (BH/Progressive Rule) + גרפים.
- 🔴 Walk‑Forward/Forward Evaluation מחמירה ושמירת "Champion RL Policy".
- 🔴 Paper Trading (IBKR) — שכבת Broker, הזמנות, סטטוס, הגנות בזמן אמת.
- 🔴 לוח RL בדשבורד (או דף ייעודי): PnL, פוזיציות, החלטות, Start/Stop/Reset.
- 🔴 Guardrails מתקדמים (Max trades/day, Kill‑Switch, Limits לכל נייר).

נעדכן סטטוסים בצד לאורך הדרך.

## היסטוריית נתונים והעשרה חינמית

מקור עיקרי: `stock_data/`. להשלמה/ריענון חינמי (לפי תנאי שימוש):
- yfinance — OHLCV, Dividends, Splits. נפוץ ונוח.
- Stooq או Yahoo CSV endpoints — גיבוי במקרה של חסימות.
- Alpha Vantage (חינם מוגבל) — אופציה למנות קטנות (Rate limit).
- אינדיקטורים — חישוב מקומי (חבילת `ta` או קוד קיים).
- תנודתיות — HV מחישוב מקומי; כבר יש VIX בתשתית.

נבנה כלי קטן (data_enrich) במידת הצורך להשלמת חסרים, מבלי לשנות את ה‑API כלפי RL.

### מוכנות נתונים ל‑RL (Spec מלא)

לקבלת פירוט מלא, סכמה, חוזי סקריפטים, וצעדי QA/ולידציה עבור כל שכבות הנתונים (PIT, יקום סמלים, Pricing, Progressive as‑of, ועוד) — ראו את המסמך: `docs/RL_DATA_READINESS.md`.

## UI ולוחות

- כדי לא להעמיס על הדשבורד: דף `/rl` ייעודי (קיים), עם הרחבה עתידית.
- בהמשך: כרטיס קטן בדשבורד הראשי שיציג סטטוס/קישור, והדף הייעודי ישאר ל‑deep‑dive.
- שדות מוצגים (בעתיד): מצב, פוזיציות, PnL, החלטות אחרונות, סיבת החלטה (Tooltip), כפתורי Start/Stop/Reset.

## בטיחות ו‑Guardrails

- Caps על גודל פוזיציה; Max trades/day.
- DD guards יומיים/כוללים; חסמי הפסד לעסקה; Kill‑Switch.
- לוג החלטות מלא, עם סיבת החלטה (features+signals) לצורך Audit.
- Gate ל‑Production רק אם עומדים בספים (Sharpe/Sortino, DD, יציבות).

## תלויות וספריות

כרגע — ללא תלויות כבדות (שלדים בלבד). כשנתקדם:
- Stable‑Baselines3, Gymnasium — אימון PPO.
- alpaca‑trade‑api או IB‑insync — שכבת ברוקר (IBKR Paper בהמשך).
- ta, numpy/pandas — עיבוד אינדיקטורים ו‑OHLCV.

נוסיף ל‑`requirements.txt` רק כשנריץ בפועל.

## How-To קצר (כשנהיה מוכנים)

- אימון Offline (דוגמא עתידית):
  1) עדכן `rl/config/rl_config.yaml` (חלון, עלויות, נתיבים).
  2) הרץ `rl/training/train_ppo.py` — ייצר מודל ותוצאות.
- הערכה/Forward: סקריפט `eval` עם Walk‑Forward.
- Paper Trading: `rl/live/paper_trader.py` לאחר חיבור שכבת IBKR.

---

שאלות/שינויים: זהו המסמך המרכזי. נעדכן אותו בכל שלב עבור סטטוסים, החלטות תכנון, וקישורים לקוד.
