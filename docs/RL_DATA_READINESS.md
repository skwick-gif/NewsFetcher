# מוכנות נתונים היסטוריים ל‑RL (Data Readiness)

מסמך זה מפרט את כל שכבות הנתונים הדרושות כדי לאמן סוכן RL “אמיתי” ללא הטיות וללא דליפות מידע, עם מיקוד ב‑Point‑In‑Time (PIT), איכות, וכשירות לפרודקשן. הדגש כאן הוא על הגדרה מעשית שניתנת ליישום בשלבים, תוך שימוש בנתונים המקומיים שב‑`stock_data/` כהתחלה.

## מטרות

- לבנות מאגר היסטורי עקבי, נקי, ו‑PIT עבור אימון, הערכה, ו‑Walk‑Forward.
- לאפשר העשרת תצפית ב‑Progressive ML “as‑of” לכל בר היסטורי, ללא דליפת עתיד.
- לשמר עקיבות לשחזור: גרסאות, חותמות זמן, מקורות, ולוגים.

## עקרונות יסוד (קריטיים)

1) Point‑In‑Time — כל ערך שנכנס לאימון או סימולציה חייב לשקף ידע שהיה זמין בנקודת הזמן באותה בר. אין שימוש במידע עתידי (כולל חישובי rolling המשתמשים בחלון שמביט קדימה).

2) שרידות ללא הטיות — הימנעות מסרידות‑מדגם (survivorship bias). בניית יקום סמלים לפי “מי היה קיים אז”, לא לפי מי ששרד עד היום.

3) התאמות תאגידיות — Splits/Dividends מתורגמים למחירי Adjusted ול‑OHLCV עקביים; יש לתעד ולוודא.

4) עלויות וסליפג' — מודל עלויות (עמלה, מרווח, סליפג') נקבע מראש ונשמר בצמוד לנתונים/קונפיג.

5) יומני מסחר (Calendars) — שימוש בלוחות מסחר (NYSE/NASDAQ) עם ימי חופש; אין ברים פיקטיביים.

6) גרסאות/שכפול — כל יצירה/עדכון של סט נתונים מתוייגים ב‑version/hash ותאריך יצירה.

## שכבות נתונים (ליבתיות)

- Pricing (Daily/Intraday):
  - שדות מינימום: DateTime (TZ=America/New_York), Open, High, Low, Close, Adj Close, Volume.
  - אחידות עמודות ופורמט תאריכים; אינדקס ייחודי ו‑monotonic.
  - מקור: `stock_data/<SYMBOL>/_price.csv` (קיים), העתק/נורמליזציה ל‑`data/rl/pricing/`.

- Corporate Actions:
  - Dividends, Splits, Mergers. נדרש לוודא שה‑Adjusted מתיישר עם האירועים.
  - מקור: נגזר מ‑yfinance או קבצים משלימים. בשלב ראשון — נשתמש ב‑Adj Close הקיים.

- Indicators/Features טכניים:
  - אם קיימים ב‑`_indicators.csv` — למזג לפי DateTime.
  - אם חסרים — לחשב מקומית (למשל ATR, RSI, EMA) או לדחות לשלב ב'.

- Macro/Market Regime:
  - ויקסים (VIX), תשואת אג"ח 10Y, אינפלציה (CPI), מחרוזות מצב שוק (Bull/Bear/Sideways).
  - מינימום: VIX כבר משולב במערכת — רצוי לכלול בסכמה עם timestamp אחיד.

- Risk‑Free Rate:
  - סדרת יומית (למשל 3M T‑Bill) לחישובי Sharpe; אפשר קבוע זעיר בשלב ראשון.

- Transaction Costs:
  - פרמטרים ב‑config: עמלה לפר‑עסקה/נפח, סליפג' בסיסי (bps), ומיסוי מחוץ לטווח.

- Borrow/Shortability (אופציונלי לשלב ב'):
  - עלות השאלה, זמינות קצר. לשלב MVP לונג‑בלבד אין צורך.

- News/Sentiment (אופציונלי):
  - פיצ'רי סנטימנט אסופים כ‑PIT; לוודא הגעה לפני בר המטרה.

## Progressive ML כ‑PIT Feature

- הפקה "as‑of" לכל בר היסטורי: עבור כל timestamp T, מריצים את ה‑Predictor על חלון עד וכולל T−ε, ושומרים:
  - לכל אופק: signal (BUY/SELL/HOLD), confidence, expected_return, SL/TP, capped, uncertainty.
  - metadata: model_version, data_version, inference_runtime, תצורה.
- אחסון: `data/rl/progressive_signals/<SYMBOL>.csv` (או Parquet), עמודות זמן תואמות ל‑pricing.
- ביצועים: cache לפי (symbol, model_version, data_version). ריצה אינקרמנטלית.

## סכמה מומלצת (CSV ידידותי; Parquet אופציונלי)

- תיקיית בסיס: `data/rl/`
  - `universe.csv` — יקום סמלים PIT: symbol, start_date, end_date, source.
  - `pricing/<SYMBOL>.csv` — עותק/נורמליזציית מחירים אחידה.
  - `indicators/<SYMBOL>.csv` — (אם בנפרד) אינדיקטורים.
  - `progressive_signals/<SYMBOL>.csv` — יציאות Progressive as‑of.
  - `calendars/market_calendar.csv` — ימי מסחר/חופשה.
  - `meta/versions.json` — רישום גרסאות datasets.

הערה: Parquet מהיר וחסכוני אך דורש `pyarrow`. בשלב זה נשארים עם CSV למינימום תלויות.

## בקרת איכות (QA) ובדיקות עקביות

- בדיקות מבניות: כפילויות timestamp, סדר לא מונוטוני, NULLs ב‑OHLCV.
- PIT: בדוק ש‑progressive_signals ב‑T לא מסתכל מעבר ל‑T.
- התאמת אינדיקטורים: אין שימוש ב‑rolling שמביט קדימה; רק past window.
- התאמות תאגידיות: Close ו‑Adj Close עקביים; קפיצות משקפות splits/dividends.
- Alignment: מיזוג left‑join לפי timestamps; גודל חלון קבוע.
- יציבות: התפלגות תשואות, volatile periods, sanity checks (winrate מול BH). 

## צנרת (Pipeline) בשלבים

1) Universe — בניית רשימת סמלים זמינים נקיים מ‑`stock_data/`:
  - סקר תיקיות, כתיבת `data/rl/universe.csv`.

2) Pricing Normalize — העתקה/ניקוי של `_price.csv` ל‑`data/rl/pricing/` עם שדות אחידים.

3) Indicators — מיזוג `_indicators.csv` אם קיים. חישוב מקומי בהמשך לפי צורך.

4) Market Calendar — יצירת לוח מסחר מאוחד מכל קבצי ה‑pricing:
  - איחוד כל ה‑dates מכל הסמלים, מיון, ספירת `active_symbols` לכל תאריך.
  - פלט: `data/rl/calendars/market_calendar.csv`.

5) Progressive as‑of (Production) — יצירת `progressive_signals` לכל סמל ללא דליפת עתיד:
  - אופקים נתמכים: 1d/7d/30d.
  - עמודות: `expected_return`, `confidence`, `signal` (BUY/SELL/HOLD), `sl`, `tp`, `capped` לכל אופק.
  - חישוב `SL/TP` באמצעות ATR(14) as‑of; החלת caps על `expected_return` וסימון `capped`.
  - גרסאות: `model_version` (hash קבצי checkpoints) ו‑`data_version` (hash קובץ pricing מנורמל).
  - הפקה תתבצע רק אם קיימים מודלים אמיתיים; אין יצירת placeholder.

6) Validation — הפעלת בדיקות QA ודו"ח כולל cross‑checks מול ה‑Calendar:
  - תאריכי signals מוכלים בתוך תאריכי pricing של אותו סמל, וכאשר קיים calendar — גם בתוך לוח המסחר.
  - בדיקת תחום אופקים, סכמות, סדר מונוטוני, ו‑caps sanity.

## חוזי סקריפטים (Contracts)

- build_universe.py
  - קלט: תיקיית `stock_data/`.
  - פלט: `data/rl/universe.csv` עם עמודה `symbol` וגבולות `start_date`/`end_date` נגזרים מה‑pricing.
  - שגיאות: אם אין pricing תקין — הסמל יידחה; הקובץ עשוי להיות ריק.

- build_pricing_dataset.py
  - קלט: `<SYMBOL>/_price.csv`.
  - פלט: `data/rl/pricing/<SYMBOL>.csv` עם עמודות סטנדרטיות ומיון לפי תאריך.
  - שגיאות: דיווח עמודות חסרות/שגויות; דילוג על סמלים לא תקינים.

- build_market_calendar.py
  - קלט: `data/rl/pricing/` לכל הסמלים התקינים.
  - פלט: `data/rl/calendars/market_calendar.csv` עם `date` ו‑`active_symbols`.
  - שגיאות: אם אין pricing — יווצר קובץ ריק או לא יווצר; מדווח על חריגות IO.

- build_asof_progressive_predictions.py
  - קלט: pricing מנורמל + מודלי Progressive זמינים (1d/7d/30d).
  - פלט: `data/rl/progressive_signals/<SYMBOL>.csv` as‑of, כולל `expected_return`, `confidence`, `signal`, `sl`, `tp`, `capped` לכל אופק, ו‑`model_version`/`data_version` כ‑metadata.
  - התנהגות: ריצה פרודקשן בלבד — אין יצירת placeholder; כתיבה מחליפה פלטים קודמים (overwrite) לשחזוריות.

- validate_dataset.py
  - קלט: `data/rl/pricing/`, `data/rl/progressive_signals/` (אם קיים), ו‑`data/rl/calendars/market_calendar.csv` (אם קיים).
  - פלט: דו"ח PASS/FAIL למסך, כולל בדיקות סכימה, PIT, ותתי‑קבוצות תאריכים מול pricing וה‑calendar.

## הוראות ריצה (Windows PowerShell)

```
# 1) בנה יקום
python rl/data_pipeline/build_universe.py

# 2) נרמל מחירים
python rl/data_pipeline/build_pricing_dataset.py

# 3) בנה לוח מסחר מאוחד
python rl/data_pipeline/build_market_calendar.py

# 4) הפק Progressive as-of (פרודקשן; ירוץ רק אם קיימים מודלים)
python rl/data_pipeline/build_asof_progressive_predictions.py

# 5) ולידציה (כולל cross-check מול calendar אם קיים)
python rl/data_pipeline/validate_dataset.py
```

הערות:
- שלב as‑of ירוץ רק אם קיימים מודלים אמיתיים; אין יצירת קבצי placeholder.
- מומלץ להגדיר את נתיב ה‑Calendar ב‑קובץ הקונפיג `rl/config/rl_config.yaml` תחת `paths.market_calendar`.

## שלבי הרחבה

- מעבר ל‑Parquet + partitioning לפי symbol/date (דורש `pyarrow`).
- חישוב אינדיקטורים מקומי (ta) ומקרו (VIX/CPI) עם alignment PIT.
- Universe עם start/end לפי תאריכים בפועל (קריאת המינימום/מקסימום לכל symbol).
- cache לריצות Progressive לפי גרסת מודל; benchmark זמן/מקום.

---

מסמך זה משלים את `rl/README_RL.md` ומגדיר את התכולה הנדרשת כדי להכין נתונים לאימון סוכן RL ללא הטיות ודליפות.

## גרסאות ושחזוריות (Reproducibility)

- model_version — hash דטרמיניסטי של קבצי ה‑checkpoints שבהם נעשה שימוש לכל אופק (לדוגמה 16 התווים הראשונים).
- data_version — hash דטרמיניסטי של קובץ ה‑pricing המנורמל לכל סמל.
- שדות גרסה נכתבים כ‑metadata יחד עם הפלטים מאפשרים audit ושחזור מלא של ריצות.
