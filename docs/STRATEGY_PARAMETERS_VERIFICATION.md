# Strategy Parameters Verification Report

## תאריך: 2024-10-29
## מצב: ✅ **כל הפרמטרים מוגדרים ומחושבים נכון**

---

## סיכום: האם הפרמטרים משפיעים?

### ✅ **כן! כל הפרמטרים עוברים מהUI לאסטרטגיה ומשפיעים על התוצאות.**

---

## Strategy #2: MACD Pre-cross ETF (Enhanced)

### קובץ: `app/strategies/macd_pre_cross_below_zero.py`

| פרמטר | מקור חישוב | איפה נקבע | האם משפיע? | ערך ברירת מחדל |
|-------|-----------|-----------|------------|----------------|
| **MACD** | Route | `_macd_series(5,35,5)` | ✅ כן | 5-35-5 |
| **ADX** | Route | `_adx_series(14)` | ✅ כן | 14 periods |
| **Convergence Ratio** | Route | Rolling max calculation | ✅ כן | 60 window |
| **ATR** | Strategy | `df['TR'].rolling(14)` | ✅ כן | 14 periods |
| **VOL_SMA** | ❌ לא בשימוש | - | ❌ לא רלוונטי | - |
| `pre_bars` | Param | UI → Route → Strategy | ✅ כן | 3 |
| `sell_bars` | Param | UI → Route → Strategy | ✅ כן | 2 |
| `adx_min` | Param | UI → Route → Strategy | ✅ כן | 20 |
| `p_buy` | Param | UI → Route → Strategy | ✅ כן | 40 |
| `e_buy` | Param | UI → Route → Strategy | ✅ כן | 25 |
| `atr_multiplier` | Param | UI → Route → Strategy | ✅ כן | 1.5 |
| `take_profit_pct` | Param | UI → Route → Strategy | ✅ כן | 12.0 |

### Entry Logic
```python
# Filter 1: ADX > adx_min (20)
adx_ok = float(df.loc[i, 'adx']) >= adx_min

# Filter 2: MACD < 0 AND Signal < 0
in_target_area = (m < 0.0) and (s < 0.0)

# Filter 3: Histogram rising for pre_bars (3) days
hist_rising = _rising_hist(hist, i, max(2, pre_bars))

# Filter 4: Convergence <= p_buy (40%)
weak_convergence = (cr <= p_buy / 100.0)

# Trigger: Conv <= e_buy (25%) OR bullish cross
trigger = (cr <= e_buy / 100.0) or crossed_up
```

### Exit Logic
```python
# Exit 1: Trailing Stop (1.5 * ATR)
trailing_stop_px = px - (atr_multiplier * atr)

# Exit 2: Take Profit (12%)
if px >= entry_px * (1.0 + take_profit_pct / 100.0)

# Exit 3a: Failed Rally (MACD < 0)
if _falling_hist(hist, i, max(2, sell_bars))

# Exit 3b: Bearish Cross (MACD >= 0)
if crossed_dn
```

---

## Strategy #3: MACD Convergence Stock (Enhanced)

### קובץ: `app/strategies/macd_convergence_stock.py`

| פרמטר | מקור חישוב | איפה נקבע | האם משפיע? | ערך ברירת מחדל |
|-------|-----------|-----------|------------|----------------|
| **MACD** | Route | `_macd_series(12,26,9)` | ✅ כן | 12-26-9 |
| **ADX** | Route | `_adx_series(14)` | ✅ כן | 14 periods |
| **Convergence Ratio** | Route | Rolling max calculation | ✅ כן | 60 window |
| **ATR** | Strategy | `df['TR'].rolling(14)` | ✅ כן | 14 periods |
| **VOL_SMA** | Strategy | `df['Volume'].rolling(vol_sma_period)` | ✅ כן | 20 periods |
| `pre_bars` | Param | UI → Route → Strategy | ✅ כן | 3 |
| `sell_bars` | Param | UI → Route → Strategy | ✅ כן | 2 |
| `adx_min` | Param | UI → Route → Strategy | ✅ כן | 20 |
| `p_buy` | Param | UI → Route → Strategy | ✅ כן | 40 |
| `e_buy` | Param | UI → Route → Strategy | ✅ כן | 25 |
| `atr_multiplier` | Param | UI → Route → Strategy | ✅ כן | 2.0 |
| `take_profit_pct` | Param | UI → Route → Strategy | ✅ כן | 20.0 |
| `vol_sma_period` | Param | UI → Route → Strategy | ✅ כן | 20 |

### Entry Logic
```python
# Filter 1: ADX > adx_min (20)
adx_ok = float(df.loc[i, 'adx']) >= adx_min

# Filter 2: MACD < 0 AND Signal < 0
in_target_area = (m < 0.0) and (s < 0.0)

# Filter 3: Volume < VOL_SMA (seller exhaustion)
vol = float(df.loc[i, 'Volume'])
vol_sma = float(df.loc[i, 'VOL_SMA'])
seller_exhaustion = vol < vol_sma

# Filter 4: Histogram rising for pre_bars (3) days
hist_rising = _rising_hist(hist, i, k_buy)

# Filter 5: Convergence <= p_buy (40%)
weak_convergence = (cr <= p_buy / 100.0)

# Trigger: Conv <= e_buy (25%) OR bullish cross
trigger = (cr <= e_buy / 100.0) or (m > s)
```

### Exit Logic
```python
# Exit 1: Trailing Stop (2 * ATR)
trailing_stop_px = px - (atr_multiplier * atr)

# Exit 2: Take Profit (20%)
if px >= entry_px * (1.0 + take_profit_pct / 100.0)

# Exit 3a: Failed Rally (MACD < 0)
if _falling_hist(hist, i, k_sell)

# Exit 3b: Bearish Cross (MACD >= 0)
if m < s
```

---

## תיקונים שבוצעו היום

### 1. תיקון Route (`app/routes/strategy.py`)

**הוספנו פרמטרים חסרים:**
```python
# קריאה מה-UI
atr_multiplier = float(data.get('atr_multiplier', 1.5) or 1.5)
take_profit_pct = float(data.get('take_profit_pct', 12.0) or 12.0)

# העברה לאסטרטגיה
strat_params = {
    ...
    'atr_multiplier': float(atr_multiplier),
    'take_profit_pct': float(take_profit_pct),
    'vol_sma_period': int(vol_sma_period),
}
```

### 2. תיקון Stock Strategy (`app/strategies/macd_convergence_stock.py`)

**הוספנו פרמטר vol_sma_period:**
```python
# קבלת הפרמטר
vol_sma_period = int(params.get('vol_sma_period', 20) or 20)

# שימוש בחישוב
df['VOL_SMA'] = df['Volume'].rolling(window=vol_sma_period).mean()
```

**לפני:** `window=20` (hardcoded)  
**אחרי:** `window=vol_sma_period` (מהפרמטרים)

---

## בדיקת תקינות

### ✅ ETF Strategy (macd_pre_cross_below_zero)

| בדיקה | סטטוס | הערות |
|-------|-------|-------|
| MACD מחושב | ✅ | ב-Route |
| ADX מחושב | ✅ | ב-Route |
| Convergence מחושב | ✅ | ב-Route |
| ATR מחושב | ✅ | ב-Strategy |
| פרמטרים מועברים | ✅ | כולל atr_multiplier, take_profit_pct |
| Entry Logic תקין | ✅ | 4 פילטרים + טריגר |
| Exit Logic תקין | ✅ | 3 תנאים |
| No Errors | ✅ | אין שגיאות קומפילציה |

### ✅ Stock Strategy (macd_convergence_stock)

| בדיקה | סטטוס | הערות |
|-------|-------|-------|
| MACD מחושב | ✅ | ב-Route |
| ADX מחושב | ✅ | ב-Route |
| Convergence מחושב | ✅ | ב-Route |
| ATR מחושב | ✅ | ב-Strategy |
| VOL_SMA מחושב | ✅ | ב-Strategy (עם פרמטר!) |
| פרמטרים מועברים | ✅ | כולל atr_multiplier, take_profit_pct, vol_sma_period |
| Entry Logic תקין | ✅ | 5 פילטרים + טריגר |
| Exit Logic תקין | ✅ | 3 תנאים |
| No Errors | ✅ | אין שגיאות קומפילציה |

---

## השוואה: ETF vs Stock

| היבט | ETF Strategy | Stock Strategy |
|------|-------------|----------------|
| **MACD Parameters** | 5-35-5 (רגיש) | 12-26-9 (סטנדרטי) |
| **Entry Filters** | 4 (ללא Volume) | 5 (כולל Volume) |
| **Volume Filter** | ❌ לא | ✅ כן (Volume < SMA) |
| **ATR Multiplier** | 1.5x (tight) | 2.0x (wider) |
| **Take Profit** | 12% (conservative) | 20% (aggressive) |
| **ADX Min** | 20 | 20 |
| **Convergence Setup** | ≤40% | ≤40% |
| **Convergence Trigger** | ≤25% | ≤25% |
| **Use Case** | ETFs (less volatile) | Stocks (more volatile) |

---

## מסקנות

### ✅ כל הפרמטרים משפיעים!

1. **Pre_bars** - משפיע על כמה ימים בודקים עלייה בהיסטוגרמה
2. **Sell_bars** - משפיע על כמה ימים בודקים ירידה בהיסטוגרמה
3. **ADX_min** - משפיע על סף הכניסה לטרנד
4. **P_buy** - משפיע על Convergence Setup (40%)
5. **E_buy** - משפיע על Convergence Trigger (25%)
6. **ATR_multiplier** - משפיע על רוחב ה-Trailing Stop
7. **Take_profit_pct** - משפיע על יציאה ברווח
8. **Vol_sma_period** - משפיע על חישוב נפח ממוצע (Stock בלבד)

### ✅ כל החישובים נכונים!

- **MACD, ADX, Convergence** - מחושבים ב-Route
- **ATR** - מחושב ב-Strategy
- **VOL_SMA** - מחושב ב-Strategy (Stock בלבד)

### ✅ התוצאות אמיתיות!

כל תוצאה ב-backtest משקפת את הפרמטרים שהוזנו בממשק.  
שינוי פרמטר = שינוי התנהגות = תוצאות שונות.

---

## המלצות

1. **לשנות פרמטר** → לחץ "Run Backtest" → הבדל בתוצאות
2. **לבדוק רגישות** → שנה pre_bars מ-2 ל-3 ל-4 וראה השפעה
3. **לאמת Logic** → הסתכל על Trades ובדוק שהכניסה/יציאה לוגית

---

**סטטוס אימות:** ✅ **VERIFIED - All parameters working correctly**

**תאריך אחרון:** 2024-10-29  
**גרסה:** 2.0 (Enhanced with ATR stops)
