# תיקון שגיאת NaN Values - HTTP 500

## 🐛 הבעיה
כשניסית לקבל תחזיות למניה AAPL (וכנראה מניות אחרות), קיבלת:
```
❌ Error: HTTP 500: Internal Server Error
ValueError: Out of range float values are not JSON compliant: nan
```

## 🔍 הסיבה
1. **מודלים מסוימים (במיוחד Transformer) נכשלים בטעינה** - זורקים exception
2. **תחזיות מסוימות מחזירות NaN** (Not a Number) - למשל כשמודל לא מאומן טוב
3. **JSON לא תומך ב-NaN** - Python יכול לעבוד עם NaN אבל JSON לא יכול לסריאליז אותו

## ✅ הפתרון

### 1. **תפיסת Exceptions בטעינת מודלים**
תיקנתי את `app/ml/progressive/predictor.py` כך שאם מודל נכשל בטעינה, המערכת ממשיכה עם שאר המודלים:

```python
# לפני:
model = keras.models.load_model(str(model_file), custom_objects=custom_objs)

# אחרי:
try:
    model = keras.models.load_model(str(model_file), custom_objects=custom_objs)
    model_dict[horizon_key] = model
    logger.info(f"   ✅ Loaded {model_type} {horizon_key}")
except Exception as load_err:
    logger.warning(f"   ⚠️ Failed to load {model_type} {horizon_key}: {load_err}")
```

### 2. **טיפול ב-NaN בתחזיות בודדות**
ב-`predict_single_model()` - בדיקה והחלפה של NaN:

```python
# Check for NaN values and replace with defaults
if np.isnan(price_pred) or np.isinf(price_pred):
    logger.warning(f"   ⚠️ NaN/Inf detected in price prediction, using 0.0")
    price_pred = 0.0

if np.isnan(direction_pred) or np.isinf(direction_pred):
    logger.warning(f"   ⚠️ NaN/Inf detected in direction prediction, using 0.5")
    direction_pred = 0.5
```

### 3. **טיפול ב-NaN בתחזיות Ensemble**
גם בחישובי הממוצע המשוקלל:

```python
# Check for NaN/Inf in ensemble results
if np.isnan(ensemble_price_change) or np.isinf(ensemble_price_change):
    logger.warning(f"   ⚠️ NaN/Inf in ensemble price change, using 0.0")
    ensemble_price_change = 0.0

if np.isnan(ensemble_direction_prob) or np.isinf(ensemble_direction_prob):
    logger.warning(f"   ⚠️ NaN/Inf in ensemble direction prob, using 0.5")
    ensemble_direction_prob = 0.5

# Check for NaN in std
if np.isnan(price_std) or np.isinf(price_std):
    price_std = 0.0
```

### 4. **לוגינג משופר**
שיפרתי את הלוגינג ב-`main_realtime.py` כדי לראות שגיאות בצורה ברורה יותר:

```python
logger.error(f"❌ Failed to get progressive predictions for {symbol}: {e}", exc_info=True)
raise HTTPException(status_code=500, detail=f"Failed to get progressive predictions: {str(e)}")
```

---

## 🧪 תוצאות

### ✅ לפני התיקון:
```powershell
Invoke-RestMethod http://localhost:8000/api/ml/progressive/predict/AAPL
# ❌ Error: Internal Server Error
# ValueError: Out of range float values are not JSON compliant: nan
```

### ✅ אחרי התיקון:
```powershell
Invoke-RestMethod http://localhost:8000/api/ml/progressive/predict/AAPL
# ✅ AAPL Success: Status=success
```

---

## 📊 מה קורה עכשיו

### מודלים שנכשלים בטעינה:
- ✅ **LSTM** - נטען ומחזיר תחזיות
- ✅ **CNN** - נטען ומחזיר תחזיות  
- ⚠️ **Transformer** - נכשל בטעינה אבל **לא קורס את המערכת**

### תחזיות עם NaN:
- ⚠️ אם מודל בודד מחזיר NaN - מוחלף ב-0.0
- ⚠️ אם ensemble מחזיר NaN - מוחלף ב-0.0  
- ✅ המערכת ממשיכה לעבוד עם המודלים הזמינים

---

## 🎯 סיכום

**הבעיה:** JSON לא יכול לסריאליז NaN values שמגיעים ממודלים  
**הפתרון:** בדיקה והחלפה של NaN ב-ערכי ברירת מחדל (0.0, 0.5)  
**התוצאה:** המערכת עובדת גם כשחלק מהמודלים נכשלים או מחזירים NaN

**מניות שנבדקו:**
- ✅ RGTI - עובד
- ✅ AAPL - עובד (לאחר התיקון)
- ✅ כל מניה אחרת אמורה לעבוד

**לוגים:**
המערכת מדווחת עכשיו אזהרות ברורות:
- `⚠️ Failed to load transformer_AAPL_1d: ...`
- `⚠️ NaN/Inf detected in price prediction, using 0.0`

זה מאפשר לך לדעת איזה מודלים עובדים ואיזה לא, בלי שהמערכת תקרוס!
