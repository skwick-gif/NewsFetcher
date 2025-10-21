# קבצי הפעלה לאימון מודלים - MarketPulse

## קבצי BAT זמינים:

### 🚀 start_training.bat - הפעלת אימון עם אפשרויות
**השימוש הראשי - עם אפשרויות התאמה אישית**
- בוחר batch size
- בוחר מספר epochs מקסימלי  
- בוחר האם להמשיך מ-checkpoint או להתחיל מחדש
- שומר לוג מפורט עם timestamp
- מראה סטטוס לפני ואחרי האימון

### ⚡ quick_train.bat - הפעלה מהירה
**להפעלה מהירה עם הגדרות ברירת מחדל**
- Batch size: 32
- Epochs: 100  
- תמיד ממשיך מ-checkpoint אם קיים
- שומר לוג בסיסי

### 📊 check_status.bat - בדיקת סטטוס בלבד
**לבדיקת מצב האימון הנוכחי מבלי להתחיל אימון**
- מראה כמה סמלים מוכנים
- מראה אם יש checkpoints קיימים
- מראה לוגים אחרונים

## איך להשתמש:

1. **לאימון רגיל:** לחץ פעמיים על `start_training.bat`
2. **לאימון מהיר:** לחץ פעמיים על `quick_train.bat` 
3. **לבדיקת מצב:** לחץ פעמיים על `check_status.bat`

## עצירת אימון בטוחה:
- לחץ `Ctrl+C` בחלון הטרמינל
- ה-checkpoint האחרון יישמר אוטומטית
- באימון הבא זה ימשיך מאותה נקודה

## מיקום קבצים:
- **Checkpoints:** `app/ml/models/checkpoints/`
- **מודלים סופיים:** `app/ml/models/`
- **לוגים:** `logs/`

## פרמטרי command line זמינים:
```bash
py app/ml/train_model.py --batch-size 64 --epochs 50 --no-resume
```

- `--batch-size X` - גודל batch (ברירת מחדל: 32)
- `--epochs X` - מספר epochs מקסימלי (ברירת מחדל: 100)  
- `--no-resume` - התחל מחדש במקום להמשיך מ-checkpoint

## דרישות:
- Python עם TensorFlow מותקן
- נתונים מוכנים ב-`ml/data/`
- די מקום פנוי לcheckpoints ולוגים