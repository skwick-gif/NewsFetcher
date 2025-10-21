# 🚀 Progressive ML Training System - תכנית עבודה

**תאריך יצירה:** 21 אוקטובר 2025  
**מטרה:** מערכת אימון ML פרוגרסיבי עם השוואת מודים לחיזוי מחירי מניות

---

## 📋 Phase 1: תכנון ותשתית
- [ ] ✅ יצירת תכנית עבודה מפורטת
- [ ] ⏳ העלאה לגיט של הקוד הנוכחי
- [ ] ⏳ בדיקת נתוני AAPL הקיימים (2020-01-01 עד היום)
- [ ] ⏳ יצירת מבנה קבצים לפרויקט החדש

## 📊 Phase 2: Data Pipeline
- [ ] ⏳ פיתוח DataLoader לקריאת CSV + JSON
- [ ] ⏳ יצירת Technical Indicators (RSI, MACD, Bollinger Bands)
- [ ] ⏳ שילוב Fundamental Data מתוך JSON
- [ ] ⏳ בדיקת איכות נתונים ו-validation

## 🧠 Phase 3: Model Architecture
### בחירת מודלים:
- [ ] ⏳ LSTM בלבד (מהיר)
- [ ] ⏳ LSTM + CNN (איזון)
- [ ] ⏳ LSTM + Transformer + CNN (מלא)

### Time Horizons:
- [ ] ⏳ 1 יום קדימה
- [ ] ⏳ 7 ימים קדימה  
- [ ] ⏳ 30 ימים קדימה

## 🔄 Phase 4: Progressive Training System
### Progressive Mode:
- [ ] ⏳ אימון נפרד לכל horizon
- [ ] ⏳ Iterative improvement (30 iterations max)
- [ ] ⏳ Fine-tuning עם rolling window

### Unified Mode:
- [ ] ⏳ Multi-output model לכל horizons
- [ ] ⏳ Shared feature learning
- [ ] ⏳ Single training process

## 📊 Phase 5: Progress Tracking
- [ ] ⏳ Real-time Progress Bar
- [ ] ⏳ ETA calculation
- [ ] ⏳ Accuracy tracking per iteration
- [ ] ⏳ Loss visualization

## 🎯 Phase 6: Training Parameters
### תאריכים:
- **Start Date:** 2025-01-01
- **Initial End Date:** 2025-09-01  
- **Progressive Window:** +7 ימים כל iteration
- **Max Iterations:** 30

### Model Parameters:
- **Batch Size:** 64 (או קטן יותר אם יש בעיות זיכרון)
- **Epochs per iteration:** 10-20
- **Early Stopping:** אם אין שיפור ב-5 iterations
- **Target Accuracy:** 70%

## 🔬 Phase 7: Results Comparison
- [ ] ⏳ השוואת Progressive vs Unified
- [ ] ⏳ Accuracy per horizon analysis
- [ ] ⏳ Training time comparison
- [ ] ⏳ Model size comparison
- [ ] ⏳ Export results to CSV/JSON

## 💻 Phase 8: Dashboard Integration
- [ ] ⏳ UI לבחירת מוד אימון
- [ ] ⏳ פרמטרים configurable
- [ ] ⏳ Progress display בזמן אמת
- [ ] ⏳ Results visualization
- [ ] ⏳ Model comparison charts

## 🏗 Phase 9: Implementation Details

### קבצים חדשים:
```
app/ml/progressive/
├── __init__.py
├── progressive_trainer.py      # Main training system
├── data_loader.py             # AAPL data loading
├── model_factory.py           # Model creation
├── progress_tracker.py        # Progress bar & stats
├── results_comparator.py      # Compare modes
└── config.py                  # Training parameters
```

### API Endpoints חדשים:
- `POST /api/ml/progressive/start` - התחלת אימון
- `GET /api/ml/progressive/status` - סטטוס אימון
- `GET /api/ml/progressive/results` - תוצאות
- `POST /api/ml/progressive/compare` - השוואת מודים

## ⚙️ Phase 10: Configuration Options

### Training Config:
```yaml
training:
  symbol: "AAPL"
  start_date: "2025-01-01"
  initial_end_date: "2025-09-01"
  window_increment: 7  # days
  max_iterations: 30
  target_accuracy: 0.70
  early_stop_patience: 5

models:
  enabled: ["lstm", "cnn", "transformer"]  # או ["lstm"] לבדיקה מהירה
  batch_size: 64
  epochs_per_iteration: 15

horizons:
  enabled: [1, 7, 30]  # days
  
modes:
  progressive: true
  unified: true
  compare_results: true
```

## 📈 Phase 11: Success Metrics
- [ ] ⏳ Direction Accuracy > 70% עבור 1 יום
- [ ] ⏳ Direction Accuracy > 65% עבור 7 ימים
- [ ] ⏳ Direction Accuracy > 60% עבור 30 ימים
- [ ] ⏳ Training time < 2 שעות (עם GPU)
- [ ] ⏳ Model size < 500MB total

## 🧪 Phase 12: Testing & Validation
- [ ] ⏳ Unit tests לכל רכיב
- [ ] ⏳ Integration tests לתהליך המלא
- [ ] ⏳ Performance benchmarks
- [ ] ⏳ Memory usage validation
- [ ] ⏳ Error handling testing

---

## 🎯 סדר ביצוע מומלץ:

1. **Setup** (Phase 1-2): תשתית ונתונים
2. **Models** (Phase 3): יצירת מודלים בסיסיים
3. **Progressive** (Phase 4): מערכת אימון פרוגרסיבי
4. **UI** (Phase 5,8): Progress tracking ודשבורד
5. **Compare** (Phase 7): השוואת תוצאות
6. **Polish** (Phase 11-12): בדיקות וביצועים

---

## 📝 הערות:
- נתחיל עם LSTM בלבד לבדיקה מהירה
- נוסיף מודלים נוספים אחרי שנוודא שהבסיס עובד
- Progress bar חייב להיות responsive ומדויק
- כל iteration שומרת checkpoint למקרה של crash
- Results נשמרים ב-JSON לanalyze מאוחר יותר


**סטטוס נוכחי:** 📋 תכנון הושלם - מוכן להתחיל Phase 1!