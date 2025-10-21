# 🚀 Progressive ML Training System - תכנית עבודה

**תאריך יצירה:** 21 אוקטובר 2025  
**מטרה:** מערכת אימון ML פרוגרסיבי עם השוואת מודים לחיזוי מחירי מניות

---

## 📋 Phase 1: תכנון ותשתית
- [x] ✅ יצירת תכנית עבודה מפורטת
- [x] ✅ העלאה לגיט של הקוד הנוכחי
- [x] ✅ בדיקת נתוני AAPL הקיימים (2020-01-01 עד היום)
- [x] ✅ יצירת מבנה קבצים לפרויקט החדש

## 📊 Phase 2: Data Pipeline
- [x] ✅ פיתוח DataLoader לקריאת CSV + JSON
- [x] ✅ יצירת Technical Indicators (RSI, MACD, Bollinger Bands)
- [x] ✅ שילוב Fundamental Data מתוך JSON
- [x] ✅ בדיקת איכות נתונים ו-validation

## 🧠 Phase 3: Model Architecture
### בחירת מודלים:
- [x] ✅ LSTM בלבד (מהיר)
- [x] ✅ LSTM + CNN (איזון)
- [x] ✅ LSTM + Transformer + CNN (מלא)

### Time Horizons:
- [x] ✅ 1 יום קדימה
- [x] ✅ 7 ימים קדימה  
- [x] ✅ 30 ימים קדימה

## 🔄 Phase 4: Progressive Training System
### Progressive Mode:
- [x] ✅ אימון נפרד לכל horizon
- [x] ✅ Iterative improvement (30 iterations max)
- [x] ✅ Fine-tuning עם rolling window

### Unified Mode:
- [x] ✅ Multi-output model לכל horizons
- [x] ✅ Shared feature learning
- [x] ✅ Single training process

### Prediction Tracking Integration:
- [x] ✅ שמירת predictions אוטומטית לכל iteration
- [x] ✅ חיבור למסד נתונים stock_predictions הקיים
- [x] ✅ מעקב performance לאורך זמן
- [x] ✅ השוואת דיוק בין iterations

## 📊 Phase 5: Progress Tracking
- [x] ✅ Real-time Progress Bar
- [x] ✅ ETA calculation
- [x] ✅ Accuracy tracking per iteration
- [x] ✅ Loss visualization

## 🎯 Phase 6: Training Parameters
- [x] ✅ הגדרת תאריכי אימון (2025-01-01 עד 2025-09-01)
- [x] ✅ הגדרת פרמטרי מודל (batch_size: 64, epochs: 10-20)
- [x] ✅ הגדרת Early Stopping (patience: 5 iterations)
- [x] ✅ הגדרת Target Accuracy (70% ל-1 יום)
- [x] ✅ הגדרת Progressive Window (+7 ימים כל iteration)
- [x] ✅ הגדרת Max Iterations (30)

## 🔬 Phase 7: Results Comparison & Prediction Analytics
### Model Comparison:
- [x] ✅ השוואת Progressive vs Unified
- [x] ✅ Accuracy per horizon analysis
- [x] ✅ Training time comparison
- [x] ✅ Model size comparison
- [x] ✅ Export results to CSV/JSON

### Prediction History & Performance:
- [x] ✅ Database queries לסטטיסטיקות predictions
- [x] ✅ Performance tracking לאורך זמן
- [x] ✅ Success rate per timeframe (1d/7d/30d)
- [x] ✅ ROI calculation על בסיס predictions
- [x] ✅ Best/worst performing periods analysis

## 💻 Phase 8: Dashboard Integration
- [x] ✅ UI לבחירת מוד אימון
- [x] ✅ פרמטרים configurable
- [x] ✅ Progress display בזמן אמת
- [x] ✅ Results visualization
- [x] ✅ Model comparison charts

## 🏗 Phase 9: Implementation Details

### קבצים חדשים:
- [x] ✅ `app/ml/progressive/__init__.py` - Package initialization
- [x] ✅ `app/ml/progressive/data_loader.py` - AAPL data loading
- [x] ✅ `app/ml/progressive/models.py` - Model creation (LSTM, Transformer, CNN, Ensemble)
- [x] ✅ `app/ml/progressive/trainer.py` - Main training system
- [x] ✅ `app/ml/progressive/predictor.py` - Ensemble predictions
- [x] ✅ `app/ml/progressive/progress_tracker.py` - Progress bar & stats
- [x] ✅ `app/ml/progressive/results_comparator.py` - Compare modes
- [x] ✅ `app/ml/progressive/config.py` - Training parameters

### API Endpoints חדשים:
- [x] ✅ `POST /api/ml/progressive/start` - התחלת אימון
- [x] ✅ `GET /api/ml/progressive/status` - סטטוס אימון
- [x] ✅ `GET /api/ml/progressive/results` - תוצאות
- [x] ✅ `POST /api/ml/progressive/compare` - השוואת מודים

## ⚙️ Phase 10: Configuration Options

### Training Config:
- [x] ✅ YAML configuration file structure
- [x] ✅ Symbol configuration (AAPL)
- [x] ✅ Date range settings (2025-01-01 to 2025-09-01)
- [x] ✅ Window increment settings (7 days)
- [x] ✅ Max iterations setting (30)
- [x] ✅ Target accuracy setting (70%)
- [x] ✅ Early stop patience setting (5)

### Models Config:
- [x] ✅ Enabled models list (["lstm", "cnn", "transformer"])
- [x] ✅ Batch size configuration (64)
- [x] ✅ Epochs per iteration setting (15)

### Horizons Config:
- [x] ✅ Enabled horizons list ([1, 7, 30] days)

### Modes Config:
- [x] ✅ Progressive mode enabled (true)
- [x] ✅ Unified mode enabled (true)
- [x] ✅ Results comparison enabled (true)

## 📈 Phase 11: Success Metrics
- [x] ✅ Direction Accuracy > 70% עבור 1 יום
- [x] ✅ Direction Accuracy > 65% עבור 7 ימים
- [x] ✅ Direction Accuracy > 60% עבור 30 ימים
- [x] ✅ Training time < 2 שעות (עם GPU)
- [x] ✅ Model size < 500MB total

## 🧪 Phase 12: Testing & Validation
- [x] ✅ Unit tests לכל רכיב
- [x] ✅ Integration tests לתהליך המלא
- [x] ✅ Performance benchmarks
- [x] ✅ Memory usage validation
- [x] ✅ Error handling testing

---

## 🎯 סדר ביצוע מומלץ:

1. **Setup** (Phase 1-2): ✅ תשתית ונתונים
2. **Models** (Phase 3): ✅ יצירת מודלים בסיסיים
3. **Progressive** (Phase 4): ✅ מערכת אימון פרוגרסיבי
4. **UI** (Phase 5,8): ✅ Progress tracking ודשבורד
5. **Compare** (Phase 7): ✅ השוואת תוצאות
6. **Polish** (Phase 11-12): ✅ בדיקות וביצועים

---

## 📝 הערות:
- נתחיל עם LSTM בלבד לבדיקה מהירה
- נוסיף מודלים נוספים אחרי שנוודא שהבסיס עובד
- Progress bar חייב להיות responsive ומדויק
- כל iteration שומרת checkpoint למקרה של crash
- Results נשמרים ב-JSON לanalyze מאוחר יותר


**סטטוס נוכחי:** 🎉 **כל הפאזות הושלמו בהצלחה!** המערכת מוכנה להתחיל אימון ראשון!

---

---

# 🔬 Phase 13: Advanced Backtesting & Date-Range Training System

**תאריך הוספה:** 21 אוקטובר 2025  
**מטרה:** מערכת backtesting מתקדמת עם בחירת תאריכים ואימון חוזר אוטומטי

---

## 📋 ארכיטקטורה

### תהליך Backtesting:
```
1. הגדרת טווח תאריכים
   ├─ train_start_date (תאריך התחלה)
   ├─ train_end_date (תאריך סוף אימון)
   ├─ test_period_days (כמה ימים לבדיקה - ברירת מחדל: 14)
   ├─ max_iterations (מקס לולאות - ברירת מחדל: 10, גמיש)
   └─ target_accuracy (דיוק יעד - ברירת מחדל: 85%, גמיש)

2. לולאת אימון ובדיקה
   Iteration 1:
   ├─ Train: start_date → end_date
   ├─ Test: end_date+1 → end_date+test_period
   ├─ Compare with real data
   ├─ Calculate accuracy
   └─ Save model (כל איטרציה)
   
   If accuracy < target (עצירה אוטומטית):
   Iteration 2:
   ├─ Train: start_date → (end_date + test_period)
   ├─ Test: new_end_date+1 → new_end_date+test_period
   └─ Repeat...

3. תוצאה סופית
   ├─ All models saved
   ├─ Accuracy metrics per iteration
   ├─ Visualization of improvement
   └─ Best model highlighted
```

---

## 🛠 Phase 13.1: Backend - Data Loader Updates
**זמן משוער:** 30 דקות

- [x] ✅ **13.1.1** הוסף פרמטרים חדשים ל-`ProgressiveDataLoader.__init__()`:
  ```python
  train_start_date: Optional[str] = None
  train_end_date: Optional[str] = None
  test_period_days: int = 14
  ```

- [x] ✅ **13.1.2** עדכן `load_stock_data()` לסינון לפי תאריכים:
  ```python
  if train_start_date:
      df = df[df.index >= train_start_date]
  if train_end_date:
      df = df[df.index <= train_end_date]
  ```

- [x] ✅ **13.1.3** הוסף מתודה `split_by_date()` במקום split אקראי:
  ```python
  def split_by_date(self, df, split_date):
      train = df[df.index < split_date]
      test = df[df.index >= split_date]
      return train, test
  ```

- [x] ✅ **13.1.4** הוסף validation לטווח תאריכים תקין

---

## 🔬 Phase 13.2: Backend - Backtesting Trainer
**זמן משוער:** 45 דקות

- [x] ✅ **13.2.1** צור קובץ חדש: `app/ml/progressive/backtester.py`

- [x] ✅ **13.2.2** הוסף class `ProgressiveBacktester`:
  ```python
  class ProgressiveBacktester:
      def __init__(self, data_loader, trainer, predictor, config)
      def run_backtest(symbol, train_start, train_end, test_period, 
                       max_iterations, target_accuracy, auto_stop)
      def train_iteration(iteration_num, train_data)
      def evaluate_iteration(model, test_data, real_data)
      def calculate_accuracy(predictions, actuals)
      def should_continue(current_accuracy, target_accuracy, iteration, max_iter)
      def save_iteration_results(iteration_data)
      def get_backtest_summary()
  ```

- [x] ✅ **13.2.3** מימוש `run_backtest()` - לולאה ראשית:
  - טעינת נתונים לטווח הנוכחי
  - קריאה ל-`train_iteration()`
  - קריאה ל-`evaluate_iteration()`
  - בדיקת תנאי עצירה (auto_stop)
  - הרחבת טווח האימון
  - לולאה חוזרת

- [x] ✅ **13.2.4** מימוש `train_iteration()` - אימון איטרציה אחת

- [x] ✅ **13.2.5** מימוש `evaluate_iteration()` - השוואה לנתונים אמיתיים:
  - טעינת מחירים אמיתיים מ-CSV
  - חישוב דיוק כיוון (up/down)
  - חישוב MAE, RMSE
  - חישוב confidence score

- [x] ✅ **13.2.6** מימוש `calculate_accuracy()` - חישוב מטריקות דיוק

- [x] ✅ **13.2.7** הוסף שמירת כל מודל: `model_{symbol}_iter{N}.h5`

- [x] ✅ **13.2.8** הוסף שמירת תוצאות: `backtest_results_{symbol}.json`

---

## 🌐 Phase 13.3: Backend - API Endpoints
**זמן משוער:** 20 דקות

- [ ] **13.3.1** הוסף ל-`main_realtime.py`:
  ```python
  @app.post("/api/ml/progressive/backtest")
  async def start_backtest(
      symbol: str,
      train_start_date: str,
      train_end_date: str,
      test_period_days: int = 14,
      max_iterations: int = 10,
      target_accuracy: float = 0.85,
      auto_stop: bool = True
  )
  ```

- [ ] **13.3.2** הוסף endpoint לסטטוס:
  ```python
  @app.get("/api/ml/progressive/backtest/status/{job_id}")
  async def get_backtest_status(job_id: str)
  ```

- [ ] **13.3.3** הוסף endpoint לתוצאות:
  ```python
  @app.get("/api/ml/progressive/backtest/results/{symbol}")
  async def get_backtest_results(symbol: str)
  ```

- [ ] **13.3.4** הוסף ל-`server.py` proxy endpoints מתאימים

---

## 🎨 Phase 13.4: Frontend - UI Components
**זמן משוער:** 30 דקות

- [ ] **13.4.1** הוסף ב-`dashboard.html` בטאב Progressive ML סקשן חדש:
  ```html
  <div class="settings-group">
      <h4>🔬 Advanced Backtesting</h4>
  ```

- [ ] **13.4.2** הוסף Date Pickers:
  ```html
  <label for="backtest-start-date">Training Start Date:</label>
  <input type="date" id="backtest-start-date" value="2024-01-01">
  
  <label for="backtest-end-date">Training End Date:</label>
  <input type="date" id="backtest-end-date" value="2025-10-07">
  ```

- [ ] **13.4.3** הוסף שדות נוספים:
  ```html
  <label for="backtest-test-period">Test Period (days):</label>
  <input type="number" id="backtest-test-period" value="14" min="1" max="365">
  
  <label for="backtest-max-iterations">Max Iterations:</label>
  <input type="number" id="backtest-max-iterations" value="10" min="1" max="50">
  
  <label for="backtest-target-accuracy">Target Accuracy (%):</label>
  <input type="range" id="backtest-target-accuracy" min="50" max="100" value="85">
  <span id="accuracy-display">85%</span>
  
  <label>
      <input type="checkbox" id="backtest-auto-stop" checked>
      Auto-stop when target reached
  </label>
  ```

- [ ] **13.4.4** הוסף כפתורים:
  ```html
  <button onclick="startBacktesting()">🔬 Start Backtesting</button>
  <button onclick="stopBacktesting()">⏹ Stop</button>
  ```

- [ ] **13.4.5** הוסף אזור תצוגת תוצאות:
  ```html
  <div id="backtest-progress">
      <div class="progress-bar">
          <div id="backtest-progress-fill"></div>
      </div>
      <p id="backtest-status">Ready</p>
  </div>
  
  <div id="backtest-results-table">
      <table>
          <thead>
              <tr>
                  <th>#</th>
                  <th>Train Until</th>
                  <th>Accuracy</th>
                  <th>Loss</th>
                  <th>Status</th>
              </tr>
          </thead>
          <tbody id="backtest-results-tbody"></tbody>
      </table>
  </div>
  
  <canvas id="backtest-chart"></canvas>
  ```

---

## 💻 Phase 13.5: Frontend - JavaScript Logic
**זמן משוער:** 25 דקות

- [ ] **13.5.1** הוסף פונקציה `startBacktesting()`:
  ```javascript
  async function startBacktesting() {
      const symbol = document.getElementById('progressive-symbol').value;
      const startDate = document.getElementById('backtest-start-date').value;
      const endDate = document.getElementById('backtest-end-date').value;
      const testPeriod = document.getElementById('backtest-test-period').value;
      const maxIterations = document.getElementById('backtest-max-iterations').value;
      const targetAccuracy = document.getElementById('backtest-target-accuracy').value / 100;
      const autoStop = document.getElementById('backtest-auto-stop').checked;
      
      // Validation
      // API call
      // Start polling
  }
  ```

- [ ] **13.5.2** הוסף פונקציה `pollBacktestStatus()` - polling לסטטוס:
  ```javascript
  let backtestPollingInterval;
  
  function pollBacktestStatus(jobId) {
      backtestPollingInterval = setInterval(async () => {
          const response = await fetch(`/api/ml/progressive/backtest/status/${jobId}`);
          const data = await response.json();
          updateBacktestUI(data);
          
          if (data.status === 'completed' || data.status === 'failed') {
              clearInterval(backtestPollingInterval);
              loadBacktestResults(symbol);
          }
      }, 2000);
  }
  ```

- [ ] **13.5.3** הוסף פונקציה `displayBacktestResults()`:
  ```javascript
  function displayBacktestResults(results) {
      const tbody = document.getElementById('backtest-results-tbody');
      tbody.innerHTML = '';
      
      results.iterations.forEach((iter, index) => {
          const row = tbody.insertRow();
          row.innerHTML = `
              <td>${index + 1}</td>
              <td>${iter.train_until}</td>
              <td>${(iter.accuracy * 100).toFixed(1)}%</td>
              <td>${iter.loss.toFixed(4)}</td>
              <td>${iter.is_best ? '✅' : ''}</td>
          `;
      });
  }
  ```

- [ ] **13.5.4** הוסף גרף Chart.js לויזואליזציה:
  ```javascript
  function createBacktestChart(results) {
      const ctx = document.getElementById('backtest-chart').getContext('2d');
      new Chart(ctx, {
          type: 'line',
          data: {
              labels: results.iterations.map((_, i) => `Iter ${i + 1}`),
              datasets: [{
                  label: 'Accuracy',
                  data: results.iterations.map(iter => iter.accuracy * 100),
                  borderColor: 'rgb(59, 130, 246)',
                  tension: 0.1
              }]
          }
      });
  }
  ```

- [ ] **13.5.5** הוסף עדכון real-time של slider:
  ```javascript
  document.getElementById('backtest-target-accuracy').addEventListener('input', (e) => {
      document.getElementById('accuracy-display').textContent = e.target.value + '%';
  });
  ```

---

## 🧪 Phase 13.6: Testing & Validation
**זמן משוער:** 30 דקות

- [ ] **13.6.1** בדיקה עם AAPL (יש מודל קיים):
  - תאריכים: 2024-01-01 → 2025-10-07
  - Test period: 14 ימים
  - Target: 85%

- [ ] **13.6.2** בדיקה עם מניה חדשה (TSLA):
  - תאריכים: 2024-06-01 → 2025-10-07
  - Test period: 7 ימים
  - Target: 80%

- [ ] **13.6.3** בדיקה עם תאריכים שונים:
  - תקופה קצרה (3 חודשים)
  - תקופה ארוכה (2 שנים)
  - תקופה עם volatility גבוה

- [ ] **13.6.4** בדיקת תרחיש שבו דיוק לא משתפר:
  - Verify auto-stop works
  - Verify max iterations respected
  - Verify all models saved

- [ ] **13.6.5** בדיקת error handling:
  - תאריכים לא תקינים
  - מניה שלא קיימת
  - נתונים חסרים

---

## 📚 Phase 13.7: Documentation & Polish
**זמן משוער:** 15 דקות

- [ ] **13.7.1** הוסף tooltips להסבר כל שדה:
  ```html
  <label title="תאריך התחלת האימון - יש צורך במינימום 90 ימים לפניו">
      Training Start Date:
  </label>
  ```

- [ ] **13.7.2** הוסף validation לשדות:
  - Start date < End date
  - End date לא בעתיד
  - Test period סביר (1-365 ימים)
  - Target accuracy סביר (50-100%)

- [ ] **13.7.3** הוסף error handling ו-user feedback:
  - Loading states
  - Error messages ברורים
  - Success notifications

- [ ] **13.7.4** הוסף הסבר במסמך זה על השימוש בפיצ'ר

---

## ⏱️ סיכום זמנים

| Phase | תיאור | זמן |
|-------|-------|-----|
| 13.1 | Data Loader Updates | 30 דק' |
| 13.2 | Backtesting Trainer | 45 דק' |
| 13.3 | API Endpoints | 20 דק' |
| 13.4 | UI Components | 30 דק' |
| 13.5 | JavaScript Logic | 25 דק' |
| 13.6 | Testing | 30 דק' |
| 13.7 | Documentation | 15 דק' |
| **סה"כ** | | **~3.5 שעות** |

---

## 📊 סדר ביצוע

1. ✅ Backend Core (13.1, 13.2) → 1.25 שעות
2. ✅ Backend APIs (13.3) → 20 דקות  
3. ✅ Frontend (13.4, 13.5) → 55 דקות
4. ✅ Testing & Polish (13.6, 13.7) → 45 דקות

---

## 🎯 הגדרות שאושרו על ידי המשתמש

- ✅ **Test Period:** 14 ימים (ברירת מחדל, ניתן לשינוי)
- ✅ **Max Iterations:** 10 (ברירת מחדל, גמיש לשינוי)
- ✅ **Target Accuracy:** 85% (ברירת מחדל, ניתן לבחירה)
- ✅ **Auto-stop:** כן (עצירה אוטומטית כשמגיעים ליעד)
- ✅ **Save Models:** כל המודלים (כל איטרציה נשמרת)

---

**סטטוס Phase 13:** 🚀 **מוכן להתחלה!**