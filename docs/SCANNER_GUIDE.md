# 📊 Market Scanner Guide

## Overview
The Market Scanner helps you discover and analyze trading opportunities from **10,889 stocks** in the local database.

## Workflow

### 1️⃣ **Filter Stage** - Screen All Stocks
Click **"Run Filter"** to scan all 10,889 stocks with these criteria:

#### Filter Criteria:
- ✅ **Price**: $3 minimum
- ✅ **Average Dollar Volume**: $1,000,000 minimum  
  (Calculated as: average price × average volume over 20 days)
- ✅ **Market Cap**: Micro-cap stocks only (< $300M)  
  (Loaded from `{SYMBOL}_advanced.json` fundamentals file)

#### What Happens:
1. Scans **ALL 10,889 stocks** from `stock_data/` directory
2. Reads CSV files: `stock_data/{SYMBOL}/{SYMBOL}_historical_data.csv`
3. Reads fundamentals: `stock_data/{SYMBOL}/{SYMBOL}_advanced.json`
4. Calculates metrics for each stock:
   - Current price
   - Volume & average dollar volume
   - Momentum (5-day vs 20-day MA)
   - Change % 
   - **Market cap** (from fundamentals)
   - Sector & Industry (from fundamentals)
5. Filters by all 3 criteria
6. Displays results in a table

#### Progress Tracking:
- Real-time progress bar
- Logs every 1,000 symbols processed
- Shows: `processed/total (%) - Passed: X`

---

### 2️⃣ **Training Stage** - Build ML Models

After filtering, you'll see a table with:
- All stocks that passed the filter
- **Train button** next to each stock
- **Indicator** showing if model already exists (✅/❌)

#### Training Options:

**A. Train Individual Stock:**
- Click **"Train"** button next to any stock
- Trains 3 Progressive ML models:
  - 🔷 Transformer
  - 🔶 LSTM  
  - 🔵 CNN
- Models saved to: `app/ml/models/{SYMBOL}_{MODEL_TYPE}_progressive.pt`

**B. Train All (Batch):**
- Click **"Train All (Transformer)"** button
- Trains all stocks that don't have models yet
- Runs in background, updates status live

#### Training Details:
- **Model Types**: Transformer, LSTM, CNN (ensemble)
- **Data**: Last 60 days of price + technical indicators
- **Horizons**: Predicts 1-day, 7-day, 30-day returns
- **Features**: 
  - Price data (OHLCV)
  - Technical indicators (RSI, MACD, Bollinger Bands, etc.)
  - Fundamentals (if available)
  - Market features (VIX correlation)

---

### 3️⃣ **Scanning Stage** - Rank & Select

After training, click **"Run Daily Scan"** with one of two modes:

#### Mode 1: **ML Strategy** (Dropdown = "ML")
- Scans only stocks **with trained models**
- Uses Progressive ML predictions:
  - Ensemble of Transformer + LSTM + CNN
  - Confidence-weighted predictions
  - 1-day, 7-day, 30-day forecasts
- Ranks by **ml_score** (predicted return %)
- Displays top stocks with:
  - Symbol, Price, ML Score
  - Expected Return
  - Confidence Level

#### Mode 2: **Technical Strategy** (Dropdown = "Technical")
✅ **MACD Convergence Setup Scanner** - Uses real-time technical analysis:
- **MACD Convergence** detection (below zero, rising histogram)
- **ADX** trending strength filter (ADX > 15)
- **Volume** exhaustion signals (Volume < SMA)
- **Histogram** momentum (rising for 2+ days)
- **Convergence Ratio** ≤ 50% (MACD approaching Signal)
- Returns **technical_score** (0-100) with detailed breakdown

---

## Data Flow

```
stock_data/{SYMBOL}/     Each of 10,889 stocks contains:
├── {SYMBOL}_historical_data.csv  (Price, Volume, OHLC)
├── {SYMBOL}_advanced.json        (Fundamentals: marketCap, sector, industry, etc.)
├── {SYMBOL}_indicators.csv       (Technical indicators)
└── {SYMBOL}_sentiment.csv        (News sentiment - optional)
    ↓
Filter               Apply 3 criteria (price, volume, micro-cap)
    ↓
Filter Results       ~50-200 stocks typically
    ↓
Training             Build ML models (optional)
    ↓
app/ml/models/       Trained models (.pt files)
    ↓
Scan (ML/Tech)       Rank & score stocks
    ↓
Top Picks            Display best opportunities
```

---

## API Endpoints

### Filter:
- `POST /api/scanner/filter/run` - Start filter job
- `GET /api/scanner/filter/status` - Get progress (state, %, processed, passed)
- `GET /api/scanner/filter/results` - Get filtered stocks

### Training:
- `POST /api/scanner/train/start?symbol={SYMBOL}` - Train one stock
- `POST /api/scanner/train/start-all` - Train all unmodeled stocks
- `GET /api/scanner/train/status` - Get training queue status
- `GET /api/scanner/train/symbol/{SYMBOL}/status` - Check if model exists

### Scanning:
- `POST /api/scanner/run?mode=ml&limit=50` - Run scan
- `GET /api/scanner/top?limit=50` - Get ranked results
- `GET /api/scanner/status` - Overall scanner status
- `GET /api/scanner/technical/convergence?min_score=60&limit=50` - **NEW**: Technical convergence scan

---

## Technical Details

### Metrics Calculation:
```python
{
    'symbol': 'AAPL',
    'current_price': 150.25,
    'change': 2.50,
    'change_percent': 1.69,
    'volume': 50000000,
    'avg_volume': 45000000,              # 20-day average
    'avg_dollar_volume': 6_761_250_000,  # price * avg_volume
    'market_cap': 3_825_555_734_528,     # ✅ From _advanced.json
    'is_micro_cap': False,               # ✅ marketCap < $300M
    'sector': 'Technology',              # ✅ From _advanced.json
    'industry': 'Consumer Electronics',  # ✅ From _advanced.json
    'momentum': 3.2,                     # 5-day MA vs 20-day MA
    'expected_return': 2.2,              # Heuristic (60% change + 40% momentum)
    'ml_score': 2.5,                     # ML prediction if model exists
    'technical_score': 75.5,             # ✅ NEW: Technical convergence score (0-100)
    'convergence_data': {                # ✅ NEW: Detailed breakdown
        'score': 75.5,
        'meets_criteria': True,
        'filters': {
            'adx': True,                 # ADX > 15
            'negative_zone': True,       # MACD < 0 and Signal < 0
            'volume_dry': True,          # Volume < VOL_SMA
            'hist_rising': True,         # Histogram rising 2+ days
            'convergence': True          # Conv ratio ≤ 50%
        },
        'passed_filters': '5/5',
        'conv_ratio': 32.5,              # Convergence ratio (%)
        'adx': 22.3,
        'macd': -0.15,
        'signal': -0.25,
        'histogram': 0.05
    },
    'has_model': True
}
```

### Fundamentals Data (`{SYMBOL}_advanced.json`):
The scanner reads real fundamentals from each stock's `_advanced.json` file:
- **marketCap**: Actual market capitalization in USD
- **sector**: Sector classification (Technology, Healthcare, etc.)
- **industry**: Industry classification (Consumer Electronics, Biotech, etc.)
- **sharesOutstanding**: Total shares
- **beta**: Stock volatility vs market
- **trailingPE**: Price-to-earnings ratio
- **dividendYield**: Current dividend yield
- **52WeekHigh/Low**: Trading range
- **averageVolume**: Historical volume metrics

This ensures **accurate micro-cap filtering** (< $300M) instead of estimations.

### Filter Criteria Logic:
```python
# Load fundamentals
fundamentals = json.load(f"{symbol}_advanced.json")
market_cap = fundamentals.get('marketCap', 0)

# Apply filters
passes_price = current_price >= 3.0
passes_volume = avg_dollar_volume >= 1_000_000.0
is_micro_cap = (market_cap > 0 and market_cap < 300_000_000)  # < $300M

if passes_price and passes_volume and is_micro_cap:
    # Stock passes filter
```

**Fallback**: If `_advanced.json` is missing, estimates micro-cap as:
```python
is_micro_cap = (price < 10.0 and avg_dollar_volume < 10_000_000)
```

---

## Performance Notes

- **Filter Time**: ~10-15 minutes for 10,889 stocks
- **Training Time**: 
  - Single stock: ~2-5 minutes (3 models)
  - Batch (200 stocks): ~6-16 hours
- **Scan Time**: ~5-10 seconds (using cached models)

---

## Troubleshooting

### Filter runs slow:
- Normal for 10,889 stocks
- Check progress in logs every 1,000 symbols
- Runs in background, won't block UI

### Training fails:
- Check if CSV has enough data (60+ days needed)
- Look for errors in logs
- Verify `app/ml/models/` directory exists

### No stocks pass filter:
- Criteria might be too strict
- Micro-cap definition: Market Cap < $300M (from fundamentals)
- Check if `_advanced.json` files exist for stocks
- Adjust thresholds in code if needed:
  ```python
  price_min = 3.0
  adv_min = 1_000_000.0
  micro_cap_threshold = 300_000_000  # $300M
  ```

### ML predictions not showing:
- Train models first (Train button or Train All)
- Check `app/ml/models/` for `.pt` files
- Look for "✅ Got ML score" in logs

---

## Technical Convergence Scanner

### Overview
The Technical Convergence Scanner identifies stocks with **MACD Pre-Cross Below Zero** setup - stocks positioned for a bullish reversal while still in negative territory.

### Scoring Algorithm

**Base Score** (0-100 points):
- Calculated from 5 filters passing: `(passed_filters / 5) × 100`

**Bonus Points**:
- Strong Convergence (ratio ≤ 25%): **+15 points**
- Moderate Convergence (ratio ≤ 40%): **+10 points**
- Strong ADX (> 25): **+10 points**
- Moderate ADX (> 20): **+5 points**
- Positive Histogram Momentum: **+5 points**

**Final Score**: `min(base_score + bonuses, 100.0)`

### Five Filters (Flexible Thresholds)

#### 1. **ADX Filter** (Trending Market)
- **Criteria**: ADX > 15
- **Purpose**: Ensure market is trending (not sideways)
- **Flexibility**: More lenient than strategy (15 vs 20) to catch early trends

#### 2. **Negative Zone Filter** (MACD Position)
- **Criteria**: MACD < 0 AND Signal < 0 (with 5% tolerance)
- **Purpose**: Stock is oversold but not capitulating
- **Flexibility**: Allows MACD slightly positive if very close to zero

#### 3. **Volume Dry Filter** (Seller Exhaustion)
- **Criteria**: Current Volume < VOL_SMA × 1.1
- **Purpose**: Selling pressure is diminishing
- **Flexibility**: Allows up to 10% above SMA (vs strict < SMA in strategy)

#### 4. **Histogram Rising** (Early Momentum)
- **Criteria**: Histogram rising for 2+ consecutive days
- **Purpose**: Momentum is building toward bullish cross
- **Flexibility**: Only 2 days required (vs 3 in strategy) for earlier signals

#### 5. **Convergence Filter** (MACD Approaching Signal)
- **Criteria**: Convergence Ratio ≤ 50%
- **Purpose**: MACD is getting closer to Signal line
- **Formula**: `|MACD - Signal| / |MACD|`
- **Flexibility**: 50% threshold (vs 40% in strategy) to catch wider setups

### Usage Examples

#### Find High-Quality Setups (Score ≥ 70):
```bash
GET /api/scanner/technical/convergence?min_score=70&limit=20
```

Returns stocks with:
- At least 4/5 filters passing
- Strong convergence or trending signals
- High probability of bullish reversal

#### Find All Potential Setups (Score ≥ 50):
```bash
GET /api/scanner/technical/convergence?min_score=50&limit=100
```

Returns broader list including:
- Early-stage setups
- Weaker but developing signals
- More candidates to monitor

#### Response Format:
```json
{
  "status": "success",
  "data": {
    "stocks": [
      {
        "symbol": "ABCD",
        "current_price": 12.45,
        "technical_score": 82.5,
        "convergence_data": {
          "score": 82.5,
          "meets_criteria": true,
          "filters": {
            "adx": true,
            "negative_zone": true,
            "volume_dry": true,
            "hist_rising": true,
            "convergence": true
          },
          "passed_filters": "5/5",
          "conv_ratio": 28.5,
          "adx": 24.3,
          "macd": -0.12,
          "signal": -0.17,
          "histogram": 0.05
        }
      }
    ],
    "total": 15,
    "total_scanned": 1089,
    "min_score": 70,
    "criteria": "MACD Convergence (ADX, Negative Zone, Volume Dry, Histogram Rising, Conv Ratio)"
  }
}
```

### Differences: Strategy vs Scanner

| Feature | Stock Strategy | Scanner |
|---------|---------------|---------|
| **ADX Threshold** | 20 | 15 (more flexible) |
| **Conv Ratio** | ≤ 40% | ≤ 50% (wider) |
| **Histogram Days** | 3 rising | 2 rising (earlier) |
| **Volume Filter** | < VOL_SMA (strict) | < VOL_SMA × 1.1 (tolerant) |
| **MACD Tolerance** | Must be < 0 | Can be slightly positive |
| **Purpose** | Execute precise trades | Discover early setups |
| **Entry Trigger** | ≤25% conv OR cross up | Any passing setup |

### Integration with Filter Results

The technical scanner works best with pre-filtered stocks:

1. **Run Filter First** → Get micro-cap stocks with liquidity
2. **Technical Scan** → Find convergence setups within filtered list
3. **Review Top Scores** → Analyze stocks with score ≥ 60
4. **Monitor Watchlist** → Track as setup develops

If no filter results exist, scanner will **sample** ~1,000 stocks from database for speed.

---

## Future Enhancements

- [ ] ~~Technical strategy implementation~~ ✅ **DONE** (Convergence Scanner)
- [ ] Custom filter criteria (user-adjustable thresholds)
- [ ] Sector/Industry filtering (using fundamentals data)
- [ ] Export results to CSV/Excel
- [ ] Backtesting integration
- [ ] Alert notifications for top picks
- [ ] Historical market cap tracking
- [ ] Financial ratios filtering (P/E, P/B, debt ratio, etc.)
- [ ] Additional technical patterns (Cup & Handle, Bull Flags, etc.)
- [ ] Multi-timeframe confirmation (Daily + Weekly alignment)
- [ ] Real-time alerts when stocks reach score thresholds
