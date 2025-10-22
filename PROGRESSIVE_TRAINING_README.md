# Progressive ML Training Guide

## Overview
Train Transformer and CNN models on all stocks in the `stock_data` directory.
The script automatically skips stocks that already have models trained.

## Files
- `train_all_models_progressive.py` - Main Python training script
- `run_progressive_training.ps1` - Interactive PowerShell runner (asks for confirmation)
- `run_progressive_training_silent.ps1` - Silent mode for Task Scheduler

## Quick Start

### Option 1: Interactive (Recommended for first run)
```powershell
.\run_progressive_training.ps1
```
This will:
- Show you the training plan
- Ask for confirmation
- Run the training
- Display progress and statistics

### Option 2: Direct Python
```powershell
py train_all_models_progressive.py
```

### Option 3: Silent Mode (for automation)
```powershell
.\run_progressive_training_silent.ps1
```

## What It Does

1. **Scans** all stocks in `stock_data/` directory
2. **Checks** which models already exist for each stock
3. **Skips** stocks that already have the model trained
4. **Trains** missing models:
   - Transformer (attention-based, good for trends)
   - CNN (convolutional, good for patterns)
5. **Saves** models to `app/ml/models/`
6. **Logs** everything to `logs/progressive_training_YYYYMMDD_HHMMSS.log`

## Model Files
Models are saved as:
- `{SYMBOL}_transformer_progressive.keras`
- `{SYMBOL}_cnn_progressive.keras`

Example: `AAPL_transformer_progressive.keras`

## Estimated Time
- **Per stock**: 2-3 minutes per model (4-6 minutes total)
- **933 stocks**: ~6-12 hours total
- **10,808 stocks**: ~3-5 days continuous

## Progress Tracking
The script shows:
- Current stock being trained
- Progress percentage
- Models being trained/skipped
- Elapsed time
- ETA for completion
- Statistics every 10 stocks

## Resuming
If training is interrupted:
- Simply run the script again
- It will automatically skip already-trained stocks
- Continue from where it stopped

## Logs
Check `logs/` folder for:
- Full training logs
- Error messages
- Performance statistics
- Model metrics

## Task Scheduler Setup
To run weekly:
```powershell
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-File `"D:\Projects\NewsFetcher\run_progressive_training_silent.ps1`""
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Sunday -At 4am
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 5)

Register-ScheduledTask -TaskName "Progressive_ML_Training" -Action $action -Trigger $trigger -Settings $settings -Description "Train Transformer and CNN models weekly"
```

## Monitoring
- Check PowerShell window for real-time progress
- Watch `logs/` folder for detailed logs
- Models appear in `app/ml/models/` as they complete

## Tips
1. **Run overnight** - Let it run while you sleep
2. **Check logs** - Review `logs/` for any issues
3. **Monitor disk space** - Models are ~50-100MB each
4. **Don't interrupt** - Let each stock complete to avoid partial training

## Troubleshooting

### "No stocks found"
- Check that `stock_data/` exists
- Verify stock folders contain `{SYMBOL}_price.csv`

### "Out of memory"
- Close other applications
- Reduce batch size in settings
- Train fewer stocks at once

### "Training failed"
- Check logs for specific error
- Verify stock has enough historical data (>60 days)
- Check CSV file format

## Next Steps
After training:
1. Models are automatically available for predictions
2. Use Dashboard to get predictions
3. Run `daily_scan.py` to scan all stocks
4. Check Hot Stocks for opportunities

## Status Check
To see which models exist:
```powershell
Get-ChildItem app/ml/models/*.keras | Measure-Object
```

To see specific stock models:
```powershell
Get-ChildItem app/ml/models/AAPL*.keras
```
