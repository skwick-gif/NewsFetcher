#!/usr/bin/env python3
"""
Monitor Advanced Backtesting Training progress and results
"""

import sys
import os
import time
import json
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

print("🔍 מכין ניטור לתהליך ה-Advanced Backtesting Training")
print("=" * 70)

def check_backtest_logs():
    """Check for backtest logs and progress"""
    print(f"\n📊 בדיקת קבצי log...")
    
    # Check logs directory
    logs_dir = "logs"
    if os.path.exists(logs_dir):
        log_files = [f for f in os.listdir(logs_dir) if f.endswith('.log')]
        print(f"   📁 נמצאו {len(log_files)} קבצי log")
        
        # Get the most recent log
        if log_files:
            latest_log = max([os.path.join(logs_dir, f) for f in log_files], 
                           key=os.path.getmtime)
            print(f"   📝 קובץ log אחרון: {latest_log}")
            
            # Show last few lines
            try:
                with open(latest_log, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    if lines:
                        print(f"   📋 שורות אחרונות:")
                        for line in lines[-5:]:
                            if 'backtest' in line.lower() or 'iteration' in line.lower():
                                print(f"      {line.strip()}")
            except Exception as e:
                print(f"   ❌ שגיאה בקריאת log: {e}")
    else:
        print(f"   📁 אין תיקיית logs")

def check_backtest_results():
    """Check for backtest result files"""
    print(f"\n📊 בדיקת תוצאות backtesting...")
    
    # Check models directory for backtest results
    models_dir = "app/ml/models"
    backtest_dir = os.path.join(models_dir, "backtest_results")
    
    if os.path.exists(backtest_dir):
        files = os.listdir(backtest_dir)
        print(f"   📁 תיקיית תוצאות: {len(files)} קבצים")
        
        # Show recent files
        if files:
            for file in sorted(files)[-5:]:  # Last 5 files
                file_path = os.path.join(backtest_dir, file)
                mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                print(f"      📄 {file} - {mod_time.strftime('%H:%M:%S')}")
    else:
        print(f"   📁 אין תיקיית תוצאות עדיין")

def check_model_files():
    """Check for new model files created during backtesting"""
    print(f"\n🤖 בדיקת מודלים חדשים...")
    
    models_dir = "app/ml/models"
    if os.path.exists(models_dir):
        # Look for recent model files
        model_files = []
        for file in os.listdir(models_dir):
            if file.endswith('.pth'):
                file_path = os.path.join(models_dir, file)
                mod_time = os.path.getmtime(file_path)
                model_files.append((file, mod_time))
        
        # Sort by modification time
        model_files.sort(key=lambda x: x[1], reverse=True)
        
        print(f"   🤖 נמצאו {len(model_files)} מודלים")
        print(f"   📝 5 מודלים אחרונים:")
        
        for file, mod_time in model_files[:5]:
            mod_datetime = datetime.fromtimestamp(mod_time)
            print(f"      {file} - {mod_datetime.strftime('%H:%M:%S')}")

def monitor_progress():
    """Monitor training progress"""
    print(f"\n⏰ מתחיל ניטור התקדמות...")
    print(f"   💡 הרץ את הbacktesting מהממשק ואני אבדוק כל 30 שניות")
    print(f"   🛑 לחץ Ctrl+C כדי לעצור את הניטור")
    
    try:
        iteration = 0
        while True:
            iteration += 1
            print(f"\n" + "="*50)
            print(f"🔍 ניטור #{iteration} - {datetime.now().strftime('%H:%M:%S')}")
            print(f"="*50)
            
            check_backtest_logs()
            check_backtest_results()
            check_model_files()
            
            print(f"\n⏳ ממתין 30 שניות לבדיקה הבאה...")
            time.sleep(30)
            
    except KeyboardInterrupt:
        print(f"\n🛑 ניטור הופסק על ידי המשתמש")

if __name__ == "__main__":
    print(f"🚀 הרץ את ה-Advanced Backtesting מהממשק עכשיו!")
    print(f"📋 הגדרות מומלצות:")
    print(f"   📊 Symbol: INTC")
    print(f"   📅 Start Date: 2024-01-01")
    print(f"   📅 End Date: 2025-09-23 (לפני חודש)")
    print(f"   🧪 Test Period: 14")
    print(f"   🔄 Iterations: 10")
    print(f"   🎯 Target Accuracy: 85%")
    print(f"   ⏹ Auto-stop: מסומן")
    
    input(f"\n⏳ לחץ Enter כשאתה מתחיל להריץ מהממשק...")
    
    monitor_progress()