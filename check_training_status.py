#!/usr/bin/env python3
"""
Check training status and data preparation
"""

import json
from pathlib import Path
import os

def check_data_status():
    """Check data preparation status"""
    print("🔍 Checking data preparation status...")
    
    # Check possible data directories
    possible_dirs = [
        Path('ml/data'),
        Path('data'),
        Path('app/ml/data')
    ]
    
    data_dir = None
    for d in possible_dirs:
        if d.exists():
            data_dir = d
            break
    
    if not data_dir:
        print("❌ No data directory found")
        return False
    
    print(f"📂 Data directory: {data_dir}")
    
    # Check preparation summary
    summary_file = data_dir / 'preparation_summary.json'
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            summary = json.load(f)
        
        print('📊 Data Preparation Status:')
        print(f'   ✅ Symbols prepared: {len(summary["symbols"])}')
        print(f'   📏 Sequence length: {summary["sequence_length"]}')
        print(f'   🔢 Feature count: {summary.get("feature_count", "Unknown")}')
        print(f'   📈 Total samples: {summary.get("total_samples", "Unknown"):,}')
        print(f'   🏠 Use local data: {summary.get("use_local", "Unknown")}')
        print(f'   💼 Use fundamentals: {summary.get("use_fundamentals", "Unknown")}')
        
        # List symbols
        print(f'   📋 Symbols: {", ".join(summary["symbols"][:5])}{"..." if len(summary["symbols"]) > 5 else ""}')
        return True
    else:
        print('❌ No preparation summary found')
        return False

def check_model_status():
    """Check existing models and checkpoints"""
    print("\n🔍 Checking model status...")
    
    # Check all possible model directories
    possible_model_dirs = [
        Path('ml/models'),
        Path('app/ml/models'),
        Path('app/app/ml/models'),
        Path('models')
    ]
    
    found_models = []
    found_dirs = []
    
    for models_dir in possible_model_dirs:
        if models_dir.exists():
            found_dirs.append(models_dir)
            model_files = list(models_dir.glob('*.keras'))
            for model in model_files:
                found_models.append((model, models_dir))
    
    if found_dirs:
        print(f"📂 Found model directories:")
        for dir_path in found_dirs:
            print(f"   📁 {dir_path}")
    else:
        print("❌ No models directories found")
        return
    
    # Check for existing models
    if found_models:
        print(f"🤖 Found {len(found_models)} saved models:")
        for model, parent_dir in found_models:
            print(f"   ✅ {model.name} (in {parent_dir})")
    else:
        print("❌ No saved models found")
    
    # Check for checkpoints in all directories
    checkpoint_dirs = []
    all_checkpoints = []
    all_states = []
    
    for models_dir in found_dirs:
        checkpoint_dir = models_dir / 'checkpoints'
        if checkpoint_dir.exists():
            checkpoint_dirs.append(checkpoint_dir)
            checkpoints = list(checkpoint_dir.glob('*_checkpoint_*.keras'))
            states = list(checkpoint_dir.glob('*_training_state.json'))
            all_checkpoints.extend([(cp, checkpoint_dir) for cp in checkpoints])
            all_states.extend([(st, checkpoint_dir) for st in states])
    
    if checkpoint_dirs:
        print(f"💾 Found checkpoint directories:")
        for cp_dir in checkpoint_dirs:
            print(f"   📁 {cp_dir}")
    
    if all_checkpoints:
        print(f"💾 Found {len(all_checkpoints)} checkpoint files:")
        
        # Group by model type
        models = {}
        for cp, cp_dir in all_checkpoints:
            model_type = cp.name.split('_checkpoint_')[0]
            if model_type not in models:
                models[model_type] = []
            epoch = int(cp.name.split('_')[-1].replace('.keras', ''))
            models[model_type].append((epoch, cp_dir))
        
        for model_type, epochs_dirs in models.items():
            latest_epoch = max(epochs_dirs, key=lambda x: x[0])
            print(f"   🔹 {model_type}: {len(epochs_dirs)} checkpoints, latest epoch {latest_epoch[0]} (in {latest_epoch[1]})")
    else:
        print("❌ No checkpoint files found")
    
    if all_states:
        print(f"📋 Found {len(all_states)} training state files:")
        for state_file, state_dir in all_states:
            with open(state_file, 'r') as f:
                state = json.load(f)
            print(f"   📊 {state['model_name']}: epoch {state['current_epoch']}/{state['total_epochs']} (in {state_dir})")
            print(f"       Batch size: {state['batch_size']}, Status: {state['status']}")
            print(f"       Last update: {state['last_update']}")
    else:
        print("❌ No training state files found")

def check_logs():
    """Check recent training logs"""
    print("\n🔍 Checking recent logs...")
    
    logs_dir = Path('logs')
    if not logs_dir.exists():
        print("❌ No logs directory found")
        return
    
    log_files = list(logs_dir.glob('*.log'))
    if log_files:
        # Sort by modification time
        log_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        print(f"📜 Found {len(log_files)} log files:")
        for i, log_file in enumerate(log_files[:3]):  # Show only last 3
            size_mb = log_file.stat().st_size / (1024 * 1024)
            print(f"   📄 {log_file.name} ({size_mb:.1f} MB)")
            
            if i == 0:  # Show last few lines of most recent log
                print("      Last 5 lines:")
                try:
                    with open(log_file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        for line in lines[-5:]:
                            print(f"      {line.strip()}")
                except:
                    print("      (Could not read log file)")
    else:
        print("❌ No log files found")

def main():
    """Main function"""
    print("🚀 MarketPulse Training Status Check")
    print("=" * 50)
    
    data_ready = check_data_status()
    check_model_status()
    check_logs()
    
    print("\n" + "=" * 50)
    if data_ready:
        print("💡 Ready to start/resume training!")
        print("\n🎯 Usage options:")
        print("   py app/ml/train_model.py                    # Resume with default settings")
        print("   py app/ml/train_model.py --batch-size 64    # Resume with custom batch size")
        print("   py app/ml/train_model.py --no-resume        # Start fresh training")
        print("   py app/ml/train_model.py --epochs 50        # Limit epochs")
    else:
        print("⚠️  Need to prepare data first!")
        print("   Run: py prepare_data.py")

if __name__ == "__main__":
    main()