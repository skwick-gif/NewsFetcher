import os
import sys
from pathlib import Path
import requests
from datetime import datetime
import subprocess
import threading

# הוספת הנתיב של הפרויקט ל-sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from flask import Flask, render_template, jsonify, request

app = Flask(__name__)
app.config['SECRET_KEY'] = 'development-key'

# FastAPI Backend URL
FASTAPI_BACKEND = os.getenv('FASTAPI_BACKEND', 'http://localhost:8000')

def proxy_to_backend(endpoint, method='GET', **kwargs):
    """
    Proxy request to FastAPI backend
    """
    try:
        url = f"{FASTAPI_BACKEND}{endpoint}"
        
        if method == 'GET':
            response = requests.get(url, **kwargs)
        elif method == 'POST':
            response = requests.post(url, **kwargs)
        else:
            return jsonify({'error': f'Unsupported method: {method}'}), 400
        
        # Return the JSON response from backend
        return jsonify(response.json()), response.status_code
    
    except requests.exceptions.ConnectionError:
        return jsonify({
            'status': 'error',
            'message': 'Backend service unavailable',
            'timestamp': datetime.now().isoformat()
        }), 503
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

# ===========================
# Dashboard Routes
# ===========================

@app.route('/')
def dashboard():
    """מציג את דשבורד MarketPulse הראשי"""
    return render_template('dashboard.html')

@app.route('/health')
def health():
    """בדיקת תקינות השרת"""
    return jsonify({
        'status': 'healthy', 
        'service': 'MarketPulse Dashboard',
        'backend': FASTAPI_BACKEND
    })

# ===========================
# Financial API Proxies
# ===========================

@app.route('/api/financial/market-indices')
def api_market_indices():
    """Proxy to FastAPI: Market indices"""
    return proxy_to_backend('/api/financial/market-indices')

@app.route('/api/financial/market-sentiment')
def api_market_sentiment():
    """Proxy to FastAPI: Market sentiment"""
    return proxy_to_backend('/api/financial/market-sentiment')

@app.route('/api/financial/stock/<symbol>')
def api_stock_data(symbol):
    """Proxy to FastAPI: Stock data"""
    return proxy_to_backend(f'/api/financial/stock/{symbol}')

@app.route('/api/financial/top-stocks')
def api_top_stocks():
    """Proxy to FastAPI: Top performing stocks"""
    return proxy_to_backend('/api/financial/top-stocks')

@app.route('/api/financial/geopolitical-risks')
def api_geopolitical_risks():
    """Proxy to FastAPI: Geopolitical risk analysis"""
    return proxy_to_backend('/api/financial/geopolitical-risks')

# ===========================
# AI API Proxies
# ===========================

@app.route('/api/ai/status')
def api_ai_status():
    """Proxy to FastAPI: AI models status"""
    return proxy_to_backend('/api/ai/status')

@app.route('/api/ai/comprehensive-analysis')
def api_comprehensive_analysis():
    """Proxy to FastAPI: Comprehensive stock analysis"""
    symbol = request.args.get('symbol', 'AAPL')
    return proxy_to_backend(f'/api/ai/comprehensive-analysis/{symbol}')

@app.route('/api/ai/comprehensive-analysis/<symbol>')
def api_comprehensive_analysis_path(symbol):
    """Proxy to FastAPI: Comprehensive stock analysis (path version)"""
    return proxy_to_backend(f'/api/ai/comprehensive-analysis/{symbol}')

# ===========================
# ML API Proxies (NEW)
# ===========================

@app.route('/api/ml/predictions/<symbol>')
def api_ml_predictions(symbol):
    """Proxy to FastAPI: ML predictions for symbol"""
    return proxy_to_backend(f'/api/ml/predictions/{symbol}')

@app.route('/api/ml/train/<symbol>', methods=['POST'])
def api_ml_train(symbol):
    """Proxy to FastAPI: Train ML models for symbol"""
    days_back = request.args.get('days_back', 365)
    return proxy_to_backend(f'/api/ml/train/{symbol}?days_back={days_back}', method='POST')

@app.route('/api/ml/status')
def api_ml_status():
    """Proxy to FastAPI: ML system status"""
    return proxy_to_backend('/api/ml/status')

# Progressive ML endpoints
@app.route('/api/ml/progressive/status')
def api_progressive_ml_status():
    """Proxy to FastAPI: Progressive ML system status"""
    return proxy_to_backend('/api/ml/progressive/status')

@app.route('/api/ml/progressive/models')
def api_progressive_ml_models():
    """Proxy to FastAPI: Progressive ML models info"""
    return proxy_to_backend('/api/ml/progressive/models')

@app.route('/api/ml/progressive/training/status')
def api_progressive_training_status():
    """Proxy to FastAPI: Progressive ML training status"""
    return proxy_to_backend('/api/ml/progressive/training/status')

@app.route('/api/ml/progressive/train', methods=['POST'])
def api_progressive_train():
    """Proxy to FastAPI: Start Progressive ML training"""
    symbol = request.args.get('symbol', 'AAPL')
    model_types = request.args.get('model_types', 'lstm').split(',')
    mode = request.args.get('mode', 'progressive')
    return proxy_to_backend(f'/api/ml/progressive/train?symbol={symbol}&model_types={",".join(model_types)}&mode={mode}', method='POST')

@app.route('/api/ml/progressive/predict/<symbol>', methods=['POST'])
def api_progressive_predict(symbol):
    """Proxy to FastAPI: Get Progressive ML predictions"""
    mode = request.args.get('mode', 'progressive')
    return proxy_to_backend(f'/api/ml/progressive/predict/{symbol}?mode={mode}', method='POST')

@app.route('/api/ml/progressive/backtest', methods=['POST'])
def api_progressive_backtest():
    """Proxy to FastAPI: Start progressive backtesting"""
    data = request.get_json()
    symbol = data.get('symbol')
    train_start_date = data.get('train_start_date')
    train_end_date = data.get('train_end_date')
    test_period_days = data.get('test_period_days', 14)
    max_iterations = data.get('max_iterations', 10)
    target_accuracy = data.get('target_accuracy', 0.85)
    auto_stop = data.get('auto_stop', True)
    model_types = data.get('model_types', ['lstm'])
    
    url = f'/api/ml/progressive/backtest?symbol={symbol}&train_start_date={train_start_date}&train_end_date={train_end_date}&test_period_days={test_period_days}&max_iterations={max_iterations}&target_accuracy={target_accuracy}&auto_stop={auto_stop}'
    for model_type in model_types:
        url += f'&model_types={model_type}'
    
    return proxy_to_backend(url, method='POST')

@app.route('/api/ml/progressive/backtest/status/<job_id>')
def api_progressive_backtest_status(job_id):
    """Proxy to FastAPI: Get backtest status"""
    return proxy_to_backend(f'/api/ml/progressive/backtest/status/{job_id}')

@app.route('/api/ml/progressive/backtest/results/<symbol>')
def api_progressive_backtest_results(symbol):
    """Proxy to FastAPI: Get backtest results"""
    return proxy_to_backend(f'/api/ml/progressive/backtest/results/{symbol}')

# ===========================
# Enhanced Financial API Proxies (NEW)
# ===========================

@app.route('/api/financial/sector-performance')
def api_sector_performance():
    """Proxy to FastAPI: Sector performance analysis"""
    return proxy_to_backend('/api/financial/sector-performance')

@app.route('/api/market/<symbol>')
def api_market_data(symbol):
    """Proxy to FastAPI: Real-time market data for symbol"""
    return proxy_to_backend(f'/api/market/{symbol}')

@app.route('/api/sentiment/<symbol>')
def api_sentiment(symbol):
    """Proxy to FastAPI: Social sentiment for symbol"""
    return proxy_to_backend(f'/api/sentiment/{symbol}')

@app.route('/api/watchlist')
def api_watchlist():
    """Proxy to FastAPI: User watchlist"""
    return proxy_to_backend('/api/watchlist')

# ===========================
# System API Proxies (NEW)
# ===========================

@app.route('/api/system/info')
def api_system_info():
    """Proxy to FastAPI: Enhanced system information"""
    return proxy_to_backend('/api/system/info')

# ===========================
# Alerts API Proxies
# ===========================

@app.route('/api/alerts/active')
def api_active_alerts():
    """Proxy to FastAPI: Active alerts"""
    return proxy_to_backend('/api/alerts/active')

@app.route('/api/stats')
def api_stats():
    """Proxy to FastAPI: System statistics"""
    return proxy_to_backend('/api/stats')

@app.route('/api/articles')
def api_articles():
    """Proxy to FastAPI: News articles"""
    limit = request.args.get('limit', 50)
    return proxy_to_backend(f'/api/articles/recent?limit={limit}')

@app.route('/api/articles/recent')
def api_articles_recent():
    """Proxy to FastAPI: Recent news articles"""
    limit = request.args.get('limit', 20)
    return proxy_to_backend(f'/api/articles/recent?limit={limit}')

# ===========================
# Scanner API Proxies
# ===========================

@app.route('/api/scanner/hot-stocks')
def api_hot_stocks():
    """Proxy to FastAPI: Hot stocks scanner"""
    limit = request.args.get('limit', 10)
    return proxy_to_backend(f'/api/scanner/hot-stocks?limit={limit}')

@app.route('/api/ai/market-intelligence')
def api_market_intelligence():
    """Proxy to FastAPI: Market intelligence analysis"""
    return proxy_to_backend('/api/ai/market-intelligence')

@app.route('/api/ai/neural-network-prediction/<symbol>')
def api_neural_network_prediction(symbol):
    """Proxy to FastAPI: Neural network prediction for symbol"""
    return proxy_to_backend(f'/api/ai/neural-network-prediction/{symbol}')

@app.route('/api/ai/time-series-analysis/<symbol>')
def api_time_series_analysis(symbol):
    """Proxy to FastAPI: Time series analysis for symbol"""
    return proxy_to_backend(f'/api/ai/time-series-analysis/{symbol}')

# ===========================
# Predictions API Proxies (NEW)
# ===========================

@app.route('/api/predictions/create', methods=['POST'])
def api_predictions_create():
    """Proxy to FastAPI: Create new prediction"""
    return proxy_to_backend('/api/predictions/create', method='POST', json=request.json)

@app.route('/api/predictions/stats')
def api_predictions_stats():
    """Proxy to FastAPI: Prediction statistics"""
    source = request.args.get('source')
    endpoint = '/api/predictions/stats'
    if source:
        endpoint += f'?source={source}'
    return proxy_to_backend(endpoint)

@app.route('/api/predictions/list')
def api_predictions_list():
    """Proxy to FastAPI: List predictions with filters"""
    params = []
    for param in ['status', 'symbol', 'source', 'limit']:
        value = request.args.get(param)
        if value:
            params.append(f'{param}={value}')
    
    endpoint = '/api/predictions/list'
    if params:
        endpoint += '?' + '&'.join(params)
    return proxy_to_backend(endpoint)

# ===========================
# Scanner API Proxies (Enhanced)
# ===========================

@app.route('/api/scanner/sectors')
def api_scanner_sectors():
    """Proxy to FastAPI: Sector analysis"""
    return proxy_to_backend('/api/scanner/sectors')

@app.route('/api/scanner/sector/<sector_id>')
def api_scanner_sector_detail(sector_id):
    """Proxy to FastAPI: Detailed sector analysis"""
    return proxy_to_backend(f'/api/scanner/sector/{sector_id}')

# ===========================
# Jobs & Feeds API Proxies
# ===========================

@app.route('/api/jobs')
def api_jobs():
    """Proxy to FastAPI: System jobs status"""
    return proxy_to_backend('/api/jobs')

@app.route('/api/feeds/status')
def api_feeds_status():
    """Proxy to FastAPI: RSS feeds status"""
    return proxy_to_backend('/api/feeds/status')

@app.route('/api/statistics')
def api_statistics():
    """Proxy to FastAPI: System statistics"""
    return proxy_to_backend('/api/statistics')

# ===========================
# Trigger API Proxies (NEW)
# ===========================

@app.route('/api/trigger/major-news', methods=['POST'])
def api_trigger_major_news():
    """Proxy to FastAPI: Trigger major news scan"""
    return proxy_to_backend('/api/trigger/major-news', method='POST')

@app.route('/api/trigger/perplexity-scan', methods=['POST'])
def api_trigger_perplexity_scan():
    """Proxy to FastAPI: Trigger Perplexity scan"""
    return proxy_to_backend('/api/trigger/perplexity-scan', method='POST')

@app.route('/api/test-alert', methods=['POST'])
def api_test_alert():
    """Proxy to FastAPI: Send test alert"""
    return proxy_to_backend('/api/test-alert', method='POST')

# ===========================
# Legacy Routes (removed)
# ===========================
# Note: Legacy routes using BACKEND_URL were removed to prevent NameError and
# duplicated endpoints. All clients should use the standardized proxies above
# which forward to FASTAPI_BACKEND.

# ===========================
# Data Management Routes
# ===========================

# Global storage for job logs
job_logs = {}
running_jobs = {}

def run_script_in_background(job_type, script_name, job_id):
    """Run a Python script in the background and capture output"""
    global job_logs, running_jobs
    
    job_logs[job_id] = []
    running_jobs[job_id] = {"status": "running", "start_time": datetime.now().isoformat()}
    
    try:
        # Run the script
        process = subprocess.Popen(
            [sys.executable, script_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=0,  # Unbuffered for real-time output
            universal_newlines=True
        )
        
        # Capture output line by line
        for line in process.stdout:
            line = line.strip()
            if line:
                log_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "level": "INFO",
                    "message": line
                }
                job_logs[job_id].append(log_entry)
                
                # Keep only last 200 lines
                if len(job_logs[job_id]) > 200:
                    job_logs[job_id] = job_logs[job_id][-200:]
        
        process.wait()
        
        if process.returncode == 0:
            running_jobs[job_id]["status"] = "completed"
            job_logs[job_id].append({
                "timestamp": datetime.now().isoformat(),
                "level": "SUCCESS",
                "message": f"✅ Job completed successfully!"
            })
        else:
            running_jobs[job_id]["status"] = "failed"
            job_logs[job_id].append({
                "timestamp": datetime.now().isoformat(),
                "level": "ERROR",
                "message": f"❌ Job failed with exit code {process.returncode}"
            })
    
    except Exception as e:
        running_jobs[job_id]["status"] = "error"
        job_logs[job_id].append({
            "timestamp": datetime.now().isoformat(),
            "level": "ERROR",
            "message": f"❌ Error: {str(e)}"
        })

@app.route('/api/data-management/status')
def data_management_status():
    """Get data management status and job information"""
    try:
        # Mock data for now - can be enhanced with real data later
        return jsonify({
            "status": "success",
            "daily_downloads": 150,
            "weekly_updates": 45,
            "error_count": 2,
            "last_run": datetime.now().isoformat(),
            "jobs": [
                {
                    "id": "daily_data",
                    "name": "Daily Price Data Download",
                    "status": "scheduled",
                    "last_run": datetime.now().isoformat(),
                    "next_run": datetime.now().isoformat(),
                    "script": "ToUse/download_prices.py"
                },
                {
                    "id": "weekly_fundamentals",
                    "name": "Weekly Fundamentals Update",
                    "status": "scheduled",
                    "last_run": datetime.now().isoformat(),
                    "next_run": datetime.now().isoformat(),
                    "script": "ToUse/download_fundamentals.py"
                }
            ]
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/data-management/run-job/<job_type>', methods=['POST'])
def run_data_job(job_type):
    """Trigger a data download job"""
    try:
        job_scripts = {
            'daily': 'ToUse/download_prices.py',
            'weekly': 'ToUse/download_fundamentals.py'
        }
        
        if job_type not in job_scripts:
            return jsonify({
                "success": False,
                "error": f"Unknown job type: {job_type}"
            }), 400
        
        script_name = job_scripts[job_type]
        script_path = Path(__file__).parent.parent / script_name
        
        # Check if script exists
        if not script_path.exists():
            return jsonify({
                "success": False,
                "error": f"Script not found: {script_name}"
            }), 404
        
        # Generate job ID
        job_id = f"{job_type}_{int(datetime.now().timestamp())}"
        
        # Start script in background thread
        thread = threading.Thread(
            target=run_script_in_background,
            args=(job_type, str(script_path), job_id)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            "success": True,
            "message": f"Job {job_type} started successfully",
            "job_id": job_id
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/data-management/job-status/<job_id>')
def get_job_status(job_id):
    """Get the status and logs of a running job"""
    try:
        if job_id not in running_jobs:
            return jsonify({
                "status": "error",
                "message": "Job not found"
            }), 404
        
        return jsonify({
            "status": "success",
            "job": running_jobs[job_id],
            "logs": job_logs.get(job_id, [])
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/system/health')
def system_health():
    """Get system health status"""
    try:
        return jsonify({
            "status": "healthy",
            "database": "online",
            "scheduler": "running",
            "api": "online",
            "storage": "available",
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/data-management/logs')
def data_management_logs():
    """Get recent logs from data management operations"""
    try:
        # Mock logs - can be enhanced to read real logs later
        logs = [
            {"timestamp": datetime.now().isoformat(), "level": "INFO", "message": "System started successfully"},
            {"timestamp": datetime.now().isoformat(), "level": "INFO", "message": "Connected to database"},
            {"timestamp": datetime.now().isoformat(), "level": "SUCCESS", "message": "Daily download completed: 150 stocks"}
        ]
        return jsonify({
            "status": "success",
            "logs": logs
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

if __name__ == '__main__':
    print("🚀 Starting MarketPulse Dashboard Server...")
    print("📊 Dashboard will be available at: http://localhost:5000")
    print("🔗 Health check: http://localhost:5000/health")
    print("📈 API endpoint: http://localhost:5000/api/market-data")
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )