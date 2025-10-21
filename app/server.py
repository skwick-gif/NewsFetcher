import os
import sys
from pathlib import Path
import requests
from datetime import datetime

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
# Legacy Routes (for backward compatibility)
# ===========================

@app.route('/api/market-data')
def legacy_market_data():
    """Legacy endpoint - redirects to new API"""
    return proxy_to_backend('/api/financial/market-indices')

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