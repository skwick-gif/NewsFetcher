from flask import Blueprint, request

from app.utils.proxy import proxy_to_backend


scanner_bp = Blueprint('scanner', __name__)

@scanner_bp.route('/api/ai/status')
def api_ai_status():
    return proxy_to_backend('/api/ai/status')

@scanner_bp.route('/api/ai/comprehensive-analysis')
def api_comprehensive_analysis_qs():
    symbol = request.args.get('symbol', 'AAPL')
    return proxy_to_backend(f'/api/ai/comprehensive-analysis/{symbol}')

@scanner_bp.route('/api/ai/comprehensive-analysis/<symbol>')
def api_comprehensive_analysis_path(symbol):
    return proxy_to_backend(f'/api/ai/comprehensive-analysis/{symbol}')


@scanner_bp.route('/api/scanner/hot-stocks')
def api_hot_stocks():
    limit = request.args.get('limit', 10)
    return proxy_to_backend(f'/api/scanner/hot-stocks?limit={limit}')


@scanner_bp.route('/api/ai/market-intelligence')
def api_market_intelligence():
    return proxy_to_backend('/api/ai/market-intelligence')


@scanner_bp.route('/api/ai/neural-network-prediction/<symbol>')
def api_neural_network_prediction(symbol):
    return proxy_to_backend(f'/api/ai/neural-network-prediction/{symbol}')


@scanner_bp.route('/api/ai/time-series-analysis/<symbol>')
def api_time_series_analysis(symbol):
    return proxy_to_backend(f'/api/ai/time-series-analysis/{symbol}')


@scanner_bp.route('/api/scanner/sectors')
def api_scanner_sectors():
    return proxy_to_backend('/api/scanner/sectors')


@scanner_bp.route('/api/scanner/sector/<sector_id>')
def api_scanner_sector_detail(sector_id):
    return proxy_to_backend(f'/api/scanner/sector/{sector_id}')


@scanner_bp.route('/api/scanner/status')
def api_scanner_status_proxy():
    qs = request.query_string.decode('utf-8') if request.query_string else ''
    endpoint = '/api/scanner/status' + (f'?{qs}' if qs else '')
    return proxy_to_backend(endpoint)


@scanner_bp.route('/api/scanner/top')
def api_scanner_top_proxy():
    qs = request.query_string.decode('utf-8') if request.query_string else ''
    endpoint = '/api/scanner/top' + (f'?{qs}' if qs else '')
    return proxy_to_backend(endpoint)


@scanner_bp.route('/api/scanner/run', methods=['POST'])
def api_scanner_run_proxy():
    qs = request.query_string.decode('utf-8') if request.query_string else ''
    endpoint = '/api/scanner/run' + (f'?{qs}' if qs else '')
    return proxy_to_backend(endpoint, method='POST')


@scanner_bp.route('/api/scanner/filter/run', methods=['POST'])
def api_scanner_filter_run_proxy():
    body = request.get_json(silent=True) or {}
    return proxy_to_backend('/api/scanner/filter/run', method='POST', json=body)


@scanner_bp.route('/api/scanner/filter/status')
def api_scanner_filter_status_proxy():
    qs = request.query_string.decode('utf-8') if request.query_string else ''
    endpoint = '/api/scanner/filter/status' + (f'?{qs}' if qs else '')
    return proxy_to_backend(endpoint)


@scanner_bp.route('/api/scanner/filter/results')
def api_scanner_filter_results_proxy():
    return proxy_to_backend('/api/scanner/filter/results')


@scanner_bp.route('/api/scanner/train/status')
def api_scanner_train_status_proxy():
    return proxy_to_backend('/api/scanner/train/status')


@scanner_bp.route('/api/scanner/train/start', methods=['POST'])
def api_scanner_train_start_proxy():
    qs = request.query_string.decode('utf-8') if request.query_string else ''
    endpoint = '/api/scanner/train/start' + (f'?{qs}' if qs else '')
    return proxy_to_backend(endpoint, method='POST')


@scanner_bp.route('/api/scanner/train/start-all', methods=['POST'])
def api_scanner_train_start_all_proxy():
    body = request.get_json(silent=True) or {}
    return proxy_to_backend('/api/scanner/train/start-all', method='POST', json=body)


@scanner_bp.route('/api/scanner/train/symbol/<symbol>/status')
def api_scanner_train_symbol_status_proxy(symbol):
    return proxy_to_backend(f'/api/scanner/train/symbol/{symbol}/status')
