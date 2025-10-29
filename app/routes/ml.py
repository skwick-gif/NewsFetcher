from flask import Blueprint, request

from app.utils.proxy import proxy_to_backend


ml_bp = Blueprint('ml', __name__)


@ml_bp.route('/api/ml/predictions/<symbol>')
def api_ml_predictions(symbol):
    return proxy_to_backend(f'/api/ml/predictions/{symbol}')


@ml_bp.route('/api/ml/train/<symbol>', methods=['POST'])
def api_ml_train(symbol):
    days_back = request.args.get('days_back', 365)
    return proxy_to_backend(f'/api/ml/train/{symbol}?days_back={days_back}', method='POST')


@ml_bp.route('/api/ml/status')
def api_ml_status():
    return proxy_to_backend('/api/ml/status')


# Progressive ML endpoints

@ml_bp.route('/api/ml/progressive/status')
def api_progressive_ml_status():
    return proxy_to_backend('/api/ml/progressive/status')


@ml_bp.route('/api/ml/progressive/models')
def api_progressive_ml_models():
    return proxy_to_backend('/api/ml/progressive/models')


@ml_bp.route('/api/ml/progressive/training/status')
def api_progressive_training_status():
    return proxy_to_backend('/api/ml/progressive/training/status')


@ml_bp.route('/api/ml/progressive/training/status/<job_id>')
def api_progressive_training_job_status(job_id):
    return proxy_to_backend(f'/api/ml/progressive/training/status/{job_id}')


@ml_bp.route('/api/ml/progressive/train', methods=['POST'])
def api_progressive_train():
    symbol = request.args.get('symbol', 'AAPL')
    model_types = request.args.get('model_types', 'lstm').split(',')
    mode = request.args.get('mode', 'progressive')
    return proxy_to_backend(f'/api/ml/progressive/train?symbol={symbol}&model_types={" ,".join(model_types).replace(" ","")}&mode={mode}', method='POST')


@ml_bp.route('/api/ml/progressive/predict/<symbol>', methods=['POST'])
def api_progressive_predict(symbol):
    mode = request.args.get('mode', 'progressive')
    return proxy_to_backend(f'/api/ml/progressive/predict/{symbol}?mode={mode}', method='POST')


@ml_bp.route('/api/ml/progressive/backtest', methods=['POST'])
def api_progressive_backtest():
    data = request.get_json(silent=True) or {}
    return proxy_to_backend('/api/ml/progressive/backtest', method='POST', timeout=60, json=data)


@ml_bp.route('/api/ml/progressive/backtest/status/<job_id>')
def api_progressive_backtest_status(job_id):
    return proxy_to_backend(f'/api/ml/progressive/backtest/status/{job_id}')


@ml_bp.route('/api/ml/progressive/backtest/results/<symbol>')
def api_progressive_backtest_results(symbol):
    return proxy_to_backend(f'/api/ml/progressive/backtest/results/{symbol}')


@ml_bp.route('/api/ml/progressive/backtest/cancel/<job_id>', methods=['POST'])
def api_progressive_backtest_cancel(job_id):
    return proxy_to_backend(f'/api/ml/progressive/backtest/cancel/{job_id}', method='POST')


@ml_bp.route('/api/ml/progressive/backtest/history/<symbol>')
def api_progressive_backtest_history(symbol):
    return proxy_to_backend(f'/api/ml/progressive/backtest/history/{symbol}')


@ml_bp.route('/api/ml/progressive/backtest/result_by_file/<symbol>/<file_name>')
def api_progressive_backtest_result_by_file(symbol, file_name):
    return proxy_to_backend(f'/api/ml/progressive/backtest/result_by_file/{symbol}/{file_name}')


@ml_bp.route('/api/ml/progressive/champions/<symbol>')
def api_progressive_champions(symbol):
    return proxy_to_backend(f'/api/ml/progressive/champions/{symbol}')


@ml_bp.route('/api/ml/progressive/champion/forward_test/<symbol>', methods=['POST'])
def api_progressive_champion_forward_test(symbol):
    data = request.get_json(silent=True) or {}
    return proxy_to_backend(f'/api/ml/progressive/champion/forward_test/{symbol}', method='POST', json=data)
