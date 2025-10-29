from flask import Blueprint

from app.utils.proxy import proxy_to_backend


financial_bp = Blueprint('financial', __name__)


@financial_bp.route('/api/financial/market-indices')
def api_market_indices():
    return proxy_to_backend('/api/financial/market-indices')


@financial_bp.route('/api/financial/market-sentiment')
def api_market_sentiment():
    return proxy_to_backend('/api/financial/market-sentiment')


@financial_bp.route('/api/financial/stock/<symbol>')
def api_stock_data(symbol):
    return proxy_to_backend(f'/api/financial/stock/{symbol}')


@financial_bp.route('/api/financial/top-stocks')
def api_top_stocks():
    return proxy_to_backend('/api/financial/top-stocks')


@financial_bp.route('/api/financial/geopolitical-risks')
def api_geopolitical_risks():
    return proxy_to_backend('/api/financial/geopolitical-risks')


@financial_bp.route('/api/financial/sector-performance')
def api_sector_performance():
    return proxy_to_backend('/api/financial/sector-performance')


@financial_bp.route('/api/market/<symbol>')
def api_market_data(symbol):
    return proxy_to_backend(f'/api/market/{symbol}')


@financial_bp.route('/api/sentiment/<symbol>')
def api_sentiment(symbol):
    return proxy_to_backend(f'/api/sentiment/{symbol}')


@financial_bp.route('/api/watchlist')
def api_watchlist():
    return proxy_to_backend('/api/watchlist')
