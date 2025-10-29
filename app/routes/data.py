from flask import Blueprint

from app.utils.proxy import proxy_to_backend


data_bp = Blueprint('data', __name__)


@data_bp.route('/api/data/ensure/<symbol>')
def api_data_ensure(symbol):
    """Proxy to FastAPI: Ensure per-symbol data exists and is fresh"""
    return proxy_to_backend(f'/api/data/ensure/{symbol}', method='GET', timeout=30)
