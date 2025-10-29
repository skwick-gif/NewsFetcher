from flask import Blueprint, request

from app.utils.proxy import proxy_to_backend


predictions_bp = Blueprint('predictions', __name__)


@predictions_bp.route('/api/predictions/create', methods=['POST'])
def api_predictions_create():
    return proxy_to_backend('/api/predictions/create', method='POST', json=request.json)


@predictions_bp.route('/api/predictions/stats')
def api_predictions_stats():
    source = request.args.get('source')
    endpoint = '/api/predictions/stats'
    if source:
        endpoint += f'?source={source}'
    return proxy_to_backend(endpoint)


@predictions_bp.route('/api/predictions/list')
def api_predictions_list():
    try:
        qs = request.query_string.decode('utf-8') if request.query_string else ''
    except Exception:
        qs = ''
    endpoint = '/api/predictions/list'
    if qs:
        endpoint += f'?{qs}'
    return proxy_to_backend(endpoint)


@predictions_bp.route('/api/articles')
def api_articles():
    limit = request.args.get('limit', 50)
    return proxy_to_backend(f'/api/articles/recent?limit={limit}')


@predictions_bp.route('/api/articles/recent')
def api_articles_recent():
    limit = request.args.get('limit', 20)
    return proxy_to_backend(f'/api/articles/recent?limit={limit}')
