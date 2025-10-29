from flask import Blueprint, request

from app.utils.proxy import proxy_to_backend


ibkr_bp = Blueprint('ibkr', __name__)


@ibkr_bp.route('/api/ibkr/status')
def ibkr_status_proxy():
    return proxy_to_backend('/api/ibkr/status')


@ibkr_bp.route('/api/ibkr/connect', methods=['POST'])
def ibkr_connect_proxy():
    data = request.get_json(silent=True) or {}
    return proxy_to_backend('/api/ibkr/connect', method='POST', json=data)


@ibkr_bp.route('/api/ibkr/place_order', methods=['POST'])
def ibkr_place_order_proxy():
    data = request.get_json(silent=True) or {}
    return proxy_to_backend('/api/ibkr/place_order', method='POST', json=data)


@ibkr_bp.route('/api/ibkr/positions')
def ibkr_positions_proxy():
    return proxy_to_backend('/api/ibkr/positions')


@ibkr_bp.route('/api/ibkr/disconnect', methods=['POST'])
def ibkr_disconnect_proxy():
    return proxy_to_backend('/api/ibkr/disconnect', method='POST')
