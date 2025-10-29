from flask import Blueprint

from app.utils.proxy import proxy_to_backend


system_bp = Blueprint('system', __name__)


@system_bp.route('/api/system/info')
def api_system_info():
    return proxy_to_backend('/api/system/info')


@system_bp.route('/api/stats')
def api_stats():
    return proxy_to_backend('/api/stats')


@system_bp.route('/api/jobs')
def api_jobs():
    return proxy_to_backend('/api/jobs')


@system_bp.route('/api/feeds/status')
def api_feeds_status():
    return proxy_to_backend('/api/feeds/status')


@system_bp.route('/api/statistics')
def api_statistics():
    return proxy_to_backend('/api/statistics')


@system_bp.route('/api/alerts/active')
def api_active_alerts():
    return proxy_to_backend('/api/alerts/active')
