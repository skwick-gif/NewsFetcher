from flask import Blueprint

from app.utils.proxy import proxy_to_backend


triggers_bp = Blueprint('triggers', __name__)


@triggers_bp.route('/api/trigger/major-news', methods=['POST'])
def api_trigger_major_news():
    return proxy_to_backend('/api/trigger/major-news', method='POST')


@triggers_bp.route('/api/trigger/perplexity-scan', methods=['POST'])
def api_trigger_perplexity_scan():
    return proxy_to_backend('/api/trigger/perplexity-scan', method='POST')


@triggers_bp.route('/api/test-alert', methods=['POST'])
def api_test_alert():
    return proxy_to_backend('/api/test-alert', method='POST')
