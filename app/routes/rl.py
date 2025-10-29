from flask import Blueprint, request

from app.utils.proxy import proxy_to_backend


rl_bp = Blueprint('rl', __name__)


@rl_bp.route('/api/rl/status')
def api_rl_status_proxy():
    return proxy_to_backend('/api/rl/status')


@rl_bp.route('/api/rl/simulate')
def api_rl_simulate_proxy():
    params = request.args.to_dict(flat=True)
    return proxy_to_backend('/api/rl/simulate', params=params)


@rl_bp.route('/api/rl/simulate/plan')
def api_rl_simulate_plan_proxy():
    params = request.args.to_dict(flat=True)
    return proxy_to_backend('/api/rl/simulate/plan', params=params)


@rl_bp.route('/api/rl/ppo/plan')
def api_rl_ppo_plan_proxy():
    params = request.args.to_dict(flat=True)
    return proxy_to_backend('/api/rl/ppo/plan', params=params)


@rl_bp.route('/api/rl/ppo/train', methods=['POST'])
def api_rl_ppo_train_proxy():
    params = request.args.to_dict(flat=True)
    return proxy_to_backend('/api/rl/ppo/train', method='POST', params=params)


@rl_bp.route('/api/rl/ppo/train/status/<job_id>')
def api_rl_ppo_train_status_proxy(job_id):
    return proxy_to_backend(f'/api/rl/ppo/train/status/{job_id}')


@rl_bp.route('/api/rl/ppo/train/stop/<job_id>', methods=['POST'])
def api_rl_ppo_train_stop_proxy(job_id):
    return proxy_to_backend(f'/api/rl/ppo/train/stop/{job_id}', method='POST')


# RL Live endpoints

@rl_bp.route('/api/rl/live/latest-model')
def api_rl_live_latest_model_proxy():
    return proxy_to_backend('/api/rl/live/latest-model')


@rl_bp.route('/api/rl/live/preview', methods=['POST'])
def api_rl_live_preview_proxy():
    data = request.get_json(silent=True) or {}
    return proxy_to_backend('/api/rl/live/preview', method='POST', json=data)


@rl_bp.route('/api/rl/live/paper/start', methods=['POST'])
def api_rl_live_paper_start_proxy():
    data = request.get_json(silent=True) or {}
    return proxy_to_backend('/api/rl/live/paper/start', method='POST', json=data)


@rl_bp.route('/api/rl/live/paper/stop', methods=['POST'])
def api_rl_live_paper_stop_proxy():
    return proxy_to_backend('/api/rl/live/paper/stop', method='POST')


@rl_bp.route('/api/rl/live/paper/status')
def api_rl_live_paper_status_proxy():
    return proxy_to_backend('/api/rl/live/paper/status')
