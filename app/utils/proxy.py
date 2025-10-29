from datetime import datetime
from typing import Any

import requests
from flask import jsonify

from app.config.runtime import get_backend_base_url


def proxy_to_backend(endpoint: str, method: str = 'GET', timeout: int | None = None, **kwargs: Any):
    """Proxy a request to the FastAPI backend and return a Flask JSON response.

    - endpoint: path starting with '/'
    - method: 'GET' or 'POST'
    - timeout: optional override; defaults to 6s GET, 12s POST
    - kwargs: forwarded to requests (e.g., params=, json=)
    """
    try:
        base = get_backend_base_url().rstrip('/')
        url = f"{base}{endpoint}"

        # Default timeouts: GET 6s, POST 12s unless overridden
        if timeout is None:
            timeout = 12 if method.upper() == 'POST' else 6

        if method.upper() == 'GET':
            response = requests.get(url, timeout=timeout, **kwargs)
        elif method.upper() == 'POST':
            response = requests.post(url, timeout=timeout, **kwargs)
        else:
            return jsonify({'error': f'Unsupported method: {method}'}), 400

        try:
            payload = response.json()
        except Exception:
            payload = {'status': 'error', 'message': f'Invalid JSON from backend ({response.status_code})'}

        return jsonify(payload), response.status_code

    except requests.exceptions.ConnectionError:
        return jsonify({
            'status': 'error',
            'message': 'Backend service unavailable',
            'endpoint': endpoint,
            'timestamp': datetime.now().isoformat()
        }), 503
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500
