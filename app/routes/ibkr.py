from flask import Blueprint, request

from app.utils.proxy import proxy_to_backend


ibkr_bp = Blueprint("ibkr", __name__)


def _proxy(
    method: str,
    path: str,
    *,
    json_body: bool = False,
    forward_query: bool = True,
    timeout: int | None = None,
):
    payload = request.get_json(silent=True) if json_body else None
    params = request.args.to_dict(flat=True) if forward_query and request.args else None
    kwargs = {}
    if payload is not None:
        kwargs["json"] = payload
    if params:
        kwargs["params"] = params
    if timeout is not None:
        kwargs["timeout"] = timeout
    return proxy_to_backend(path, method=method, **kwargs)


@ibkr_bp.route("/api/ibkr/status")
def ibkr_status_proxy():
    return _proxy("GET", "/api/ibkr/status")


@ibkr_bp.route("/api/ibkr/connect", methods=["POST"])
def ibkr_connect_proxy():
    return _proxy("POST", "/api/ibkr/connect", json_body=True)


@ibkr_bp.route("/api/ibkr/orders/stock", methods=["POST"])
def ibkr_stock_order_proxy():
    return _proxy("POST", "/api/ibkr/orders/stock", json_body=True)


@ibkr_bp.route("/api/ibkr/orders/stock/<int:order_id>")
def ibkr_stock_order_status_proxy(order_id: int):
    return _proxy("GET", f"/api/ibkr/orders/stock/{order_id}")


@ibkr_bp.route("/api/ibkr/orders/stock/<int:order_id>/cancel", methods=["POST"])
def ibkr_stock_cancel_proxy(order_id: int):
    return _proxy("POST", f"/api/ibkr/orders/stock/{order_id}/cancel")


@ibkr_bp.route("/api/ibkr/orders/stock")
def ibkr_stock_orders_active_proxy():
    return _proxy("GET", "/api/ibkr/orders/stock")


@ibkr_bp.route("/api/ibkr/orders/forex", methods=["POST"])
def ibkr_forex_order_proxy():
    return _proxy("POST", "/api/ibkr/orders/forex", json_body=True)


@ibkr_bp.route("/api/ibkr/orders/forex/<int:order_id>")
def ibkr_forex_order_status_proxy(order_id: int):
    return _proxy("GET", f"/api/ibkr/orders/forex/{order_id}")


@ibkr_bp.route("/api/ibkr/orders/forex")
def ibkr_forex_orders_active_proxy():
    return _proxy("GET", "/api/ibkr/orders/forex")


@ibkr_bp.route("/api/ibkr/orders/updates")
def ibkr_order_updates_proxy():
    return _proxy("GET", "/api/ibkr/orders/updates")


@ibkr_bp.route("/api/ibkr/market/historical")
def ibkr_market_historical_proxy():
    # Historical queries can take longer while IBKR aggregates bars, so relax the proxy timeout.
    return _proxy("GET", "/api/ibkr/market/historical", timeout=30)


@ibkr_bp.route("/api/ibkr/account")
def ibkr_account_proxy():
    return _proxy("GET", "/api/ibkr/account")


@ibkr_bp.route("/api/ibkr/portfolio")
def ibkr_portfolio_proxy():
    return _proxy("GET", "/api/ibkr/portfolio")


@ibkr_bp.route("/api/ibkr/stream/subscribe", methods=["POST"])
def ibkr_stream_subscribe_proxy():
    return _proxy("POST", "/api/ibkr/stream/subscribe", json_body=True)


@ibkr_bp.route("/api/ibkr/stream/unsubscribe", methods=["POST"])
def ibkr_stream_unsubscribe_proxy():
    return _proxy("POST", "/api/ibkr/stream/unsubscribe", json_body=True)


@ibkr_bp.route("/api/ibkr/stream/subscriptions")
def ibkr_stream_subscriptions_proxy():
    return _proxy("GET", "/api/ibkr/stream/subscriptions")


@ibkr_bp.route("/api/ibkr/stream/quotes")
def ibkr_stream_quotes_proxy():
    return _proxy("GET", "/api/ibkr/stream/quotes")
