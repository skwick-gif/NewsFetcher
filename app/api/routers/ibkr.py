"""FastAPI router exposing Interactive Brokers bridge endpoints."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException

from app.integrations.ibkr_bridge import IBKRBridgeError, get_ibkr_bridge

router = APIRouter(prefix="/api/ibkr", tags=["IBKR"])


async def _call_bridge(action: str, *args: Any, **kwargs: Any) -> Any:
    bridge = await get_ibkr_bridge()
    try:
        method = getattr(bridge, action)
    except AttributeError as exc:  # pragma: no cover - developer error
        raise HTTPException(status_code=500, detail=f"Unavailable IBKR action '{action}'") from exc
    try:
        return await method(*args, **kwargs)
    except IBKRBridgeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/connect")
async def connect(body: Optional[Dict[str, Any]] = None) -> Any:
    body = body or {}
    return await _call_bridge(
        "connect",
        host=body.get("host"),
        port=body.get("port"),
        client_id=body.get("client_id") or body.get("clientId"),
    )


@router.get("/status")
async def connection_status() -> Any:
    bridge = await get_ibkr_bridge()
    try:
        payload = await bridge.connection_status()
    except IBKRBridgeError as exc:
        return {"status": "bridge_unavailable", "isConnected": False, "detail": str(exc)}
    return payload


@router.get("/account")
async def account_summary() -> Any:
    return await _call_bridge("account")


@router.get("/portfolio")
async def portfolio() -> Any:
    return await _call_bridge("portfolio")


@router.post("/orders/stock")
async def place_stock_order(body: Dict[str, Any]) -> Any:
    if "symbol" not in body or "action" not in body or "quantity" not in body:
        raise HTTPException(status_code=400, detail="symbol, action, quantity are required")

    def _resolve_bool(value: Any, default: bool = False) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        if isinstance(value, str):
            return value.strip().lower() in {"true", "1", "yes", "y", "on"}
        return default

    return await _call_bridge(
        "place_stock_order",
        symbol=body["symbol"],
        action=body["action"],
        quantity=int(body["quantity"]),
        order_type=body.get("order_type") or body.get("orderType", "MKT"),
        limit_price=body.get("limit_price") or body.get("limitPrice"),
        stop_price=body.get("stop_price") or body.get("stopPrice"),
        exchange=body.get("exchange", "SMART"),
        time_in_force=body.get("time_in_force") or body.get("timeInForce", "DAY"),
        outside_regular_trading_hours=_resolve_bool(body.get("outside_rth") or body.get("outsideRth")),
        allow_partial_fills=_resolve_bool(body.get("allow_partial_fills") or body.get("allowPartialFills"), default=True),
    )


@router.post("/orders/stock/{order_id}/cancel")
async def cancel_stock_order(order_id: int) -> Any:
    return await _call_bridge("cancel_stock_order", order_id)


@router.get("/orders/stock/{order_id}")
async def stock_order_status(order_id: int) -> Any:
    return await _call_bridge("stock_order_status", order_id)


@router.get("/orders/stock")
async def stock_orders_active() -> Any:
    return await _call_bridge("stock_orders_active")


@router.post("/orders/forex")
async def place_forex_order(body: Dict[str, Any]) -> Any:
    required = {
        "base_currency": ["base_currency", "baseCurrency"],
        "quote_currency": ["quote_currency", "quoteCurrency"],
        "action": ["action"],
        "quantity": ["quantity"],
        "limit_price": ["limit_price", "limitPrice"],
    }

    resolved: Dict[str, Any] = {}
    for target, aliases in required.items():
        value = None
        for alias in aliases:
            if alias in body and body[alias] not in (None, ""):
                value = body[alias]
                break
        if value is None:
            raise HTTPException(status_code=400, detail=f"Missing required field '{target}'")
        resolved[target] = value

    return await _call_bridge(
        "place_forex_order",
        base_currency=resolved["base_currency"],
        quote_currency=resolved["quote_currency"],
        action=resolved["action"],
        quantity=int(resolved["quantity"]),
        limit_price=float(resolved["limit_price"]),
    )


@router.get("/orders/forex/{order_id}")
async def forex_order_status(order_id: int) -> Any:
    return await _call_bridge("forex_order_status", order_id)


@router.get("/orders/forex")
async def forex_orders_active() -> Any:
    return await _call_bridge("forex_orders_active")


@router.get("/orders/updates")
async def order_updates(limit: int = 50) -> Any:
    bridge = await get_ibkr_bridge()
    updates = await bridge.get_order_updates(limit=limit)
    return {"status": "ok", "updates": updates}


@router.post("/stream/subscribe")
async def stream_subscribe(body: Dict[str, Any]) -> Any:
    symbol = body.get("symbol")
    sec_type = body.get("sec_type") or body.get("secType")
    exchange = body.get("exchange")

    if not symbol:
        base = body.get("base_currency") or body.get("baseCurrency")
        quote = body.get("quote_currency") or body.get("quoteCurrency")
        if not base or not quote:
            raise HTTPException(status_code=400, detail="symbol or base_currency+quote_currency required")
        symbol = f"{str(base).upper()}/{str(quote).upper()}"
        sec_type = sec_type or "CASH"
        exchange = exchange or "IDEALPRO"
    else:
        symbol = str(symbol).upper()
        sec_type = sec_type or "STK"
        exchange = exchange or "SMART"

    bridge = await get_ibkr_bridge()
    try:
        return await bridge.subscribe_symbol(symbol, sec_type=sec_type, exchange=exchange)
    except IBKRBridgeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/stream/unsubscribe")
async def stream_unsubscribe(body: Dict[str, Any]) -> Any:
    symbol = body.get("symbol")
    if not symbol:
        base = body.get("base_currency") or body.get("baseCurrency")
        quote = body.get("quote_currency") or body.get("quoteCurrency")
        if base and quote:
            symbol = f"{str(base).upper()}/{str(quote).upper()}"
    if not symbol:
        raise HTTPException(status_code=400, detail="symbol required")

    bridge = await get_ibkr_bridge()
    try:
        return await bridge.unsubscribe_symbol(symbol)
    except IBKRBridgeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.get("/stream/subscriptions")
async def stream_subscriptions() -> Any:
    bridge = await get_ibkr_bridge()
    return await bridge.list_subscriptions()


@router.get("/stream/quotes")
async def stream_quotes(symbols: Optional[str] = None) -> Any:
    bridge = await get_ibkr_bridge()
    parsed: Optional[List[str]] = None
    if symbols:
        parsed = [sym.strip().upper() for sym in symbols.split(',') if sym.strip()]
    quotes = await bridge.get_quotes(parsed)
    return {"status": "ok", "quotes": quotes}


@router.get("/market/historical")
async def market_historical_data(
    symbol: str,
    timeframe: Optional[str] = None,
    bar_size: Optional[str] = None,
    duration: Optional[str] = None,
    bar_count: Optional[int] = None,
    include_outside_rth: bool = True,
    what_to_show: str = "TRADES",
) -> Any:
    bridge = await get_ibkr_bridge()
    try:
        payload = await bridge.get_historical_data(
            symbol,
            timeframe=timeframe,
            bar_size=bar_size,
            duration=duration,
            bar_count=bar_count,
            include_outside_rth=include_outside_rth,
            what_to_show=what_to_show,
        )
    except IBKRBridgeError as exc:
        detail = str(exc)
        status_code = 502
        try:
            import re
            match = re.search(r"Bridge returned (\d+)", detail)
            if match:
                candidate = int(match.group(1))
                if 400 <= candidate < 600:
                    status_code = candidate
        except Exception:
            pass
        raise HTTPException(status_code=status_code, detail=detail) from exc
    return payload
