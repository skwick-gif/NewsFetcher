"""HTTP/SignalR client for the external IBKR C# bridge service."""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Callable, Deque, Dict, List, Optional

import httpx
from signalrcore.hub_connection_builder import HubConnectionBuilder

from app.config.settings import IBKRConfig, get_config


class IBKRBridgeError(RuntimeError):
    """Raised when the bridge service returns an error or is unreachable."""


@dataclass
class RegisteredHandler:
    event: str
    callback: Callable[[Any], Any]


class IBKRBridgeClient:
    """Thin wrapper around the C# IBKR bridge REST and SignalR APIs."""

    def __init__(self, config: IBKRConfig) -> None:
        self._config = config
        self._logger = logging.getLogger("IBKRBridgeClient")
        self._http: Optional[httpx.AsyncClient] = None
        self._http_lock = asyncio.Lock()
        self._hub = None
        self._hub_lock = asyncio.Lock()
        self._handlers: Dict[str, RegisteredHandler] = {}
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._subscriptions: Dict[str, Dict[str, str]] = {}
        self._order_updates: Deque[Dict[str, Any]] = deque(maxlen=500)
        self._order_lock = asyncio.Lock()
        self._quote_cache: Dict[str, Dict[str, Any]] = {}
        self._quote_lock = asyncio.Lock()

        # Default handlers for bridge events
        self.register_handler("OrderStatus", self._handle_order_status)
        self.register_handler("StockQuote", self._handle_stock_quote)
        self.register_handler("ForexQuote", self._handle_forex_quote)
        self.register_handler("MarketDataBatch", self._handle_market_data_batch)

    @property
    def rest_base_url(self) -> str:
        return self._config.rest_base_url.rstrip("/")

    async def _ensure_http(self) -> httpx.AsyncClient:
        async with self._http_lock:
            if self._http is None:
                timeout = httpx.Timeout(
                    connect=self._config.connect_timeout_seconds,
                    read=self._config.request_timeout_seconds,
                    write=self._config.request_timeout_seconds,
                    pool=None,
                )
                self._http = httpx.AsyncClient(base_url=self.rest_base_url, timeout=timeout)
            return self._http

    async def close(self) -> None:
        if self._hub is not None:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._hub.stop)
            self._hub = None
            self._loop = None
        async with self._http_lock:
            if self._http is not None:
                await self._http.aclose()
                self._http = None

    async def _request(self, method: str, path: str, *, params: Optional[Dict[str, Any]] = None, json: Optional[Any] = None) -> Any:
        client = await self._ensure_http()
        url = path if path.startswith("/") else f"/{path}"
        try:
            response = await client.request(method, url, params=params, json=json)
        except httpx.RequestError as exc:  # pragma: no cover - network failure
            raise IBKRBridgeError(f"Bridge request failed: {exc}") from exc
        if response.status_code >= 400:
            detail = response.text
            try:
                payload = response.json()
                detail = payload.get("error") or payload
            except Exception:
                pass
            raise IBKRBridgeError(f"Bridge returned {response.status_code}: {detail}")
        if "application/json" in response.headers.get("content-type", ""):
            return response.json()
        return response.text

    async def connect(self, host: Optional[str] = None, port: Optional[int] = None, client_id: Optional[int] = None) -> Any:
        params = {
            "host": host or self._config.host,
            "port": port or self._config.port,
            "clientId": client_id or self._config.client_id,
        }
        self._logger.info("Connecting to IBKR bridge host=%s port=%s client_id=%s", params["host"], params["port"], params["clientId"])
        return await self._request("POST", "/connect", params=params)

    async def connection_status(self) -> Any:
        return await self._request("GET", "/connection-status")

    async def account(self) -> Any:
        return await self._request("GET", "/account")

    async def portfolio(self) -> Any:
        return await self._request("GET", "/portfolio")

    async def place_stock_order(
        self,
        symbol: str,
        action: str,
        quantity: int,
        order_type: str = "MKT",
        *,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        exchange: str = "SMART",
        time_in_force: str = "DAY",
        outside_regular_trading_hours: bool = False,
        allow_partial_fills: bool = True,
    ) -> Any:
        params: Dict[str, Any] = {
            "symbol": symbol,
            "action": action,
            "quantity": quantity,
            "orderType": order_type,
            "exchange": exchange,
            "timeInForce": time_in_force,
            "outsideRth": str(bool(outside_regular_trading_hours)).lower(),
            "allowPartialFills": str(bool(allow_partial_fills)).lower(),
        }
        if limit_price is not None:
            params["limitPrice"] = limit_price
        if stop_price is not None:
            params["stopPrice"] = stop_price
        return await self._request("POST", "/stocks/order", params=params)

    async def get_historical_data(
        self,
        symbol: str,
        *,
        timeframe: Optional[str] = None,
        bar_size: Optional[str] = None,
        duration: Optional[str] = None,
        bar_count: Optional[int] = None,
        include_outside_rth: bool = True,
        what_to_show: str = "TRADES",
    ) -> Any:
        params: Dict[str, Any] = {"symbol": symbol, "includeOutsideRth": str(bool(include_outside_rth)).lower(), "whatToShow": what_to_show}
        if timeframe:
            params["timeframe"] = timeframe
        if bar_size:
            params["barSize"] = bar_size
        if duration:
            params["duration"] = duration
        if bar_count is not None:
            params["barCount"] = bar_count
        try:
            return await self._request("GET", "/api/market/historical", params=params)
        except IBKRBridgeError as exc:
            detail = str(exc)
            if "404" not in detail and "not found" not in detail.lower():
                raise
            # Older bridge builds exposed the endpoint without the /api prefix.
            try:
                return await self._request("GET", "/market/historical", params=params)
            except IBKRBridgeError as fallback_exc:
                raise IBKRBridgeError(
                    "Historical data endpoint unavailable on IBKR bridge; "
                    "attempted /api/market/historical and /market/historical. "
                    f"Last error: {fallback_exc}"
                ) from fallback_exc

    async def cancel_stock_order(self, order_id: int) -> Any:
        return await self._request("POST", "/stocks/order/cancel", params={"orderId": order_id})

    async def stock_order_status(self, order_id: int) -> Any:
        return await self._request("GET", "/stocks/order/status", params={"orderId": order_id})

    async def stock_orders_active(self) -> Any:
        return await self._request("GET", "/stocks/orders/active")

    async def place_forex_order(self, base_currency: str, quote_currency: str, action: str, quantity: int, *, limit_price: float) -> Any:
        params = {
            "baseCurrency": base_currency,
            "quoteCurrency": quote_currency,
            "action": action,
            "quantity": quantity,
            "limitPrice": limit_price,
        }
        return await self._request("POST", "/forex/order", params=params)

    async def forex_order_status(self, order_id: int) -> Any:
        return await self._request("GET", "/forex/order/status", params={"orderId": order_id})

    async def forex_orders_active(self) -> Any:
        return await self._request("GET", "/forex/orders/active")

    def register_handler(self, event: str, callback: Callable[[Any], Any]) -> None:
        self._handlers[event] = RegisteredHandler(event=event, callback=callback)
        if self._hub is not None:
            self._hub.on(event, self._wrap_handler(event))

    def _wrap_handler(self, event: str) -> Callable[[Any], None]:
        async def _invoke(payload: Any) -> None:
            handler = self._handlers.get(event)
            if handler is None:
                return
            try:
                result = handler.callback(payload)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as exc:  # pragma: no cover - handler failure
                self._logger.exception("IBKR handler '%s' failed: %s", event, exc)

        def _sync(payload: Any) -> None:
            if self._loop is None:
                return
            asyncio.run_coroutine_threadsafe(_invoke(payload), self._loop)

        return _sync

    async def ensure_signalr(self) -> None:
        async with self._hub_lock:
            if self._hub is not None:
                return

            loop = asyncio.get_running_loop()
            self._loop = loop

            def _build_hub():
                connection = (
                    HubConnectionBuilder()
                    .with_url(
                        self._config.signalr_url,
                        options={
                            "verify_ssl": False,
                            "headers": {},
                            "skip_negotiation": True,
                            "transport": "websockets",
                        },
                    )
                    .build()
                )
                for event in self._handlers:
                    connection.on(event, self._wrap_handler(event))
                connection.start()
                return connection

            self._hub = await loop.run_in_executor(None, _build_hub)
            self._logger.info("SignalR hub connected")

    async def wait_until_ready(self) -> None:
        status = await self.connection_status()
        if not status.get("isConnected"):
            raise IBKRBridgeError("IBKR bridge is not connected to TWS")
        await self.ensure_signalr()

    async def _hub_send(self, method: str, *args: Any) -> None:
        await self.ensure_signalr()
        if self._hub is None:
            raise IBKRBridgeError("SignalR hub is not connected")

        loop = asyncio.get_running_loop()

        def _send() -> None:
            self._hub.send(method, list(args))

        await loop.run_in_executor(None, _send)

    async def subscribe_symbol(self, symbol: str, *, sec_type: str = "STK", exchange: str = "SMART") -> Dict[str, Any]:
        clean_symbol = symbol.upper()
        await self._hub_send("SubscribeToSymbol", clean_symbol, sec_type, exchange)
        self._subscriptions[clean_symbol] = {"sec_type": sec_type, "exchange": exchange}
        return {"symbol": clean_symbol, "sec_type": sec_type, "exchange": exchange, "status": "subscribed"}

    async def unsubscribe_symbol(self, symbol: str) -> Dict[str, Any]:
        clean_symbol = symbol.upper()
        await self._hub_send("UnsubscribeFromSymbol", clean_symbol)
        self._subscriptions.pop(clean_symbol, None)
        return {"symbol": clean_symbol, "status": "unsubscribed"}

    async def list_subscriptions(self) -> Dict[str, Any]:
        return {"subscriptions": [
            {"symbol": symbol, **info} for symbol, info in sorted(self._subscriptions.items())
        ]}

    async def get_order_updates(self, limit: int = 100) -> List[Dict[str, Any]]:
        async with self._order_lock:
            if limit <= 0:
                return list(self._order_updates)
            return list(self._order_updates)[-limit:]

    async def get_quotes(self, symbols: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        async with self._quote_lock:
            if not symbols:
                return {symbol: dict(data) for symbol, data in self._quote_cache.items()}
            result: Dict[str, Dict[str, Any]] = {}
            for symbol in symbols:
                sym = str(symbol).upper()
                data = self._quote_cache.get(sym)
                if data:
                    result[sym] = dict(data)
            return result

    async def _handle_order_status(self, payload: Any) -> None:
        if not isinstance(payload, dict):
            payload = dict(payload)
        async with self._order_lock:
            self._order_updates.append(payload)

    async def _handle_stock_quote(self, payload: Any) -> None:
        if not isinstance(payload, dict):
            payload = dict(payload)
        symbol = str(payload.get("symbol") or payload.get("Symbol") or "").upper()
        if not symbol:
            return
        snapshot = dict(payload)
        snapshot.setdefault("type", "stock")
        async with self._quote_lock:
            self._quote_cache[symbol] = snapshot

    async def _handle_forex_quote(self, payload: Any) -> None:
        if not isinstance(payload, dict):
            payload = dict(payload)
        base = payload.get("baseCurrency") or payload.get("BaseCurrency")
        quote = payload.get("quoteCurrency") or payload.get("QuoteCurrency")
        if not base or not quote:
            return
        symbol = f"{str(base).upper()}/{str(quote).upper()}"
        snapshot = dict(payload)
        snapshot.setdefault("type", "forex")
        snapshot["symbol"] = symbol
        async with self._quote_lock:
            self._quote_cache[symbol] = snapshot

    async def _handle_market_data_batch(self, payload: Any) -> None:
        if not isinstance(payload, dict):
            payload = dict(payload)
        stocks = payload.get("StockQuotes") or payload.get("stockQuotes") or []
        for quote in stocks:
            await self._handle_stock_quote(quote)
        forex = payload.get("ForexQuotes") or payload.get("forexQuotes") or []
        for quote in forex:
            await self._handle_forex_quote(quote)


_bridge_client: Optional[IBKRBridgeClient] = None
_bridge_lock = asyncio.Lock()


async def get_ibkr_bridge() -> IBKRBridgeClient:
    global _bridge_client
    if _bridge_client is not None:
        return _bridge_client
    async with _bridge_lock:
        if _bridge_client is None:
            config = get_config().config.ibkr
            _bridge_client = IBKRBridgeClient(config)
        return _bridge_client


@asynccontextmanager
async def ibkr_bridge_session() -> Any:
    bridge = await get_ibkr_bridge()
    try:
        yield bridge
    finally:
        pass
