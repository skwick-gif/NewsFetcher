"""
Lightweight in-memory IBKR client stub.

This provides a minimal interface for connecting, placing orders, fetching
positions, and disconnecting. It's intended as a placeholder until the
external C# bridge is integrated. Thread-safety is not guaranteed; FastAPI
should run with a single event loop for now, and we keep this simple.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import time


@dataclass
class Order:
    order_id: str
    symbol: str
    quantity: float
    side: str  # BUY or SELL
    order_type: str  # MKT or LMT
    limit_price: Optional[float] = None
    status: str = "filled"  # stub: immediately filled
    created_at: float = field(default_factory=lambda: time.time())


class IBKRClient:
    def __init__(self) -> None:
        self._connected: bool = False
        self._host: Optional[str] = None
        self._port: Optional[int] = None
        self._client_id: Optional[int] = None
        self._orders: List[Order] = []
        self._positions: Dict[str, float] = {}

    def connect(self, host: Optional[str] = None, port: Optional[int] = None, client_id: Optional[int] = None) -> Dict[str, Any]:
        self._connected = True
        self._host = host
        self._port = port
        self._client_id = client_id
        return {
            "connected": self._connected,
            "host": self._host,
            "port": self._port,
            "client_id": self._client_id,
        }

    def status(self) -> Dict[str, Any]:
        return {
            "connected": self._connected,
            "host": self._host,
            "port": self._port,
            "client_id": self._client_id,
            "orders_count": len(self._orders),
            "positions_count": len(self._positions),
        }

    def place_order(self, symbol: str, quantity: float, side: str, order_type: str, limit_price: Optional[float] = None) -> Dict[str, Any]:
        if not self._connected:
            return {"ok": False, "error": "Not connected"}

        order_id = f"SIM-{int(time.time() * 1000)}"
        order = Order(
            order_id=order_id,
            symbol=symbol.upper(),
            quantity=float(quantity),
            side=side.upper(),
            order_type=order_type.upper(),
            limit_price=limit_price,
        )
        self._orders.append(order)
        # naive position update
        mult = 1 if order.side == "BUY" else -1
        self._positions[order.symbol] = self._positions.get(order.symbol, 0.0) + mult * order.quantity

        return {"ok": True, "order_id": order_id, "status": order.status}

    def positions(self) -> List[Dict[str, Any]]:
        return [{"symbol": sym, "quantity": qty} for sym, qty in sorted(self._positions.items())]

    def disconnect(self) -> Dict[str, Any]:
        self._connected = False
        return {"connected": self._connected}


# Singleton instance used by FastAPI endpoints
ibkr = IBKRClient()
