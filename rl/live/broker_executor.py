"""Execution helper that routes RL trades through the IBKR bridge."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from app.integrations.ibkr_bridge import IBKRBridgeError, get_ibkr_bridge

logger = logging.getLogger("rl.broker_executor")


@dataclass
class TradeRequest:
    symbol: str
    quantity: float
    side: str
    order_type: str = "MKT"
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    exchange: str = "SMART"
    time_in_force: str = "DAY"
    outside_rth: bool = False
    allow_partial_fills: bool = True

    def __post_init__(self) -> None:
        self.symbol = str(self.symbol).upper()
        self.side = str(self.side).upper()
        self.order_type = str(self.order_type).upper()
        self.exchange = str(self.exchange).upper()
        self.time_in_force = str(self.time_in_force).upper()
        self.quantity = abs(float(self.quantity))

    @classmethod
    def from_weights(cls, symbol: str, target: float, current: float, equity: float) -> Optional["TradeRequest"]:
        delta_weight = target - current
        if abs(delta_weight) < 1e-6:
            return None
        shares = int(delta_weight * equity)
        if shares == 0:
            return None
        side = "BUY" if shares > 0 else "SELL"
        return cls(symbol=symbol, quantity=abs(shares), side=side)


class BrokerExecutor:
    def __init__(self) -> None:
        self._ready = asyncio.Event()

    async def ensure_ready(self) -> None:
        if self._ready.is_set():
            return
        bridge = await get_ibkr_bridge()
        await bridge.wait_until_ready()
        self._ready.set()

    async def sync_account(self) -> Dict[str, Any]:
        await self.ensure_ready()
        bridge = await get_ibkr_bridge()
        return {
            "account": await bridge.account(),
            "portfolio": await bridge.portfolio(),
        }

    async def place_trades(self, trades: List[TradeRequest]) -> List[Dict[str, Any]]:
        await self.ensure_ready()
        bridge = await get_ibkr_bridge()
        results: List[Dict[str, Any]] = []
        for trade in trades:
            if int(trade.quantity) == 0:
                results.append({"symbol": trade.symbol, "success": False, "error": "quantity rounds to zero"})
                continue
            try:
                response = await bridge.place_stock_order(
                    symbol=trade.symbol,
                    action=trade.side,
                    quantity=int(trade.quantity),
                    order_type=trade.order_type,
                    limit_price=trade.limit_price,
                    stop_price=trade.stop_price,
                    exchange=trade.exchange,
                    time_in_force=trade.time_in_force,
                    outside_regular_trading_hours=trade.outside_rth,
                    allow_partial_fills=trade.allow_partial_fills,
                )
                results.append({"symbol": trade.symbol, "success": True, "order": response})
            except IBKRBridgeError as exc:
                logger.error("IBKR order failed for %s: %s", trade.symbol, exc)
                results.append({"symbol": trade.symbol, "success": False, "error": str(exc)})
        return results

    async def order_updates(self, limit: int = 50) -> List[Dict[str, Any]]:
        await self.ensure_ready()
        bridge = await get_ibkr_bridge()
        return await bridge.get_order_updates(limit=limit)


_executor: Optional[BrokerExecutor] = None
_executor_lock = asyncio.Lock()


async def get_executor() -> BrokerExecutor:
    global _executor
    if _executor is not None:
        return _executor
    async with _executor_lock:
        if _executor is None:
            _executor = BrokerExecutor()
        return _executor
