from __future__ import annotations

from typing import Dict

from .base import StrategyFn


_REGISTRY: Dict[str, StrategyFn] = {}


def register(name: str, fn: StrategyFn) -> None:
    _REGISTRY[name] = fn


def get(name: str) -> StrategyFn | None:
    return _REGISTRY.get(name)


def available() -> Dict[str, StrategyFn]:
    return dict(_REGISTRY)
