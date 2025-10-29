from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Any


@dataclass
class StrategyResult:
    trades: List[dict]
    in_position: List[int]
    equity_curve: List[float]
    stops_count: int = 0


# Strategy run signature: accepts a pandas-like DataFrame and params dict, returns StrategyResult
StrategyFn = Callable[[Any, Dict], StrategyResult]
