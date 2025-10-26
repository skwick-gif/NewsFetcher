"""
Lightweight DataManager for Progressive ML pipeline.

Goals:
- Resolve stock_data directory reliably in this repo.
- Ensure basic presence of per-symbol data (price is mandatory; indicators optional but attempted).
- Avoid heavy imports at module import time to keep FastAPI startup robust.

This module is safe to import even when optional providers are unavailable; any enrichment attempts
are guarded behind try/except and only run when ensure_symbol_data is called.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any


@dataclass
class EnsureSummary:
    symbol: str
    price_exists: bool
    indicators_exists: bool
    sentiment_exists: bool
    advanced_exists: bool
    price_path: str | None
    indicators_path: str | None
    sentiment_path: str | None
    advanced_path: str | None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "price_exists": self.price_exists,
            "indicators_exists": self.indicators_exists,
            "sentiment_exists": self.sentiment_exists,
            "advanced_exists": self.advanced_exists,
            "price_path": self.price_path,
            "indicators_path": self.indicators_path,
            "sentiment_path": self.sentiment_path,
            "advanced_path": self.advanced_path,
        }


class DataManager:
    def __init__(self, stock_data_dir: str | Path = "stock_data") -> None:
        # Resolve a sensible absolute path for stock_data
        base = Path(stock_data_dir)
        if not base.is_absolute():
            candidates = [
                Path(__file__).resolve().parents[2] / stock_data_dir,  # repo root
                Path.cwd() / stock_data_dir,                            # current working dir
                Path(stock_data_dir),                                   # relative as-is
            ]
            for c in candidates:
                if c.exists():
                    base = c
                    break
        self.stock_data_dir: Path = base
        self.stock_data_dir.mkdir(parents=True, exist_ok=True)

    def ensure_symbol_data(self, symbol: str, include_fundamentals: bool = True) -> EnsureSummary:
        """
        Ensure presence of core files for a symbol. Tries best-effort enrichment when missing.
        Only raises if the mandatory price CSV cannot be found/created.
        """
        symbol = symbol.upper().strip()
        sym_dir = self.stock_data_dir / symbol
        sym_dir.mkdir(parents=True, exist_ok=True)

        price_csv = sym_dir / f"{symbol}_price.csv"
        ind_csv = sym_dir / f"{symbol}_indicators.csv"
        sent_csv = sym_dir / f"{symbol}_sentiment.csv"
        adv_json = sym_dir / f"{symbol}_advanced.json"

        # Try to populate missing files without making import-time heavy deps mandatory
        if not price_csv.exists():
            try:
                import importlib
                stocks = importlib.import_module("app.data.stocks")
                start_date = getattr(getattr(stocks, "Config", object), "START_DATE", "2020-01-01")
                stocks.update_price_data(symbol, start_date, str(self.stock_data_dir))
            except Exception:
                # Ignore; we'll validate existence below and raise if still missing
                pass

        if not ind_csv.exists() and price_csv.exists():
            try:
                from app.data import compute_indicators
                compute_indicators.process_ticker(symbol)
            except Exception:
                pass

        if not sent_csv.exists():
            try:
                from app.data.sentiment_analyzer import SentimentAnalyzer
                SentimentAnalyzer(data_folder=str(self.stock_data_dir)).process_ticker(symbol)
            except Exception:
                pass

        if include_fundamentals and not adv_json.exists():
            try:
                import importlib
                stocks = importlib.import_module("app.data.stocks")
                stocks.scrape_all_data(symbol, str(self.stock_data_dir))
            except Exception:
                pass

        # Final state and mandatory checks
        price_exists = price_csv.exists()
        ind_exists = ind_csv.exists()
        sent_exists = sent_csv.exists()
        adv_exists = adv_json.exists()

        if not price_exists:
            raise FileNotFoundError(f"Price CSV not found for {symbol}: {price_csv}")

        return EnsureSummary(
            symbol=symbol,
            price_exists=price_exists,
            indicators_exists=ind_exists,
            sentiment_exists=sent_exists,
            advanced_exists=adv_exists,
            price_path=str(price_csv) if price_exists else None,
            indicators_path=str(ind_csv) if ind_exists else None,
            sentiment_path=str(sent_csv) if sent_exists else None,
            advanced_path=str(adv_json) if adv_exists else None,
        )


__all__ = ["DataManager", "EnsureSummary"]
