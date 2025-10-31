"""
Financial Historical Data Endpoints
CSV-based historical price data loading
"""
from fastapi import APIRouter, HTTPException
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any
from app.core.config import logger, data_manager

router = APIRouter()

@router.get("/financial/historical/{symbol}")
async def get_historical_data(symbol: str, timeframe: str = "1D"):
    """
    Get REAL historical price data from LOCAL CSV files
    Timeframes: 1D (1 day), 1W (5 days), 1M (1 month), 3M (3 months)
    """
    try:
        import pandas as pd

        # Get stock_data directory (two levels up from app/)
        project_root = Path(__file__).parent.parent.parent
        stock_file = project_root / "stock_data" / symbol / f"{symbol}_price.csv"

        if not stock_file.exists():
            raise HTTPException(status_code=404, detail=f"No local data found for {symbol}")

        # Read CSV file (robust to index column and case-insensitive 'Date')
        try:
            df = pd.read_csv(stock_file)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to read CSV for {symbol}: {e}")

        # Normalize date column
        date_col = None
        for cand in ['Date', 'date']:
            if cand in df.columns:
                date_col = cand
                break
        # If Date not found as a named column, try first column as Date
        if date_col is None and df.shape[1] >= 1:
            date_col = df.columns[0]

        if date_col is None:
            raise HTTPException(status_code=500, detail=f"Invalid CSV: missing Date column for {symbol}")

        # Convert to datetime robustly and make timezone-naive
        try:
            # Prefer strict ISO-8601 when possible
            df[date_col] = pd.to_datetime(df[date_col], format='ISO8601', utc=True)
        except Exception:
            try:
                # Pandas >=2.0 supports mixed formats
                df[date_col] = pd.to_datetime(df[date_col], format='mixed', utc=True)
            except Exception:
                # Fallback: let pandas infer; coerce invalids to NaT
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce', utc=True)
        # Drop invalid and strip timezone
        df[date_col] = df[date_col].dt.tz_localize(None)
        df = df.dropna(subset=[date_col]).sort_values(date_col)

        # Map timeframe to number of days
        timeframe_days = {
            "1D": 1,    # Last 1 day
            "1W": 5,    # Last 5 trading days (1 week)
            "1M": 30,   # Last 30 days (1 month)
            "3M": 90    # Last 90 days (3 months)
        }

        days = timeframe_days.get(timeframe, 1)

        # Filter to last N days
        cutoff_date = datetime.now() - timedelta(days=days)
        df_filtered = df[df[date_col] >= cutoff_date].copy()

        if len(df_filtered) == 0:
            # If no data in timeframe, get last N rows
            df_filtered = df.tail(days)

        # Format data for chart (both line and candlestick)
        labels = []
        prices = []
        ohlc = []  # list of dicts: {t, o, h, l, c}

        for _, row in df_filtered.iterrows():
            date = pd.to_datetime(row[date_col])
            if timeframe == "1D":
                # For 1 day, show time if available, else show date
                labels.append(date.strftime("%H:%M" if date.hour != 0 else "%m/%d"))
            else:
                # For longer periods, show date
                labels.append(date.strftime("%m/%d"))

            # Handle different casings for Close
            close_val = None
            if 'Close' in df.columns:
                close_val = row['Close']
            elif 'close' in df.columns:
                close_val = row['close']
            else:
                raise HTTPException(status_code=500, detail=f"Invalid CSV: missing Close column for {symbol}")
            prices.append(float(close_val))

            # Build OHLC if columns exist
            def _get(colA, colB=None):
                if colA in df.columns:
                    return row[colA]
                if colB and colB in df.columns:
                    return row[colB]
                return None
            o = _get('Open', 'open')
            h = _get('High', 'high')
            l = _get('Low', 'low')
            c = close_val
            if o is not None and h is not None and l is not None and c is not None:
                # Use ISO-8601 with Z to avoid client-side parser format errors
                ohlc.append({
                    "t": date.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "o": float(o),
                    "h": float(h),
                    "l": float(l),
                    "c": float(c)
                })

        # Calculate statistics
        current_price = prices[-1] if prices else 0
        start_price = prices[0] if prices else 0
        change = current_price - start_price
        change_percent = (change / start_price * 100) if start_price != 0 else 0

        logger.info(f"✅ Loaded {symbol} from local CSV: {len(prices)} data points")

        return {
            "status": "success",
            "data": {
                "symbol": symbol,
                "timeframe": timeframe,
                "labels": labels,
                "prices": prices,
                "ohlc": ohlc,
                "current_price": round(current_price, 2),
                "change": round(change, 2),
                "change_percent": round(change_percent, 2),
                "is_positive": change >= 0,
                "data_points": len(prices)
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error loading local data for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading data: {str(e)}")