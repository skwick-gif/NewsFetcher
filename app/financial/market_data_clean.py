"""
Real-time Financial Data Provider - REAL DATA ONLY
No demo data - All sources must provide actual market data
"""

import yfinance as yf
import requests
import asyncio
from datetime import datetime, timedelta, time as dt_time
from typing import Dict, List, Optional, Tuple
import logging
import os
import pytz

logger = logging.getLogger(__name__)

class FinancialDataProvider:
    """Real-time financial data provider - REAL DATA ONLY"""
    
    def __init__(self):
        self.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        
        # Market indices symbols
        self.market_indices = {
            'SP500': '^GSPC',  # S&P 500
            'NASDAQ': '^IXIC', # NASDAQ Composite  
            'DOW': '^DJI',     # Dow Jones
            'VIX': '^VIX'      # Volatility Index
        }
        
        # Key stocks to monitor
        self.key_stocks = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 
            'NFLX', 'AMD', 'INTC', 'PFE', 'JNJ', 'JPM', 'BAC'
        ]
        
        logger.info("✅ FinancialDataProvider initialized - REAL DATA ONLY MODE")
    
    def _is_market_open(self) -> bool:
        """Check if US stock market is currently open"""
        try:
            et = pytz.timezone('US/Eastern')
            now_et = datetime.now(et)
            
            market_open = dt_time(9, 30)
            market_close = dt_time(16, 0)
            
            is_weekday = now_et.weekday() < 5
            is_market_hours = market_open <= now_et.time() <= market_close
            
            return is_weekday and is_market_hours
        except Exception as e:
            logger.warning(f"Could not determine market hours: {e}")
            return False
    
    async def get_live_price(self, symbol: str) -> Optional[float]:
        """Get the most recent available price - REAL DATA ONLY"""
        try:
            ticker = yf.Ticker(symbol)
            
            # If market is open, try real-time data
            if self._is_market_open():
                hist = ticker.history(period='1d', interval='1m')
                if len(hist) > 0:
                    price = float(hist['Close'].iloc[-1])
                    logger.info(f"🔴 LIVE {symbol}: ${price}")
                    return price
            
            # Market closed - get most recent close
            hist = ticker.history(period='1d')
            if len(hist) > 0:
                price = float(hist['Close'].iloc[-1])
                logger.info(f"🔵 CLOSE {symbol}: ${price}")
                return price
                
            logger.warning(f"⚠️ No price data for {symbol}")
            return None
        except Exception as e:
            logger.error(f"❌ Failed to get price for {symbol}: {e}")
            return None
    
    async def get_stock_data(self, symbol: str) -> Optional[Dict]:
        """Get detailed stock data - REAL DATA ONLY"""
        try:
            ticker = yf.Ticker(symbol)
            
            # Get price history
            hist = ticker.history(period="5d")
            if len(hist) < 2:
                logger.warning(f"⚠️ Insufficient data for {symbol}")
                return None
            
            # Get current and previous prices
            current_price = float(hist['Close'].iloc[-1])
            previous_price = float(hist['Close'].iloc[-2])
            change = current_price - previous_price
            change_percent = (change / previous_price) * 100
            
            # Get additional info
            info = ticker.info
            
            stock_data = {
                'symbol': symbol,
                'name': info.get('longName', symbol),
                'current_price': round(current_price, 2),
                'price_change': round(change, 2),
                'price_change_percent': round(change_percent, 2),
                'volume': info.get('volume', 0),
                'market_cap': info.get('marketCap', 0),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'pe_ratio': info.get('trailingPE', None),
                'eps': info.get('trailingEps', None),
                'dividend_yield': info.get('dividendYield', 0),
                'day_high': round(float(hist['High'].iloc[-1]), 2),
                'day_low': round(float(hist['Low'].iloc[-1]), 2),
                'week_52_high': info.get('fiftyTwoWeekHigh', None),
                'week_52_low': info.get('fiftyTwoWeekLow', None),
                'market_status': 'open' if self._is_market_open() else 'closed',
                'last_updated': datetime.now().isoformat(),
                'data_source': 'Yahoo Finance'
            }
            
            logger.info(f"✅ Got real data for {symbol}: ${current_price}")
            return stock_data
            
        except Exception as e:
            logger.error(f"❌ Failed to get data for {symbol}: {e}")
            return None
    
    async def get_market_indices(self) -> Dict:
        """Get current market indices - REAL DATA ONLY"""
        try:
            indices_data = {}
            
            for name, symbol in [('sp500', '^GSPC'), ('nasdaq', '^IXIC'), ('dow', '^DJI'), ('vix', '^VIX')]:
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period='2d')
                    
                    if len(hist) >= 2:
                        current_price = float(hist['Close'].iloc[-1])
                        previous_price = float(hist['Close'].iloc[-2])
                        change = current_price - previous_price
                        change_percent = (change / previous_price) * 100
                        
                        indices_data[name] = {
                            'symbol': symbol,
                            'price': round(current_price, 2),
                            'change': round(change, 2),
                            'change_percent': round(change_percent, 2),
                            'is_positive': change >= 0,
                            'last_updated': datetime.now().isoformat(),
                            'data_source': 'Yahoo Finance'
                        }
                        logger.info(f"✅ {name.upper()}: ${current_price}")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to get {name}: {e}")
                    continue
            
            if indices_data:
                logger.info(f"✅ Retrieved {len(indices_data)} market indices")
                return indices_data
            else:
                logger.error("❌ Failed to retrieve any market indices")
                return {}
                
        except Exception as e:
            logger.error(f"❌ Error fetching market indices: {e}")
            return {}
    
    async def get_key_stocks(self) -> List[Dict]:
        """Get data for key stocks - REAL DATA ONLY"""
        stocks_data = []
        
        for symbol in self.key_stocks:
            stock_data = await self.get_stock_data(symbol)
            if stock_data:
                stocks_data.append(stock_data)
            await asyncio.sleep(0.1)  # Rate limiting
        
        logger.info(f"✅ Retrieved {len(stocks_data)}/{len(self.key_stocks)} stocks")
        return stocks_data
    
    async def get_stock_fundamentals(self, symbol: str) -> Optional[Dict]:
        """Get fundamental data for a stock - REAL DATA ONLY"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            fundamentals = {
                'symbol': symbol,
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', None),
                'forward_pe': info.get('forwardPE', None),
                'peg_ratio': info.get('pegRatio', None),
                'price_to_book': info.get('priceToBook', None),
                'revenue': info.get('totalRevenue', 0),
                'profit_margin': info.get('profitMargins', None),
                'operating_margin': info.get('operatingMargins', None),
                'roe': info.get('returnOnEquity', None),
                'debt_to_equity': info.get('debtToEquity', None),
                'current_ratio': info.get('currentRatio', None),
                'quick_ratio': info.get('quickRatio', None),
                'eps': info.get('trailingEps', None),
                'forward_eps': info.get('forwardEps', None),
                'dividend_yield': info.get('dividendYield', 0),
                'payout_ratio': info.get('payoutRatio', None),
                'analyst_rating': info.get('recommendationKey', 'none'),
                'target_price': info.get('targetMeanPrice', None),
                'last_updated': datetime.now().isoformat(),
                'data_source': 'Yahoo Finance'
            }
            
            logger.info(f"✅ Got fundamentals for {symbol}")
            return fundamentals
            
        except Exception as e:
            logger.error(f"❌ Failed to get fundamentals for {symbol}: {e}")
            return None
    
    async def get_stock_history(self, symbol: str, period: str = "1mo") -> Optional[Dict]:
        """Get historical price data - REAL DATA ONLY"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            
            if len(hist) == 0:
                logger.warning(f"⚠️ No history for {symbol}")
                return None
            
            history_data = {
                'symbol': symbol,
                'period': period,
                'dates': hist.index.strftime('%Y-%m-%d').tolist(),
                'open': hist['Open'].tolist(),
                'high': hist['High'].tolist(),
                'low': hist['Low'].tolist(),
                'close': hist['Close'].tolist(),
                'volume': hist['Volume'].tolist(),
                'data_points': len(hist),
                'last_updated': datetime.now().isoformat(),
                'data_source': 'Yahoo Finance'
            }
            
            logger.info(f"✅ Got {len(hist)} days history for {symbol}")
            return history_data
            
        except Exception as e:
            logger.error(f"❌ Failed to get history for {symbol}: {e}")
            return None
    
    async def get_sector_performance(self) -> Dict:
        """Get sector performance - REAL DATA ONLY"""
        try:
            sector_etfs = {
                'Technology': 'XLK',
                'Healthcare': 'XLV', 
                'Financial': 'XLF',
                'Energy': 'XLE',
                'Consumer': 'XLY',
                'Industrial': 'XLI',
                'Defense': 'ITA'
            }
            
            sector_data = {}
            
            for sector, etf_symbol in sector_etfs.items():
                try:
                    ticker = yf.Ticker(etf_symbol)
                    hist = ticker.history(period="2d")
                    
                    if len(hist) >= 2:
                        current_price = float(hist['Close'].iloc[-1])
                        previous_price = float(hist['Close'].iloc[-2])
                        change_percent = ((current_price - previous_price) / previous_price) * 100
                        
                        sector_data[sector] = {
                            'etf_symbol': etf_symbol,
                            'price': round(current_price, 2),
                            'change_percent': round(change_percent, 2),
                            'is_positive': change_percent >= 0,
                            'last_updated': datetime.now().isoformat(),
                            'data_source': 'Yahoo Finance'
                        }
                        logger.info(f"✅ {sector}: {change_percent:+.2f}%")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to get {sector}: {e}")
                    continue
            
            if sector_data:
                logger.info(f"✅ Retrieved {len(sector_data)} sectors")
            return sector_data
            
        except Exception as e:
            logger.error(f"❌ Error fetching sectors: {e}")
            return {}
    
    async def get_crypto_prices(self) -> Dict:
        """Get cryptocurrency prices - REAL DATA ONLY"""
        try:
            crypto_symbols = {
                'BTC': 'BTC-USD',
                'ETH': 'ETH-USD',
                'BNB': 'BNB-USD',
                'XRP': 'XRP-USD'
            }
            
            crypto_data = {}
            
            for name, symbol in crypto_symbols.items():
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period="2d")
                    
                    if len(hist) >= 2:
                        current_price = float(hist['Close'].iloc[-1])
                        previous_price = float(hist['Close'].iloc[-2])
                        change_percent = ((current_price - previous_price) / previous_price) * 100
                        
                        crypto_data[name] = {
                            'symbol': symbol,
                            'price': round(current_price, 2),
                            'change_percent': round(change_percent, 2),
                            'is_positive': change_percent >= 0,
                            'last_updated': datetime.now().isoformat(),
                            'data_source': 'Yahoo Finance'
                        }
                        logger.info(f"✅ {name}: ${current_price}")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to get {name}: {e}")
                    continue
            
            return crypto_data
            
        except Exception as e:
            logger.error(f"❌ Error fetching crypto: {e}")
            return {}
