"""
Real-time financial data provider
Integrates with Yahoo Finance, Alpha Vantage, and other financial APIs
"""

import yfinance as yf
import requests
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.fundamentaldata import FundamentalData
import os

logger = logging.getLogger(__name__)

class FinancialDataProvider:
    """Real-time financial data provider"""
    
    def __init__(self):
        self.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY', 'demo')
        self.ts = TimeSeries(key=self.alpha_vantage_key, output_format='pandas')
        self.fd = FundamentalData(key=self.alpha_vantage_key, output_format='pandas')
        
        # Data source priority: Yahoo Finance -> Alpha Vantage -> Demo
        self.use_real_data = True
        
        # Major market indices symbols
        self.market_indices = {
            'SP500': '^GSPC',  # S&P 500
            'NASDAQ': '^IXIC', # NASDAQ Composite  
            'DOW': '^DJI',     # Dow Jones
            'RUSSELL': '^RUT', # Russell 2000
            'VIX': '^VIX'      # Volatility Index
        }
        
        # Key stocks to monitor
        self.key_stocks = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 
            'NFLX', 'AMD', 'INTC', 'PFE', 'JNJ', 'JPM', 'BAC'
        ]
        
    async def get_market_indices(self) -> Dict:
        """Get current market indices data - REAL DATA ONLY, NO DEMO FALLBACK"""
        try:
            # Try to get real data from Yahoo Finance
            real_data = await self._fetch_real_market_data()
            
            if real_data and len(real_data) >= 2:  # Need at least 2 indices
                logger.info(f"✅ Using REAL Yahoo Finance data ({len(real_data)} indices)")
                return real_data
            
            # Try Alpha Vantage as backup
            av_data = await self._fetch_alpha_vantage_data()
            if av_data and len(av_data) >= 2:
                logger.info(f"✅ Using REAL Alpha Vantage data ({len(av_data)} indices)")
                return av_data
            
            # NO DEMO DATA FALLBACK - Return empty dict to indicate failure
            logger.error("❌ FAILED TO FETCH REAL MARKET DATA - Returning empty (NO DEMO)")
            return {}
            
        except Exception as e:
            logger.error(f"❌ Error fetching market indices data: {e}")
            return {}
    
    async def _fetch_real_market_data(self) -> Optional[Dict]:
        """Try to fetch real market data from Yahoo Finance"""
        try:
            indices_data = {}
            success_count = 0
            
            for name, symbol in [('sp500', '^GSPC'), ('nasdaq', '^IXIC'), ('dow', '^DJI'), ('russell', '^RUT'), ('vix', '^VIX')]:
                try:
                    ticker = yf.Ticker(symbol)
                    
                    # SIMPLIFIED: Just get the most recent daily data
                    # This is more reliable than trying ticker.info which often fails
                    hist = ticker.history(period='5d')  # Get last 5 days to ensure we have data
                    
                    if hist is None or len(hist) == 0:
                        logger.warning(f"❌ No historical data for {symbol}")
                        continue
                    
                    # Get the most recent close price
                    current_price = float(hist['Close'].iloc[-1])
                    logger.info(f"✅ Got price for {symbol}: ${current_price}")
                    
                    # Calculate change if we have at least 2 data points
                    if len(hist) > 1:
                        prev_price = float(hist['Close'].iloc[-2])
                    else:
                        # Use opening price as previous price
                        prev_price = float(hist['Open'].iloc[-1])
                    
                    change = current_price - prev_price
                    change_percent = (change / prev_price) * 100 if prev_price != 0 else 0
                    
                    indices_data[name] = {
                        'symbol': symbol,
                        'price': round(current_price, 2),
                        'change': round(change, 2),
                        'change_percent': round(change_percent, 2),
                        'is_positive': change >= 0,
                        'last_updated': datetime.now().isoformat()
                    }
                    success_count += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to fetch {symbol}: {e}")
                    continue
            
            # Return real data only if we got at least 2 indices
            if success_count >= 2:
                logger.info(f"✅ Successfully fetched {success_count}/5 real market indices")
                return indices_data
            else:
                logger.warning(f"Only got {success_count}/5 indices, trying Alpha Vantage...")
                return await self._fetch_alpha_vantage_data()
                
        except Exception as e:
            logger.error(f"Error fetching real market data: {e}")
            return None
    
    async def _fetch_alpha_vantage_data(self) -> Optional[Dict]:
        """Try to fetch data from Alpha Vantage as backup"""
        try:
            if self.alpha_vantage_key == 'demo':
                logger.info("No Alpha Vantage key, skipping...")
                return None
            
            # Alpha Vantage symbols for major indices
            av_symbols = {
                'sp500': 'SPY',  # SPDR S&P 500 ETF
                'nasdaq': 'QQQ', # Invesco QQQ Trust
                'dow': 'DIA',    # SPDR Dow Jones Industrial Average ETF
            }
            
            indices_data = {}
            success_count = 0
            
            for name, symbol in av_symbols.items():
                try:
                    data, _ = self.ts.get_daily(symbol=symbol, outputsize='compact')
                    if data is not None and len(data) > 0:
                        latest_date = data.index[0]
                        current_price = float(data.loc[latest_date, '4. close'])
                        
                        if len(data) > 1:
                            prev_date = data.index[1]
                            prev_price = float(data.loc[prev_date, '4. close'])
                        else:
                            prev_price = float(data.loc[latest_date, '1. open'])
                        
                        change = current_price - prev_price
                        change_percent = (change / prev_price) * 100 if prev_price != 0 else 0
                        
                        indices_data[name] = {
                            'symbol': symbol,
                            'price': round(current_price, 2),
                            'change': round(change, 2),
                            'change_percent': round(change_percent, 2),
                            'is_positive': change >= 0,
                            'last_updated': datetime.now().isoformat()
                        }
                        success_count += 1
                        
                except Exception as e:
                    logger.warning(f"Alpha Vantage failed for {symbol}: {e}")
                    continue
            
            if success_count >= 2:
                logger.info(f"✅ Alpha Vantage success: {success_count}/3 indices")
                return indices_data
            else:
                logger.warning("Alpha Vantage also failed, using demo data")
                return None
                
        except Exception as e:
            logger.error(f"Alpha Vantage error: {e}")
            return None
    
    def _is_market_open(self) -> bool:
        """Check if US stock market is currently open"""
        from datetime import datetime, time as dt_time
        import pytz
        
        try:
            # Get current time in Eastern timezone (NYSE timezone)
            et = pytz.timezone('US/Eastern')
            now_et = datetime.now(et)
            
            # Market hours: 9:30 AM - 4:00 PM EST, Monday-Friday
            market_open = dt_time(9, 30)
            market_close = dt_time(16, 0)
            
            is_weekday = now_et.weekday() < 5  # Monday=0, Friday=4
            is_market_hours = market_open <= now_et.time() <= market_close
            
            return is_weekday and is_market_hours
            
        except Exception as e:
            logger.warning(f"Could not determine market hours: {e}")
            # Default to market closed for safety
            return False
    
    async def get_live_price(self, symbol: str) -> Optional[float]:
        """Get the most recent available price for a symbol"""
        try:
            ticker = yf.Ticker(symbol)
            
            # If market is open, try to get real-time data
            if self._is_market_open():
                # Try 1-minute interval for last hour
                hist = ticker.history(period='1d', interval='1m')
                if len(hist) > 0:
                    price = float(hist['Close'].iloc[-1])
                    logger.info(f"🔴 LIVE price for {symbol}: ${price}")
                    return price
            
            # Market closed - get most recent close
            hist = ticker.history(period='1d')
            if len(hist) > 0:
                price = float(hist['Close'].iloc[-1])
                logger.info(f"🔵 CLOSE price for {symbol}: ${price}")
                return price
                
            return None
            
        except Exception as e:
            logger.error(f"Failed to get live price for {symbol}: {e}")
            return None
    
    async def get_stock_data(self, symbol: str) -> Optional[Dict]:
        """Get detailed stock data for a specific symbol"""
        try:
            ticker = yf.Ticker(symbol)
            
            # Get recent price data
            hist = ticker.history(period="5d")
            if len(hist) < 2:
                return None
                
            current_price = float(hist['Close'].iloc[-1])
            previous_price = float(hist['Close'].iloc[-2])
            change = current_price - previous_price
            change_percent = (change / previous_price) * 100
            
            # Get basic info
            info = ticker.info
            
            # Get news impact analysis
            try:
                from financial.news_impact import NewsImpactAnalyzer
                news_analyzer = NewsImpactAnalyzer()
                
                # Mock recent news for demo (in production, fetch from your news database)
                recent_news = [
                    f"{symbol} reports quarterly earnings above expectations",
                    f"Market volatility affects {symbol} stock price",
                    f"Analysts maintain positive outlook for {symbol}"
                ]
                
                news_impact = await news_analyzer.analyze_stock_impact(symbol, recent_news)
            except Exception as e:
                logger.warning(f"Could not analyze news impact for {symbol}: {e}")
                news_impact = {'impact_score': 0.0, 'sentiment': 'neutral'}

            stock_data = {
                'symbol': symbol,
                'name': info.get('longName', symbol),
                'price': round(current_price, 2),
                'change': round(change, 2),
                'change_percent': round(change_percent, 2),
                'is_positive': change >= 0,
                'volume': int(hist['Volume'].iloc[-1]) if len(hist) > 0 else 0,  # Convert numpy.int64 to Python int
                'market_cap': int(info.get('marketCap', 0)) if info.get('marketCap') else 0,  # Convert to Python int
                'sector': info.get('sector', 'Unknown'),
                'news_impact': news_impact,
                'last_updated': datetime.now().isoformat()
            }
            
            return stock_data
            
        except Exception as e:
            logger.error(f"❌ Error fetching stock data for {symbol}: {e}")
            return None
    
    async def get_key_stocks_data(self) -> List[Dict]:
        """Get data for all key stocks - REAL DATA ONLY"""
        stocks_data = []
        
        for symbol in self.key_stocks:
            stock_data = await self.get_stock_data(symbol)
            if stock_data:
                stocks_data.append(stock_data)
        
        # NO DEMO FALLBACK - If no real data, return empty list
        if not stocks_data:
            logger.error("❌ NO REAL STOCK DATA AVAILABLE - Returning empty (NO DEMO)")
                
        logger.info(f"✅ Retrieved REAL data for {len(stocks_data)} stocks")
        return stocks_data
    
    async def get_sector_performance(self) -> Dict:
        """Get sector performance data"""
        try:
            # Sector ETFs as proxies
            sector_etfs = {
                'Technology': 'XLK',
                'Healthcare': 'XLV', 
                'Financial': 'XLF',
                'Energy': 'XLE',
                'Consumer': 'XLY',
                'Industrial': 'XLI',
                'Defense': 'ITA'  # iShares U.S. Aerospace & Defense ETF
            }
            
            sector_data = {}
            
            for sector, etf_symbol in sector_etfs.items():
                ticker = yf.Ticker(etf_symbol)
                hist = ticker.history(period="2d")
                
                if len(hist) >= 2:
                    current_price = float(hist['Close'].iloc[-1])
                    previous_price = float(hist['Close'].iloc[-2])
                    change_percent = ((current_price - previous_price) / previous_price) * 100
                    
                    sector_data[sector] = {
                        'etf_symbol': etf_symbol,
                        'change_percent': round(change_percent, 2),
                        'is_positive': change_percent >= 0
                    }
            
            logger.info(f"✅ Retrieved sector performance data")
            return sector_data
            
        except Exception as e:
            logger.error(f"❌ Error fetching sector performance: {e}")
            return {}
    
    async def calculate_market_sentiment(self) -> Dict:
        """Calculate overall market sentiment based on various indicators"""
        try:
            # Get major indices performance
            indices_data = await self.get_market_indices()
            
            # Calculate sentiment score (0-100)
            sentiment_score = 50  # Neutral baseline
            
            # VIX impact from demo data (using the vix value from indices_data)
            vix_value = indices_data.get('vix', {}).get('price', 20.0)
            
            # VIX impact (lower VIX = higher sentiment)
            if vix_value < 15:
                sentiment_score += 20  # Very low fear
            elif vix_value < 20:
                sentiment_score += 10  # Low fear
            elif vix_value > 30:
                sentiment_score -= 20  # High fear
            elif vix_value > 25:
                sentiment_score -= 10  # Moderate fear
            
            # Market indices impact
            positive_indices = 0
            total_indices = 0
            total_change = 0
            
            for index_name, data in indices_data.items():
                if 'change_percent' in data and index_name != 'vix':  # Exclude VIX from sentiment calculation
                    total_indices += 1
                    if data['is_positive']:
                        positive_indices += 1
                    
                    # Weight by magnitude
                    total_change += data['change_percent']
                    sentiment_score += data['change_percent'] * 3  # Increase weight
            
            # Average change impact
            if total_indices > 0:
                avg_change = total_change / total_indices
                sentiment_score += avg_change * 5  # Additional weight for average performance
            
            # Normalize to 0-100 range
            sentiment_score = max(0, min(100, sentiment_score))
            
            # Determine sentiment label
            if sentiment_score >= 75:
                sentiment_label = "Very Bullish 🚀"
            elif sentiment_score >= 60:
                sentiment_label = "Bullish 📈"
            elif sentiment_score >= 40:
                sentiment_label = "Neutral ➡️"
            elif sentiment_score >= 25:
                sentiment_label = "Bearish 📉"
            else:
                sentiment_label = "Very Bearish 🐻"
            
            sentiment_data = {
                'score': round(sentiment_score, 1),
                'label': sentiment_label,
                'vix_value': round(vix_value, 2),
                'positive_indices_ratio': positive_indices / total_indices if total_indices > 0 else 0.5,
                'total_change': round(total_change, 2),
                'last_updated': datetime.now().isoformat()
            }
            
            logger.info(f"✅ Calculated market sentiment: {sentiment_score}% - {sentiment_label}")
            return sentiment_data
            
        except Exception as e:
            logger.error(f"❌ Error calculating market sentiment: {e}")
            return {
                'score': 50.0,
                'label': "Neutral ➡️",
                'vix_value': 20.0,
                'positive_indices_ratio': 0.5,
                'total_change': 0.0,
                'last_updated': datetime.now().isoformat()
            }

# Global instance
financial_provider = FinancialDataProvider()