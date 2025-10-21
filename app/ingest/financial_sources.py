"""
Financial Intelligence Data Sources
Monitoring: News, Patents, Drug Approvals, Geopolitical Events, Regulatory Filings
"""

import logging
import asyncio
import httpx
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import feedparser
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

class FinancialDataSources:
    """Comprehensive financial intelligence data aggregator"""
    
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
        
        # News sources - Financial outlets
        self.news_sources = {
            "reuters_finance": "https://www.reuters.com/finance/rss",
            "reuters_markets": "https://www.reuters.com/markets/rss",
            "bloomberg": "https://www.bloomberg.com/feed/podcast/etf-report.xml",
            "wsj_markets": "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
            "ft_companies": "https://www.ft.com/companies?format=rss",
            "cnbc": "https://www.cnbc.com/id/100003114/device/rss/rss.html",
            "marketwatch": "https://www.marketwatch.com/rss/marketpulse",
            "seeking_alpha": "https://seekingalpha.com/feed.xml",
        }
        
        # Government/Regulatory sources
        self.regulatory_sources = {
            "sec_filings": "https://www.sec.gov/cgi-bin/browse-edgar",
            "fda_approvals": "https://www.fda.gov/about-fda/contact-fda/stay-informed/rss-feeds/fda-newsroom/rss.xml",
            "uspto_patents": "https://www.uspto.gov/",
            "ftc_news": "https://www.ftc.gov/news-events/news/rss",
        }
        
        # Economic data sources
        self.economic_sources = {
            "fred": "https://fred.stlouisfed.org/",
            "bls": "https://www.bls.gov/",
            "census": "https://www.census.gov/",
        }
        
        # Geopolitical sources
        self.geopolitical_sources = {
            "reuters_world": "https://www.reuters.com/world/rss",
            "bbc_business": "http://feeds.bbci.co.uk/news/business/rss.xml",
            "ap_business": "https://apnews.com/apf-business",
        }
    
    async def fetch_financial_news(self, category: str = "all") -> List[Dict[str, Any]]:
        """Fetch latest financial news from multiple sources"""
        all_articles = []
        
        sources_to_check = self.news_sources
        if category == "regulatory":
            sources_to_check = self.regulatory_sources
        elif category == "geopolitical":
            sources_to_check = self.geopolitical_sources
        
        for source_name, url in sources_to_check.items():
            try:
                articles = await self._fetch_rss_feed(url, source_name)
                all_articles.extend(articles)
                logger.info(f"✅ Fetched {len(articles)} articles from {source_name}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to fetch from {source_name}: {e}")
        
        # Sort by date
        all_articles.sort(key=lambda x: x.get('published', ''), reverse=True)
        return all_articles[:50]  # Return top 50
    
    async def fetch_sec_filings(self, symbols: List[str]) -> List[Dict[str, Any]]:
        """Fetch SEC filings for specific companies"""
        filings = []
        
        for symbol in symbols:
            try:
                # SEC EDGAR API
                url = f"https://data.sec.gov/submissions/CIK{symbol}.json"
                response = await self.client.get(url, headers={
                    "User-Agent": "MarketPulse Financial Intelligence info@marketpulse.ai"
                })
                
                if response.status_code == 200:
                    data = response.json()
                    recent_filings = data.get("filings", {}).get("recent", {})
                    
                    for i in range(min(10, len(recent_filings.get("form", [])))):
                        filing = {
                            "symbol": symbol,
                            "form_type": recent_filings["form"][i],
                            "filing_date": recent_filings["filingDate"][i],
                            "accession_number": recent_filings["accessionNumber"][i],
                            "url": f"https://www.sec.gov/cgi-bin/viewer?action=view&cik={symbol}&accession_number={recent_filings['accessionNumber'][i]}&xbrl_type=v",
                            "source": "SEC EDGAR"
                        }
                        filings.append(filing)
                    
                    logger.info(f"✅ Fetched {len(recent_filings.get('form', []))} SEC filings for {symbol}")
                    
            except Exception as e:
                logger.warning(f"⚠️ Failed to fetch SEC filings for {symbol}: {e}")
        
        return filings
    
    async def fetch_fda_approvals(self, days_back: int = 30) -> List[Dict[str, Any]]:
        """Fetch recent FDA drug approvals"""
        try:
            feed = feedparser.parse("https://www.fda.gov/about-fda/contact-fda/stay-informed/rss-feeds/fda-newsroom/rss.xml")
            
            approvals = []
            cutoff_date = datetime.now() - timedelta(days=days_back)
            
            for entry in feed.entries[:50]:
                # Check if related to drug approval
                title_lower = entry.title.lower()
                if any(keyword in title_lower for keyword in ["approval", "approved", "drug", "medication", "therapy"]):
                    
                    published = datetime(*entry.published_parsed[:6])
                    if published >= cutoff_date:
                        approval = {
                            "title": entry.title,
                            "summary": entry.summary if hasattr(entry, 'summary') else "",
                            "published": published.isoformat(),
                            "link": entry.link,
                            "source": "FDA",
                            "category": "drug_approval"
                        }
                        approvals.append(approval)
            
            logger.info(f"✅ Found {len(approvals)} FDA approvals in last {days_back} days")
            return approvals
            
        except Exception as e:
            logger.error(f"❌ Failed to fetch FDA approvals: {e}")
            return []
    
    async def fetch_patent_data(self, keywords: List[str]) -> List[Dict[str, Any]]:
        """Fetch patent information (USPTO)"""
        patents = []
        
        # Note: USPTO requires specific API access
        # This is a placeholder for patent monitoring
        logger.info("📋 Patent monitoring requires USPTO API access")
        logger.info(f"Keywords to monitor: {keywords}")
        
        # Alternative: Google Patents RSS
        for keyword in keywords[:5]:
            try:
                url = f"https://patents.google.com/xhr/query?q={keyword}&num=10"
                logger.info(f"🔍 Searching patents for: {keyword}")
                
                # This would need proper Google Patents API integration
                # For now, return placeholder
                
            except Exception as e:
                logger.warning(f"⚠️ Patent search failed for {keyword}: {e}")
        
        return patents
    
    async def fetch_geopolitical_events(self) -> List[Dict[str, Any]]:
        """Fetch geopolitical events affecting markets"""
        events = []
        
        for source_name, url in self.geopolitical_sources.items():
            try:
                articles = await self._fetch_rss_feed(url, source_name)
                
                # Filter for market-relevant geopolitical news
                market_keywords = [
                    "trade", "tariff", "sanctions", "embargo", "war", "conflict",
                    "election", "policy", "regulation", "treaty", "agreement",
                    "oil", "energy", "currency", "central bank", "interest rate"
                ]
                
                for article in articles:
                    text_lower = (article.get('title', '') + ' ' + article.get('summary', '')).lower()
                    if any(keyword in text_lower for keyword in market_keywords):
                        article['category'] = 'geopolitical'
                        article['market_relevant'] = True
                        events.append(article)
                
            except Exception as e:
                logger.warning(f"⚠️ Failed to fetch geopolitical news from {source_name}: {e}")
        
        logger.info(f"✅ Found {len(events)} market-relevant geopolitical events")
        return events[:30]
    
    async def fetch_earnings_calendar(self, symbols: List[str]) -> List[Dict[str, Any]]:
        """Fetch upcoming earnings announcements"""
        earnings = []
        
        for symbol in symbols:
            try:
                # Yahoo Finance earnings calendar
                url = f"https://finance.yahoo.com/calendar/earnings?symbol={symbol}"
                
                # This requires web scraping
                logger.info(f"📅 Checking earnings calendar for {symbol}")
                
                # Placeholder for earnings data
                earnings.append({
                    "symbol": symbol,
                    "source": "earnings_calendar",
                    "checked_at": datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.warning(f"⚠️ Failed to fetch earnings for {symbol}: {e}")
        
        return earnings
    
    async def _fetch_rss_feed(self, url: str, source_name: str) -> List[Dict[str, Any]]:
        """Fetch and parse RSS feed"""
        try:
            feed = feedparser.parse(url)
            articles = []
            
            for entry in feed.entries[:20]:
                article = {
                    "title": entry.title,
                    "summary": entry.summary if hasattr(entry, 'summary') else "",
                    "link": entry.link,
                    "published": entry.published if hasattr(entry, 'published') else "",
                    "source": source_name,
                    "fetched_at": datetime.now().isoformat()
                }
                articles.append(article)
            
            return articles
            
        except Exception as e:
            logger.error(f"❌ Failed to parse RSS feed from {source_name}: {e}")
            return []
    
    async def get_comprehensive_intelligence(self, symbols: List[str]) -> Dict[str, Any]:
        """Get comprehensive financial intelligence for symbols"""
        logger.info(f"🔍 Gathering comprehensive intelligence for {len(symbols)} symbols...")
        
        # Fetch all data sources in parallel
        results = await asyncio.gather(
            self.fetch_financial_news("all"),
            self.fetch_sec_filings(symbols),
            self.fetch_fda_approvals(30),
            self.fetch_geopolitical_events(),
            self.fetch_earnings_calendar(symbols),
            return_exceptions=True
        )
        
        intelligence = {
            "timestamp": datetime.now().isoformat(),
            "symbols": symbols,
            "data": {
                "financial_news": results[0] if not isinstance(results[0], Exception) else [],
                "sec_filings": results[1] if not isinstance(results[1], Exception) else [],
                "fda_approvals": results[2] if not isinstance(results[2], Exception) else [],
                "geopolitical_events": results[3] if not isinstance(results[3], Exception) else [],
                "earnings_calendar": results[4] if not isinstance(results[4], Exception) else []
            },
            "stats": {
                "total_news": len(results[0]) if not isinstance(results[0], Exception) else 0,
                "total_filings": len(results[1]) if not isinstance(results[1], Exception) else 0,
                "total_fda": len(results[2]) if not isinstance(results[2], Exception) else 0,
                "total_geopolitical": len(results[3]) if not isinstance(results[3], Exception) else 0,
                "total_earnings": len(results[4]) if not isinstance(results[4], Exception) else 0
            }
        }
        
        total_items = sum(intelligence["stats"].values())
        logger.info(f"✅ Gathered {total_items} intelligence items from {len(symbols)} sources")
        
        return intelligence
    
    async def close(self):
        """Close HTTP client"""
        await self.client.aclose()

# Test function
async def test_sources():
    """Test financial data sources"""
    sources = FinancialDataSources()
    
    print("🔍 Testing Financial Data Sources\n")
    
    # Test financial news
    print("1. Fetching financial news...")
    news = await sources.fetch_financial_news("all")
    print(f"   ✅ Got {len(news)} news articles")
    if news:
        print(f"   Latest: {news[0]['title'][:80]}...")
    
    # Test FDA approvals
    print("\n2. Fetching FDA approvals...")
    fda = await sources.fetch_fda_approvals(30)
    print(f"   ✅ Got {len(fda)} FDA approvals")
    
    # Test geopolitical
    print("\n3. Fetching geopolitical events...")
    geo = await sources.fetch_geopolitical_events()
    print(f"   ✅ Got {len(geo)} geopolitical events")
    
    # Test comprehensive
    print("\n4. Getting comprehensive intelligence...")
    intel = await sources.get_comprehensive_intelligence(["AAPL", "MSFT", "GOOGL"])
    print(f"   ✅ Total intelligence items: {sum(intel['stats'].values())}")
    print(f"   Stats: {intel['stats']}")
    
    await sources.close()
    print("\n✅ All tests completed!")

if __name__ == "__main__":
    asyncio.run(test_sources())
