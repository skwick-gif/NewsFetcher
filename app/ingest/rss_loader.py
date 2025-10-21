"""
Financial RSS/API Data Loader for MarketPulse
Handles:
- Financial news RSS feeds (Reuters, Bloomberg, WSJ, etc.)
- Regulatory filings (SEC, FDA)
- Social media mentions
- Perplexity AI scheduled searches
"""
import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional
import httpx
import feedparser
from urllib.parse import urljoin
import yaml
import hashlib
import re

logger = logging.getLogger(__name__)


class FinancialDataLoader:
    """Load financial data from multiple sources"""
    
    def __init__(self, config_path: str = "app/config/data_sources.yaml"):
        """Initialize with configuration"""
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        
        self.timeout = 30
        self.user_agent = "MarketPulse/1.0 (Financial Intelligence Platform)"
        self.max_retries = 3
        
        # Track processed articles to avoid duplicates
        self.seen_articles = set()
        
    def _generate_article_hash(self, title: str, url: str) -> str:
        """Generate unique hash for article deduplication"""
        content = f"{title}{url}".lower().strip()
        return hashlib.md5(content.encode()).hexdigest()
    
    def _is_duplicate(self, title: str, url: str) -> bool:
        """Check if article was already processed"""
        article_hash = self._generate_article_hash(title, url)
        if article_hash in self.seen_articles:
            return True
        self.seen_articles.add(article_hash)
        return False
    
    def _extract_stock_symbols(self, text: str) -> List[str]:
        """Extract stock symbols from text using regex"""
        # Match $SYMBOL or (NASDAQ: SYMBOL) or (NYSE: SYMBOL)
        patterns = [
            r'\$([A-Z]{1,5})',  # $AAPL
            r'\((?:NASDAQ|NYSE|AMEX):\s*([A-Z]{1,5})\)',  # (NASDAQ: AAPL)
            r'\b([A-Z]{2,5})\s+stock\b',  # AAPL stock
        ]
        
        symbols = set()
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            symbols.update([s.upper() for s in matches])
        
        return list(symbols)
    
    async def fetch_rss_feed(self, feed_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Fetch and parse RSS feed"""
        url = feed_config.get("url")
        name = feed_config.get("name", url)
        weight = feed_config.get("weight", 1.0)
        category = feed_config.get("category", "general")
        
        logger.info(f"Fetching RSS feed: {name}")
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for attempt in range(self.max_retries):
                try:
                    response = await client.get(
                        url,
                        headers={"User-Agent": self.user_agent},
                        follow_redirects=True
                    )
                    response.raise_for_status()
                    
                    # Parse RSS feed
                    feed = feedparser.parse(response.text)
                    
                    if feed.bozo:
                        logger.warning(f"Feed parsing warning for {name}: {feed.bozo_exception}")
                    
                    articles = []
                    for entry in feed.entries:
                        article = self._parse_rss_entry(entry, feed_config)
                        if article and not self._is_duplicate(article["title"], article["url"]):
                            articles.append(article)
                    
                    logger.info(f"✅ Fetched {len(articles)} new articles from {name}")
                    return articles
                    
                except httpx.TimeoutException:
                    logger.warning(f"Timeout for {name} (attempt {attempt + 1}/{self.max_retries})")
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(2 ** attempt)
                    else:
                        logger.error(f"❌ All attempts timed out for {name}")
                        return []
                        
                except Exception as e:
                    logger.error(f"Attempt {attempt + 1} failed for {name}: {e}")
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(2 ** attempt)
                    else:
                        logger.error(f"❌ All attempts failed for {name}")
                        return []
        
        return []
    
    def _parse_rss_entry(self, entry, feed_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse individual RSS entry"""
        try:
            # Extract basic info
            title = entry.get("title", "").strip()
            link = entry.get("link", "").strip()
            
            if not title or not link:
                return None
            
            # Validation from config
            validation = self.config.get("validation", {})
            if len(title) < validation.get("min_title_length", 20):
                return None
            
            # Extract content
            content = ""
            if hasattr(entry, "content") and entry.content:
                content = entry.content[0].value if entry.content else ""
            elif hasattr(entry, "description"):
                content = entry.description
            elif hasattr(entry, "summary"):
                content = entry.summary
            
            # Remove HTML tags
            content = re.sub(r'<[^>]+>', '', content)
            
            if len(content) < validation.get("min_content_length", 100):
                logger.debug(f"Skipping article (content too short): {title[:50]}")
                return None
            
            # Parse publication date
            published_at = None
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                published_at = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
            elif hasattr(entry, "updated_parsed") and entry.updated_parsed:
                published_at = datetime(*entry.updated_parsed[:6], tzinfo=timezone.utc)
            else:
                published_at = datetime.now(timezone.utc)
            
            # Check age limit
            max_age_hours = validation.get("max_age_hours", 48)
            age = datetime.now(timezone.utc) - published_at
            if age > timedelta(hours=max_age_hours):
                logger.debug(f"Skipping old article ({age.total_seconds() / 3600:.1f}h old): {title[:50]}")
                return None
            
            # Extract stock symbols
            combined_text = f"{title} {content}"
            symbols = self._extract_stock_symbols(combined_text)
            
            # Extract author
            author = getattr(entry, "author", "")
            
            return {
                "title": title,
                "content": content,
                "url": link,
                "published_at": published_at.isoformat(),
                "author": author,
                "source_name": feed_config.get("name", ""),
                "source_category": feed_config.get("category", "general"),
                "source_weight": feed_config.get("weight", 1.0),
                "source_type": "rss",
                "symbols": symbols,
                "sectors": feed_config.get("sectors", []),
                "regions": feed_config.get("regions", []),
                "keywords_boost": feed_config.get("keywords_boost", 1.0),
                "fetched_at": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to parse RSS entry: {e}")
            return None
    
    async def fetch_all_rss_feeds(self, tier: Optional[str] = None) -> List[Dict[str, Any]]:
        """Fetch all RSS feeds (optionally filtered by tier)"""
        all_articles = []
        rss_feeds = self.config.get("rss_feeds", {})
        
        tiers_to_fetch = [tier] if tier else list(rss_feeds.keys())
        
        for tier_name in tiers_to_fetch:
            feeds = rss_feeds.get(tier_name, [])
            
            for feed_config in feeds:
                try:
                    articles = await self.fetch_rss_feed(feed_config)
                    all_articles.extend(articles)
                    
                    # Rate limiting
                    await asyncio.sleep(0.5)
                    
                except Exception as e:
                    logger.error(f"Failed to fetch feed {feed_config.get('name', 'unknown')}: {e}")
                    continue
        
        logger.info(f"📰 Total RSS articles fetched: {len(all_articles)}")
        return all_articles
    
    async def fetch_sec_filings(self) -> List[Dict[str, Any]]:
        """Fetch recent SEC filings"""
        sec_config = self.config.get("regulatory_sources", {}).get("sec_edgar", [])
        
        if not sec_config:
            return []
        
        articles = []
        for source in sec_config:
            try:
                url = source.get("url")
                logger.info(f"Fetching SEC filings from {url}")
                
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.get(
                        url,
                        headers={"User-Agent": self.user_agent}
                    )
                    response.raise_for_status()
                    
                    # Parse ATOM feed
                    feed = feedparser.parse(response.text)
                    
                    for entry in feed.entries:
                        # Extract filing type and company
                        title = entry.get("title", "")
                        
                        # Match filing types we care about
                        filing_types = source.get("filing_types", [])
                        matched_type = None
                        for ftype in filing_types:
                            if ftype in title:
                                matched_type = ftype
                                break
                        
                        if not matched_type:
                            continue
                        
                        # Extract CIK and symbol
                        symbols = self._extract_stock_symbols(title)
                        
                        article = {
                            "title": title,
                            "content": entry.get("summary", ""),
                            "url": entry.get("link", ""),
                            "published_at": datetime.now(timezone.utc).isoformat(),
                            "source_name": "SEC EDGAR",
                            "source_category": "regulatory",
                            "source_weight": source.get("weight", 1.2),
                            "source_type": "sec_filing",
                            "filing_type": matched_type,
                            "symbols": symbols,
                            "sectors": [],
                            "fetched_at": datetime.now(timezone.utc).isoformat()
                        }
                        
                        if not self._is_duplicate(article["title"], article["url"]):
                            articles.append(article)
                
                logger.info(f"📋 Fetched {len(articles)} SEC filings")
                
            except Exception as e:
                logger.error(f"Failed to fetch SEC filings: {e}")
        
        return articles
    
    async def fetch_fda_approvals(self) -> List[Dict[str, Any]]:
        """Fetch FDA drug/device approvals and safety alerts"""
        fda_sources = self.config.get("regulatory_sources", {}).get("fda", [])
        
        all_articles = []
        for source in fda_sources:
            try:
                url = source.get("url")
                name = source.get("name", "FDA")
                
                logger.info(f"Fetching {name}")
                
                feed_config = {
                    "url": url,
                    "name": name,
                    "weight": source.get("weight", 1.3),
                    "category": "regulatory",
                    "sectors": source.get("sectors", ["pharma", "biotech"])
                }
                
                articles = await self.fetch_rss_feed(feed_config)
                all_articles.extend(articles)
                
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Failed to fetch FDA data: {e}")
        
        logger.info(f"💊 Fetched {len(all_articles)} FDA updates")
        return all_articles
    
    async def fetch_perplexity_market_scan(self) -> List[Dict[str, Any]]:
        """
        Execute scheduled Perplexity searches to DISCOVER new events
        This is different from analysis - this is proactive scanning
        """
        from app.financial.perplexity_analyzer import PerplexityAnalyzer
        
        scan_queries = self.config.get("perplexity_scheduled_searches", {}).get("market_scanning", [])
        
        analyzer = PerplexityAnalyzer()
        scan_results = []
        
        for query_config in scan_queries:
            try:
                query = query_config.get("query_template")
                purpose = query_config.get("purpose", "scan")
                priority = query_config.get("priority", "medium")
                
                logger.info(f"🔍 Perplexity scan: {purpose}")
                
                # Execute search
                result = await analyzer.analyze_async(
                    symbol="MARKET",  # General market scan
                    user_query=query,
                    analysis_type=purpose
                )
                
                if result and result.get("analysis"):
                    scan_results.append({
                        "title": f"Perplexity Market Scan: {purpose}",
                        "content": result.get("analysis", ""),
                        "url": f"perplexity://scan/{purpose}",
                        "published_at": datetime.now(timezone.utc).isoformat(),
                        "source_name": "Perplexity AI",
                        "source_category": "ai_scan",
                        "source_weight": 1.0,
                        "source_type": "perplexity_scan",
                        "purpose": purpose,
                        "priority": priority,
                        "symbols": self._extract_stock_symbols(result.get("analysis", "")),
                        "sectors": query_config.get("sectors", []),
                        "citations": result.get("citations", []),
                        "fetched_at": datetime.now(timezone.utc).isoformat()
                    })
                
                # Rate limit: Perplexity costs money!
                await asyncio.sleep(2)
                
            except Exception as e:
                logger.error(f"Failed Perplexity scan {query_config.get('purpose')}: {e}")
        
        logger.info(f"🤖 Completed {len(scan_results)} Perplexity scans")
        return scan_results
    
    async def fetch_all_data(self, include_perplexity: bool = True) -> Dict[str, List[Dict[str, Any]]]:
        """Fetch all data sources"""
        logger.info("=" * 80)
        logger.info("🚀 Starting comprehensive data fetch...")
        logger.info("=" * 80)
        
        results = {
            "rss_articles": [],
            "sec_filings": [],
            "fda_updates": [],
            "perplexity_scans": []
        }
        
        # Fetch RSS feeds (major news)
        logger.info("\n📰 Fetching RSS feeds...")
        results["rss_articles"] = await self.fetch_all_rss_feeds()
        
        # Fetch SEC filings
        logger.info("\n📋 Fetching SEC filings...")
        results["sec_filings"] = await self.fetch_sec_filings()
        
        # Fetch FDA updates
        logger.info("\n💊 Fetching FDA updates...")
        results["fda_updates"] = await self.fetch_fda_approvals()
        
        # Fetch Perplexity scans
        if include_perplexity:
            logger.info("\n🤖 Running Perplexity market scans...")
            try:
                results["perplexity_scans"] = await self.fetch_perplexity_market_scan()
            except Exception as e:
                logger.error(f"Perplexity scans failed: {e}")
                results["perplexity_scans"] = []
        
        # Summary
        total = sum(len(v) for v in results.values())
        logger.info("\n" + "=" * 80)
        logger.info(f"✅ Fetch complete! Total items: {total}")
        logger.info(f"   - RSS articles: {len(results['rss_articles'])}")
        logger.info(f"   - SEC filings: {len(results['sec_filings'])}")
        logger.info(f"   - FDA updates: {len(results['fda_updates'])}")
        logger.info(f"   - Perplexity scans: {len(results['perplexity_scans'])}")
        logger.info("=" * 80)
        
        return results
    
    def get_schedule_config(self) -> Dict[str, int]:
        """Get scheduler intervals from config"""
        return self.config.get("scheduler", {})


async def main():
    """Test the financial data loader"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    loader = FinancialDataLoader()
    
    # Test RSS feeds only (fast test)
    print("\n🧪 Testing RSS feeds (major news only)...")
    articles = await loader.fetch_all_rss_feeds(tier="major_news")
    
    if articles:
        print(f"\n✅ Sample articles ({len(articles)} total):")
        for i, article in enumerate(articles[:5]):
            print(f"\n{i+1}. {article['title'][:100]}")
            print(f"   Source: {article['source_name']} (weight: {article['source_weight']})")
            print(f"   Symbols: {', '.join(article['symbols']) if article['symbols'] else 'None'}")
            print(f"   URL: {article['url'][:80]}")
    
    # Full test (uncomment to test everything including Perplexity)
    # print("\n🧪 Testing full data fetch (RSS + SEC + FDA + Perplexity)...")
    # results = await loader.fetch_all_data(include_perplexity=False)  # Set True to test Perplexity


if __name__ == "__main__":
    asyncio.run(main())
