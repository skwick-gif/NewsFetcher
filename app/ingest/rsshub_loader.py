"""
RSSHub data loader for fetching RSS feeds from various sources
Handles WeChat Official Accounts, government websites, and news sources
"""
import asyncio
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import httpx
import feedparser
from urllib.parse import urljoin
import yaml

logger = logging.getLogger(__name__)


class RSSHubLoader:
    """RSSHub client for loading RSS feeds"""
    
    def __init__(self, config: Dict[str, Any]):
        self.base_url = config.get("rsshub_base_url", "https://rsshub.app")
        self.timeout = config.get("timeout", 30)
        self.user_agent = config.get("user_agent", "TariffRadar/1.0")
        self.max_retries = config.get("max_retries", 3)
        self.request_delay = config.get("request_delay", 1)
        
        self.sources = config.get("sources", {}).get("rss", [])
        
    async def fetch_feed(self, source: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Fetch and parse RSS feed from a single source"""
        # Support both 'route' (old) and 'url' (new) formats
        url = source.get("url")
        if not url:
            route = source.get("route")
            if not route:
                logger.error(f"No URL or route specified for source: {source}")
                return []
            url = urljoin(self.base_url, route)
        
        name = source.get("name", url)
        priority = source.get("priority", "medium")
        
        logger.info(f"Fetching feed: {name} from {url}")
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for attempt in range(self.max_retries):
                try:
                    response = await client.get(
                        url,
                        headers={"User-Agent": self.user_agent}
                    )
                    response.raise_for_status()
                    
                    # Parse RSS feed
                    feed = feedparser.parse(response.text)
                    
                    if feed.bozo:
                        logger.warning(f"Feed parsing warning for {name}: {feed.bozo_exception}")
                    
                    articles = []
                    for entry in feed.entries:
                        article = self._parse_entry(entry, source)
                        if article:
                            articles.append(article)
                    
                    logger.info(f"Fetched {len(articles)} articles from {name}")
                    return articles
                    
                except Exception as e:
                    logger.error(f"Attempt {attempt + 1} failed for {name}: {e}")
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    else:
                        logger.error(f"All attempts failed for {name}")
                        return []
                        
        return []
    
    def _parse_entry(self, entry, source: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse individual RSS entry"""
        try:
            # Extract basic info
            title = entry.get("title", "").strip()
            link = entry.get("link", "").strip()
            
            if not title or not link:
                return None
            
            # Extract content
            content = ""
            if hasattr(entry, "content") and entry.content:
                content = entry.content[0].value if entry.content else ""
            elif hasattr(entry, "description"):
                content = entry.description
            elif hasattr(entry, "summary"):
                content = entry.summary
            
            # Parse publication date
            published_at = None
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                published_at = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
            elif hasattr(entry, "updated_parsed") and entry.updated_parsed:
                published_at = datetime(*entry.updated_parsed[:6], tzinfo=timezone.utc)
            
            # Extract author/source info
            author = getattr(entry, "author", "")
            
            return {
                "title": title,
                "content": content,
                "url": link,
                "published_at": published_at,
                "author": author,
                "source_name": source.get("name", ""),
                "source_priority": source.get("priority", "medium"),
                "source_type": "rss",
                "raw_entry": entry  # Keep for debugging
            }
            
        except Exception as e:
            logger.error(f"Failed to parse entry: {e}")
            return None
    
    async def fetch_direct_rss(self, url: str, name: str, priority: str = "medium") -> List[Dict[str, Any]]:
        """Fetch RSS feed directly (not via RSSHub) - more reliable!"""
        logger.info(f"Fetching direct RSS: {name} from {url}")
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
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
                    article = self._parse_entry(entry, {"name": name, "priority": priority})
                    if article:
                        articles.append(article)
                
                logger.info(f"Fetched {len(articles)} articles from {name}")
                return articles
                
            except Exception as e:
                logger.error(f"Failed to fetch direct RSS {name}: {e}")
                return []
    
    def load_feed(self, url: str) -> List[Dict[str, Any]]:
        """Legacy method for backward compatibility - determines method based on URL"""
        if "rsshub.app" in url:
            # Use RSSHub
            source_config = {"url": url, "name": url, "priority": "medium"}
            return asyncio.run(self.fetch_feed(source_config))
        else:
            # Use direct RSS
            return asyncio.run(self.fetch_direct_rss(url, url, "medium"))
    
    async def fetch_all_feeds(self) -> List[Dict[str, Any]]:
        """Fetch all configured RSS feeds"""
        all_articles = []
        
        for source in self.sources:
            if source.get("type") == "rsshub":
                try:
                    articles = await self.fetch_feed(source)
                    all_articles.extend(articles)
                    
                    # Rate limiting
                    if self.request_delay > 0:
                        await asyncio.sleep(self.request_delay)
                        
                except Exception as e:
                    logger.error(f"Failed to fetch source {source.get('name', 'unknown')}: {e}")
                    continue
            elif source.get("type") == "direct_rss":
                # NEW: Support for direct RSS feeds (bypasses RSSHub)
                try:
                    articles = await self.fetch_direct_rss(
                        url=source.get("url"),
                        name=source.get("name", "Unknown"),
                        priority=source.get("priority", "medium")
                    )
                    all_articles.extend(articles)
                    
                    if self.request_delay > 0:
                        await asyncio.sleep(self.request_delay)
                        
                except Exception as e:
                    logger.error(f"Failed to fetch direct RSS {source.get('name', 'unknown')}: {e}")
                    continue
        
        logger.info(f"Total articles fetched: {len(all_articles)}")
        return all_articles
    
    def get_popular_routes(self) -> Dict[str, str]:
        """Get popular RSSHub routes for US-China trade monitoring"""
        return {
            # WeChat Official Accounts
            "mofcom_weixin": "/weixin/account/mofcom_gov",
            "gacc_weixin": "/weixin/account/customs_china", 
            "ccpit_weixin": "/weixin/account/chinachamber",
            
            # Government websites
            "mofcom_news": "/mofcom/news",
            "mofcom_announcements": "/mofcom/announcements", 
            "gacc_news": "/gacc/news",
            "gacc_announcements": "/gacc/announcements",
            
            # State Council
            "gov_cn_policies": "/gov/beijing/zhengce",
            "gov_cn_news": "/gov/beijing/news",
            
            # Think tanks & research
            "csis_china": "/csis/briefs/china",
            "piie_trade": "/piie/blogs/trade-and-investment-policy-watch",
            
            # News media
            "reuters_china_trade": "/reuters/world/china/trade",
            "ft_china_trade": "/ft/china/trade",
            "scmp_china_economy": "/scmp/china/economy",
            
            # Trade associations
            "uschamber_china": "/uschamber/china",
            "amcham_china": "/amcham/china/policy"
        }
    
    async def test_source(self, route: str) -> bool:
        """Test if a specific RSSHub route works"""
        url = urljoin(self.base_url, route)
        
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(url, headers={"User-Agent": self.user_agent})
                response.raise_for_status()
                
                feed = feedparser.parse(response.text)
                return len(feed.entries) > 0
                
        except Exception as e:
            logger.error(f"Failed to test route {route}: {e}")
            return False


def load_config() -> Dict[str, Any]:
    """Load configuration from config.yaml"""
    try:
        with open("config.yaml", "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return {}


async def main():
    """Test the RSSHub loader"""
    config = load_config()
    loader = RSSHubLoader(config)
    
    # Test a few routes
    test_routes = [
        "/mofcom/news",
        "/gacc/news", 
        "/gov/beijing/zhengce"
    ]
    
    for route in test_routes:
        print(f"Testing {route}...")
        works = await loader.test_source(route)
        print(f"  {'✓' if works else '✗'} {route}")
    
    # Fetch all feeds
    print("\nFetching all configured feeds...")
    articles = await loader.fetch_all_feeds()
    print(f"Total articles: {len(articles)}")
    
    # Show sample
    if articles:
        print("\nSample articles:")
        for i, article in enumerate(articles[:3]):
            print(f"{i+1}. {article['title'][:80]}...")
            print(f"   Source: {article['source_name']}")
            print(f"   URL: {article['url']}")
            print()


if __name__ == "__main__":
    asyncio.run(main())