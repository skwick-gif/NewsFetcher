"""
HTML scraper for websites that don't have RSS feeds
Uses Playwright for JavaScript-heavy sites and requests for simple HTML
"""
import asyncio
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import httpx
from bs4 import BeautifulSoup
try:
    from selectolax.parser import HTMLParser
    HAS_SELECTOLAX = True
except ImportError:
    HAS_SELECTOLAX = False
try:
    from playwright.async_api import async_playwright, Page, Browser
    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False
from urllib.parse import urljoin, urlparse
import re

logger = logging.getLogger(__name__)


class HTMLScraper:
    """HTML scraper with fallback from requests to Playwright"""
    
    def __init__(self, config: Dict[str, Any]):
        self.timeout = config.get("timeout", 30)
        self.user_agent = config.get("user_agent", "TariffRadar/1.0")
        self.max_retries = config.get("max_retries", 3)
        self.request_delay = config.get("request_delay", 2)
        self.use_playwright = config.get("use_playwright", False)
        
        self.scrapers = config.get("sources", {}).get("scrapers", [])
        
    async def scrape_with_httpx(self, source: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Scrape using httpx/requests (faster, no JS)"""
        url = source.get("url")
        selector = source.get("selector", "a")  # CSS selector for links
        name = source.get("name", url)
        
        logger.info(f"Scraping {name} with httpx")
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.get(
                    url,
                    headers={
                        "User-Agent": self.user_agent,
                        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
                    }
                )
                response.raise_for_status()
                
                # Parse with selectolax (faster than BeautifulSoup)
                tree = HTMLParser(response.text)
                articles = self._extract_articles(tree, source, url)
                
                logger.info(f"Scraped {len(articles)} articles from {name}")
                return articles
                
            except Exception as e:
                logger.error(f"Failed to scrape {name} with httpx: {e}")
                return []
    
    async def scrape_with_playwright(self, source: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Scrape using Playwright (handles JS, slower)"""
        url = source.get("url")
        name = source.get("name", url)
        
        logger.info(f"Scraping {name} with Playwright")
        
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                
                # Set user agent
                await page.set_extra_http_headers({
                    "User-Agent": self.user_agent
                })
                
                # Navigate and wait for content
                await page.goto(url, wait_until="networkidle", timeout=self.timeout * 1000)
                
                # Wait for dynamic content
                await asyncio.sleep(2)
                
                # Get HTML content
                html_content = await page.content()
                await browser.close()
                
                # Parse content
                tree = HTMLParser(html_content)
                articles = self._extract_articles(tree, source, url)
                
                logger.info(f"Scraped {len(articles)} articles from {name} with Playwright")
                return articles
                
        except Exception as e:
            logger.error(f"Failed to scrape {name} with Playwright: {e}")
            return []
    
    def _extract_articles(
        self, 
        tree: HTMLParser, 
        source: Dict[str, Any], 
        base_url: str
    ) -> List[Dict[str, Any]]:
        """Extract article links and metadata from parsed HTML"""
        selector = source.get("selector", "a")
        name = source.get("name", "unknown")
        priority = source.get("priority", "medium")
        
        articles = []
        
        try:
            # Find all matching elements
            elements = tree.css(selector)
            
            for element in elements:
                article = self._extract_article_from_element(element, source, base_url)
                if article:
                    articles.append(article)
                    
        except Exception as e:
            logger.error(f"Failed to extract articles from {name}: {e}")
            
        return articles
    
    def _extract_article_from_element(
        self, 
        element, 
        source: Dict[str, Any], 
        base_url: str
    ) -> Optional[Dict[str, Any]]:
        """Extract article data from HTML element"""
        try:
            # Get link
            link = element.attributes.get("href", "")
            if not link:
                return None
                
            # Make absolute URL
            full_url = urljoin(base_url, link)
            
            # Get title (try different methods)
            title = ""
            if element.attributes.get("title"):
                title = element.attributes.get("title")
            elif element.text():
                title = element.text().strip()
            elif element.css_first("img"):
                img = element.css_first("img")
                title = img.attributes.get("alt", "")
            
            if not title or len(title) < 10:
                return None
                
            # Try to extract date from URL or nearby elements
            published_at = self._extract_date_from_url(full_url)
            
            # Get any additional metadata
            metadata = self._extract_metadata(element)
            
            return {
                "title": title.strip(),
                "url": full_url,
                "published_at": published_at,
                "source_name": source.get("name", ""),
                "source_priority": source.get("priority", "medium"),
                "source_type": "scraper",
                "metadata": metadata
            }
            
        except Exception as e:
            logger.error(f"Failed to extract article from element: {e}")
            return None
    
    def _extract_date_from_url(self, url: str) -> Optional[datetime]:
        """Try to extract date from URL pattern"""
        try:
            # Common patterns: /2024/10/17/, /20241017/, etc.
            patterns = [
                r'/(\d{4})/(\d{1,2})/(\d{1,2})/',
                r'/(\d{4})(\d{2})(\d{2})/',
                r'(\d{4})-(\d{1,2})-(\d{1,2})',
                r'date[=_](\d{4})-(\d{1,2})-(\d{1,2})'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, url)
                if match:
                    year, month, day = map(int, match.groups())
                    return datetime(year, month, day, tzinfo=timezone.utc)
                    
        except Exception as e:
            logger.debug(f"Failed to extract date from URL {url}: {e}")
            
        return None
    
    def _extract_metadata(self, element) -> Dict[str, Any]:
        """Extract additional metadata from element"""
        metadata = {}
        
        try:
            # Look for date in nearby elements
            parent = element.parent
            if parent:
                # Common date selectors
                date_selectors = [".date", ".time", ".publish-time", "[data-date]"]
                for selector in date_selectors:
                    date_elem = parent.css_first(selector)
                    if date_elem and date_elem.text():
                        metadata["date_text"] = date_elem.text().strip()
                        break
            
            # Look for category/tags
            if element.attributes.get("data-category"):
                metadata["category"] = element.attributes.get("data-category")
                
        except Exception as e:
            logger.debug(f"Failed to extract metadata: {e}")
            
        return metadata
    
    async def scrape_article_content(self, url: str) -> Optional[str]:
        """Scrape full content of an individual article"""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    url,
                    headers={"User-Agent": self.user_agent}
                )
                response.raise_for_status()
                
                # Parse and extract main content
                tree = HTMLParser(response.text)
                content = self._extract_main_content(tree)
                
                return content
                
        except Exception as e:
            logger.error(f"Failed to scrape content from {url}: {e}")
            return None
    
    def _extract_main_content(self, tree: HTMLParser) -> str:
        """Extract main article content from HTML"""
        # Common content selectors (prioritized)
        content_selectors = [
            "article",
            ".article-content",
            ".post-content", 
            ".content",
            "#content",
            ".main-content",
            "main",
            ".entry-content",
            "[role='main']"
        ]
        
        for selector in content_selectors:
            content_elem = tree.css_first(selector)
            if content_elem:
                # Clean up content
                text = content_elem.text(separator=" ", strip=True)
                if len(text) > 200:  # Minimum content length
                    return text
                    
        # Fallback: get all p tags
        p_tags = tree.css("p")
        if p_tags:
            content = " ".join([p.text(strip=True) for p in p_tags if p.text(strip=True)])
            if len(content) > 200:
                return content
                
        return ""
    
    async def scrape_all_sources(self) -> List[Dict[str, Any]]:
        """Scrape all configured sources"""
        all_articles = []
        
        for source in self.scrapers:
            try:
                # Try httpx first, fallback to Playwright if needed
                articles = await self.scrape_with_httpx(source)
                
                if not articles and self.use_playwright:
                    logger.info(f"Falling back to Playwright for {source.get('name')}")
                    articles = await self.scrape_with_playwright(source)
                
                all_articles.extend(articles)
                
                # Rate limiting
                if self.request_delay > 0:
                    await asyncio.sleep(self.request_delay)
                    
            except Exception as e:
                logger.error(f"Failed to scrape source {source.get('name', 'unknown')}: {e}")
                continue
        
        logger.info(f"Total articles scraped: {len(all_articles)}")
        return all_articles


async def main():
    """Test the HTML scraper"""
    config = {
        "sources": {
            "scrapers": [
                {
                    "url": "http://gss.mof.gov.cn/zhengwuxinxi/zhengcefabu/",
                    "name": "MOF Policy",
                    "selector": ".news-list a",
                    "priority": "high"
                },
                {
                    "url": "http://www.gov.cn/zhengce/",
                    "name": "State Council",
                    "selector": ".news_list li a",
                    "priority": "high"
                }
            ]
        },
        "timeout": 30,
        "request_delay": 2
    }
    
    scraper = HTMLScraper(config)
    articles = await scraper.scrape_all_sources()
    
    print(f"Scraped {len(articles)} articles")
    for i, article in enumerate(articles[:5]):
        print(f"{i+1}. {article['title'][:60]}...")
        print(f"   URL: {article['url']}")
        print()


if __name__ == "__main__":
    asyncio.run(main())