"""
Chinese Government Websites Scraper
Specialized scraper for MOFCOM, GACC, MOF and other Chinese government sites
"""
import asyncio
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import httpx
from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin

logger = logging.getLogger(__name__)


class ChineseGovScraper:
    """Scraper specialized for Chinese government websites"""
    
    def __init__(self):
        self.timeout = 30
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Accept-Encoding": "gzip, deflate",
        }
        
    async def scrape_mofcom_trade_policy(self) -> List[Dict[str, Any]]:
        """
        Scrape MOFCOM Trade Policy
        Ministry of Commerce - International Trade (商务部 - 国际经贸关系)
        This is THE FIRST source for official Chinese tariff announcements!
        """
        url = "http://www.mofcom.gov.cn/article/ae/"
        articles = []
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url, headers=self.headers)
                response.encoding = 'utf-8'
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Find article list (MOFCOM uses <ul class="artitle_list">)
                article_list = soup.find('ul', class_='artitle_list')
                if not article_list:
                    logger.warning("Could not find article list on MOFCOM")
                    return []
                
                items = article_list.find_all('li')
                
                for item in items[:20]:  # Get latest 20
                    try:
                        # Extract link and title
                        link_tag = item.find('a')
                        if not link_tag:
                            continue
                        
                        title = link_tag.get_text(strip=True)
                        href = link_tag.get('href', '')
                        
                        # Make absolute URL
                        if href:
                            article_url = urljoin(url, href)
                        else:
                            continue
                        
                        # Extract date (MOFCOM format: YYYY-MM-DD)
                        date_match = re.search(r'(\d{4})-(\d{2})-(\d{2})', item.get_text())
                        published_at = None
                        if date_match:
                            try:
                                published_at = datetime(
                                    int(date_match.group(1)),
                                    int(date_match.group(2)),
                                    int(date_match.group(3)),
                                    tzinfo=timezone.utc
                                )
                            except:
                                pass
                        
                        articles.append({
                            'title': title,
                            'url': article_url,
                            'published_at': published_at or datetime.now(timezone.utc),
                            'source': 'MOFCOM Trade Policy',
                            'language': 'zh',
                            'priority': 'high',
                            'content_snippet': '',
                        })
                        
                    except Exception as e:
                        logger.error(f"Error parsing MOFCOM item: {e}")
                        continue
                
                logger.info(f"Scraped {len(articles)} articles from MOFCOM Trade Policy")
                return articles
                
        except Exception as e:
            logger.error(f"Failed to scrape MOFCOM: {e}")
            return []
    
    async def scrape_mofcom_news(self) -> List[Dict[str, Any]]:
        """
        Scrape MOFCOM General News
        商务部 - 商务新闻
        """
        url = "http://www.mofcom.gov.cn/article/b/c/"
        articles = []
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url, headers=self.headers)
                response.encoding = 'utf-8'
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # MOFCOM news uses different structure
                items = soup.select('.content_list li')
                
                for item in items[:15]:
                    try:
                        link_tag = item.find('a')
                        if not link_tag:
                            continue
                        
                        title = link_tag.get_text(strip=True)
                        href = link_tag.get('href', '')
                        article_url = urljoin(url, href) if href else None
                        
                        if not article_url:
                            continue
                        
                        # Extract date
                        date_match = re.search(r'(\d{4})-(\d{2})-(\d{2})', item.get_text())
                        published_at = None
                        if date_match:
                            try:
                                published_at = datetime(
                                    int(date_match.group(1)),
                                    int(date_match.group(2)),
                                    int(date_match.group(3)),
                                    tzinfo=timezone.utc
                                )
                            except:
                                pass
                        
                        articles.append({
                            'title': title,
                            'url': article_url,
                            'published_at': published_at or datetime.now(timezone.utc),
                            'source': 'MOFCOM News',
                            'language': 'zh',
                            'priority': 'high',
                            'content_snippet': '',
                        })
                        
                    except Exception as e:
                        logger.error(f"Error parsing MOFCOM news item: {e}")
                        continue
                
                logger.info(f"Scraped {len(articles)} articles from MOFCOM News")
                return articles
                
        except Exception as e:
            logger.error(f"Failed to scrape MOFCOM News: {e}")
            return []
    
    async def scrape_gacc_news(self) -> List[Dict[str, Any]]:
        """
        Scrape GACC (Customs) News
        海关总署 - 新闻发布
        Important for tariff implementation details
        """
        url = "http://www.customs.gov.cn/customs/xwfb34/index.html"
        articles = []
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url, headers=self.headers)
                response.encoding = 'utf-8'
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # GACC uses <div class="list_con">
                items = soup.select('.list_con li')
                
                for item in items[:15]:
                    try:
                        link_tag = item.find('a')
                        if not link_tag:
                            continue
                        
                        title = link_tag.get_text(strip=True)
                        href = link_tag.get('href', '')
                        article_url = urljoin(url, href) if href else None
                        
                        if not article_url:
                            continue
                        
                        # Extract date
                        date_text = item.find('span', class_='time')
                        published_at = None
                        if date_text:
                            date_match = re.search(r'(\d{4})-(\d{2})-(\d{2})', date_text.get_text())
                            if date_match:
                                try:
                                    published_at = datetime(
                                        int(date_match.group(1)),
                                        int(date_match.group(2)),
                                        int(date_match.group(3)),
                                        tzinfo=timezone.utc
                                    )
                                except:
                                    pass
                        
                        articles.append({
                            'title': title,
                            'url': article_url,
                            'published_at': published_at or datetime.now(timezone.utc),
                            'source': 'GACC News',
                            'language': 'zh',
                            'priority': 'high',
                            'content_snippet': '',
                        })
                        
                    except Exception as e:
                        logger.error(f"Error parsing GACC item: {e}")
                        continue
                
                logger.info(f"Scraped {len(articles)} articles from GACC")
                return articles
                
        except Exception as e:
            logger.error(f"Failed to scrape GACC: {e}")
            return []
    
    async def scrape_mof_policies(self) -> List[Dict[str, Any]]:
        """
        Scrape MOF (Ministry of Finance) Tax Policies
        财政部 - 政策发布
        Important for tariff-related tax policies
        """
        url = "http://gss.mof.gov.cn/zhengwuxinxi/zhengcefabu/"
        articles = []
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url, headers=self.headers)
                response.encoding = 'utf-8'
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # MOF uses <ul class="news_list">
                items = soup.select('.news_list li, .news-list li')
                
                for item in items[:10]:
                    try:
                        link_tag = item.find('a')
                        if not link_tag:
                            continue
                        
                        title = link_tag.get_text(strip=True)
                        href = link_tag.get('href', '')
                        article_url = urljoin(url, href) if href else None
                        
                        if not article_url:
                            continue
                        
                        # Extract date
                        date_match = re.search(r'(\d{4})-(\d{2})-(\d{2})', item.get_text())
                        published_at = None
                        if date_match:
                            try:
                                published_at = datetime(
                                    int(date_match.group(1)),
                                    int(date_match.group(2)),
                                    int(date_match.group(3)),
                                    tzinfo=timezone.utc
                                )
                            except:
                                pass
                        
                        articles.append({
                            'title': title,
                            'url': article_url,
                            'published_at': published_at or datetime.now(timezone.utc),
                            'source': 'MOF Tax Policy',
                            'language': 'zh',
                            'priority': 'medium',
                            'content_snippet': '',
                        })
                        
                    except Exception as e:
                        logger.error(f"Error parsing MOF item: {e}")
                        continue
                
                logger.info(f"Scraped {len(articles)} articles from MOF")
                return articles
                
        except Exception as e:
            logger.error(f"Failed to scrape MOF: {e}")
            return []
    
    async def scrape_statecouncil(self) -> List[Dict[str, Any]]:
        """
        Scrape State Council Policies
        国务院 - 政策
        Highest level government announcements
        """
        url = "http://www.gov.cn/zhengce/"
        articles = []
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url, headers=self.headers)
                response.encoding = 'utf-8'
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # State Council uses various structures
                items = soup.select('.news_box li, .listBox li')
                
                for item in items[:10]:
                    try:
                        link_tag = item.find('a')
                        if not link_tag:
                            continue
                        
                        title = link_tag.get_text(strip=True)
                        href = link_tag.get('href', '')
                        article_url = urljoin(url, href) if href else None
                        
                        if not article_url:
                            continue
                        
                        # Extract date
                        date_match = re.search(r'(\d{4})[/-](\d{2})[/-](\d{2})', item.get_text())
                        published_at = None
                        if date_match:
                            try:
                                published_at = datetime(
                                    int(date_match.group(1)),
                                    int(date_match.group(2)),
                                    int(date_match.group(3)),
                                    tzinfo=timezone.utc
                                )
                            except:
                                pass
                        
                        articles.append({
                            'title': title,
                            'url': article_url,
                            'published_at': published_at or datetime.now(timezone.utc),
                            'source': 'State Council',
                            'language': 'zh',
                            'priority': 'high',
                            'content_snippet': '',
                        })
                        
                    except Exception as e:
                        logger.error(f"Error parsing State Council item: {e}")
                        continue
                
                logger.info(f"Scraped {len(articles)} articles from State Council")
                return articles
                
        except Exception as e:
            logger.error(f"Failed to scrape State Council: {e}")
            return []
    
    async def scrape_all(self) -> List[Dict[str, Any]]:
        """Scrape all Chinese government sources"""
        logger.info("Starting Chinese government sources scraping...")
        
        # Run all scrapers in parallel
        results = await asyncio.gather(
            self.scrape_mofcom_trade_policy(),
            self.scrape_mofcom_news(),
            self.scrape_gacc_news(),
            self.scrape_mof_policies(),
            self.scrape_statecouncil(),
            return_exceptions=True
        )
        
        # Combine all articles
        all_articles = []
        for result in results:
            if isinstance(result, list):
                all_articles.extend(result)
            else:
                logger.error(f"Scraper failed with exception: {result}")
        
        logger.info(f"Total articles scraped from Chinese gov sources: {len(all_articles)}")
        return all_articles


# Async helper function for Celery tasks
async def scrape_chinese_sources() -> List[Dict[str, Any]]:
    """Helper function to be called from Celery tasks"""
    scraper = ChineseGovScraper()
    return await scraper.scrape_all()
