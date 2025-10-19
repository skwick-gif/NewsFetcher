"""
Twitter/X Scraper for monitoring Trump and other key accounts
This is often THE FIRST source for tariff announcements!
"""
import asyncio
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import httpx
from bs4 import BeautifulSoup
import re
import os

logger = logging.getLogger(__name__)


class TwitterScraper:
    """
    Twitter/X scraper for monitoring key accounts
    
    Important accounts for tariff monitoring:
    - @realDonaldTrump - Often announces tariffs here FIRST
    - @POTUS - Official presidential account
    - @WhiteHouse - White House official
    - @USTreasury - Treasury Department
    - @CommerceGov - Commerce Department
    """
    
    def __init__(self):
        self.timeout = 30
        
        # Twitter API v2 credentials (Free tier: 500,000 tweets/month)
        self.bearer_token = os.getenv("TWITTER_BEARER_TOKEN", "")
        
        # Nitter instances (Twitter frontend without API - backup method)
        self.nitter_instances = [
            "https://nitter.net",
            "https://nitter.it",
            "https://nitter.privacydev.net"
        ]
        
        # RSSHub Twitter routes (another backup)
        self.rsshub_url = os.getenv("RSSHUB_URL", "http://rsshub:1200")
        
    async def scrape_via_twitter_api(self, username: str) -> List[Dict[str, Any]]:
        """
        Scrape using official Twitter API v2
        
        Requires: TWITTER_BEARER_TOKEN environment variable
        Get free token at: https://developer.twitter.com
        """
        if not self.bearer_token:
            logger.warning("Twitter API token not configured")
            return []
        
        try:
            # Twitter API v2 endpoint
            url = "https://api.twitter.com/2/tweets/search/recent"
            
            headers = {
                "Authorization": f"Bearer {self.bearer_token}",
                "User-Agent": "TariffRadar/1.0"
            }
            
            # Search query - tariff-related tweets from specific accounts
            query = f"from:{username} (tariff OR 关税 OR trade OR 贸易)"
            
            params = {
                "query": query,
                "max_results": 100,  # Free tier: 10-100 tweets per request
                "tweet.fields": "created_at,text,author_id,public_metrics",
                "expansions": "author_id",
                "user.fields": "username,name"
            }
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url, headers=headers, params=params)
                response.raise_for_status()
                
                data = response.json()
                
                tweets = []
                if data.get("data"):
                    for tweet in data["data"]:
                        tweets.append({
                            'title': f"@{username}: {tweet['text'][:100]}...",
                            'content': tweet['text'],
                            'url': f"https://twitter.com/{username}/status/{tweet['id']}",
                            'published_at': datetime.fromisoformat(tweet['created_at'].replace('Z', '+00:00')),
                            'source': f"Twitter @{username}",
                            'language': 'en',
                            'priority': 'high',
                            'tweet_id': tweet['id'],
                            'metrics': tweet.get('public_metrics', {})
                        })
                
                logger.info(f"Scraped {len(tweets)} tweets from @{username} via API")
                return tweets
                
        except Exception as e:
            logger.error(f"Failed to scrape @{username} via Twitter API: {e}")
            return []
    
    async def scrape_via_rsshub(self, username: str) -> List[Dict[str, Any]]:
        """
        Scrape using RSSHub Twitter route
        No API key needed!
        """
        try:
            url = f"{self.rsshub_url}/twitter/user/{username}"
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url)
                response.raise_for_status()
                
                # Parse RSS
                from xml.etree import ElementTree as ET
                root = ET.fromstring(response.content)
                
                tweets = []
                for item in root.findall('.//item')[:50]:  # Latest 50
                    title = item.find('title').text if item.find('title') is not None else ""
                    link = item.find('link').text if item.find('link') is not None else ""
                    description = item.find('description').text if item.find('description') is not None else ""
                    pub_date = item.find('pubDate').text if item.find('pubDate') is not None else ""
                    
                    # Parse date
                    published_at = None
                    if pub_date:
                        try:
                            from email.utils import parsedate_to_datetime
                            published_at = parsedate_to_datetime(pub_date)
                        except:
                            pass
                    
                    # Clean description (remove HTML)
                    soup = BeautifulSoup(description, 'html.parser')
                    clean_text = soup.get_text(strip=True)
                    
                    tweets.append({
                        'title': f"@{username}: {title[:100]}",
                        'content': clean_text,
                        'url': link,
                        'published_at': published_at or datetime.now(timezone.utc),
                        'source': f"Twitter @{username}",
                        'language': 'en',
                        'priority': 'high'
                    })
                
                logger.info(f"Scraped {len(tweets)} tweets from @{username} via RSSHub")
                return tweets
                
        except Exception as e:
            logger.error(f"Failed to scrape @{username} via RSSHub: {e}")
            return []
    
    async def scrape_via_nitter(self, username: str) -> List[Dict[str, Any]]:
        """
        Scrape using Nitter (Twitter frontend)
        No API key needed, but less reliable
        """
        for nitter_instance in self.nitter_instances:
            try:
                url = f"{nitter_instance}/{username}"
                
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.get(url)
                    response.raise_for_status()
                    
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    tweets = []
                    tweet_items = soup.select('.timeline-item')
                    
                    for item in tweet_items[:30]:
                        try:
                            # Extract tweet text
                            tweet_content = item.select_one('.tweet-content')
                            if not tweet_content:
                                continue
                            
                            text = tweet_content.get_text(strip=True)
                            
                            # Extract tweet link
                            tweet_link = item.select_one('.tweet-link')
                            if tweet_link:
                                href = tweet_link.get('href', '')
                                full_url = f"https://twitter.com{href}"
                            else:
                                continue
                            
                            # Extract date
                            date_elem = item.select_one('.tweet-date a')
                            published_at = datetime.now(timezone.utc)
                            if date_elem:
                                date_str = date_elem.get('title', '')
                                try:
                                    published_at = datetime.strptime(date_str, '%b %d, %Y · %I:%M %p %Z')
                                    published_at = published_at.replace(tzinfo=timezone.utc)
                                except:
                                    pass
                            
                            tweets.append({
                                'title': f"@{username}: {text[:100]}...",
                                'content': text,
                                'url': full_url,
                                'published_at': published_at,
                                'source': f"Twitter @{username}",
                                'language': 'en',
                                'priority': 'high'
                            })
                            
                        except Exception as e:
                            logger.error(f"Error parsing tweet from Nitter: {e}")
                            continue
                    
                    if tweets:
                        logger.info(f"Scraped {len(tweets)} tweets from @{username} via Nitter ({nitter_instance})")
                        return tweets
                        
            except Exception as e:
                logger.warning(f"Nitter instance {nitter_instance} failed: {e}")
                continue
        
        logger.error(f"All Nitter instances failed for @{username}")
        return []
    
    async def scrape_account(self, username: str) -> List[Dict[str, Any]]:
        """
        Scrape Twitter account with automatic fallback
        
        Priority:
        1. RSSHub (most reliable, no API key)
        2. Twitter API (if configured)
        3. Nitter (backup)
        """
        logger.info(f"Scraping Twitter @{username}")
        
        # Try RSSHub first (most reliable)
        tweets = await self.scrape_via_rsshub(username)
        if tweets:
            return tweets
        
        # Try official API if configured
        if self.bearer_token:
            tweets = await self.scrape_via_twitter_api(username)
            if tweets:
                return tweets
        
        # Fallback to Nitter
        tweets = await self.scrape_via_nitter(username)
        return tweets
    
    async def scrape_key_accounts(self) -> List[Dict[str, Any]]:
        """
        Scrape all key accounts for tariff monitoring
        
        Priority accounts:
        1. @realDonaldTrump - MOST IMPORTANT! Often announces tariffs here first
        2. @POTUS - Official presidential account
        3. @WhiteHouse - Official White House
        4. @USTreasury - Treasury Secretary
        5. @SecRaimondo - Commerce Secretary (Gina Raimondo)
        6. @USTradeRep - USTR (Katherine Tai)
        """
        accounts = [
            "realDonaldTrump",  # Trump personal - HIGHEST PRIORITY
            "POTUS",            # Official president
            "WhiteHouse",       # White House
            "USTreasury",       # Treasury
            "SecRaimondo",      # Commerce Secretary
            "USTradeRep"        # Trade Representative
        ]
        
        # Scrape all accounts in parallel
        results = await asyncio.gather(
            *[self.scrape_account(account) for account in accounts],
            return_exceptions=True
        )
        
        # Combine all tweets
        all_tweets = []
        for i, result in enumerate(results):
            if isinstance(result, list):
                all_tweets.extend(result)
            else:
                logger.error(f"Failed to scrape @{accounts[i]}: {result}")
        
        # Filter for tariff-related content
        tariff_keywords = [
            'tariff', 'tariffs', 'trade war', 'china trade', 
            'import tax', 'customs', '关税', '贸易战',
            'duties', 'trade deal', 'trade policy'
        ]
        
        filtered_tweets = []
        for tweet in all_tweets:
            text_lower = tweet['content'].lower()
            if any(keyword in text_lower for keyword in tariff_keywords):
                filtered_tweets.append(tweet)
        
        logger.info(f"Found {len(filtered_tweets)} tariff-related tweets from {len(all_tweets)} total")
        return filtered_tweets


# Helper function for Celery tasks
async def scrape_twitter_accounts() -> List[Dict[str, Any]]:
    """Helper function to be called from Celery tasks"""
    scraper = TwitterScraper()
    return await scraper.scrape_key_accounts()
