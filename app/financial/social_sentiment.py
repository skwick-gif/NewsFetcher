"""
Social Media Sentiment Analysis Module
Integrates with Twitter, Reddit, Discord APIs for real-time sentiment analysis
"""

import asyncio
import aiohttp
import re
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
import os

logger = logging.getLogger(__name__)

@dataclass
class SocialPost:
    """Social media post data structure"""
    platform: str
    content: str
    author: str
    timestamp: datetime
    engagement_score: float  # likes, retweets, upvotes etc.
    follower_count: int
    sentiment_raw: str

class SocialMediaAnalyzer:
    """Real-time social media sentiment analysis"""
    
    def __init__(self):
        # API credentials
        self.twitter_bearer_token = os.getenv('TWITTER_BEARER_TOKEN')
        self.reddit_client_id = os.getenv('REDDIT_CLIENT_ID')
        self.reddit_client_secret = os.getenv('REDDIT_CLIENT_SECRET')
        self.discord_bot_token = os.getenv('DISCORD_BOT_TOKEN')
        
        # Stock-related keywords and hashtags
        self.stock_hashtags = {
            'AAPL': ['#AAPL', '#Apple', '#iPhone', '#AppleStock', '$AAPL'],
            'TSLA': ['#TSLA', '#Tesla', '#ElonMusk', '#TeslaStock', '$TSLA'],
            'NVDA': ['#NVDA', '#NVIDIA', '#AIStock', '$NVDA'],
            'MSFT': ['#MSFT', '#Microsoft', '$MSFT'],
            'GOOGL': ['#GOOGL', '#Google', '#Alphabet', '$GOOGL'],
            'AMZN': ['#AMZN', '#Amazon', '$AMZN'],
            'META': ['#META', '#Facebook', '#Metaverse', '$META']
        }
        
        # Influential accounts to track (higher weight)
        self.influential_accounts = {
            'twitter': ['elonmusk', 'cathiedwood', 'chamath', 'naval', 'unusual_whales'],
            'reddit': ['wallstreetbets', 'stocks', 'investing', 'SecurityAnalysis'],
            'discord': []  # Will be channel IDs
        }
        
        # Sentiment scoring weights
        self.engagement_weights = {
            'twitter': {
                'retweets': 3.0,
                'likes': 1.0,
                'replies': 2.0,
                'follower_multiplier': 0.001  # Per 1000 followers
            },
            'reddit': {
                'upvotes': 2.0,
                'comments': 1.5,
                'awards': 5.0
            },
            'discord': {
                'reactions': 1.0,
                'replies': 1.5
            }
        }

    async def get_twitter_sentiment(self, symbol: str, hours_back: int = 24) -> List[SocialPost]:
        """Get Twitter sentiment for a stock symbol"""
        try:
            if not self.twitter_bearer_token:
                logger.warning("No Twitter API token - using demo data")
                return self._generate_demo_twitter_data(symbol)
            
            hashtags = self.stock_hashtags.get(symbol, [f'${symbol}'])
            query = ' OR '.join(hashtags)
            
            # Twitter API v2 recent search
            url = "https://api.twitter.com/2/tweets/search/recent"
            headers = {"Authorization": f"Bearer {self.twitter_bearer_token}"}
            
            params = {
                'query': f'({query}) -is:retweet lang:en',
                'tweet.fields': 'public_metrics,created_at,author_id',
                'user.fields': 'public_metrics',
                'expansions': 'author_id',
                'max_results': 100
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_twitter_data(data, symbol)
                    else:
                        logger.error(f"Twitter API error: {response.status}")
                        return []
                        
        except Exception as e:
            logger.error(f"Twitter sentiment error: {e}")
            return self._generate_demo_twitter_data(symbol)
    
    async def get_reddit_sentiment(self, symbol: str, hours_back: int = 24) -> List[SocialPost]:
        """Get Reddit sentiment for a stock symbol"""
        try:
            if not self.reddit_client_id:
                logger.warning("No Reddit API credentials - using demo data")
                return self._generate_demo_reddit_data(symbol)
            
            # Reddit OAuth and search
            auth_url = "https://www.reddit.com/api/v1/access_token"
            auth_data = {
                'grant_type': 'client_credentials'
            }
            
            auth = aiohttp.BasicAuth(self.reddit_client_id, self.reddit_client_secret)
            
            async with aiohttp.ClientSession() as session:
                # Get access token
                async with session.post(auth_url, data=auth_data, auth=auth) as response:
                    auth_response = await response.json()
                    access_token = auth_response.get('access_token')
                
                if not access_token:
                    return self._generate_demo_reddit_data(symbol)
                
                # Search Reddit posts
                headers = {
                    'Authorization': f'bearer {access_token}',
                    'User-Agent': 'MarketPulse/1.0'
                }
                
                posts = []
                for subreddit in self.influential_accounts['reddit']:
                    search_url = f"https://oauth.reddit.com/r/{subreddit}/search"
                    params = {
                        'q': f'{symbol} OR ${symbol}',
                        'sort': 'new',
                        'limit': 25,
                        't': 'day'
                    }
                    
                    async with session.get(search_url, headers=headers, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            posts.extend(self._parse_reddit_data(data, symbol))
                
                return posts
                
        except Exception as e:
            logger.error(f"Reddit sentiment error: {e}")
            return self._generate_demo_reddit_data(symbol)
    
    def _parse_twitter_data(self, data: Dict, symbol: str) -> List[SocialPost]:
        """Parse Twitter API response into SocialPost objects"""
        posts = []
        
        try:
            tweets = data.get('data', [])
            users = {user['id']: user for user in data.get('includes', {}).get('users', [])}
            
            for tweet in tweets:
                author_id = tweet.get('author_id')
                author = users.get(author_id, {})
                
                metrics = tweet.get('public_metrics', {})
                author_metrics = author.get('public_metrics', {})
                
                # Calculate engagement score
                engagement = (
                    metrics.get('retweet_count', 0) * 3 +
                    metrics.get('like_count', 0) * 1 +
                    metrics.get('reply_count', 0) * 2
                )
                
                post = SocialPost(
                    platform='twitter',
                    content=tweet.get('text', ''),
                    author=author.get('username', 'unknown'),
                    timestamp=datetime.fromisoformat(tweet.get('created_at', '').replace('Z', '+00:00')),
                    engagement_score=engagement,
                    follower_count=author_metrics.get('followers_count', 0),
                    sentiment_raw=tweet.get('text', '')
                )
                posts.append(post)
                
        except Exception as e:
            logger.error(f"Error parsing Twitter data: {e}")
        
        return posts
    
    def _parse_reddit_data(self, data: Dict, symbol: str) -> List[SocialPost]:
        """Parse Reddit API response into SocialPost objects"""
        posts = []
        
        try:
            for post_data in data.get('data', {}).get('children', []):
                post = post_data.get('data', {})
                
                # Calculate engagement score
                engagement = (
                    post.get('ups', 0) * 2 +
                    post.get('num_comments', 0) * 1.5 +
                    post.get('total_awards_received', 0) * 5
                )
                
                social_post = SocialPost(
                    platform='reddit',
                    content=f"{post.get('title', '')} {post.get('selftext', '')}",
                    author=post.get('author', 'unknown'),
                    timestamp=datetime.fromtimestamp(post.get('created_utc', 0)),
                    engagement_score=engagement,
                    follower_count=post.get('subreddit_subscribers', 0),
                    sentiment_raw=post.get('title', '')
                )
                posts.append(social_post)
                
        except Exception as e:
            logger.error(f"Error parsing Reddit data: {e}")
        
        return posts
    
    def _generate_demo_twitter_data(self, symbol: str) -> List[SocialPost]:
        """Generate realistic demo Twitter data"""
        import random
        
        demo_tweets = [
            f"${symbol} looking bullish today! 🚀 #investing",
            f"Just bought more ${symbol} on this dip 💎🙌",
            f"${symbol} earnings report was impressive! Strong buy",
            f"Concerned about ${symbol} valuation at these levels 📉",
            f"${symbol} technical analysis shows resistance at current price",
            f"Love the ${symbol} product roadmap for 2025! 📈",
            f"${symbol} insider selling worries me... thoughts?",
            f"${symbol} vs competitors - who wins long term?",
        ]
        
        posts = []
        for i, tweet in enumerate(demo_tweets):
            engagement = random.randint(10, 1000)
            followers = random.randint(100, 50000)
            
            post = SocialPost(
                platform='twitter',
                content=tweet,
                author=f'trader_{random.randint(1000, 9999)}',
                timestamp=datetime.now() - timedelta(hours=random.randint(1, 24)),
                engagement_score=engagement,
                follower_count=followers,
                sentiment_raw=tweet
            )
            posts.append(post)
        
        logger.info(f"Generated {len(posts)} demo Twitter posts for {symbol}")
        return posts
    
    def _generate_demo_reddit_data(self, symbol: str) -> List[SocialPost]:
        """Generate realistic demo Reddit data"""
        import random
        
        demo_posts = [
            f"DD: Why {symbol} is undervalued - Analysis inside",
            f"{symbol} Q3 earnings discussion thread",
            f"Thoughts on {symbol} after today's drop?",
            f"YOLO: All in on {symbol} - here's why",
            f"{symbol} technical breakout confirmed?",
            f"Bear case for {symbol} - change my mind",
            f"{symbol} options flow looking interesting",
            f"Long term outlook for {symbol} - 5 year hold",
        ]
        
        posts = []
        for post_text in demo_posts:
            engagement = random.randint(50, 2000)
            subscribers = random.randint(1000, 1000000)
            
            post = SocialPost(
                platform='reddit',
                content=post_text,
                author=f'user_{random.randint(100, 999)}',
                timestamp=datetime.now() - timedelta(hours=random.randint(1, 48)),
                engagement_score=engagement,
                follower_count=subscribers,
                sentiment_raw=post_text
            )
            posts.append(post)
        
        logger.info(f"Generated {len(posts)} demo Reddit posts for {symbol}")
        return posts
    
    async def analyze_social_sentiment(self, symbol: str) -> Dict:
        """Analyze overall social media sentiment for a stock"""
        try:
            # Get data from all platforms
            twitter_posts = await self.get_twitter_sentiment(symbol)
            reddit_posts = await self.get_reddit_sentiment(symbol)
            
            all_posts = twitter_posts + reddit_posts
            
            if not all_posts:
                return {
                    'overall_sentiment': 'neutral',
                    'sentiment_score': 0.0,
                    'confidence': 0.0,
                    'total_posts': 0,
                    'platform_breakdown': {},
                    'top_themes': []
                }
            
            # Analyze sentiment for each post
            sentiment_scores = []
            platform_stats = {'twitter': 0, 'reddit': 0}
            themes = []
            
            for post in all_posts:
                platform_stats[post.platform] += 1
                
                # Simple sentiment analysis (can be replaced with ML model)
                sentiment_score = self._calculate_post_sentiment(post)
                
                # Weight by engagement and follower count
                weight = 1 + (post.engagement_score / 100) + (post.follower_count / 10000)
                weighted_score = sentiment_score * min(weight, 10)  # Cap weight at 10x
                
                sentiment_scores.append(weighted_score)
                
                # Extract themes
                themes.extend(self._extract_themes(post.content))
            
            # Calculate overall metrics
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
            
            # Determine sentiment label
            if avg_sentiment > 0.3:
                sentiment_label = 'bullish'
            elif avg_sentiment < -0.3:
                sentiment_label = 'bearish'
            else:
                sentiment_label = 'neutral'
            
            # Calculate confidence based on volume and consistency
            confidence = min(1.0, len(all_posts) / 50)  # More posts = higher confidence
            
            # Top themes
            theme_counts = {}
            for theme in themes:
                theme_counts[theme] = theme_counts.get(theme, 0) + 1
            
            top_themes = sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            
            return {
                'overall_sentiment': sentiment_label,
                'sentiment_score': round(avg_sentiment, 3),
                'confidence': round(confidence, 3),
                'total_posts': len(all_posts),
                'platform_breakdown': platform_stats,
                'top_themes': [theme for theme, count in top_themes],
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing social sentiment for {symbol}: {e}")
            return {
                'overall_sentiment': 'neutral',
                'sentiment_score': 0.0,
                'confidence': 0.0,
                'total_posts': 0,
                'platform_breakdown': {},
                'top_themes': [],
                'error': str(e)
            }
    
    def _calculate_post_sentiment(self, post: SocialPost) -> float:
        """Calculate sentiment score for a single post (-1 to +1)"""
        content = post.content.lower()
        
        # Positive indicators
        positive_words = [
            'bullish', 'buy', 'moon', 'rocket', '🚀', 'diamond', 'hands', '💎',
            'hold', 'strong', 'up', 'gain', 'profit', 'bullish', 'calls',
            'breakout', 'pump', 'surge', 'rally', 'growth', 'promising'
        ]
        
        # Negative indicators
        negative_words = [
            'bearish', 'sell', 'dump', 'crash', 'drop', 'fall', 'puts',
            'short', 'overvalued', 'bubble', 'correction', 'fear', 'panic',
            'loss', 'bag', 'holder', 'rip', 'dead', 'scam'
        ]
        
        positive_count = sum(1 for word in positive_words if word in content)
        negative_count = sum(1 for word in negative_words if word in content)
        
        # Calculate score
        if positive_count + negative_count == 0:
            return 0.0
        
        score = (positive_count - negative_count) / max(positive_count + negative_count, 1)
        return max(-1.0, min(1.0, score))
    
    def _extract_themes(self, content: str) -> List[str]:
        """Extract key themes from post content"""
        content = content.lower()
        themes = []
        
        theme_keywords = {
            'earnings': ['earnings', 'eps', 'revenue', 'profit', 'guidance'],
            'technical_analysis': ['support', 'resistance', 'breakout', 'chart', 'pattern'],
            'options': ['calls', 'puts', 'strike', 'expiry', 'iv', 'gamma'],
            'fundamentals': ['valuation', 'pe', 'growth', 'debt', 'cash'],
            'news': ['announcement', 'partnership', 'acquisition', 'launch'],
            'macro': ['fed', 'rates', 'inflation', 'economy', 'gdp']
        }
        
        for theme, keywords in theme_keywords.items():
            if any(keyword in content for keyword in keywords):
                themes.append(theme)
        
        return themes