"""
News Impact Analysis Module
Analyzes how news articles impact specific stocks and sectors
"""

import re
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class NewsImpactAnalyzer:
    """Analyzes news impact on stocks and markets"""
    
    def __init__(self):
        # Stock symbol mapping
        self.stock_keywords = {
            'AAPL': ['apple', 'iphone', 'ipad', 'mac', 'tim cook', 'cupertino'],
            'MSFT': ['microsoft', 'windows', 'azure', 'office', 'satya nadella'],
            'GOOGL': ['google', 'alphabet', 'youtube', 'android', 'sundar pichai'],
            'AMZN': ['amazon', 'aws', 'prime', 'bezos', 'andy jassy'],
            'TSLA': ['tesla', 'elon musk', 'electric vehicle', 'ev', 'model 3', 'model y'],
            'NVDA': ['nvidia', 'gpu', 'ai chip', 'jensen huang', 'cuda'],
            'META': ['meta', 'facebook', 'instagram', 'whatsapp', 'mark zuckerberg'],
            'PFE': ['pfizer', 'vaccine', 'covid', 'pharma', 'drug approval'],
            'JNJ': ['johnson', 'johnson & johnson', 'medical device', 'pharmaceutical'],
            'JPM': ['jpmorgan', 'jp morgan', 'jamie dimon', 'investment bank'],
            'BAC': ['bank of america', 'boa', 'banking'],
        }
        
        # News impact keywords and their sentiment scores
        self.impact_keywords = {
            # Positive impact keywords
            'earnings beat': 8.5,
            'revenue growth': 7.0,
            'profit surge': 8.0,
            'upgraded': 6.5,
            'buyback': 6.0,
            'dividend increase': 5.5,
            'merger': 7.5,
            'acquisition': 7.0,
            'partnership': 5.0,
            'innovation': 4.5,
            'breakthrough': 6.5,
            'fda approval': 8.0,
            'patent': 4.0,
            'expansion': 4.5,
            
            # Negative impact keywords  
            'earnings miss': -8.0,
            'revenue decline': -6.5,
            'loss': -7.0,
            'downgraded': -6.0,
            'lawsuit': -5.5,
            'investigation': -6.0,
            'recall': -7.5,
            'bankruptcy': -9.0,
            'scandal': -7.0,
            'layoffs': -5.0,
            'closure': -6.5,
            'fine': -4.5,
            'regulatory': -4.0,
            'tariff': -5.5,  # Tariff impact from original system
            'trade war': -6.0,
            'sanctions': -7.0,
        }
        
        # Sector keywords
        self.sector_keywords = {
            'Technology': ['tech', 'software', 'ai', 'artificial intelligence', 'cloud', 'digital'],
            'Healthcare': ['healthcare', 'medical', 'pharma', 'drug', 'vaccine', 'fda'],
            'Financial': ['bank', 'financial', 'fed', 'interest rate', 'credit'],
            'Energy': ['energy', 'oil', 'gas', 'renewable', 'solar', 'wind'],
            'Defense': ['defense', 'military', 'weapon', 'aerospace', 'security'],
            'Consumer': ['retail', 'consumer', 'shopping', 'brand'],
            'Industrial': ['manufacturing', 'industrial', 'factory', 'supply chain']
        }
        
        # Geopolitical risk keywords (from original tariff system)
        self.geopolitical_keywords = {
            'china trade': -6.0,
            'tariff': -5.5,
            'trade war': -6.5,
            'sanctions': -7.0,
            'geopolitical': -4.0,
            'tension': -3.5,
            'diplomatic': -2.0,
            'policy change': -3.0,
            'election': -2.5,
            'fed decision': -4.5,
            'interest rate': -3.0,
        }
    
    def analyze_article_impact(self, article: Dict) -> Dict:
        """Analyze the impact of a news article on stocks and sectors"""
        try:
            title = article.get('title', '').lower()
            content = article.get('content', '').lower()
            text = f"{title} {content}"
            
            # Find affected stocks
            affected_stocks = self._find_affected_stocks(text)
            
            # Calculate impact scores
            impact_score = self._calculate_impact_score(text)
            
            # Find affected sectors
            affected_sectors = self._find_affected_sectors(text)
            
            # Calculate geopolitical risk
            geo_risk = self._calculate_geopolitical_risk(text)
            
            # Determine overall sentiment
            sentiment = self._determine_sentiment(impact_score)
            
            analysis = {
                'article_id': article.get('id', 'unknown'),
                'title': article.get('title', ''),
                'source': article.get('source', ''),
                'published_at': article.get('published_at', datetime.now().isoformat()),
                'overall_impact_score': round(impact_score, 2),
                'sentiment': sentiment,
                'affected_stocks': affected_stocks,
                'affected_sectors': affected_sectors,
                'geopolitical_risk_score': round(geo_risk, 2),
                'confidence_level': self._calculate_confidence(text, affected_stocks),
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"✅ Analyzed article impact: {len(affected_stocks)} stocks, score: {impact_score}")
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Error analyzing article impact: {e}")
            return {}
    
    def _find_affected_stocks(self, text: str) -> List[Dict]:
        """Find stocks mentioned or likely affected by the article"""
        affected_stocks = []
        
        for symbol, keywords in self.stock_keywords.items():
            mentions = 0
            matched_keywords = []
            
            for keyword in keywords:
                if keyword in text:
                    mentions += 1
                    matched_keywords.append(keyword)
            
            if mentions > 0:
                # Calculate confidence based on number of keyword matches
                confidence = min(90, mentions * 25 + 40)
                
                affected_stocks.append({
                    'symbol': symbol,
                    'mentions': mentions,
                    'matched_keywords': matched_keywords,
                    'confidence': confidence
                })
        
        # Sort by confidence
        affected_stocks.sort(key=lambda x: x['confidence'], reverse=True)
        return affected_stocks[:5]  # Top 5 most relevant stocks
    
    def _calculate_impact_score(self, text: str) -> float:
        """Calculate the overall impact score of the article"""
        total_score = 0.0
        keyword_count = 0
        
        for keyword, score in self.impact_keywords.items():
            if keyword in text:
                total_score += score
                keyword_count += 1
        
        # Add geopolitical impact
        for keyword, score in self.geopolitical_keywords.items():
            if keyword in text:
                total_score += score
                keyword_count += 1
        
        # Normalize score (average if multiple keywords found)
        if keyword_count > 0:
            return total_score / keyword_count
        else:
            return 0.0
    
    def _find_affected_sectors(self, text: str) -> List[Dict]:
        """Find sectors mentioned or likely affected"""
        affected_sectors = []
        
        for sector, keywords in self.sector_keywords.items():
            mentions = 0
            for keyword in keywords:
                if keyword in text:
                    mentions += 1
            
            if mentions > 0:
                affected_sectors.append({
                    'sector': sector,
                    'mentions': mentions,
                    'relevance_score': mentions * 2.5
                })
        
        affected_sectors.sort(key=lambda x: x['relevance_score'], reverse=True)
        return affected_sectors
    
    def _calculate_geopolitical_risk(self, text: str) -> float:
        """Calculate geopolitical risk score based on keywords"""
        risk_score = 0.0
        risk_factors = 0
        
        for keyword, impact in self.geopolitical_keywords.items():
            if keyword in text:
                risk_score += abs(impact)  # Use absolute value for risk
                risk_factors += 1
        
        # Normalize to 0-10 scale
        if risk_factors > 0:
            normalized_risk = min(10.0, (risk_score / risk_factors) * 1.5)
            return normalized_risk
        
        return 0.0
    
    def _determine_sentiment(self, impact_score: float) -> str:
        """Determine sentiment based on impact score"""
        if impact_score >= 6.0:
            return "Very Positive"
        elif impact_score >= 3.0:
            return "Positive"
        elif impact_score >= -3.0:
            return "Neutral"
        elif impact_score >= -6.0:
            return "Negative"
        else:
            return "Very Negative"
    
    def _calculate_confidence(self, text: str, affected_stocks: List[Dict]) -> int:
        """Calculate confidence level of the analysis"""
        base_confidence = 50
        
        # Higher confidence if specific stocks are mentioned
        if affected_stocks:
            base_confidence += len(affected_stocks) * 10
        
        # Higher confidence if multiple impact keywords found
        impact_keywords_found = sum(1 for keyword in self.impact_keywords.keys() if keyword in text)
        base_confidence += impact_keywords_found * 5
        
        return min(95, base_confidence)
    
    async def calculate_geopolitical_risk(self, news_articles: List[str]) -> Dict:
        """Calculate geopolitical risk score based on news articles"""
        if not news_articles:
            return {
                'risk_score': 0.5,
                'risk_level': 'Medium',
                'factors': [],
                'affected_sectors': [],
                'analysis_timestamp': datetime.now().isoformat()
            }
        
        risk_factors = []
        total_risk = 0.0
        affected_sectors = set()
        
        for article in news_articles:
            article_risk = self._calculate_geopolitical_risk(article)
            total_risk += article_risk
            
            # Extract risk factors from article
            article_lower = article.lower()
            if any(keyword in article_lower for keyword in ['war', 'conflict', 'sanctions']):
                risk_factors.append({
                    'factor': 'Geopolitical Tensions',
                    'severity': article_risk,
                    'source': article[:50] + '...'
                })
                affected_sectors.update(['Energy', 'Defense', 'Technology'])
            
            if any(keyword in article_lower for keyword in ['trade', 'tariff', 'economic']):
                risk_factors.append({
                    'factor': 'Trade Relations',
                    'severity': article_risk,
                    'source': article[:50] + '...'
                })
                affected_sectors.update(['Manufacturing', 'Technology', 'Consumer Goods'])
        
        # Normalize risk score (0-1)
        risk_score = min(total_risk / len(news_articles), 1.0) if news_articles else 0.5
        
        # Determine risk level
        if risk_score < 0.3:
            risk_level = 'Low'
        elif risk_score < 0.6:
            risk_level = 'Medium'
        elif risk_score < 0.8:
            risk_level = 'High'
        else:
            risk_level = 'Critical'
        
        return {
            'risk_score': round(risk_score, 2),
            'risk_level': risk_level,
            'factors': risk_factors[:3],  # Top 3 factors
            'affected_sectors': list(affected_sectors),
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    async def analyze_stock_impact(self, symbol: str, news_articles: List[str]) -> Dict:
        """Analyze how news articles impact a specific stock"""
        try:
            if not news_articles:
                return {
                    'impact_score': 0.0,
                    'sentiment': 'neutral',
                    'confidence': 0.5,
                    'key_themes': []
                }
            
            total_impact = 0.0
            relevant_articles = 0
            themes = []
            
            # Check each article for stock relevance and impact
            for article in news_articles:
                article_lower = article.lower()
                
                # Check if article mentions this stock
                stock_keywords = self.stock_keywords.get(symbol, [])
                is_relevant = any(keyword in article_lower for keyword in stock_keywords)
                
                # Also check for symbol mention
                if symbol.lower() in article_lower:
                    is_relevant = True
                
                if is_relevant:
                    relevant_articles += 1
                    
                    # Calculate impact score for this article
                    impact = self._calculate_impact_score(article_lower)
                    total_impact += impact
                    
                    # Extract themes
                    for keyword, score in self.impact_keywords.items():
                        if keyword in article_lower:
                            themes.append(keyword.replace('_', ' ').title())
            
            # Calculate average impact
            if relevant_articles > 0:
                avg_impact = total_impact / relevant_articles
                confidence = min(0.95, 0.5 + (relevant_articles * 0.1))
            else:
                # No relevant articles - check for general market sentiment
                avg_impact = sum(self._calculate_impact_score(article) for article in news_articles) / len(news_articles)
                confidence = 0.3  # Low confidence for general market impact
            
            # Determine sentiment
            if avg_impact > 1.0:
                sentiment = 'positive'
            elif avg_impact < -1.0:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
            
            # Limit themes to top 3
            themes = list(set(themes))[:3]
            if not themes:
                themes = ['market volatility', 'general news', 'sector trends']
            
            return {
                'impact_score': round(avg_impact, 2),
                'sentiment': sentiment,
                'confidence': round(confidence, 2),
                'key_themes': themes,
                'relevant_articles': relevant_articles,
                'total_articles': len(news_articles)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing stock impact for {symbol}: {e}")
            return {
                'impact_score': 0.0,
                'sentiment': 'neutral',
                'confidence': 0.5,
                'key_themes': ['analysis_error']
            }

# Global instance
news_impact_analyzer = NewsImpactAnalyzer()