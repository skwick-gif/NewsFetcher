"""
Perplexity AI Integration for Financial Analysis
Using updated Sonar models (2025): sonar, sonar-pro, sonar-reasoning
"""

import os
import json
import logging
import httpx
from typing import Dict, Any, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)

class PerplexityFinancialAnalyzer:
    """Perplexity AI for financial news and market analysis"""
    
    def __init__(self):
        self.api_key = os.getenv('PERPLEXITY_API_KEY')
        self.model = "sonar"  # Updated 2025 model - fast search with grounding
        self.base_url = "https://api.perplexity.ai"
        
        # Initialize client
        self.client = None
        if self.api_key and self.api_key not in ['YOUR_PERPLEXITY_API_KEY', '']:
            try:
                self.client = httpx.AsyncClient(
                    base_url=self.base_url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    timeout=30.0
                )
                logger.info(f"✅ Perplexity client initialized with model: {self.model}")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Perplexity client: {e}")
        else:
            logger.warning("⚠️ Perplexity API key not configured")
    
    async def analyze_stock_news(self, symbol: str, news_headlines: List[str]) -> Optional[Dict[str, Any]]:
        """Analyze news headlines for a stock using Perplexity"""
        if not self.client:
            logger.warning("Perplexity client not available")
            return None
        
        try:
            # Create prompt
            headlines_text = "\n".join([f"- {h}" for h in news_headlines[:10]])
            
            prompt = f"""Analyze these recent news headlines about {symbol} stock and provide financial insights:

{headlines_text}

Provide a JSON response with:
{{
    "overall_sentiment": "positive/negative/neutral",
    "market_impact": "bullish/bearish/neutral",
    "confidence": 0.0-1.0,
    "key_insights": ["insight1", "insight2", "insight3"],
    "risk_factors": ["risk1", "risk2"],
    "recommendation": "buy/hold/sell",
    "reasoning": "brief explanation"
}}"""

            response = await self.client.post(
                "/chat/completions",
                json={
                    "model": self.model,
                    "messages": [
                        {
                            "role": "system", 
                            "content": "You are a financial analyst specializing in stock market analysis and news interpretation."
                        },
                        {
                            "role": "user", 
                            "content": prompt
                        }
                    ],
                    "temperature": 0.2,
                    "max_tokens": 1000
                }
            )
            
            response.raise_for_status()
            result = response.json()
            
            # Parse response
            content = result["choices"][0]["message"]["content"]
            analysis = self._parse_json_response(content)
            
            if analysis:
                logger.info(f"✅ Perplexity analyzed {symbol}: {analysis.get('market_impact')}")
                return analysis
            else:
                logger.warning(f"⚠️ Failed to parse Perplexity response for {symbol}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Perplexity analysis failed for {symbol}: {e}")
            return None
    
    async def analyze_market_event(self, event_description: str) -> Optional[Dict[str, Any]]:
        """Analyze a market event or news using Perplexity's online search"""
        if not self.client:
            return None
        
        try:
            prompt = f"""Analyze this market event and its potential impact:

Event: {event_description}

Provide detailed analysis in JSON format:
{{
    "event_type": "earnings/geopolitical/regulatory/economic/other",
    "severity": "high/medium/low",
    "affected_sectors": ["sector1", "sector2"],
    "market_sentiment": "positive/negative/neutral",
    "short_term_impact": "description",
    "long_term_impact": "description",
    "stocks_to_watch": ["SYMBOL1", "SYMBOL2"],
    "recommendation": "detailed trading strategy",
    "confidence": 0.0-1.0
}}"""

            response = await self.client.post(
                "/chat/completions",
                json={
                    "model": self.model,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are an expert market analyst with real-time market data access."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "temperature": 0.3,
                    "max_tokens": 1500
                }
            )
            
            response.raise_for_status()
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            
            return self._parse_json_response(content)
            
        except Exception as e:
            logger.error(f"❌ Market event analysis failed: {e}")
            return None
    
    async def get_stock_insights(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive stock insights using Perplexity's online search"""
        if not self.client:
            return None
        
        try:
            prompt = f"""Analyze {symbol} stock and provide insights on:
1. Current price outlook and analyst consensus
2. Recent news and developments
3. Key risks and opportunities
4. Recommendation (buy/hold/sell)

Be concise and actionable."""

            response = await self.client.post(
                "/chat/completions",
                json={
                    "model": self.model,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a professional stock analyst providing real-time market insights."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "temperature": 0.2,
                    "max_tokens": 1000
                }
            )
            
            response.raise_for_status()
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            citations = result.get("citations", [])
            
            # Return structured response with actual content (not trying to parse as JSON)
            insights = {
                "symbol": symbol,
                "analysis": content,
                "citations": citations[:5],  # Top 5 sources
                "timestamp": datetime.now().isoformat(),
                "model": self.model
            }
            
            logger.info(f"✅ Got Perplexity insights for {symbol} from {len(citations)} sources")
            return insights
            
        except Exception as e:
            logger.error(f"❌ Failed to get insights for {symbol}: {e}")
            return None
    
    async def analyze_sector_trends(self, sector: str) -> Optional[Dict[str, Any]]:
        """Analyze sector trends and news"""
        if not self.client:
            return None
        
        try:
            prompt = f"""Analyze current trends and outlook for the {sector} sector:

Provide JSON analysis:
{{
    "sector_sentiment": "bullish/bearish/neutral",
    "key_trends": ["trend1", "trend2", "trend3"],
    "leading_stocks": ["SYMBOL1", "SYMBOL2"],
    "lagging_stocks": ["SYMBOL3", "SYMBOL4"],
    "catalysts": ["catalyst1", "catalyst2"],
    "headwinds": ["headwind1", "headwind2"],
    "outlook": "short description",
    "confidence": 0.0-1.0
}}"""

            response = await self.client.post(
                "/chat/completions",
                json={
                    "model": self.model,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a sector analyst tracking industry trends and competitive dynamics."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "temperature": 0.3,
                    "max_tokens": 1200
                }
            )
            
            response.raise_for_status()
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            
            return self._parse_json_response(content)
            
        except Exception as e:
            logger.error(f"❌ Sector analysis failed for {sector}: {e}")
            return None
    
    async def analyze_async(
        self,
        symbol: str,
        user_query: str,
        analysis_type: str = "event_analysis"
    ) -> Optional[Dict[str, Any]]:
        """
        Generic analysis endpoint for scheduler integration
        
        Args:
            symbol: Stock ticker symbol
            user_query: The query or context to analyze
            analysis_type: Type of analysis (event_analysis, stock_news, etc.)
        
        Returns:
            Dict with analysis, recommendation, confidence, citations
        """
        if not self.client:
            logger.warning("Perplexity client not available")
            return None
        
        try:
            # Create system prompt based on analysis type
            if analysis_type == "event_analysis":
                system_prompt = "You are a financial analyst specializing in real-time market event analysis and trading recommendations."
            elif analysis_type == "stock_news":
                system_prompt = "You are a financial analyst specializing in stock news interpretation and sentiment analysis."
            else:
                system_prompt = "You are a professional financial analyst providing data-driven market insights."
            
            # Request analysis
            response = await self.client.post(
                "/chat/completions",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_query}
                    ],
                    "temperature": 0.2,
                    "max_tokens": 1500
                }
            )
            
            response.raise_for_status()
            result = response.json()
            
            # Extract content
            content = result["choices"][0]["message"]["content"]
            
            # Try to parse as JSON first
            parsed = self._parse_json_response(content)
            
            if parsed:
                # Structured response
                return {
                    "analysis": parsed.get("reasoning", content),
                    "recommendation": parsed.get("recommendation", "HOLD"),
                    "confidence": parsed.get("confidence", 0.5),
                    "citations": parsed.get("citations", [])
                }
            else:
                # Plain text response
                return {
                    "analysis": content,
                    "recommendation": "HOLD",
                    "confidence": 0.5,
                    "citations": []
                }
                
        except Exception as e:
            logger.error(f"❌ Perplexity analyze_async failed for {symbol}: {e}")
            return None
    
    def _parse_json_response(self, content: str) -> Optional[Dict[str, Any]]:
        """Parse JSON from LLM response"""
        try:
            # Clean response
            cleaned = content.strip()
            
            # Remove markdown code blocks if present
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            elif cleaned.startswith("```"):
                cleaned = cleaned[3:]
            
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            
            cleaned = cleaned.strip()
            
            # Parse JSON
            parsed = json.loads(cleaned)
            return parsed
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.debug(f"Raw content: {content[:200]}")
            return None
    
    async def close(self):
        """Close the HTTP client"""
        if self.client:
            await self.client.aclose()
    
    def is_available(self) -> bool:
        """Check if Perplexity is available"""
        return self.client is not None

# Test function
async def test_perplexity():
    """Test Perplexity integration"""
    analyzer = PerplexityFinancialAnalyzer()
    
    if not analyzer.is_available():
        print("❌ Perplexity not available - check API key")
        return False
    
    print("✅ Perplexity client initialized")
    
    # Test stock insights
    print("\n🔍 Testing stock insights for AAPL...")
    insights = await analyzer.get_stock_insights("AAPL")
    
    if insights:
        print("✅ Got insights:")
        print(json.dumps(insights, indent=2))
        return True
    else:
        print("❌ Failed to get insights")
        return False

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_perplexity())
