"""
Perplexity AI Integration for Real Stock Analysis
Advanced financial analysis using Perplexity's Sonar Reasoning Pro model
"""

import os
import requests
import logging
import asyncio
from typing import Dict, Optional
from datetime import datetime
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class PerplexityFinanceAnalyzer:
    """Real AI-powered financial analysis using Perplexity"""
    
    def __init__(self):
        self.api_key = os.getenv('PERPLEXITY_API_KEY')
        self.base_url = "https://api.perplexity.ai/chat/completions"
        self.model = "sonar-reasoning-pro"  # Best for complex financial analysis
        
        if not self.api_key:
            raise ValueError("PERPLEXITY_API_KEY not found in environment variables")
            
        logger.info("✅ Perplexity Finance Analyzer initialized")
    
    async def analyze_stock(self, symbol: str, current_price: float) -> Dict:
        """
        Get comprehensive AI analysis for a stock using Perplexity
        """
        return await asyncio.get_event_loop().run_in_executor(
            None, self._analyze_stock_sync, symbol, current_price
        )
    
    def _analyze_stock_sync(self, symbol: str, current_price: float) -> Dict:
        """
        Get comprehensive AI analysis for a stock using Perplexity
        """
        try:
            # Create comprehensive financial analysis prompt
            prompt = self._create_financial_prompt(symbol, current_price)
            
            # Send request to Perplexity
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 2000,
                "temperature": 0.1  # Low temperature for consistent financial analysis
            }
            
            logger.info(f"🤖 Sending {symbol} analysis request to Perplexity...")
            
            response = requests.post(
                self.base_url, 
                json=data, 
                headers=headers, 
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                ai_content = result['choices'][0]['message']['content']
                
                # Parse the AI response into structured data
                parsed_analysis = self._parse_ai_response(ai_content, symbol, current_price)
                
                logger.info(f"✅ AI analysis completed for {symbol}")
                return {
                    "status": "success",
                    "symbol": symbol,
                    "ai_analysis": parsed_analysis,
                    "raw_response": ai_content,
                    "citations": result.get('citations', []),
                    "search_results": result.get('search_results', []),
                    "timestamp": datetime.now().isoformat(),
                    "cost": result.get('usage', {}).get('cost', {})
                }
            else:
                logger.error(f"❌ Perplexity API error: {response.status_code} - {response.text}")
                return {
                    "status": "error",
                    "message": f"API error: {response.status_code}",
                    "symbol": symbol
                }
                
        except Exception as e:
            logger.error(f"❌ Error analyzing {symbol}: {e}")
            return {
                "status": "error", 
                "message": str(e),
                "symbol": symbol
            }
    
    def _create_financial_prompt(self, symbol: str, current_price: float) -> str:
        """Create comprehensive financial analysis prompt"""
        return f"""
You are an expert financial analyst. Provide a comprehensive analysis of {symbol} stock currently trading at ${current_price}.

Please analyze and provide specific data for:

1. **Technical Analysis**:
   - Current trend direction (bullish/bearish/neutral)
   - Support and resistance levels (specific prices)
   - RSI indicator interpretation
   - MACD signal (buy/sell/hold)
   - Key technical indicators

2. **Fundamental Analysis**:
   - P/E ratio and valuation metrics
   - Debt-to-equity ratio
   - Recent revenue growth trends
   - Profit margins and financial health
   - Competitive position in sector

3. **Sentiment & News Analysis**:
   - Recent news sentiment (positive/negative/neutral)
   - Social media sentiment trends
   - Analyst ratings and price targets
   - Market buzz and investor sentiment

4. **Risk Assessment**:
   - Volatility analysis (low/moderate/high)
   - Beta coefficient if available
   - Key risk factors and threats
   - Liquidity assessment

5. **AI Prediction**:
   - Direction forecast (up/down/sideways)
   - Confidence level (0-100%)
   - Target price estimate
   - Time horizon (1 month outlook)

Please search for the most recent financial data and news about {symbol}. Focus on actionable insights and specific numerical data where possible.

Format your response with clear sections and specific values that can be used for investment decisions.
"""
    
    def _parse_ai_response(self, ai_content: str, symbol: str, current_price: float) -> Dict:
        """
        Parse AI response into structured financial data
        This extracts key information from the AI's natural language response
        """
        try:
            # Initialize default structure
            analysis = {
                "technical": {
                    "trend": "neutral",
                    "support_level": round(current_price * 0.90, 2),
                    "resistance_level": round(current_price * 1.10, 2),
                    "rsi": 50,
                    "macd_signal": "hold"
                },
                "fundamental": {
                    "pe_ratio": "N/A",
                    "debt_to_equity": "N/A",
                    "revenue_growth": "N/A",
                    "profit_margin": "N/A"
                },
                "sentiment": {
                    "news_sentiment": 0.5,
                    "social_sentiment": 0.5,
                    "analyst_rating": "hold",
                    "price_target": current_price
                },
                "risk_assessment": {
                    "volatility": "moderate",
                    "liquidity": "high",
                    "beta": 1.0,
                    "risk_score": 0.5
                },
                "prediction": {
                    "direction": "sideways",
                    "confidence": 0.6,
                    "target_price": current_price,
                    "time_horizon": "1_month"
                }
            }
            
            # Try to extract specific information using basic text parsing
            content_lower = ai_content.lower()
            
            # Extract trend
            if "bullish" in content_lower or "uptrend" in content_lower:
                analysis["technical"]["trend"] = "bullish"
                analysis["prediction"]["direction"] = "up"
            elif "bearish" in content_lower or "downtrend" in content_lower:
                analysis["technical"]["trend"] = "bearish" 
                analysis["prediction"]["direction"] = "down"
            
            # Extract sentiment
            if "positive" in content_lower and "sentiment" in content_lower:
                analysis["sentiment"]["news_sentiment"] = 0.7
            elif "negative" in content_lower and "sentiment" in content_lower:
                analysis["sentiment"]["news_sentiment"] = 0.3
                
            # Extract analyst rating
            if "buy" in content_lower and ("rating" in content_lower or "recommend" in content_lower):
                analysis["sentiment"]["analyst_rating"] = "buy"
            elif "sell" in content_lower and ("rating" in content_lower or "recommend" in content_lower):
                analysis["sentiment"]["analyst_rating"] = "sell"
            
            # Extract volatility
            if "high volatility" in content_lower or "volatile" in content_lower:
                analysis["risk_assessment"]["volatility"] = "high"
                analysis["risk_assessment"]["risk_score"] = 0.7
            elif "low volatility" in content_lower or "stable" in content_lower:
                analysis["risk_assessment"]["volatility"] = "low"
                analysis["risk_assessment"]["risk_score"] = 0.3
            
            # Try to extract numerical values (basic regex patterns)
            import re
            
            # Look for support/resistance levels
            price_pattern = r'\$?(\d+\.?\d*)'
            support_matches = re.findall(rf'support.*?{price_pattern}', content_lower)
            if support_matches:
                try:
                    analysis["technical"]["support_level"] = float(support_matches[0])
                except:
                    pass
                    
            resistance_matches = re.findall(rf'resistance.*?{price_pattern}', content_lower)
            if resistance_matches:
                try:
                    analysis["technical"]["resistance_level"] = float(resistance_matches[0])
                except:
                    pass
            
            # Look for target price
            target_matches = re.findall(rf'target.*?{price_pattern}', content_lower)
            if target_matches:
                try:
                    target_price = float(target_matches[0])
                    analysis["sentiment"]["price_target"] = target_price
                    analysis["prediction"]["target_price"] = target_price
                except:
                    pass
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error parsing AI response: {e}")
            # Return default analysis if parsing fails
            return {
                "technical": {
                    "trend": "neutral",
                    "support_level": round(current_price * 0.90, 2),
                    "resistance_level": round(current_price * 1.10, 2),
                    "rsi": 50,
                    "macd_signal": "hold"
                },
                "prediction": {
                    "direction": "sideways", 
                    "confidence": 0.6,
                    "target_price": current_price,
                    "time_horizon": "1_month"
                },
                "error": "Failed to parse AI response"
            }

# Create global instance
perplexity_analyzer = PerplexityFinanceAnalyzer()