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

    async def analyze_async(self, symbol: str, user_query: str, analysis_type: str = "generic") -> Dict:
        """
        Lightweight async analysis for scheduler/event-driven use.

        Returns a simplified schema expected by the scheduler:
        - analysis: str (full AI text)
        - recommendation: str (best-effort extraction)
        - confidence: float (0.0-1.0, heuristic)
        - citations: list (from API if available)

        This method intentionally avoids heavy parsing and is resilient to
        schema changes from the upstream API.
        """
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            data = {
                "model": self.model,
                "messages": [
                    {"role": "user", "content": user_query}
                ],
                "max_tokens": 1200,
                "temperature": 0.2,
            }

            logger.info(f"🤖 Perplexity async analysis for {symbol} ({analysis_type})...")
            resp = requests.post(self.base_url, json=data, headers=headers, timeout=60)
            if resp.status_code != 200:
                logger.error(f"❌ Perplexity API error ({resp.status_code}): {resp.text[:200]}")
                return {
                    "analysis": "",
                    "recommendation": "",
                    "confidence": 0.0,
                    "citations": [],
                }

            result = resp.json()
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            citations = result.get("citations", []) or []

            # Best-effort recommendation extraction
            recommendation = ""
            try:
                import re
                match = re.search(r"\b(RECOMMENDATION|Action)\s*:\s*(BUY|SELL|HOLD)\b", content, re.I)
                if match:
                    recommendation = match.group(2).upper()
            except Exception:
                pass

            # Heuristic confidence extraction (0-1)
            confidence = 0.0
            try:
                import re
                m2 = re.search(r"CONFIDENCE\s*[:=]\s*(\d{1,3})%", content, re.I)
                if m2:
                    pct = float(m2.group(1))
                    confidence = max(0.0, min(1.0, pct / 100.0))
            except Exception:
                pass

            return {
                "analysis": content,
                "recommendation": recommendation,
                "confidence": confidence,
                "citations": citations,
            }
        except Exception as e:
            logger.error(f"❌ Perplexity async analysis error: {e}")
            return {
                "analysis": "",
                "recommendation": "",
                "confidence": 0.0,
                "citations": [],
            }
    
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
            logger.info(f"📤 PROMPT SENT:\n{prompt}")
            logger.info(f"🔧 REQUEST DATA: {json.dumps(data, indent=2)}")
            
            response = requests.post(
                self.base_url, 
                json=data, 
                headers=headers, 
                timeout=60
            )
            
            logger.info(f"📥 RESPONSE STATUS: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                ai_content = result['choices'][0]['message']['content']
                
                logger.info(f"✅ PERPLEXITY RESPONSE:\n{ai_content}")
                logger.info(f"📊 CITATIONS: {len(result.get('citations', []))} sources")
                logger.info(f"🔍 SEARCH RESULTS: {len(result.get('search_results', []))} results")
                
                # Parse the AI response into structured data
                parsed_analysis = self._parse_ai_response(ai_content, symbol, current_price)
                
                logger.info(f"🎯 PARSED ANALYSIS: {json.dumps(parsed_analysis, indent=2)}")
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
Analyze the stock {symbol} using all accessible financial data within the Perplexity API. 
Include current and historical price performance, financial metrics, SEC filings, analyst sentiment, and relevant macroeconomic indicators. 
Incorporate market volatility (VIX), sector trends, and interest rate environment into the evaluation.

Current trading price: ${current_price}

IMPORTANT: Please structure your response in the following format, providing specific numerical data and clear reasoning:

**TECHNICAL ANALYSIS**
Trend: [bullish/bearish/neutral] - [reasoning with specific indicators]
Support Level: $[price] - [basis for this level]
Resistance Level: $[price] - [basis for this level]
RSI: [value] - [interpretation]
MACD Signal: [buy/sell/hold] - [current values and reasoning]
Volume Analysis: [description of recent volume patterns and momentum]

**FUNDAMENTAL ANALYSIS**
P/E Ratio: [value] vs Sector Average: [value]
Price-to-Book: [value] - [valuation assessment]
Debt-to-Equity: [value] - [financial health assessment]
Revenue Growth: [quarterly %] (latest quarter), [annual %] (year-over-year)
Profit Margins: Gross [%], Operating [%], Net [%] - [trend analysis]
Free Cash Flow: $[amount] - [cash position strength]
ROE: [%], ROA: [%] - [efficiency analysis]

**MARKET CONTEXT & MACROECONOMIC FACTORS**
Interest Rate Impact: [analysis of current environment effect]
Sector Performance: [relative to broader market with %]
VIX Impact: [current level and volatility considerations]
Economic Indicators: [relevant sector-specific factors]
Currency Impact: [if applicable for international exposure]

**SENTIMENT & NEWS ANALYSIS**
News Sentiment: [positive/negative/neutral] - [recent developments]
Analyst Ratings: [consensus rating and average price target]
Social Media Sentiment: [trend analysis]
Insider Activity: [recent trading patterns if available]
Institutional Changes: [ownership trends]

**RISK ASSESSMENT**
Volatility: Beta [value], 30-day volatility [%]
Business Risks: [key operational and market risks]
Regulatory Risks: [industry-specific considerations]
Liquidity: [trading volume and market depth]
Overall Risk: [low/moderate/high] - [explanation]

**INVESTMENT RECOMMENDATION**
Direction: [up/down/sideways] with [probability %]
Confidence Level: [0-100%] - [basis for confidence]
Target Price (1-month): $[price] - [rationale and method]
Key Catalysts: [upcoming events/factors that could move stock]
Stop-Loss: $[price], Take-Profit: $[price] - [if applicable]

Please search for the most recent SEC filings, earnings reports, analyst notes, and relevant news about {symbol}. 
Provide specific numerical values wherever possible and cite your data sources.
Use the most current data available and ensure all analysis is actionable for investment decisions.
"""
    
    def _parse_ai_response(self, ai_content: str, symbol: str, current_price: float) -> Dict:
        """
        Parse AI response into structured financial data
        Enhanced parser for the new structured format
        """
        try:
            import re
            
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
            
            # Parse structured response sections
            content = ai_content.upper()
            
            # Extract Technical Analysis
            tech_section = self._extract_section(ai_content, "TECHNICAL ANALYSIS")
            if tech_section:
                # Trend
                if "BULLISH" in tech_section:
                    analysis["technical"]["trend"] = "bullish"
                    analysis["prediction"]["direction"] = "up"
                elif "BEARISH" in tech_section:
                    analysis["technical"]["trend"] = "bearish"
                    analysis["prediction"]["direction"] = "down"
                
                # Extract numerical values
                support = self._extract_price(tech_section, r"SUPPORT LEVEL:\s*\$?(\d+\.?\d*)")
                if support:
                    analysis["technical"]["support_level"] = support
                    
                resistance = self._extract_price(tech_section, r"RESISTANCE LEVEL:\s*\$?(\d+\.?\d*)")
                if resistance:
                    analysis["technical"]["resistance_level"] = resistance
                    
                rsi = self._extract_number(tech_section, r"RSI:\s*(\d+\.?\d*)")
                if rsi:
                    analysis["technical"]["rsi"] = rsi
                    
                if "BUY" in tech_section:
                    analysis["technical"]["macd_signal"] = "buy"
                elif "SELL" in tech_section:
                    analysis["technical"]["macd_signal"] = "sell"
            
            # Extract Fundamental Analysis
            fund_section = self._extract_section(ai_content, "FUNDAMENTAL ANALYSIS")
            if fund_section:
                pe_ratio = self._extract_number(fund_section, r"P/E RATIO:\s*(\d+\.?\d*)")
                if pe_ratio:
                    analysis["fundamental"]["pe_ratio"] = pe_ratio
                    
                debt_eq = self._extract_number(fund_section, r"DEBT-TO-EQUITY:\s*(\d+\.?\d*)")
                if debt_eq:
                    analysis["fundamental"]["debt_to_equity"] = debt_eq
                    
                revenue_growth = self._extract_number(fund_section, r"REVENUE GROWTH:\s*(\d+\.?\d*)%?")
                if revenue_growth:
                    analysis["fundamental"]["revenue_growth"] = f"{revenue_growth}%"
            
            # Extract Investment Recommendation
            rec_section = self._extract_section(ai_content, "INVESTMENT RECOMMENDATION")
            if rec_section:
                if "UP" in rec_section:
                    analysis["prediction"]["direction"] = "up"
                elif "DOWN" in rec_section:
                    analysis["prediction"]["direction"] = "down"
                    
                confidence = self._extract_number(rec_section, r"CONFIDENCE LEVEL:\s*(\d+\.?\d*)%?")
                if confidence:
                    analysis["prediction"]["confidence"] = confidence / 100
                    
                target = self._extract_price(rec_section, r"TARGET PRICE.*?\$?(\d+\.?\d*)")
                if target:
                    analysis["prediction"]["target_price"] = target
                    analysis["sentiment"]["price_target"] = target
            
            # Extract Risk Assessment
            risk_section = self._extract_section(ai_content, "RISK ASSESSMENT")
            if risk_section:
                if "HIGH" in risk_section and "RISK" in risk_section:
                    analysis["risk_assessment"]["volatility"] = "high"
                    analysis["risk_assessment"]["risk_score"] = 0.8
                elif "LOW" in risk_section and "RISK" in risk_section:
                    analysis["risk_assessment"]["volatility"] = "low"
                    analysis["risk_assessment"]["risk_score"] = 0.3
                    
                beta = self._extract_number(risk_section, r"BETA\s*(\d+\.?\d*)")
                if beta:
                    analysis["risk_assessment"]["beta"] = beta
            
            # Extract Sentiment
            sent_section = self._extract_section(ai_content, "SENTIMENT & NEWS ANALYSIS")
            if sent_section:
                if "POSITIVE" in sent_section:
                    analysis["sentiment"]["news_sentiment"] = 0.7
                elif "NEGATIVE" in sent_section:
                    analysis["sentiment"]["news_sentiment"] = 0.3
                    
                if "BUY" in sent_section and "RATING" in sent_section:
                    analysis["sentiment"]["analyst_rating"] = "buy"
                elif "SELL" in sent_section and "RATING" in sent_section:
                    analysis["sentiment"]["analyst_rating"] = "sell"
            
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
    
    def _extract_section(self, content: str, section_name: str) -> str:
        """Extract a specific section from the structured response"""
        try:
            import re
            # Look for section header followed by content until next section or end
            pattern = rf'\*\*{section_name}\*\*(.*?)(?=\*\*[A-Z\s&]+\*\*|$)'
            match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
            return match.group(1).strip() if match else ""
        except:
            return ""
    
    def _extract_price(self, text: str, pattern: str) -> float:
        """Extract price value from text using regex pattern"""
        try:
            import re
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return float(match.group(1))
        except:
            pass
        return None
    
    def _extract_number(self, text: str, pattern: str) -> float:
        """Extract numerical value from text using regex pattern"""
        try:
            import re
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return float(match.group(1))
        except:
            pass
        return None

# Create global instance
perplexity_analyzer = PerplexityFinanceAnalyzer()