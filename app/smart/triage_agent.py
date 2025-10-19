"""
LLM-powered triage agent for final relevance determination and summarization
Uses OpenAI/Anthropic/Perplexity APIs for intelligent article analysis
"""
import logging
import json
from typing import Dict, Any, List, Optional
import openai
import anthropic
import httpx
from datetime import datetime
import os

logger = logging.getLogger(__name__)


class TriageAgent:
    """LLM-powered agent for article triage and summarization"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # Load config.yaml directly (more reliable than settings.py)
        import yaml
        
        config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                full_config = yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Could not load config.yaml: {e}, using defaults")
            full_config = {}
        
        # Get LLM config
        self.config = config if config else full_config.get("llm", {})
        
        # LLM configuration with fallbacks
        self.provider = self.config.get("provider", "perplexity")
        self.model = self.config.get("model", "llama-3.1-sonar-small-128k-online")
        self.temperature = self.config.get("temperature", 0.1)
        self.max_tokens = self.config.get("max_tokens", 300)
        self.timeout = self.config.get("timeout", 30)
        
        # API keys from environment (always preferred)
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        self.perplexity_api_key = os.getenv("PERPLEXITY_API_KEY")
        
        # Prompts
        self.triage_prompt = self.config.get("triage_prompt", self._get_default_prompt())
        
        # Initialize API clients
        self.openai_client = None
        self.anthropic_client = None
        self.perplexity_client = None
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize LLM API clients"""
        try:
            if self.provider == "openai":
                if self.openai_api_key:
                    self.openai_client = openai.OpenAI(api_key=self.openai_api_key)
                    logger.info("Initialized OpenAI client")
                else:
                    logger.warning("OpenAI API key not found")
            
            elif self.provider == "anthropic":
                if self.anthropic_api_key:
                    self.anthropic_client = anthropic.Anthropic(api_key=self.anthropic_api_key)
                    logger.info("Initialized Anthropic client")
                else:
                    logger.warning("Anthropic API key not found")
            
            elif self.provider == "perplexity":
                if self.perplexity_api_key:
                    self.perplexity_client = httpx.AsyncClient(
                        base_url="https://api.perplexity.ai",
                        headers={"Authorization": f"Bearer {self.perplexity_api_key}"},
                        timeout=self.timeout
                    )
                    logger.info("Initialized Perplexity client")
                else:
                    logger.warning("Perplexity API key not found")
                    
        except Exception as e:
            logger.error(f"Failed to initialize LLM clients: {e}")
    
    def _get_default_prompt(self) -> str:
        """Get default triage prompt"""
        return """You are analyzing news about US-China trade relations and tariffs.

Article Details:
Title: {title}
Source: {source}
Language: {language}
Content: {content}

Previous Analysis:
- Keywords Score: {keyword_score}
- Semantic Score: {semantic_score}
- Classifier Score: {classifier_score}

Your Task:
Determine if this article is relevant to US-China tariffs, trade war, countermeasures, or export controls.

Questions:
1. Is this relevant to US-China trade/tariff disputes? (YES/NO)
2. Provide a 2-3 sentence summary in Hebrew or English
3. Select relevant tags: Retaliation, Exemption, List_Update, 301_Tariffs, Semiconductors, EV, Solar, Steel, Export_Control, Negotiation, WTO, Investigation, Other
4. Confidence level (0.0-1.0)

Respond ONLY in valid JSON format:
{
    "relevant": true/false,
    "summary": "Brief summary here",
    "tags": ["tag1", "tag2"],
    "confidence": 0.95,
    "reasoning": "Brief explanation"
}"""
    
    async def analyze_article(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze article with LLM triage"""
        if not self._is_client_available():
            return self._get_fallback_analysis(article)
        
        try:
            # Prepare content for analysis
            content = self._prepare_content(article)
            
            # Get LLM response
            response = await self._call_llm(content)
            
            if response:
                # Parse and validate response
                analysis = self._parse_llm_response(response)
                
                # Add metadata
                analysis["llm_provider"] = self.provider
                analysis["llm_model"] = self.model
                analysis["analysis_timestamp"] = datetime.utcnow().isoformat()
                
                return analysis
            else:
                return self._get_fallback_analysis(article)
                
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            return self._get_fallback_analysis(article)
    
    def _prepare_content(self, article: Dict[str, Any]) -> str:
        """Prepare article content for LLM analysis"""
        title = article.get("title", "")[:200]  # Limit title length
        content = article.get("content", "")[:1500]  # Limit content length
        source = article.get("source_name", "")
        language = article.get("language", "unknown")
        
        # Get previous analysis scores
        keyword_score = article.get("keyword_score", 0)
        semantic_score = article.get("semantic_score", 0)
        classifier_score = article.get("classifier_score", 0)
        
        # Format prompt
        formatted_prompt = self.triage_prompt.format(
            title=title,
            source=source,
            language=language,
            content=content,
            keyword_score=keyword_score,
            semantic_score=semantic_score,
            classifier_score=classifier_score
        )
        
        return formatted_prompt
    
    async def _call_llm(self, prompt: str) -> Optional[str]:
        """Call LLM API and get response"""
        try:
            if self.provider == "openai" and self.openai_client:
                response = self.openai_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a trade policy analyst specializing in US-China relations."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    timeout=self.timeout
                )
                return response.choices[0].message.content
                
            elif self.provider == "anthropic" and self.anthropic_client:
                response = self.anthropic_client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                return response.content[0].text
            
            elif self.provider == "perplexity" and self.perplexity_client:
                # Perplexity API call
                response = await self.perplexity_client.post(
                    "/chat/completions",
                    json={
                        "model": self.model or "llama-3.1-sonar-small-128k-online",
                        "messages": [
                            {"role": "system", "content": "You are a trade policy analyst specializing in US-China relations."},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": self.temperature,
                        "max_tokens": self.max_tokens
                    }
                )
                response.raise_for_status()
                result = response.json()
                return result["choices"][0]["message"]["content"]
                
        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            return None
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse and validate LLM JSON response"""
        try:
            # Clean response (remove markdown formatting if present)
            cleaned_response = response.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]
            cleaned_response = cleaned_response.strip()
            
            # Parse JSON
            parsed = json.loads(cleaned_response)
            
            # Validate required fields and types
            result = {
                "llm_relevant": bool(parsed.get("relevant", False)),
                "llm_summary": str(parsed.get("summary", ""))[:500],  # Limit summary length
                "llm_tags": self._validate_tags(parsed.get("tags", [])),
                "llm_confidence": float(max(0.0, min(1.0, parsed.get("confidence", 0.5)))),
                "llm_reasoning": str(parsed.get("reasoning", ""))[:200]
            }
            
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM JSON response: {e}")
            logger.debug(f"Raw response: {response}")
            return self._extract_fallback_from_text(response)
        except Exception as e:
            logger.error(f"Error processing LLM response: {e}")
            return self._get_default_llm_analysis()
    
    def _validate_tags(self, tags: List[str]) -> List[str]:
        """Validate and filter tags"""
        valid_tags = {
            "Retaliation", "Exemption", "List_Update", "301_Tariffs", 
            "Semiconductors", "EV", "Solar", "Steel", "Export_Control",
            "Negotiation", "WTO", "Investigation", "Other"
        }
        
        validated = []
        for tag in tags[:5]:  # Limit to 5 tags
            if isinstance(tag, str) and tag in valid_tags:
                validated.append(tag)
        
        return validated
    
    def _extract_fallback_from_text(self, response: str) -> Dict[str, Any]:
        """Try to extract information from non-JSON response"""
        try:
            # Look for YES/NO in response
            response_lower = response.lower()
            relevant = "yes" in response_lower and "no" not in response_lower
            
            # Extract confidence if mentioned
            import re
            confidence_match = re.search(r'confidence[:\s]*([0-9.]+)', response_lower)
            confidence = float(confidence_match.group(1)) if confidence_match else 0.5
            
            # Extract summary (first sentence or paragraph)
            sentences = response.split('.')
            summary = sentences[0][:200] if sentences else ""
            
            return {
                "llm_relevant": relevant,
                "llm_summary": summary,
                "llm_tags": ["Other"],
                "llm_confidence": confidence,
                "llm_reasoning": "Fallback text extraction"
            }
            
        except Exception as e:
            logger.error(f"Fallback extraction failed: {e}")
            return self._get_default_llm_analysis()
    
    def _get_default_llm_analysis(self) -> Dict[str, Any]:
        """Get default analysis when LLM fails"""
        return {
            "llm_relevant": None,
            "llm_summary": "",
            "llm_tags": [],
            "llm_confidence": 0.0,
            "llm_reasoning": "LLM analysis failed"
        }
    
    def _get_fallback_analysis(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """Get rule-based fallback analysis when LLM unavailable"""
        # Simple rule-based analysis
        title = article.get("title", "").lower()
        content = article.get("content", "").lower()
        text = f"{title} {content}"
        
        # Check for key indicators
        tariff_indicators = ["关税", "tariff", "税则", "customs", "duties"]
        trade_indicators = ["贸易", "trade", "商务", "commerce", "经贸"]
        
        has_tariff = any(indicator in text for indicator in tariff_indicators)
        has_trade = any(indicator in text for indicator in trade_indicators)
        
        relevant = has_tariff or has_trade
        confidence = 0.7 if relevant else 0.3
        
        # Determine tags based on keywords
        tags = []
        if "关税" in text or "tariff" in text:
            tags.append("301_Tariffs")
        if "半导体" in text or "semiconductor" in text:
            tags.append("Semiconductors")
        if "反制" in text or "retaliation" in text:
            tags.append("Retaliation")
        
        if not tags:
            tags = ["Other"]
        
        return {
            "llm_relevant": relevant,
            "llm_summary": f"Rule-based analysis: {'Relevant' if relevant else 'Not relevant'} to trade/tariffs",
            "llm_tags": tags,
            "llm_confidence": confidence,
            "llm_reasoning": "Rule-based fallback (LLM unavailable)",
            "llm_provider": "fallback",
            "llm_model": "rule_based",
            "analysis_timestamp": datetime.utcnow().isoformat()
        }
    
    def _is_client_available(self) -> bool:
        """Check if LLM client is available"""
        if self.provider == "openai":
            return self.openai_client is not None
        elif self.provider == "anthropic":
            return self.anthropic_client is not None
        elif self.provider == "perplexity":
            return self.perplexity_client is not None
        return False
    
    def calculate_final_score(self, article: Dict[str, Any]) -> float:
        """Calculate final relevance score combining all analyses"""
        try:
            # Get individual scores
            keyword_score = article.get("keyword_score", 0)
            semantic_score = article.get("semantic_score", 0) 
            classifier_score = article.get("classifier_score", 0)
            llm_confidence = article.get("llm_confidence", 0)
            llm_relevant = article.get("llm_relevant", False)
            
            # Normalize scores to 0-1 range
            keyword_norm = min(1.0, keyword_score / 10.0)  # Assuming max keyword score ~10
            semantic_norm = semantic_score  # Already 0-1
            classifier_norm = classifier_score  # Already 0-1
            llm_norm = llm_confidence if llm_relevant else llm_confidence * 0.3
            
            # Weighted combination
            weights = {
                "keyword": 0.2,
                "semantic": 0.3,
                "classifier": 0.2,
                "llm": 0.3
            }
            
            final_score = (
                keyword_norm * weights["keyword"] +
                semantic_norm * weights["semantic"] +
                classifier_norm * weights["classifier"] +
                llm_norm * weights["llm"]
            )
            
            # Source priority boost
            source_priority = article.get("source_priority", "medium")
            if source_priority == "high":
                final_score *= 1.2
            elif source_priority == "low":
                final_score *= 0.8
            
            # Government document boost
            metadata = article.get("metadata", {})
            if metadata.get("is_government_document", False):
                final_score *= 1.1
            
            return min(1.0, final_score)
            
        except Exception as e:
            logger.error(f"Failed to calculate final score: {e}")
            return 0.0
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about the triage agent"""
        return {
            "provider": self.provider,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "client_available": self._is_client_available(),
            "fallback_enabled": True
        }


def triage_articles(articles: List[Dict[str, Any]], config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Run triage analysis on articles"""
    agent = TriageAgent(config)
    
    for article in articles:
        try:
            # Run LLM triage
            llm_analysis = agent.analyze_article(article)
            article.update(llm_analysis)
            
            # Calculate final score
            final_score = agent.calculate_final_score(article)
            article["final_score"] = final_score
            
        except Exception as e:
            logger.error(f"Failed to triage article: {e}")
            # Set default values
            article.update({
                "llm_relevant": None,
                "llm_summary": "",
                "llm_tags": [],
                "llm_confidence": 0.0,
                "final_score": 0.0
            })
    
    return articles


def main():
    """Test the triage agent"""
    config = {
        "llm": {
            "provider": "openai",
            "model": "gpt-3.5-turbo",
            "temperature": 0.1,
            "max_tokens": 300
        }
    }
    
    test_article = {
        "title": "国务院关税税则委员会发布对美加征关税商品清单",
        "content": "根据《中华人民共和国对外贸易法》等法律法规和《国务院关税税则委员会关于对美加征关税的公告》，国务院关税税则委员会决定，对美国进口的部分商品加征关税。",
        "source_name": "MOFCOM",
        "language": "zh",
        "keyword_score": 8.5,
        "semantic_score": 0.92,
        "classifier_score": 0.88
    }
    
    agent = TriageAgent(config)
    
    print("Agent info:")
    print(agent.get_agent_info())
    print()
    
    print("Analyzing article...")
    result = agent.analyze_article(test_article)
    
    for key, value in result.items():
        print(f"{key}: {value}")
    
    print(f"\nFinal score: {agent.calculate_final_score({**test_article, **result}):.3f}")


if __name__ == "__main__":
    main()