"""
Financial Keywords Engine for MarketPulse
Analyzes articles for bullish/bearish sentiment using keyword matching
Implements scoring system from keywords.yaml
"""
import re
import logging
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
import yaml
import math

logger = logging.getLogger(__name__)


@dataclass
class KeywordMatch:
    """Represents a matched keyword with context"""
    keyword: str
    category: str  # very_bullish, bullish, bearish, very_bearish
    base_score: float  # +3, +2, -2, -3
    position: int
    context: str
    in_title: bool = False


class FinancialKeywordsEngine:
    """Keywords-based sentiment analysis and scoring for financial articles"""
    
    def __init__(self, keywords_config_path: str = "keywords.yaml"):
        """Initialize with keywords configuration"""
        try:
            with open(keywords_config_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Keywords config not found at {keywords_config_path}, using defaults")
            self.config = self._get_default_config()
        
        # Load keyword categories
        self.keywords = self.config.get("keywords", {})
        self.scoring_rules = self.config.get("scoring", {})
        self.alert_thresholds = self.config.get("alert_thresholds", {})
        
        # Compile regex patterns for fast matching
        self.patterns = self._compile_all_patterns()
        
        # Source weights (from data_sources.yaml if available)
        self.source_weights = {
            "Reuters": 1.0,
            "Bloomberg": 1.0,
            "WSJ": 1.0,
            "Financial Times": 1.0,
            "CNBC": 1.1,  # Breaking news
            "SEC EDGAR": 1.2,  # Regulatory
            "FDA": 1.3,  # High impact
            "Perplexity AI": 1.0,
            "Twitter": 0.6,  # Social media
            "Reddit": 0.5
        }
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Default keywords configuration if file not found"""
        return {
            "keywords": {
                "very_bullish": {
                    "score": 3,
                    "keywords": ["FDA approval", "record earnings", "beat expectations", 
                                "breakthrough", "blockbuster drug"]
                },
                "bullish": {
                    "score": 2,
                    "keywords": ["upgrade", "strong demand", "exceeds forecast", "growth", "partnership"]
                },
                "neutral": {
                    "score": 0,
                    "keywords": ["announced", "reported", "statement", "filed"]
                },
                "bearish": {
                    "score": -2,
                    "keywords": ["downgrade", "miss expectations", "lowered guidance", "decline", "weak"]
                },
                "very_bearish": {
                    "score": -3,
                    "keywords": ["bankruptcy", "fraud", "investigation", "recall", "scandal"]
                }
            },
            "scoring": {
                "title_amplifier": 1.5,
                "time_decay_hours": 24,
                "decay_factor": 0.5
            },
            "alert_thresholds": {
                "critical": 2.5,
                "important": 1.5,
                "watch": 0.8
            }
        }
    
    def _compile_all_patterns(self) -> Dict[str, List[Tuple[re.Pattern, float]]]:
        """Compile regex patterns for all keyword categories"""
        patterns = {}
        
        for category, config in self.keywords.items():
            # Skip if not a dict (could be metadata)
            if not isinstance(config, dict):
                continue
            
            score = config.get("score", 0)
            keywords_list = config.get("keywords", [])
            
            # Skip if it's not a keyword category
            if not keywords_list:
                continue
            
            patterns[category] = []
            for keyword in keywords_list:
                # Case-insensitive, word boundary matching
                pattern = re.compile(r'\b' + re.escape(keyword) + r'\b', re.IGNORECASE)
                patterns[category].append((pattern, score))
        
        return patterns
    
    def analyze_article(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze article and return comprehensive keyword analysis
        
        Returns:
            {
                "keyword_score": float,  # Final calculated score
                "sentiment": str,  # "very_bullish", "bullish", "neutral", "bearish", "very_bearish"
                "alert_level": str,  # "critical", "important", "watch", "none"
                "keyword_matches": List[dict],  # All matched keywords
                "category_breakdown": dict,  # Score by category
                "confidence": float  # 0-1, based on number of matches
            }
        """
        title = article.get("title", "")
        content = article.get("content", "")
        source_name = article.get("source_name", "Unknown")
        published_at = article.get("published_at")
        
        # Find all keyword matches
        matches = []
        
        # Check title (higher weight)
        title_matches = self._find_matches_in_text(title, in_title=True)
        matches.extend(title_matches)
        
        # Check content
        content_matches = self._find_matches_in_text(content, in_title=False)
        matches.extend(content_matches)
        
        # Calculate scores
        base_score = self._calculate_base_score(matches)
        source_weight = self.source_weights.get(source_name, 1.0)
        time_weight = self._calculate_time_weight(published_at)
        
        # Final score formula: base_score × source_weight × time_weight
        final_score = base_score * source_weight * time_weight
        
        # Determine sentiment
        sentiment = self._score_to_sentiment(final_score)
        
        # Determine alert level
        alert_level = self._score_to_alert_level(abs(final_score))
        
        # Calculate confidence (0-1) based on number of matches
        confidence = min(1.0, len(matches) / 5.0)  # 5+ matches = 100% confidence
        
        # Category breakdown
        category_breakdown = self._get_category_breakdown(matches)
        
        return {
            "keyword_score": round(final_score, 2),
            "base_score": round(base_score, 2),
            "source_weight": source_weight,
            "time_weight": round(time_weight, 2),
            "sentiment": sentiment,
            "alert_level": alert_level,
            "confidence": round(confidence, 2),
            "keyword_matches": [
                {
                    "keyword": match.keyword,
                    "category": match.category,
                    "score": match.base_score,
                    "in_title": match.in_title,
                    "context": match.context[:100]
                }
                for match in matches
            ],
            "category_breakdown": category_breakdown,
            "total_matches": len(matches),
            "title_matches": len(title_matches),
            "content_matches": len(content_matches)
        }
    
    def _find_matches_in_text(self, text: str, in_title: bool = False) -> List[KeywordMatch]:
        """Find all keyword matches in a piece of text"""
        matches = []
        
        for category, patterns_with_scores in self.patterns.items():
            for pattern, base_score in patterns_with_scores:
                for match in pattern.finditer(text):
                    # Extract context (50 chars before and after)
                    start = max(0, match.start() - 50)
                    end = min(len(text), match.end() + 50)
                    context = text[start:end].strip()
                    
                    keyword_match = KeywordMatch(
                        keyword=match.group(),
                        category=category,
                        base_score=base_score,
                        position=match.start(),
                        context=context,
                        in_title=in_title
                    )
                    matches.append(keyword_match)
        
        return matches
    
    def _calculate_base_score(self, matches: List[KeywordMatch]) -> float:
        """Calculate base score from all matches"""
        if not matches:
            return 0.0
        
        total_score = 0.0
        title_amplifier = self.scoring_rules.get("title_amplifier", 1.5)
        
        for match in matches:
            score = match.base_score
            
            # Apply title amplifier
            if match.in_title:
                score *= title_amplifier
            
            total_score += score
        
        return total_score
    
    def _calculate_time_weight(self, published_at: Optional[str]) -> float:
        """Calculate time decay weight (newer = higher weight)"""
        if not published_at:
            return 1.0
        
        try:
            # Parse timestamp
            if isinstance(published_at, str):
                pub_time = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
            else:
                pub_time = published_at
            
            # Calculate age in hours
            now = datetime.now(timezone.utc)
            age_hours = (now - pub_time).total_seconds() / 3600
            
            # Apply time decay
            decay_hours = self.scoring_rules.get("time_decay_hours", 24)
            decay_factor = self.scoring_rules.get("decay_factor", 0.5)
            
            if age_hours < decay_hours:
                # Linear decay over decay_hours period
                weight = 1.0 - (age_hours / decay_hours) * (1.0 - decay_factor)
            else:
                # Minimum weight after decay period
                weight = decay_factor
            
            return weight
            
        except Exception as e:
            logger.debug(f"Failed to calculate time weight: {e}")
            return 1.0
    
    def _score_to_sentiment(self, score: float) -> str:
        """Convert numerical score to sentiment label"""
        if score >= 2.5:
            return "very_bullish"
        elif score >= 1.0:
            return "bullish"
        elif score <= -2.5:
            return "very_bearish"
        elif score <= -1.0:
            return "bearish"
        else:
            return "neutral"
    
    def _score_to_alert_level(self, abs_score: float) -> str:
        """Convert absolute score to alert level"""
        thresholds = self.alert_thresholds
        
        if abs_score >= thresholds.get("critical", 2.5):
            return "critical"
        elif abs_score >= thresholds.get("important", 1.5):
            return "important"
        elif abs_score >= thresholds.get("watch", 0.8):
            return "watch"
        else:
            return "none"
    
    def _get_category_breakdown(self, matches: List[KeywordMatch]) -> Dict[str, Any]:
        """Get score breakdown by category"""
        breakdown = {}
        
        for match in matches:
            category = match.category
            if category not in breakdown:
                breakdown[category] = {
                    "count": 0,
                    "score": 0.0
                }
            
            breakdown[category]["count"] += 1
            breakdown[category]["score"] += match.base_score
        
        return breakdown
    
    def filter_articles_by_threshold(self, articles: List[Dict[str, Any]], 
                                    min_alert_level: str = "watch") -> List[Dict[str, Any]]:
        """Filter articles that meet minimum alert threshold"""
        alert_priority = {"critical": 3, "important": 2, "watch": 1, "none": 0}
        min_priority = alert_priority.get(min_alert_level, 0)
        
        filtered = []
        for article in articles:
            # Analyze if not already analyzed
            if "keyword_score" not in article:
                analysis = self.analyze_article(article)
                article.update(analysis)
            
            article_priority = alert_priority.get(article.get("alert_level", "none"), 0)
            
            if article_priority >= min_priority:
                filtered.append(article)
        
        logger.info(f"Filtered {len(filtered)}/{len(articles)} articles meeting '{min_alert_level}' threshold")
        return filtered
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about keywords configuration"""
        total_keywords = sum(
            len(config.get("keywords", []))
            for cat, config in self.keywords.items()
            if isinstance(config, dict) and "keywords" in config
        )
        
        return {
            "total_keywords": total_keywords,
            "categories": [k for k, v in self.keywords.items() if isinstance(v, dict) and "keywords" in v],
            "alert_thresholds": self.alert_thresholds,
            "scoring_rules": self.scoring_rules,
            "source_weights": self.source_weights
        }
    
    def explain_score(self, article: Dict[str, Any]) -> str:
        """Generate human-readable explanation of score"""
        analysis = self.analyze_article(article)
        
        explanation = f"""
Keyword Analysis for: {article.get('title', 'Unknown')[:80]}
{'=' * 80}

Final Score: {analysis['keyword_score']} ({analysis['sentiment']})
Alert Level: {analysis['alert_level']}
Confidence: {analysis['confidence'] * 100:.0f}%

Score Breakdown:
- Base Score: {analysis['base_score']}
- Source Weight: {analysis['source_weight']} ({article.get('source_name', 'Unknown')})
- Time Weight: {analysis['time_weight']}

Matched Keywords ({analysis['total_matches']} total):
"""
        
        for i, match in enumerate(analysis['keyword_matches'][:10], 1):
            location = "TITLE" if match['in_title'] else "content"
            explanation += f"  {i}. '{match['keyword']}' ({match['category']}, score: {match['score']:+.1f}) [{location}]\n"
        
        if len(analysis['keyword_matches']) > 10:
            explanation += f"  ... and {len(analysis['keyword_matches']) - 10} more\n"
        
        return explanation


def analyze_articles_batch(articles: List[Dict[str, Any]], 
                           keywords_engine: Optional[FinancialKeywordsEngine] = None) -> List[Dict[str, Any]]:
    """Analyze a batch of articles and add keyword analysis"""
    if keywords_engine is None:
        keywords_engine = FinancialKeywordsEngine()
    
    analyzed_articles = []
    
    for article in articles:
        try:
            analysis = keywords_engine.analyze_article(article)
            article.update(analysis)
            analyzed_articles.append(article)
            
        except Exception as e:
            logger.error(f"Failed to analyze article: {e}")
            continue
    
    # Sort by score (highest absolute value first)
    analyzed_articles.sort(key=lambda x: abs(x.get("keyword_score", 0)), reverse=True)
    
    logger.info(f"✅ Analyzed {len(analyzed_articles)} articles")
    return analyzed_articles


def main():
    """Test the financial keywords engine"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Test articles
    test_articles = [
        {
            "title": "AAPL beats earnings expectations with record iPhone sales",
            "content": "Apple announced breakthrough quarterly results, showing strong demand across all product lines. The company beat analyst expectations and raised guidance.",
            "source_name": "Bloomberg",
            "published_at": datetime.now(timezone.utc).isoformat(),
            "symbols": ["AAPL"]
        },
        {
            "title": "FDA approves Pfizer's blockbuster drug for cancer treatment",
            "content": "The FDA today approved Pfizer's breakthrough cancer drug after successful clinical trials. This is expected to be a major revenue driver.",
            "source_name": "FDA",
            "published_at": datetime.now(timezone.utc).isoformat(),
            "symbols": ["PFE"]
        },
        {
            "title": "Tesla faces investigation over autopilot safety concerns",
            "content": "Federal regulators have opened an investigation into Tesla's autopilot system following multiple incidents. The company's stock declined on the news.",
            "source_name": "Reuters",
            "published_at": (datetime.now(timezone.utc) - timedelta(hours=5)).isoformat(),
            "symbols": ["TSLA"]
        },
        {
            "title": "GOOGL announces quarterly dividend",
            "content": "Google parent Alphabet announced its regular quarterly dividend payment to shareholders.",
            "source_name": "CNBC",
            "published_at": datetime.now(timezone.utc).isoformat(),
            "symbols": ["GOOGL"]
        }
    ]
    
    # Initialize engine
    engine = FinancialKeywordsEngine()
    
    print("\n" + "=" * 80)
    print("FINANCIAL KEYWORDS ENGINE TEST")
    print("=" * 80)
    
    # Show statistics
    stats = engine.get_statistics()
    print(f"\n📊 Engine Statistics:")
    print(f"   Total keywords: {stats['total_keywords']}")
    print(f"   Categories: {', '.join(stats['categories'])}")
    print(f"   Alert thresholds: {stats['alert_thresholds']}")
    
    # Analyze each article
    print(f"\n🔍 Analyzing {len(test_articles)} test articles...\n")
    
    analyzed = analyze_articles_batch(test_articles, engine)
    
    for article in analyzed:
        print("\n" + "-" * 80)
        print(f"📰 {article['title']}")
        print(f"   Score: {article['keyword_score']:+.2f} | Sentiment: {article['sentiment']} | Alert: {article['alert_level']}")
        print(f"   Confidence: {article['confidence'] * 100:.0f}% | Matches: {article['total_matches']}")
        print(f"   Source: {article['source_name']} (weight: {article['source_weight']})")
        
        if article['keyword_matches']:
            print(f"   Keywords matched:")
            for match in article['keyword_matches'][:5]:
                location = "TITLE" if match['in_title'] else "content"
                print(f"      • '{match['keyword']}' ({match['category']}, {match['score']:+.1f}) [{location}]")
    
    # Show filtered results
    print("\n" + "=" * 80)
    print("🚨 CRITICAL ALERTS")
    print("=" * 80)
    critical = engine.filter_articles_by_threshold(analyzed, "critical")
    for article in critical:
        print(f"   🔴 {article['title'][:70]} (score: {article['keyword_score']:+.2f})")
    
    print("\n" + "=" * 80)
    print("🟡 IMPORTANT ALERTS")
    print("=" * 80)
    important = engine.filter_articles_by_threshold(analyzed, "important")
    for article in important:
        if article['alert_level'] == "important":  # Exclude criticals
            print(f"   🟡 {article['title'][:70]} (score: {article['keyword_score']:+.2f})")


if __name__ == "__main__":
    main()
