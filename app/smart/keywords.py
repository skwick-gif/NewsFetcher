"""
Keywords-based filtering system for initial article screening
Uses Chinese and English keywords with fuzzy matching and scoring
"""
import re
import logging
from typing import Dict, Any, List, Tuple, Set
from dataclasses import dataclass
import yaml

logger = logging.getLogger(__name__)


@dataclass
class KeywordMatch:
    """Represents a keyword match with context"""
    keyword: str
    category: str
    weight: float
    position: int
    context: str


class KeywordFilter:
    """Keywords-based article filtering and scoring"""
    
    def __init__(self, config: Dict[str, Any]):
        self.keywords_config = config.get("keywords", {})
        self.min_score = config.get("thresholds", {}).get("keyword_min", 1)
        
        # Load keyword dictionaries
        self.chinese_keywords = self._load_chinese_keywords()
        self.english_keywords = self._load_english_keywords()
        
        # Compile regex patterns for faster matching
        self.chinese_patterns = self._compile_patterns(self.chinese_keywords)
        self.english_patterns = self._compile_patterns(self.english_keywords)
        
        # Weights for different keyword categories
        self.category_weights = {
            "primary": 3.0,      # Core tariff/trade terms
            "secondary": 2.0,    # Related trade terms 
            "control": 2.5,      # Export control terms
            "sectors": 1.5,      # Specific sectors/products
            "organizations": 2.0, # Government orgs
            "indicators": 1.0    # Document type indicators
        }
    
    def _load_chinese_keywords(self) -> Dict[str, List[str]]:
        """Load Chinese keyword categories"""
        return {
            "primary": self.keywords_config.get("chinese", {}).get("primary", [
                "关税", "加征关税", "关税清单", "关税排除", "关税豁免", 
                "税委会公告", "国务院关税税则委员会", "反倾销", "反补贴"
            ]),
            "secondary": self.keywords_config.get("chinese", {}).get("secondary", [
                "反制措施", "对等反制", "301关税", "中美贸易", "贸易战", 
                "贸易摩擦", "贸易争端", "贸易制裁", "经贸磋商"
            ]),
            "control": self.keywords_config.get("chinese", {}).get("control", [
                "出口管制", "不可靠实体清单", "实体清单", "科技限制", 
                "技术出口", "禁运", "制裁名单"
            ]),
            "sectors": self.keywords_config.get("chinese", {}).get("sectors", [
                "半导体", "芯片", "稀土", "光刻机", "电动车", "太阳能", 
                "钢铁", "铝材", "农产品", "大豆", "汽车"
            ]),
            "organizations": [
                "商务部", "海关总署", "发改委", "工信部", "外交部",
                "MOFCOM", "GACC", "NDRC"
            ],
            "indicators": [
                "公告", "通知", "决定", "办法", "规定", "意见", "通告", 
                "实施", "暂停", "恢复", "调整", "修订", "补充"
            ]
        }
    
    def _load_english_keywords(self) -> Dict[str, List[str]]:
        """Load English keyword categories"""
        return {
            "primary": self.keywords_config.get("english", {}).get("primary", [
                "tariff", "tariffs", "surcharge", "customs duties", "301 tariffs",
                "exclusion", "exemption", "anti-dumping", "countervailing"
            ]),
            "secondary": self.keywords_config.get("english", {}).get("secondary", [
                "retaliation", "countermeasure", "trade war", "MOFCOM", "GACC",
                "trade dispute", "trade sanctions", "trade negotiations"
            ]),
            "control": self.keywords_config.get("english", {}).get("control", [
                "export control", "entity list", "unreliable entity", 
                "technology export", "embargo", "sanctions list"
            ]),
            "sectors": [
                "semiconductor", "semiconductors", "chips", "rare earth", 
                "lithography", "electric vehicle", "EV", "solar panels",
                "steel", "aluminum", "agriculture", "soybeans", "automotive"
            ],
            "organizations": [
                "MOFCOM", "GACC", "USTR", "Commerce Department", "Treasury",
                "State Council", "Tariff Commission"
            ],
            "indicators": [
                "announcement", "notice", "decision", "implementation", 
                "suspension", "resume", "adjustment", "revision"
            ]
        }
    
    def _compile_patterns(self, keywords_dict: Dict[str, List[str]]) -> Dict[str, List[re.Pattern]]:
        """Compile regex patterns for faster matching"""
        patterns = {}
        
        for category, keywords in keywords_dict.items():
            patterns[category] = []
            for keyword in keywords:
                # Create case-insensitive pattern with word boundaries for English
                if re.search(r'[a-zA-Z]', keyword):
                    pattern = re.compile(r'\b' + re.escape(keyword) + r'\b', re.IGNORECASE)
                else:
                    # Chinese doesn't need word boundaries
                    pattern = re.compile(re.escape(keyword), re.IGNORECASE)
                patterns[category].append(pattern)
        
        return patterns
    
    def analyze_article(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze article and return keyword analysis results"""
        title = article.get("title", "")
        content = article.get("content", "")
        language = article.get("language", "unknown")
        
        # Combine title and content for analysis (title weighted more heavily)
        text_for_analysis = f"{title} {title} {content}"  # Title appears twice for higher weight
        
        # Find matches based on detected language
        if language == "zh":
            matches = self._find_matches(text_for_analysis, self.chinese_patterns)
        elif language == "en":
            matches = self._find_matches(text_for_analysis, self.english_patterns)
        else:
            # Try both languages
            matches_zh = self._find_matches(text_for_analysis, self.chinese_patterns)
            matches_en = self._find_matches(text_for_analysis, self.english_patterns)
            matches = matches_zh + matches_en
        
        # Calculate scores
        scores = self._calculate_scores(matches)
        
        # Determine if article passes threshold
        passes_filter = scores["total_score"] >= self.min_score
        
        return {
            "keyword_score": scores["total_score"],
            "keyword_matches": [
                {
                    "keyword": match.keyword,
                    "category": match.category,
                    "weight": match.weight,
                    "context": match.context[:100]  # First 100 chars of context
                }
                for match in matches
            ],
            "category_scores": scores["category_scores"],
            "passes_keyword_filter": passes_filter,
            "matched_categories": list(scores["category_scores"].keys()),
            "primary_keywords_count": len([m for m in matches if m.category == "primary"])
        }
    
    def score_article(self, article: Dict[str, Any]) -> float:
        """Simple scoring method for backward compatibility"""
        analysis = self.analyze_article(article)
        return analysis.get("keyword_score", 0.0)
    
    def _find_matches(self, text: str, patterns: Dict[str, List[re.Pattern]]) -> List[KeywordMatch]:
        """Find all keyword matches in text"""
        matches = []
        
        for category, category_patterns in patterns.items():
            weight = self.category_weights.get(category, 1.0)
            
            for pattern in category_patterns:
                for match in pattern.finditer(text):
                    # Extract context around the match
                    start = max(0, match.start() - 50)
                    end = min(len(text), match.end() + 50)
                    context = text[start:end].strip()
                    
                    keyword_match = KeywordMatch(
                        keyword=match.group(),
                        category=category,
                        weight=weight,
                        position=match.start(),
                        context=context
                    )
                    matches.append(keyword_match)
        
        return matches
    
    def _calculate_scores(self, matches: List[KeywordMatch]) -> Dict[str, Any]:
        """Calculate various scoring metrics from matches"""
        if not matches:
            return {
                "total_score": 0.0,
                "category_scores": {},
                "unique_keywords": 0
            }
        
        # Group by category
        category_scores = {}
        unique_keywords = set()
        
        for match in matches:
            category = match.category
            if category not in category_scores:
                category_scores[category] = 0.0
            
            category_scores[category] += match.weight
            unique_keywords.add(match.keyword.lower())
        
        # Calculate total score with diminishing returns for same category
        total_score = 0.0
        for category, score in category_scores.items():
            # Apply logarithmic scaling to prevent category spam
            if score > 0:
                import math
                adjusted_score = math.log(1 + score) * self.category_weights.get(category, 1.0)
                total_score += adjusted_score
        
        return {
            "total_score": total_score,
            "category_scores": category_scores,
            "unique_keywords": len(unique_keywords)
        }
    
    def get_keyword_suggestions(self, text: str) -> List[str]:
        """Get suggested keywords that might be relevant but not in our lists"""
        # Simple extraction of potential terms
        suggestions = []
        
        # Look for patterns like "XX关税", "XX清单", etc.
        chinese_patterns = [
            r'[\u4e00-\u9fff]{2,}关税',
            r'[\u4e00-\u9fff]{2,}清单', 
            r'[\u4e00-\u9fff]{2,}措施',
            r'[\u4e00-\u9fff]{2,}协议'
        ]
        
        for pattern in chinese_patterns:
            matches = re.findall(pattern, text)
            suggestions.extend(matches)
        
        # English patterns
        english_patterns = [
            r'\b[A-Z][a-z]+ tariff\b',
            r'\b[A-Z][a-z]+ sanctions?\b',
            r'\b[A-Z][a-z]+ agreement\b'
        ]
        
        for pattern in english_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            suggestions.extend(matches)
        
        return list(set(suggestions))  # Remove duplicates
    
    def update_keywords(self, new_keywords: Dict[str, Dict[str, List[str]]]):
        """Update keyword lists (for adaptive learning)"""
        # Merge new keywords with existing ones
        for lang, categories in new_keywords.items():
            if lang == "chinese":
                for category, keywords in categories.items():
                    if category in self.chinese_keywords:
                        self.chinese_keywords[category].extend(keywords)
                        # Remove duplicates
                        self.chinese_keywords[category] = list(set(self.chinese_keywords[category]))
            elif lang == "english":
                for category, keywords in categories.items():
                    if category in self.english_keywords:
                        self.english_keywords[category].extend(keywords)
                        self.english_keywords[category] = list(set(self.english_keywords[category]))
        
        # Recompile patterns
        self.chinese_patterns = self._compile_patterns(self.chinese_keywords)
        self.english_patterns = self._compile_patterns(self.english_keywords)
        
        logger.info("Updated keyword lists")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about keyword configuration"""
        chinese_count = sum(len(keywords) for keywords in self.chinese_keywords.values())
        english_count = sum(len(keywords) for keywords in self.english_keywords.values())
        
        return {
            "total_chinese_keywords": chinese_count,
            "total_english_keywords": english_count,
            "chinese_categories": list(self.chinese_keywords.keys()),
            "english_categories": list(self.english_keywords.keys()),
            "category_weights": self.category_weights,
            "min_score_threshold": self.min_score
        }


def filter_articles_by_keywords(articles: List[Dict[str, Any]], config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Filter articles using keyword analysis"""
    keyword_filter = KeywordFilter(config)
    filtered_articles = []
    
    for article in articles:
        try:
            analysis = keyword_filter.analyze_article(article)
            
            # Add analysis results to article
            article.update(analysis)
            
            # Only keep articles that pass the filter
            if analysis["passes_keyword_filter"]:
                filtered_articles.append(article)
            else:
                logger.debug(f"Article filtered out by keywords: {article.get('title', '')[:50]}")
                
        except Exception as e:
            logger.error(f"Error filtering article: {e}")
            continue
    
    logger.info(f"Keywords filter: {len(filtered_articles)}/{len(articles)} articles passed")
    return filtered_articles


def main():
    """Test the keyword filter"""
    config = {
        "keywords": {
            "chinese": {
                "primary": ["关税", "加征关税", "关税清单"],
                "secondary": ["反制措施", "301关税", "贸易战"]
            },
            "english": {
                "primary": ["tariff", "customs duties"],
                "secondary": ["retaliation", "trade war"]
            }
        },
        "thresholds": {
            "keyword_min": 1.0
        }
    }
    
    test_articles = [
        {
            "title": "国务院关税税则委员会发布对美加征关税商品清单",
            "content": "根据相关法律法规和《国务院关税税则委员会关于对美加征关税商品清单》...",
            "language": "zh"
        },
        {
            "title": "US announces new tariff measures on Chinese imports",
            "content": "The US Trade Representative announced new tariff measures...",
            "language": "en"
        },
        {
            "title": "今日天气预报",
            "content": "明天多云转晴，气温适宜...",
            "language": "zh"
        }
    ]
    
    keyword_filter = KeywordFilter(config)
    
    for i, article in enumerate(test_articles):
        print(f"Article {i+1}: {article['title']}")
        analysis = keyword_filter.analyze_article(article)
        print(f"  Score: {analysis['keyword_score']:.2f}")
        print(f"  Passes: {analysis['passes_keyword_filter']}")
        print(f"  Matches: {len(analysis['keyword_matches'])}")
        if analysis['keyword_matches']:
            for match in analysis['keyword_matches'][:3]:
                print(f"    - {match['keyword']} ({match['category']})")
        print()


if __name__ == "__main__":
    main()