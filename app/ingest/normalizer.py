"""
Text normalizer and preprocessor
Cleans HTML, detects language, normalizes text for analysis
"""
import re
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import html
from bs4 import BeautifulSoup
try:
    from langdetect import detect, LangDetectException as LangDetectError
except ImportError:
    from langdetect import detect
    LangDetectError = Exception

logger = logging.getLogger(__name__)


class TextNormalizer:
    """Text normalization and cleaning utilities"""
    
    def __init__(self):
        # Chinese punctuation and symbols
        self.chinese_punct = r'[，。！？；：""''（）【】《》〈〉「」『』〔〕…—·]'
        
        # Regex patterns for common noise
        self.noise_patterns = [
            r'点击查看更多',
            r'阅读原文',
            r'点击阅读全文',
            r'更多详情请关注',
            r'来源[:：].*',
            r'责任编辑[:：].*',
            r'编辑[:：].*',
            r'记者[:：].*',
            r'\[.*?\]',  # Remove content in brackets
            r'【.*?】',   # Remove content in Chinese brackets
        ]
        
        # Government document indicators
        self.gov_indicators = [
            '公告', '通知', '决定', '办法', '规定', '意见', '通告', '公报',
            '条例', '法规', '实施细则', '暂行规定', '管理办法'
        ]
    
    def normalize_article(self, raw_article: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize a single article"""
        try:
            # Clean and normalize title
            title = self.clean_text(raw_article.get("title", ""))
            normalized_title = self.normalize_title(title)
            
            # Clean and normalize content
            content = raw_article.get("content", "")
            if content:
                content = self.clean_html(content)
                content = self.clean_text(content)
            
            # Detect language
            language = self.detect_language(title + " " + content)
            
            # Generate content hash for deduplication
            content_hash = self.generate_content_hash(normalized_title, content)
            
            # Extract metadata
            metadata = self.extract_metadata(title, content)
            
            return {
                "title": title,
                "normalized_title": normalized_title,
                "content": content,
                "content_hash": content_hash,
                "language": language,
                "url": raw_article.get("url", ""),
                "published_at": raw_article.get("published_at"),
                "source_name": raw_article.get("source_name", ""),
                "source_priority": raw_article.get("source_priority", "medium"),
                "source_type": raw_article.get("source_type", "unknown"),
                "metadata": {**raw_article.get("metadata", {}), **metadata}
            }
            
        except Exception as e:
            logger.error(f"Failed to normalize article: {e}")
            return raw_article
    
    def normalize(self, raw_article: Dict[str, Any], source: str = None, category: str = None) -> Dict[str, Any]:
        """Alias for normalize_article for backward compatibility"""
        # Add source and category to the article if provided
        if source and 'source_name' not in raw_article:
            raw_article['source_name'] = source
        if category and 'source_type' not in raw_article:
            raw_article['source_type'] = category
        return self.normalize_article(raw_article)
    
    def clean_html(self, html_text: str) -> str:
        """Remove HTML tags and decode entities"""
        if not html_text:
            return ""
            
        try:
            # Decode HTML entities
            text = html.unescape(html_text)
            
            # Parse with BeautifulSoup to remove tags
            soup = BeautifulSoup(text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "header", "footer"]):
                script.decompose()
            
            # Get text content
            text = soup.get_text()
            
            # Clean up whitespace
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()
            
            return text
            
        except Exception as e:
            logger.error(f"Failed to clean HTML: {e}")
            return html_text
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        if not text:
            return ""
        
        try:
            # Remove noise patterns
            for pattern in self.noise_patterns:
                text = re.sub(pattern, '', text, flags=re.IGNORECASE)
            
            # Normalize whitespace
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'\n\s*\n', '\n', text)
            
            # Remove extra punctuation
            text = re.sub(r'[.]{3,}', '...', text)
            text = re.sub(r'[!]{2,}', '!', text)
            text = re.sub(r'[?]{2,}', '?', text)
            
            # Trim
            text = text.strip()
            
            return text
            
        except Exception as e:
            logger.error(f"Failed to clean text: {e}")
            return text
    
    def normalize_title(self, title: str) -> str:
        """Normalize title for deduplication"""
        if not title:
            return ""
        
        try:
            # Convert to lowercase for comparison
            normalized = title.lower()
            
            # Remove common prefixes/suffixes
            prefixes_to_remove = [
                r'^【.*?】\s*',
                r'^关于\s*',
                r'^通知[:：]\s*',
                r'^公告[:：]\s*',
            ]
            
            for prefix in prefixes_to_remove:
                normalized = re.sub(prefix, '', normalized)
            
            # Remove punctuation for comparison
            normalized = re.sub(r'[^\w\s]', '', normalized)
            
            # Normalize whitespace
            normalized = re.sub(r'\s+', ' ', normalized).strip()
            
            return normalized
            
        except Exception as e:
            logger.error(f"Failed to normalize title: {e}")
            return title
    
    def detect_language(self, text: str) -> str:
        """Detect text language"""
        if not text or len(text) < 20:
            return "unknown"
        
        try:
            # Use first 500 chars for detection
            sample = text[:500]
            lang = detect(sample)
            
            # Map to our standard codes
            lang_mapping = {
                "zh-cn": "zh",
                "zh": "zh", 
                "en": "en",
                "ja": "ja",
                "ko": "ko"
            }
            
            return lang_mapping.get(lang, lang)
            
        except LangDetectError:
            # Fallback: check for Chinese characters
            if re.search(r'[\u4e00-\u9fff]', text):
                return "zh"
            elif re.search(r'[a-zA-Z]', text):
                return "en"
            else:
                return "unknown"
        except Exception as e:
            logger.error(f"Language detection failed: {e}")
            return "unknown"
    
    def generate_content_hash(self, title: str, content: str) -> str:
        """Generate hash for deduplication"""
        import hashlib
        
        # Combine title and first part of content
        text_for_hash = title + (content[:500] if content else "")
        text_for_hash = self.normalize_title(text_for_hash)
        
        return hashlib.sha256(text_for_hash.encode('utf-8')).hexdigest()
    
    def extract_metadata(self, title: str, content: str) -> Dict[str, Any]:
        """Extract metadata from title and content"""
        metadata = {}
        
        try:
            # Check if it's a government document
            is_gov_doc = any(indicator in title for indicator in self.gov_indicators)
            metadata["is_government_document"] = is_gov_doc
            
            # Extract document type
            doc_types = []
            for indicator in self.gov_indicators:
                if indicator in title:
                    doc_types.append(indicator)
            metadata["document_types"] = doc_types
            
            # Check for urgency indicators
            urgency_keywords = ['紧急', '重要', '特急', '加急', '立即', '马上']
            is_urgent = any(keyword in title for keyword in urgency_keywords)
            metadata["is_urgent"] = is_urgent
            
            # Extract numbers/references
            doc_numbers = re.findall(r'[〔（\[]\d{4}[〕）\]]\d+号?', title)
            if doc_numbers:
                metadata["document_numbers"] = doc_numbers
            
            # Estimate content length category
            if content:
                content_len = len(content)
                if content_len < 500:
                    metadata["content_length"] = "short"
                elif content_len < 2000:
                    metadata["content_length"] = "medium"
                else:
                    metadata["content_length"] = "long"
            
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to extract metadata: {e}")
            return {}
    
    def is_duplicate_title(self, title1: str, title2: str, threshold: float = 0.9) -> bool:
        """Check if two titles are likely duplicates"""
        if not title1 or not title2:
            return False
        
        try:
            norm1 = self.normalize_title(title1)
            norm2 = self.normalize_title(title2)
            
            if norm1 == norm2:
                return True
            
            # Calculate simple similarity
            if len(norm1) == 0 or len(norm2) == 0:
                return False
                
            # Jaccard similarity on words
            words1 = set(norm1.split())
            words2 = set(norm2.split())
            
            if not words1 or not words2:
                return False
            
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            
            similarity = intersection / union if union > 0 else 0
            
            return similarity >= threshold
            
        except Exception as e:
            logger.error(f"Failed to check title similarity: {e}")
            return False


def normalize_articles(articles: list) -> list:
    """Normalize a list of articles"""
    normalizer = TextNormalizer()
    normalized = []
    
    for article in articles:
        try:
            normalized_article = normalizer.normalize_article(article)
            normalized.append(normalized_article)
        except Exception as e:
            logger.error(f"Failed to normalize article: {e}")
            continue
    
    return normalized


def main():
    """Test the normalizer"""
    test_articles = [
        {
            "title": "【重要公告】关于对美加征关税商品清单的通知",
            "content": """<p>根据国务院关税税则委员会公告...</p><script>tracking();</script>""",
            "url": "http://example.com/article1",
            "source_name": "MOFCOM"
        },
        {
            "title": "US announces new tariffs on Chinese imports   ",
            "content": "The US Trade Representative office announced...",
            "url": "http://example.com/article2", 
            "source_name": "Reuters"
        }
    ]
    
    normalizer = TextNormalizer()
    
    for i, article in enumerate(test_articles):
        print(f"Article {i+1}:")
        print(f"Original title: {article['title']}")
        
        normalized = normalizer.normalize_article(article)
        print(f"Normalized title: {normalized['normalized_title']}")
        print(f"Language: {normalized['language']}")
        print(f"Hash: {normalized['content_hash'][:16]}...")
        print(f"Metadata: {normalized['metadata']}")
        print()


if __name__ == "__main__":
    main()