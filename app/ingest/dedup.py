"""
Deduplication engine for detecting and managing duplicate articles
Handles both exact duplicates and near-duplicates with change detection
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from storage.models import Article
from storage.db import get_db_session
from ingest.normalizer import TextNormalizer
import difflib

logger = logging.getLogger(__name__)


class DeduplicationEngine:
    """Handle article deduplication and change detection"""
    
    def __init__(self, config: Dict[str, Any]):
        self.similarity_threshold = config.get("deduplication", {}).get("similarity_threshold", 0.95)
        self.hash_fields = config.get("deduplication", {}).get("hash_fields", ["normalized_title", "source", "date"])
        
        # Change detection settings
        self.change_detection_enabled = config.get("change_detection", {}).get("enabled", True)
        self.significant_keywords = config.get("change_detection", {}).get("significant_keywords", [
            "公告", "补充", "修订", "实施", "调整", "暂停", "恢复", "取消", "新增"
        ])
        self.min_change_ratio = config.get("change_detection", {}).get("min_change_ratio", 0.1)
        
        self.normalizer = TextNormalizer()
    
    def check_duplicates(self, article: Dict[str, Any], db: Session) -> Tuple[bool, Optional[int], Optional[str]]:
        """
        Check if article is duplicate and return (is_duplicate, existing_id, relationship_type)
        relationship_type can be: 'exact', 'near_duplicate', 'updated_version'
        """
        try:
            content_hash = article.get("content_hash")
            normalized_title = article.get("normalized_title")
            source_name = article.get("source_name")
            
            if not content_hash or not normalized_title:
                return False, None, None
            
            # 1. Check for exact hash match (exact duplicate)
            existing_exact = db.query(Article).filter(
                Article.content_hash == content_hash
            ).first()
            
            if existing_exact:
                logger.info(f"Found exact duplicate: {existing_exact.id}")
                return True, existing_exact.id, "exact"
            
            # 2. Check for near-duplicates by title and source
            # Look for articles from same source with similar titles in last 30 days
            recent_cutoff = datetime.utcnow() - timedelta(days=30)
            
            existing_articles = db.query(Article).filter(
                Article.source_id == self._get_source_id(source_name, db),
                Article.discovered_at >= recent_cutoff
            ).all()
            
            for existing in existing_articles:
                # Check title similarity
                if self.normalizer.is_duplicate_title(
                    normalized_title, 
                    existing.normalized_title, 
                    threshold=self.similarity_threshold
                ):
                    # Check if this might be an update
                    if self._is_potential_update(article, existing):
                        logger.info(f"Found potential update of article {existing.id}")
                        return True, existing.id, "updated_version"
                    else:
                        logger.info(f"Found near-duplicate: {existing.id}")
                        return True, existing.id, "near_duplicate"
            
            return False, None, None
            
        except Exception as e:
            logger.error(f"Error checking duplicates: {e}")
            return False, None, None
    
    def is_duplicate(self, article: Dict[str, Any]) -> bool:
        """Simple duplicate check - returns True if article is duplicate"""
        db = get_db_session()
        try:
            is_dup, _, _ = self.check_duplicates(article, db)
            return is_dup
        finally:
            db.close()
    
    def _get_source_id(self, source_name: str, db: Session) -> Optional[int]:
        """Get source ID by name"""
        from storage.models import Source
        source = db.query(Source).filter(Source.name == source_name).first()
        return source.id if source else None
    
    def _is_potential_update(self, new_article: Dict[str, Any], existing_article: Article) -> bool:
        """Check if new article is likely an update of existing one"""
        try:
            # Check publication time - updates usually come later
            new_pub = new_article.get("published_at")
            # Normalize timezone to avoid naive/aware comparison errors
            try:
                if isinstance(new_pub, str):
                    # Accept ISO strings with or without Z
                    new_pub_dt = datetime.fromisoformat(new_pub.replace('Z', '+00:00'))
                else:
                    new_pub_dt = new_pub
                if isinstance(new_pub_dt, datetime) and new_pub_dt is not None:
                    # Convert aware -> UTC naive; leave naive as-is
                    if new_pub_dt.tzinfo is not None:
                        new_pub_dt = new_pub_dt.astimezone(tz=None).replace(tzinfo=None)
                new_pub = new_pub_dt
            except Exception:
                # If parsing fails, fall back to original value
                pass

            if new_pub and existing_article.published_at:
                if new_pub <= existing_article.published_at:
                    return False  # Older article, not an update
            
            # Check for update keywords in title
            title = new_article.get("title", "")
            for keyword in self.significant_keywords:
                if keyword in title:
                    return True
            
            # Check content changes if available
            if new_article.get("content") and existing_article.content:
                change_ratio = self._calculate_content_change_ratio(
                    new_article["content"], 
                    existing_article.content
                )
                if change_ratio >= self.min_change_ratio:
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking if article is update: {e}")
            return False
    
    def _calculate_content_change_ratio(self, new_content: str, old_content: str) -> float:
        """Calculate ratio of content that changed"""
        try:
            # Use difflib to compare content
            old_lines = old_content.split('\n')
            new_lines = new_content.split('\n')
            
            diff = list(difflib.unified_diff(old_lines, new_lines, n=0))
            
            # Count changed lines (lines starting with + or -)
            changed_lines = len([line for line in diff if line.startswith(('+', '-'))])
            total_lines = max(len(old_lines), len(new_lines))
            
            return changed_lines / total_lines if total_lines > 0 else 0
            
        except Exception as e:
            logger.error(f"Error calculating content change ratio: {e}")
            return 0
    
    def handle_duplicate(
        self, 
        article: Dict[str, Any], 
        existing_id: int, 
        relationship_type: str, 
        db: Session
    ) -> Optional[Article]:
        """Handle found duplicate based on relationship type"""
        try:
            existing_article = db.query(Article).get(existing_id)
            if not existing_article:
                return None
            
            if relationship_type == "exact":
                # Just mark as duplicate and reference existing
                existing_article.is_duplicate = True
                db.commit()
                return existing_article
                
            elif relationship_type == "near_duplicate":
                # Mark as duplicate but might want to keep for analysis
                existing_article.is_duplicate = True
                db.commit()
                return existing_article
                
            elif relationship_type == "updated_version":
                # Create new version and link to previous
                return self._create_updated_version(article, existing_article, db)
                
        except Exception as e:
            logger.error(f"Error handling duplicate: {e}")
            return None
    
    def _create_updated_version(
        self, 
        new_article: Dict[str, Any], 
        original_article: Article, 
        db: Session
    ) -> Article:
        """Create new version of article with change tracking"""
        try:
            # Create change summary
            change_summary = self._generate_change_summary(new_article, original_article)
            
            # Create new article as updated version
            from storage.models import create_article, ArticleCreate
            
            article_create = ArticleCreate(
                title=new_article["title"],
                content=new_article.get("content"),
                url=new_article["url"],
                source_id=original_article.source_id,
                published_at=new_article.get("published_at")
            )
            
            updated_article = create_article(db, article_create, new_article["content_hash"])
            
            # Set version info
            updated_article.version = original_article.version + 1
            updated_article.previous_version_id = original_article.id
            updated_article.change_summary = change_summary
            
            # Copy analysis scores from original (will be re-analyzed)
            updated_article.keyword_score = original_article.keyword_score
            updated_article.semantic_score = original_article.semantic_score
            updated_article.classifier_score = original_article.classifier_score
            
            db.commit()
            db.refresh(updated_article)
            
            logger.info(f"Created updated version {updated_article.id} of article {original_article.id}")
            return updated_article
            
        except Exception as e:
            logger.error(f"Error creating updated version: {e}")
            db.rollback()
            return None
    
    def _generate_change_summary(self, new_article: Dict[str, Any], original_article: Article) -> str:
        """Generate summary of what changed"""
        changes = []
        
        try:
            # Title changes
            if new_article["title"] != original_article.title:
                changes.append(f"标题更新")
            
            # Content changes
            new_content = new_article.get("content", "")
            if new_content and original_article.content:
                if new_content != original_article.content:
                    change_ratio = self._calculate_content_change_ratio(new_content, original_article.content)
                    changes.append(f"内容变更 ({change_ratio:.1%})")
            
            # Check for significant keywords
            title = new_article["title"]
            for keyword in self.significant_keywords:
                if keyword in title:
                    changes.append(f"发现关键词: {keyword}")
            
            return "; ".join(changes) if changes else "内容更新"
            
        except Exception as e:
            logger.error(f"Error generating change summary: {e}")
            return "内容更新"
    
    def cleanup_old_duplicates(self, db: Session, days: int = 90):
        """Clean up old duplicate entries"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            # Find old duplicate articles
            old_duplicates = db.query(Article).filter(
                Article.is_duplicate == True,
                Article.discovered_at < cutoff_date
            ).all()
            
            for duplicate in old_duplicates:
                # Archive instead of delete to preserve history
                duplicate.status = "archived"
            
            db.commit()
            logger.info(f"Archived {len(old_duplicates)} old duplicates")
            
        except Exception as e:
            logger.error(f"Error cleaning up duplicates: {e}")
            db.rollback()
    
    def get_duplicate_statistics(self, db: Session) -> Dict[str, Any]:
        """Get statistics about duplicates"""
        try:
            total_articles = db.query(Article).count()
            duplicates = db.query(Article).filter(Article.is_duplicate == True).count()
            
            # Get recent duplicates (last 7 days)
            recent_cutoff = datetime.utcnow() - timedelta(days=7)
            recent_duplicates = db.query(Article).filter(
                Article.is_duplicate == True,
                Article.discovered_at >= recent_cutoff
            ).count()
            
            # Get versions
            versioned_articles = db.query(Article).filter(Article.version > 1).count()
            
            return {
                "total_articles": total_articles,
                "duplicates": duplicates,
                "duplicate_rate": duplicates / total_articles if total_articles > 0 else 0,
                "recent_duplicates": recent_duplicates,
                "versioned_articles": versioned_articles
            }
            
        except Exception as e:
            logger.error(f"Error getting duplicate statistics: {e}")
            return {}


def deduplicate_articles(articles: List[Dict[str, Any]], config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Deduplicate a batch of articles"""
    dedup_engine = DeduplicationEngine(config)
    unique_articles = []
    db = get_db_session()
    
    try:
        for article in articles:
            is_duplicate, existing_id, relationship_type = dedup_engine.check_duplicates(article, db)
            
            if not is_duplicate:
                unique_articles.append(article)
            else:
                # Handle the duplicate
                handled_article = dedup_engine.handle_duplicate(
                    article, existing_id, relationship_type, db
                )
                if handled_article and relationship_type == "updated_version":
                    # Include updated versions in processing
                    article_dict = {
                        "id": handled_article.id,
                        "title": handled_article.title,
                        "content": handled_article.content,
                        "url": handled_article.url,
                        "content_hash": handled_article.content_hash,
                        "is_update": True,
                        "previous_version_id": handled_article.previous_version_id
                    }
                    unique_articles.append(article_dict)
        
        return unique_articles
        
    finally:
        db.close()


def main():
    """Test deduplication"""
    config = {
        "deduplication": {
            "similarity_threshold": 0.95,
            "hash_fields": ["normalized_title", "source", "date"]
        },
        "change_detection": {
            "enabled": True,
            "significant_keywords": ["公告", "补充", "修订", "实施"],
            "min_change_ratio": 0.1
        }
    }
    
    # Test articles
    test_articles = [
        {
            "title": "关于对美加征关税商品清单的公告",
            "normalized_title": "关于对美加征关税商品清单的公告",
            "content": "根据国务院关税税则委员会公告...",
            "content_hash": "hash1",
            "source_name": "MOFCOM",
            "url": "http://example.com/1"
        },
        {
            "title": "关于对美加征关税商品清单的公告", 
            "normalized_title": "关于对美加征关税商品清单的公告",
            "content": "根据国务院关税税则委员会公告...",
            "content_hash": "hash1",  # Same hash = exact duplicate
            "source_name": "MOFCOM", 
            "url": "http://example.com/1"
        },
        {
            "title": "关于对美加征关税商品清单的补充公告",
            "normalized_title": "关于对美加征关税商品清单的补充公告", 
            "content": "根据国务院关税税则委员会最新公告...",
            "content_hash": "hash2",
            "source_name": "MOFCOM",
            "url": "http://example.com/2"
        }
    ]
    
    unique = deduplicate_articles(test_articles, config)
    print(f"Original: {len(test_articles)} articles")
    print(f"After deduplication: {len(unique)} articles")


if __name__ == "__main__":
    main()