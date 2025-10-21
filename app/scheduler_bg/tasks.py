import os
from celery import Celery
from datetime import datetime, timedelta
import asyncio
from typing import Dict, Any

# Import our pipeline components
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from ingest.rsshub_loader import RSSHubLoader
from ingest.html_scraper import HTMLScraper
from ingest.chinese_scraper import ChineseGovScraper, scrape_chinese_sources
from ingest.twitter_scraper import TwitterScraper, scrape_twitter_accounts
from ingest.normalizer import TextNormalizer as ArticleNormalizer
from ingest.dedup import DeduplicationEngine as DuplicateDetector
from smart.keywords import KeywordFilter
from smart.embedder import EmbeddingGenerator
from smart.classifier import RelevanceClassifier as ArticleClassifier
from smart.triage_agent import TriageAgent
from smart.sector_scanner import SectorScanner
from notify.wecom import WeCom
from notify.emailer import EmailNotifier
from notify.telegram import TelegramNotifier
from storage.db import SessionLocal
from storage.models import Article, Source

# Initialize Celery
celery_app = Celery(
    'tariff_radar',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/0'
)

# Configuration
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    result_expires=3600,
    task_soft_time_limit=300,  # 5 minutes
    task_time_limit=600,       # 10 minutes hard limit
)

# Load configuration
import yaml
import os

# Try multiple config paths (Docker vs local)
config_paths = [
    "/app/config.yaml",  # Docker path
    os.path.join(os.path.dirname(__file__), "..", "config.yaml"),  # Local relative
    "config.yaml",  # Current directory
]

config = None
for config_path in config_paths:
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            print(f"✅ Loaded config from: {config_path}")
            break
    except Exception as e:
        continue

if config is None:
    print(f"⚠️ Could not load config.yaml from any path, using defaults")
    config = {"sources": {"rss": []}, "scraping": {}, "ml": {}, "llm": {}}

# Initialize components
loader = RSSHubLoader(config)
scraper = HTMLScraper(config.get("scraping", {}))
normalizer = ArticleNormalizer()
dedup_detector = DuplicateDetector(config.get("deduplication", {}))
keyword_filter = KeywordFilter(config.get("keywords", {}))

# Optional ML components (LAZY LOADING - don't load heavy models at startup!)
embedder = None
classifier = None
triage_agent = None
sector_scanner = None  # Lazy load sector scanner
USE_EMBEDDINGS = True  # Will try to load on first use
USE_CLASSIFIER = True

def get_embedder():
    """Lazy load embedder only when needed"""
    global embedder
    if embedder is None:
        try:
            print("🔄 Loading ML embedding model (this may take 1-2 minutes)...")
            embedder = EmbeddingGenerator(config.get("ml", {}))
            print("✅ Embedding model loaded successfully!")
        except Exception as e:
            print(f"⚠️ Embeddings disabled: {e}")
            embedder = False  # Mark as failed, don't retry
    return embedder if embedder is not False else None

def get_classifier():
    """Lazy load classifier only when needed"""
    global classifier
    if classifier is None:
        try:
            print("🔄 Loading ML classifier model...")
            classifier = ArticleClassifier(config.get("ml", {}))
            print("✅ Classifier model loaded successfully!")
        except Exception as e:
            print(f"⚠️ Classifier disabled: {e}")
            classifier = False  # Mark as failed, don't retry
    return classifier if classifier is not False else None

def get_triage_agent():
    """Lazy load triage agent only when needed"""
    global triage_agent
    if triage_agent is None:
        try:
            print("🔄 Loading LLM triage agent...")
            triage_agent = TriageAgent(config.get("llm", {}))
            print("✅ Triage agent loaded successfully!")
        except Exception as e:
            print(f"⚠️ LLM triage agent disabled: {e}")
            triage_agent = False
    return triage_agent if triage_agent is not False else None

def get_sector_scanner():
    """Lazy load sector scanner only when needed"""
    global sector_scanner
    if sector_scanner is None:
        try:
            print("🔄 Loading Sector Scanner for intelligent stock discovery...")
            sector_scanner = SectorScanner()
            print("✅ Sector Scanner loaded successfully!")
        except Exception as e:
            print(f"⚠️ Sector Scanner disabled: {e}")
            sector_scanner = False
    return sector_scanner if sector_scanner is not False else None

# Notification channels (pass config or use defaults)
try:
    wecom = WeCom(config.get("notifications", {}).get("wecom", {}))
except:
    wecom = WeCom({})

try:
    emailer = EmailNotifier(config.get("notifications", {}).get("email", {}))
except:
    emailer = EmailNotifier({})

try:
    telegram = TelegramNotifier(config.get("notifications", {}).get("telegram", {}))
except:
    telegram = TelegramNotifier({})


@celery_app.task(bind=True, max_retries=3)
def ingest_articles_task(self, source_configs: list = None):
    """
    Main ingestion task - fetches, processes, and analyzes new articles
    
    Args:
        source_configs: List of source configurations to process.
                       If None, uses default RSS sources.
    """
    try:
        print(f"Starting article ingestion at {datetime.utcnow()}")
        
        # Load config in task (Celery workers don't share global state)
        task_config = None
        for config_path in config_paths:
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    task_config = yaml.safe_load(f)
                    print(f"✅ Loaded config from: {config_path}")
                    break
            except Exception as e:
                continue
        if task_config is None:
            print(f"⚠️ Could not load config.yaml, using defaults")
            task_config = {"sources": {"rss": []}, "scraping": {}, "ml": {}, "llm": {}}
        
        # Initialize components with task config
        task_loader = RSSHubLoader(task_config)
        task_scraper = HTMLScraper(task_config.get("scraping", {}))
        task_normalizer = ArticleNormalizer()
        task_dedup_detector = DuplicateDetector(task_config.get("deduplication", {}))
        task_keyword_filter = KeywordFilter(task_config.get("keywords", {}))
        
        # Use default RSS sources if none provided
        if not source_configs:
            # Use sources from config
            rss_sources = task_config.get("sources", {}).get("rss", [])
            if rss_sources:
                source_configs = rss_sources
            else:
                # Fallback to hardcoded if config empty
                source_configs = [
                    {
                        "type": "direct_rss",
                        "url": "http://feeds.reuters.com/reuters/businessNews",
                        "name": "Reuters Business",
                        "priority": "high"
                    },
                    {
                        "type": "direct_rss", 
                        "url": "http://feeds.reuters.com/reuters/companyNews",
                        "name": "Reuters Company News",
                        "priority": "high"
                    },
                    {
                        "type": "direct_rss",
                        "url": "https://rss.nytimes.com/services/xml/rss/nyt/Business.xml",
                        "name": "NY Times Business",
                        "priority": "medium"
                    }
                ]
        
        # Initialize database session
        db = SessionLocal()
        total_processed = 0
        new_articles = []
        
        try:
            for config in source_configs:
                print(f"Processing source: {config.get('name', config.get('source', 'Unknown'))}")
                
                # Step 1: Load articles from RSS
                source_type = config.get("type", "rss")
                if source_type in ["direct_rss", "rsshub", "rss"]:
                    if source_type == "direct_rss":
                        raw_articles = asyncio.run(task_loader.fetch_direct_rss(
                            config["url"],
                            config["name"],
                            config.get("priority", "medium")
                        ))
                    elif source_type == "rsshub":
                        raw_articles = asyncio.run(task_loader.fetch_feed(config))
                    else:  # legacy "rss"
                        if "rsshub.app" in config["url"]:
                            raw_articles = asyncio.run(task_loader.fetch_feed({
                                "url": config["url"],
                                "name": config.get("source", config["url"])
                            }))
                        else:
                            raw_articles = asyncio.run(task_loader.fetch_direct_rss(
                                config["url"],
                                config.get("source", "Unknown"),
                                "medium"
                            ))
                else:
                    continue  # Skip unsupported types
                
                print(f"Found {len(raw_articles)} raw articles from {config.get('name', config.get('source', 'Unknown'))}")
                
                for raw_article in raw_articles:
                    try:
                        # Step 2: Scrape full content
                        if raw_article.get('link'):
                            scraped_content = task_scraper.scrape_article(raw_article['link'])
                            if scraped_content:
                                raw_article.update(scraped_content)
                        
                        # Step 3: Normalize article data
                        normalized = task_normalizer.normalize(
                            raw_article, 
                            source=config.get('name', config.get('source', 'Unknown')),
                            category=config.get("category")
                        )
                        
                        if not normalized:
                            continue
                        
                        # Step 4: Check for duplicates
                        if task_dedup_detector.is_duplicate(normalized):
                            print(f"Skipping duplicate article: {normalized['title'][:50]}...")
                            continue
                        
                        # Keyword filtering
                        keyword_score = task_keyword_filter.score_article(normalized)
                        print(f"Article: {normalized['title'][:50]}... Keyword score: {keyword_score:.2f}")
                        if keyword_score < 0.1:  # Lower threshold to see more articles
                            print(f"Skipping low keyword score: {keyword_score}")
                            continue
                        
                        # Step 6: Create article record
                        # Get or create source
                        source_name = config.get('name', config.get('source', 'Unknown'))
                        source_obj = db.query(Source).filter(Source.name == source_name).first()
                        if not source_obj:
                            source_obj = Source(
                                name=source_name,
                                url=config.get('url', ''),
                                source_type=config.get('type', 'rss'),
                                priority=config.get('priority', 'medium')
                            )
                            db.add(source_obj)
                            db.commit()
                            db.refresh(source_obj)
                        
                        article = Article(
                            title=normalized['title'],
                            normalized_title=normalized.get('normalized_title', normalized['title']),
                            content=normalized.get('content', ''),
                            url=normalized['url'],
                            source_id=source_obj.id,
                            language=normalized.get('language', 'en'),
                            published_at=normalized.get('published_at'),
                            discovered_at=datetime.utcnow(),
                            keyword_score=keyword_score,
                            status='pending'
                        )
                        
                        # Step 7: Generate embeddings for semantic analysis (optional)
                        if USE_EMBEDDINGS:
                            try:
                                emb = get_embedder()
                                if emb:
                                    embedding_result = emb.calculate_semantic_score(normalized)
                                article.semantic_score = embedding_result.get('semantic_score', 0.0)
                                article.topic_tags = embedding_result.get('best_topic', [])
                            except Exception as e:
                                print(f"⚠️ Embedding generation failed: {e}")
                                article.semantic_score = 0.0
                        else:
                            article.semantic_score = 0.0
                        
                        # Step 8: ML classification (optional)
                        if USE_CLASSIFIER:
                            try:
                                cls = get_classifier()
                                if cls:
                                    ml_result = cls.classify_article(normalized)
                                article.classifier_score = ml_result.get('classifier_score', 0.0)
                                article.ml_category = ml_result.get('classifier_prediction', 'unknown')
                            except Exception as e:
                                print(f"⚠️ ML classification failed: {e}")
                                article.classifier_score = 0.0
                        else:
                            article.classifier_score = 0.0
                        
                        # Step 9: Calculate final score and LLM triage
                        article.final_score = max(
                            keyword_score, 
                            article.semantic_score if hasattr(article, 'semantic_score') else 0.0,
                            article.classifier_score if hasattr(article, 'classifier_score') else 0.0
                        )
                        
                        # LLM triage for articles above threshold
                        if article.final_score >= 0.4:  # Lower threshold to catch more articles
                            try:
                                agent = get_triage_agent()
                                if agent:
                                    llm_result = asyncio.run(agent.analyze_article(normalized))
                                article.llm_relevant = llm_result.get('relevant', False)
                                article.llm_summary = llm_result.get('summary', '')
                                article.llm_tags = llm_result.get('tags', [])
                                article.llm_confidence = llm_result.get('confidence', 0.0)
                                
                                # Update final score based on LLM relevance
                                if article.llm_relevant:
                                    article.final_score = max(article.final_score, article.llm_confidence)
                            except Exception as e:
                                print(f"⚠️ LLM triage failed: {e}")
                                article.llm_relevant = None
                                article.llm_summary = ""
                        
                        # Save article to database
                        db.add(article)
                        db.commit()
                        db.refresh(article)
                        
                        new_articles.append(article)
                        total_processed += 1
                        
                        print(f"Processed article: {article.title[:50]}... (Score: {article.final_score:.2f})")
                        
                    except Exception as e:
                        print(f"Error processing article: {e}")
                        continue
            
            print(f"Ingestion completed. Processed {total_processed} new articles")
            
            # Trigger notification task for high-priority articles
            high_priority_articles = [a for a in new_articles if a.final_score >= 0.8]
            if high_priority_articles:
                send_notifications_task.delay(
                    [{"id": a.id, "title": a.title, "score": a.final_score} for a in high_priority_articles],
                    urgency="high"
                )
            
            return {
                "success": True,
                "processed": total_processed,
                "high_priority": len(high_priority_articles),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        finally:
            db.close()
            
    except Exception as exc:
        print(f"Ingestion task failed: {exc}")
        # Retry with exponential backoff
        raise self.retry(exc=exc, countdown=60 * (2 ** self.request.retries))


@celery_app.task(bind=True, max_retries=2)
def send_notifications_task(self, articles_data: list, urgency: str = "medium"):
    """
    Send notifications for important articles
    
    Args:
        articles_data: List of article data dicts with id, title, score
        urgency: Notification urgency level (low/medium/high)
    """
    try:
        print(f"Sending {urgency} priority notifications for {len(articles_data)} articles")
        
        db = SessionLocal()
        results = {"wecom": False, "email": False, "telegram": False}
        
        try:
            # Prepare notification content
            if urgency == "high":
                title = "🚨 High Priority US-China Trade Alert"
                urgency_emoji = "🔴"
            elif urgency == "medium":
                title = "⚠️ US-China Trade Update"
                urgency_emoji = "🟡"
            else:
                title = "ℹ️ US-China Trade News"
                urgency_emoji = "🟢"
            
            message_lines = [f"{title}\n"]
            
            for article_data in articles_data[:5]:  # Limit to top 5 articles
                article = db.query(Article).filter(Article.id == article_data["id"]).first()
                if article:
                    score_emoji = "🔥" if article.final_score >= 0.9 else "⭐" if article.final_score >= 0.7 else "📰"
                    message_lines.append(
                        f"{score_emoji} **{article.title[:80]}{'...' if len(article.title) > 80 else ''}**"
                    )
                    message_lines.append(f"   Score: {article.final_score:.2f} | Source: {article.source}")
                    if article.llm_summary:
                        message_lines.append(f"   Summary: {article.llm_summary[:100]}{'...' if len(article.llm_summary) > 100 else ''}")
                    message_lines.append(f"   Link: {article.url}\n")
            
            message = "\n".join(message_lines)
            
            # Send to all channels based on urgency
            notification_channels = ["wecom", "email"]
            if urgency == "high":
                notification_channels.append("telegram")
            
            # WeChat Work (WeCom) - for team notifications
            if "wecom" in notification_channels:
                try:
                    wecom_result = wecom.send_message(message, urgency=urgency)
                    results["wecom"] = wecom_result.get("success", False)
                    print(f"WeCom notification: {'✅' if results['wecom'] else '❌'}")
                except Exception as e:
                    print(f"WeCom notification failed: {e}")
            
            # Email notifications
            if "email" in notification_channels:
                try:
                    email_result = emailer.send_alert(
                        subject=title,
                        content=message,
                        priority=urgency,
                        articles=[{"id": a["id"], "title": a["title"], "score": a["score"]} for a in articles_data]
                    )
                    results["email"] = email_result.get("success", False)
                    print(f"Email notification: {'✅' if results['email'] else '❌'}")
                except Exception as e:
                    print(f"Email notification failed: {e}")
            
            # Telegram - for high urgency only
            if "telegram" in notification_channels:
                try:
                    telegram_result = telegram.send_alert(message, urgency=urgency)
                    results["telegram"] = telegram_result.get("success", False)
                    print(f"Telegram notification: {'✅' if results['telegram'] else '❌'}")
                except Exception as e:
                    print(f"Telegram notification failed: {e}")
            
            # Update notification log in database
            for article_data in articles_data:
                article = db.query(Article).filter(Article.id == article_data["id"]).first()
                if article:
                    if not article.notifications_sent:
                        article.notifications_sent = []
                    article.notifications_sent.append({
                        "timestamp": datetime.utcnow().isoformat(),
                        "urgency": urgency,
                        "channels": results,
                        "success": any(results.values())
                    })
                    db.commit()
            
            success_count = sum(1 for v in results.values() if v)
            print(f"Notifications completed. {success_count}/{len(notification_channels)} channels successful")
            
            return {
                "success": success_count > 0,
                "results": results,
                "articles_notified": len(articles_data),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        finally:
            db.close()
            
    except Exception as exc:
        print(f"Notification task failed: {exc}")
        raise self.retry(exc=exc, countdown=30 * (2 ** self.request.retries))


@celery_app.task
def cleanup_old_articles_task(days_to_keep: int = 30):
    """
    Clean up old articles and maintain database health
    
    Args:
        days_to_keep: Number of days to retain articles (default: 30)
    """
    try:
        print(f"Starting cleanup of articles older than {days_to_keep} days")
        
        db = SessionLocal()
        cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
        
        try:
            # Archive old articles instead of deleting
            old_articles = db.query(Article).filter(
                Article.discovered_at < cutoff_date,
                Article.status != 'archived'
            ).all()
            
            archived_count = 0
            for article in old_articles:
                # Keep high-scoring articles longer
                if article.final_score >= 0.8:
                    continue
                    
                article.status = 'archived'
                archived_count += 1
            
            db.commit()
            
            # Clean up duplicate detection cache
            dedup_detector.cleanup_cache(days_to_keep)
            
            print(f"Cleanup completed. Archived {archived_count} articles")
            
            return {
                "success": True,
                "archived_articles": archived_count,
                "cutoff_date": cutoff_date.isoformat(),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        finally:
            db.close()
            
    except Exception as e:
        print(f"Cleanup task failed: {e}")
        return {"success": False, "error": str(e)}


@celery_app.task
def health_check_task():
    """
    Perform system health checks and monitoring
    """
    try:
        print("Performing system health check")
        
        db = SessionLocal()
        health_status = {
            "database": False,
            "redis": False,
            "article_pipeline": False,
            "notifications": False,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        try:
            # Database health check
            try:
                article_count = db.query(Article).count()
                recent_articles = db.query(Article).filter(
                    Article.discovered_at >= datetime.utcnow() - timedelta(hours=24)
                ).count()
                health_status["database"] = True
                health_status["article_count"] = article_count
                health_status["recent_articles"] = recent_articles
            except Exception as e:
                print(f"Database health check failed: {e}")
            
            # Redis connectivity check
            try:
                celery_app.backend.get('health_check_key')
                health_status["redis"] = True
            except Exception as e:
                print(f"Redis health check failed: {e}")
            
            # Pipeline components check
            try:
                # Test each component briefly
                loader_status = loader.health_check() if hasattr(loader, 'health_check') else True
                scraper_status = scraper.health_check() if hasattr(scraper, 'health_check') else True
                health_status["article_pipeline"] = loader_status and scraper_status
            except Exception as e:
                print(f"Pipeline health check failed: {e}")
            
            # Notification channels check
            try:
                wecom_status = wecom.health_check() if hasattr(wecom, 'health_check') else True
                email_status = emailer.health_check() if hasattr(emailer, 'health_check') else True
                health_status["notifications"] = wecom_status and email_status
            except Exception as e:
                print(f"Notifications health check failed: {e}")
            
            overall_health = all([
                health_status["database"],
                health_status["redis"],
                health_status["article_pipeline"]
            ])
            health_status["overall_healthy"] = overall_health
            
            print(f"Health check completed. Overall status: {'✅' if overall_health else '❌'}")
            return health_status
            
        finally:
            db.close()
            
    except Exception as e:
        print(f"Health check failed: {e}")
        return {
            "overall_healthy": False,
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


@celery_app.task(bind=True, max_retries=3)
def scrape_chinese_gov_sites_task(self):
    """
    Scrape Chinese government websites directly
    This is THE MOST IMPORTANT task for identifying "who published first"!
    
    Sources:
    - MOFCOM (Ministry of Commerce) - ALWAYS first for tariff announcements
    - GACC (Customs) - Implementation details
    - MOF (Ministry of Finance) - Tax policies
    - State Council - High-level policies
    """
    try:
        print("🇨🇳 Starting Chinese government websites scraping...")
        
        # Run async scraping
        articles = asyncio.run(scrape_chinese_sources())
        
        if not articles:
            print("⚠️ No articles found from Chinese sources")
            return {
                "success": True,
                "processed": 0,
                "message": "No articles found"
            }
        
        print(f"📰 Found {len(articles)} articles from Chinese government sources")
        
        # Process articles through the pipeline
        db = SessionLocal()
        new_articles = []
        processed_count = 0
        
        try:
            for raw_article in articles:
                try:
                    # Normalize
                    normalized = normalizer.normalize(
                        raw_article,
                        source=raw_article.get('source', 'Chinese Government'),
                        category='government'
                    )
                    
                    if not normalized:
                        continue
                    
                    # Check for duplicates
                    if dedup_detector.is_duplicate(normalized):
                        continue
                    
                    # Keyword filtering
                    keyword_score = keyword_filter.score_article(normalized)
                    if keyword_score < 0.2:  # Lower threshold for government sources
                        continue
                    
                    # Create article record
                    # Get or create source
                    source_name = normalized.get('source', 'Chinese Government')
                    source_obj = db.query(Source).filter(Source.name == source_name).first()
                    if not source_obj:
                        source_obj = Source(
                            name=source_name,
                            url='',
                            source_type='scraper',
                            priority='high'
                        )
                        db.add(source_obj)
                        db.commit()
                        db.refresh(source_obj)
                    
                    article = Article(
                        title=normalized['title'],
                        normalized_title=normalized.get('normalized_title', normalized['title']),
                        content=normalized.get('content', ''),
                        url=normalized['url'],
                        source_id=source_obj.id,
                        language=normalized.get('language', 'zh'),
                        published_at=normalized.get('published_at'),
                        discovered_at=datetime.utcnow(),
                        keyword_score=keyword_score,
                        status='pending'
                    )
                    
                    # Generate embeddings
                    if USE_EMBEDDINGS:
                        try:
                            emb = get_embedder()
                            if emb:
                                embedding_result = emb.calculate_semantic_score(normalized)
                            article.semantic_score = embedding_result.get('semantic_score', 0.0)
                            article.topic_tags = embedding_result.get('best_topic', [])
                        except Exception as e:
                            print(f"⚠️ Embedding generation failed: {e}")
                            article.semantic_score = 0.0
                    
                    # ML classification
                    if USE_CLASSIFIER:
                        try:
                            clf = get_classifier()
                            if clf:
                                ml_result = clf.classify_article(normalized)
                                article.classifier_score = ml_result.get('classifier_score', 0.0)
                                article.ml_category = ml_result.get('classifier_prediction', 'unknown')
                        except Exception as e:
                            article.classifier_score = 0.0
                    
                    # Calculate final score
                    article.final_score = max(
                        keyword_score,
                        article.semantic_score if hasattr(article, 'semantic_score') else 0.0,
                        article.classifier_score if hasattr(article, 'classifier_score') else 0.0
                    )
                    
                    # LLM triage for high-scoring articles
                    if article.final_score >= 0.4:
                        try:
                            agent = get_triage_agent()
                            if agent:
                                llm_result = asyncio.run(agent.analyze_article(normalized))
                            article.llm_relevant = llm_result.get('relevant', False)
                            article.llm_summary = llm_result.get('summary', '')
                            article.llm_tags = llm_result.get('tags', [])
                            article.llm_confidence = llm_result.get('confidence', 0.0)
                            
                            if article.llm_relevant:
                                article.final_score = max(article.final_score, article.llm_confidence)
                        except Exception as e:
                            print(f"⚠️ LLM triage failed: {e}")
                    
                    # Save to database
                    db.add(article)
                    db.commit()
                    db.refresh(article)
                    
                    new_articles.append(article)
                    processed_count += 1
                    
                    print(f"✅ {article.source}: {article.title[:50]}... (Score: {article.final_score:.2f})")
                    
                except Exception as e:
                    print(f"❌ Error processing Chinese article: {e}")
                    continue
            
            # Send notifications for high-priority articles
            high_priority = [a for a in new_articles if a.final_score >= 0.75]
            if high_priority:
                print(f"🚨 Found {len(high_priority)} high-priority Chinese government articles!")
                send_notifications_task.delay(
                    [{"id": a.id, "title": a.title, "score": a.final_score} for a in high_priority],
                    urgency="high"
                )
            
            return {
                "success": True,
                "processed": processed_count,
                "high_priority": len(high_priority),
                "timestamp": datetime.utcnow().isoformat(),
                "sources_scraped": list(set([a.source for a in new_articles]))
            }
            
        finally:
            db.close()
            
    except Exception as exc:
        print(f"❌ Chinese government scraping failed: {exc}")
        raise self.retry(exc=exc, countdown=120 * (2 ** self.request.retries))


@celery_app.task(bind=True, max_retries=3)
def scrape_twitter_accounts_task(self):
    """
    Scrape Twitter/X accounts for tariff announcements
    
    THIS IS THE MOST IMPORTANT TASK!
    Trump often tweets tariff announcements BEFORE official statements
    
    Monitored accounts:
    - @realDonaldTrump - Personal account (FIRST source!)
    - @POTUS - Official presidential
    - @WhiteHouse - White House official
    - @USTreasury, @SecRaimondo, @USTradeRep - Government officials
    """
    try:
        print("🐦 Starting Twitter/X monitoring for tariff announcements...")
        
        # Run async Twitter scraping
        tweets = asyncio.run(scrape_twitter_accounts())
        
        if not tweets:
            print("ℹ️  No tariff-related tweets found")
            return {
                "success": True,
                "processed": 0,
                "message": "No relevant tweets"
            }
        
        print(f"🐦 Found {len(tweets)} tariff-related tweets")
        
        # Process tweets through the pipeline
        db = SessionLocal()
        new_articles = []
        processed_count = 0
        
        try:
            for tweet in tweets:
                try:
                    # Normalize
                    normalized = normalizer.normalize(
                        tweet,
                        source=tweet.get('source', 'Twitter'),
                        category='social_media'
                    )
                    
                    if not normalized:
                        continue
                    
                    # Check for duplicates
                    if dedup_detector.is_duplicate(normalized):
                        continue
                    
                    # Keyword filtering (tweets already pre-filtered)
                    keyword_score = keyword_filter.score_article(normalized)
                    if keyword_score < 0.1:  # Very low threshold for tweets
                        continue
                    
                    # Create article record
                    # Get or create source
                    source_name = normalized.get('source', 'Twitter')
                    source_obj = db.query(Source).filter(Source.name == source_name).first()
                    if not source_obj:
                        source_obj = Source(
                            name=source_name,
                            url='',
                            source_type='scraper',
                            priority='high'
                        )
                        db.add(source_obj)
                        db.commit()
                        db.refresh(source_obj)
                    
                    article = Article(
                        title=normalized['title'],
                        normalized_title=normalized.get('normalized_title', normalized['title']),
                        content=normalized.get('content', ''),
                        url=normalized['url'],
                        source_id=source_obj.id,
                        language=normalized.get('language', 'en'),
                        published_at=normalized.get('published_at'),
                        discovered_at=datetime.utcnow(),
                        keyword_score=keyword_score,
                        status='pending'
                    )
                    
                    # Embeddings
                    if USE_EMBEDDINGS:
                        try:
                            emb = get_embedder()
                            if emb:
                                embedding_result = emb.calculate_semantic_score(normalized)
                                article.semantic_score = embedding_result.get('semantic_score', 0.0)
                                article.topic_tags = embedding_result.get('best_topic', [])
                            else:
                                article.semantic_score = 0.0
                        except Exception as e:
                            article.semantic_score = 0.0
                    
                    # ML classification
                    if USE_CLASSIFIER:
                        try:
                            cls = get_classifier()
                            if cls:
                                ml_result = cls.classify_article(normalized)
                            article.classifier_score = ml_result.get('classifier_score', 0.0)
                            article.ml_category = ml_result.get('classifier_prediction', 'unknown')
                        except Exception as e:
                            article.classifier_score = 0.0
                    
                    # Calculate final score (boost tweets from official accounts)
                    base_score = max(
                        keyword_score,
                        article.semantic_score if hasattr(article, 'semantic_score') else 0.0,
                        article.classifier_score if hasattr(article, 'classifier_score') else 0.0
                    )
                    
                    # Boost Trump's tweets significantly
                    if '@realDonaldTrump' in article.source or '@POTUS' in article.source:
                        base_score = min(base_score * 1.5, 1.0)  # 50% boost, cap at 1.0
                    
                    article.final_score = base_score
                    
                    # LLM triage (important for tweets)
                    if article.final_score >= 0.3:
                        try:
                            agent = get_triage_agent()
                            if agent:
                                llm_result = asyncio.run(agent.analyze_article(normalized))
                            article.llm_relevant = llm_result.get('relevant', False)
                            article.llm_summary = llm_result.get('summary', '')
                            article.llm_tags = llm_result.get('tags', [])
                            article.llm_confidence = llm_result.get('confidence', 0.0)
                            
                            if article.llm_relevant:
                                article.final_score = max(article.final_score, article.llm_confidence)
                        except Exception as e:
                            print(f"⚠️ LLM triage failed: {e}")
                    
                    # Save to database
                    db.add(article)
                    db.commit()
                    db.refresh(article)
                    
                    new_articles.append(article)
                    processed_count += 1
                    
                    print(f"🐦 {article.source}: {article.title[:60]}... (Score: {article.final_score:.2f})")
                    
                except Exception as e:
                    print(f"❌ Error processing tweet: {e}")
                    continue
            
            # Send IMMEDIATE notifications for Trump tweets!
            trump_tweets = [a for a in new_articles if 'Trump' in a.source and a.final_score >= 0.6]
            if trump_tweets:
                print(f"🚨 URGENT! {len(trump_tweets)} tariff-related tweets from Trump!")
                send_notifications_task.delay(
                    [{"id": a.id, "title": a.title, "score": a.final_score} for a in trump_tweets],
                    urgency="high"
                )
            
            # Other high-priority tweets
            other_high = [a for a in new_articles if a not in trump_tweets and a.final_score >= 0.7]
            if other_high:
                send_notifications_task.delay(
                    [{"id": a.id, "title": a.title, "score": a.final_score} for a in other_high],
                    urgency="medium"
                )
            
            return {
                "success": True,
                "processed": processed_count,
                "trump_tweets": len(trump_tweets),
                "other_high_priority": len(other_high),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        finally:
            db.close()
            
    except Exception as exc:
        print(f"❌ Twitter scraping failed: {exc}")
        raise self.retry(exc=exc, countdown=60 * (2 ** self.request.retries))


@celery_app.task(bind=True, max_retries=2)
def scan_rss_for_stock_opportunities_task(self):
    """
    🔍 INTELLIGENT STOCK DISCOVERY - Background scanner for high-potential opportunities
    
    This task continuously scans RSS articles using the Sector Scanner to identify:
    - Sector-specific news (FDA approvals, Quantum breakthroughs, AI chip advances, etc.)
    - Companies mentioned in high-relevance articles
    - ML-validated stock opportunities (expected return > 2%, confidence > 65%)
    
    Runs every 15 minutes to discover opportunities before they hit mainstream news.
    """
    try:
        print("🔍 Starting intelligent stock opportunity scanner...")
        print("=" * 70)
        
        # Initialize components
        scanner = get_sector_scanner()
        if not scanner:
            print("⚠️ Sector Scanner not available, skipping task")
            return {"success": False, "message": "Sector Scanner unavailable"}
        
        db = SessionLocal()
        
        try:
            # Get recent articles from the last hour (not yet scanned for stocks)
            one_hour_ago = datetime.utcnow() - timedelta(hours=1)
            recent_articles = db.query(Article).filter(
                Article.discovered_at >= one_hour_ago,
                Article.final_score >= 0.3  # Only scan articles with some relevance
            ).order_by(Article.discovered_at.desc()).limit(50).all()
            
            if not recent_articles:
                print("ℹ️  No recent articles to scan")
                return {
                    "success": True,
                    "message": "No recent articles found",
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            print(f"📰 Scanning {len(recent_articles)} recent articles for stock opportunities...")
            
            opportunities = []
            sectors_found = {}
            
            for article in recent_articles:
                try:
                    # Prepare article data for scanner
                    article_data = {
                        'title': article.title,
                        'content': article.content or '',
                        'summary': article.llm_summary or '',
                        'url': article.url,
                        'published_at': article.published_at.isoformat() if article.published_at else None
                    }
                    
                    # Scan article for sector relevance
                    scan_result = scanner.scan_article(article_data)
                    
                    # Skip low-relevance articles
                    if scan_result['relevance_score'] < 8:
                        continue
                    
                    print(f"\n🎯 HIGH RELEVANCE ARTICLE (Score: {scan_result['relevance_score']:.1f})")
                    print(f"   Title: {article.title[:80]}...")
                    print(f"   Sectors: {', '.join(scan_result['sectors_matched'])}")
                    print(f"   Potential Tickers: {', '.join(scan_result['potential_tickers'][:5])}")
                    
                    # Track sectors
                    for sector in scan_result['sectors_matched']:
                        sectors_found[sector] = sectors_found.get(sector, 0) + 1
                    
                    # Validate each ticker with ML
                    for ticker in scan_result['potential_tickers'][:3]:  # Top 3 tickers per article
                        try:
                            # Run async ML validation
                            ml_result = asyncio.run(scanner.scan_ticker_with_ml(ticker))
                            
                            if ml_result.get('has_potential', False):
                                opportunity = {
                                    'ticker': ticker,
                                    'article_id': article.id,
                                    'article_title': article.title,
                                    'sectors': scan_result['sectors_matched'],
                                    'relevance_score': scan_result['relevance_score'],
                                    'sentiment': scan_result['sentiment'],
                                    'ml_score': ml_result.get('ml_score', 0),
                                    'recommendation': ml_result.get('recommendation', 'HOLD'),
                                    'confidence': ml_result.get('confidence', 0),
                                    'expected_return': ml_result.get('expected_return', 0),
                                    'predicted_price': ml_result.get('predicted_price', 0),
                                    'current_price': ml_result.get('current_price', 0),
                                    'discovered_at': datetime.utcnow().isoformat()
                                }
                                
                                opportunities.append(opportunity)
                                
                                print(f"   ✅ {ticker}: {ml_result['recommendation']} "
                                      f"(Expected: +{ml_result['expected_return']:.2f}%, "
                                      f"Confidence: {ml_result['confidence']*100:.1f}%)")
                            
                        except Exception as e:
                            print(f"   ⚠️ Error validating {ticker}: {e}")
                            continue
                    
                except Exception as e:
                    print(f"⚠️ Error scanning article {article.id}: {e}")
                    continue
            
            print("\n" + "=" * 70)
            print(f"🎯 SCAN COMPLETE - Found {len(opportunities)} high-potential opportunities!")
            
            if opportunities:
                print("\n📊 SECTOR BREAKDOWN:")
                for sector, count in sorted(sectors_found.items(), key=lambda x: x[1], reverse=True):
                    print(f"   {sector}: {count} mentions")
                
                print("\n🚀 TOP OPPORTUNITIES:")
                # Sort by expected return
                top_opportunities = sorted(opportunities, key=lambda x: x['expected_return'], reverse=True)[:5]
                for i, opp in enumerate(top_opportunities, 1):
                    print(f"   {i}. {opp['ticker']} - {opp['recommendation']}")
                    print(f"      Expected Return: +{opp['expected_return']:.2f}% | Confidence: {opp['confidence']*100:.1f}%")
                    print(f"      Article: {opp['article_title'][:60]}...")
                    print(f"      Sectors: {', '.join(opp['sectors'][:2])}")
                
                # Store opportunities in cache or database for hot-stocks endpoint
                # For now, just log them. The /api/scanner/hot-stocks endpoint
                # will run its own scan when called.
                
                # Optional: Send notification for VERY high-confidence opportunities
                urgent_opportunities = [
                    opp for opp in opportunities 
                    if opp['expected_return'] > 5 and opp['confidence'] > 0.8
                ]
                
                if urgent_opportunities:
                    print(f"\n🚨 URGENT: {len(urgent_opportunities)} VERY HIGH-CONFIDENCE OPPORTUNITIES!")
                    # Could trigger notification here if needed
            else:
                print("ℹ️  No high-potential opportunities detected in this scan")
            
            print("=" * 70)
            
            return {
                "success": True,
                "articles_scanned": len(recent_articles),
                "opportunities_found": len(opportunities),
                "sectors_detected": list(sectors_found.keys()),
                "top_opportunities": opportunities[:10],  # Return top 10
                "timestamp": datetime.utcnow().isoformat()
            }
            
        finally:
            db.close()
            
    except Exception as exc:
        print(f"❌ Stock opportunity scanner failed: {exc}")
        raise self.retry(exc=exc, countdown=120 * (2 ** self.request.retries))


# Task routing and scheduling will be handled by beat.py
if __name__ == "__main__":
    # For testing tasks individually
    print("Tariff Radar Celery Tasks Ready")
    print("Available tasks:")
    print("- ingest_articles_task: Main article ingestion pipeline")
    print("- send_notifications_task: Send alerts for important articles")
    print("- cleanup_old_articles_task: Archive old articles")
    print("- health_check_task: System health monitoring")
    print("- scan_rss_for_stock_opportunities_task: 🔍 Intelligent stock discovery scanner")