"""
Background Scheduler for MarketPulse
Runs automated tasks:
- RSS feed scraping
- Keyword analysis
- Perplexity AI scans
- Alert triggering
- WebSocket broadcasting
"""
import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Dict, Any, List
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
import yaml

# Import our components
from app.ingest.rss_loader import FinancialDataLoader
from app.smart.keywords_engine import FinancialKeywordsEngine, analyze_articles_batch
from app.smart.perplexity_finance import PerplexityFinanceAnalyzer

logger = logging.getLogger(__name__)


class MarketPulseScheduler:
    """Background scheduler for automated monitoring"""
    
    def __init__(self, config_path: str = "config/data_sources.yaml"):
        """Initialize scheduler with configuration"""
        self.scheduler = AsyncIOScheduler(timezone="UTC")
        
        # Load configuration with proper path resolution
        if not os.path.isabs(config_path):
            # Make relative to this file's directory
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), config_path)
        
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        
        # Initialize components
        self.data_loader = FinancialDataLoader(config_path)
        self.keywords_engine = FinancialKeywordsEngine()
        self.perplexity = PerplexityFinanceAnalyzer()
        
        # Statistics tracking
        self.stats = {
            "total_articles_fetched": 0,
            "total_alerts_fired": 0,
            "last_run": None,
            "runs_count": 0
        }
        
        # Alert cache (prevent duplicate alerts)
        self.alert_cache = set()
        
        # WebSocket connections (will be set by main app)
        self.websocket_broadcast_callback = None
        
        logger.info("✅ MarketPulseScheduler initialized")
    
    def setup_schedules(self):
        """Setup all scheduled jobs based on config"""
        schedule_config = self.config.get("scheduler", {})
        
        # ============================================================
        # TIER 1: Major News RSS (Every 5 minutes)
        # ============================================================
        self.scheduler.add_job(
            self._fetch_major_news,
            trigger=IntervalTrigger(
                minutes=schedule_config.get("rss_feeds", {}).get("major_news_interval", 300) // 60
            ),
            id="major_news_rss",
            name="Fetch Major News RSS",
            replace_existing=True,
            max_instances=1
        )
        
        # ============================================================
        # TIER 2: Market-Specific RSS (Every 10 minutes)
        # ============================================================
        self.scheduler.add_job(
            self._fetch_market_specific,
            trigger=IntervalTrigger(
                minutes=schedule_config.get("rss_feeds", {}).get("market_specific_interval", 600) // 60
            ),
            id="market_specific_rss",
            name="Fetch Market-Specific RSS",
            replace_existing=True,
            max_instances=1
        )
        
        # ============================================================
        # TIER 3: Sector-Specific RSS (Every 15 minutes)
        # ============================================================
        self.scheduler.add_job(
            self._fetch_sector_specific,
            trigger=IntervalTrigger(
                minutes=schedule_config.get("rss_feeds", {}).get("sector_specific_interval", 900) // 60
            ),
            id="sector_specific_rss",
            name="Fetch Sector-Specific RSS",
            replace_existing=True,
            max_instances=1
        )
        
        # ============================================================
        # SEC Filings (Every hour)
        # ============================================================
        self.scheduler.add_job(
            self._fetch_sec_filings,
            trigger=IntervalTrigger(
                minutes=schedule_config.get("regulatory", {}).get("sec_edgar_interval", 3600) // 60
            ),
            id="sec_filings",
            name="Fetch SEC Filings",
            replace_existing=True,
            max_instances=1
        )
        
        # ============================================================
        # FDA Updates (Every 2 hours)
        # ============================================================
        self.scheduler.add_job(
            self._fetch_fda_updates,
            trigger=IntervalTrigger(
                minutes=schedule_config.get("regulatory", {}).get("fda_interval", 7200) // 60
            ),
            id="fda_updates",
            name="Fetch FDA Updates",
            replace_existing=True,
            max_instances=1
        )
        
        # ============================================================
        # Perplexity Market Scans (Every 30 minutes)
        # ============================================================
        self.scheduler.add_job(
            self._run_perplexity_scans,
            trigger=IntervalTrigger(
                minutes=schedule_config.get("perplexity", {}).get("market_scan_interval", 1800) // 60
            ),
            id="perplexity_scans",
            name="Run Perplexity Market Scans",
            replace_existing=True,
            max_instances=1
        )
        
        # ============================================================
        # Statistics Report (Every hour)
        # ============================================================
        self.scheduler.add_job(
            self._log_statistics,
            trigger=IntervalTrigger(hours=1),
            id="statistics_report",
            name="Log Statistics",
            replace_existing=True
        )
        
        logger.info("✅ All schedules configured")
    
    async def _fetch_major_news(self):
        """Fetch major financial news (Tier 1)"""
        try:
            logger.info("📰 Fetching major news RSS feeds...")
            articles = await self.data_loader.fetch_all_rss_feeds(tier="major_news")
            
            await self._process_articles(articles, "major_news")
            
        except Exception as e:
            logger.error(f"❌ Major news fetch failed: {e}")
    
    async def _fetch_market_specific(self):
        """Fetch market-specific news (Tier 2)"""
        try:
            logger.info("📰 Fetching market-specific RSS feeds...")
            articles = await self.data_loader.fetch_all_rss_feeds(tier="market_specific")
            
            await self._process_articles(articles, "market_specific")
            
        except Exception as e:
            logger.error(f"❌ Market-specific fetch failed: {e}")
    
    async def _fetch_sector_specific(self):
        """Fetch sector-specific news (Tier 3)"""
        try:
            logger.info("📰 Fetching sector-specific RSS feeds...")
            
            # Fetch all sector feeds
            sectors = ["sector_technology", "sector_biotech", "sector_energy", 
                      "sector_finance", "sector_retail", "sector_automotive"]
            
            all_articles = []
            for sector in sectors:
                articles = await self.data_loader.fetch_all_rss_feeds(tier=sector)
                all_articles.extend(articles)
            
            await self._process_articles(all_articles, "sector_specific")
            
        except Exception as e:
            logger.error(f"❌ Sector-specific fetch failed: {e}")
    
    async def _fetch_sec_filings(self):
        """Fetch SEC filings"""
        try:
            logger.info("📋 Fetching SEC filings...")
            filings = await self.data_loader.fetch_sec_filings()
            
            await self._process_articles(filings, "sec_filings")
            
        except Exception as e:
            logger.error(f"❌ SEC filings fetch failed: {e}")
    
    async def _fetch_fda_updates(self):
        """Fetch FDA updates"""
        try:
            logger.info("💊 Fetching FDA updates...")
            updates = await self.data_loader.fetch_fda_approvals()
            
            await self._process_articles(updates, "fda_updates")
            
        except Exception as e:
            logger.error(f"❌ FDA updates fetch failed: {e}")
    
    async def _run_perplexity_scans(self):
        """Run Perplexity AI market scans"""
        try:
            logger.info("🤖 Running Perplexity market scans...")
            scans = await self.data_loader.fetch_perplexity_market_scan()
            
            await self._process_articles(scans, "perplexity_scans")
            
        except Exception as e:
            logger.error(f"❌ Perplexity scans failed: {e}")
    
    async def _process_articles(self, articles: List[Dict[str, Any]], source_type: str):
        """
        Process fetched articles:
        1. Keyword analysis
        2. Alert determination
        3. Perplexity deep analysis (for high-priority)
        4. WebSocket broadcast
        5. Database storage
        """
        if not articles:
            logger.debug(f"No articles from {source_type}")
            return
        
        logger.info(f"🔍 Processing {len(articles)} articles from {source_type}")
        
        # ============================================================
        # STEP 1: Keyword Analysis
        # ============================================================
        analyzed_articles = analyze_articles_batch(articles, self.keywords_engine)
        
        # ============================================================
        # STEP 2: Filter by Alert Threshold
        # ============================================================
        # Only process "watch" level and above
        alertable = self.keywords_engine.filter_articles_by_threshold(analyzed_articles, "watch")
        
        if not alertable:
            logger.debug(f"No articles from {source_type} met alert threshold")
            self.stats["total_articles_fetched"] += len(articles)
            return
        
        logger.info(f"🚨 {len(alertable)} articles triggered alerts")
        
        # ============================================================
        # STEP 3: Perplexity Deep Analysis (for important/critical only)
        # ============================================================
        for article in alertable:
            alert_level = article.get("alert_level", "none")
            
            # Deep analysis for important/critical alerts
            if alert_level in ["important", "critical"]:
                await self._run_perplexity_analysis(article)
            
            # ============================================================
            # STEP 4: Trigger Alert
            # ============================================================
            await self._trigger_alert(article)
        
        # Update statistics
        self.stats["total_articles_fetched"] += len(articles)
        self.stats["total_alerts_fired"] += len(alertable)
        self.stats["last_run"] = datetime.now(timezone.utc).isoformat()
        self.stats["runs_count"] += 1
    
    async def _run_perplexity_analysis(self, article: Dict[str, Any]):
        """Run Perplexity AI deep analysis on article"""
        try:
            symbols = article.get("symbols", [])
            if not symbols:
                logger.debug("No symbols found, skipping Perplexity analysis")
                return
            
            # Analyze first symbol (primary)
            primary_symbol = symbols[0]
            
            logger.info(f"🤖 Running Perplexity analysis for {primary_symbol}...")
            
            # Use event_analysis prompt
            query = f"""
An event was detected:
- Symbol: {primary_symbol}
- Headline: {article.get('title')}
- Source: {article.get('source_name')}
- Keywords matched: {', '.join([m['keyword'] for m in article.get('keyword_matches', [])[:5]])}
- Sentiment score: {article.get('keyword_score')}

Provide:
1. Immediate impact assessment (bullish/bearish/neutral)
2. Price target implications
3. Key risks or opportunities
4. Recommended action (BUY/SELL/HOLD)
5. Confidence level (0-1)
"""
            
            result = await self.perplexity.analyze_async(
                symbol=primary_symbol,
                user_query=query,
                analysis_type="event_analysis"
            )
            
            if result:
                article["perplexity_analysis"] = result.get("analysis", "")
                article["perplexity_recommendation"] = result.get("recommendation", "")
                article["perplexity_confidence"] = result.get("confidence", 0.0)
                article["perplexity_citations"] = result.get("citations", [])
                
                logger.info(f"✅ Perplexity analysis complete for {primary_symbol}")
            
        except Exception as e:
            logger.error(f"❌ Perplexity analysis failed: {e}")
    
    async def _trigger_alert(self, article: Dict[str, Any]):
        """
        Trigger multi-channel alert based on priority
        
        Alert Levels:
        - critical: WhatsApp + Telegram + Dashboard + Email
        - important: Telegram + Dashboard
        - watch: Dashboard only
        """
        alert_level = article.get("alert_level", "none")
        
        # Check if already alerted (deduplication)
        alert_hash = f"{article.get('url', '')}_{alert_level}"
        if alert_hash in self.alert_cache:
            logger.debug(f"Duplicate alert skipped: {article.get('title', '')[:50]}")
            return
        
        self.alert_cache.add(alert_hash)
        
        # Cleanup old cache entries (keep last 1000)
        if len(self.alert_cache) > 1000:
            self.alert_cache = set(list(self.alert_cache)[-1000:])
        
        # ============================================================
        # Format Alert Message
        # ============================================================
        alert_msg = self._format_alert_message(article)
        
        # ============================================================
        # Send to Channels Based on Priority
        # ============================================================
        if alert_level == "critical":
            # TODO: Send WhatsApp (Twilio)
            # TODO: Send Telegram
            # TODO: Send Email
            await self._broadcast_dashboard(article, "🔴 CRITICAL")
            logger.info(f"🔴 CRITICAL alert fired: {article.get('title', '')[:60]}")
        
        elif alert_level == "important":
            # TODO: Send Telegram
            await self._broadcast_dashboard(article, "🟡 IMPORTANT")
            logger.info(f"🟡 IMPORTANT alert fired: {article.get('title', '')[:60]}")
        
        elif alert_level == "watch":
            await self._broadcast_dashboard(article, "🟢 WATCH")
            logger.info(f"🟢 WATCH alert logged: {article.get('title', '')[:60]}")
    
    def _format_alert_message(self, article: Dict[str, Any]) -> str:
        """Format alert message for notifications"""
        symbols = article.get("symbols", [])
        symbol_str = ", ".join(symbols[:3]) if symbols else "MARKET"
        
        title = article.get("title", "")
        source = article.get("source_name", "Unknown")
        score = article.get("keyword_score", 0)
        sentiment = article.get("sentiment", "neutral")
        
        msg = f"""
🔔 {article.get('alert_level', 'ALERT').upper()}

Symbol: {symbol_str}
{title}

Sentiment: {sentiment} (Score: {score:+.2f})
Source: {source}
        """
        
        # Add Perplexity analysis if available
        if "perplexity_recommendation" in article:
            msg += f"\nAI Recommendation: {article['perplexity_recommendation']}"
        
        return msg.strip()
    
    async def _broadcast_dashboard(self, article: Dict[str, Any], alert_prefix: str):
        """Broadcast alert to dashboard via WebSocket"""
        if not self.websocket_broadcast_callback:
            logger.debug("WebSocket callback not set, skipping dashboard broadcast")
            return
        
        try:
            # Prepare WebSocket message
            ws_message = {
                "type": "alert",
                "level": article.get("alert_level", "watch"),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "title": f"{alert_prefix}: {article.get('title', '')}",
                "symbols": article.get("symbols", []),
                "score": article.get("keyword_score", 0),
                "sentiment": article.get("sentiment", "neutral"),
                "source": article.get("source_name", "Unknown"),
                "url": article.get("url", ""),
                "keywords": [m["keyword"] for m in article.get("keyword_matches", [])[:5]],
                "perplexity_analysis": article.get("perplexity_analysis", "")
            }
            
            # Broadcast to all connected clients
            await self.websocket_broadcast_callback(ws_message)
            
        except Exception as e:
            logger.error(f"❌ WebSocket broadcast failed: {e}")
    
    async def _log_statistics(self):
        """Log scheduler statistics"""
        logger.info("=" * 80)
        logger.info("📊 SCHEDULER STATISTICS")
        logger.info(f"   Total articles fetched: {self.stats['total_articles_fetched']}")
        logger.info(f"   Total alerts fired: {self.stats['total_alerts_fired']}")
        logger.info(f"   Runs count: {self.stats['runs_count']}")
        logger.info(f"   Last run: {self.stats['last_run']}")
        logger.info(f"   Alert cache size: {len(self.alert_cache)}")
        logger.info("=" * 80)
    
    def start(self):
        """Start the scheduler"""
        self.setup_schedules()
        self.scheduler.start()
        logger.info("✅ MarketPulse scheduler started!")
        logger.info(f"   Jobs configured: {len(self.scheduler.get_jobs())}")
        
        # Print schedule
        for job in self.scheduler.get_jobs():
            logger.info(f"   - {job.name}: {job.trigger}")
    
    def stop(self):
        """Stop the scheduler"""
        self.scheduler.shutdown()
        logger.info("⏹️  MarketPulse scheduler stopped")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current statistics"""
        return {
            **self.stats,
            "jobs": [
                {
                    "id": job.id,
                    "name": job.name,
                    "next_run": job.next_run_time.isoformat() if job.next_run_time else None
                }
                for job in self.scheduler.get_jobs()
            ]
        }


# Singleton instance
_scheduler_instance = None

def get_scheduler() -> MarketPulseScheduler:
    """Get or create scheduler singleton"""
    global _scheduler_instance
    if _scheduler_instance is None:
        _scheduler_instance = MarketPulseScheduler()
    return _scheduler_instance


async def main():
    """Test the scheduler"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("🧪 Testing MarketPulse Scheduler")
    
    scheduler = get_scheduler()
    scheduler.start()
    
    logger.info("\n⏰ Scheduler is running. Jobs:")
    for job in scheduler.scheduler.get_jobs():
        logger.info(f"   {job.name}: next run at {job.next_run_time}")
    
    logger.info("\n⏳ Running for 60 seconds (press Ctrl+C to stop)...")
    
    try:
        await asyncio.sleep(60)
    except KeyboardInterrupt:
        logger.info("\n⏹️  Stopping scheduler...")
    finally:
        scheduler.stop()
        logger.info("✅ Test complete")


if __name__ == "__main__":
    asyncio.run(main())
