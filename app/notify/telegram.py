"""
Telegram notification handler
Sends alerts through Telegram Bot API
"""
import logging
import asyncio
from typing import Dict, Any, List, Optional
import httpx
import os
from datetime import datetime

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """Telegram Bot notification handler"""
    
    def __init__(self, config: Dict[str, Any]):
        self.bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
        self.parse_mode = config.get("parse_mode", "Markdown")
        
        # API configuration
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"
        self.timeout = 30
        
        # Rate limiting (Telegram limits: 30 messages/second to different chats)
        self.max_messages_per_second = 20
        self.last_message_time = 0
        
        if not self.bot_token or not self.chat_id:
            logger.warning("Telegram configuration incomplete - notifications will be disabled")
    
    async def send_alert(self, article: Dict[str, Any]) -> bool:
        """Send single article alert"""
        if not self._is_configured():
            logger.warning("Telegram not configured, skipping notification")
            return False
        
        try:
            # Rate limiting
            await self._rate_limit()
            
            # Format message
            message = self._format_message(article)
            
            # Send message
            return await self._send_message(message)
            
        except Exception as e:
            logger.error(f"Failed to send Telegram alert: {e}")
            return False
    
    async def send_batch_summary(self, articles: List[Dict[str, Any]], batch_type: str = "daily") -> bool:
        """Send batch summary message"""
        if not articles or not self._is_configured():
            return False
        
        try:
            await self._rate_limit()
            
            # Format batch message
            message = self._format_batch_message(articles, batch_type)
            
            # Send message
            return await self._send_message(message)
            
        except Exception as e:
            logger.error(f"Failed to send Telegram batch summary: {e}")
            return False
    
    async def send_test_message(self) -> bool:
        """Send test message"""
        if not self._is_configured():
            logger.error("Telegram not configured for testing")
            return False
        
        test_message = f"""🤖 *Tariff Radar Test*

This is a test message to verify Telegram notifications are working.

*System Status:* ✅ Operational
*Time:* `{datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")}`

[Dashboard](http://localhost:8000/dashboard)"""
        
        result = await self._send_message(test_message)
        
        if result:
            logger.info("Telegram test message sent successfully")
        else:
            logger.error("Telegram test message failed")
        
        return result
    
    def _format_message(self, article: Dict[str, Any]) -> str:
        """Format article for Telegram message"""
        title = article.get("title", "No Title")
        source = article.get("source_name", "Unknown")
        url = article.get("url", "")
        score = article.get("final_score", 0)
        summary = article.get("llm_summary", "No summary available")
        tags = article.get("llm_tags", [])
        confidence = article.get("llm_confidence", 0)
        
        # Priority emoji
        if score >= 0.8:
            priority_emoji = "🚨"
            priority_text = "HIGH PRIORITY"
        elif score >= 0.6:
            priority_emoji = "⚠️"
            priority_text = "MEDIUM PRIORITY"
        else:
            priority_emoji = "ℹ️"
            priority_text = "LOW PRIORITY"
        
        # Format publication date
        pub_date = article.get("published_at", "")
        if pub_date:
            try:
                if isinstance(pub_date, str):
                    pub_date = datetime.fromisoformat(pub_date.replace('Z', '+00:00'))
                date_str = pub_date.strftime("%Y-%m-%d %H:%M")
            except:
                date_str = str(pub_date)[:16]
        else:
            date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        # Format tags
        tags_str = " ".join([f"#{tag}" for tag in tags]) if tags else "#General"
        
        # Escape special characters for Markdown
        def escape_markdown(text):
            special_chars = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
            for char in special_chars:
                text = text.replace(char, f'\\{char}')
            return text
        
        # Truncate title and summary for Telegram limits
        title_truncated = title[:80] + "..." if len(title) > 80 else title
        summary_truncated = summary[:200] + "..." if len(summary) > 200 else summary
        
        message = f"""{priority_emoji} *{priority_text}*

*{escape_markdown(title_truncated)}*

*Source:* {escape_markdown(source)}
*Date:* `{date_str}`
*Score:* `{score:.2f}` \\| *Confidence:* `{confidence:.2f}`

*Summary:*
{escape_markdown(summary_truncated)}

*Tags:* {escape_markdown(tags_str)}

[📄 Read Original]({url}) \\| [📊 Dashboard](http://localhost:8000/dashboard)"""
        
        return message
    
    def _format_batch_message(self, articles: List[Dict[str, Any]], batch_type: str) -> str:
        """Format batch summary message"""
        total_count = len(articles)
        high_priority = len([a for a in articles if a.get("final_score", 0) >= 0.8])
        medium_priority = len([a for a in articles if 0.6 <= a.get("final_score", 0) < 0.8])
        
        # Group by tags
        tag_counts = {}
        for article in articles:
            for tag in article.get("llm_tags", []):
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        # Top tags
        top_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        tags_summary = " ".join([f"#{tag}({count})" for tag, count in top_tags])
        
        # Top articles
        top_articles = sorted(articles, key=lambda x: x.get("final_score", 0), reverse=True)[:3]
        
        def escape_markdown(text):
            special_chars = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
            for char in special_chars:
                text = text.replace(char, f'\\{char}')
            return text
        
        message = f"""📊 *{batch_type.title()} Tariff Radar Report*

*Summary:*
• Total Articles: `{total_count}`
• High Priority: `{high_priority}` 🚨
• Medium Priority: `{medium_priority}` ⚠️
• Low Priority: `{total_count - high_priority - medium_priority}` ℹ️

*Top Topics:*
{escape_markdown(tags_summary)}

*Key Articles:*"""
        
        for i, article in enumerate(top_articles, 1):
            title = article.get("title", "No Title")
            score = article.get("final_score", 0)
            
            priority_emoji = "🚨" if score >= 0.8 else "⚠️" if score >= 0.6 else "ℹ️"
            title_short = title[:50] + "..." if len(title) > 50 else title
            
            message += f"\n{i}\\. {priority_emoji} [{escape_markdown(title_short)}]({article.get('url', '#')}) \\(`{score:.2f}`\\)"
        
        message += f"\n\n[📊 Full Dashboard](http://localhost:8000/dashboard)\n\n*Generated:* `{datetime.now().strftime('%Y-%m-%d %H:%M UTC')}`"
        
        return message
    
    async def _send_message(self, message: str, disable_web_page_preview: bool = True) -> bool:
        """Send message via Telegram Bot API"""
        try:
            url = f"{self.base_url}/sendMessage"
            
            payload = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": self.parse_mode,
                "disable_web_page_preview": disable_web_page_preview
            }
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                
                result = response.json()
                
                if result.get("ok"):
                    logger.debug("Telegram message sent successfully")
                    return True
                else:
                    logger.error(f"Telegram API error: {result.get('description')}")
                    return False
                    
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return False
    
    async def _rate_limit(self):
        """Apply rate limiting"""
        current_time = datetime.now().timestamp()
        
        # Ensure minimum interval between messages
        min_interval = 1.0 / self.max_messages_per_second
        time_since_last = current_time - self.last_message_time
        
        if time_since_last < min_interval:
            sleep_time = min_interval - time_since_last
            await asyncio.sleep(sleep_time)
        
        self.last_message_time = datetime.now().timestamp()
    
    def _is_configured(self) -> bool:
        """Check if Telegram is properly configured"""
        return bool(self.bot_token and self.chat_id)
    
    async def get_bot_info(self) -> Optional[Dict[str, Any]]:
        """Get bot information"""
        if not self.bot_token:
            return None
        
        try:
            url = f"{self.base_url}/getMe"
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url)
                response.raise_for_status()
                
                result = response.json()
                
                if result.get("ok"):
                    return result.get("result")
                else:
                    return None
                    
        except Exception as e:
            logger.error(f"Failed to get bot info: {e}")
            return None
    
    def get_status(self) -> Dict[str, Any]:
        """Get Telegram client status"""
        return {
            "configured": self._is_configured(),
            "bot_token": "***" + self.bot_token[-6:] if self.bot_token else None,
            "chat_id": self.chat_id,
            "parse_mode": self.parse_mode
        }


async def main():
    """Test Telegram functionality"""
    config = {
        "parse_mode": "MarkdownV2"
    }
    
    test_article = {
        "title": "国务院关税税则委员会发布对美加征关税商品清单",
        "source_name": "MOFCOM",
        "url": "http://example.com/article",
        "final_score": 0.92,
        "llm_summary": "中国宣布对美国商品实施新的关税措施，作为对美方301调查的回应。",
        "llm_tags": ["Retaliation", "301_Tariffs"],
        "llm_confidence": 0.95,
        "published_at": datetime.now()
    }
    
    notifier = TelegramNotifier(config)
    
    print("Telegram Status:")
    print(notifier.get_status())
    print()
    
    if notifier._is_configured():
        # Get bot info
        bot_info = await notifier.get_bot_info()
        if bot_info:
            print(f"Bot: @{bot_info.get('username')} ({bot_info.get('first_name')})")
        
        print("Sending test message...")
        result = await notifier.send_test_message()
        print(f"Test result: {result}")
    else:
        print("Telegram not configured - set environment variables:")
        print("- TELEGRAM_BOT_TOKEN")
        print("- TELEGRAM_CHAT_ID")


if __name__ == "__main__":
    asyncio.run(main())