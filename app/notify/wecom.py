"""
WeCom (企业微信) notification handler
Sends alerts through WeChat Work API for Chinese government/business communications
"""
import logging
import json
import asyncio
from typing import Dict, Any, List, Optional
import httpx
import os
from datetime import datetime

logger = logging.getLogger(__name__)


class WeCom:
    """WeChat Work (WeCom) notification client"""
    
    def __init__(self, config: Dict[str, Any]):
        self.corp_id = os.getenv("WECOM_CORP_ID", "")
        self.agent_id = os.getenv("WECOM_AGENT_ID", "")
        self.corp_secret = os.getenv("WECOM_CORP_SECRET", "")
        self.to_user = config.get("to_user", "@all")
        self.message_format = config.get("message_format", "markdown")
        
        # API endpoints
        self.base_url = "https://qyapi.weixin.qq.com/cgi-bin"
        self.access_token = None
        self.token_expires_at = None
        
        # Rate limiting
        self.max_requests_per_minute = 1000  # WeCom limit
        self.request_count = 0
        self.last_reset_time = datetime.now()
        
        # Validate configuration
        if not all([self.corp_id, self.agent_id, self.corp_secret]):
            logger.warning("WeCom configuration incomplete - notifications will be disabled")
    
    async def get_access_token(self) -> Optional[str]:
        """Get WeCom access token"""
        # Check if current token is still valid
        if (self.access_token and self.token_expires_at and 
            datetime.now() < self.token_expires_at):
            return self.access_token
        
        try:
            url = f"{self.base_url}/gettoken"
            params = {
                "corpid": self.corp_id,
                "corpsecret": self.corp_secret
            }
            
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                
                data = response.json()
                
                if data.get("errcode") == 0:
                    self.access_token = data["access_token"]
                    # Token expires in 7200 seconds, we'll refresh earlier
                    expires_in = data.get("expires_in", 7200) - 300  # 5 min buffer
                    self.token_expires_at = datetime.now().timestamp() + expires_in
                    
                    logger.info("Successfully obtained WeCom access token")
                    return self.access_token
                else:
                    logger.error(f"WeCom API error: {data.get('errmsg')}")
                    return None
                    
        except Exception as e:
            logger.error(f"Failed to get WeCom access token: {e}")
            return None
    
    async def send_message(self, article: Dict[str, Any]) -> bool:
        """Send article alert to WeCom"""
        if not self._is_configured():
            logger.warning("WeCom not configured, skipping notification")
            return False
        
        # Rate limiting check
        if not self._check_rate_limit():
            logger.warning("WeCom rate limit exceeded, skipping notification")
            return False
        
        try:
            access_token = await self.get_access_token()
            if not access_token:
                return False
            
            # Format message
            message_content = self._format_message(article)
            
            # Prepare message payload
            message_data = {
                "touser": self.to_user,
                "msgtype": self.message_format,
                "agentid": int(self.agent_id),
            }
            
            if self.message_format == "markdown":
                message_data["markdown"] = {"content": message_content}
            else:
                message_data["text"] = {"content": message_content}
            
            # Send message
            url = f"{self.base_url}/message/send"
            params = {"access_token": access_token}
            
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(
                    url, 
                    params=params, 
                    json=message_data,
                    headers={"Content-Type": "application/json"}
                )
                response.raise_for_status()
                
                result = response.json()
                
                if result.get("errcode") == 0:
                    logger.info(f"Successfully sent WeCom notification for article: {article.get('title', '')[:50]}")
                    return True
                else:
                    logger.error(f"WeCom message send error: {result.get('errmsg')}")
                    return False
                    
        except Exception as e:
            logger.error(f"Failed to send WeCom message: {e}")
            return False
    
    def _format_message(self, article: Dict[str, Any]) -> str:
        """Format article for WeCom message"""
        title = article.get("title", "No Title")
        source = article.get("source_name", "Unknown")
        url = article.get("url", "")
        score = article.get("final_score", 0)
        
        # Get LLM analysis if available
        summary = article.get("llm_summary", "")
        tags = article.get("llm_tags", [])
        confidence = article.get("llm_confidence", 0)
        
        # Publication date
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
        
        # Priority indicator
        priority_emoji = "🚨" if score >= 0.8 else "⚠️" if score >= 0.6 else "ℹ️"
        
        # Tags formatting
        tags_str = " | ".join(tags) if tags else "General"
        
        if self.message_format == "markdown":
            # Markdown format for rich display
            message = f"""**{priority_emoji} 关税贸易监控提醒**

**标题:** {title}

**来源:** {source} | {date_str}
**评分:** {score:.2f} | **置信度:** {confidence:.2f}
**标签:** {tags_str}

**摘要:** {summary}

**操作:**
- [查看原文]({url})
- [打开监控面板](http://localhost:8000/dashboard)

---
*Tariff Radar Alert System*"""

        else:
            # Plain text format
            message = f"""【{priority_emoji} 关税贸易监控】

{title}

来源: {source} | {date_str}
评分: {score:.2f} | 置信度: {confidence:.2f}
标签: {tags_str}

摘要: {summary}

查看详情: {url}
控制台: http://localhost:8000/dashboard"""
        
        return message
    
    async def send_batch_summary(self, articles: List[Dict[str, Any]], batch_type: str = "daily") -> bool:
        """Send batch summary of articles"""
        if not articles or not self._is_configured():
            return False
        
        try:
            access_token = await self.get_access_token()
            if not access_token:
                return False
            
            # Create summary message
            summary_message = self._format_batch_summary(articles, batch_type)
            
            message_data = {
                "touser": self.to_user,
                "msgtype": "markdown",
                "agentid": int(self.agent_id),
                "markdown": {"content": summary_message}
            }
            
            url = f"{self.base_url}/message/send"
            params = {"access_token": access_token}
            
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(url, params=params, json=message_data)
                response.raise_for_status()
                
                result = response.json()
                return result.get("errcode") == 0
                
        except Exception as e:
            logger.error(f"Failed to send batch summary: {e}")
            return False
    
    def _format_batch_summary(self, articles: List[Dict[str, Any]], batch_type: str) -> str:
        """Format batch summary message"""
        total_count = len(articles)
        high_priority = len([a for a in articles if a.get("final_score", 0) >= 0.8])
        medium_priority = len([a for a in articles if 0.6 <= a.get("final_score", 0) < 0.8])
        
        # Group by tags
        tag_counts = {}
        for article in articles:
            tags = article.get("llm_tags", [])
            for tag in tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        # Top tags
        top_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        tags_summary = " | ".join([f"{tag}({count})" for tag, count in top_tags])
        
        # Sample high-priority articles
        high_priority_articles = [a for a in articles if a.get("final_score", 0) >= 0.8][:3]
        articles_list = "\n".join([
            f"• {a.get('title', '')[:60]}... ({a.get('final_score', 0):.2f})"
            for a in high_priority_articles
        ])
        
        message = f"""**📊 {batch_type.title()} 关税贸易监控汇总**

**统计信息:**
- 总文章数: {total_count}
- 高优先级: {high_priority} 🚨
- 中优先级: {medium_priority} ⚠️

**主要话题:** {tags_summary}

**重要文章:**
{articles_list}

**操作:**
[查看详细报告](http://localhost:8000/dashboard)

---
*{datetime.now().strftime("%Y-%m-%d %H:%M")} | Tariff Radar*"""
        
        return message
    
    async def send_test_message(self) -> bool:
        """Send a test message to verify configuration"""
        if not self._is_configured():
            logger.error("WeCom not configured for testing")
            return False
        
        test_article = {
            "title": "WeCom 测试消息 - Tariff Radar 系统正常运行",
            "source_name": "System",
            "url": "http://localhost:8000",
            "final_score": 0.9,
            "llm_summary": "这是一条系统测试消息，用于验证 WeCom 通知功能是否正常工作。",
            "llm_tags": ["Test", "System"],
            "llm_confidence": 1.0,
            "published_at": datetime.now()
        }
        
        result = await self.send_message(test_article)
        
        if result:
            logger.info("WeCom test message sent successfully")
        else:
            logger.error("WeCom test message failed")
        
        return result
    
    def _is_configured(self) -> bool:
        """Check if WeCom is properly configured"""
        return bool(self.corp_id and self.agent_id and self.corp_secret)
    
    def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits"""
        now = datetime.now()
        
        # Reset counter every minute
        if (now - self.last_reset_time).total_seconds() >= 60:
            self.request_count = 0
            self.last_reset_time = now
        
        if self.request_count >= self.max_requests_per_minute:
            return False
        
        self.request_count += 1
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """Get WeCom client status"""
        return {
            "configured": self._is_configured(),
            "has_access_token": bool(self.access_token),
            "token_expires_at": self.token_expires_at,
            "requests_this_minute": self.request_count,
            "corp_id": self.corp_id[:6] + "***" if self.corp_id else None,
            "agent_id": self.agent_id
        }


async def main():
    """Test WeCom functionality"""
    config = {
        "to_user": "@all",
        "message_format": "markdown"
    }
    
    # Test message
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
    
    wecom = WeCom(config)
    
    print("WeCom Status:")
    print(wecom.get_status())
    print()
    
    if wecom._is_configured():
        print("Sending test message...")
        result = await wecom.send_test_message()
        print(f"Test result: {result}")
    else:
        print("WeCom not configured - set environment variables:")
        print("- WECOM_CORP_ID")
        print("- WECOM_AGENT_ID") 
        print("- WECOM_CORP_SECRET")


if __name__ == "__main__":
    asyncio.run(main())