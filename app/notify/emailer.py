"""
Email notification handler
Sends formatted email alerts with HTML content and attachments
"""
import logging
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from typing import Dict, Any, List, Optional
import os
from datetime import datetime
from jinja2 import Template
import json

logger = logging.getLogger(__name__)


class EmailNotifier:
    """Email notification handler"""
    
    def __init__(self, config: Dict[str, Any]):
        self.smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.smtp_user = os.getenv("SMTP_USER", "")
        self.smtp_password = os.getenv("SMTP_PASSWORD", "")
        self.email_to = os.getenv("EMAIL_TO", "")
        
        self.subject_prefix = config.get("subject_prefix", "[Tariff Alert]")
        self.from_name = config.get("from_name", "Tariff Radar")
        
        # Email templates
        self.html_template = self._get_html_template()
        self.text_template = self._get_text_template()
        
        # Validate configuration
        if not all([self.smtp_host, self.smtp_user, self.smtp_password, self.email_to]):
            logger.warning("Email configuration incomplete - notifications will be disabled")
    
    def send_alert(self, article: Dict[str, Any], alert_type: str = "immediate") -> bool:
        """Send email alert for single article"""
        if not self._is_configured():
            logger.warning("Email not configured, skipping notification")
            return False
        
        try:
            # Prepare email content
            subject = self._format_subject(article, alert_type)
            html_content = self._format_html_content(article)
            text_content = self._format_text_content(article)
            
            # Create message
            message = MIMEMultipart("alternative")
            message["Subject"] = subject
            message["From"] = f"{self.from_name} <{self.smtp_user}>"
            message["To"] = self.email_to
            message["X-Priority"] = "2" if article.get("final_score", 0) >= 0.8 else "3"
            
            # Add text and HTML parts
            text_part = MIMEText(text_content, "plain", "utf-8")
            html_part = MIMEText(html_content, "html", "utf-8")
            
            message.attach(text_part)
            message.attach(html_part)
            
            # Send email
            return self._send_email(message)
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            return False
    
    def send_batch_report(self, articles: List[Dict[str, Any]], report_type: str = "daily") -> bool:
        """Send batch report with multiple articles"""
        if not articles or not self._is_configured():
            return False
        
        try:
            # Prepare batch report content
            subject = f"{self.subject_prefix} {report_type.title()} Report - {len(articles)} Articles"
            html_content = self._format_batch_html(articles, report_type)
            text_content = self._format_batch_text(articles, report_type)
            
            # Create message
            message = MIMEMultipart("alternative")
            message["Subject"] = subject
            message["From"] = f"{self.from_name} <{self.smtp_user}>"
            message["To"] = self.email_to
            
            # Add content
            text_part = MIMEText(text_content, "plain", "utf-8")
            html_part = MIMEText(html_content, "html", "utf-8")
            
            message.attach(text_part)
            message.attach(html_part)
            
            # Optionally attach JSON report
            if len(articles) > 0:
                json_attachment = self._create_json_attachment(articles)
                if json_attachment:
                    message.attach(json_attachment)
            
            return self._send_email(message)
            
        except Exception as e:
            logger.error(f"Failed to send batch report: {e}")
            return False
    
    def send_test_email(self) -> bool:
        """Send test email to verify configuration"""
        if not self._is_configured():
            logger.error("Email not configured for testing")
            return False
        
        test_article = {
            "title": "Email Test - Tariff Radar System Operational",
            "source_name": "System Test",
            "url": "http://localhost:8000",
            "final_score": 0.9,
            "llm_summary": "This is a test message to verify email notification functionality is working correctly.",
            "llm_tags": ["Test", "System"],
            "llm_confidence": 1.0,
            "published_at": datetime.now()
        }
        
        result = self.send_alert(test_article, "test")
        
        if result:
            logger.info("Test email sent successfully")
        else:
            logger.error("Test email failed")
        
        return result
    
    def _format_subject(self, article: Dict[str, Any], alert_type: str) -> str:
        """Format email subject line"""
        title = article.get("title", "No Title")[:50]
        score = article.get("final_score", 0)
        
        # Priority indicator
        if score >= 0.8:
            priority = "🚨 HIGH"
        elif score >= 0.6:
            priority = "⚠️ MEDIUM"
        else:
            priority = "ℹ️ LOW"
        
        subject = f"{self.subject_prefix} {priority} - {title}"
        
        if len(subject) > 100:
            subject = subject[:97] + "..."
        
        return subject
    
    def _format_html_content(self, article: Dict[str, Any]) -> str:
        """Format HTML email content"""
        template = Template(self.html_template)
        
        # Prepare template variables
        template_vars = {
            "article": article,
            "title": article.get("title", "No Title"),
            "source": article.get("source_name", "Unknown"),
            "url": article.get("url", "#"),
            "score": article.get("final_score", 0),
            "summary": article.get("llm_summary", "No summary available"),
            "tags": article.get("llm_tags", []),
            "confidence": article.get("llm_confidence", 0),
            "published_at": article.get("published_at", datetime.now()),
            "dashboard_url": "http://localhost:8000/dashboard",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
        }
        
        # Format publication date
        pub_date = template_vars["published_at"]
        if isinstance(pub_date, str):
            try:
                pub_date = datetime.fromisoformat(pub_date.replace('Z', '+00:00'))
            except:
                pub_date = datetime.now()
        template_vars["formatted_date"] = pub_date.strftime("%Y-%m-%d %H:%M")
        
        # Priority styling
        if template_vars["score"] >= 0.8:
            template_vars["priority_class"] = "high-priority"
            template_vars["priority_text"] = "🚨 HIGH PRIORITY"
        elif template_vars["score"] >= 0.6:
            template_vars["priority_class"] = "medium-priority"
            template_vars["priority_text"] = "⚠️ MEDIUM PRIORITY"
        else:
            template_vars["priority_class"] = "low-priority"
            template_vars["priority_text"] = "ℹ️ LOW PRIORITY"
        
        return template.render(**template_vars)
    
    def _format_text_content(self, article: Dict[str, Any]) -> str:
        """Format plain text email content"""
        template = Template(self.text_template)
        
        template_vars = {
            "title": article.get("title", "No Title"),
            "source": article.get("source_name", "Unknown"),
            "url": article.get("url", "#"),
            "score": article.get("final_score", 0),
            "summary": article.get("llm_summary", "No summary available"),
            "tags": ", ".join(article.get("llm_tags", [])),
            "confidence": article.get("llm_confidence", 0),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
        }
        
        # Format date
        pub_date = article.get("published_at", datetime.now())
        if isinstance(pub_date, str):
            try:
                pub_date = datetime.fromisoformat(pub_date.replace('Z', '+00:00'))
            except:
                pub_date = datetime.now()
        template_vars["formatted_date"] = pub_date.strftime("%Y-%m-%d %H:%M")
        
        return template.render(**template_vars)
    
    def _format_batch_html(self, articles: List[Dict[str, Any]], report_type: str) -> str:
        """Format HTML batch report"""
        # Statistics
        total_count = len(articles)
        high_priority = len([a for a in articles if a.get("final_score", 0) >= 0.8])
        medium_priority = len([a for a in articles if 0.6 <= a.get("final_score", 0) < 0.8])
        low_priority = total_count - high_priority - medium_priority
        
        # Group by tags
        tag_counts = {}
        for article in articles:
            for tag in article.get("llm_tags", []):
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        # Sort articles by score
        sorted_articles = sorted(articles, key=lambda x: x.get("final_score", 0), reverse=True)
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Tariff Radar {report_type.title()} Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background: #2c3e50; color: white; padding: 20px; border-radius: 5px; }}
                .stats {{ background: #ecf0f1; padding: 15px; margin: 20px 0; border-radius: 5px; }}
                .article {{ border-left: 4px solid #3498db; padding: 15px; margin: 10px 0; background: #f9f9f9; }}
                .high-priority {{ border-left-color: #e74c3c; }}
                .medium-priority {{ border-left-color: #f39c12; }}
                .low-priority {{ border-left-color: #95a5a6; }}
                .score {{ font-weight: bold; color: #2c3e50; }}
                .tags {{ font-size: 0.9em; color: #7f8c8d; }}
                .footer {{ margin-top: 30px; padding-top: 20px; border-top: 1px solid #bdc3c7; font-size: 0.9em; color: #7f8c8d; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>📊 Tariff Radar {report_type.title()} Report</h1>
                <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")}</p>
            </div>
            
            <div class="stats">
                <h2>Summary Statistics</h2>
                <ul>
                    <li><strong>Total Articles:</strong> {total_count}</li>
                    <li><strong>High Priority (≥0.8):</strong> {high_priority} 🚨</li>
                    <li><strong>Medium Priority (0.6-0.8):</strong> {medium_priority} ⚠️</li>
                    <li><strong>Low Priority (<0.6):</strong> {low_priority} ℹ️</li>
                </ul>
                
                <h3>Top Tags</h3>
                <p>{" | ".join([f"{tag}({count})" for tag, count in sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10]])}</p>
            </div>
            
            <h2>Articles</h2>
        """
        
        for article in sorted_articles[:20]:  # Limit to top 20
            score = article.get("final_score", 0)
            if score >= 0.8:
                css_class = "high-priority"
            elif score >= 0.6:
                css_class = "medium-priority"
            else:
                css_class = "low-priority"
            
            tags_str = " | ".join(article.get("llm_tags", []))
            
            html += f"""
            <div class="article {css_class}">
                <h3><a href="{article.get('url', '#')}">{article.get('title', 'No Title')}</a></h3>
                <p><strong>Source:</strong> {article.get('source_name', 'Unknown')} | 
                   <span class="score">Score: {score:.2f}</span></p>
                <p>{article.get('llm_summary', 'No summary available')[:200]}...</p>
                <p class="tags"><strong>Tags:</strong> {tags_str}</p>
            </div>
            """
        
        html += f"""
            <div class="footer">
                <p>This report was generated by Tariff Radar automated monitoring system.</p>
                <p><a href="http://localhost:8000/dashboard">View Full Dashboard</a></p>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _format_batch_text(self, articles: List[Dict[str, Any]], report_type: str) -> str:
        """Format plain text batch report"""
        total_count = len(articles)
        high_priority = len([a for a in articles if a.get("final_score", 0) >= 0.8])
        medium_priority = len([a for a in articles if 0.6 <= a.get("final_score", 0) < 0.8])
        
        text = f"""TARIFF RADAR {report_type.upper()} REPORT
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")}

SUMMARY STATISTICS:
- Total Articles: {total_count}
- High Priority (≥0.8): {high_priority}
- Medium Priority (0.6-0.8): {medium_priority}
- Low Priority (<0.6): {total_count - high_priority - medium_priority}

TOP ARTICLES:
"""
        
        # Sort and show top articles
        sorted_articles = sorted(articles, key=lambda x: x.get("final_score", 0), reverse=True)
        for i, article in enumerate(sorted_articles[:10], 1):
            score = article.get("final_score", 0)
            priority = "🚨" if score >= 0.8 else "⚠️" if score >= 0.6 else "ℹ️"
            
            text += f"""
{i}. {priority} {article.get('title', 'No Title')}
   Source: {article.get('source_name', 'Unknown')} | Score: {score:.2f}
   Tags: {', '.join(article.get('llm_tags', []))}
   URL: {article.get('url', '#')}
   Summary: {article.get('llm_summary', 'No summary')[:150]}...

"""
        
        text += f"""
Full dashboard: http://localhost:8000/dashboard

---
Tariff Radar Automated Monitoring System
"""
        
        return text
    
    def _create_json_attachment(self, articles: List[Dict[str, Any]]) -> Optional[MIMEBase]:
        """Create JSON attachment with article data"""
        try:
            # Prepare data for JSON export
            export_data = {
                "generated_at": datetime.now().isoformat(),
                "total_articles": len(articles),
                "articles": []
            }
            
            for article in articles:
                export_data["articles"].append({
                    "title": article.get("title", ""),
                    "source": article.get("source_name", ""),
                    "url": article.get("url", ""),
                    "published_at": str(article.get("published_at", "")),
                    "final_score": article.get("final_score", 0),
                    "summary": article.get("llm_summary", ""),
                    "tags": article.get("llm_tags", []),
                    "confidence": article.get("llm_confidence", 0)
                })
            
            # Create attachment
            json_data = json.dumps(export_data, indent=2, ensure_ascii=False)
            
            attachment = MIMEBase('application', 'json')
            attachment.set_payload(json_data.encode('utf-8'))
            encoders.encode_base64(attachment)
            
            filename = f"tariff_radar_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            attachment.add_header('Content-Disposition', f'attachment; filename="{filename}"')
            
            return attachment
            
        except Exception as e:
            logger.error(f"Failed to create JSON attachment: {e}")
            return None
    
    def _send_email(self, message: MIMEMultipart) -> bool:
        """Send email using SMTP"""
        try:
            # Create SSL context
            context = ssl.create_default_context()
            
            # Connect and send
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls(context=context)
                server.login(self.smtp_user, self.smtp_password)
                server.send_message(message)
            
            logger.info(f"Email sent successfully to {self.email_to}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False
    
    def _is_configured(self) -> bool:
        """Check if email is properly configured"""
        return bool(self.smtp_host and self.smtp_user and self.smtp_password and self.email_to)
    
    def _get_html_template(self) -> str:
        """Get HTML email template"""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Tariff Radar Alert</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f4f4f4; }
                .container { max-width: 600px; margin: 0 auto; background: white; border-radius: 10px; overflow: hidden; box-shadow: 0 0 20px rgba(0,0,0,0.1); }
                .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; text-align: center; }
                .priority { padding: 15px; text-align: center; font-weight: bold; }
                .high-priority { background: #ff6b6b; color: white; }
                .medium-priority { background: #feca57; color: white; }
                .low-priority { background: #a4b0be; color: white; }
                .content { padding: 20px; }
                .article-title { font-size: 1.2em; font-weight: bold; color: #2c3e50; margin-bottom: 10px; }
                .meta { color: #7f8c8d; font-size: 0.9em; margin-bottom: 15px; }
                .summary { background: #ecf0f1; padding: 15px; border-radius: 5px; margin: 15px 0; }
                .tags { margin: 15px 0; }
                .tag { display: inline-block; background: #3498db; color: white; padding: 3px 8px; border-radius: 3px; margin: 2px; font-size: 0.8em; }
                .actions { text-align: center; margin: 20px 0; }
                .btn { display: inline-block; padding: 10px 20px; background: #3498db; color: white; text-decoration: none; border-radius: 5px; margin: 5px; }
                .btn:hover { background: #2980b9; }
                .footer { background: #34495e; color: white; padding: 15px; text-align: center; font-size: 0.9em; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🎯 Tariff Radar Alert</h1>
                    <p>US-China Trade Monitoring System</p>
                </div>
                
                <div class="priority {{ priority_class }}">
                    {{ priority_text }}
                </div>
                
                <div class="content">
                    <div class="article-title">{{ title }}</div>
                    <div class="meta">
                        <strong>Source:</strong> {{ source }} | 
                        <strong>Published:</strong> {{ formatted_date }} |
                        <strong>Score:</strong> {{ "%.2f"|format(score) }} |
                        <strong>Confidence:</strong> {{ "%.2f"|format(confidence) }}
                    </div>
                    
                    <div class="summary">
                        <strong>Summary:</strong><br>
                        {{ summary }}
                    </div>
                    
                    <div class="tags">
                        <strong>Tags:</strong><br>
                        {% for tag in tags %}
                        <span class="tag">{{ tag }}</span>
                        {% endfor %}
                    </div>
                    
                    <div class="actions">
                        <a href="{{ url }}" class="btn">📄 Read Original</a>
                        <a href="{{ dashboard_url }}" class="btn">📊 Dashboard</a>
                    </div>
                </div>
                
                <div class="footer">
                    Generated on {{ timestamp }}<br>
                    Tariff Radar Automated Monitoring System
                </div>
            </div>
        </body>
        </html>
        """
    
    def _get_text_template(self) -> str:
        """Get plain text email template"""
        return """
TARIFF RADAR ALERT
==================

{{ title }}

Source: {{ source }}
Published: {{ formatted_date }}
Score: {{ "%.2f"|format(score) }}
Confidence: {{ "%.2f"|format(confidence) }}

SUMMARY:
{{ summary }}

TAGS: {{ tags }}

LINKS:
- Original Article: {{ url }}
- Dashboard: http://localhost:8000/dashboard

---
Generated: {{ timestamp }}
Tariff Radar Automated Monitoring System
        """
    
    def get_status(self) -> Dict[str, Any]:
        """Get email client status"""
        return {
            "configured": self._is_configured(),
            "smtp_host": self.smtp_host,
            "smtp_port": self.smtp_port,
            "smtp_user": self.smtp_user[:5] + "***" if self.smtp_user else None,
            "email_to": self.email_to[:5] + "***" if self.email_to else None,
        }


def main():
    """Test email functionality"""
    config = {
        "subject_prefix": "[Tariff Alert]",
        "from_name": "Tariff Radar"
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
    
    notifier = EmailNotifier(config)
    
    print("Email Status:")
    print(notifier.get_status())
    print()
    
    if notifier._is_configured():
        print("Sending test email...")
        result = notifier.send_test_email()
        print(f"Test result: {result}")
    else:
        print("Email not configured - set environment variables:")
        print("- SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, EMAIL_TO")


if __name__ == "__main__":
    main()