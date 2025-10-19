"""
Database models for Tariff Radar system
SQLAlchemy models for articles, sources, and alerts
"""
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, List, Any
from sqlalchemy import Column, Integer, String, Text, DateTime, Float, Boolean, JSON, ForeignKey, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, Session
from pydantic import BaseModel

Base = declarative_base()


class ArticleStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    ARCHIVED = "archived"


class AlertStatus(str, Enum):
    PENDING = "pending"
    SENT = "sent"
    FAILED = "failed"


class Source(Base):
    """Data sources configuration"""
    __tablename__ = "sources"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), unique=True, nullable=False, index=True)
    url = Column(Text, nullable=False)
    source_type = Column(String(50), nullable=False)  # rss, scraper, api
    priority = Column(String(20), default="medium")  # high, medium, low
    config = Column(JSON, default={})
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship
    articles = relationship("Article", back_populates="source")


class Article(Base):
    """Main articles table"""
    __tablename__ = "articles"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(Text, nullable=False)
    normalized_title = Column(Text, nullable=False, index=True)
    content = Column(Text)
    summary = Column(Text)
    url = Column(Text, nullable=False)
    canonical_url = Column(String(500), index=True)
    
    # Source info
    source_id = Column(Integer, ForeignKey("sources.id"), nullable=False)
    published_at = Column(DateTime, index=True)
    discovered_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Content analysis
    language = Column(String(10), default="zh")
    content_hash = Column(String(64), unique=True, index=True)
    
    # AI Analysis results
    keyword_score = Column(Float, default=0.0)
    semantic_score = Column(Float, default=0.0)
    classifier_score = Column(Float, default=0.0)
    llm_score = Column(Float, default=0.0)
    final_score = Column(Float, default=0.0, index=True)
    
    # LLM Analysis
    llm_relevant = Column(Boolean)
    llm_summary = Column(Text)
    llm_tags = Column(JSON, default=[])
    llm_confidence = Column(Float)
    
    # Status & metadata
    status = Column(String(20), default=ArticleStatus.PENDING, index=True)
    is_duplicate = Column(Boolean, default=False, index=True)
    duplicate_of = Column(Integer, ForeignKey("articles.id"))
    
    # Change tracking
    version = Column(Integer, default=1)
    previous_version_id = Column(Integer, ForeignKey("articles.id"))
    change_summary = Column(Text)
    
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    source = relationship("Source", back_populates="articles")
    alerts = relationship("Alert", back_populates="article")
    
    # Indexes
    __table_args__ = (
        Index('idx_article_score_status', 'final_score', 'status'),
        Index('idx_article_source_date', 'source_id', 'published_at'),
        Index('idx_article_discovery_score', 'discovered_at', 'final_score'),
    )


class Alert(Base):
    """Alert/notification tracking"""
    __tablename__ = "alerts"
    
    id = Column(Integer, primary_key=True, index=True)
    article_id = Column(Integer, ForeignKey("articles.id"), nullable=False)
    
    # Alert details
    alert_type = Column(String(50), nullable=False)  # immediate, batched, manual
    channel = Column(String(50), nullable=False)     # wecom, email, telegram
    recipient = Column(String(255))
    
    # Message content
    subject = Column(Text)
    message = Column(Text)
    formatted_message = Column(Text)  # Final formatted version
    
    # Status tracking
    status = Column(String(20), default=AlertStatus.PENDING, index=True)
    sent_at = Column(DateTime)
    error_message = Column(Text)
    retry_count = Column(Integer, default=0)
    
    # External IDs for tracking
    external_id = Column(String(255))  # WeChat message ID, email ID, etc.
    
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    article = relationship("Article", back_populates="alerts")


class ProcessingLog(Base):
    """Processing pipeline logs"""
    __tablename__ = "processing_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    article_id = Column(Integer, ForeignKey("articles.id"))
    step = Column(String(50), nullable=False)  # ingestion, normalization, analysis, etc.
    status = Column(String(20), nullable=False)  # success, error, warning
    
    message = Column(Text)
    details = Column(JSON, default={})
    processing_time = Column(Float)  # seconds
    
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


# Pydantic Models for API

class ArticleBase(BaseModel):
    title: str
    content: Optional[str] = None
    url: str
    language: Optional[str] = "zh"


class ArticleCreate(ArticleBase):
    source_id: int
    published_at: Optional[datetime] = None


class ArticleResponse(ArticleBase):
    id: int
    normalized_title: str
    source_id: int
    published_at: Optional[datetime]
    discovered_at: datetime
    
    keyword_score: float
    semantic_score: float
    classifier_score: float
    final_score: float
    
    status: ArticleStatus
    is_duplicate: bool
    
    llm_relevant: Optional[bool]
    llm_summary: Optional[str]
    llm_tags: List[str]
    llm_confidence: Optional[float]
    
    created_at: datetime
    updated_at: datetime
    
    class Config:
        orm_mode = True


class SourceBase(BaseModel):
    name: str
    url: str
    source_type: str
    priority: str = "medium"
    config: Dict[str, Any] = {}
    is_active: bool = True


class SourceCreate(SourceBase):
    pass


class SourceResponse(SourceBase):
    id: int
    created_at: datetime
    updated_at: datetime
    
    class Config:
        orm_mode = True


class AlertCreate(BaseModel):
    article_id: int
    alert_type: str
    channel: str
    recipient: Optional[str] = None
    subject: Optional[str] = None
    message: Optional[str] = None


class AlertResponse(BaseModel):
    id: int
    article_id: int
    alert_type: str
    channel: str
    status: AlertStatus
    sent_at: Optional[datetime]
    created_at: datetime
    
    class Config:
        orm_mode = True


# Database utility functions

def get_or_create_source(db: Session, name: str, **kwargs) -> Source:
    """Get existing source or create new one"""
    source = db.query(Source).filter(Source.name == name).first()
    if not source:
        source = Source(name=name, **kwargs)
        db.add(source)
        db.commit()
        db.refresh(source)
    return source


def create_article(db: Session, article: ArticleCreate, content_hash: str) -> Article:
    """Create new article with deduplication check"""
    # Check for duplicates
    existing = db.query(Article).filter(Article.content_hash == content_hash).first()
    if existing:
        existing.is_duplicate = True
        return existing
    
    db_article = Article(
        **article.dict(),
        content_hash=content_hash,
        normalized_title=normalize_title(article.title)
    )
    db.add(db_article)
    db.commit()
    db.refresh(db_article)
    return db_article


def normalize_title(title: str) -> str:
    """Normalize title for deduplication"""
    import re
    # Remove extra spaces, punctuation, convert to lowercase
    normalized = re.sub(r'[^\w\s]', '', title.lower())
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    return normalized