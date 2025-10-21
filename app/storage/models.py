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


class StockPrediction(Base):
    """Stock predictions tracking for ML model improvement"""
    __tablename__ = "stock_predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Stock identification
    symbol = Column(String(20), nullable=False, index=True)  # AAPL, TSLA, etc.
    company_name = Column(String(255))
    
    # Prediction details (when discovered)
    prediction_date = Column(DateTime, default=datetime.utcnow, index=True)
    prediction_source = Column(String(50), nullable=False)  # hot_stocks, sector_scanner, ml_model
    
    # Price at prediction time
    price_at_prediction = Column(Float, nullable=False)
    
    # Prediction parameters
    predicted_direction = Column(String(10))  # up, down, neutral
    confidence_score = Column(Float)  # 0-100
    timeframe = Column(String(20))  # 1D, 1W, 1M, 3M
    target_price = Column(Float)  # Optional price target
    expected_return = Column(Float)  # Expected % return
    
    # Reasoning
    reason = Column(Text)  # Why was this stock flagged?
    sector = Column(String(100))  # Which sector triggered it
    keywords_matched = Column(JSON, default=[])  # Keywords that matched
    news_sentiment = Column(Float)  # Sentiment score from news
    
    # ML scores that led to prediction
    ml_score = Column(Float)
    technical_score = Column(Float)
    fundamental_score = Column(Float)
    
    # Related article (if came from news)
    article_id = Column(Integer, ForeignKey("articles.id"))
    
    # Outcome tracking (filled later)
    actual_price_1d = Column(Float)
    actual_price_1w = Column(Float)
    actual_price_1m = Column(Float)
    actual_price_3m = Column(Float)
    
    actual_return_1d = Column(Float)  # Actual % return after 1 day
    actual_return_1w = Column(Float)  # Actual % return after 1 week
    actual_return_1m = Column(Float)  # Actual % return after 1 month
    actual_return_3m = Column(Float)  # Actual % return after 3 months
    
    # Performance evaluation
    prediction_accuracy = Column(Float)  # 0-100, how close was the prediction
    was_correct = Column(Boolean)  # Did it move in predicted direction?
    max_gain_achieved = Column(Float)  # Highest % gain reached
    max_loss_suffered = Column(Float)  # Worst % loss reached
    
    # Status tracking
    status = Column(String(20), default="active", index=True)  # active, completed, expired, invalid
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    outcome_recorded_at = Column(DateTime)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    notes = Column(Text)  # Admin notes for review
    
    # Indexes for queries
    __table_args__ = (
        Index('idx_prediction_symbol_date', 'symbol', 'prediction_date'),
        Index('idx_prediction_performance', 'was_correct', 'confidence_score'),
        Index('idx_prediction_sector', 'sector', 'prediction_date'),
    )


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


class StockPredictionCreate(BaseModel):
    symbol: str
    company_name: Optional[str] = None
    prediction_source: str
    price_at_prediction: float
    predicted_direction: Optional[str] = None
    confidence_score: Optional[float] = None
    timeframe: Optional[str] = "1W"
    target_price: Optional[float] = None
    expected_return: Optional[float] = None
    reason: Optional[str] = None
    sector: Optional[str] = None
    keywords_matched: List[str] = []
    news_sentiment: Optional[float] = None
    ml_score: Optional[float] = None
    technical_score: Optional[float] = None
    fundamental_score: Optional[float] = None
    article_id: Optional[int] = None


class StockPredictionResponse(BaseModel):
    id: int
    symbol: str
    company_name: Optional[str]
    prediction_date: datetime
    prediction_source: str
    price_at_prediction: float
    predicted_direction: Optional[str]
    confidence_score: Optional[float]
    timeframe: Optional[str]
    expected_return: Optional[float]
    reason: Optional[str]
    sector: Optional[str]
    
    # Outcome data
    actual_return_1d: Optional[float]
    actual_return_1w: Optional[float]
    actual_return_1m: Optional[float]
    actual_return_3m: Optional[float]
    
    prediction_accuracy: Optional[float]
    was_correct: Optional[bool]
    status: str
    
    created_at: datetime
    last_updated: datetime
    
    class Config:
        orm_mode = True


class PredictionPerformanceStats(BaseModel):
    """Statistics about prediction performance"""
    total_predictions: int
    completed_predictions: int
    active_predictions: int
    
    overall_accuracy: float  # % of correct predictions
    average_confidence: float
    
    # By timeframe
    accuracy_1d: Optional[float]
    accuracy_1w: Optional[float]
    accuracy_1m: Optional[float]
    accuracy_3m: Optional[float]
    
    # By sector
    best_sector: Optional[str]
    worst_sector: Optional[str]
    
    # Returns
    average_return: float
    best_return: float
    worst_return: float
    
    # By source
    performance_by_source: Dict[str, float]


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


def create_stock_prediction(db: Session, prediction: StockPredictionCreate) -> StockPrediction:
    """Create new stock prediction for tracking"""
    db_prediction = StockPrediction(**prediction.dict())
    db.add(db_prediction)
    db.commit()
    db.refresh(db_prediction)
    return db_prediction


def update_prediction_outcome(
    db: Session, 
    prediction_id: int, 
    actual_price: float, 
    days_elapsed: int
) -> Optional[StockPrediction]:
    """Update prediction with actual outcome after time period"""
    prediction = db.query(StockPrediction).filter(StockPrediction.id == prediction_id).first()
    if not prediction:
        return None
    
    # Calculate actual return
    actual_return = ((actual_price - prediction.price_at_prediction) / prediction.price_at_prediction) * 100
    
    # Update appropriate field based on days elapsed
    if days_elapsed == 1:
        prediction.actual_price_1d = actual_price
        prediction.actual_return_1d = actual_return
    elif days_elapsed == 7:
        prediction.actual_price_1w = actual_price
        prediction.actual_return_1w = actual_return
    elif days_elapsed == 30:
        prediction.actual_price_1m = actual_price
        prediction.actual_return_1m = actual_return
    elif days_elapsed == 90:
        prediction.actual_price_3m = actual_price
        prediction.actual_return_3m = actual_return
    
    # Check if prediction was correct
    if prediction.predicted_direction:
        if prediction.predicted_direction == "up" and actual_return > 0:
            prediction.was_correct = True
        elif prediction.predicted_direction == "down" and actual_return < 0:
            prediction.was_correct = True
        else:
            prediction.was_correct = False
    
    # Calculate accuracy (how close to expected return)
    if prediction.expected_return:
        accuracy = 100 - abs(prediction.expected_return - actual_return)
        prediction.prediction_accuracy = max(0, accuracy)
    
    prediction.last_updated = datetime.utcnow()
    prediction.outcome_recorded_at = datetime.utcnow()
    
    # Mark as completed if all timeframes recorded
    if (prediction.actual_return_1d is not None and 
        prediction.actual_return_1w is not None and 
        prediction.actual_return_1m is not None):
        prediction.status = "completed"
    
    db.commit()
    db.refresh(prediction)
    return prediction


def get_prediction_stats(db: Session, source: Optional[str] = None) -> PredictionPerformanceStats:
    """Get performance statistics for predictions"""
    query = db.query(StockPrediction)
    if source:
        query = query.filter(StockPrediction.prediction_source == source)
    
    all_predictions = query.all()
    total = len(all_predictions)
    
    if total == 0:
        return PredictionPerformanceStats(
            total_predictions=0,
            completed_predictions=0,
            active_predictions=0,
            overall_accuracy=0.0,
            average_confidence=0.0,
            average_return=0.0,
            best_return=0.0,
            worst_return=0.0,
            performance_by_source={}
        )
    
    completed = [p for p in all_predictions if p.status == "completed"]
    active = [p for p in all_predictions if p.status == "active"]
    
    # Calculate accuracy
    correct_predictions = [p for p in completed if p.was_correct is True]
    overall_accuracy = (len(correct_predictions) / len(completed) * 100) if completed else 0
    
    # Average confidence
    confidences = [p.confidence_score for p in all_predictions if p.confidence_score is not None]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
    
    # Returns
    returns_1w = [p.actual_return_1w for p in all_predictions if p.actual_return_1w is not None]
    avg_return = sum(returns_1w) / len(returns_1w) if returns_1w else 0
    best_return = max(returns_1w) if returns_1w else 0
    worst_return = min(returns_1w) if returns_1w else 0
    
    # Accuracy by timeframe
    acc_1d = _calc_timeframe_accuracy(all_predictions, '1d')
    acc_1w = _calc_timeframe_accuracy(all_predictions, '1w')
    acc_1m = _calc_timeframe_accuracy(all_predictions, '1m')
    acc_3m = _calc_timeframe_accuracy(all_predictions, '3m')
    
    # Performance by source
    sources = {}
    for pred in completed:
        if pred.prediction_source not in sources:
            sources[pred.prediction_source] = []
        if pred.was_correct is not None:
            sources[pred.prediction_source].append(1 if pred.was_correct else 0)
    
    perf_by_source = {
        source: (sum(vals) / len(vals) * 100) if vals else 0
        for source, vals in sources.items()
    }
    
    return PredictionPerformanceStats(
        total_predictions=total,
        completed_predictions=len(completed),
        active_predictions=len(active),
        overall_accuracy=overall_accuracy,
        average_confidence=avg_confidence,
        accuracy_1d=acc_1d,
        accuracy_1w=acc_1w,
        accuracy_1m=acc_1m,
        accuracy_3m=acc_3m,
        best_sector=None,  # TODO: Calculate
        worst_sector=None,  # TODO: Calculate
        average_return=avg_return,
        best_return=best_return,
        worst_return=worst_return,
        performance_by_source=perf_by_source
    )


def _calc_timeframe_accuracy(predictions: List[StockPrediction], timeframe: str) -> Optional[float]:
    """Calculate accuracy for specific timeframe"""
    field_map = {
        '1d': 'actual_return_1d',
        '1w': 'actual_return_1w',
        '1m': 'actual_return_1m',
        '3m': 'actual_return_3m'
    }
    
    field = field_map.get(timeframe)
    if not field:
        return None
    
    relevant = [p for p in predictions if getattr(p, field) is not None and p.was_correct is not None]
    if not relevant:
        return None
    
    correct = sum(1 for p in relevant if p.was_correct)
    return (correct / len(relevant) * 100) if relevant else None
