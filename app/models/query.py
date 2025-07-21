"""
Query-related Database Models
Models for logging and tracking query processing
"""

from datetime import datetime
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import Integer, String, Float, DateTime, Boolean, Text, JSON, ForeignKey

from .base import Base, TimestampMixin

class QueryLog(Base, TimestampMixin):
    """Log of all queries processed by the router"""
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    
    # Query details
    query_text: Mapped[str] = mapped_column(Text, nullable=False)
    query_hash: Mapped[str] = mapped_column(String(64), nullable=True, index=True)
    
    # User information
    user_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    session_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    
    # Classification results
    category: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    subcategory: Mapped[str | None] = mapped_column(String(64), nullable=True)
    
    # Routing information
    agent_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    agent_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    model_used: Mapped[str | None] = mapped_column(String(128), nullable=True)
    
    # Processing results
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    response_time: Mapped[float | None] = mapped_column(Float, nullable=True)
    tokens_used: Mapped[int | None] = mapped_column(Integer, nullable=True)
    cost: Mapped[float | None] = mapped_column(Float, nullable=True)
    
    # Additional metadata
    metadata: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    
    # Cache information
    cache_hit: Mapped[bool] = mapped_column(Boolean, default=False)
    cache_key: Mapped[str | None] = mapped_column(String(128), nullable=True)
    
    def __repr__(self):
        return f"<QueryLog {self.id}: {self.category} - {self.status}>"
    
    def to_dict(self):
        """Convert to dictionary representation"""
        return {
            'id': self.id,
            'query_text': self.query_text[:100] + '...' if len(self.query_text) > 100 else self.query_text,
            'category': self.category,
            'confidence': self.confidence,
            'agent_name': self.agent_name,
            'status': self.status,
            'response_time': self.response_time,
            'tokens_used': self.tokens_used,
            'cost': self.cost,
            'cache_hit': self.cache_hit,
            'created_at': self.created_at.isoformat(),
            'metadata': self.metadata
        }

class QueryMetrics(Base):
    """Aggregated metrics for query processing"""
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    
    # Time period
    period_start: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    period_end: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    granularity: Mapped[str] = mapped_column(String(16), nullable=False)  # hour, day, week
    
    # Query counts
    total_queries: Mapped[int] = mapped_column(Integer, default=0)
    successful_queries: Mapped[int] = mapped_column(Integer, default=0)
    failed_queries: Mapped[int] = mapped_column(Integer, default=0)
    cached_queries: Mapped[int] = mapped_column(Integer, default=0)
    
    # Performance metrics
    avg_response_time: Mapped[float | None] = mapped_column(Float, nullable=True)
    min_response_time: Mapped[float | None] = mapped_column(Float, nullable=True)
    max_response_time: Mapped[float | None] = mapped_column(Float, nullable=True)
    p95_response_time: Mapped[float | None] = mapped_column(Float, nullable=True)
    
    # Resource usage
    total_tokens: Mapped[int] = mapped_column(Integer, default=0)
    total_cost: Mapped[float] = mapped_column(Float, default=0.0)
    
    # Category breakdown
    category_distribution: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    agent_distribution: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    model_distribution: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    
    # Error tracking
    error_rate: Mapped[float] = mapped_column(Float, default=0.0)
    error_types: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<QueryMetrics {self.granularity}: {self.period_start} - {self.total_queries} queries>"
    
    def to_dict(self):
        """Convert to dictionary representation"""
        return {
            'id': self.id,
            'period_start': self.period_start.isoformat(),
            'period_end': self.period_end.isoformat(),
            'granularity': self.granularity,
            'total_queries': self.total_queries,
            'successful_queries': self.successful_queries,
            'failed_queries': self.failed_queries,
            'cached_queries': self.cached_queries,
            'avg_response_time': self.avg_response_time,
            'total_tokens': self.total_tokens,
            'total_cost': self.total_cost,
            'error_rate': self.error_rate,
            'category_distribution': self.category_distribution,
            'agent_distribution': self.agent_distribution,
            'model_distribution': self.model_distribution
        }

# Index definitions for performance
from sqlalchemy import Index

# Create composite indexes for common queries
Index('idx_query_log_user_created', QueryLog.user_id, QueryLog.created_at)
Index('idx_query_log_category_created', QueryLog.category, QueryLog.created_at)
Index('idx_query_log_status_created', QueryLog.status, QueryLog.created_at)
Index('idx_query_metrics_period', QueryMetrics.granularity, QueryMetrics.period_start)
