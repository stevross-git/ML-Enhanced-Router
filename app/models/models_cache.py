"""
Cache-related Database Models
Models for AI response caching and cache management
"""

from datetime import datetime, timedelta
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import Integer, String, Float, DateTime, Boolean, Text, JSON, LargeBinary
import hashlib
import json

from .base import Base, TimestampMixin, generate_id

class AICacheEntry(Base, TimestampMixin):
    """Cache entries for AI model responses"""
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_id)
    
    # Cache key information
    cache_key: Mapped[str] = mapped_column(String(128), unique=True, nullable=False, index=True)
    query_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    
    # Request information
    model_id: Mapped[str] = mapped_column(String(200), nullable=False, index=True)
    provider: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    
    # Query details
    query_text: Mapped[str] = mapped_column(Text, nullable=False)
    parameters: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    
    # Response data
    response_data: Mapped[dict] = mapped_column(JSON, nullable=False)
    response_metadata: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    
    # Cache metadata
    ttl_seconds: Mapped[int] = mapped_column(Integer, default=3600)  # 1 hour default
    expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    
    # Usage tracking
    hit_count: Mapped[int] = mapped_column(Integer, default=0)
    last_accessed: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    
    # Performance metrics
    original_response_time: Mapped[float | None] = mapped_column(Float, nullable=True)
    tokens_saved: Mapped[int] = mapped_column(Integer, default=0)
    cost_saved: Mapped[float] = mapped_column(Float, default=0.0)
    
    # Cache status
    is_valid: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    invalidation_reason: Mapped[str | None] = mapped_column(String(100), nullable=True)
    
    # Additional metadata
    tags: Mapped[list | None] = mapped_column(JSON, nullable=True)
    metadata: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    
    def __repr__(self):
        return f"<AICacheEntry {self.cache_key[:20]}... - {self.hit_count} hits>"
    
    @classmethod
    def generate_cache_key(cls, query: str, model_id: str, parameters: dict = None) -> str:
        """Generate a cache key for the given query and parameters"""
        # Create a consistent hash of the query and parameters
        content = {
            'query': query.strip().lower(),
            'model_id': model_id,
            'parameters': parameters or {}
        }
        
        # Sort parameters for consistent hashing
        content_str = json.dumps(content, sort_keys=True)
        return hashlib.sha256(content_str.encode()).hexdigest()[:32]
    
    @classmethod
    def generate_query_hash(cls, query: str) -> str:
        """Generate a hash for the query text only"""
        return hashlib.sha256(query.strip().lower().encode()).hexdigest()[:16]
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired"""
        return datetime.utcnow() > self.expires_at
    
    def is_cache_valid(self) -> bool:
        """Check if cache entry is valid and not expired"""
        return self.is_valid and not self.is_expired()
    
    def record_hit(self):
        """Record a cache hit"""
        self.hit_count += 1
        self.last_accessed = datetime.utcnow()
    
    def extend_ttl(self, additional_seconds: int):
        """Extend the TTL of the cache entry"""
        self.expires_at = datetime.utcnow() + timedelta(seconds=additional_seconds)
        self.ttl_seconds = additional_seconds
    
    def invalidate(self, reason: str = None):
        """Invalidate the cache entry"""
        self.is_valid = False
        self.invalidation_reason = reason
    
    def to_dict(self):
        """Convert to dictionary representation"""
        return {
            'id': self.id,
            'cache_key': self.cache_key,
            'model_id': self.model_id,
            'provider': self.provider,
            'query_text': self.query_text[:100] + '...' if len(self.query_text) > 100 else self.query_text,
            'hit_count': self.hit_count,
            'ttl_seconds': self.ttl_seconds,
            'expires_at': self.expires_at.isoformat(),
            'last_accessed': self.last_accessed.isoformat() if self.last_accessed else None,
            'is_valid': self.is_valid,
            'tokens_saved': self.tokens_saved,
            'cost_saved': self.cost_saved,
            'created_at': self.created_at.isoformat(),
            'tags': self.tags
        }

class AICacheStats(Base):
    """Aggregated cache statistics"""
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    
    # Time period
    period_start: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    period_end: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    granularity: Mapped[str] = mapped_column(String(16), nullable=False)  # hour, day, week
    
    # Cache metrics
    total_requests: Mapped[int] = mapped_column(Integer, default=0)
    cache_hits: Mapped[int] = mapped_column(Integer, default=0)
    cache_misses: Mapped[int] = mapped_column(Integer, default=0)
    
    # Performance metrics
    hit_rate: Mapped[float] = mapped_column(Float, default=0.0)
    avg_response_time_cached: Mapped[float | None] = mapped_column(Float, nullable=True)
    avg_response_time_uncached: Mapped[float | None] = mapped_column(Float, nullable=True)
    time_saved: Mapped[float] = mapped_column(Float, default=0.0)
    
    # Resource savings
    tokens_saved: Mapped[int] = mapped_column(Integer, default=0)
    cost_saved: Mapped[float] = mapped_column(Float, default=0.0)
    
    # Cache size metrics
    total_entries: Mapped[int] = mapped_column(Integer, default=0)
    valid_entries: Mapped[int] = mapped_column(Integer, default=0)
    expired_entries: Mapped[int] = mapped_column(Integer, default=0)
    
    # Model breakdown
    model_breakdown: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    provider_breakdown: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    
    # Cache efficiency
    avg_hits_per_entry: Mapped[float] = mapped_column(Float, default=0.0)
    entries_created: Mapped[int] = mapped_column(Integer, default=0)
    entries_invalidated: Mapped[int] = mapped_column(Integer, default=0)
    entries_expired: Mapped[int] = mapped_column(Integer, default=0)
    
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<AICacheStats {self.granularity}: {self.period_start} - {self.hit_rate:.1f}% hit rate>"
    
    @property
    def miss_rate(self) -> float:
        """Calculate cache miss rate percentage"""
        return 100.0 - self.hit_rate
    
    @property
    def efficiency_score(self) -> float:
        """Calculate cache efficiency score (0-100)"""
        if self.total_entries == 0:
            return 0.0
        
        # Combine hit rate and average hits per entry
        hit_rate_score = self.hit_rate
        utilization_score = min(self.avg_hits_per_entry * 10, 100)  # Cap at 100
        
        return (hit_rate_score + utilization_score) / 2
    
    def to_dict(self):
        """Convert to dictionary representation"""
        return {
            'id': self.id,
            'period_start': self.period_start.isoformat(),
            'period_end': self.period_end.isoformat(),
            'granularity': self.granularity,
            'total_requests': self.total_requests,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': self.hit_rate,
            'miss_rate': self.miss_rate,
            'time_saved': self.time_saved,
            'tokens_saved': self.tokens_saved,
            'cost_saved': self.cost_saved,
            'total_entries': self.total_entries,
            'valid_entries': self.valid_entries,
            'expired_entries': self.expired_entries,
            'efficiency_score': self.efficiency_score,
            'avg_hits_per_entry': self.avg_hits_per_entry,
            'model_breakdown': self.model_breakdown,
            'provider_breakdown': self.provider_breakdown,
            'created_at': self.created_at.isoformat()
        }

# Create indexes for performance
from sqlalchemy import Index

Index('idx_cache_entry_expires_valid', AICacheEntry.expires_at, AICacheEntry.is_valid)
Index('idx_cache_entry_model_provider', AICacheEntry.model_id, AICacheEntry.provider)
Index('idx_cache_entry_query_hash', AICacheEntry.query_hash)
Index('idx_cache_stats_period_granularity', AICacheStats.granularity, AICacheStats.period_start)
