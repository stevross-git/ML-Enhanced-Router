"""
Agent-related Database Models
Models for agent registration and management
"""

from datetime import datetime
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import Integer, String, Float, DateTime, Boolean, Text, JSON, Index

from .base import Base, TimestampMixin


class Agent(Base, TimestampMixin):
    """Main agent model"""
    
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    
    # Basic agent information
    name: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    version: Mapped[str | None] = mapped_column(String(32), nullable=True)
    type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    
    # Connection details
    endpoint: Mapped[str] = mapped_column(String(512), nullable=False)
    auth_type: Mapped[str] = mapped_column(String(32), default='none')  # none, bearer, api_key
    auth_credentials: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    
    # Status and configuration
    status: Mapped[str] = mapped_column(String(32), default='active', index=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    is_healthy: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    
    # Capacity management
    max_concurrent_sessions: Mapped[int] = mapped_column(Integer, default=10)
    active_sessions: Mapped[int] = mapped_column(Integer, default=0)
    
    # Performance tracking
    total_requests: Mapped[int] = mapped_column(Integer, default=0)
    successful_requests: Mapped[int] = mapped_column(Integer, default=0)
    failed_requests: Mapped[int] = mapped_column(Integer, default=0)
    avg_response_time: Mapped[float] = mapped_column(Float, default=0.0)
    
    # Timestamps
    last_seen: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    last_used: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    last_health_check: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    
    # Metadata (renamed to avoid SQLAlchemy reserved word)
    agent_metadata: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    
    # Relationships
    capabilities = relationship("AgentCapability", back_populates="agent", cascade="all, delete-orphan")
    metrics = relationship("AgentMetrics", back_populates="agent", cascade="all, delete-orphan")
    sessions = relationship("AgentSession", back_populates="agent", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Agent {self.id}: {self.name}>"
    
    @property
    def success_rate(self):
        """Calculate success rate percentage"""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100
    
    def update_health_status(self, is_healthy: bool):
        """Update agent health status"""
        self.is_healthy = is_healthy
        self.last_health_check = datetime.utcnow()
        self.last_seen = datetime.utcnow()
    
    def record_request(self, success: bool, response_time: float):
        """Record a request and update metrics"""
        self.total_requests += 1
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
        
        # Update average response time
        if self.total_requests == 1:
            self.avg_response_time = response_time
        else:
            self.avg_response_time = (
                (self.avg_response_time * (self.total_requests - 1) + response_time) 
                / self.total_requests
            )
        
        self.last_seen = datetime.utcnow()
    
    def to_dict(self):
        """Convert to dictionary representation"""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'version': self.version,
            'type': self.type,
            'endpoint': self.endpoint,
            'status': self.status,
            'is_active': self.is_active,
            'is_healthy': self.is_healthy,
            'max_concurrent_sessions': self.max_concurrent_sessions,
            'active_sessions': self.active_sessions,
            'success_rate': self.success_rate,
            'total_requests': self.total_requests,
            'avg_response_time': self.avg_response_time,
            'last_seen': self.last_seen.isoformat() if self.last_seen else None,
            'created_at': self.created_at.isoformat(),
            'metadata': self.agent_metadata
        }


class AgentCapability(Base, TimestampMixin):
    """Agent capabilities"""
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    
    # Foreign key
    agent_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    
    # Capability details
    capability: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    confidence_score: Mapped[float] = mapped_column(Float, default=1.0)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    
    # Metadata (renamed to avoid SQLAlchemy reserved word)
    capability_metadata: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    
    # Relationship
    agent = relationship("Agent", back_populates="capabilities")
    
    def __repr__(self):
        return f"<AgentCapability {self.agent_id}: {self.capability}>"
    
    def to_dict(self):
        """Convert to dictionary representation"""
        return {
            'id': self.id,
            'agent_id': self.agent_id,
            'capability': self.capability,
            'confidence_score': self.confidence_score,
            'is_active': self.is_active,
            'metadata': self.capability_metadata,
            'created_at': self.created_at.isoformat()
        }


class AgentSession(Base, TimestampMixin):
    """Active agent sessions"""
    
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    
    # Foreign key
    agent_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    
    # Session details
    session_type: Mapped[str] = mapped_column(String(64), default='chat')
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    
    # Session metadata
    user_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    context: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    
    # Performance tracking
    request_count: Mapped[int] = mapped_column(Integer, default=0)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0)
    total_cost: Mapped[float] = mapped_column(Float, default=0.0)
    
    # Timestamps
    started_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    last_activity: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    
    # Relationship
    agent = relationship("Agent", back_populates="sessions")
    
    def __repr__(self):
        return f"<AgentSession {self.id}: {self.agent_id}>"
    
    def to_dict(self):
        """Convert to dictionary representation"""
        return {
            'id': self.id,
            'agent_id': self.agent_id,
            'session_type': self.session_type,
            'is_active': self.is_active,
            'user_id': self.user_id,
            'request_count': self.request_count,
            'total_tokens': self.total_tokens,
            'total_cost': self.total_cost,
            'started_at': self.started_at.isoformat(),
            'last_activity': self.last_activity.isoformat(),
            'ended_at': self.ended_at.isoformat() if self.ended_at else None,
            'context': self.context
        }


class AgentMetrics(Base):
    """Detailed metrics for agent performance"""
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    
    # Agent reference
    agent_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    agent_name: Mapped[str] = mapped_column(String(128), nullable=False)
    
    # Time period
    period_start: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    period_end: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    granularity: Mapped[str] = mapped_column(String(16), nullable=False)  # hour, day, week
    
    # Request metrics
    total_requests: Mapped[int] = mapped_column(Integer, default=0)
    successful_requests: Mapped[int] = mapped_column(Integer, default=0)
    failed_requests: Mapped[int] = mapped_column(Integer, default=0)
    timeout_requests: Mapped[int] = mapped_column(Integer, default=0)
    
    # Performance metrics
    average_response_time: Mapped[float] = mapped_column(Float, default=0.0)  # Note: named to match service usage
    avg_response_time: Mapped[float | None] = mapped_column(Float, nullable=True)
    min_response_time: Mapped[float | None] = mapped_column(Float, nullable=True)
    max_response_time: Mapped[float | None] = mapped_column(Float, nullable=True)
    p95_response_time: Mapped[float | None] = mapped_column(Float, nullable=True)
    
    # Resource usage
    total_tokens: Mapped[int] = mapped_column(Integer, default=0)
    total_cost: Mapped[float] = mapped_column(Float, default=0.0)
    
    # Category performance
    category_breakdown: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error_breakdown: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    
    # Health metrics
    uptime_percentage: Mapped[float] = mapped_column(Float, default=100.0)
    health_check_failures: Mapped[int] = mapped_column(Integer, default=0)
    
    # Timestamps
    last_updated: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    # Relationship
    agent = relationship("Agent", back_populates="metrics")
    
    def __repr__(self):
        return f"<AgentMetrics {self.agent_id}: {self.period_start} - {self.total_requests} requests>"
    
    @property
    def success_rate(self):
        """Calculate success rate percentage"""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100
    
    def to_dict(self):
        """Convert to dictionary representation"""
        return {
            'id': self.id,
            'agent_id': self.agent_id,
            'agent_name': self.agent_name,
            'period_start': self.period_start.isoformat(),
            'period_end': self.period_end.isoformat(),
            'granularity': self.granularity,
            'total_requests': self.total_requests,
            'successful_requests': self.successful_requests,
            'failed_requests': self.failed_requests,
            'success_rate': self.success_rate,
            'average_response_time': self.average_response_time,
            'total_tokens': self.total_tokens,
            'total_cost': self.total_cost,
            'uptime_percentage': self.uptime_percentage,
            'category_breakdown': self.category_breakdown,
            'error_breakdown': self.error_breakdown,
            'last_updated': self.last_updated.isoformat()
        }


# Legacy model for backward compatibility (if needed)
class AgentRegistration(Base, TimestampMixin):
    """Legacy agent registration model - kept for backward compatibility"""
    
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    
    # Basic agent information
    name: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    version: Mapped[str | None] = mapped_column(String(32), nullable=True)
    
    # Connection details
    endpoint: Mapped[str] = mapped_column(String(512), nullable=False)
    auth_type: Mapped[str] = mapped_column(String(32), default='none')
    auth_credentials: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    
    # Agent capabilities
    categories: Mapped[dict] = mapped_column(JSON, nullable=False)
    capabilities: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    supported_formats: Mapped[list | None] = mapped_column(JSON, nullable=True)
    
    # Configuration
    priority: Mapped[int] = mapped_column(Integer, default=100)
    max_requests_per_minute: Mapped[int] = mapped_column(Integer, default=60)
    timeout_seconds: Mapped[int] = mapped_column(Integer, default=30)
    
    # Status tracking
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    is_healthy: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    last_health_check: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    last_seen: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    # Performance tracking
    total_requests: Mapped[int] = mapped_column(Integer, default=0)
    successful_requests: Mapped[int] = mapped_column(Integer, default=0)
    failed_requests: Mapped[int] = mapped_column(Integer, default=0)
    avg_response_time: Mapped[float] = mapped_column(Float, default=0.0)
    
    # Additional metadata
    agent_metadata: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    tags: Mapped[list | None] = mapped_column(JSON, nullable=True)
    
    def __repr__(self):
        return f"<AgentRegistration {self.id}: {self.name}>"
    
    @property
    def success_rate(self):
        """Calculate success rate percentage"""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100


# Create indexes for performance
Index('idx_agent_status_active', Agent.status, Agent.is_active)
Index('idx_agent_capability_lookup', AgentCapability.agent_id, AgentCapability.capability, AgentCapability.is_active)
Index('idx_agent_session_active', AgentSession.agent_id, AgentSession.is_active)
Index('idx_agent_metrics_agent_period', AgentMetrics.agent_id, AgentMetrics.period_start)
Index('idx_agent_registration_active_healthy', AgentRegistration.is_active, AgentRegistration.is_healthy)