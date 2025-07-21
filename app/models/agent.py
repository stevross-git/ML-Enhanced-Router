"""
Agent-related Database Models
Models for agent registration and management
"""

from datetime import datetime
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import Integer, String, Float, DateTime, Boolean, Text, JSON

from .base import Base, TimestampMixin

class AgentRegistration(Base, TimestampMixin):
    """Registry of all available agents"""
    
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    
    # Basic agent information
    name: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    version: Mapped[str | None] = mapped_column(String(32), nullable=True)
    
    # Connection details
    endpoint: Mapped[str] = mapped_column(String(512), nullable=False)
    auth_type: Mapped[str] = mapped_column(String(32), default='none')  # none, bearer, api_key
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
    metadata: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    tags: Mapped[list | None] = mapped_column(JSON, nullable=True)
    
    def __repr__(self):
        return f"<AgentRegistration {self.id}: {self.name}>"
    
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
            'endpoint': self.endpoint,
            'categories': self.categories,
            'capabilities': self.capabilities,
            'priority': self.priority,
            'is_active': self.is_active,
            'is_healthy': self.is_healthy,
            'success_rate': self.success_rate,
            'total_requests': self.total_requests,
            'avg_response_time': self.avg_response_time,
            'last_seen': self.last_seen.isoformat() if self.last_seen else None,
            'created_at': self.created_at.isoformat(),
            'metadata': self.metadata,
            'tags': self.tags
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
    
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
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
            'avg_response_time': self.avg_response_time,
            'total_tokens': self.total_tokens,
            'total_cost': self.total_cost,
            'uptime_percentage': self.uptime_percentage,
            'category_breakdown': self.category_breakdown,
            'error_breakdown': self.error_breakdown
        }

# Create indexes for performance
from sqlalchemy import Index

Index('idx_agent_metrics_agent_period', AgentMetrics.agent_id, AgentMetrics.period_start)
Index('idx_agent_registration_active_healthy', AgentRegistration.is_active, AgentRegistration.is_healthy)
