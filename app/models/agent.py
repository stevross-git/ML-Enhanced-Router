"""
Agent-related Database Models
Models for agent registration and management
"""

from datetime import datetime
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import Integer, String, Float, DateTime, Boolean, Text, JSON, ForeignKey

from .base import Base, TimestampMixin, generate_id

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
            'metadata': self.agent_metadata,
            'tags': self.tags
        }

class Agent(Base, TimestampMixin):
    """Agent model for service layer operations"""
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_id)
    
    # Basic agent information
    name: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    endpoint: Mapped[str] = mapped_column(String(512), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    version: Mapped[str] = mapped_column(String(32), default='1.0.0')
    
    # Status and configuration
    status: Mapped[str] = mapped_column(String(32), default='active', index=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    max_concurrent_sessions: Mapped[int] = mapped_column(Integer, default=10)
    
    last_used: Mapped[datetime | None] = mapped_column(DateTime, nullable=True, index=True)
    last_seen: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    active_sessions: Mapped[int] = mapped_column(Integer, default=0)
    
    # Additional metadata
    agent_metadata: Mapped[dict] = mapped_column(JSON, default=dict)
    
    # Relationships
    capabilities: Mapped[list["AgentCapability"]] = relationship("AgentCapability", back_populates="agent", cascade="all, delete-orphan")
    sessions: Mapped[list["AgentSession"]] = relationship("AgentSession", back_populates="agent", cascade="all, delete-orphan")
    metrics: Mapped[list["AgentMetrics"]] = relationship("AgentMetrics", back_populates="agent", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Agent {self.id}: {self.name}>"
    
    def to_dict(self):
        """Convert to dictionary representation"""
        return {
            'id': self.id,
            'name': self.name,
            'type': self.type,
            'endpoint': self.endpoint,
            'description': self.description,
            'version': self.version,
            'status': self.status,
            'is_active': self.is_active,
            'max_concurrent_sessions': self.max_concurrent_sessions,
            'last_used': self.last_used.isoformat() if self.last_used else None,
            'last_seen': self.last_seen.isoformat(),
            'active_sessions': self.active_sessions,
            'created_at': self.created_at.isoformat(),
            'agent_metadata': self.agent_metadata
        }

class AgentCapability(Base, TimestampMixin):
    """Agent capability model for tracking agent abilities"""
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    
    # Agent reference
    agent_id: Mapped[str] = mapped_column(String(36), ForeignKey('agent.id'), nullable=False, index=True)
    agent: Mapped["Agent"] = relationship("Agent", back_populates="capabilities")
    
    capability: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    confidence_score: Mapped[float] = mapped_column(Float, default=1.0)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    
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
            'created_at': self.created_at.isoformat()
        }

class AgentSession(Base, TimestampMixin):
    """Agent session model for tracking active sessions"""
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_id)
    
    # Agent reference
    agent_id: Mapped[str] = mapped_column(String(36), ForeignKey('agent.id'), nullable=False, index=True)
    agent: Mapped["Agent"] = relationship("Agent", back_populates="sessions")
    
    # Session details
    query: Mapped[str] = mapped_column(Text, nullable=False)
    context: Mapped[dict] = mapped_column(JSON, default=dict)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    
    def __repr__(self):
        return f"<AgentSession {self.id} for {self.agent_id}>"
    
    def to_dict(self):
        """Convert to dictionary representation"""
        return {
            'id': self.id,
            'agent_id': self.agent_id,
            'query': self.query,
            'context': self.context,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat()
        }

class AgentMetrics(Base, TimestampMixin):
    """Agent metrics model for performance tracking"""
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    
    # Agent reference
    agent_id: Mapped[str] = mapped_column(String(36), ForeignKey('agent.id'), nullable=False, index=True)
    agent: Mapped["Agent"] = relationship("Agent", back_populates="metrics")
    
    # Request metrics
    total_requests: Mapped[int] = mapped_column(Integer, default=0)
    successful_requests: Mapped[int] = mapped_column(Integer, default=0)
    failed_requests: Mapped[int] = mapped_column(Integer, default=0)
    
    # Performance metrics
    average_response_time: Mapped[float] = mapped_column(Float, default=0.0)
    last_updated: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<AgentMetrics {self.agent_id}: {self.total_requests} requests>"
    
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
            'total_requests': self.total_requests,
            'successful_requests': self.successful_requests,
            'failed_requests': self.failed_requests,
            'success_rate': self.success_rate,
            'average_response_time': self.average_response_time,
            'last_updated': self.last_updated.isoformat(),
            'created_at': self.created_at.isoformat()
        }

# Create indexes for performance
from sqlalchemy import Index

Index('idx_agent_active_status', Agent.is_active, Agent.status)
Index('idx_agent_type_active', Agent.type, Agent.is_active)
Index('idx_agent_last_used', Agent.last_used)
Index('idx_agent_capability_agent_active', AgentCapability.agent_id, AgentCapability.is_active)
Index('idx_agent_capability_capability', AgentCapability.capability)
Index('idx_agent_session_agent_active', AgentSession.agent_id, AgentSession.is_active)
Index('idx_agent_metrics_agent', AgentMetrics.agent_id)
Index('idx_agent_registration_active_healthy', AgentRegistration.is_active, AgentRegistration.is_healthy)
