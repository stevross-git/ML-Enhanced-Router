"""
AI Model Registry Models
Models for managing AI models and their configurations
"""

from datetime import datetime
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import Integer, String, Float, DateTime, Boolean, Text, JSON, ForeignKey
from enum import Enum

from .base import Base, TimestampMixin, generate_id

class ModelType(Enum):
    """Types of AI models"""
    COMPLETION = "completion"
    CHAT = "chat"
    EMBEDDING = "embedding"
    IMAGE = "image"
    AUDIO = "audio"
    MULTIMODAL = "multimodal"

class ModelProvider(Enum):
    """AI model providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    MICROSOFT = "microsoft"
    COHERE = "cohere"
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"
    CUSTOM = "custom"

class MLModelRegistry(Base, TimestampMixin):
    """Registry of available AI models"""
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_id)
    
    # Model identification
    name: Mapped[str] = mapped_column(String(200), nullable=False, index=True)
    model_id: Mapped[str] = mapped_column(String(200), nullable=False, index=True)  # Provider's model ID
    version: Mapped[str | None] = mapped_column(String(50), nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    
    # Provider information
    provider: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    model_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    
    # Connection details
    endpoint: Mapped[str | None] = mapped_column(String(512), nullable=True)
    api_key_env: Mapped[str | None] = mapped_column(String(100), nullable=True)  # Environment variable name
    
    # Model capabilities
    categories: Mapped[list] = mapped_column(JSON, nullable=False)  # Categories this model can handle
    max_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    supports_streaming: Mapped[bool] = mapped_column(Boolean, default=False)
    supports_functions: Mapped[bool] = mapped_column(Boolean, default=False)
    supports_vision: Mapped[bool] = mapped_column(Boolean, default=False)
    
    # Configuration parameters
    temperature: Mapped[float] = mapped_column(Float, default=0.7)
    top_p: Mapped[float | None] = mapped_column(Float, nullable=True)
    frequency_penalty: Mapped[float | None] = mapped_column(Float, nullable=True)
    presence_penalty: Mapped[float | None] = mapped_column(Float, nullable=True)
    
    # Status and priority
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    is_default: Mapped[bool] = mapped_column(Boolean, default=False)
    priority: Mapped[int] = mapped_column(Integer, default=100)  # Lower number = higher priority
    
    # Cost information
    cost_per_input_token: Mapped[float | None] = mapped_column(Float, nullable=True)
    cost_per_output_token: Mapped[float | None] = mapped_column(Float, nullable=True)
    cost_per_request: Mapped[float | None] = mapped_column(Float, nullable=True)
    
    # Performance tracking
    last_used: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    usage_count: Mapped[int] = mapped_column(Integer, default=0)
    avg_response_time: Mapped[float | None] = mapped_column(Float, nullable=True)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0)
    total_cost: Mapped[float] = mapped_column(Float, default=0.0)
    success_rate: Mapped[float] = mapped_column(Float, default=1.0)
    
    # Health status
    is_healthy: Mapped[bool] = mapped_column(Boolean, default=True)
    last_health_check: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    health_check_failures: Mapped[int] = mapped_column(Integer, default=0)
    
    # Additional configuration - FIXED: renamed to avoid SQLAlchemy conflict
    model_metadata: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    
    # Relationships
    configurations: Mapped[list["ModelConfiguration"]] = relationship(
        "ModelConfiguration", 
        back_populates="model", 
        cascade="all, delete-orphan"
    )
    
    def __repr__(self):
        return f"<MLModelRegistry {self.name} ({self.provider})>"
    
    def is_available(self) -> bool:
        """Check if model is available for use"""
        return self.is_active and self.is_healthy
    
    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for a request"""
        cost = 0.0
        
        if self.cost_per_input_token:
            cost += input_tokens * self.cost_per_input_token / 1000
        
        if self.cost_per_output_token:
            cost += output_tokens * self.cost_per_output_token / 1000
        
        if self.cost_per_request:
            cost += self.cost_per_request
        
        return cost
    
    def update_usage_stats(self, response_time: float, tokens_used: int, cost: float, success: bool):
        """Update model usage statistics"""
        self.usage_count += 1
        self.last_used = datetime.utcnow()
        self.total_tokens += tokens_used
        self.total_cost += cost
        
        # Update average response time
        if self.avg_response_time is None:
            self.avg_response_time = response_time
        else:
            # Exponential moving average
            alpha = 0.1
            self.avg_response_time = alpha * response_time + (1 - alpha) * self.avg_response_time
        
        # Update success rate
        if success:
            self.success_rate = (self.success_rate * (self.usage_count - 1) + 1.0) / self.usage_count
        else:
            self.success_rate = (self.success_rate * (self.usage_count - 1)) / self.usage_count
    
    def to_dict(self, include_stats=False):
        """Convert to dictionary representation"""
        result = {
            'id': self.id,
            'name': self.name,
            'model_id': self.model_id,
            'version': self.version,
            'description': self.description,
            'provider': self.provider,
            'model_type': self.model_type,
            'endpoint': self.endpoint,
            'categories': self.categories,
            'max_tokens': self.max_tokens,
            'supports_streaming': self.supports_streaming,
            'supports_functions': self.supports_functions,
            'supports_vision': self.supports_vision,
            'temperature': self.temperature,
            'is_active': self.is_active,
            'is_default': self.is_default,
            'priority': self.priority,
            'is_healthy': self.is_healthy,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'metadata': self.model_metadata  # Return as 'metadata' for API compatibility
        }
        
        if include_stats:
            result.update({
                'usage_count': self.usage_count,
                'avg_response_time': self.avg_response_time,
                'total_tokens': self.total_tokens,
                'total_cost': self.total_cost,
                'success_rate': self.success_rate,
                'last_used': self.last_used.isoformat() if self.last_used else None,
                'last_health_check': self.last_health_check.isoformat() if self.last_health_check else None
            })
        
        return result

class ModelConfiguration(Base, TimestampMixin):
    """Specific configurations for models"""
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_id)
    
    # Configuration identification
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    
    # Model association
    model_id: Mapped[str] = mapped_column(String(36), ForeignKey('ml_model_registry.id'), nullable=False)
    model: Mapped["MLModelRegistry"] = relationship("MLModelRegistry", back_populates="configurations")
    
    # Configuration parameters
    temperature: Mapped[float | None] = mapped_column(Float, nullable=True)
    max_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    top_p: Mapped[float | None] = mapped_column(Float, nullable=True)
    frequency_penalty: Mapped[float | None] = mapped_column(Float, nullable=True)
    presence_penalty: Mapped[float | None] = mapped_column(Float, nullable=True)
    
    # System prompts and context
    system_prompt: Mapped[str | None] = mapped_column(Text, nullable=True)
    context_instructions: Mapped[str | None] = mapped_column(Text, nullable=True)
    
    # Usage constraints
    categories: Mapped[list | None] = mapped_column(JSON, nullable=True)  # Specific categories for this config
    user_roles: Mapped[list | None] = mapped_column(JSON, nullable=True)  # User roles that can use this config
    
    # Status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_default: Mapped[bool] = mapped_column(Boolean, default=False)
    
    # Additional configuration - FIXED: renamed to avoid SQLAlchemy conflict
    config_metadata: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    
    def __repr__(self):
        return f"<ModelConfiguration {self.name} for {self.model.name if self.model else 'unknown'}>"
    
    def get_effective_config(self) -> dict:
        """Get the effective configuration merging model defaults with this config"""
        config = {}
        
        if self.model:
            # Start with model defaults
            config.update({
                'temperature': self.model.temperature,
                'max_tokens': self.model.max_tokens,
                'top_p': self.model.top_p,
                'frequency_penalty': self.model.frequency_penalty,
                'presence_penalty': self.model.presence_penalty,
            })
        
        # Override with configuration-specific values
        if self.temperature is not None:
            config['temperature'] = self.temperature
        if self.max_tokens is not None:
            config['max_tokens'] = self.max_tokens
        if self.top_p is not None:
            config['top_p'] = self.top_p
        if self.frequency_penalty is not None:
            config['frequency_penalty'] = self.frequency_penalty
        if self.presence_penalty is not None:
            config['presence_penalty'] = self.presence_penalty
        
        # Add prompts
        if self.system_prompt:
            config['system_prompt'] = self.system_prompt
        if self.context_instructions:
            config['context_instructions'] = self.context_instructions
        
        return config
    
    def to_dict(self):
        """Convert to dictionary representation"""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'model_id': self.model_id,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'top_p': self.top_p,
            'frequency_penalty': self.frequency_penalty,
            'presence_penalty': self.presence_penalty,
            'system_prompt': self.system_prompt,
            'context_instructions': self.context_instructions,
            'categories': self.categories,
            'user_roles': self.user_roles,
            'is_active': self.is_active,
            'is_default': self.is_default,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'metadata': self.config_metadata  # Return as 'metadata' for API compatibility
        }

# Index definitions for performance
from sqlalchemy import Index

# Create indexes for common queries
Index('idx_ml_model_provider_active', MLModelRegistry.provider, MLModelRegistry.is_active)
Index('idx_ml_model_type_active', MLModelRegistry.model_type, MLModelRegistry.is_active)
Index('idx_ml_model_priority', MLModelRegistry.priority, MLModelRegistry.is_active)
Index('idx_model_config_model_active', ModelConfiguration.model_id, ModelConfiguration.is_active)