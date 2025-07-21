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
    version: Mapped[str | None] = mapped_column(String(50),