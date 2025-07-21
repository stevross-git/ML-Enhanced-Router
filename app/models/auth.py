"""
Authentication and User Management Models
Models for user authentication, API keys, and sessions
"""

from datetime import datetime, timedelta
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import Integer, String, DateTime, Boolean, Text, JSON, ForeignKey
from werkzeug.security import generate_password_hash, check_password_hash
import jwt
import secrets

from .base import Base, TimestampMixin, generate_id

class APIKey(Base, TimestampMixin):
    """API key model for external access"""
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_id)
    
    # Key details
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    key_hash: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True)
    key_prefix: Mapped[str] = mapped_column(String(20), nullable=False, index=True)  # First few chars for identification
    
    # User association
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey('user.id'), nullable=False, index=True)
    
    # Key status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    
    # Usage tracking
    usage_count: Mapped[int] = mapped_column(Integer, default=0)
    last_used: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    
    # Permissions and limits
    permissions: Mapped[list | None] = mapped_column(JSON, nullable=True)
    rate_limit: Mapped[int | None] = mapped_column(Integer, nullable=True)  # requests per minute
    
    # Additional metadata
    user_key_metadata: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    
    def __repr__(self):
        return f"<APIKey {self.name} ({self.key_prefix}...)>"
    
    @classmethod
    def generate_key(cls) -> tuple[str, str]:
        """Generate new API key and return (key, hash)"""
        key = f"mlr_{secrets.token_urlsafe(32)}"
        key_hash = generate_password_hash(key)
        return key, key_hash
    
    def check_key(self, key: str) -> bool:
        """Check if provided key matches this API key"""
        return check_password_hash(self.key_hash, key)
    
    def is_valid(self) -> bool:
        """Check if API key is valid and not expired"""
        if not self.is_active:
            return False
        if self.expires_at and self.expires_at < datetime.utcnow():
            return False
        return True
    
    def record_usage(self):
        """Record API key usage"""
        self.usage_count += 1
        self.last_used = datetime.utcnow()
    
    def to_dict(self):
        """Convert to dictionary representation"""
        return {
            'id': self.id,
            'name': self.name,
            'key_prefix': self.key_prefix,
            'user_id': self.user_id,
            'is_active': self.is_active,
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'usage_count': self.usage_count,
            'last_used': self.last_used.isoformat() if self.last_used else None,
            'permissions': self.permissions,
            'rate_limit': self.rate_limit,
            'created_at': self.created_at.isoformat(),
            'metadata': self.user_metadata
        }

class UserSession(Base, TimestampMixin):
    """User session tracking model"""
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_id)
    
    # Session details
    session_token: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    
    # User association
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey('user.id'), nullable=False, index=True)
    
    # Session metadata
    ip_address: Mapped[str | None] = mapped_column(String(45), nullable=True)  # IPv6 compatible
    user_agent: Mapped[str | None] = mapped_column(Text, nullable=True)
    
    # Session status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    last_activity: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    # Additional data
    session_data: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    
    def __repr__(self):
        return f"<UserSession {self.id} for {self.user_id}>"
    
    def is_valid(self) -> bool:
        """Check if session is valid and not expired"""
        if not self.is_active:
            return False
        if self.expires_at < datetime.utcnow():
            return False
        return True
    
    def refresh(self, extend_hours: int = 24):
        """Refresh session expiration"""
        self.expires_at = datetime.utcnow() + timedelta(hours=extend_hours)
        self.last_activity = datetime.utcnow()
    
    def invalidate(self):
        """Invalidate session"""
        self.is_active = False
    
    def to_dict(self):
        """Convert to dictionary representation"""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'ip_address': self.ip_address,
            'is_active': self.is_active,
            'expires_at': self.expires_at.isoformat(),
            'last_activity': self.last_activity.isoformat(),
            'created_at': self.created_at.isoformat(),
            'session_data': self.session_data
        }

# Create indexes for performance
from sqlalchemy import Index

Index('idx_apikey_user_active', APIKey.user_id, APIKey.is_active)
Index('idx_session_user_active', UserSession.user_id, UserSession.is_active)
Index('idx_session_expires', UserSession.expires_at)
