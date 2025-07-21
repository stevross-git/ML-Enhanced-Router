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

class User(Base, TimestampMixin):
    """User account model"""
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_id)
    
    # Basic user information
    username: Mapped[str] = mapped_column(String(80), unique=True, nullable=False, index=True)
    email: Mapped[str] = mapped_column(String(120), unique=True, nullable=False, index=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    
    # User details
    first_name: Mapped[str | None] = mapped_column(String(50), nullable=True)
    last_name: Mapped[str | None] = mapped_column(String(50), nullable=True)
    
    # Account status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False)
    
    # Role and permissions
    role: Mapped[str] = mapped_column(String(20), default='user', index=True)  # user, admin, superuser
    permissions: Mapped[list | None] = mapped_column(JSON, nullable=True)
    
    # Login tracking
    last_login: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    login_count: Mapped[int] = mapped_column(Integer, default=0)
    failed_login_attempts: Mapped[int] = mapped_column(Integer, default=0)
    locked_until: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    
    # Additional metadata
    user_key_metadata: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    
    # Relationships
    api_keys: Mapped[list["APIKey"]] = relationship("APIKey", back_populates="user", cascade="all, delete-orphan")
    sessions: Mapped[list["UserSession"]] = relationship("UserSession", back_populates="user", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<User {self.username}>"
    
    def set_password(self, password: str):
        """Set password hash"""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password: str) -> bool:
        """Check password against hash"""
        return check_password_hash(self.password_hash, password)
    
    def is_locked(self) -> bool:
        """Check if account is locked"""
        return self.locked_until and self.locked_until > datetime.utcnow()
    
    def lock_account(self, duration_minutes: int = 30):
        """Lock account for specified duration"""
        self.locked_until = datetime.utcnow() + timedelta(minutes=duration_minutes)
    
    def unlock_account(self):
        """Unlock account"""
        self.locked_until = None
        self.failed_login_attempts = 0
    
    def record_login(self, success: bool):
        """Record login attempt"""
        if success:
            self.last_login = datetime.utcnow()
            self.login_count += 1
            self.failed_login_attempts = 0
        else:
            self.failed_login_attempts += 1
            if self.failed_login_attempts >= 5:  # Lock after 5 failed attempts
                self.lock_account()
    
    def has_permission(self, permission: str) -> bool:
        """Check if user has specific permission"""
        if self.role == 'superuser':
            return True
        if not self.permissions:
            return False
        return permission in self.permissions
    
    def generate_jwt_token(self, secret_key: str, expires_delta: timedelta = None) -> str:
        """Generate JWT token for user"""
        if expires_delta is None:
            expires_delta = timedelta(hours=1)
        
        payload = {
            'user_id': self.id,
            'username': self.username,
            'role': self.role,
            'exp': datetime.utcnow() + expires_delta,
            'iat': datetime.utcnow()
        }
        
        return jwt.encode(payload, secret_key, algorithm='HS256')
    
    def to_dict(self):
        """Convert to dictionary representation"""
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'first_name': self.first_name,
            'last_name': self.last_name,
            'is_active': self.is_active,
            'is_verified': self.is_verified,
            'role': self.role,
            'permissions': self.permissions,
            'last_login': self.last_login.isoformat() if self.last_login else None,
            'login_count': self.login_count,
            'created_at': self.created_at.isoformat(),
            'metadata': self.user_metadata
        }

class APIKey(Base, TimestampMixin):
    """API key model for external access"""
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_id)
    
    # Key details
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    key_hash: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True)
    key_prefix: Mapped[str] = mapped_column(String(20), nullable=False, index=True)  # First few chars for identification
    
    # User association
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey('user.id'), nullable=False, index=True)
    user: Mapped["User"] = relationship("User", back_populates="api_keys")
    
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
    user: Mapped["User"] = relationship("User", back_populates="sessions")
    
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

Index('idx_user_email_active', User.email, User.is_active)
Index('idx_apikey_user_active', APIKey.user_id, APIKey.is_active)
Index('idx_session_user_active', UserSession.user_id, UserSession.is_active)
Index('idx_session_expires', UserSession.expires_at)
