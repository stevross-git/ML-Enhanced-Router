"""
User Model - Complete version with all features
app/models/user.py
"""

from datetime import datetime, timedelta
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import String, Boolean, DateTime, Integer, JSON, Text
from werkzeug.security import generate_password_hash, check_password_hash
import jwt
import secrets

from .base import Base, TimestampMixin, generate_id


class User(Base, TimestampMixin):
    """User account model with full features"""
    
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
    permissions: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    
    # Login tracking
    last_login: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    login_count: Mapped[int] = mapped_column(Integer, default=0)
    failed_login_attempts: Mapped[int] = mapped_column(Integer, default=0)
    locked_until: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    
    # Additional metadata
    user_metadata: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    
    # Profile information
    avatar_url: Mapped[str | None] = mapped_column(String(255), nullable=True)
    timezone: Mapped[str | None] = mapped_column(String(50), nullable=True)
    language: Mapped[str] = mapped_column(String(10), default='en')
    
    # Preferences
    preferences: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    
    # Verification
    email_verification_token: Mapped[str | None] = mapped_column(String(255), nullable=True)
    email_verification_sent_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    password_reset_token: Mapped[str | None] = mapped_column(String(255), nullable=True)
    password_reset_sent_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    
    def __repr__(self):
        return f"<User {self.username}>"
    
    def set_password(self, password: str):
        """Set user password with proper hashing"""
        self.password_hash = generate_password_hash(password)
        self.failed_login_attempts = 0  # Reset on password change
        self.locked_until = None
    
    def check_password(self, password: str) -> bool:
        """Check if provided password is correct"""
        if self.is_locked():
            return False
        return check_password_hash(self.password_hash, password)
    
    def is_admin(self) -> bool:
        """Check if user is admin"""
        return self.role in ['admin', 'superuser']
    
    def is_locked(self) -> bool:
        """Check if user account is locked"""
        if self.locked_until is None:
            return False
        return datetime.utcnow() < self.locked_until
    
    def lock_account(self, minutes: int = 30):
        """Lock user account for specified minutes"""
        self.locked_until = datetime.utcnow() + timedelta(minutes=minutes)
    
    def unlock_account(self):
        """Unlock user account"""
        self.locked_until = None
        self.failed_login_attempts = 0
    
    def record_login_attempt(self, successful: bool = True):
        """Record a login attempt"""
        if successful:
            self.last_login = datetime.utcnow()
            self.login_count += 1
            self.failed_login_attempts = 0
            self.locked_until = None
        else:
            self.failed_login_attempts += 1
            # Lock account after 5 failed attempts
            if self.failed_login_attempts >= 5:
                self.lock_account(30)  # Lock for 30 minutes
    
    def has_permission(self, permission: str) -> bool:
        """Check if user has specific permission"""
        if self.role == 'superuser':
            return True
        
        if not self.permissions:
            return False
        
        # Check direct permission
        if permission in self.permissions:
            return self.permissions[permission]
        
        # Check wildcard permissions
        permission_parts = permission.split('.')
        for i in range(len(permission_parts)):
            wildcard = '.'.join(permission_parts[:i+1]) + '.*'
            if wildcard in self.permissions:
                return self.permissions[wildcard]
        
        return False
    
    def grant_permission(self, permission: str):
        """Grant a permission to user"""
        if self.permissions is None:
            self.permissions = {}
        self.permissions[permission] = True
    
    def revoke_permission(self, permission: str):
        """Revoke a permission from user"""
        if self.permissions and permission in self.permissions:
            del self.permissions[permission]
    
    def get_default_permissions(self) -> dict:
        """Get default permissions based on role"""
        if self.role == 'superuser':
            return {'*': True}
        elif self.role == 'admin':
            return {
                'users.view': True,
                'users.create': True,
                'users.edit': True,
                'models.view': True,
                'models.create': True,
                'models.edit': True,
                'agents.view': True,
                'agents.create': True,
                'agents.edit': True,
                'system.config': True,
                'system.stats': True
            }
        elif self.role == 'user':
            return {
                'models.view': True,
                'models.create': True,
                'agents.view': True,
                'agents.create': True,
                'queries.create': True,
                'queries.view': True
            }
        else:  # readonly or other
            return {
                'models.view': True,
                'agents.view': True,
                'queries.view': True
            }
    
    def generate_email_verification_token(self) -> str:
        """Generate email verification token"""
        token = secrets.token_urlsafe(32)
        self.email_verification_token = token
        self.email_verification_sent_at = datetime.utcnow()
        return token
    
    def generate_password_reset_token(self) -> str:
        """Generate password reset token"""
        token = secrets.token_urlsafe(32)
        self.password_reset_token = token
        self.password_reset_sent_at = datetime.utcnow()
        return token
    
    def verify_email_token(self, token: str) -> bool:
        """Verify email verification token"""
        if not self.email_verification_token:
            return False
        
        # Check if token is expired (24 hours)
        if self.email_verification_sent_at:
            expires_at = self.email_verification_sent_at + timedelta(hours=24)
            if datetime.utcnow() > expires_at:
                return False
        
        if self.email_verification_token == token:
            self.is_verified = True
            self.email_verification_token = None
            self.email_verification_sent_at = None
            return True
        
        return False
    
    def verify_password_reset_token(self, token: str) -> bool:
        """Verify password reset token"""
        if not self.password_reset_token:
            return False
        
        # Check if token is expired (1 hour)
        if self.password_reset_sent_at:
            expires_at = self.password_reset_sent_at + timedelta(hours=1)
            if datetime.utcnow() > expires_at:
                return False
        
        return self.password_reset_token == token
    
    def clear_password_reset_token(self):
        """Clear password reset token"""
        self.password_reset_token = None
        self.password_reset_sent_at = None
    
    def generate_jwt_token(self, secret_key: str, expires_hours: int = 24) -> str:
        """Generate JWT token for user"""
        payload = {
            'user_id': self.id,
            'username': self.username,
            'email': self.email,
            'role': self.role,
            'exp': datetime.utcnow() + timedelta(hours=expires_hours),
            'iat': datetime.utcnow()
        }
        
        return jwt.encode(payload, secret_key, algorithm='HS256')
    
    def update_preferences(self, new_preferences: dict):
        """Update user preferences"""
        if self.preferences is None:
            self.preferences = {}
        self.preferences.update(new_preferences)
    
    def get_preference(self, key: str, default=None):
        """Get user preference value"""
        if not self.preferences:
            return default
        return self.preferences.get(key, default)
    
    def set_preference(self, key: str, value):
        """Set user preference value"""
        if self.preferences is None:
            self.preferences = {}
        self.preferences[key] = value
    
    @property
    def full_name(self) -> str:
        """Get user's full name"""
        if self.first_name and self.last_name:
            return f"{self.first_name} {self.last_name}"
        elif self.first_name:
            return self.first_name
        elif self.last_name:
            return self.last_name
        else:
            return self.username
    
    @property
    def is_email_verified(self) -> bool:
        """Check if email is verified"""
        return self.is_verified
    
    @property
    def days_since_last_login(self) -> int:
        """Get days since last login"""
        if not self.last_login:
            return -1
        delta = datetime.utcnow() - self.last_login
        return delta.days
    
    def to_dict(self, include_sensitive=False):
        """Convert to dictionary representation"""
        data = {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'first_name': self.first_name,
            'last_name': self.last_name,
            'full_name': self.full_name,
            'is_active': self.is_active,
            'is_verified': self.is_verified,
            'role': self.role,
            'permissions': self.permissions,
            'last_login': self.last_login.isoformat() if self.last_login else None,
            'login_count': self.login_count,
            'avatar_url': self.avatar_url,
            'timezone': self.timezone,
            'language': self.language,
            'preferences': self.preferences,
            'is_locked': self.is_locked(),
            'days_since_last_login': self.days_since_last_login,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }
        
        if include_sensitive:
            data.update({
                'failed_login_attempts': self.failed_login_attempts,
                'locked_until': self.locked_until.isoformat() if self.locked_until else None,
                'metadata': self.user_metadata,
                'email_verification_sent_at': self.email_verification_sent_at.isoformat() if self.email_verification_sent_at else None,
                'password_reset_sent_at': self.password_reset_sent_at.isoformat() if self.password_reset_sent_at else None
            })
        
        return data
    
    @classmethod
    def find_by_username_or_email(cls, identifier: str):
        """Find user by username or email"""
        from app.extensions import db
        return db.session.query(cls).filter(
            (cls.username == identifier) | (cls.email == identifier)
        ).first()
    
    @classmethod
    def create_user(cls, username: str, email: str, password: str, role: str = 'user', **kwargs):
        """Create new user with validation"""
        from app.extensions import db
        
        # Check if user exists
        existing = cls.find_by_username_or_email(username) or cls.find_by_username_or_email(email)
        if existing:
            raise ValueError("User with this username or email already exists")
        
        user = cls(
            username=username,
            email=email,
            role=role,
            first_name=kwargs.get('first_name'),
            last_name=kwargs.get('last_name'),
            timezone=kwargs.get('timezone'),
            language=kwargs.get('language', 'en'),
            user_metadata=kwargs.get('metadata', {})
        )
        
        user.set_password(password)
        user.permissions = user.get_default_permissions()
        
        db.session.add(user)
        db.session.commit()
        
        return user
