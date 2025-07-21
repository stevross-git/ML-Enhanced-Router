"""
Authentication Service
Handles user authentication, JWT tokens, and authorization
"""

import os
import jwt
import bcrypt
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
from functools import wraps
from flask import current_app, request, session, g
from sqlalchemy.exc import IntegrityError

from app.extensions import db
from app.models.user import User
from app.models.auth import UserSession, APIKey
from app.utils.exceptions import AuthenticationError, AuthorizationError, ValidationError


class AuthService:
    """Service for handling authentication and authorization"""
    
    def __init__(self):
        self.jwt_secret = None
        self.jwt_algorithm = 'HS256'
        self.token_expiry_hours = 24
    
    def initialize(self, app):
        """Initialize the auth service with app configuration"""
        self.jwt_secret = app.config.get('JWT_SECRET_KEY', app.config.get('SECRET_KEY'))
        self.jwt_algorithm = app.config.get('JWT_ALGORITHM', 'HS256')
        self.token_expiry_hours = app.config.get('JWT_EXPIRY_HOURS', 24)
        
        if not self.jwt_secret:
            raise ValueError("JWT_SECRET_KEY or SECRET_KEY must be configured")
    
    def authenticate_user(self, username: str, password: str) -> Dict[str, Any]:
        """
        Authenticate user with username/password
        
        Args:
            username: User's username or email
            password: Plain text password
            
        Returns:
            Dict containing user info and token
            
        Raises:
            AuthenticationError: If credentials are invalid
        """
        try:
            user = User.query.filter(
                (User.username == username) | (User.email == username)
            ).first()
            
            if not user:
                raise AuthenticationError("Invalid credentials")
            
            if not user.is_active:
                raise AuthenticationError("Account is disabled")
            
            if not self._verify_password(password, user.password_hash):
                raise AuthenticationError("Invalid credentials")
            
            user.last_login = datetime.utcnow()
            db.session.commit()
            
            token = self.generate_jwt_token(user.id)
            
            session_record = UserSession(
                user_id=user.id,
                token_hash=self._hash_token(token),
                expires_at=datetime.utcnow() + timedelta(hours=self.token_expiry_hours),
                ip_address=request.remote_addr,
                user_agent=request.headers.get('User-Agent', '')
            )
            db.session.add(session_record)
            db.session.commit()
            
            return {
                'user': {
                    'id': user.id,
                    'username': user.username,
                    'email': user.email,
                    'role': user.role,
                    'is_active': user.is_active
                },
                'token': token,
                'expires_at': session_record.expires_at.isoformat()
            }
            
        except Exception as e:
            db.session.rollback()
            if isinstance(e, AuthenticationError):
                raise
            current_app.logger.error(f"Authentication error: {str(e)}")
            raise AuthenticationError("Authentication failed")
    
    def generate_jwt_token(self, user_id: int) -> str:
        """
        Generate JWT token for user
        
        Args:
            user_id: User's ID
            
        Returns:
            JWT token string
        """
        payload = {
            'user_id': user_id,
            'iat': datetime.utcnow(),
            'exp': datetime.utcnow() + timedelta(hours=self.token_expiry_hours)
        }
        
        return jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)
    
    def validate_jwt_token(self, token: str) -> Dict[str, Any]:
        """
        Validate JWT token and return user info
        
        Args:
            token: JWT token string
            
        Returns:
            Dict containing user info
            
        Raises:
            AuthenticationError: If token is invalid or expired
        """
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm])
            user_id = payload.get('user_id')
            
            if not user_id:
                raise AuthenticationError("Invalid token payload")
            
            token_hash = self._hash_token(token)
            session_record = UserSession.query.filter_by(
                user_id=user_id,
                token_hash=token_hash,
                is_active=True
            ).first()
            
            if not session_record:
                raise AuthenticationError("Session not found")
            
            if session_record.expires_at < datetime.utcnow():
                session_record.is_active = False
                db.session.commit()
                raise AuthenticationError("Token expired")
            
            user = User.query.get(user_id)
            if not user or not user.is_active:
                raise AuthenticationError("User not found or inactive")
            
            return {
                'user_id': user.id,
                'username': user.username,
                'email': user.email,
                'role': user.role,
                'session_id': session_record.id
            }
            
        except jwt.ExpiredSignatureError:
            raise AuthenticationError("Token expired")
        except jwt.InvalidTokenError:
            raise AuthenticationError("Invalid token")
        except Exception as e:
            current_app.logger.error(f"Token validation error: {str(e)}")
            raise AuthenticationError("Token validation failed")
    
    def check_permissions(self, user_id: int, resource: str, action: str = 'read') -> bool:
        """
        Check if user has permission for resource/action
        
        Args:
            user_id: User's ID
            resource: Resource name (e.g., 'models', 'queries', 'admin')
            action: Action type ('read', 'write', 'delete', 'admin')
            
        Returns:
            True if user has permission, False otherwise
        """
        try:
            user = User.query.get(user_id)
            if not user or not user.is_active:
                return False
            
            if user.role == 'admin':
                return True
            
            permissions = {
                'user': {
                    'queries': ['read', 'write'],
                    'models': ['read'],
                    'profile': ['read', 'write']
                },
                'moderator': {
                    'queries': ['read', 'write', 'delete'],
                    'models': ['read', 'write'],
                    'users': ['read'],
                    'profile': ['read', 'write']
                },
                'admin': {
                    '*': ['*']  # All permissions
                }
            }
            
            user_permissions = permissions.get(user.role, {})
            resource_permissions = user_permissions.get(resource, [])
            
            return action in resource_permissions or '*' in resource_permissions
            
        except Exception as e:
            current_app.logger.error(f"Permission check error: {str(e)}")
            return False
    
    def create_user(self, user_data: Dict[str, Any]) -> str:
        """
        Create new user account
        
        Args:
            user_data: Dict containing username, email, password, role
            
        Returns:
            User ID of created user
            
        Raises:
            ValidationError: If user data is invalid
        """
        try:
            required_fields = ['username', 'email', 'password']
            for field in required_fields:
                if not user_data.get(field):
                    raise ValidationError(f"Missing required field: {field}")
            
            existing_user = User.query.filter(
                (User.username == user_data['username']) | 
                (User.email == user_data['email'])
            ).first()
            
            if existing_user:
                raise ValidationError("Username or email already exists")
            
            password_hash = self._hash_password(user_data['password'])
            
            user = User(
                username=user_data['username'],
                email=user_data['email'],
                password_hash=password_hash,
                role=user_data.get('role', 'user'),
                is_active=user_data.get('is_active', True),
                created_at=datetime.utcnow()
            )
            
            db.session.add(user)
            db.session.commit()
            
            current_app.logger.info(f"User created: {user.username} (ID: {user.id})")
            return user.id
            
        except IntegrityError:
            db.session.rollback()
            raise ValidationError("Username or email already exists")
        except Exception as e:
            db.session.rollback()
            current_app.logger.error(f"User creation error: {str(e)}")
            raise ValidationError("Failed to create user")
    
    def reset_password(self, user_id: int, new_password: str) -> bool:
        """
        Reset user password
        
        Args:
            user_id: User's ID
            new_password: New plain text password
            
        Returns:
            True if password was reset successfully
        """
        try:
            user = User.query.get(user_id)
            if not user:
                raise ValidationError("User not found")
            
            password_hash = self._hash_password(new_password)
            user.password_hash = password_hash
            user.password_changed_at = datetime.utcnow()
            
            UserSession.query.filter_by(user_id=user_id, is_active=True).update({
                'is_active': False
            })
            
            db.session.commit()
            
            current_app.logger.info(f"Password reset for user ID: {user_id}")
            return True
            
        except Exception as e:
            db.session.rollback()
            current_app.logger.error(f"Password reset error: {str(e)}")
            return False
    
    def logout_user(self, token: str) -> bool:
        """
        Logout user by invalidating session
        
        Args:
            token: JWT token to invalidate
            
        Returns:
            True if logout successful
        """
        try:
            token_hash = self._hash_token(token)
            session_record = UserSession.query.filter_by(
                token_hash=token_hash,
                is_active=True
            ).first()
            
            if session_record:
                session_record.is_active = False
                session_record.logged_out_at = datetime.utcnow()
                db.session.commit()
            
            return True
            
        except Exception as e:
            current_app.logger.error(f"Logout error: {str(e)}")
            return False
    
    def get_current_user(self) -> Optional[Dict[str, Any]]:
        """
        Get current authenticated user from request context
        
        Returns:
            User info dict or None if not authenticated
        """
        if hasattr(g, 'current_user'):
            return g.current_user
        return None
    
    def require_auth(self, roles: List[str] = None):
        """
        Decorator to require authentication for routes
        
        Args:
            roles: List of required roles (optional)
        """
        def decorator(f):
            @wraps(f)
            def decorated_function(*args, **kwargs):
                auth_header = request.headers.get('Authorization')
                if not auth_header or not auth_header.startswith('Bearer '):
                    raise AuthenticationError("Missing or invalid authorization header")
                
                token = auth_header.split(' ')[1]
                
                try:
                    user_info = self.validate_jwt_token(token)
                    g.current_user = user_info
                    
                    if roles and user_info.get('role') not in roles:
                        raise AuthorizationError("Insufficient permissions")
                    
                    return f(*args, **kwargs)
                    
                except (AuthenticationError, AuthorizationError):
                    raise
                except Exception as e:
                    current_app.logger.error(f"Auth decorator error: {str(e)}")
                    raise AuthenticationError("Authentication failed")
            
            return decorated_function
        return decorator
    
    def _hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    def _verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password against hash"""
        return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))
    
    def _hash_token(self, token: str) -> str:
        """Hash token for storage"""
        import hashlib
        return hashlib.sha256(token.encode('utf-8')).hexdigest()


# Singleton instance
_auth_service = None

def get_auth_service() -> AuthService:
    """Get singleton auth service instance"""
    global _auth_service
    if _auth_service is None:
        _auth_service = AuthService()
        if current_app:
            _auth_service.initialize(current_app)
    return _auth_service

def init_auth_service(app):
    """Initialize auth service with Flask app"""
    service = get_auth_service()
    service.initialize(app)
    return service
