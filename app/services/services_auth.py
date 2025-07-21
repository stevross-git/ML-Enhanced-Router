"""
Authentication Service
Handles user authentication, authorization, and session management
"""

import jwt
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, List

from flask import current_app
from ..models import User, APIKey, UserSession
from ..extensions import db
from ..utils.exceptions import AuthenticationError, ValidationError

class AuthService:
    """
    Authentication and authorization service
    """
    
    def __init__(self):
        self.initialized = False
        self.users = {}  # In-memory cache for users
        self.api_keys = {}  # In-memory cache for API keys
        
    def initialize(self):
        """Initialize the auth service"""
        try:
            # Load users from database
            self._load_users()
            
            # Create default admin user if none exists
            self._ensure_admin_user()
            
            self.initialized = True
            
        except Exception as e:
            current_app.logger.error(f"Auth service initialization failed: {e}")
            raise
    
    def _load_users(self):
        """Load users from database into memory cache"""
        try:
            users = db.session.query(User).filter_by(is_active=True).all()
            self.users = {user.username: user for user in users}
            
        except Exception as e:
            current_app.logger.error(f"Failed to load users: {e}")
            self.users = {}
    
    def _ensure_admin_user(self):
        """Ensure default admin user exists"""
        try:
            admin_user = db.session.query(User).filter_by(username='admin').first()
            
            if not admin_user:
                # Create default admin user
                admin_user = User(
                    username='admin',
                    email='admin@mlrouter.local',
                    role='admin',
                    is_active=True,
                    is_verified=True
                )
                admin_user.set_password('admin123')  # Should be changed on first login
                
                db.session.add(admin_user)
                db.session.commit()
                
                # Create API key for admin
                api_key, key_hash = APIKey.generate_key()
                admin_api_key = APIKey(
                    name='Admin Default Key',
                    key_hash=key_hash,
                    key_prefix=api_key[:8],
                    user_id=admin_user.id,
                    permissions=['admin']
                )
                
                db.session.add(admin_api_key)
                db.session.commit()
                
                # Add to cache
                self.users['admin'] = admin_user
                
                current_app.logger.info("Created default admin user with API key")
                
        except Exception as e:
            current_app.logger.error(f"Failed to ensure admin user: {e}")
            db.session.rollback()
    
    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """
        Authenticate user with username and password
        
        Args:
            username: Username
            password: Password
            
        Returns:
            User object if authentication successful, None otherwise
        """
        try:
            user = db.session.query(User).filter_by(username=username, is_active=True).first()
            
            if not user:
                return None
            
            # Check if account is locked
            if user.is_locked():
                raise AuthenticationError("Account is locked. Please try again later.")
            
            # Verify password
            if user.check_password(password):
                # Record successful login
                user.record_login(success=True)
                db.session.commit()
                
                # Update cache
                self.users[username] = user
                
                return user
            else:
                # Record failed login
                user.record_login(success=False)
                db.session.commit()
                
                return None
                
        except Exception as e:
            current_app.logger.error(f"Authentication error: {e}")
            return None
    
    def authenticate_api_key(self, api_key: str) -> Optional[User]:
        """
        Authenticate user with API key
        
        Args:
            api_key: API key string
            
        Returns:
            User object if authentication successful, None otherwise
        """
        try:
            # Find API key by prefix
            key_prefix = api_key[:8]
            api_key_obj = db.session.query(APIKey).filter_by(
                key_prefix=key_prefix,
                is_active=True
            ).first()
            
            if not api_key_obj:
                return None
            
            # Verify key
            if not api_key_obj.check_key(api_key):
                return None
            
            # Check if key is valid
            if not api_key_obj.is_valid():
                return None
            
            # Get user
            user = db.session.query(User).filter_by(
                id=api_key_obj.user_id,
                is_active=True
            ).first()
            
            if user:
                # Record API key usage
                api_key_obj.record_usage()
                db.session.commit()
            
            return user
            
        except Exception as e:
            current_app.logger.error(f"API key authentication error: {e}")
            return None
    
    def authenticate_jwt(self, token: str) -> Optional[User]:
        """
        Authenticate user with JWT token
        
        Args:
            token: JWT token string
            
        Returns:
            User object if authentication successful, None otherwise
        """
        try:
            # Decode JWT token
            secret_key = current_app.config.get('JWT_SECRET_KEY')
            payload = jwt.decode(token, secret_key, algorithms=['HS256'])
            
            user_id = payload.get('user_id')
            if not user_id:
                return None
            
            # Get user
            user = db.session.query(User).filter_by(
                id=user_id,
                is_active=True
            ).first()
            
            return user
            
        except jwt.ExpiredSignatureError:
            current_app.logger.warning("JWT token expired")
            return None
        except jwt.InvalidTokenError:
            current_app.logger.warning("Invalid JWT token")
            return None
        except Exception as e:
            current_app.logger.error(f"JWT authentication error: {e}")
            return None
    
    def create_user(self, username: str, email: str, password: str, 
                   role: str = 'user') -> Optional[User]:
        """
        Create a new user
        
        Args:
            username: Username
            email: Email address
            password: Password
            role: User role
            
        Returns:
            User object if creation successful, None otherwise
        """
        try:
            # Check if user already exists
            existing_user = db.session.query(User).filter(
                (User.username == username) | (User.email == email)
            ).first()
            
            if existing_user:
                raise ValidationError("Username or email already exists")
            
            # Create user
            user = User(
                username=username,
                email=email,
                role=role,
                is_active=True
            )
            user.set_password(password)
            
            db.session.add(user)
            db.session.commit()
            
            # Add to cache
            self.users[username] = user
            
            return user
            
        except ValidationError:
            raise
        except Exception as e:
            current_app.logger.error(f"User creation error: {e}")
            db.session.rollback()
            return None
    
    def get_user(self, user_id: str) -> Optional[User]:
        """
        Get user by ID
        
        Args:
            user_id: User ID
            
        Returns:
            User object if found, None otherwise
        """
        try:
            return db.session.query(User).filter_by(id=user_id, is_active=True).first()
        except Exception as e:
            current_app.logger.error(f"Error getting user: {e}")
            return None
    
    def get_all_users(self) -> List[User]:
        """
        Get all users
        
        Returns:
            List of User objects
        """
        try:
            return db.session.query(User).filter_by(is_active=True).all()
        except Exception as e:
            current_app.logger.error(f"Error getting all users: {e}")
            return []
    
    def generate_jwt_token(self, user_id: str, expires_delta: Optional[timedelta] = None) -> str:
        """
        Generate JWT token for user
        
        Args:
            user_id: User ID
            expires_delta: Token expiration time
            
        Returns:
            JWT token string
        """
        try:
            user = self.get_user(user_id)
            if not user:
                raise AuthenticationError("User not found")
            
            if expires_delta is None:
                expires_delta = timedelta(hours=1)
            
            payload = {
                'user_id': user.id,
                'username': user.username,
                'role': user.role,
                'exp': datetime.utcnow() + expires_delta,
                'iat': datetime.utcnow()
            }
            
            secret_key = current_app.config.get('JWT_SECRET_KEY')
            return jwt.encode(payload, secret_key, algorithm='HS256')
            
        except Exception as e:
            current_app.logger.error(f"JWT generation error: {e}")
            raise AuthenticationError("Failed to generate token")
    
    def verify_jwt_token(self, token: str) -> Optional[Dict]:
        """
        Verify JWT token and return payload
        
        Args:
            token: JWT token string
            
        Returns:
            Token payload if valid, None otherwise
        """
        try:
            secret_key = current_app.config.get('JWT_SECRET_KEY')
            payload = jwt.decode(token, secret_key, algorithms=['HS256'])
            return payload
            
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
        except Exception as e:
            current_app.logger.error(f"JWT verification error: {e}")
            return None
    
    def regenerate_api_key(self, user_id: str) -> Optional[str]:
        """
        Regenerate API key for user
        
        Args:
            user_id: User ID
            
        Returns:
            New API key if successful, None otherwise
        """
        try:
            user = self.get_user(user_id)
            if not user:
                return None
            
            # Find existing API key
            existing_key = db.session.query(APIKey).filter_by(user_id=user_id).first()
            
            if existing_key:
                # Deactivate existing key
                existing_key.is_active = False
            
            # Generate new API key
            api_key, key_hash = APIKey.generate_key()
            new_api_key = APIKey(
                name=f'Regenerated Key - {datetime.now().strftime("%Y-%m-%d")}',
                key_hash=key_hash,
                key_prefix=api_key[:8],
                user_id=user_id,
                permissions=existing_key.permissions if existing_key else ['user']
            )
            
            db.session.add(new_api_key)
            db.session.commit()
            
            return api_key
            
        except Exception as e:
            current_app.logger.error(f"API key regeneration error: {e}")
            db.session.rollback()
            return None
    
    def change_password(self, user_id: str, current_password: str, new_password: str) -> bool:
        """
        Change user password
        
        Args:
            user_id: User ID
            current_password: Current password
            new_password: New password
            
        Returns:
            True if successful, False otherwise
        """
        try:
            user = self.get_user(user_id)
            if not user:
                return False
            
            # Verify current password
            if not user.check_password(current_password):
                return False
            
            # Set new password
            user.set_password(new_password)
            db.session.commit()
            
            return True
            
        except Exception as e:
            current_app.logger.error(f"Password change error: {e}")
            db.session.rollback()
            return False
    
    def create_session(self, user_id: str, ip_address: str = None, 
                      user_agent: str = None) -> Optional[UserSession]:
        """
        Create user session
        
        Args:
            user_id: User ID
            ip_address: Client IP address
            user_agent: Client user agent
            
        Returns:
            UserSession object if successful, None otherwise
        """
        try:
            session_token = secrets.token_urlsafe(32)
            expires_at = datetime.utcnow() + timedelta(hours=24)
            
            session = UserSession(
                session_token=session_token,
                user_id=user_id,
                ip_address=ip_address,
                user_agent=user_agent,
                expires_at=expires_at
            )
            
            db.session.add(session)
            db.session.commit()
            
            return session
            
        except Exception as e:
            current_app.logger.error(f"Session creation error: {e}")
            db.session.rollback()
            return None
    
    def validate_session(self, session_token: str) -> Optional[UserSession]:
        """
        Validate user session
        
        Args:
            session_token: Session token
            
        Returns:
            UserSession object if valid, None otherwise
        """
        try:
            session = db.session.query(UserSession).filter_by(
                session_token=session_token,
                is_active=True
            ).first()
            
            if session and session.is_valid():
                # Refresh session
                session.refresh()
                db.session.commit()
                return session
            
            return None
            
        except Exception as e:
            current_app.logger.error(f"Session validation error: {e}")
            return None
    
    def invalidate_session(self, session_token: str) -> bool:
        """
        Invalidate user session
        
        Args:
            session_token: Session token
            
        Returns:
            True if successful, False otherwise
        """
        try:
            session = db.session.query(UserSession).filter_by(
                session_token=session_token
            ).first()
            
            if session:
                session.invalidate()
                db.session.commit()
                return True
            
            return False
            
        except Exception as e:
            current_app.logger.error(f"Session invalidation error: {e}")
            db.session.rollback()
            return False

# Global service instance
_auth_service = None

def get_auth_service() -> Optional[AuthService]:
    """Get the global auth service instance"""
    global _auth_service
    
    if _auth_service is None:
        try:
            _auth_service = AuthService()
            _auth_service.initialize()
        except Exception as e:
            current_app.logger.error(f"Failed to create auth service: {e}")
            return None
    
    return _auth_service
