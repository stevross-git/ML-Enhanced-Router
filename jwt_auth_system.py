"""
JWT Authentication System for AI-to-AI Communication
Provides secure token-based authentication with role-based access control
"""

import jwt
import time
import uuid
import hashlib
import secrets
import logging
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import json
import os

logger = logging.getLogger(__name__)

class Permission(Enum):
    """Available permissions for AI agents"""
    READ_CONTEXT = "read_context"
    WRITE_CONTEXT = "write_context"
    DELETE_CONTEXT = "delete_context"
    QUERY_AI = "query_ai"
    RECEIVE_QUERIES = "receive_queries"
    SYSTEM_STATUS = "system_status"
    ADMIN = "admin"
    EMERGENCY_ACCESS = "emergency_access"

class AgentRole(Enum):
    """Predefined agent roles"""
    GUEST = "guest"
    AGENT = "agent"
    COORDINATOR = "coordinator"
    ADMIN = "admin"
    SYSTEM = "system"

@dataclass
class AgentCredentials:
    """Agent authentication credentials"""
    agent_id: str
    secret_key: str
    role: AgentRole
    permissions: List[Permission]
    created_at: int
    last_used: int
    is_active: bool = True
    max_token_lifetime: int = 3600  # 1 hour default
    allowed_origins: List[str] = None
    
    def __post_init__(self):
        if self.allowed_origins is None:
            self.allowed_origins = []

@dataclass
class TokenClaims:
    """JWT token claims structure"""
    agent_id: str
    role: str
    permissions: List[str]
    issued_at: int
    expires_at: int
    issuer: str
    token_id: str
    session_id: Optional[str] = None
    ip_address: Optional[str] = None

class JWTAuthSystem:
    """JWT-based authentication system for AI agents"""
    
    def __init__(
        self,
        secret_key: Optional[str] = None,
        algorithm: str = 'HS256',
        default_token_lifetime: int = 3600,
        issuer: str = 'ai-communication-system',
        key_rotation_interval: int = 86400  # 24 hours
    ):
        """Initialize JWT authentication system"""
        
        self.algorithm = algorithm
        self.default_token_lifetime = default_token_lifetime
        self.issuer = issuer
        self.key_rotation_interval = key_rotation_interval
        
        # Initialize secret key
        if secret_key:
            self.secret_key = secret_key
        else:
            self.secret_key = self._generate_secret_key()
        
        # Store for agent credentials
        self.agent_credentials: Dict[str, AgentCredentials] = {}
        
        # Revoked tokens store (in production, use Redis)
        self.revoked_tokens: Set[str] = set()
        
        # Rate limiting store
        self.rate_limits: Dict[str, List[int]] = {}
        
        # Key rotation tracking
        self.key_created_at = int(time.time())
        self.previous_keys: List[Tuple[str, int]] = []  # (key, created_at)
        
        # Role-based permission mapping
        self.role_permissions = {
            AgentRole.GUEST: [Permission.READ_CONTEXT],
            AgentRole.AGENT: [
                Permission.READ_CONTEXT, 
                Permission.WRITE_CONTEXT, 
                Permission.QUERY_AI, 
                Permission.RECEIVE_QUERIES
            ],
            AgentRole.COORDINATOR: [
                Permission.READ_CONTEXT, 
                Permission.WRITE_CONTEXT, 
                Permission.DELETE_CONTEXT,
                Permission.QUERY_AI, 
                Permission.RECEIVE_QUERIES, 
                Permission.SYSTEM_STATUS
            ],
            AgentRole.ADMIN: [
                Permission.READ_CONTEXT, 
                Permission.WRITE_CONTEXT, 
                Permission.DELETE_CONTEXT,
                Permission.QUERY_AI, 
                Permission.RECEIVE_QUERIES, 
                Permission.SYSTEM_STATUS, 
                Permission.ADMIN
            ],
            AgentRole.SYSTEM: [perm for perm in Permission]  # All permissions
        }
        
        logger.info(f"JWT authentication system initialized with {algorithm} algorithm")
    
    def register_agent(
        self,
        agent_id: str,
        role: AgentRole = AgentRole.AGENT,
        custom_permissions: Optional[List[Permission]] = None,
        max_token_lifetime: Optional[int] = None,
        allowed_origins: Optional[List[str]] = None
    ) -> str:
        """Register a new AI agent and return its secret key"""
        
        if agent_id in self.agent_credentials:
            raise ValueError(f"Agent {agent_id} already exists")
        
        # Generate secret key for the agent
        secret_key = self._generate_agent_secret()
        
        # Determine permissions
        if custom_permissions:
            permissions = custom_permissions
        else:
            permissions = self.role_permissions.get(role, [])
        
        # Create credentials
        credentials = AgentCredentials(
            agent_id=agent_id,
            secret_key=secret_key,
            role=role,
            permissions=permissions,
            created_at=int(time.time()),
            last_used=int(time.time()),
            max_token_lifetime=max_token_lifetime or self.default_token_lifetime,
            allowed_origins=allowed_origins or []
        )
        
        self.agent_credentials[agent_id] = credentials
        
        logger.info(f"Registered agent {agent_id} with role {role.value}")
        return secret_key
    
    def authenticate_agent(self, agent_id: str, secret_key: str) -> bool:
        """Authenticate an agent using its credentials"""
        
        credentials = self.agent_credentials.get(agent_id)
        if not credentials:
            logger.warning(f"Authentication failed: Agent {agent_id} not found")
            return False
        
        if not credentials.is_active:
            logger.warning(f"Authentication failed: Agent {agent_id} is inactive")
            return False
        
        if credentials.secret_key != secret_key:
            logger.warning(f"Authentication failed: Invalid secret key for {agent_id}")
            return False
        
        # Update last used timestamp
        credentials.last_used = int(time.time())
        
        logger.info(f"Agent {agent_id} authenticated successfully")
        return True
    
    def generate_token(
        self,
        agent_id: str,
        session_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        custom_lifetime: Optional[int] = None
    ) -> str:
        """Generate JWT token for authenticated agent"""
        
        credentials = self.agent_credentials.get(agent_id)
        if not credentials or not credentials.is_active:
            raise ValueError(f"Agent {agent_id} not found or inactive")
        
        # Determine token lifetime
        lifetime = custom_lifetime or credentials.max_token_lifetime
        
        # Create token claims
        current_time = int(time.time())
        token_id = str(uuid.uuid4())
        
        claims = TokenClaims(
            agent_id=agent_id,
            role=credentials.role.value,
            permissions=[perm.value for perm in credentials.permissions],
            issued_at=current_time,
            expires_at=current_time + lifetime,
            issuer=self.issuer,
            token_id=token_id,
            session_id=session_id,
            ip_address=ip_address
        )
        
        # Generate JWT token
        payload = asdict(claims)
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        
        # Update rate limiting
        self._update_rate_limit(agent_id)
        
        logger.info(f"Generated token for agent {agent_id} (expires in {lifetime}s)")
        return token
    
    def verify_token(self, token: str, required_permission: Optional[Permission] = None) -> Optional[TokenClaims]:
        """Verify JWT token and return claims if valid"""
        
        try:
            # Check if token is revoked
            if token in self.revoked_tokens:
                logger.warning("Token verification failed: Token is revoked")
                return None
            
            # Decode token (try current key first, then previous keys)
            payload = None
            
            try:
                payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            except jwt.InvalidTokenError:
                # Try previous keys for graceful key rotation
                for prev_key, _ in self.previous_keys:
                    try:
                        payload = jwt.decode(token, prev_key, algorithms=[self.algorithm])
                        break
                    except jwt.InvalidTokenError:
                        continue
            
            if not payload:
                logger.warning("Token verification failed: Invalid token")
                return None
            
            # Parse claims
            claims = TokenClaims(**payload)
            
            # Verify expiration
            current_time = int(time.time())
            if claims.expires_at < current_time:
                logger.warning("Token verification failed: Token expired")
                return None
            
            # Verify agent exists and is active
            credentials = self.agent_credentials.get(claims.agent_id)
            if not credentials or not credentials.is_active:
                logger.warning(f"Token verification failed: Agent {claims.agent_id} not found or inactive")
                return None
            
            # Verify required permission
            if required_permission:
                if required_permission.value not in claims.permissions:
                    logger.warning(f"Token verification failed: Missing permission {required_permission.value}")
                    return None
            
            logger.debug(f"Token verified successfully for agent {claims.agent_id}")
            return claims
            
        except Exception as e:
            logger.error(f"Token verification error: {e}")
            return None
    
    def revoke_token(self, token: str) -> bool:
        """Revoke a specific token"""
        try:
            # Decode token to get token ID
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm], options={"verify_exp": False})
            token_id = payload.get('token_id')
            
            if token_id:
                self.revoked_tokens.add(token_id)
                logger.info(f"Token {token_id} revoked successfully")
                return True
            
        except Exception as e:
            logger.error(f"Error revoking token: {e}")
        
        # Fallback: add full token to revoked list
        self.revoked_tokens.add(token)
        return True
    
    def revoke_all_agent_tokens(self, agent_id: str) -> bool:
        """Revoke all tokens for a specific agent"""
        credentials = self.agent_credentials.get(agent_id)
        if not credentials:
            return False
        
        # Generate new secret key for the agent (invalidates all existing tokens)
        credentials.secret_key = self._generate_agent_secret()
        credentials.last_used = int(time.time())
        
        logger.info(f"All tokens revoked for agent {agent_id}")
        return True
    
    def deactivate_agent(self, agent_id: str) -> bool:
        """Deactivate an agent (prevents new token generation)"""
        credentials = self.agent_credentials.get(agent_id)
        if not credentials:
            return False
        
        credentials.is_active = False
        logger.info(f"Agent {agent_id} deactivated")
        return True
    
    def update_agent_permissions(self, agent_id: str, permissions: List[Permission]) -> bool:
        """Update agent permissions"""
        credentials = self.agent_credentials.get(agent_id)
        if not credentials:
            return False
        
        credentials.permissions = permissions
        logger.info(f"Updated permissions for agent {agent_id}")
        return True
    
    def rotate_secret_key(self) -> str:
        """Rotate the main secret key"""
        # Store previous key for graceful transition
        self.previous_keys.append((self.secret_key, self.key_created_at))
        
        # Keep only last 3 keys for rotation
        if len(self.previous_keys) > 3:
            self.previous_keys.pop(0)
        
        # Generate new key
        self.secret_key = self._generate_secret_key()
        self.key_created_at = int(time.time())
        
        logger.info("Secret key rotated successfully")
        return self.secret_key
    
    def cleanup_expired_tokens(self) -> int:
        """Clean up expired tokens and old keys"""
        current_time = int(time.time())
        cleanup_count = 0
        
        # Remove old revoked tokens (keep for 24 hours after expiration)
        tokens_to_remove = []
        for token in self.revoked_tokens:
            try:
                payload = jwt.decode(token, options={"verify_signature": False, "verify_exp": False})
                if payload.get('expires_at', 0) + 86400 < current_time:
                    tokens_to_remove.append(token)
            except:
                # If we can't decode, remove it
                tokens_to_remove.append(token)
        
        for token in tokens_to_remove:
            self.revoked_tokens.remove(token)
            cleanup_count += 1
        
        # Remove old previous keys (older than 24 hours)
        keys_to_remove = []
        for key, created_at in self.previous_keys:
            if created_at + 86400 < current_time:
                keys_to_remove.append((key, created_at))
        
        for key_pair in keys_to_remove:
            self.previous_keys.remove(key_pair)
            cleanup_count += 1
        
        if cleanup_count > 0:
            logger.info(f"Cleaned up {cleanup_count} expired tokens/keys")
        
        return cleanup_count
    
    def get_agent_info(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get agent information (excluding secret key)"""
        credentials = self.agent_credentials.get(agent_id)
        if not credentials:
            return None
        
        return {
            "agent_id": credentials.agent_id,
            "role": credentials.role.value,
            "permissions": [perm.value for perm in credentials.permissions],
            "created_at": credentials.created_at,
            "last_used": credentials.last_used,
            "is_active": credentials.is_active,
            "max_token_lifetime": credentials.max_token_lifetime,
            "allowed_origins": credentials.allowed_origins
        }
    
    def list_agents(self) -> List[Dict[str, Any]]:
        """List all registered agents"""
        return [self.get_agent_info(agent_id) for agent_id in self.agent_credentials.keys()]
    
    def check_rate_limit(self, agent_id: str, max_requests: int = 60, time_window: int = 60) -> bool:
        """Check if agent is within rate limits"""
        current_time = int(time.time())
        
        # Get agent's request history
        if agent_id not in self.rate_limits:
            self.rate_limits[agent_id] = []
        
        requests = self.rate_limits[agent_id]
        
        # Remove old requests outside time window
        requests[:] = [req_time for req_time in requests if req_time > current_time - time_window]
        
        # Check if within limit
        return len(requests) < max_requests
    
    def _update_rate_limit(self, agent_id: str):
        """Update rate limiting for agent"""
        current_time = int(time.time())
        
        if agent_id not in self.rate_limits:
            self.rate_limits[agent_id] = []
        
        self.rate_limits[agent_id].append(current_time)
    
    def _generate_secret_key(self) -> str:
        """Generate a secure secret key"""
        return secrets.token_urlsafe(64)
    
    def _generate_agent_secret(self) -> str:
        """Generate a secure secret for agent authentication"""
        return secrets.token_urlsafe(32)
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get authentication system statistics"""
        current_time = int(time.time())
        
        active_agents = sum(1 for cred in self.agent_credentials.values() if cred.is_active)
        total_agents = len(self.agent_credentials)
        
        # Calculate recent activity
        recent_activity = sum(
            1 for cred in self.agent_credentials.values() 
            if cred.last_used > current_time - 3600  # Last hour
        )
        
        return {
            "total_agents": total_agents,
            "active_agents": active_agents,
            "recent_activity": recent_activity,
            "revoked_tokens": len(self.revoked_tokens),
            "key_rotation_age": current_time - self.key_created_at,
            "key_rotation_due": (current_time - self.key_created_at) > self.key_rotation_interval,
            "previous_keys_count": len(self.previous_keys)
        }
    
    def export_config(self) -> Dict[str, Any]:
        """Export system configuration (excluding secrets)"""
        return {
            "algorithm": self.algorithm,
            "default_token_lifetime": self.default_token_lifetime,
            "issuer": self.issuer,
            "key_rotation_interval": self.key_rotation_interval,
            "agents": [self.get_agent_info(agent_id) for agent_id in self.agent_credentials.keys()]
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on authentication system"""
        try:
            # Test token generation and verification
            test_agent_id = "health_check_agent"
            
            # Temporarily register test agent
            if test_agent_id not in self.agent_credentials:
                secret = self.register_agent(test_agent_id, AgentRole.GUEST)
                test_token = self.generate_token(test_agent_id)
                claims = self.verify_token(test_token)
                
                # Cleanup
                del self.agent_credentials[test_agent_id]
                
                is_healthy = claims is not None and claims.agent_id == test_agent_id
            else:
                is_healthy = True
            
            return {
                "healthy": is_healthy,
                "stats": self.get_system_stats(),
                "errors": []
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "healthy": False,
                "errors": [str(e)],
                "stats": {}
            }

# Utility functions
def create_system_agent(auth_system: JWTAuthSystem) -> Tuple[str, str]:
    """Create a system agent with full permissions"""
    agent_id = "system_agent"
    secret_key = auth_system.register_agent(agent_id, AgentRole.SYSTEM)
    return agent_id, secret_key

def create_admin_agent(auth_system: JWTAuthSystem, agent_id: str) -> str:
    """Create an admin agent"""
    return auth_system.register_agent(agent_id, AgentRole.ADMIN)

# Decorator for token-based authentication
def require_auth(permission: Optional[Permission] = None):
    """Decorator to require authentication for function calls"""
    def decorator(func):
        def wrapper(self, token: str, *args, **kwargs):
            # Assuming the class has an auth_system attribute
            if not hasattr(self, 'auth_system'):
                raise RuntimeError("No authentication system available")
            
            claims = self.auth_system.verify_token(token, permission)
            if not claims:
                raise PermissionError("Authentication failed")
            
            # Add claims to kwargs for the function to use
            kwargs['auth_claims'] = claims
            return func(self, *args, **kwargs)
        
        return wrapper
    return decorator