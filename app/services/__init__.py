"""
Services Package
Business logic layer for the ML Query Router application
"""

# Export service interfaces for easy import
from .ml_router import get_ml_router, MLRouterService
from .ai_models import get_ai_model_manager, AIModelManager
from .auth_service import get_auth_service, AuthService
from .cache_service import get_cache_manager, CacheService
from .rag_service import get_rag_service, RAGService
from .agent_service import get_agent_service, AgentService
from .email_service import get_email_service, EmailService

__all__ = [
    # Service getters
    'get_ml_router',
    'get_ai_model_manager',
    'get_auth_service',
    'get_cache_manager',
    'get_rag_service',
    'get_agent_service',
    'get_email_service',
    
    # Service classes
    'MLRouterService',
    'AIModelManager',
    'AuthService',
    'CacheService',
    'RAGService',
    'AgentService',
    'EmailService'
]
