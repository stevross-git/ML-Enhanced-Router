"""
Database Models Package
Centralized import and initialization of all database models
"""

from ..extensions import db

# Import base classes first
from .base import Base, TimestampMixin

# Import all model classes with error handling
try:
    from .user import User
    USER_MODELS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: User models not available: {e}")
    USER_MODELS_AVAILABLE = False

try:
    from .agent import Agent, AgentCapability, AgentSession, AgentMetrics, AgentRegistration
    AGENT_MODELS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Agent models not available: {e}")
    AGENT_MODELS_AVAILABLE = False

try:
    from .query import QueryLog, QueryMetrics
    QUERY_MODELS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Query models not available: {e}")
    QUERY_MODELS_AVAILABLE = False

try:
    from .ai_model import MLModelRegistry, ModelConfiguration
    AI_MODEL_MODELS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: AI model models not available: {e}")
    AI_MODEL_MODELS_AVAILABLE = False

try:
    from .cache import AICacheEntry, AICacheStats
    CACHE_MODELS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Cache models not available: {e}")
    CACHE_MODELS_AVAILABLE = False

try:
    from .auth import APIKey, UserSession
    AUTH_MODELS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Auth models not available: {e}")
    AUTH_MODELS_AVAILABLE = False

try:
    from .rag import Document, DocumentChunk, RAGQuery
    RAG_MODELS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: RAG models not available: {e}")
    RAG_MODELS_AVAILABLE = False

def init_db(app):
    """
    Initialize database with Flask app
    
    Args:
        app: Flask application instance
    """
    try:
        with app.app_context():
            # Create all tables
            db.create_all()
            
            # Create initial data if needed
            _create_initial_data()
            
            app.logger.info("Database initialized successfully")
            
    except Exception as e:
        app.logger.error(f"Database initialization failed: {e}")
        raise

def _create_initial_data():
    """Create initial data for the application"""
    try:
        # Only create initial data if tables are empty
        if USER_MODELS_AVAILABLE:
            from .user import User
            if User.query.count() == 0:
                _create_default_user()
        
    except Exception as e:
        print(f"Warning: Could not create initial data: {e}")

def _create_default_user():
    """Create default admin user"""
    try:
        from .user import User
        from werkzeug.security import generate_password_hash
        
        admin_user = User(
            username='admin',
            email='admin@peoplesainetwork.com',
            password_hash=generate_password_hash('admin123'),
            first_name='Admin',
            last_name='User',
            is_active=True,
            is_verified=True,
            role='admin'
        )
        
        db.session.add(admin_user)
        db.session.commit()
        
        print("Created default admin user (admin/admin123)")
        
    except Exception as e:
        db.session.rollback()
        print(f"Could not create default user: {e}")

# Export all available models
__all__ = [
    'Base',
    'TimestampMixin',
    'init_db'
]

# Add available models to exports
if USER_MODELS_AVAILABLE:
    __all__.append('User')

if AGENT_MODELS_AVAILABLE:
    __all__.extend(['Agent', 'AgentCapability', 'AgentSession', 'AgentMetrics', 'AgentRegistration'])

if QUERY_MODELS_AVAILABLE:
    __all__.extend(['QueryLog', 'QueryMetrics'])

if AI_MODEL_MODELS_AVAILABLE:
    __all__.extend(['MLModelRegistry', 'ModelConfiguration'])

if CACHE_MODELS_AVAILABLE:
    __all__.extend(['AICacheEntry', 'AICacheStats'])

if AUTH_MODELS_AVAILABLE:
    __all__.extend(['APIKey', 'UserSession'])

if RAG_MODELS_AVAILABLE:
    __all__.extend(['Document', 'DocumentChunk', 'RAGQuery'])