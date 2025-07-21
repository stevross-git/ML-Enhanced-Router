"""
Database Models Package
Centralized import and initialization of all database models
"""

from ..extensions import db
from .base import Base, TimestampMixin, SoftDeleteMixin, UUIDMixin, generate_id

# Import all model classes with error handling
try:
    from .query import QueryLog, QueryMetrics
    QUERY_MODELS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Query models not available: {e}")
    QUERY_MODELS_AVAILABLE = False

try:
    from .agent import AgentRegistration, AgentMetrics, Agent, AgentCapability, AgentSession
    AGENT_MODELS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Agent models not available: {e}")
    AGENT_MODELS_AVAILABLE = False

try:
    from .auth import User, APIKey, UserSession
    AUTH_MODELS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Auth models not available: {e}")
    AUTH_MODELS_AVAILABLE = False

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
    # Create all tables in app context
    with app.app_context():
        try:
            # Create all tables
            db.create_all()
            app.logger.info("✅ Database tables created successfully")
            
            # Run any initial data setup
            _create_initial_data()
            
        except Exception as e:
            app.logger.error(f"❌ Database initialization failed: {e}")
            # Don't raise - let the app start anyway for debugging
            app.logger.warning("⚠️ Continuing without database initialization...")

def _create_initial_data():
    """Create initial data if needed"""
    try:
        # Only create admin user if auth models are available
        if AUTH_MODELS_AVAILABLE:
            existing_admin = db.session.query(User).filter_by(email='admin@mlrouter.local').first()
            
            if not existing_admin:
                admin_user = User(
                    email='admin@mlrouter.local',
                    username='admin',
                    role='admin',
                    is_active=True
                )
                
                # Set password with error handling
                try:
                    if hasattr(admin_user, 'set_password'):
                        admin_user.set_password('admin123')
                    else:
                        from werkzeug.security import generate_password_hash
                        admin_user.password_hash = generate_password_hash('admin123')
                except Exception as e:
                    print(f"Warning: Could not set admin password: {e}")
                
                db.session.add(admin_user)
                print("✅ Created default admin user")
        
        # Create default agent if models are available
        if AGENT_MODELS_AVAILABLE:
            existing_agent = db.session.query(AgentRegistration).first()
            if not existing_agent:
                default_agent = AgentRegistration(
                    id='default-agent-001',
                    name='Default Agent',
                    description='Default conversational agent',
                    endpoint='internal://default',
                    categories=['conversational', 'general'],
                    capabilities={'max_tokens': 2000, 'temperature': 0.7},
                    is_active=True
                )
                db.session.add(default_agent)
                print("✅ Created default agent")
        
        # Commit all changes
        db.session.commit()
        
    except Exception as e:
        db.session.rollback()
        print(f"⚠️ Could not create initial data: {e}")
        # Don't raise - let the app continue

# Export available models dynamically
__all__ = ['db', 'init_db', 'Base', 'TimestampMixin', 'SoftDeleteMixin', 'UUIDMixin', 'generate_id']

if QUERY_MODELS_AVAILABLE:
    __all__.extend(['QueryLog', 'QueryMetrics'])
if AGENT_MODELS_AVAILABLE:
    __all__.extend(['AgentRegistration', 'AgentMetrics', 'Agent', 'AgentCapability', 'AgentSession'])
if AUTH_MODELS_AVAILABLE:
    __all__.extend(['User', 'APIKey', 'UserSession'])
if AI_MODEL_MODELS_AVAILABLE:
    __all__.extend(['MLModelRegistry', 'ModelConfiguration'])
if CACHE_MODELS_AVAILABLE:
    __all__.extend(['AICacheEntry', 'AICacheStats'])
if RAG_MODELS_AVAILABLE:
    __all__.extend(['Document', 'DocumentChunk', 'RAGQuery'])
