"""
Database Models Package
Centralized import and initialization of all database models
"""

from ..extensions import db
from .base import Base, TimestampMixin, SoftDeleteMixin, UUIDMixin, generate_id

# Import all model classes
from .query import QueryLog, QueryMetrics
from .agent import AgentRegistration, AgentMetrics
from .auth import User, APIKey, UserSession
from .ai_model import MLModelRegistry, ModelConfiguration
from .cache import AICacheEntry, AICacheStats
from .rag import Document, DocumentChunk, RAGQuery

def init_db(app):
    """
    Initialize database with Flask app
    
    Args:
        app: Flask application instance
    """
    # Initialize SQLAlchemy with app
    db.init_app(app)
    
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
            raise

def _create_initial_data():
    """Create initial data if needed"""
    try:
        # Check if we need to create default admin user
        if not User.query.filter_by(email='admin@mlrouter.local').first():
            admin_user = User(
                email='admin@mlrouter.local',
                username='admin',
                role='admin',
                is_active=True
            )
            admin_user.set_password('admin123')  # Should be changed on first login
            db.session.add(admin_user)
        
        # Create default model configurations if none exist
        if not MLModelRegistry.query.first():
            default_models = [
                {
                    'name': 'OpenAI GPT-4',
                    'provider': 'openai',
                    'model_id': 'gpt-4',
                    'model_type': 'completion',
                    'is_active': False,
                    'categories': ['general', 'reasoning', 'coding']
                },
                {
                    'name': 'Anthropic Claude',
                    'provider': 'anthropic', 
                    'model_id': 'claude-3-sonnet',
                    'model_type': 'completion',
                    'is_active': False,
                    'categories': ['general', 'analysis', 'writing']
                }
            ]
            
            for model_data in default_models:
                model = MLModelRegistry(**model_data)
                db.session.add(model)
        
        db.session.commit()
        
    except Exception as e:
        db.session.rollback()
        raise e

# Export all models and database instance
__all__ = [
    # Database
    'db',
    'init_db',
    
    # Base classes
    'Base',
    'TimestampMixin',
    'SoftDeleteMixin', 
    'UUIDMixin',
    'generate_id',
    
    # Model classes
    'QueryLog',
    'QueryMetrics',
    'AgentRegistration',
    'AgentMetrics',
    'User',
    'APIKey',
    'UserSession',
    'MLModelRegistry',
    'ModelConfiguration',
    'AICacheEntry',
    'AICacheStats',
    'Document',
    'DocumentChunk',
    'RAGQuery'
]
