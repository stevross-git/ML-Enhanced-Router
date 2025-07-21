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
                    'name': 'OpenAI GPT-3.5 Turbo',
                    'provider': 'openai',
                    'model_id': 'gpt-3.5-turbo',
                    'model_type': 'completion',
                    'is_active': True,
                    'categories': ['general', 'chat']
                },
                {
                    'name': 'Claude 3 Sonnet',
                    'provider': 'anthropic',
                    'model_id': 'claude-3-sonnet-20240229',
                    'model_type': 'completion',
                    'is_active': False,
                    'categories': ['general', 'reasoning', 'analysis']
                },
                {
                    'name': 'Gemini Pro',
                    'provider': 'google',
                    'model_id': 'gemini-pro',
                    'model_type': 'completion',
                    'is_active': False,
                    'categories': ['general', 'multimodal']
                }
            ]
            
            for model_data in default_models:
                model = MLModelRegistry(
                    id=model_data['model_id'],
                    name=model_data['name'],
                    model_type=model_data['model_type'],
                    categories=model_data['categories'],
                    config={
                        'provider': model_data['provider'],
                        'model_id': model_data['model_id'],
                        'max_tokens': 4000,
                        'temperature': 0.7
                    },
                    is_active=model_data['is_active']
                )
                db.session.add(model)
        
        # Create default agent registrations
        if not AgentRegistration.query.first():
            default_agents = [
                {
                    'id': 'general_assistant',
                    'name': 'General Assistant',
                    'description': 'General purpose AI assistant for various tasks',
                    'endpoint': 'internal://general',
                    'categories': ['general', 'chat', 'help', 'questions']
                },
                {
                    'id': 'coding_assistant',
                    'name': 'Coding Assistant', 
                    'description': 'Specialized assistant for programming and code-related tasks',
                    'endpoint': 'internal://coding',
                    'categories': ['coding', 'programming', 'debug', 'development']
                },
                {
                    'id': 'analysis_assistant',
                    'name': 'Analysis Assistant',
                    'description': 'Data analysis and research specialist',
                    'endpoint': 'internal://analysis',
                    'categories': ['analysis', 'research', 'data', 'statistics']
                },
                {
                    'id': 'writing_assistant',
                    'name': 'Writing Assistant',
                    'description': 'Creative and technical writing specialist',
                    'endpoint': 'internal://writing',
                    'categories': ['writing', 'creative', 'content', 'documentation']
                }
            ]
            
            for agent_data in default_agents:
                agent = AgentRegistration(
                    id=agent_data['id'],
                    name=agent_data['name'],
                    description=agent_data['description'],
                    endpoint=agent_data['endpoint'],
                    categories=agent_data['categories'],
                    capabilities={
                        'max_tokens': 4000,
                        'temperature': 0.7,
                        'supports_streaming': True
                    },
                    is_active=True
                )
                db.session.add(agent)
        
        # Commit all initial data
        db.session.commit()
        
    except Exception as e:
        db.session.rollback()
        raise e

# Export all models and utilities
__all__ = [
    # Base classes
    'Base',
    'TimestampMixin',
    'SoftDeleteMixin', 
    'UUIDMixin',
    'generate_id',
    
    # Query models
    'QueryLog',
    'QueryMetrics',
    
    # Agent models
    'AgentRegistration',
    'AgentMetrics',
    
    # Auth models
    'User',
    'APIKey',
    'UserSession',
    
    # AI model registry
    'MLModelRegistry',
    'ModelConfiguration',
    
    # Cache models
    'AICacheEntry',
    'AICacheStats',
    
    # RAG models
    'Document',
    'DocumentChunk', 
    'RAGQuery',
    
    # Database initialization
    'init_db'
]