"""
Application Factory for ML Query Router
Implements clean Flask application factory pattern
"""

import logging
import threading
from flask import Flask

from .extensions import init_extensions
from .models import init_db
from .routes import register_blueprints
from .utils.exceptions import register_error_handlers
from config import get_config

def create_app(config_name='development'):
    """
    Application factory function
    
    Args:
        config_name: Configuration name (development, production, testing)
        
    Returns:
        Flask application instance
    """
    # Create Flask app
    app = Flask(__name__)
    
    # Load configuration
    config = get_config(config_name)
    app.config.from_object(config)
    
    # Setup logging
    _setup_logging(app)
    
    # Initialize extensions
    init_extensions(app)
    
    # Initialize database
    init_db(app)
    
    # Register blueprints
    register_blueprints(app)
    
    # Register error handlers
    register_error_handlers(app)
    
    # Initialize background services
    _init_background_services(app)
    
    app.logger.info(f"✅ ML Query Router initialized - {config_name} mode")
    
    return app

def _setup_logging(app):
    """Setup application logging"""
    log_level = app.config.get('LOG_LEVEL', 'INFO')
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s [%(levelname)8s] %(name)s - %(message)s'
    )
    
    # Set Flask logger level
    app.logger.setLevel(getattr(logging, log_level))

def _init_background_services(app):
    """Initialize background services in separate threads"""
    def init_services():
        """Initialize ML services in background"""
        with app.app_context():
            try:
                from .services.ml_router import get_ml_router
                from .services.ai_models import get_ai_model_manager
                from .services.cache_service import get_cache_manager
                from .services.rag_service import get_rag_service
                
                # Initialize services
                ml_router = get_ml_router()
                ai_manager = get_ai_model_manager()
                cache_manager = get_cache_manager()
                rag_service = get_rag_service()
                
                app.logger.info("🤖 Background services initialized")
                
            except Exception as e:
                app.logger.error(f"❌ Background services initialization failed: {e}")
    
    # Start background initialization
    init_thread = threading.Thread(target=init_services, daemon=True)
    init_thread.start()
