"""
Routes Package
Centralized registration of all application blueprints
"""

from flask import Flask

def register_blueprints(app: Flask):
    """
    Register all blueprints with the Flask application
    
    Args:
        app: Flask application instance
    """
    # Import blueprints
    from .main import main_bp
    from .api import api_bp
    from .auth import auth_bp
    from .models import models_bp
    from .cache import cache_bp
    from .rag import rag_bp
    from .config import config_bp
    from .graphql import graphql_bp
    
    # Register blueprints with URL prefixes
    app.register_blueprint(main_bp)
    app.register_blueprint(api_bp, url_prefix='/api')
    app.register_blueprint(auth_bp, url_prefix='/auth')
    app.register_blueprint(models_bp, url_prefix='/api/models')
    app.register_blueprint(cache_bp, url_prefix='/api/cache')
    app.register_blueprint(rag_bp, url_prefix='/api/rag')
    app.register_blueprint(config_bp, url_prefix='/api/config')
    app.register_blueprint(graphql_bp, url_prefix='/graphql')
    
    app.logger.info("✅ All blueprints registered successfully")

# Export for use in application factory
__all__ = ['register_blueprints']
