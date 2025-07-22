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
    # Import core blueprints (these should exist)
    from .main import main_bp
    from .api import api_bp
    from .auth import auth_bp
    from .models import models_bp
    from .cache import cache_bp
    from .rag import rag_bp
    from .config import config_bp
    from .graphql import graphql_bp
    
    # Register core blueprints with URL prefixes
    app.register_blueprint(main_bp)
    app.register_blueprint(api_bp, url_prefix='/api')
    app.register_blueprint(auth_bp, url_prefix='/auth')
    app.register_blueprint(models_bp, url_prefix='/api/models')
    app.register_blueprint(cache_bp, url_prefix='/api/cache')
    app.register_blueprint(rag_bp, url_prefix='/api/rag')
    app.register_blueprint(config_bp, url_prefix='/api/config')
    app.register_blueprint(graphql_bp, url_prefix='/graphql')
    
    # Import advanced blueprints (only if they exist)
    try:
        from .streaming import streaming_bp
        app.register_blueprint(streaming_bp)  # No prefix, handles /api/stream
        app.logger.info("✅ Streaming blueprint registered")
    except ImportError:
        app.logger.warning("⚠️ Streaming blueprint not found - skipping")
    
    try:
        from .chains import chains_bp
        app.register_blueprint(chains_bp, url_prefix='/api/chains')
        app.logger.info("✅ Chains blueprint registered")
    except ImportError:
        app.logger.warning("⚠️ Chains blueprint not found - skipping")
    
    try:
        from .multimodal import multimodal_bp
        app.register_blueprint(multimodal_bp, url_prefix='/api/multimodal')
        app.logger.info("✅ Multimodal blueprint registered")
    except ImportError:
        app.logger.warning("⚠️ Multimodal blueprint not found - skipping")
    
    try:
        from .email_advanced import email_advanced_bp
        app.register_blueprint(email_advanced_bp, url_prefix='/api/email')
        app.logger.info("✅ Email advanced blueprint registered")
    except ImportError:
        app.logger.warning("⚠️ Email advanced blueprint not found - skipping")
    
    try:
        from .evaluation import evaluation_bp
        app.register_blueprint(evaluation_bp, url_prefix='/api/evaluation')
        app.logger.info("✅ Evaluation blueprint registered")
    except ImportError:
        app.logger.warning("⚠️ Evaluation blueprint not found - skipping")
    
    try:
        from .peer_teaching import peer_teaching_bp
        app.register_blueprint(peer_teaching_bp, url_prefix='/api/peer-teaching')
        app.logger.info("✅ Peer teaching blueprint registered")
    except ImportError:
        app.logger.warning("⚠️ Peer teaching blueprint not found - skipping")
    
    try:
        from .personal_ai import personal_ai_bp
        app.register_blueprint(personal_ai_bp, url_prefix='/api/personal-ai')
        app.logger.info("✅ Personal AI blueprint registered")
    except ImportError:
        app.logger.warning("⚠️ Personal AI blueprint not found - skipping")
    
    try:
        from .shared_memory import shared_memory_bp
        app.register_blueprint(shared_memory_bp, url_prefix='/api/shared-memory')
        app.logger.info("✅ Shared memory blueprint registered")
    except ImportError:
        app.logger.warning("⚠️ Shared memory blueprint not found - skipping")
    
    app.logger.info(f"✅ Blueprint registration completed - {len(app.blueprints)} blueprints total")

# Export for use in application factory
__all__ = ['register_blueprints']