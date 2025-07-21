"""
Development Configuration
Settings optimized for local development
"""

import os
from .base import BaseConfig

class DevelopmentConfig(BaseConfig):
    """Development environment configuration"""
    
    DEBUG = True
    TESTING = False
    
    # Database - Use SQLite for development
    SQLALCHEMY_DATABASE_URI = os.environ.get('DEV_DATABASE_URL') or \
        'sqlite:///ml_router_dev.db'
    
    # Disable CSRF for easier development
    WTF_CSRF_ENABLED = False
    
    # Session settings for development
    SESSION_COOKIE_SECURE = False  # Allow HTTP in development
    
    # Cache settings - Use in-memory cache for development
    CACHE_TYPE = 'simple'
    CACHE_DEFAULT_TIMEOUT = 300
    
    # Logging
    LOG_LEVEL = 'DEBUG'
    
    # Development-specific ML settings
    ML_MODEL_CACHE_SIZE = 100  # Smaller cache for development
    ML_INIT_TIMEOUT = 30  # Shorter timeout for development
    
    # API Rate limiting - More lenient for development
    RATE_LIMIT_PER_MINUTE = 1000
    RATE_LIMIT_BURST = 2000
    
    # File upload settings
    UPLOAD_FOLDER = 'dev_uploads'
    
    # Development features
    SWAGGER_ENABLED = True
    PROFILER_ENABLED = False  # Can be enabled for performance debugging
    
    # RAG settings for development
    RAG_CHUNK_SIZE = 500  # Smaller chunks for faster processing
    RAG_MAX_DOCUMENTS = 100
    
    # Email settings - Use console backend for development
    MAIL_SUPPRESS_SEND = True  # Don't actually send emails
    MAIL_DEBUG = 1
    
    @staticmethod
    def init_app(app):
        """Initialize development-specific settings"""
        # Enable detailed error pages
        app.config['PROPAGATE_EXCEPTIONS'] = True
        
        # Development logging to console
        import logging
        logging.basicConfig(level=logging.DEBUG)
        
        # Print useful development info
        app.logger.info("🔧 Development mode activated")
        app.logger.info(f"📁 Database: {app.config['SQLALCHEMY_DATABASE_URI']}")
        app.logger.info(f"📂 Upload folder: {app.config['UPLOAD_FOLDER']}")
