"""
Production Configuration
Settings optimized for production deployment
"""

import os
from .base import BaseConfig

class ProductionConfig(BaseConfig):
    """Production environment configuration"""
    
    DEBUG = False
    TESTING = False
    
    # Database - Use PostgreSQL for production
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
        os.environ.get('POSTGRESQL_URL') or \
        'postgresql://mlrouter:password@localhost:5432/mlrouter'
    
    # Enhanced security settings
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Strict'
    WTF_CSRF_ENABLED = True
    
    # Production cache settings - Use Redis
    CACHE_TYPE = 'redis'
    CACHE_REDIS_URL = os.environ.get('REDIS_URL') or 'redis://localhost:6379/0'
    CACHE_DEFAULT_TIMEOUT = 3600
    
    # Logging - More conservative in production
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    
    # Production ML settings
    ML_MODEL_CACHE_SIZE = 1000
    ML_INIT_TIMEOUT = 120
    
    # Strict rate limiting for production
    RATE_LIMIT_PER_MINUTE = int(os.environ.get('RATE_LIMIT_PER_MINUTE', 100))
    RATE_LIMIT_BURST = int(os.environ.get('RATE_LIMIT_BURST', 200))
    
    # Production file upload settings
    UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', '/app/uploads')
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB for production
    
    # Production features
    SWAGGER_ENABLED = False  # Disable Swagger in production
    PROFILER_ENABLED = False
    
    # Enhanced RAG settings for production
    RAG_CHUNK_SIZE = 1000
    RAG_MAX_DOCUMENTS = 10000
    
    # Email settings for production
    MAIL_SUPPRESS_SEND = False
    MAIL_DEBUG = 0
    
    # Performance settings
    SQLALCHEMY_ENGINE_OPTIONS = {
        'pool_size': 20,
        'pool_recycle': 3600,
        'pool_pre_ping': True,
        'max_overflow': 40
    }
    
    # Security headers
    SECURITY_HEADERS = {
        'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
        'X-Content-Type-Options': 'nosniff',
        'X-Frame-Options': 'DENY',
        'X-XSS-Protection': '1; mode=block',
        'Content-Security-Policy': "default-src 'self'"
    }
    
    @staticmethod
    def init_app(app):
        """Initialize production-specific settings"""
        # Production logging to file
        import logging
        from logging.handlers import RotatingFileHandler
        
        if not app.debug:
            file_handler = RotatingFileHandler(
                'logs/mlrouter.log',
                maxBytes=10240000,
                backupCount=10
            )
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
            ))
            file_handler.setLevel(logging.INFO)
            app.logger.addHandler(file_handler)
            
            app.logger.setLevel(logging.INFO)
            app.logger.info('🚀 ML Router production startup')
        
        # Add security headers middleware
        @app.after_request
        def add_security_headers(response):
            for header, value in ProductionConfig.SECURITY_HEADERS.items():
                response.headers[header] = value
            return response
