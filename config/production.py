"""
Production Configuration
Optimized settings for production deployment with Docker
"""

import os
from .base import BaseConfig

class ProductionConfig(BaseConfig):
    """Production configuration with Docker database credentials"""
    
    # Database Configuration - matches docker-compose.yml
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        'DATABASE_URL',
        'postgresql://ml_router_user:ml_router_password@db:5432/ml_router_db'
    )
    
    # Alternative format for SQLAlchemy
    if not SQLALCHEMY_DATABASE_URI:
        DB_USER = os.environ.get('DB_USER', 'ml_router_user')
        DB_PASSWORD = os.environ.get('DB_PASSWORD', 'ml_router_password')
        DB_HOST = os.environ.get('DB_HOST', 'db')
        DB_PORT = os.environ.get('DB_PORT', '5432')
        DB_NAME = os.environ.get('DB_NAME', 'ml_router_db')
        SQLALCHEMY_DATABASE_URI = f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}'
    
    # Redis Configuration - matches docker-compose.yml
    REDIS_URL = os.environ.get('REDIS_URL', 'redis://redis:6379/0')
    CACHE_REDIS_URL = os.environ.get('CACHE_REDIS_URL', 'redis://redis:6379/0')
    
    # Security - MUST be set in production
    SECRET_KEY = os.environ.get('SECRET_KEY')
    if not SECRET_KEY:
        raise RuntimeError("SECRET_KEY environment variable must be set in production")
    
    SESSION_SECRET = os.environ.get('SESSION_SECRET')
    if not SESSION_SECRET:
        raise RuntimeError("SESSION_SECRET environment variable must be set in production")
    
    # SSL and Security Headers
    PREFERRED_URL_SCHEME = 'https'
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    
    # Database Performance
    SQLALCHEMY_ENGINE_OPTIONS = {
        **BaseConfig.SQLALCHEMY_ENGINE_OPTIONS,  # Inherit base engine options
        'pool_size': 20,
        'max_overflow': 30,
        'pool_recycle': 1800,  # 30 minutes
        'pool_pre_ping': True,
        'echo': False  # Disable SQL logging in production
    }
    
    # Production optimizations
    SQLALCHEMY_ECHO = False
    SQLALCHEMY_RECORD_QUERIES = False
    
    # Cache Configuration
    CACHE_TYPE = 'redis'
    CACHE_DEFAULT_TIMEOUT = 300
    CACHE_KEY_PREFIX = 'ml_router:'
    
    # ML Production Settings
    ML_MODEL_CACHE_SIZE = 1000
    ML_INIT_TIMEOUT = 120
    ML_INFERENCE_TIMEOUT = 60
    
    # Production logging
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_FORMAT = '%(asctime)s %(levelname)s [%(name)s] [%(filename)s:%(lineno)d] - %(message)s'
    
    # Production rate limiting
    RATE_LIMIT_PER_MINUTE = 60
    RATE_LIMIT_BURST = 100
    
    # File upload settings
    UPLOAD_FOLDER = '/app/uploads'
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB
    
    # RAG settings for production
    RAG_CHUNK_SIZE = 1000
    RAG_MAX_DOCUMENTS = 10000
    
    # Monitoring
    METRICS_ENABLED = True
    PROFILER_ENABLED = False
    
    # Email settings
    MAIL_SERVER = os.environ.get('MAIL_SERVER')
    MAIL_PORT = int(os.environ.get('MAIL_PORT', '587'))
    MAIL_USE_TLS = os.environ.get('MAIL_USE_TLS', 'true').lower() == 'true'
    MAIL_USERNAME = os.environ.get('MAIL_USERNAME')
    MAIL_PASSWORD = os.environ.get('MAIL_PASSWORD')
    
    # Gunicorn settings
    WORKER_CLASS = 'sync'
    WORKERS = os.environ.get('WEB_CONCURRENCY', '4')
    WORKER_CONNECTIONS = 1000
    MAX_REQUESTS = 1000
    MAX_REQUESTS_JITTER = 50
    TIMEOUT = 30
    KEEPALIVE = 2
    
    @staticmethod
    def init_app(app):
        """Initialize production-specific settings"""
        # Configure production logging
        import logging
        from logging.handlers import RotatingFileHandler
        
        if not app.debug and not app.testing:
            # Create logs directory
            import os
            if not os.path.exists('logs'):
                os.mkdir('logs')
            
            # Setup rotating file handler
            file_handler = RotatingFileHandler(
                'logs/ml_router.log', 
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
        
        # Production-specific configurations
        app.logger.info("🏭 Production mode activated")
        app.logger.info(f"📊 Workers: {app.config.get('WORKERS')}")
        app.logger.info(f"🔒 SSL: {app.config.get('SESSION_COOKIE_SECURE')}")