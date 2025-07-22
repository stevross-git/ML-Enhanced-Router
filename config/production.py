"""
Production Configuration
Optimized settings for production deployment with Docker
"""

import os
from .base import Config

class ProductionConfig(Config):
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
    SECRET_KEY = os.environ.get('SECRET_KEY', 'your-super-secret-key-change-in-production')
    SESSION_SECRET = os.environ.get('SESSION_SECRET', 'your-session-secret-key-change-in-production')
    
    # SSL and Security Headers
    PREFERRED_URL_SCHEME = 'https'
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    
    # Database Performance
    SQLALCHEMY_ENGINE_OPTIONS = {
        'pool_size': 20,
        'pool_recycle': 3600,
        'pool_pre_ping': True,
        'max_overflow': 30,
        'pool_timeout': 30,
    }
    
    # Caching
    CACHE_TYPE = 'redis'
    CACHE_REDIS_URL = REDIS_URL
    CACHE_DEFAULT_TIMEOUT = 3600
    
    # Rate Limiting
    RATELIMIT_STORAGE_URL = REDIS_URL
    RATELIMIT_ENABLED = True
    
    # ML Router Configuration
    ML_ROUTER_ENABLED = True
    ML_ROUTER_MAX_TOKENS = 4096
    ML_ROUTER_TEMPERATURE = 0.7
    ML_ROUTER_TIMEOUT = 30
    
    # Authentication
    AUTH_ENABLED = True
    JWT_SECRET_KEY = SECRET_KEY
    JWT_ACCESS_TOKEN_EXPIRES = 3600  # 1 hour
    JWT_REFRESH_TOKEN_EXPIRES = 86400  # 24 hours
    
    # RAG Configuration
    RAG_ENABLED = True
    RAG_CHUNK_SIZE = 1000
    RAG_CHUNK_OVERLAP = 200
    RAG_MAX_CHUNKS = 10
    
    # Email Configuration
    MAIL_SERVER = os.environ.get('MAIL_SERVER', 'localhost')
    MAIL_PORT = int(os.environ.get('MAIL_PORT', 587))
    MAIL_USE_TLS = os.environ.get('MAIL_USE_TLS', 'true').lower() == 'true'
    MAIL_USERNAME = os.environ.get('MAIL_USERNAME')
    MAIL_PASSWORD = os.environ.get('MAIL_PASSWORD')
    
    # API Keys
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
    ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY')
    GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
    COHERE_API_KEY = os.environ.get('COHERE_API_KEY')
    HUGGINGFACE_API_KEY = os.environ.get('HUGGINGFACE_API_KEY')
    
    # Logging
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_TO_STDOUT = True
    
    # Performance
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB max file upload
    REQUEST_TIMEOUT = 300  # 5 minutes
    MAX_CONCURRENT_REQUESTS = 100
    
    # Monitoring
    METRICS_ENABLED = True
    HEALTH_CHECK_ENABLED = True
    
    # File Upload
    UPLOAD_FOLDER = '/app/instance/uploads'
    MAX_UPLOAD_SIZE = 25 * 1024 * 1024  # 25MB
    
    # Docker specific settings
    DOCKER_ENV = True
    USE_DOCKER_NETWORKING = True
    
    @staticmethod
    def init_app(app):
        """Initialize production-specific configurations"""
        Config.init_app(app)
        
        # Create necessary directories
        import os
        os.makedirs('/app/instance', exist_ok=True)
        os.makedirs('/app/logs', exist_ok=True)
        os.makedirs('/app/instance/uploads', exist_ok=True)
        
        # Set up logging for production
        import logging
        from logging.handlers import RotatingFileHandler
        
        if not app.debug and not app.testing:
            # File logging
            file_handler = RotatingFileHandler(
                '/app/logs/ml_router.log',
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=10
            )
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
            ))
            file_handler.setLevel(logging.INFO)
            app.logger.addHandler(file_handler)
            
            app.logger.setLevel(logging.INFO)
            app.logger.info('ML Router production startup')