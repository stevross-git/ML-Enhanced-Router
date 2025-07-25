"""
Base Configuration for ML Query Router
Contains common configuration settings across all environments
"""

import os
from datetime import timedelta

class BaseConfig:
    """Base configuration class with common settings"""
    
    # Flask Core Settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    
    # Database Configuration
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = {
        'pool_recycle': 300,
        'pool_pre_ping': True,
        'pool_size': 10,
        'max_overflow': 20
    }
    
    # Session Configuration
    PERMANENT_SESSION_LIFETIME = timedelta(hours=24)
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    
    # ML Router Settings
    ML_ROUTER_ENABLED = os.environ.get('ML_ROUTER_ENABLED', 'true').lower() == 'true'
    ROUTER_TIMEOUT = int(os.environ.get('ROUTER_TIMEOUT', 30))
    ROUTER_MAX_RETRIES = int(os.environ.get('ROUTER_MAX_RETRIES', 3))
    
    # Cache Configuration
    CACHE_ENABLED = os.environ.get('CACHE_ENABLED', 'true').lower() == 'true'
    CACHE_TTL = int(os.environ.get('CACHE_TTL', 3600))
    CACHE_MAX_SIZE = int(os.environ.get('CACHE_MAX_SIZE', 1000))
    
    # Rate Limiting
    RATE_LIMIT_ENABLED = os.environ.get('RATE_LIMIT_ENABLED', 'true').lower() == 'true'
    RATE_LIMIT_PER_MINUTE = int(os.environ.get('RATE_LIMIT_PER_MINUTE', 100))
    RATE_LIMIT_BURST = int(os.environ.get('RATE_LIMIT_BURST', 200))
    
    # Authentication
    AUTH_ENABLED = os.environ.get('AUTH_ENABLED', 'false').lower() == 'true'
    JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY') or SECRET_KEY
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=1)
    JWT_REFRESH_TOKEN_EXPIRES = timedelta(days=30)
    
    # API Keys Storage
    API_KEYS_ENCRYPTION_KEY = os.environ.get('API_KEYS_ENCRYPTION_KEY')
    
    # RAG System Settings
    RAG_ENABLED = os.environ.get('RAG_ENABLED', 'true').lower() == 'true'
    RAG_CHUNK_SIZE = int(os.environ.get('RAG_CHUNK_SIZE', 1000))
    RAG_CHUNK_OVERLAP = int(os.environ.get('RAG_CHUNK_OVERLAP', 100))
    RAG_MAX_DOCUMENTS = int(os.environ.get('RAG_MAX_DOCUMENTS', 1000))
    
    # Multimodal AI Settings
    MULTIMODAL_ENABLED = os.environ.get('MULTIMODAL_ENABLED', 'false').lower() == 'true'
    
    # File Upload Settings
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', 'uploads')
    ALLOWED_EXTENSIONS = {
        'txt', 'pdf', 'doc', 'docx', 'md', 'html', 'json', 'csv', 'xlsx'
    }
    
    # Logging Configuration
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_FORMAT = '%(asctime)s [%(levelname)8s] %(name)s - %(message)s'
    
    # Performance Settings
    MAX_WORKERS = int(os.environ.get('MAX_WORKERS', 4))
    ASYNC_TIMEOUT = int(os.environ.get('ASYNC_TIMEOUT', 60))
    
    # Monitoring
    METRICS_ENABLED = os.environ.get('METRICS_ENABLED', 'true').lower() == 'true'
    HEALTH_CHECK_ENABLED = True
    
    # Email Configuration (if enabled)
    MAIL_SERVER = os.environ.get('MAIL_SERVER')
    MAIL_PORT = int(os.environ.get('MAIL_PORT', 587))
    MAIL_USE_TLS = os.environ.get('MAIL_USE_TLS', 'true').lower() == 'true'
    MAIL_USERNAME = os.environ.get('MAIL_USERNAME')
    MAIL_PASSWORD = os.environ.get('MAIL_PASSWORD')
    
    @staticmethod
    def init_app(app):
        """Initialize application with this configuration"""
        pass
