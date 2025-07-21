"""
Testing Configuration
Settings optimized for running tests
"""

import os
from .base import BaseConfig

class TestingConfig(BaseConfig):
    """Testing environment configuration"""
    
    TESTING = True
    DEBUG = True
    
    # Use in-memory SQLite for fast tests
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
    
    # Disable CSRF for easier testing
    WTF_CSRF_ENABLED = False
    
    # Disable security features that interfere with testing
    SESSION_COOKIE_SECURE = False
    
    # Use simple cache for testing
    CACHE_TYPE = 'simple'
    CACHE_DEFAULT_TIMEOUT = 60
    
    # Faster settings for testing
    ML_INIT_TIMEOUT = 5
    ML_MODEL_CACHE_SIZE = 10
    
    # Disable rate limiting for tests
    RATE_LIMIT_ENABLED = False
    
    # Disable authentication for tests (unless specifically testing auth)
    AUTH_ENABLED = False
    
    # Test-specific upload settings
    UPLOAD_FOLDER = 'test_uploads'
    MAX_CONTENT_LENGTH = 1024 * 1024  # 1MB for tests
    
    # Disable external integrations for testing
    MAIL_SUPPRESS_SEND = True
    METRICS_ENABLED = False
    
    # Small RAG settings for testing
    RAG_CHUNK_SIZE = 100
    RAG_MAX_DOCUMENTS = 10
    
    # Logging for tests
    LOG_LEVEL = 'WARNING'  # Reduce noise in test output
    
    @staticmethod
    def init_app(app):
        """Initialize testing-specific settings"""
        # Ensure we're in testing mode
        app.config['TESTING'] = True
        
        # Suppress most logging during tests
        import logging
        logging.disable(logging.CRITICAL)
        
        app.logger.info("🧪 Testing mode activated")
