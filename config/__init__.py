"""
Configuration module for ML Query Router
Provides environment-specific configuration classes
"""

import os
from .base import BaseConfig
from .development import DevelopmentConfig
from .production import ProductionConfig
from .testing import TestingConfig

# Configuration mapping
config_map = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}

def get_config(config_name=None):
    """
    Get configuration class for specified environment
    
    Args:
        config_name: Environment name (development, production, testing)
        
    Returns:
        Configuration class
    """
    if config_name is None:
        config_name = os.environ.get('FLASK_ENV', 'default')
    
    return config_map.get(config_name, DevelopmentConfig)

# Export configuration classes
__all__ = [
    'BaseConfig',
    'DevelopmentConfig', 
    'ProductionConfig',
    'TestingConfig',
    'get_config'
]
