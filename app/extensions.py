"""
Flask Extensions Initialization
Centralized initialization of all Flask extensions
"""

from flask_sqlalchemy import SQLAlchemy
from flask_caching import Cache
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_cors import CORS
from flask_compress import Compress

# Initialize extensions without app context
db = SQLAlchemy()
cache = Cache()
limiter = Limiter(key_func=get_remote_address)
cors = CORS()
compress = Compress()

def init_extensions(app):
    """
    Initialize all Flask extensions with the app
    
    Args:
        app: Flask application instance
    """
    # Database
    db.init_app(app)
    
    # Caching
    cache.init_app(app)
    
    # Rate limiting
    if app.config.get('RATE_LIMIT_ENABLED', True):
        limiter.init_app(app)
    
    # CORS - Configure based on environment
    if app.config.get('DEBUG', False):
        # Allow all origins in development
        cors.init_app(app, resources={
            r"/api/*": {"origins": "*"},
            r"/graphql": {"origins": "*"}
        })
    else:
        # Restrict origins in production
        allowed_origins = app.config.get('ALLOWED_ORIGINS', ['http://localhost:3000'])
        cors.init_app(app, resources={
            r"/api/*": {"origins": allowed_origins},
            r"/graphql": {"origins": allowed_origins}
        })
    
    # Compression
    compress.init_app(app)
    
    app.logger.info("✅ Flask extensions initialized")

# Export extensions for use in other modules
__all__ = ['db', 'cache', 'limiter', 'cors', 'compress', 'init_extensions']
