"""
Flask Extensions
Centralized initialization of Flask extensions
"""

from flask_sqlalchemy import SQLAlchemy
from werkzeug.middleware.proxy_fix import ProxyFix
from flask_migrate import Migrate
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_cors import CORS
import redis

# Initialize extensions (without app context)
db = SQLAlchemy()
migrate = Migrate()
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["1000 per hour", "100 per minute"]
)
cors = CORS()

# Global variables for other services
redis_client = None

def init_extensions(app):
    """
    Initialize Flask extensions with app context
    
    Args:
        app: Flask application instance
    """
    # Initialize SQLAlchemy
    db.init_app(app)
    
    # Initialize migrations
    migrate.init_app(app, db)
    
    # Initialize rate limiting
    _init_rate_limiting(app)
    
    # Initialize CORS
    cors.init_app(app, resources={
        r"/api/*": {"origins": "*"},
        r"/graphql/*": {"origins": "*"}
    })
    
    # Initialize Redis (if available)
    _init_redis(app)
    
    # Setup proxy fix for production
    if app.config.get('PROXY_FIX_ENABLED', False):
        app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)
    
    app.logger.info("✅ Extensions initialized successfully")

def _init_rate_limiting(app):
    """Initialize rate limiting"""
    try:
        # Try to use Redis for distributed rate limiting
        redis_url = app.config.get('REDIS_URL')
        if redis_url:
            limiter.init_app(app, storage_uri=redis_url)
        else:
            # Fallback to in-memory rate limiting
            limiter.init_app(app)
        
        app.logger.info("✅ Rate limiting initialized")
        
    except Exception as e:
        app.logger.warning(f"⚠️ Rate limiting initialization failed: {e}")
        app.logger.warning("ℹ️ Continuing with in-memory rate limiting")
        # Try basic initialization
        try:
            limiter.init_app(app)
        except Exception as e2:
            app.logger.warning(f"⚠️ Basic rate limiting failed: {e2}")
            # Continue without rate limiting

def _init_redis(app):
    """Initialize Redis connection"""
    global redis_client
    
    try:
        redis_url = app.config.get('REDIS_URL')
        if redis_url:
            redis_client = redis.from_url(redis_url)
            # Test connection
            redis_client.ping()
            app.logger.info("✅ Redis connection established")
        else:
            app.logger.info("ℹ️ Redis not configured - using in-memory caching")
            
    except Exception as e:
        app.logger.warning(f"⚠️ Redis connection failed: {e}")
        redis_client = None

def get_redis():
    """Get Redis client instance"""
    return redis_client