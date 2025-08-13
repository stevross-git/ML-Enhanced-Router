"""
CSRF Protection Middleware
Implements Cross-Site Request Forgery protection
"""

import secrets
import hashlib
import hmac
from functools import wraps
from flask import request, jsonify, session, current_app, g

class CSRFProtection:
    """CSRF protection middleware"""
    
    def __init__(self, app=None):
        self.app = app
        if app is not None:
            self.init_app(app)
    
    def init_app(self, app):
        """Initialize CSRF protection with Flask app"""
        app.before_request(self.before_request)
        
        # Add CSRF token generation route
        @app.route('/api/csrf-token', methods=['GET'])
        def get_csrf_token():
            """Get CSRF token for client-side forms"""
            token = self.generate_csrf_token()
            return jsonify({'csrf_token': token})
    
    def before_request(self):
        """Check CSRF token on state-changing requests"""
        # Skip CSRF check for safe methods
        if request.method in ['GET', 'HEAD', 'OPTIONS']:
            return
            
        # Skip CSRF check if disabled in development
        if current_app.config.get('CSRF_ENABLED', True) is False:
            if current_app.config.get('FLASK_ENV') != 'production':
                return
        
        # Skip CSRF check for API key authentication
        if request.headers.get('X-API-Key'):
            return
            
        # Check CSRF token for authenticated requests
        if hasattr(g, 'current_user') and g.current_user:
            if not self.validate_csrf_token():
                return jsonify({
                    'error': 'CSRF token missing or invalid',
                    'code': 'CSRF_TOKEN_INVALID'
                }), 403
    
    def generate_csrf_token(self):
        """Generate CSRF token"""
        if 'csrf_token' not in session:
            session['csrf_token'] = secrets.token_urlsafe(32)
        
        # Create HMAC of session token with secret
        secret = current_app.config.get('SECRET_KEY', '').encode('utf-8')
        token = session['csrf_token'].encode('utf-8')
        signature = hmac.new(secret, token, hashlib.sha256).hexdigest()
        
        return f"{session['csrf_token']}.{signature}"
    
    def validate_csrf_token(self):
        """Validate CSRF token from request"""
        # Get token from header or form data
        token = request.headers.get('X-CSRF-Token') or request.form.get('csrf_token')
        
        if not token:
            return False
        
        # Split token and signature
        try:
            session_token, signature = token.split('.', 1)
        except ValueError:
            return False
        
        # Check if session token matches
        if session.get('csrf_token') != session_token:
            return False
        
        # Validate signature
        secret = current_app.config.get('SECRET_KEY', '').encode('utf-8')
        expected_sig = hmac.new(secret, session_token.encode('utf-8'), hashlib.sha256).hexdigest()
        
        return hmac.compare_digest(signature, expected_sig)

def csrf_protect():
    """Decorator to enforce CSRF protection on specific routes"""
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            csrf = CSRFProtection()
            
            if request.method not in ['GET', 'HEAD', 'OPTIONS']:
                if not csrf.validate_csrf_token():
                    return jsonify({
                        'error': 'CSRF token missing or invalid',
                        'code': 'CSRF_TOKEN_INVALID'
                    }), 403
            
            return f(*args, **kwargs)
        return wrapper
    return decorator

def get_csrf_token():
    """Get CSRF token for current session"""
    csrf = CSRFProtection()
    return csrf.generate_csrf_token()