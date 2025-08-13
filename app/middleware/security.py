"""
Security Middleware
Implements security headers, CSP, and other security measures
"""

import hashlib
import secrets
from flask import g, request, current_app

class SecurityMiddleware:
    """Security middleware for adding security headers and protections"""
    
    def __init__(self, app=None):
        self.app = app
        if app is not None:
            self.init_app(app)
    
    def init_app(self, app):
        """Initialize security middleware with Flask app"""
        app.before_request(self.before_request)
        app.after_request(self.after_request)
    
    def before_request(self):
        """Generate nonce for CSP before each request"""
        g.csp_nonce = secrets.token_urlsafe(16)
    
    def after_request(self, response):
        """Add security headers to all responses"""
        # Content Security Policy
        csp_policy = self._build_csp_policy()
        response.headers['Content-Security-Policy'] = csp_policy
        
        # Security Headers
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
        
        # HTTPS enforcement
        if current_app.config.get('PREFERRED_URL_SCHEME') == 'https':
            response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
        
        # Prevent caching of sensitive responses
        if request.endpoint and any(sensitive in request.endpoint for sensitive in ['auth', 'api', 'admin']):
            response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
            response.headers['Pragma'] = 'no-cache'
            response.headers['Expires'] = '0'
        
        return response
    
    def _build_csp_policy(self):
        """Build Content Security Policy"""
        nonce = getattr(g, 'csp_nonce', '')
        
        # Base CSP policy - secure configuration without unsafe-eval
        policy = [
            "default-src 'self'",
            f"script-src 'self' 'nonce-{nonce}' https://cdnjs.cloudflare.com https://cdn.jsdelivr.net",
            f"style-src 'self' 'nonce-{nonce}' 'unsafe-inline' https://cdnjs.cloudflare.com https://fonts.googleapis.com",
            "font-src 'self' https://fonts.gstatic.com",
            "img-src 'self' data: https:",
            "connect-src 'self'",
            "frame-ancestors 'none'",
            "base-uri 'self'",
            "form-action 'self'",
            "object-src 'none'"
        ]
        
        return '; '.join(policy)

def get_csp_nonce():
    """Get CSP nonce for current request"""
    return getattr(g, 'csp_nonce', '')