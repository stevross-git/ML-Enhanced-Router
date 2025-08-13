"""
Request Limits Middleware
Implements request size and content validation
"""

from flask import request, jsonify, current_app
from functools import wraps

class RequestLimitsMiddleware:
    """Middleware for enforcing request size and content limits"""
    
    def __init__(self, app=None):
        self.app = app
        if app is not None:
            self.init_app(app)
    
    def init_app(self, app):
        """Initialize request limits with Flask app"""
        app.before_request(self.before_request)
    
    def before_request(self):
        """Validate request before processing"""
        # Check content length
        if not self.check_content_length():
            return jsonify({
                'error': 'Request payload too large',
                'max_size': self.get_max_content_length()
            }), 413
        
        # Check JSON payload size for JSON requests
        if request.is_json:
            if not self.check_json_size():
                return jsonify({
                    'error': 'JSON payload too large',
                    'max_size': current_app.config.get('MAX_JSON_PAYLOAD_SIZE', 1024*1024)
                }), 413
    
    def check_content_length(self):
        """Check if content length is within limits"""
        content_length = request.content_length
        if content_length is None:
            return True
        
        max_length = self.get_max_content_length()
        return content_length <= max_length
    
    def check_json_size(self):
        """Check JSON payload size"""
        if not request.is_json:
            return True
        
        try:
            data = request.get_data()
            max_json_size = current_app.config.get('MAX_JSON_PAYLOAD_SIZE', 1024*1024)
            return len(data) <= max_json_size
        except Exception:
            return False
    
    def get_max_content_length(self):
        """Get maximum content length based on content type"""
        if request.is_json:
            return current_app.config.get('MAX_JSON_PAYLOAD_SIZE', 1024*1024)
        elif request.content_type and 'multipart/form-data' in request.content_type:
            return current_app.config.get('MAX_FORM_DATA_SIZE', 5*1024*1024)
        else:
            return current_app.config.get('MAX_CONTENT_LENGTH', 16*1024*1024)

def validate_query_length():
    """Decorator to validate query length in request data"""
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            if request.is_json:
                data = request.get_json()
                if data and 'query' in data:
                    query = data['query']
                    max_length = current_app.config.get('MAX_QUERY_LENGTH', 10000)
                    
                    if len(query) > max_length:
                        return jsonify({
                            'error': 'Query too long',
                            'max_length': max_length,
                            'current_length': len(query)
                        }), 413
            
            return f(*args, **kwargs)
        return wrapper
    return decorator

def limit_request_size(max_size_mb=1):
    """Decorator to limit request size for specific endpoints"""
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            max_bytes = max_size_mb * 1024 * 1024
            content_length = request.content_length
            
            if content_length and content_length > max_bytes:
                return jsonify({
                    'error': 'Request too large',
                    'max_size_mb': max_size_mb,
                    'received_size_mb': round(content_length / (1024*1024), 2)
                }), 413
            
            return f(*args, **kwargs)
        return wrapper
    return decorator