"""
Custom Decorators
Reusable decorators for authentication, validation, rate limiting, etc.
"""

import functools
import json
from datetime import datetime
from typing import List, Optional, Callable, Any

from flask import request, jsonify, session, current_app, g
from ..extensions import limiter
from ..utils.exceptions import ValidationError, AuthenticationError

def rate_limit(limit_string: str):
    """
    Rate limiting decorator
    
    Args:
        limit_string: Rate limit specification (e.g., "100 per minute")
    """
    def decorator(f):
        @functools.wraps(f)
        @limiter.limit(limit_string)
        def wrapper(*args, **kwargs):
            return f(*args, **kwargs)
        return wrapper
    return decorator

def require_auth(roles: Optional[List[str]] = None, optional: bool = False):
    """
    Authentication requirement decorator
    
    Args:
        roles: Required user roles (optional)
        optional: Whether authentication is optional
    """
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            try:
                # Check if authentication is enabled
                if not current_app.config.get('AUTH_ENABLED', True):
                    if optional:
                        return f(*args, **kwargs)
                    # If auth is disabled but required, deny access
                    return jsonify({'error': 'Authentication is required but disabled in configuration'}), 503
                
                # Get authentication service
                from ..services.auth_service import get_auth_service
                auth_service = get_auth_service()
                
                if not auth_service:
                    if optional:
                        return f(*args, **kwargs)
                    return jsonify({'error': 'Authentication service unavailable'}), 503
                
                # Check for API key authentication
                api_key = request.headers.get('X-API-Key')
                if api_key:
                    user = auth_service.authenticate_api_key(api_key)
                    if user:
                        g.current_user = user
                        return _check_roles_and_proceed(f, user, roles, *args, **kwargs)
                
                # Check for session authentication
                user_id = session.get('user_id')
                if user_id:
                    user = auth_service.get_user(user_id)
                    if user and user.is_active:
                        g.current_user = user
                        return _check_roles_and_proceed(f, user, roles, *args, **kwargs)
                
                # Check for JWT token
                auth_header = request.headers.get('Authorization')
                if auth_header and auth_header.startswith('Bearer '):
                    token = auth_header.split(' ')[1]
                    user = auth_service.authenticate_jwt(token)
                    if user:
                        g.current_user = user
                        return _check_roles_and_proceed(f, user, roles, *args, **kwargs)
                
                # No valid authentication found
                if optional:
                    g.current_user = None
                    return f(*args, **kwargs)
                
                return jsonify({'error': 'Authentication required'}), 401
                
            except AuthenticationError as e:
                return jsonify({'error': str(e)}), 401
            except Exception as e:
                current_app.logger.error(f"Authentication error: {e}")
                return jsonify({'error': 'Authentication failed'}), 500
                
        return wrapper
    return decorator

def _check_roles_and_proceed(f: Callable, user: Any, required_roles: Optional[List[str]], 
                           *args, **kwargs):
    """Check user roles and proceed with function execution"""
    if required_roles:
        if user.role not in required_roles and user.role != 'superuser':
            return jsonify({'error': 'Insufficient permissions'}), 403
    
    return f(*args, **kwargs)

def validate_json(required_fields: List[str] = None, optional_fields: List[str] = None):
    """
    JSON validation decorator
    
    Args:
        required_fields: List of required field names
        optional_fields: List of optional field names
    """
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            try:
                # Check content type
                if not request.is_json:
                    return jsonify({'error': 'Content-Type must be application/json'}), 400
                
                # Get JSON data
                data = request.get_json()
                if data is None:
                    return jsonify({'error': 'Invalid JSON data'}), 400
                
                # Validate required fields
                if required_fields:
                    missing_fields = []
                    for field in required_fields:
                        if field not in data:
                            missing_fields.append(field)
                    
                    if missing_fields:
                        return jsonify({
                            'error': 'Missing required fields',
                            'missing_fields': missing_fields
                        }), 400
                
                # Validate field types and values
                validation_errors = _validate_field_values(data, required_fields, optional_fields)
                if validation_errors:
                    return jsonify({
                        'error': 'Validation failed',
                        'validation_errors': validation_errors
                    }), 400
                
                return f(*args, **kwargs)
                
            except json.JSONDecodeError:
                return jsonify({'error': 'Invalid JSON format'}), 400
            except Exception as e:
                current_app.logger.error(f"JSON validation error: {e}")
                return jsonify({'error': 'Validation failed'}), 400
                
        return wrapper
    return decorator

def _validate_field_values(data: dict, required_fields: List[str] = None, 
                         optional_fields: List[str] = None) -> List[str]:
    """Validate field values and types"""
    errors = []
    
    # Define validation rules
    validation_rules = {
        'query': {'type': str, 'min_length': 1, 'max_length': 10000},
        'model_id': {'type': str, 'min_length': 1, 'max_length': 200},
        'parameters': {'type': dict},
        'session_id': {'type': str, 'min_length': 1, 'max_length': 100},
        'email': {'type': str, 'pattern': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'},
        'password': {'type': str, 'min_length': 8, 'max_length': 128},
        'username': {'type': str, 'min_length': 3, 'max_length': 50}
    }
    
    all_fields = (required_fields or []) + (optional_fields or [])
    
    for field in all_fields:
        if field in data:
            value = data[field]
            rules = validation_rules.get(field, {})
            
            # Type validation
            expected_type = rules.get('type')
            if expected_type and not isinstance(value, expected_type):
                errors.append(f"{field} must be of type {expected_type.__name__}")
                continue
            
            # String validations
            if isinstance(value, str):
                min_length = rules.get('min_length')
                if min_length and len(value) < min_length:
                    errors.append(f"{field} must be at least {min_length} characters")
                
                max_length = rules.get('max_length')
                if max_length and len(value) > max_length:
                    errors.append(f"{field} must be at most {max_length} characters")
                
                pattern = rules.get('pattern')
                if pattern:
                    import re
                    if not re.match(pattern, value):
                        errors.append(f"{field} format is invalid")
    
    return errors

def log_requests(include_response: bool = False):
    """
    Request logging decorator
    
    Args:
        include_response: Whether to log response data
    """
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            start_time = datetime.now()
            
            # Log request
            current_app.logger.info(
                f"Request: {request.method} {request.path} "
                f"from {request.remote_addr}"
            )
            
            try:
                # Execute function
                result = f(*args, **kwargs)
                
                # Log response
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                
                status_code = getattr(result, 'status_code', 200)
                current_app.logger.info(
                    f"Response: {status_code} in {duration:.3f}s"
                )
                
                if include_response and hasattr(result, 'data'):
                    current_app.logger.debug(f"Response data: {result.data}")
                
                return result
                
            except Exception as e:
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                
                current_app.logger.error(
                    f"Request failed in {duration:.3f}s: {str(e)}"
                )
                raise
                
        return wrapper
    return decorator

def cache_response(ttl: int = 300, key_func: Optional[Callable] = None):
    """
    Response caching decorator
    
    Args:
        ttl: Time to live in seconds
        key_func: Function to generate cache key
    """
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            try:
                from ..extensions import cache
                
                # Generate cache key
                if key_func:
                    cache_key = key_func(*args, **kwargs)
                else:
                    cache_key = f"{f.__name__}:{request.path}:{request.query_string.decode()}"
                
                # Try to get from cache
                cached_result = cache.get(cache_key)
                if cached_result is not None:
                    return cached_result
                
                # Execute function and cache result
                result = f(*args, **kwargs)
                cache.set(cache_key, result, timeout=ttl)
                
                return result
                
            except Exception as e:
                current_app.logger.error(f"Cache error: {e}")
                # Fall back to executing function without caching
                return f(*args, **kwargs)
                
        return wrapper
    return decorator

def retry(max_attempts: int = 3, delay: float = 1.0, exponential_backoff: bool = True):
    """
    Retry decorator for handling transient failures
    
    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        exponential_backoff: Whether to use exponential backoff
    """
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            import time
            
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_attempts):
                try:
                    return f(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if attempt < max_attempts - 1:  # Don't delay on last attempt
                        current_app.logger.warning(
                            f"Attempt {attempt + 1} failed: {str(e)}. "
                            f"Retrying in {current_delay}s..."
                        )
                        time.sleep(current_delay)
                        
                        if exponential_backoff:
                            current_delay *= 2
                    else:
                        current_app.logger.error(
                            f"All {max_attempts} attempts failed. Last error: {str(e)}"
                        )
            
            # Re-raise the last exception
            raise last_exception
            
        return wrapper
    return decorator

def require_content_type(*content_types):
    """
    Require specific content types
    
    Args:
        content_types: Allowed content types
    """
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            if request.content_type not in content_types:
                return jsonify({
                    'error': f'Content-Type must be one of: {", ".join(content_types)}'
                }), 400
            
            return f(*args, **kwargs)
            
        return wrapper
    return decorator

def measure_performance(metric_name: Optional[str] = None):
    """
    Performance measurement decorator
    
    Args:
        metric_name: Custom metric name (defaults to function name)
    """
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            start_time = datetime.now()
            
            try:
                result = f(*args, **kwargs)
                success = True
            except Exception as e:
                success = False
                raise
            finally:
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                
                name = metric_name or f.__name__
                
                # Log performance metric
                current_app.logger.info(
                    f"Performance: {name} completed in {duration:.3f}s "
                    f"(success: {success})"
                )
                
                # TODO: Send to metrics system (Prometheus, etc.)
                # metrics.histogram('function_duration', duration, tags={'function': name, 'success': success})
            
            return result
            
        return wrapper
    return decorator
