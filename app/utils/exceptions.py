"""
Custom Exception Classes
Application-specific exceptions for better error handling
"""

from flask import jsonify, current_app
from datetime import datetime

class BaseMLRouterException(Exception):
    """Base exception for ML Router application"""
    
    def __init__(self, message: str, error_code: str = None, details: dict = None):
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        self.timestamp = datetime.now().isoformat()
        super().__init__(self.message)
    
    def to_dict(self):
        """Convert exception to dictionary representation"""
        return {
            'error': self.error_code,
            'message': self.message,
            'details': self.details,
            'timestamp': self.timestamp
        }

class ValidationError(BaseMLRouterException):
    """Raised when input validation fails"""
    pass

class AuthenticationError(BaseMLRouterException):
    """Raised when authentication fails"""
    pass

class AuthorizationError(BaseMLRouterException):
    """Raised when user lacks required permissions"""
    pass

class ServiceError(BaseMLRouterException):
    """Raised when a service operation fails"""
    pass

class ModelError(BaseMLRouterException):
    """Raised when AI model operations fail"""
    pass

class CacheError(BaseMLRouterException):
    """Raised when cache operations fail"""
    pass

class RAGError(BaseMLRouterException):
    """Raised when RAG system operations fail"""
    pass

class AgentError(BaseMLRouterException):
    """Raised when agent operations fail"""
    pass

class ConfigurationError(BaseMLRouterException):
    """Raised when configuration is invalid"""
    pass

class RateLimitError(BaseMLRouterException):
    """Raised when rate limits are exceeded"""
    pass

class ExternalServiceError(BaseMLRouterException):
    """Raised when external service calls fail"""
    pass

class DatabaseError(BaseMLRouterException):
    """Raised when database operations fail"""
    pass

# Error handler registration function
def register_error_handlers(app):
    """Register error handlers with Flask app"""
    
    @app.errorhandler(ValidationError)
    def handle_validation_error(error):
        """Handle validation errors"""
        app.logger.warning(f"Validation error: {error.message}")
        return jsonify(error.to_dict()), 400
    
    @app.errorhandler(AuthenticationError)
    def handle_authentication_error(error):
        """Handle authentication errors"""
        app.logger.warning(f"Authentication error: {error.message}")
        return jsonify(error.to_dict()), 401
    
    @app.errorhandler(AuthorizationError)
    def handle_authorization_error(error):
        """Handle authorization errors"""
        app.logger.warning(f"Authorization error: {error.message}")
        return jsonify(error.to_dict()), 403
    
    @app.errorhandler(ServiceError)
    def handle_service_error(error):
        """Handle service errors"""
        app.logger.error(f"Service error: {error.message}")
        return jsonify(error.to_dict()), 503
    
    @app.errorhandler(ModelError)
    def handle_model_error(error):
        """Handle AI model errors"""
        app.logger.error(f"Model error: {error.message}")
        return jsonify(error.to_dict()), 500
    
    @app.errorhandler(CacheError)
    def handle_cache_error(error):
        """Handle cache errors"""
        app.logger.error(f"Cache error: {error.message}")
        return jsonify(error.to_dict()), 500
    
    @app.errorhandler(RAGError)
    def handle_rag_error(error):
        """Handle RAG system errors"""
        app.logger.error(f"RAG error: {error.message}")
        return jsonify(error.to_dict()), 500
    
    @app.errorhandler(AgentError)
    def handle_agent_error(error):
        """Handle agent errors"""
        app.logger.error(f"Agent error: {error.message}")
        return jsonify(error.to_dict()), 500
    
    @app.errorhandler(ConfigurationError)
    def handle_configuration_error(error):
        """Handle configuration errors"""
        app.logger.error(f"Configuration error: {error.message}")
        return jsonify(error.to_dict()), 500
    
    @app.errorhandler(RateLimitError)
    def handle_rate_limit_error(error):
        """Handle rate limit errors"""
        app.logger.warning(f"Rate limit error: {error.message}")
        return jsonify(error.to_dict()), 429
    
    @app.errorhandler(ExternalServiceError)
    def handle_external_service_error(error):
        """Handle external service errors"""
        app.logger.error(f"External service error: {error.message}")
        return jsonify(error.to_dict()), 502
    
    @app.errorhandler(DatabaseError)
    def handle_database_error(error):
        """Handle database errors"""
        app.logger.error(f"Database error: {error.message}")
        return jsonify(error.to_dict()), 500
    
    @app.errorhandler(404)
    def handle_not_found(error):
        """Handle 404 errors"""
        return jsonify({
            'error': 'NotFound',
            'message': 'The requested resource was not found',
            'timestamp': datetime.now().isoformat()
        }), 404
    
    @app.errorhandler(405)
    def handle_method_not_allowed(error):
        """Handle 405 errors"""
        return jsonify({
            'error': 'MethodNotAllowed',
            'message': 'The requested method is not allowed for this resource',
            'timestamp': datetime.now().isoformat()
        }), 405
    
    @app.errorhandler(500)
    def handle_internal_error(error):
        """Handle 500 errors"""
        app.logger.error(f"Internal server error: {str(error)}")
        return jsonify({
            'error': 'InternalServerError',
            'message': 'An internal server error occurred',
            'timestamp': datetime.now().isoformat()
        }), 500
    
    @app.errorhandler(Exception)
    def handle_unexpected_error(error):
        """Handle unexpected errors"""
        app.logger.error(f"Unexpected error: {str(error)}", exc_info=True)
        return jsonify({
            'error': 'UnexpectedError',
            'message': 'An unexpected error occurred',
            'timestamp': datetime.now().isoformat()
        }), 500

# Context managers for specific error handling
class service_error_handler:
    """Context manager for handling service errors"""
    
    def __init__(self, service_name: str, operation: str = None):
        self.service_name = service_name
        self.operation = operation
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            operation_info = f" during {self.operation}" if self.operation else ""
            error_msg = f"{self.service_name} service failed{operation_info}: {str(exc_val)}"
            current_app.logger.error(error_msg)
            
            # Convert to ServiceError if it's not already a custom exception
            if not isinstance(exc_val, BaseMLRouterException):
                raise ServiceError(
                    message=error_msg,
                    details={'original_error': str(exc_val), 'service': self.service_name}
                ) from exc_val
        
        return False  # Don't suppress the exception

class model_error_handler:
    """Context manager for handling model errors"""
    
    def __init__(self, model_id: str, operation: str = None):
        self.model_id = model_id
        self.operation = operation
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            operation_info = f" during {self.operation}" if self.operation else ""
            error_msg = f"Model {self.model_id} failed{operation_info}: {str(exc_val)}"
            current_app.logger.error(error_msg)
            
            if not isinstance(exc_val, BaseMLRouterException):
                raise ModelError(
                    message=error_msg,
                    details={'original_error': str(exc_val), 'model_id': self.model_id}
                ) from exc_val
        
        return False

# Utility functions for error handling
def safe_execute(func, *args, default_return=None, log_errors=True, **kwargs):
    """
    Safely execute a function with error handling
    
    Args:
        func: Function to execute
        *args: Function arguments
        default_return: Default return value on error
        log_errors: Whether to log errors
        **kwargs: Function keyword arguments
        
    Returns:
        Function result or default_return on error
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if log_errors:
            current_app.logger.error(f"Safe execution failed: {str(e)}")
        return default_return

def validate_and_raise(condition, message: str, error_class=ValidationError, **kwargs):
    """
    Validate a condition and raise an exception if it fails
    
    Args:
        condition: Condition to validate
        message: Error message
        error_class: Exception class to raise
        **kwargs: Additional arguments for exception
    """
    if not condition:
        raise error_class(message, **kwargs)

def chain_exceptions(*exception_classes):
    """
    Decorator to chain multiple exception types into a single handler
    
    Args:
        exception_classes: Exception classes to catch and chain
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except exception_classes as e:
                # Chain the exception with additional context
                func_name = getattr(func, '__name__', 'unknown')
                raise ServiceError(
                    message=f"Chained exception in {func_name}: {str(e)}",
                    details={'original_exception': type(e).__name__, 'function': func_name}
                ) from e
        return wrapper
    return decorator
