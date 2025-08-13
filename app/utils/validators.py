"""
Input Validation Utilities
Functions for validating API inputs and data structures
"""

import re
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime

@dataclass
class ValidationResult:
    """Result of validation operation"""
    is_valid: bool
    errors: List[str]
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []

def validate_query_request(data: Dict[str, Any]) -> ValidationResult:
    """
    Validate query request data
    
    Args:
        data: Request data dictionary
        
    Returns:
        ValidationResult with validation status and errors
    """
    errors = []
    warnings = []
    
    # Required fields
    if 'query' not in data:
        errors.append("'query' field is required")
    elif not isinstance(data['query'], str):
        errors.append("'query' must be a string")
    elif not data['query'].strip():
        errors.append("'query' cannot be empty")
    elif len(data['query']) > 10000:
        errors.append("'query' exceeds maximum length of 10,000 characters")
    
    # Optional fields validation
    if 'model_id' in data:
        if not isinstance(data['model_id'], str):
            errors.append("'model_id' must be a string")
        elif len(data['model_id']) > 200:
            errors.append("'model_id' exceeds maximum length of 200 characters")
    
    if 'parameters' in data:
        if not isinstance(data['parameters'], dict):
            errors.append("'parameters' must be a dictionary")
        else:
            param_errors = validate_parameters(data['parameters'])
            errors.extend(param_errors)
    
    if 'session_id' in data:
        if not isinstance(data['session_id'], str):
            errors.append("'session_id' must be a string")
        elif len(data['session_id']) > 100:
            errors.append("'session_id' exceeds maximum length of 100 characters")
    
    # Check for suspicious content
    query_text = data.get('query', '')
    if _contains_suspicious_content(query_text):
        warnings.append("Query contains potentially suspicious content")
    
    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings
    )

def validate_parameters(parameters: Dict[str, Any]) -> List[str]:
    """Validate query parameters"""
    errors = []
    
    # Validate temperature
    if 'temperature' in parameters:
        temp = parameters['temperature']
        if not isinstance(temp, (int, float)):
            errors.append("'temperature' must be a number")
        elif temp < 0 or temp > 2:
            errors.append("'temperature' must be between 0 and 2")
    
    # Validate max_tokens
    if 'max_tokens' in parameters:
        tokens = parameters['max_tokens']
        if not isinstance(tokens, int):
            errors.append("'max_tokens' must be an integer")
        elif tokens < 1 or tokens > 8192:
            errors.append("'max_tokens' must be between 1 and 8192")
    
    # Validate top_p
    if 'top_p' in parameters:
        top_p = parameters['top_p']
        if not isinstance(top_p, (int, float)):
            errors.append("'top_p' must be a number")
        elif top_p < 0 or top_p > 1:
            errors.append("'top_p' must be between 0 and 1")
    
    # Validate presence_penalty
    if 'presence_penalty' in parameters:
        penalty = parameters['presence_penalty']
        if not isinstance(penalty, (int, float)):
            errors.append("'presence_penalty' must be a number")
        elif penalty < -2 or penalty > 2:
            errors.append("'presence_penalty' must be between -2 and 2")
    
    # Validate frequency_penalty
    if 'frequency_penalty' in parameters:
        penalty = parameters['frequency_penalty']
        if not isinstance(penalty, (int, float)):
            errors.append("'frequency_penalty' must be a number")
        elif penalty < -2 or penalty > 2:
            errors.append("'frequency_penalty' must be between -2 and 2")
    
    return errors

def validate_email(email: str) -> bool:
    """Validate email address"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def validate_username(username: str) -> List[str]:
    """Validate username"""
    errors = []
    
    if not isinstance(username, str):
        errors.append("Username must be a string")
        return errors
    
    if len(username) < 3:
        errors.append("Username must be at least 3 characters")
    
    if len(username) > 50:
        errors.append("Username must be at most 50 characters")
    
    if not re.match(r'^[a-zA-Z0-9_-]+$', username):
        errors.append("Username can only contain letters, numbers, underscore, and hyphen")
    
    return errors

def validate_password(password: str, strict: bool = True) -> List[str]:
    """
    Validate password strength
    
    Args:
        password: Password to validate
        strict: Whether to apply strict validation rules
    """
    errors = []
    
    if not isinstance(password, str):
        errors.append("Password must be a string")
        return errors
    
    # Basic length requirements
    min_length = 12 if strict else 8
    if len(password) < min_length:
        errors.append(f"Password must be at least {min_length} characters")
    
    if len(password) > 128:
        errors.append("Password must be at most 128 characters")
    
    # Check for common weak patterns
    weak_patterns = [
        r'123456', r'password', r'qwerty', r'admin', r'root',
        r'(.)\1{3,}',  # Repeated characters (4+ times)
        r'012345', r'abcdef'
    ]
    
    for pattern in weak_patterns:
        if re.search(pattern, password.lower()):
            errors.append("Password contains common weak patterns")
            break
    
    if strict:
        # Strict requirements
        if len(password) < 12:
            errors.append("Password must be at least 12 characters in production")
        
        # Check for at least one uppercase letter
        if not re.search(r'[A-Z]', password):
            errors.append("Password must contain at least one uppercase letter")
        
        # Check for at least one lowercase letter
        if not re.search(r'[a-z]', password):
            errors.append("Password must contain at least one lowercase letter")
        
        # Check for at least one digit
        if not re.search(r'\d', password):
            errors.append("Password must contain at least one digit")
        
        # Check for at least one special character
        if not re.search(r'[!@#$%^&*(),.?":{}|<>_\-+=\[\]\\\/~`]', password):
            errors.append("Password must contain at least one special character")
        
        # Check for at least two different character types
        char_types = 0
        if re.search(r'[A-Z]', password): char_types += 1
        if re.search(r'[a-z]', password): char_types += 1
        if re.search(r'\d', password): char_types += 1
        if re.search(r'[!@#$%^&*(),.?":{}|<>_\-+=\[\]\\\/~`]', password): char_types += 1
        
        if char_types < 3:
            errors.append("Password must contain at least 3 different character types")
    
    return errors

def enforce_password_policy():
    """Decorator to enforce password policy on routes"""
    def decorator(f):
        import functools
        from flask import request, jsonify, current_app
        
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            if request.is_json:
                data = request.get_json()
                if data and 'password' in data:
                    is_production = current_app.config.get('FLASK_ENV') == 'production'
                    errors = validate_password(data['password'], strict=is_production)
                    if errors:
                        return jsonify({
                            'error': 'Password does not meet security requirements',
                            'password_errors': errors
                        }), 400
            
            return f(*args, **kwargs)
        return wrapper
    return decorator

def validate_model_config(config: Dict[str, Any]) -> List[str]:
    """Validate AI model configuration"""
    errors = []
    
    required_fields = ['name', 'provider', 'model_id']
    for field in required_fields:
        if field not in config:
            errors.append(f"'{field}' is required")
    
    # Validate provider
    if 'provider' in config:
        valid_providers = [
            'openai', 'anthropic', 'google', 'microsoft', 'xai',
            'perplexity', 'cohere', 'huggingface', 'ollama', 'custom'
        ]
        if config['provider'] not in valid_providers:
            errors.append(f"'provider' must be one of: {', '.join(valid_providers)}")
    
    # Validate categories
    if 'categories' in config:
        if not isinstance(config['categories'], list):
            errors.append("'categories' must be a list")
        elif not all(isinstance(cat, str) for cat in config['categories']):
            errors.append("All categories must be strings")
    
    # Validate endpoint URL
    if 'endpoint' in config:
        if not isinstance(config['endpoint'], str):
            errors.append("'endpoint' must be a string")
        elif not config['endpoint'].startswith(('http://', 'https://')):
            errors.append("'endpoint' must be a valid URL")
    
    return errors

def validate_file_upload(file_data: Dict[str, Any]) -> List[str]:
    """Validate file upload data"""
    errors = []
    
    # Check file size
    max_size = 50 * 1024 * 1024  # 50MB
    if 'size' in file_data and file_data['size'] > max_size:
        errors.append(f"File size exceeds maximum of {max_size // (1024*1024)}MB")
    
    # Check file type
    allowed_extensions = {
        'txt', 'md', 'pdf', 'doc', 'docx', 'html', 'json', 'csv', 'xlsx'
    }
    if 'filename' in file_data:
        extension = file_data['filename'].split('.')[-1].lower()
        if extension not in allowed_extensions:
            errors.append(f"File type '{extension}' not allowed. Allowed types: {', '.join(allowed_extensions)}")
    
    return errors

def _contains_suspicious_content(text: str) -> bool:
    """Check for potentially suspicious content"""
    suspicious_patterns = [
        r'<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>',  # Script tags
        r'javascript:',  # JavaScript URLs
        r'data:.*base64',  # Base64 data URLs
        r'eval\s*\(',  # eval() calls
        r'document\.cookie',  # Cookie access
        r'localStorage',  # Local storage access
        r'sessionStorage',  # Session storage access
    ]
    
    for pattern in suspicious_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    
    return False

def sanitize_input(text: str) -> str:
    """Sanitize input text"""
    if not isinstance(text, str):
        return str(text)
    
    # Remove null bytes
    text = text.replace('\x00', '')
    
    # Limit length
    if len(text) > 50000:
        text = text[:50000] + '...'
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text

def validate_json_structure(data: Any, schema: Dict[str, Any]) -> List[str]:
    """Validate JSON data against a schema"""
    errors = []
    
    def _validate_recursive(obj, schema_part, path=""):
        if 'type' in schema_part:
            expected_type = schema_part['type']
            if expected_type == 'object' and not isinstance(obj, dict):
                errors.append(f"{path}: expected object, got {type(obj).__name__}")
                return
            elif expected_type == 'array' and not isinstance(obj, list):
                errors.append(f"{path}: expected array, got {type(obj).__name__}")
                return
            elif expected_type == 'string' and not isinstance(obj, str):
                errors.append(f"{path}: expected string, got {type(obj).__name__}")
                return
            elif expected_type == 'number' and not isinstance(obj, (int, float)):
                errors.append(f"{path}: expected number, got {type(obj).__name__}")
                return
            elif expected_type == 'boolean' and not isinstance(obj, bool):
                errors.append(f"{path}: expected boolean, got {type(obj).__name__}")
                return
        
        if 'properties' in schema_part and isinstance(obj, dict):
            for prop, prop_schema in schema_part['properties'].items():
                prop_path = f"{path}.{prop}" if path else prop
                if prop in obj:
                    _validate_recursive(obj[prop], prop_schema, prop_path)
                elif prop_schema.get('required', False):
                    errors.append(f"{prop_path}: required property missing")
    
    _validate_recursive(data, schema)
    return errors