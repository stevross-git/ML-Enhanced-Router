"""
Production Environment Validation
Ensures all required environment variables and security settings are configured for production
"""

import os
import sys
from typing import List, Tuple

def validate_production_environment() -> Tuple[bool, List[str]]:
    """
    Validate that all required production environment variables are set
    
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    # Required environment variables for production
    required_vars = [
        'SECRET_KEY',
        'JWT_SECRET_KEY', 
        'SESSION_SECRET',
        'POSTGRES_USER',
        'POSTGRES_PASSWORD',
        'PGADMIN_PASSWORD'
    ]
    
    # Check required environment variables
    for var in required_vars:
        value = os.environ.get(var)
        if not value:
            errors.append(f"Missing required environment variable: {var}")
        elif len(value) < 32:  # Minimum length for secrets
            errors.append(f"Environment variable {var} is too short (minimum 32 characters)")
    
    # Check CORS origins are set
    cors_origins = os.environ.get('CORS_ORIGINS')
    if not cors_origins:
        errors.append("CORS_ORIGINS must be set in production")
    
    # Check Flask environment
    flask_env = os.environ.get('FLASK_ENV', 'development')
    if flask_env != 'production':
        errors.append(f"FLASK_ENV must be set to 'production', currently: {flask_env}")
    
    # Validate debug mode is disabled
    flask_debug = os.environ.get('FLASK_DEBUG', 'False').lower()
    if flask_debug == 'true':
        errors.append("FLASK_DEBUG must be False or unset in production")
    
    # Check database configuration
    database_url = os.environ.get('DATABASE_URL')
    if database_url and 'ml_router_password' in database_url:
        errors.append("Database URL contains default password - use environment variables")
    
    return len(errors) == 0, errors

def check_security_configuration() -> Tuple[bool, List[str]]:
    """
    Check security-related configuration
    
    Returns:
        Tuple of (is_valid, list_of_warnings)
    """
    warnings = []
    
    # Check SSL/TLS configuration
    if not os.environ.get('SSL_CERT_PATH'):
        warnings.append("SSL_CERT_PATH not set - ensure HTTPS is configured at load balancer level")
    
    # Check rate limiting
    if not os.environ.get('REDIS_URL'):
        warnings.append("REDIS_URL not set - rate limiting will use in-memory storage")
    
    # Check authentication
    auth_enabled = os.environ.get('AUTH_ENABLED', 'true').lower()
    if auth_enabled != 'true':
        warnings.append("AUTH_ENABLED is not set to true - authentication may be disabled")
    
    return True, warnings

def production_startup_checks():
    """
    Perform production startup checks and exit if critical issues found
    """
    print("🔍 Running production environment validation...")
    
    # Check environment variables
    env_valid, env_errors = validate_production_environment()
    
    if not env_valid:
        print("❌ CRITICAL: Production environment validation failed!")
        print("\nRequired fixes:")
        for error in env_errors:
            print(f"  • {error}")
        
        print("\n💡 To fix these issues:")
        print("  1. Set all required environment variables")
        print("  2. Ensure secrets are at least 32 characters long")
        print("  3. Set FLASK_ENV=production")
        print("  4. Configure CORS_ORIGINS with trusted domains")
        
        sys.exit(1)
    
    # Check security configuration
    security_valid, security_warnings = check_security_configuration()
    
    if security_warnings:
        print("⚠️  Security configuration warnings:")
        for warning in security_warnings:
            print(f"  • {warning}")
    
    print("✅ Production environment validation passed!")
    return True

if __name__ == "__main__":
    production_startup_checks()