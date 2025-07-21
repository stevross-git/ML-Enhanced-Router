"""
Configuration Routes
API endpoints for system configuration management
"""

from flask import Blueprint, request, jsonify, current_app
from datetime import datetime

from ..utils.decorators import rate_limit, require_auth, validate_json
from ..utils.exceptions import ValidationError, ServiceError, ConfigurationError

# Create blueprint
config_bp = Blueprint('config', __name__)

@config_bp.route('/', methods=['GET'])
@rate_limit("100 per minute")
@require_auth(roles=['admin'])
def get_system_config():
    """Get system configuration"""
    try:
        # Get current configuration from app config
        config_data = {
            'ml_router': {
                'enabled': current_app.config.get('ML_ROUTER_ENABLED', True),
                'confidence_threshold': current_app.config.get('ML_CONFIDENCE_THRESHOLD', 0.6),
                'max_tokens': current_app.config.get('ML_MAX_TOKENS', 4000),
                'temperature': current_app.config.get('ML_TEMPERATURE', 0.7)
            },
            'cache': {
                'enabled': current_app.config.get('CACHE_TYPE') != 'null',
                'ttl_seconds': current_app.config.get('CACHE_TTL_SECONDS', 3600),
                'max_size': current_app.config.get('CACHE_MAX_SIZE', 10000),
                'type': current_app.config.get('CACHE_TYPE', 'simple')
            },
            'rate_limiting': {
                'enabled': current_app.config.get('RATE_LIMIT_ENABLED', True),
                'per_minute': current_app.config.get('RATE_LIMIT_PER_MINUTE', 100),
                'window_size': current_app.config.get('RATE_LIMIT_WINDOW_SIZE', 60)
            },
            'authentication': {
                'enabled': current_app.config.get('AUTH_ENABLED', False),
                'jwt_expires_hours': current_app.config.get('JWT_ACCESS_TOKEN_EXPIRES', {}).get('hours', 1) if hasattr(current_app.config.get('JWT_ACCESS_TOKEN_EXPIRES', {}), 'get') else 1
            },
            'rag': {
                'enabled': current_app.config.get('RAG_ENABLED', True),
                'chunk_size': current_app.config.get('RAG_CHUNK_SIZE', 1000),
                'chunk_overlap': current_app.config.get('RAG_CHUNK_OVERLAP', 200),
                'vector_db': current_app.config.get('RAG_VECTOR_DB', 'chroma')
            },
            'performance': {
                'max_concurrent_requests': current_app.config.get('MAX_CONCURRENT_REQUESTS', 50),
                'request_timeout': current_app.config.get('REQUEST_TIMEOUT', 30),
                'max_workers': current_app.config.get('MAX_WORKERS', 4)
            },
            'security': {
                'cors_enabled': current_app.config.get('CORS_ENABLED', True),
                'allowed_origins': current_app.config.get('ALLOWED_ORIGINS', []),
                'bcrypt_rounds': current_app.config.get('BCRYPT_LOG_ROUNDS', 12)
            },
            'logging': {
                'level': current_app.config.get('LOG_LEVEL', 'INFO'),
                'format': current_app.config.get('LOG_FORMAT', '%(asctime)s [%(levelname)8s] %(name)s - %(message)s')
            }
        }
        
        return jsonify({
            'config': config_data,
            'timestamp': datetime.utcnow().isoformat(),
            'environment': current_app.config.get('FLASK_ENV', 'development')
        })
        
    except Exception as e:
        current_app.logger.error(f"Error getting system config: {e}")
        return jsonify({'error': 'Failed to retrieve system configuration'}), 500

@config_bp.route('/', methods=['PUT'])
@rate_limit("20 per minute")
@require_auth(roles=['admin'])
@validate_json()
def update_system_config():
    """Update system configuration"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No configuration data provided'}), 400
        
        updated_keys = []
        errors = []
        
        # Update ML Router configuration
        if 'ml_router' in data:
            ml_config = data['ml_router']
            
            if 'enabled' in ml_config:
                current_app.config['ML_ROUTER_ENABLED'] = bool(ml_config['enabled'])
                updated_keys.append('ml_router.enabled')
            
            if 'confidence_threshold' in ml_config:
                threshold = float(ml_config['confidence_threshold'])
                if 0.0 <= threshold <= 1.0:
                    current_app.config['ML_CONFIDENCE_THRESHOLD'] = threshold
                    updated_keys.append('ml_router.confidence_threshold')
                else:
                    errors.append('ml_router.confidence_threshold must be between 0.0 and 1.0')
            
            if 'max_tokens' in ml_config:
                tokens = int(ml_config['max_tokens'])
                if 1 <= tokens <= 32000:
                    current_app.config['ML_MAX_TOKENS'] = tokens
                    updated_keys.append('ml_router.max_tokens')
                else:
                    errors.append('ml_router.max_tokens must be between 1 and 32000')
            
            if 'temperature' in ml_config:
                temp = float(ml_config['temperature'])
                if 0.0 <= temp <= 2.0:
                    current_app.config['ML_TEMPERATURE'] = temp
                    updated_keys.append('ml_router.temperature')
                else:
                    errors.append('ml_router.temperature must be between 0.0 and 2.0')
        
        # Update Cache configuration
        if 'cache' in data:
            cache_config = data['cache']
            
            if 'ttl_seconds' in cache_config:
                ttl = int(cache_config['ttl_seconds'])
                if ttl > 0:
                    current_app.config['CACHE_TTL_SECONDS'] = ttl
                    updated_keys.append('cache.ttl_seconds')
                else:
                    errors.append('cache.ttl_seconds must be positive')
            
            if 'max_size' in cache_config:
                size = int(cache_config['max_size'])
                if size > 0:
                    current_app.config['CACHE_MAX_SIZE'] = size
                    updated_keys.append('cache.max_size')
                else:
                    errors.append('cache.max_size must be positive')
        
        # Update Rate Limiting configuration
        if 'rate_limiting' in data:
            rate_config = data['rate_limiting']
            
            if 'enabled' in rate_config:
                current_app.config['RATE_LIMIT_ENABLED'] = bool(rate_config['enabled'])
                updated_keys.append('rate_limiting.enabled')
            
            if 'per_minute' in rate_config:
                rate = int(rate_config['per_minute'])
                if rate > 0:
                    current_app.config['RATE_LIMIT_PER_MINUTE'] = rate
                    updated_keys.append('rate_limiting.per_minute')
                else:
                    errors.append('rate_limiting.per_minute must be positive')
        
        # Update Authentication configuration
        if 'authentication' in data:
            auth_config = data['authentication']
            
            if 'enabled' in auth_config:
                current_app.config['AUTH_ENABLED'] = bool(auth_config['enabled'])
                updated_keys.append('authentication.enabled')
        
        # Update RAG configuration
        if 'rag' in data:
            rag_config = data['rag']
            
            if 'enabled' in rag_config:
                current_app.config['RAG_ENABLED'] = bool(rag_config['enabled'])
                updated_keys.append('rag.enabled')
            
            if 'chunk_size' in rag_config:
                size = int(rag_config['chunk_size'])
                if size > 0:
                    current_app.config['RAG_CHUNK_SIZE'] = size
                    updated_keys.append('rag.chunk_size')
                else:
                    errors.append('rag.chunk_size must be positive')
            
            if 'chunk_overlap' in rag_config:
                overlap = int(rag_config['chunk_overlap'])
                if overlap >= 0:
                    current_app.config['RAG_CHUNK_OVERLAP'] = overlap
                    updated_keys.append('rag.chunk_overlap')
                else:
                    errors.append('rag.chunk_overlap must be non-negative')
        
        # Update Performance configuration
        if 'performance' in data:
            perf_config = data['performance']
            
            if 'max_concurrent_requests' in perf_config:
                max_req = int(perf_config['max_concurrent_requests'])
                if max_req > 0:
                    current_app.config['MAX_CONCURRENT_REQUESTS'] = max_req
                    updated_keys.append('performance.max_concurrent_requests')
                else:
                    errors.append('performance.max_concurrent_requests must be positive')
            
            if 'request_timeout' in perf_config:
                timeout = int(perf_config['request_timeout'])
                if timeout > 0:
                    current_app.config['REQUEST_TIMEOUT'] = timeout
                    updated_keys.append('performance.request_timeout')
                else:
                    errors.append('performance.request_timeout must be positive')
        
        # Update Logging configuration
        if 'logging' in data:
            log_config = data['logging']
            
            if 'level' in log_config:
                level = log_config['level'].upper()
                valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
                if level in valid_levels:
                    current_app.config['LOG_LEVEL'] = level
                    # Update logger level
                    import logging
                    current_app.logger.setLevel(getattr(logging, level))
                    updated_keys.append('logging.level')
                else:
                    errors.append(f'logging.level must be one of: {", ".join(valid_levels)}')
        
        # Return results
        if errors:
            return jsonify({
                'error': 'Configuration validation failed',
                'details': errors,
                'updated_keys': updated_keys
            }), 400
        
        if not updated_keys:
            return jsonify({'error': 'No valid configuration updates provided'}), 400
        
        current_app.logger.info(f"Configuration updated: {updated_keys}")
        
        return jsonify({
            'message': 'Configuration updated successfully',
            'updated_keys': updated_keys,
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except ValueError as e:
        return jsonify({'error': f'Invalid value in configuration: {str(e)}'}), 400
    except Exception as e:
        current_app.logger.error(f"Error updating system config: {e}")
        return jsonify({'error': 'Failed to update system configuration'}), 500

@config_bp.route('/features', methods=['GET'])
@rate_limit("100 per minute")
def get_feature_flags():
    """Get feature flags and system capabilities"""
    try:
        features = {
            'ml_router': current_app.config.get('ML_ROUTER_ENABLED', True),
            'cache': current_app.config.get('CACHE_TYPE') != 'null',
            'rate_limiting': current_app.config.get('RATE_LIMIT_ENABLED', True),
            'authentication': current_app.config.get('AUTH_ENABLED', False),
            'rag': current_app.config.get('RAG_ENABLED', True),
            'email_integration': current_app.config.get('EMAIL_ENABLED', False),
            'network_integration': current_app.config.get('NETWORK_ENABLED', False),
            'metrics': current_app.config.get('METRICS_ENABLED', True),
            'swagger_docs': current_app.config.get('SWAGGER_ENABLED', True),
            'cors': current_app.config.get('CORS_ENABLED', True),
            'compression': True,  # Always enabled
            'health_checks': True  # Always enabled
        }
        
        return jsonify({
            'features': features,
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        current_app.logger.error(f"Error getting feature flags: {e}")
        return jsonify({'error': 'Failed to retrieve feature flags'}), 500

@config_bp.route('/environment', methods=['GET'])
@rate_limit("100 per minute")
@require_auth(roles=['admin'])
def get_environment_info():
    """Get environment information"""
    try:
        import sys
        import platform
        
        env_info = {
            'python_version': sys.version,
            'platform': platform.platform(),
            'flask_env': current_app.config.get('FLASK_ENV', 'development'),
            'debug_mode': current_app.debug,
            'testing_mode': current_app.testing,
            'database_url': current_app.config.get('SQLALCHEMY_DATABASE_URI', '').split('://', 1)[0] + '://***' if current_app.config.get('SQLALCHEMY_DATABASE_URI') else None,
            'cache_type': current_app.config.get('CACHE_TYPE', 'simple'),
            'log_level': current_app.config.get('LOG_LEVEL', 'INFO'),
            'max_content_length': current_app.config.get('MAX_CONTENT_LENGTH'),
            'secret_key_set': bool(current_app.config.get('SECRET_KEY')),
            'api_keys_configured': {
                'openai': bool(current_app.config.get('OPENAI_API_KEY')),
                'anthropic': bool(current_app.config.get('ANTHROPIC_API_KEY')),
                'google': bool(current_app.config.get('GOOGLE_API_KEY')),
                'cohere': bool(current_app.config.get('COHERE_API_KEY')),
                'mistral': bool(current_app.config.get('MISTRAL_API_KEY'))
            }
        }
        
        return jsonify({
            'environment': env_info,
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        current_app.logger.error(f"Error getting environment info: {e}")
        return jsonify({'error': 'Failed to retrieve environment information'}), 500

@config_bp.route('/reset', methods=['POST'])
@rate_limit("5 per minute")
@require_auth(roles=['admin'])
@validate_json(['confirm'])
def reset_config():
    """Reset configuration to defaults"""
    try:
        data = request.get_json()
        
        if not data.get('confirm'):
            return jsonify({
                'error': 'Confirmation required',
                'message': 'Set "confirm": true to proceed with configuration reset'
            }), 400
        
        # Reset to default values
        defaults = {
            'ML_ROUTER_ENABLED': True,
            'ML_CONFIDENCE_THRESHOLD': 0.6,
            'ML_MAX_TOKENS': 4000,
            'ML_TEMPERATURE': 0.7,
            'CACHE_TTL_SECONDS': 3600,
            'CACHE_MAX_SIZE': 10000,
            'RATE_LIMIT_ENABLED': True,
            'RATE_LIMIT_PER_MINUTE': 100,
            'RAG_ENABLED': True,
            'RAG_CHUNK_SIZE': 1000,
            'RAG_CHUNK_OVERLAP': 200,
            'MAX_CONCURRENT_REQUESTS': 50,
            'REQUEST_TIMEOUT': 30,
            'LOG_LEVEL': 'INFO'
        }
        
        # Apply defaults
        for key, value in defaults.items():
            current_app.config[key] = value
        
        current_app.logger.warning("Configuration reset to defaults")
        
        return jsonify({
            'message': 'Configuration reset to defaults',
            'reset_keys': list(defaults.keys()),
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        current_app.logger.error(f"Error resetting config: {e}")
        return jsonify({'error': 'Failed to reset configuration'}), 500

@config_bp.route('/export', methods=['GET'])
@rate_limit("10 per minute")
@require_auth(roles=['admin'])
def export_config():
    """Export current configuration"""
    try:
        # Get all relevant configuration
        config_export = {}
        
        # Define exportable config keys (excluding secrets)
        exportable_keys = [
            'ML_ROUTER_ENABLED', 'ML_CONFIDENCE_THRESHOLD', 'ML_MAX_TOKENS', 'ML_TEMPERATURE',
            'CACHE_TTL_SECONDS', 'CACHE_MAX_SIZE', 'RATE_LIMIT_ENABLED', 'RATE_LIMIT_PER_MINUTE',
            'AUTH_ENABLED', 'RAG_ENABLED', 'RAG_CHUNK_SIZE', 'RAG_CHUNK_OVERLAP',
            'MAX_CONCURRENT_REQUESTS', 'REQUEST_TIMEOUT', 'LOG_LEVEL', 'CORS_ENABLED'
        ]
        
        for key in exportable_keys:
            if key in current_app.config:
                config_export[key] = current_app.config[key]
        
        return jsonify({
            'config': config_export,
            'exported_at': datetime.utcnow().isoformat(),
            'environment': current_app.config.get('FLASK_ENV', 'development'),
            'version': '1.0'
        })
        
    except Exception as e:
        current_app.logger.error(f"Error exporting config: {e}")
        return jsonify({'error': 'Failed to export configuration'}), 500

@config_bp.route('/import', methods=['POST'])
@rate_limit("5 per minute")
@require_auth(roles=['admin'])
@validate_json(['config'])
def import_config():
    """Import configuration from export"""
    try:
        data = request.get_json()
        config_data = data['config']
        
        if not isinstance(config_data, dict):
            return jsonify({'error': 'Configuration must be a dictionary'}), 400
        
        # Validate and import configuration
        imported_keys = []
        errors = []
        
        for key, value in config_data.items():
            try:
                # Basic validation based on key patterns
                if key.endswith('_ENABLED') and not isinstance(value, bool):
                    errors.append(f'{key} must be a boolean')
                    continue
                elif key.endswith('_SECONDS') and not isinstance(value, int):
                    errors.append(f'{key} must be an integer')
                    continue
                elif key == 'ML_CONFIDENCE_THRESHOLD' and not (0.0 <= value <= 1.0):
                    errors.append(f'{key} must be between 0.0 and 1.0')
                    continue
                elif key == 'LOG_LEVEL' and value not in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
                    errors.append(f'{key} must be a valid log level')
                    continue
                
                # Apply configuration
                current_app.config[key] = value
                imported_keys.append(key)
                
            except Exception as e:
                errors.append(f'Error importing {key}: {str(e)}')
        
        if errors:
            return jsonify({
                'error': 'Configuration import failed',
                'details': errors,
                'imported_keys': imported_keys
            }), 400
        
        current_app.logger.info(f"Configuration imported: {imported_keys}")
        
        return jsonify({
            'message': 'Configuration imported successfully',
            'imported_keys': imported_keys,
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except ValidationError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        current_app.logger.error(f"Error importing config: {e}")
        return jsonify({'error': 'Failed to import configuration'}), 500