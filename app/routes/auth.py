"""
Authentication Routes
User authentication, registration, and session management
"""

from flask import Blueprint, request, jsonify, session, current_app
from datetime import datetime
import logging

from ..services.auth_service import get_auth_service
from ..utils.decorators import rate_limit, validate_json, require_auth
from ..utils.exceptions import AuthenticationError, ValidationError

# Create blueprint
auth_bp = Blueprint('auth', __name__)

# Set up logging
logger = logging.getLogger(__name__)

@auth_bp.route('/current-user', methods=['GET'])
@rate_limit("100 per minute")
def get_current_user():
    """Get current user info"""
    try:
        auth_service = get_auth_service()
        if not auth_service:
            return jsonify({'error': 'Auth service not initialized'}), 503
        
        # For now, return the admin user
        admin_user = auth_service.users.get('admin')
        if not admin_user:
            return jsonify({'status': 'error', 'error': 'No user found'}), 404
        
        user_data = {
            'id': admin_user.id,
            'username': admin_user.username,
            'email': admin_user.email,
            'role': admin_user.role.value,
            'api_key': admin_user.api_key,
            'created_at': admin_user.created_at.isoformat(),
            'last_login': admin_user.last_login.isoformat() if admin_user.last_login else None,
            'is_active': admin_user.is_active,
            'permissions': admin_user.permissions
        }
        
        return jsonify({'status': 'success', 'user': user_data})
        
    except Exception as e:
        logger.error(f"Error getting current user: {e}")
        return jsonify({'status': 'error', 'error': 'Failed to get current user'}), 500

@auth_bp.route('/users', methods=['GET'])
@require_auth(roles=['admin'])
@rate_limit("50 per minute")
def get_all_users():
    """Get all users"""
    try:
        auth_service = get_auth_service()
        if not auth_service:
            return jsonify({'error': 'Auth service not initialized'}), 503
        
        users = auth_service.get_all_users()
        users_data = []
        
        for user in users:
            users_data.append({
                'id': user.id,
                'username': user.username,
                'email': user.email,
                'role': user.role.value,
                'created_at': user.created_at.isoformat(),
                'last_login': user.last_login.isoformat() if user.last_login else None,
                'is_active': user.is_active
            })
        
        return jsonify({'status': 'success', 'users': users_data})
        
    except Exception as e:
        logger.error(f"Error getting users: {e}")
        return jsonify({'status': 'error', 'error': 'Failed to get users'}), 500

@auth_bp.route('/regenerate-api-key', methods=['POST'])
@require_auth()
@rate_limit("5 per hour")
def regenerate_api_key():
    """Regenerate API key for current user"""
    try:
        auth_service = get_auth_service()
        if not auth_service:
            return jsonify({'error': 'Auth service not initialized'}), 503
        
        data = request.get_json() or {}
        user_id = data.get('user_id', 'admin')  # Default to admin for now
        
        new_api_key = auth_service.regenerate_api_key(user_id)
        
        if new_api_key:
            return jsonify({
                'status': 'success',
                'message': 'API key regenerated successfully',
                'api_key': new_api_key
            })
        else:
            return jsonify({'status': 'error', 'error': 'Failed to regenerate API key'}), 500
            
    except Exception as e:
        logger.error(f"Error regenerating API key: {e}")
        return jsonify({'status': 'error', 'error': 'Failed to regenerate API key'}), 500

@auth_bp.route('/login', methods=['POST'])
@rate_limit("10 per minute")
@validate_json(['username', 'password'])
def login():
    """User login endpoint"""
    try:
        auth_service = get_auth_service()
        if not auth_service:
            return jsonify({'error': 'Auth service not initialized'}), 503
        
        data = request.get_json()
        username = data['username']
        password = data['password']
        
        # Authenticate user
        user = auth_service.authenticate_user(username, password)
        if not user:
            return jsonify({'status': 'error', 'error': 'Invalid credentials'}), 401
        
        if not user.is_active:
            return jsonify({'status': 'error', 'error': 'Account is deactivated'}), 403
        
        # Generate JWT token
        token = auth_service.generate_jwt_token(user.id)
        
        # Update last login
        user.last_login = datetime.now()
        auth_service.save_users()
        
        # Set session
        session['user_id'] = user.id
        session['username'] = user.username
        session['role'] = user.role.value
        
        return jsonify({
            'status': 'success',
            'message': 'Login successful',
            'token': token,
            'user': {
                'id': user.id,
                'username': user.username,
                'email': user.email,
                'role': user.role.value
            }
        })
        
    except ValidationError as e:
        return jsonify({'status': 'error', 'error': str(e)}), 400
    except AuthenticationError as e:
        return jsonify({'status': 'error', 'error': str(e)}), 401
    except Exception as e:
        logger.error(f"Login error: {e}")
        return jsonify({'status': 'error', 'error': 'Login failed'}), 500

@auth_bp.route('/logout', methods=['POST'])
@rate_limit("30 per minute")
def logout():
    """User logout endpoint"""
    try:
        # Clear session
        session.clear()
        
        return jsonify({
            'status': 'success',
            'message': 'Logout successful'
        })
        
    except Exception as e:
        logger.error(f"Logout error: {e}")
        return jsonify({'status': 'error', 'error': 'Logout failed'}), 500

@auth_bp.route('/register', methods=['POST'])
@rate_limit("5 per hour")
@validate_json(['username', 'email', 'password'])
def register():
    """User registration endpoint"""
    try:
        auth_service = get_auth_service()
        if not auth_service:
            return jsonify({'error': 'Auth service not initialized'}), 503
        
        data = request.get_json()
        username = data['username']
        email = data['email']
        password = data['password']
        role = data.get('role', 'user')
        
        # Validate input
        if len(username) < 3:
            return jsonify({'status': 'error', 'error': 'Username must be at least 3 characters'}), 400
        
        if len(password) < 6:
            return jsonify({'status': 'error', 'error': 'Password must be at least 6 characters'}), 400
        
        # Check if user exists
        if auth_service.get_user_by_username(username):
            return jsonify({'status': 'error', 'error': 'Username already exists'}), 409
        
        if auth_service.get_user_by_email(email):
            return jsonify({'status': 'error', 'error': 'Email already registered'}), 409
        
        # Create user
        user = auth_service.create_user(username, email, password, role)
        
        if user:
            return jsonify({
                'status': 'success',
                'message': 'User registered successfully',
                'user': {
                    'id': user.id,
                    'username': user.username,
                    'email': user.email,
                    'role': user.role.value
                }
            }), 201
        else:
            return jsonify({'status': 'error', 'error': 'Failed to create user'}), 500
            
    except ValidationError as e:
        return jsonify({'status': 'error', 'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Registration error: {e}")
        return jsonify({'status': 'error', 'error': 'Registration failed'}), 500

@auth_bp.route('/verify-token', methods=['POST'])
@rate_limit("100 per minute")
def verify_token():
    """Verify JWT token"""
    try:
        auth_service = get_auth_service()
        if not auth_service:
            return jsonify({'error': 'Auth service not initialized'}), 503
        
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'status': 'error', 'error': 'No token provided'}), 401
        
        # Remove 'Bearer ' prefix if present
        if token.startswith('Bearer '):
            token = token[7:]
        
        # Verify token
        payload = auth_service.verify_jwt_token(token)
        if not payload:
            return jsonify({'status': 'error', 'error': 'Invalid token'}), 401
        
        # Get user info
        user = auth_service.get_user_by_id(payload.get('user_id'))
        if not user:
            return jsonify({'status': 'error', 'error': 'User not found'}), 404
        
        if not user.is_active:
            return jsonify({'status': 'error', 'error': 'Account is deactivated'}), 403
        
        return jsonify({
            'status': 'success',
            'user': {
                'id': user.id,
                'username': user.username,
                'email': user.email,
                'role': user.role.value
            }
        })
        
    except Exception as e:
        logger.error(f"Token verification error: {e}")
        return jsonify({'status': 'error', 'error': 'Token verification failed'}), 500

@auth_bp.route('/usage-stats', methods=['GET'])
@require_auth()
@rate_limit("50 per minute")
def get_usage_stats():
    """Get API usage statistics for current user"""
    try:
        auth_service = get_auth_service()
        if not auth_service:
            return jsonify({'error': 'Auth service not initialized'}), 503
        
        user_id = session.get('user_id', 'admin')
        
        # Mock usage statistics (in real implementation, fetch from database)
        stats = {
            'total_requests': 1250,
            'requests_today': 87,
            'requests_this_month': 2340,
            'error_rate': 2.1,
            'avg_response_time': 450,
            'last_request': datetime.now().isoformat(),
            'quota_limit': 10000,
            'quota_remaining': 7660
        }
        
        return jsonify({
            'status': 'success',
            'stats': stats
        })
        
    except Exception as e:
        logger.error(f"Error getting usage stats: {e}")
        return jsonify({'status': 'error', 'error': 'Failed to get usage statistics'}), 500

@auth_bp.route('/api-key-status', methods=['GET'])
@require_auth()
@rate_limit("100 per minute")
def get_api_key_status():
    """Get API key status and validation"""
    try:
        auth_service = get_auth_service()
        if not auth_service:
            return jsonify({'error': 'Auth service not initialized'}), 503
        
        user_id = session.get('user_id', 'admin')
        user = auth_service.get_user_by_id(user_id)
        
        if not user:
            return jsonify({'status': 'error', 'error': 'User not found'}), 404
        
        # Check API key status
        status_info = {
            'api_key': user.api_key,
            'is_valid': True,
            'created_at': user.created_at.isoformat(),
            'last_used': user.last_login.isoformat() if user.last_login else None,
            'is_active': user.is_active
        }
        
        return jsonify({
            'status': 'success',
            'status_info': status_info
        })
        
    except Exception as e:
        logger.error(f"Error getting API key status: {e}")
        return jsonify({'status': 'error', 'error': 'Failed to get API key status'}), 500

@auth_bp.route('/save-api-keys', methods=['POST'])
@require_auth(roles=['admin'])
@rate_limit("10 per hour")
@validate_json(['keys'])
def save_api_keys():
    """Save API keys (admin only)"""
    try:
        auth_service = get_auth_service()
        if not auth_service:
            return jsonify({'error': 'Auth service not initialized'}), 503
        
        data = request.get_json()
        api_keys = data['keys']
        
        # Validate API keys format
        required_fields = ['openai_api_key', 'anthropic_api_key', 'google_api_key']
        for field in required_fields:
            if field not in api_keys:
                return jsonify({'status': 'error', 'error': f'Missing required field: {field}'}), 400
        
        # Save API keys (in real implementation, encrypt and store securely)
        saved_keys = auth_service.save_system_api_keys(api_keys)
        
        if saved_keys:
            return jsonify({
                'status': 'success',
                'message': 'API keys saved successfully',
                'saved_count': len(saved_keys)
            })
        else:
            return jsonify({'status': 'error', 'error': 'Failed to save API keys'}), 500
            
    except ValidationError as e:
        return jsonify({'status': 'error', 'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Error saving API keys: {e}")
        return jsonify({'status': 'error', 'error': 'Failed to save API keys'}), 500

# Error handlers for authentication blueprint
@auth_bp.errorhandler(AuthenticationError)
def handle_auth_error(error):
    """Handle authentication errors"""
    return jsonify({'status': 'error', 'error': str(error)}), 401

@auth_bp.errorhandler(ValidationError)
def handle_validation_error(error):
    """Handle validation errors"""
    return jsonify({'status': 'error', 'error': str(error)}), 400

@auth_bp.errorhandler(429)
def handle_rate_limit_error(error):
    """Handle rate limit errors"""
    return jsonify({
        'status': 'error', 
        'error': 'Rate limit exceeded',
        'message': 'Too many requests. Please try again later.'
    }), 429