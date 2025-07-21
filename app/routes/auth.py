"""
Authentication Routes
User authentication, registration, and session management
"""

from flask import Blueprint, request, jsonify, session, current_app
from datetime import datetime

from ..services.auth_service import get_auth_service
from ..utils.decorators import rate_limit, validate_json, require_auth
from ..utils.exceptions import AuthenticationError, ValidationError

# Create blueprint
auth_bp = Blueprint('auth', __name__)

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
        current_app.logger.error(f"Error getting current user: {e}")
        return jsonify({'status': 'error', 'error': 'Failed to get current user'}), 500

@auth_bp.route('/users', methods=['GET'])
@require_auth(roles=['admin'])
@rate_limit("50 per minute")
def get_all_users():
    """Get all users (admin only)"""
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
        current_app.logger.error(f"Error getting users: {e}")
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
        
        # For now, regenerate for admin user
        new_api_key = auth_service.regenerate_api_key('admin')
        if not new_api_key:
            return jsonify({'status': 'error', 'error': 'Failed to regenerate API key'}), 500
        
        return jsonify({'status': 'success', 'api_key': new_api_key})
        
    except Exception as e:
        current_app.logger.error(f"Error regenerating API key: {e}")
        return jsonify({'status': 'error', 'error': 'Failed to regenerate API key'}), 500

@auth_bp.route('/generate-jwt', methods=['POST'])
@require_auth()
@rate_limit("10 per hour")
@validate_json()
def generate_jwt():
    """Generate JWT token"""
    try:
        auth_service = get_auth_service()
        if not auth_service:
            return jsonify({'error': 'Auth service not initialized'}), 503
        
        data = request.get_json()
        expires_in = data.get('expires_in', 3600)
        
        # For now, generate for admin user
        token = auth_service.generate_jwt_token('admin', expires_in)
        
        return jsonify({'status': 'success', 'token': token})
        
    except Exception as e:
        current_app.logger.error(f"Error generating JWT: {e}")
        return jsonify({'status': 'error', 'error': 'Failed to generate JWT'}), 500

@auth_bp.route('/usage-stats', methods=['GET'])
@require_auth()
@rate_limit("100 per hour")
def get_usage_stats():
    """Get API usage statistics"""
    try:
        # Return mock data for now
        stats = {
            'total_requests': 150,
            'requests_today': 25,
            'error_rate': 2.5,
            'last_request': datetime.now().isoformat()
        }
        
        return jsonify({'status': 'success', 'stats': stats})
        
    except Exception as e:
        current_app.logger.error(f"Error getting usage stats: {e}")
        return jsonify({'status': 'error', 'error': 'Failed to get usage stats'}), 500

@auth_bp.route('/login', methods=['POST'])
@rate_limit("10 per minute")
@validate_json(['username', 'password'])
def login():
    """User login"""
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
            raise AuthenticationError("Invalid username or password")
        
        # Create session
        session['user_id'] = user.id
        session['username'] = user.username
        session['role'] = user.role.value
        
        # Generate JWT token
        token = auth_service.generate_jwt_token(user.id)
        
        return jsonify({
            'status': 'success',
            'message': 'Login successful',
            'user': {
                'id': user.id,
                'username': user.username,
                'email': user.email,
                'role': user.role.value
            },
            'token': token
        })
        
    except AuthenticationError as e:
        return jsonify({'status': 'error', 'error': str(e)}), 401
    except Exception as e:
        current_app.logger.error(f"Login error: {e}")
        return jsonify({'status': 'error', 'error': 'Login failed'}), 500

@auth_bp.route('/logout', methods=['POST'])
@rate_limit("20 per minute")
def logout():
    """User logout"""
    try:
        user_id = session.get('user_id')
        
        # Clear session
        session.clear()
        
        return jsonify({
            'status': 'success',
            'message': 'Logout successful',
            'user_id': user_id
        })
        
    except Exception as e:
        current_app.logger.error(f"Logout error: {e}")
        return jsonify({'status': 'error', 'error': 'Logout failed'}), 500

@auth_bp.route('/register', methods=['POST'])
@rate_limit("5 per hour")
@validate_json(['username', 'email', 'password'])
def register():
    """User registration"""
    try:
        auth_service = get_auth_service()
        if not auth_service:
            return jsonify({'error': 'Auth service not initialized'}), 503
        
        data = request.get_json()
        username = data['username']
        email = data['email']
        password = data['password']
        
        # Create user
        user = auth_service.create_user(
            username=username,
            email=email,
            password=password
        )
        
        if not user:
            raise ValidationError("Failed to create user")
        
        return jsonify({
            'status': 'success',
            'message': 'Registration successful',
            'user': {
                'id': user.id,
                'username': user.username,
                'email': user.email,
                'role': user.role.value
            }
        })
        
    except ValidationError as e:
        return jsonify({'status': 'error', 'error': str(e)}), 400
    except Exception as e:
        current_app.logger.error(f"Registration error: {e}")
        return jsonify({'status': 'error', 'error': 'Registration failed'}), 500

@auth_bp.route('/verify-token', methods=['POST'])
@rate_limit("100 per minute")
@validate_json(['token'])
def verify_token():
    """Verify JWT token"""
    try:
        auth_service = get_auth_service()
        if not auth_service:
            return jsonify({'error': 'Auth service not initialized'}), 503
        
        data = request.get_json()
        token = data['token']
        
        # Verify token
        payload = auth_service.verify_jwt_token(token)
        if not payload:
            raise AuthenticationError("Invalid or expired token")
        
        return jsonify({
            'status': 'success',
            'valid': True,
            'payload': payload
        })
        
    except AuthenticationError as e:
        return jsonify({'status': 'error', 'valid': False, 'error': str(e)}), 401
    except Exception as e:
        current_app.logger.error(f"Token verification error: {e}")
        return jsonify({'status': 'error', 'valid': False, 'error': 'Token verification failed'}), 500

@auth_bp.route('/change-password', methods=['POST'])
@require_auth()
@rate_limit("5 per hour")
@validate_json(['current_password', 'new_password'])
def change_password():
    """Change user password"""
    try:
        auth_service = get_auth_service()
        if not auth_service:
            return jsonify({'error': 'Auth service not initialized'}), 503
        
        data = request.get_json()
        current_password = data['current_password']
        new_password = data['new_password']
        user_id = session.get('user_id')
        
        # Change password
        success = auth_service.change_password(user_id, current_password, new_password)
        if not success:
            raise AuthenticationError("Invalid current password")
        
        return jsonify({
            'status': 'success',
            'message': 'Password changed successfully'
        })
        
    except AuthenticationError as e:
        return jsonify({'status': 'error', 'error': str(e)}), 401
    except Exception as e:
        current_app.logger.error(f"Password change error: {e}")
        return jsonify({'status': 'error', 'error': 'Password change failed'}), 500
