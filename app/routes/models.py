"""
Model Management Routes
Complete implementation for AI model registration, configuration, and lifecycle management
Matches all frontend API calls and backend functionality
"""

from flask import Blueprint, request, jsonify, current_app, session
from datetime import datetime
import logging
import json
import uuid
import os

from ..services.ai_models import get_ai_model_manager
from ..utils.decorators import rate_limit, validate_json, require_auth
from ..utils.exceptions import ValidationError, ServiceError

# Create blueprint
models_bp = Blueprint('models', __name__)

# Set up logging
logger = logging.getLogger(__name__)

# ============================================================================
# STANDARD MODEL MANAGEMENT ENDPOINTS
# ============================================================================

@models_bp.route('/', methods=['GET'])
@rate_limit("100 per minute")
def get_models():
    """Get all models - Frontend: GET /api/models"""
    try:
        model_manager = get_ai_model_manager()
        if not model_manager:
            return jsonify({'error': 'Model manager not initialized'}), 503
        
        # Get filter parameters
        provider_filter = request.args.get('provider', None)
        active_only = request.args.get('active_only', 'false').lower() == 'true'
        page = request.args.get('page', 1, type=int)
        per_page = min(request.args.get('per_page', 50, type=int), 100)
        
        models = model_manager.get_all_models()
        
        # Apply filters
        if provider_filter:
            models = [m for m in models if m.provider == provider_filter]
        
        if active_only:
            models = [m for m in models if m.is_active]
        
        # Pagination
        total = len(models)
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        paginated_models = models[start_idx:end_idx]
        
        return jsonify({
            'status': 'success',
            'models': [model.to_dict() for model in paginated_models],
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total': total,
                'pages': (total + per_page - 1) // per_page
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting models: {e}")
        return jsonify({'status': 'error', 'error': 'Failed to get models'}), 500

@models_bp.route('/', methods=['POST'])
@require_auth()
@rate_limit("20 per hour")
@validate_json(['name', 'provider', 'model_id'])
def create_model():
    """Create a new model - Frontend: POST /api/models"""
    try:
        model_manager = get_ai_model_manager()
        if not model_manager:
            return jsonify({'error': 'Model manager not initialized'}), 503
        
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate required fields
        required_fields = ['name', 'provider', 'model_id']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Check if model with same name already exists
        existing_model = model_manager.get_model_by_name(data['name'])
        if existing_model:
            return jsonify({'error': 'Model with this name already exists'}), 409
        
        # Extract model data with all possible fields
        model_data = {
            'name': data['name'],
            'provider': data['provider'],
            'model_id': data['model_id'],
            'description': data.get('description', ''),
            'capabilities': data.get('capabilities', []),
            'parameters': data.get('parameters', {}),
            'is_active': data.get('is_active', False),
            'max_tokens': data.get('max_tokens', 4096),
            'temperature': data.get('temperature', 0.7),
            'top_p': data.get('top_p', 1.0),
            'frequency_penalty': data.get('frequency_penalty', 0.0),
            'presence_penalty': data.get('presence_penalty', 0.0),
            'cost_per_token': data.get('cost_per_token', 0.0),
            'endpoint_url': data.get('endpoint_url', ''),
            'api_key_name': data.get('api_key_name', ''),
            'tags': data.get('tags', []),
            'context_window': data.get('context_window', 4096),
            'supports_functions': data.get('supports_functions', False),
            'supports_streaming': data.get('supports_streaming', False),
            'created_by': session.get('user_id', 'unknown')
        }
        
        # Validate specific fields
        if model_data['temperature'] < 0 or model_data['temperature'] > 2:
            return jsonify({'error': 'Temperature must be between 0 and 2'}), 400
        
        if model_data['max_tokens'] < 1 or model_data['max_tokens'] > 100000:
            return jsonify({'error': 'Max tokens must be between 1 and 100000'}), 400
        
        # Create model
        model = model_manager.create_model(model_data)
        
        if model:
            return jsonify({
                'status': 'success',
                'message': 'Model created successfully',
                'model': model.to_dict()
            }), 201
        else:
            return jsonify({'error': 'Failed to create model'}), 500
            
    except ValidationError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Error creating model: {e}")
        return jsonify({'error': 'Failed to create model'}), 500
    
# app/routes/models.py - ADD THESE MISSING ENDPOINTS

@models_bp.route('/active', methods=['GET'])
@rate_limit("100 per minute")
def get_active_model():
    """Get currently active AI model"""
    try:
        model_manager = get_ai_model_manager()
        if not model_manager:
            return jsonify({'error': 'Model manager not initialized'}), 503
        
        active_model = model_manager.get_active_model()
        if not active_model:
            return jsonify({'status': 'error', 'error': 'No active model'}), 404
        
        return jsonify({
            'status': 'success',
            'model': active_model.to_dict()
        })
        
    except Exception as e:
        logger.error(f"Error getting active model: {e}")
        return jsonify({'error': 'Failed to get active model'}), 500
    
# app/routes/models.py - ADD THESE MISSING ENDPOINTS

@models_bp.route('/active', methods=['GET'])
@rate_limit("100 per minute")
def get_active_model():
    """Get currently active AI model - Frontend: GET /api/models/active"""
    try:
        model_manager = get_ai_model_manager()
        if not model_manager:
            return jsonify({'error': 'Model manager not initialized'}), 503
        
        active_model = model_manager.get_active_model()
        if not active_model:
            return jsonify({'status': 'error', 'error': 'No active model'}), 404
        
        return jsonify({
            'status': 'success',
            'model': active_model.to_dict()
        })
        
    except Exception as e:
        logger.error(f"Error getting active model: {e}")
        return jsonify({'error': 'Failed to get active model'}), 500

@models_bp.route('/api-key-status', methods=['GET'])
@rate_limit("100 per minute") 
def get_api_key_status():
    """Get API key status for all providers - Frontend: GET /api/models/api-key-status"""
    try:
        model_manager = get_ai_model_manager()
        if not model_manager:
            return jsonify({'error': 'Model manager not initialized'}), 503
        
        status_info = model_manager.get_api_key_status()
        
        return jsonify({
            'status': 'success',
            'status_info': status_info
        })
        
    except Exception as e:
        logger.error(f"Error getting API key status: {e}")
        return jsonify({'error': 'Failed to get API key status'}), 500

@models_bp.route('/activate/<model_id>', methods=['POST'])
@require_auth()
@rate_limit("50 per hour")
def activate_model(model_id):
    """Activate specific model - Frontend: POST /api/models/activate/{id}"""
    try:
        model_manager = get_ai_model_manager()
        if not model_manager:
            return jsonify({'error': 'Model manager not initialized'}), 503
        
        success = model_manager.activate_model(model_id)
        if not success:
            return jsonify({'error': 'Model not found'}), 404
        
        return jsonify({
            'status': 'success',
            'message': 'Model activated successfully'
        })
        
    except Exception as e:
        logger.error(f"Error activating model {model_id}: {e}")
        return jsonify({'error': 'Failed to activate model'}), 500

@models_bp.route('/test/<model_id>', methods=['POST'])
@require_auth()
@rate_limit("20 per hour")
def test_model(model_id):
    """Test specific model - Frontend: POST /api/models/test/{id}"""
    try:
        model_manager = get_ai_model_manager()
        if not model_manager:
            return jsonify({'error': 'Model manager not initialized'}), 503
        
        data = request.get_json() or {}
        test_query = data.get('query', 'Hello! Can you confirm you are working correctly?')
        
        result = model_manager.test_model(model_id, test_query)
        
        return jsonify({
            'status': 'success',
            'response': result.get('response', 'Test completed'),
            'model_id': model_id
        })
        
    except Exception as e:
        logger.error(f"Error testing model {model_id}: {e}")
        return jsonify({'error': 'Failed to test model'}), 500

@models_bp.route('/config', methods=['POST'])
@require_auth()
@rate_limit("50 per hour")
def save_model_config():
    """Save model configuration - Frontend: POST /api/models/config"""
    try:
        model_manager = get_ai_model_manager()
        if not model_manager:
            return jsonify({'error': 'Model manager not initialized'}), 503
        
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No configuration data provided'}), 400
        
        # Save configuration
        success = model_manager.save_configuration(data)
        
        if success:
            return jsonify({
                'status': 'success',
                'message': 'Configuration saved successfully'
            })
        else:
            return jsonify({'error': 'Failed to save configuration'}), 500
            
    except Exception as e:
        logger.error(f"Error saving model config: {e}")
        return jsonify({'error': 'Failed to save configuration'}), 500

@models_bp.route('/api-key-status', methods=['GET'])
@rate_limit("100 per minute") 
def get_api_key_status():
    """Get API key status for all providers"""
    try:
        model_manager = get_ai_model_manager()
        if not model_manager:
            return jsonify({'error': 'Model manager not initialized'}), 503
        
        status_info = model_manager.get_api_key_status()
        
        return jsonify({
            'status': 'success',
            'status_info': status_info
        })
        
    except Exception as e:
        logger.error(f"Error getting API key status: {e}")
        return jsonify({'error': 'Failed to get API key status'}), 500

@models_bp.route('/<model_id>', methods=['GET'])
@rate_limit("100 per minute")
def get_model(model_id):
    """Get specific model by ID - Frontend: GET /api/models/{id}"""
    try:
        model_manager = get_ai_model_manager()
        if not model_manager:
            return jsonify({'error': 'Model manager not initialized'}), 503
        
        model = model_manager.get_model_by_id(model_id)
        if not model:
            return jsonify({'error': 'Model not found'}), 404
        
        # Include additional stats if requested
        include_stats = request.args.get('include_stats', 'false').lower() == 'true'
        
        model_data = model.to_dict()
        
        if include_stats:
            try:
                stats = model_manager.get_model_stats(model_id)
                model_data['stats'] = stats
            except Exception as e:
                logger.warning(f"Could not get stats for model {model_id}: {e}")
                model_data['stats'] = None
        
        return jsonify({
            'status': 'success',
            'model': model_data
        })
        
    except Exception as e:
        logger.error(f"Error getting model {model_id}: {e}")
        return jsonify({'error': 'Failed to get model'}), 500

@models_bp.route('/<model_id>', methods=['PUT'])
@require_auth()
@rate_limit("50 per hour")
def update_model(model_id):
    """Update model configuration - Frontend: PUT /api/models/{id}"""
    try:
        model_manager = get_ai_model_manager()
        if not model_manager:
            return jsonify({'error': 'Model manager not initialized'}), 503
        
        model = model_manager.get_model_by_id(model_id)
        if not model:
            return jsonify({'error': 'Model not found'}), 404
        
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Updatable fields
        updatable_fields = [
            'name', 'description', 'capabilities', 'parameters', 'is_active',
            'max_tokens', 'temperature', 'top_p', 'frequency_penalty', 'presence_penalty',
            'cost_per_token', 'endpoint_url', 'api_key_name', 'tags', 'context_window',
            'supports_functions', 'supports_streaming'
        ]
        
        updated_data = {}
        for field in updatable_fields:
            if field in data:
                updated_data[field] = data[field]
        
        # Validate specific fields
        if 'temperature' in updated_data:
            temp = updated_data['temperature']
            if temp < 0 or temp > 2:
                return jsonify({'error': 'Temperature must be between 0 and 2'}), 400
        
        if 'max_tokens' in updated_data:
            tokens = updated_data['max_tokens']
            if tokens < 1 or tokens > 100000:
                return jsonify({'error': 'Max tokens must be between 1 and 100000'}), 400
        
        # Update model
        success = model_manager.update_model(model_id, updated_data)
        
        if success:
            updated_model = model_manager.get_model_by_id(model_id)
            return jsonify({
                'status': 'success',
                'message': 'Model updated successfully',
                'model': updated_model.to_dict()
            })
        else:
            return jsonify({'error': 'Failed to update model'}), 500
            
    except Exception as e:
        logger.error(f"Error updating model {model_id}: {e}")
        return jsonify({'error': 'Failed to update model'}), 500

@models_bp.route('/<model_id>', methods=['DELETE'])
@require_auth(roles=['admin'])
@rate_limit("20 per hour")
def delete_model(model_id):
    """Delete model - Frontend: DELETE /api/models/{id}"""
    try:
        model_manager = get_ai_model_manager()
        if not model_manager:
            return jsonify({'error': 'Model manager not initialized'}), 503
        
        model = model_manager.get_model_by_id(model_id)
        if not model:
            return jsonify({'error': 'Model not found'}), 404
        
        # Check if model is currently active
        active_model = model_manager.get_active_model()
        if active_model and active_model.id == model_id:
            return jsonify({'error': 'Cannot delete active model. Please activate another model first.'}), 409
        
        success = model_manager.delete_model(model_id)
        
        if success:
            return jsonify({
                'status': 'success',
                'message': 'Model deleted successfully'
            })
        else:
            return jsonify({'error': 'Failed to delete model'}), 500
            
    except Exception as e:
        logger.error(f"Error deleting model {model_id}: {e}")
        return jsonify({'error': 'Failed to delete model'}), 500

# ============================================================================
# MODEL LIFECYCLE MANAGEMENT
# ============================================================================

@models_bp.route('/<model_id>/activate', methods=['POST'])
@require_auth()
@rate_limit("30 per hour")
def activate_model(model_id):
    """Activate model - Frontend: POST /api/models/{id}/activate"""
    try:
        model_manager = get_ai_model_manager()
        if not model_manager:
            return jsonify({'error': 'Model manager not initialized'}), 503
        
        model = model_manager.get_model_by_id(model_id)
        if not model:
            return jsonify({'error': 'Model not found'}), 404
        
        success = model_manager.activate_model(model_id)
        
        if success:
            return jsonify({
                'status': 'success',
                'message': f'Model "{model.name}" activated successfully',
                'active_model': model.to_dict()
            })
        else:
            return jsonify({'error': 'Failed to activate model'}), 500
            
    except Exception as e:
        logger.error(f"Error activating model {model_id}: {e}")
        return jsonify({'error': 'Failed to activate model'}), 500

@models_bp.route('/<model_id>/deactivate', methods=['POST'])
@require_auth()
@rate_limit("30 per hour")
def deactivate_model(model_id):
    """Deactivate model - Frontend: POST /api/models/{id}/deactivate"""
    try:
        model_manager = get_ai_model_manager()
        if not model_manager:
            return jsonify({'error': 'Model manager not initialized'}), 503
        
        model = model_manager.get_model_by_id(model_id)
        if not model:
            return jsonify({'error': 'Model not found'}), 404
        
        success = model_manager.deactivate_model(model_id)
        
        if success:
            return jsonify({
                'status': 'success',
                'message': f'Model "{model.name}" deactivated successfully'
            })
        else:
            return jsonify({'error': 'Failed to deactivate model'}), 500
            
    except Exception as e:
        logger.error(f"Error deactivating model {model_id}: {e}")
        return jsonify({'error': 'Failed to deactivate model'}), 500

@models_bp.route('/<model_id>/test', methods=['POST'])
@require_auth()
@rate_limit("10 per hour")
def test_model(model_id):
    """Test model with sample query - Frontend: POST /api/models/{id}/test"""
    try:
        model_manager = get_ai_model_manager()
        if not model_manager:
            return jsonify({'error': 'Model manager not initialized'}), 503
        
        model = model_manager.get_model_by_id(model_id)
        if not model:
            return jsonify({'error': 'Model not found'}), 404
        
        data = request.get_json() or {}
        test_query = data.get('query', 'Hello, this is a test message. Please respond with a simple greeting.')
        test_params = data.get('parameters', {})
        
        # Test model with custom parameters
        result = model_manager.test_model(
            model_id=model_id, 
            test_query=test_query,
            parameters=test_params
        )
        
        if result['success']:
            return jsonify({
                'status': 'success',
                'message': 'Model test completed successfully',
                'test_result': {
                    'model_name': model.name,
                    'model_id': model_id,
                    'test_query': test_query,
                    'response': result.get('response', ''),
                    'response_time': result.get('response_time', 0),
                    'tokens_used': result.get('tokens_used', 0),
                    'cost_estimate': result.get('cost_estimate', 0),
                    'parameters_used': result.get('parameters_used', {}),
                    'success': True,
                    'timestamp': datetime.now().isoformat()
                }
            })
        else:
            return jsonify({
                'status': 'error',
                'error': 'Model test failed',
                'details': result.get('error', 'Unknown error'),
                'model_name': model.name,
                'test_query': test_query
            }), 500
            
    except Exception as e:
        logger.error(f"Error testing model {model_id}: {e}")
        return jsonify({'error': 'Failed to test model'}), 500

@models_bp.route('/<model_id>/train', methods=['POST'])
@require_auth()
@rate_limit("5 per hour")
def train_model(model_id):
    """Train/fine-tune model - Frontend: POST /api/models/{id}/train"""
    try:
        model_manager = get_ai_model_manager()
        if not model_manager:
            return jsonify({'error': 'Model manager not initialized'}), 503
        
        model = model_manager.get_model_by_id(model_id)
        if not model:
            return jsonify({'error': 'Model not found'}), 404
        
        data = request.get_json() or {}
        training_config = {
            'dataset': data.get('dataset', ''),
            'epochs': data.get('epochs', 1),
            'learning_rate': data.get('learning_rate', 0.001),
            'batch_size': data.get('batch_size', 32),
            'validation_split': data.get('validation_split', 0.2),
            'training_type': data.get('training_type', 'fine_tune'),
            'base_model': model.model_id,
            'provider': model.provider
        }
        
        # Validate training configuration
        if not training_config['dataset']:
            return jsonify({'error': 'Dataset is required for training'}), 400
        
        if training_config['epochs'] < 1 or training_config['epochs'] > 100:
            return jsonify({'error': 'Epochs must be between 1 and 100'}), 400
        
        # Start training job (mock implementation)
        training_job_id = f"training_{model_id}_{int(datetime.now().timestamp())}"
        
        return jsonify({
            'status': 'success',
            'message': 'Model training started',
            'training_job': {
                'id': training_job_id,
                'model_id': model_id,
                'model_name': model.name,
                'status': 'queued',
                'config': training_config,
                'started_at': datetime.now().isoformat(),
                'estimated_completion': None
            }
        })
        
    except Exception as e:
        logger.error(f"Error training model {model_id}: {e}")
        return jsonify({'error': 'Failed to start model training'}), 500

@models_bp.route('/<model_id>/stats', methods=['GET'])
@rate_limit("100 per minute")
def get_model_stats(model_id):
    """Get model performance statistics - Frontend: GET /api/models/{id}/stats"""
    try:
        model_manager = get_ai_model_manager()
        if not model_manager:
            return jsonify({'error': 'Model manager not initialized'}), 503
        
        model = model_manager.get_model_by_id(model_id)
        if not model:
            return jsonify({'error': 'Model not found'}), 404
        
        # Get comprehensive model statistics
        stats = model_manager.get_model_stats(model_id)
        
        # Add real-time performance metrics
        performance_stats = {
            'total_requests': stats.get('total_requests', 0),
            'successful_requests': stats.get('successful_requests', 0),
            'failed_requests': stats.get('failed_requests', 0),
            'average_response_time': stats.get('average_response_time', 0),
            'average_tokens_per_request': stats.get('average_tokens_per_request', 0),
            'total_tokens_processed': stats.get('total_tokens_processed', 0),
            'total_cost': stats.get('total_cost', 0),
            'success_rate': stats.get('success_rate', 0),
            'last_used': stats.get('last_used'),
            'usage_trend': stats.get('usage_trend', []),
            'error_distribution': stats.get('error_distribution', {}),
            'performance_score': stats.get('performance_score', 0)
        }
        
        return jsonify({
            'status': 'success',
            'model_id': model_id,
            'model_name': model.name,
            'stats': performance_stats,
            'generated_at': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting model stats {model_id}: {e}")
        return jsonify({'error': 'Failed to get model statistics'}), 500

# ============================================================================
# ACTIVE MODEL MANAGEMENT
# ============================================================================

@models_bp.route('/active', methods=['GET'])
@rate_limit("100 per minute")
def get_active_model():
    """Get currently active model - Frontend: GET /api/models/active"""
    try:
        model_manager = get_ai_model_manager()
        if not model_manager:
            return jsonify({'error': 'Model manager not initialized'}), 503
        
        active_model = model_manager.get_active_model()
        
        if active_model:
            return jsonify({
                'status': 'success',
                'active_model': active_model.to_dict(),
                'activated_at': active_model.updated_at.isoformat() if hasattr(active_model, 'updated_at') else None
            })
        else:
            return jsonify({
                'status': 'success',
                'active_model': None,
                'message': 'No active model configured'
            })
        
    except Exception as e:
        logger.error(f"Error getting active model: {e}")
        return jsonify({'error': 'Failed to get active model'}), 500

@models_bp.route('/active', methods=['POST'])
@require_auth()
@rate_limit("30 per hour")
@validate_json(['model_id'])
def set_active_model():
    """Set active model - Frontend: POST /api/models/active"""
    try:
        model_manager = get_ai_model_manager()
        if not model_manager:
            return jsonify({'error': 'Model manager not initialized'}), 503
        
        data = request.get_json()
        model_id = data['model_id']
        
        model = model_manager.get_model_by_id(model_id)
        if not model:
            return jsonify({'error': 'Model not found'}), 404
        
        success = model_manager.activate_model(model_id)
        
        if success:
            return jsonify({
                'status': 'success',
                'message': f'Model "{model.name}" is now active',
                'active_model': model.to_dict()
            })
        else:
            return jsonify({'error': 'Failed to set active model'}), 500
            
    except ValidationError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Error setting active model: {e}")
        return jsonify({'error': 'Failed to set active model'}), 500

# ============================================================================
# MODEL IMPORT/EXPORT
# ============================================================================

@models_bp.route('/import', methods=['POST'])
@require_auth()
@rate_limit("5 per hour")
def import_models():
    """Import models from JSON file - Frontend: POST /api/models/import"""
    try:
        model_manager = get_ai_model_manager()
        if not model_manager:
            return jsonify({'error': 'Model manager not initialized'}), 503
        
        # Check if file is uploaded
        if 'jsonFile' not in request.files:
            return jsonify({'status': 'error', 'error': 'No file uploaded'}), 400
        
        file = request.files['jsonFile']
        if file.filename == '':
            return jsonify({'status': 'error', 'error': 'No file selected'}), 400
        
        if not file.filename.endswith('.json'):
            return jsonify({'status': 'error', 'error': 'File must be a JSON file'}), 400
        
        # Check file size (max 10MB)
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        
        if file_size > 10 * 1024 * 1024:  # 10MB
            return jsonify({'status': 'error', 'error': 'File size too large. Maximum 10MB allowed.'}), 400
        
        try:
            # Parse JSON data
            models_data = json.load(file)
            
            if not isinstance(models_data, dict) or 'models' not in models_data:
                return jsonify({
                    'status': 'error', 
                    'error': 'Invalid JSON format. Expected format: {"models": [...]}'
                }), 400
            
            models_list = models_data['models']
            if not isinstance(models_list, list):
                return jsonify({'status': 'error', 'error': 'Models must be an array'}), 400
            
            if len(models_list) == 0:
                return jsonify({'status': 'warning', 'message': 'No models found in file'}), 400
            
            if len(models_list) > 100:
                return jsonify({'status': 'error', 'error': 'Too many models in file. Maximum 100 allowed.'}), 400
            
            # Import models
            imported_count = 0
            updated_count = 0
            skipped_count = 0
            errors = []
            
            for idx, model_data in enumerate(models_list):
                try:
                    # Validate required fields
                    required_fields = ['name', 'provider', 'model_id']
                    for field in required_fields:
                        if field not in model_data:
                            errors.append(f"Model #{idx+1}: Missing required field '{field}'")
                            continue
                    
                    # Check for valid provider
                    valid_providers = ['openai', 'anthropic', 'google', 'xai', 'perplexity', 'ollama', 'custom']
                    if model_data['provider'] not in valid_providers:
                        errors.append(f"Model #{idx+1} ({model_data['name']}): Invalid provider. Must be one of: {', '.join(valid_providers)}")
                        continue
                    
                    # Check if model already exists
                    existing_model = model_manager.get_model_by_name(model_data['name'])
                    
                    # Prepare complete model data
                    complete_model_data = {
                        'name': model_data['name'],
                        'provider': model_data['provider'],
                        'model_id': model_data['model_id'],
                        'description': model_data.get('description', ''),
                        'capabilities': model_data.get('capabilities', []),
                        'parameters': model_data.get('parameters', {}),
                        'is_active': model_data.get('is_active', False),
                        'max_tokens': model_data.get('max_tokens', 4096),
                        'temperature': model_data.get('temperature', 0.7),
                        'top_p': model_data.get('top_p', 1.0),
                        'frequency_penalty': model_data.get('frequency_penalty', 0.0),
                        'presence_penalty': model_data.get('presence_penalty', 0.0),
                        'cost_per_token': model_data.get('cost_per_token', 0.0),
                        'endpoint_url': model_data.get('endpoint_url', ''),
                        'api_key_name': model_data.get('api_key_name', ''),
                        'tags': model_data.get('tags', []),
                        'context_window': model_data.get('context_window', 4096),
                        'supports_functions': model_data.get('supports_functions', False),
                        'supports_streaming': model_data.get('supports_streaming', False),
                        'created_by': session.get('user_id', 'import')
                    }
                    
                    if existing_model:
                        # Update existing model
                        success = model_manager.update_model(existing_model.id, complete_model_data)
                        if success:
                            updated_count += 1
                        else:
                            errors.append(f"Model #{idx+1} ({model_data['name']}): Failed to update existing model")
                    else:
                        # Create new model
                        model = model_manager.create_model(complete_model_data)
                        if model:
                            imported_count += 1
                        else:
                            errors.append(f"Model #{idx+1} ({model_data['name']}): Failed to create model")
                        
                except Exception as e:
                    errors.append(f"Model #{idx+1} ({model_data.get('name', 'unknown')}): {str(e)}")
            
            # Build response
            total_processed = imported_count + updated_count
            status = 'success' if total_processed > 0 else 'error'
            
            if total_processed == len(models_list) and not errors:
                status = 'success'
            elif total_processed > 0:
                status = 'partial_success'
            
            response = {
                'status': status,
                'message': f'Import completed. {imported_count} new models imported, {updated_count} models updated, {skipped_count} skipped.',
                'summary': {
                    'total_models': len(models_list),
                    'imported_count': imported_count,
                    'updated_count': updated_count,
                    'skipped_count': skipped_count,
                    'error_count': len(errors)
                }
            }
            
            if errors:
                response['errors'] = errors[:10]  # Limit to first 10 errors
                if len(errors) > 10:
                    response['additional_errors'] = len(errors) - 10
            
            return jsonify(response)
            
        except json.JSONDecodeError as e:
            return jsonify({'status': 'error', 'error': f'Invalid JSON file: {str(e)}'}), 400
            
    except Exception as e:
        logger.error(f"Error importing models: {e}")
        return jsonify({'status': 'error', 'error': 'Failed to import models'}), 500

@models_bp.route('/export', methods=['GET'])
@require_auth()
@rate_limit("10 per hour")
def export_models():
    """Export all models to JSON - Frontend: GET /api/models/export"""
    try:
        model_manager = get_ai_model_manager()
        if not model_manager:
            return jsonify({'error': 'Model manager not initialized'}), 503
        
        # Get export options
        include_inactive = request.args.get('include_inactive', 'true').lower() == 'true'
        provider_filter = request.args.get('provider', None)
        
        models = model_manager.get_all_models()
        
        # Apply filters
        if not include_inactive:
            models = [m for m in models if m.is_active]
        
        if provider_filter:
            models = [m for m in models if m.provider == provider_filter]
        
        # Create export data
        export_data = {
            'metadata': {
                'export_date': datetime.now().isoformat(),
                'version': '1.0',
                'exported_by': session.get('user_id', 'unknown'),
                'total_models': len(models),
                'filters_applied': {
                    'include_inactive': include_inactive,
                    'provider_filter': provider_filter
                }
            },
            'models': [model.to_dict() for model in models]
        }
        
        return jsonify({
            'status': 'success',
            'export_data': export_data,
            'download_info': {
                'filename': f'models_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json',
                'size_estimate': f'{len(json.dumps(export_data))} bytes'
            }
        })
        
    except Exception as e:
        logger.error(f"Error exporting models: {e}")
        return jsonify({'error': 'Failed to export models'}), 500

# ============================================================================
# MODEL SEARCH AND FILTERING
# ============================================================================

@models_bp.route('/search', methods=['POST'])
@rate_limit("100 per minute")
@validate_json(['query'])
def search_models():
    """Search models - Frontend: POST /api/models/search"""
    try:
        model_manager = get_ai_model_manager()
        if not model_manager:
            return jsonify({'error': 'Model manager not initialized'}), 503
        
        data = request.get_json()
        query = data['query'].strip()
        
        if not query:
            return jsonify({'error': 'Search query cannot be empty'}), 400
        
        # Search parameters
        search_fields = data.get('fields', ['name', 'description', 'provider', 'tags'])
        limit = min(data.get('limit', 20), 100)
        
        models = model_manager.get_all_models()
        results = []
        
        # Simple text search across specified fields
        query_lower = query.lower()
        
        for model in models:
            score = 0
            matched_fields = []
            
            model_dict = model.to_dict()
            
            if 'name' in search_fields and query_lower in model_dict.get('name', '').lower():
                score += 10
                matched_fields.append('name')
            
            if 'description' in search_fields and query_lower in model_dict.get('description', '').lower():
                score += 5
                matched_fields.append('description')
            
            if 'provider' in search_fields and query_lower in model_dict.get('provider', '').lower():
                score += 8
                matched_fields.append('provider')
            
            if 'tags' in search_fields:
                for tag in model_dict.get('tags', []):
                    if query_lower in tag.lower():
                        score += 3
                        matched_fields.append('tags')
                        break
            
            if score > 0:
                results.append({
                    'model': model_dict,
                    'score': score,
                    'matched_fields': matched_fields
                })
        
        # Sort by score and limit results
        results.sort(key=lambda x: x['score'], reverse=True)
        results = results[:limit]
        
        return jsonify({
            'status': 'success',
            'query': query,
            'results': [r['model'] for r in results],
            'search_info': {
                'total_results': len(results),
                'search_fields': search_fields,
                'execution_time': 0  # Mock value
            }
        })
        
    except ValidationError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Error searching models: {e}")
        return jsonify({'error': 'Failed to search models'}), 500

@models_bp.route('/providers', methods=['GET'])
@rate_limit("100 per minute")
def get_providers():
    """Get list of available providers - Frontend: GET /api/models/providers"""
    try:
        model_manager = get_ai_model_manager()
        if not model_manager:
            return jsonify({'error': 'Model manager not initialized'}), 503
        
        models = model_manager.get_all_models()
        
        # Count models by provider
        provider_stats = {}
        for model in models:
            provider = model.provider
            if provider not in provider_stats:
                provider_stats[provider] = {
                    'name': provider,
                    'total_models': 0,
                    'active_models': 0
                }
            
            provider_stats[provider]['total_models'] += 1
            if model.is_active:
                provider_stats[provider]['active_models'] += 1
        
        return jsonify({
            'status': 'success',
            'providers': list(provider_stats.values())
        })
        
    except Exception as e:
        logger.error(f"Error getting providers: {e}")
        return jsonify({'error': 'Failed to get providers'}), 500

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@models_bp.errorhandler(ValidationError)
def handle_validation_error(error):
    """Handle validation errors"""
    return jsonify({'status': 'error', 'error': str(error)}), 400

@models_bp.errorhandler(ServiceError)
def handle_service_error(error):
    """Handle service errors"""
    return jsonify({'status': 'error', 'error': str(error)}), 500

@models_bp.errorhandler(429)
def handle_rate_limit_error(error):
    """Handle rate limit errors"""
    return jsonify({
        'status': 'error', 
        'error': 'Rate limit exceeded',
        'message': 'Too many requests. Please try again later.'
    }), 429

@models_bp.errorhandler(413)
def handle_file_too_large(error):
    """Handle file too large errors"""
    return jsonify({
        'status': 'error',
        'error': 'File too large',
        'message': 'The uploaded file exceeds the maximum allowed size.'
    }), 413