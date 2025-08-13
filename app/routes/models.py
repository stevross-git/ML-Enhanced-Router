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
@require_auth()
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
        return jsonify({'error': 'Failed to get models'}), 500

@models_bp.route('/', methods=['POST'])
@require_auth()
@rate_limit("50 per hour")
@validate_json(['name', 'provider', 'model_name'])
def create_model():
    """Create new model - Frontend: POST /api/models"""
    try:
        model_manager = get_ai_model_manager()
        if not model_manager:
            return jsonify({'error': 'Model manager not initialized'}), 503
        
        data = request.get_json()
        
        # Add metadata
        data['created_by'] = session.get('user_id', 'anonymous')
        data['created_at'] = datetime.utcnow().isoformat()
        
        model = model_manager.create_model(data)
        if not model:
            return jsonify({'error': 'Failed to create model'}), 500
        
        return jsonify({
            'status': 'success',
            'model': model.to_dict(),
            'message': 'Model created successfully'
        }), 201
        
    except ValidationError as e:
        return jsonify({'error': f'Validation error: {e}'}), 400
    except Exception as e:
        logger.error(f"Error creating model: {e}")
        return jsonify({'error': 'Failed to create model'}), 500

# ============================================================================
# ACTIVE MODEL MANAGEMENT
# ============================================================================

@models_bp.route('/active', methods=['GET'])
@require_auth()
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
        if not success:
            return jsonify({'error': 'Failed to activate model'}), 500
        
        return jsonify({
            'status': 'success',
            'active_model': model.to_dict(),
            'message': f'Model {model.name} activated successfully'
        })
        
    except Exception as e:
        logger.error(f"Error setting active model: {e}")
        return jsonify({'error': 'Failed to set active model'}), 500

# ============================================================================
# MODEL OPERATIONS
# ============================================================================

@models_bp.route('/api-key-status', methods=['GET'])
@require_auth()
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

@models_bp.route('/<model_id>', methods=['GET'])
@require_auth()
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
    """Update model - Frontend: PUT /api/models/{id}"""
    try:
        model_manager = get_ai_model_manager()
        if not model_manager:
            return jsonify({'error': 'Model manager not initialized'}), 503
        
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No update data provided'}), 400
        
        # Add update metadata
        data['updated_by'] = session.get('user_id', 'anonymous')
        data['updated_at'] = datetime.utcnow().isoformat()
        
        success = model_manager.update_model(model_id, data)
        if not success:
            return jsonify({'error': 'Model not found or update failed'}), 404
        
        updated_model = model_manager.get_model_by_id(model_id)
        
        return jsonify({
            'status': 'success',
            'model': updated_model.to_dict(),
            'message': 'Model updated successfully'
        })
        
    except ValidationError as e:
        return jsonify({'error': f'Validation error: {e}'}), 400
    except Exception as e:
        logger.error(f"Error updating model {model_id}: {e}")
        return jsonify({'error': 'Failed to update model'}), 500

@models_bp.route('/<model_id>', methods=['DELETE'])
@require_auth()
@rate_limit("30 per hour")
def delete_model(model_id):
    """Delete model - Frontend: DELETE /api/models/{id}"""
    try:
        model_manager = get_ai_model_manager()
        if not model_manager:
            return jsonify({'error': 'Model manager not initialized'}), 503
        
        success = model_manager.delete_model(model_id)
        if not success:
            return jsonify({'error': 'Model not found'}), 404
        
        return jsonify({
            'status': 'success',
            'message': 'Model deleted successfully'
        })
        
    except Exception as e:
        logger.error(f"Error deleting model {model_id}: {e}")
        return jsonify({'error': 'Failed to delete model'}), 500

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

# ============================================================================
# MODEL CONFIGURATION AND MANAGEMENT  
# ============================================================================

@models_bp.route('/providers', methods=['GET'])
@rate_limit("100 per minute")
def get_providers():
    """Get available AI providers - Frontend: GET /api/models/providers"""
    try:
        model_manager = get_ai_model_manager()
        if not model_manager:
            return jsonify({'error': 'Model manager not initialized'}), 503
        
        # Get unique providers from all models
        models = model_manager.get_all_models()
        providers = []
        
        for model in models:
            provider_info = {
                'name': model.provider.value if hasattr(model.provider, 'value') else str(model.provider),
                'display_name': model.provider.value.title() if hasattr(model.provider, 'value') else str(model.provider).title()
            }
            if provider_info not in providers:
                providers.append(provider_info)
        
        return jsonify({
            'status': 'success',
            'providers': providers
        })
        
    except Exception as e:
        logger.error(f"Error getting providers: {e}")
        return jsonify({'error': 'Failed to get providers'}), 500

@models_bp.route('/capabilities', methods=['GET'])
@rate_limit("100 per minute")
def get_capabilities():
    """Get available model capabilities - Frontend: GET /api/models/capabilities"""
    try:
        model_manager = get_ai_model_manager()
        if not model_manager:
            return jsonify({'error': 'Model manager not initialized'}), 503
        
        # Get unique capabilities from all models
        models = model_manager.get_all_models()
        capabilities = set()
        
        for model in models:
            if hasattr(model, 'capabilities'):
                for cap in model.capabilities:
                    cap_name = cap.value if hasattr(cap, 'value') else str(cap)
                    capabilities.add(cap_name)
        
        return jsonify({
            'status': 'success',
            'capabilities': list(capabilities)
        })
        
    except Exception as e:
        logger.error(f"Error getting capabilities: {e}")
        return jsonify({'error': 'Failed to get capabilities'}), 500

@models_bp.route('/by-provider/<provider>', methods=['GET'])
@rate_limit("100 per minute")
def get_models_by_provider(provider):
    """Get models by provider - Frontend: GET /api/models/by-provider/{provider}"""
    try:
        model_manager = get_ai_model_manager()
        if not model_manager:
            return jsonify({'error': 'Model manager not initialized'}), 503
        
        models = model_manager.get_models_by_provider(provider)
        
        return jsonify({
            'status': 'success',
            'models': [model.to_dict() if hasattr(model, 'to_dict') else {
                'id': model.id,
                'name': model.name,
                'provider': model.provider.value if hasattr(model.provider, 'value') else str(model.provider)
            } for model in models],
            'provider': provider,
            'count': len(models)
        })
        
    except Exception as e:
        logger.error(f"Error getting models by provider {provider}: {e}")
        return jsonify({'error': 'Failed to get models by provider'}), 500

@models_bp.route('/by-capability/<capability>', methods=['GET'])
@rate_limit("100 per minute")
def get_models_by_capability(capability):
    """Get models by capability - Frontend: GET /api/models/by-capability/{capability}"""
    try:
        model_manager = get_ai_model_manager()
        if not model_manager:
            return jsonify({'error': 'Model manager not initialized'}), 503
        
        models = model_manager.get_models_by_capability(capability)
        
        return jsonify({
            'status': 'success',
            'models': [model.to_dict() if hasattr(model, 'to_dict') else {
                'id': model.id,
                'name': model.name,
                'provider': model.provider.value if hasattr(model.provider, 'value') else str(model.provider)
            } for model in models],
            'capability': capability,
            'count': len(models)
        })
        
    except Exception as e:
        logger.error(f"Error getting models by capability {capability}: {e}")
        return jsonify({'error': 'Failed to get models by capability'}), 500

@models_bp.route('/custom', methods=['POST'])
@require_auth()
@rate_limit("10 per hour")
@validate_json(['model_id', 'name', 'endpoint'])
def add_custom_model():
    """Add custom model - Frontend: POST /api/models/custom"""
    try:
        model_manager = get_ai_model_manager()
        if not model_manager:
            return jsonify({'error': 'Model manager not initialized'}), 503
        
        data = request.get_json()
        
        model = model_manager.add_custom_model(
            model_id=data['model_id'],
            name=data['name'],
            endpoint=data['endpoint'],
            api_key_env=data.get('api_key_env', ''),
            model_name=data.get('model_name', data['model_id']),
            max_tokens=data.get('max_tokens', 4096),
            temperature=data.get('temperature', 0.7),
            custom_headers=data.get('custom_headers', {})
        )
        
        if model:
            return jsonify({
                'status': 'success',
                'model': model.to_dict() if hasattr(model, 'to_dict') else {
                    'id': model.id,
                    'name': model.name
                },
                'message': 'Custom model added successfully'
            }), 201
        else:
            return jsonify({'error': 'Failed to add custom model'}), 500
        
    except Exception as e:
        logger.error(f"Error adding custom model: {e}")
        return jsonify({'error': 'Failed to add custom model'}), 500

# ============================================================================
# BATCH OPERATIONS
# ============================================================================

@models_bp.route('/batch/activate', methods=['POST'])
@require_auth()
@rate_limit("30 per hour")
@validate_json(['model_ids'])
def batch_activate_models():
    """Batch activate models - Frontend: POST /api/models/batch/activate"""
    try:
        model_manager = get_ai_model_manager()
        if not model_manager:
            return jsonify({'error': 'Model manager not initialized'}), 503
        
        data = request.get_json()
        model_ids = data['model_ids']
        
        results = model_manager.batch_activate_models(model_ids)
        
        return jsonify({
            'status': 'success',
            'results': results,
            'message': f'Batch activation completed'
        })
        
    except Exception as e:
        logger.error(f"Error in batch activate: {e}")
        return jsonify({'error': 'Failed to batch activate models'}), 500

@models_bp.route('/batch/deactivate', methods=['POST'])
@require_auth()
@rate_limit("30 per hour")
@validate_json(['model_ids'])
def batch_deactivate_models():
    """Batch deactivate models - Frontend: POST /api/models/batch/deactivate"""
    try:
        model_manager = get_ai_model_manager()
        if not model_manager:
            return jsonify({'error': 'Model manager not initialized'}), 503
        
        data = request.get_json()
        model_ids = data['model_ids']
        
        results = model_manager.batch_deactivate_models(model_ids)
        
        return jsonify({
            'status': 'success',
            'results': results,
            'message': f'Batch deactivation completed'
        })
        
    except Exception as e:
        logger.error(f"Error in batch deactivate: {e}")
        return jsonify({'error': 'Failed to batch deactivate models'}), 500

@models_bp.route('/batch/test', methods=['POST'])
@require_auth()
@rate_limit("10 per hour")
@validate_json(['model_ids'])
def batch_test_models():
    """Batch test models - Frontend: POST /api/models/batch/test"""
    try:
        model_manager = get_ai_model_manager()
        if not model_manager:
            return jsonify({'error': 'Model manager not initialized'}), 503
        
        data = request.get_json()
        model_ids = data['model_ids']
        test_query = data.get('query', 'Hello! Can you confirm you are working correctly?')
        
        results = model_manager.batch_test_models(model_ids, test_query)
        
        return jsonify({
            'status': 'success',
            'results': results,
            'message': f'Batch testing completed'
        })
        
    except Exception as e:
        logger.error(f"Error in batch test: {e}")
        return jsonify({'error': 'Failed to batch test models'}), 500

# ============================================================================
# IMPORT/EXPORT OPERATIONS
# ============================================================================

@models_bp.route('/export', methods=['GET'])
@require_auth()
@rate_limit("10 per hour")
def export_models():
    """Export models configuration - Frontend: GET /api/models/export"""
    try:
        model_manager = get_ai_model_manager()
        if not model_manager:
            return jsonify({'error': 'Model manager not initialized'}), 503
        
        export_format = request.args.get('format', 'json')
        include_inactive = request.args.get('include_inactive', 'false').lower() == 'true'
        
        exported_data = model_manager.export_models(
            format=export_format,
            include_inactive=include_inactive
        )
        
        return jsonify({
            'status': 'success',
            'data': exported_data,
            'export_time': datetime.utcnow().isoformat(),
            'format': export_format
        })
        
    except Exception as e:
        logger.error(f"Error exporting models: {e}")
        return jsonify({'error': 'Failed to export models'}), 500

@models_bp.route('/import', methods=['POST'])
@require_auth()
@rate_limit("5 per hour")
def import_models():
    """Import models configuration - Frontend: POST /api/models/import"""
    try:
        model_manager = get_ai_model_manager()
        if not model_manager:
            return jsonify({'error': 'Model manager not initialized'}), 503
        
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No import data provided'}), 400
        
        models_data = data.get('models', [])
        overwrite_existing = data.get('overwrite_existing', False)
        
        if not models_data:
            return jsonify({'error': 'No models data found in import'}), 400
        
        results = model_manager.import_models(models_data, overwrite_existing)
        
        return jsonify({
            'status': 'success',
            'results': results,
            'message': f'Import completed. {results.get("imported", 0)} models imported, {results.get("skipped", 0)} skipped'
        })
        
    except Exception as e:
        logger.error(f"Error importing models: {e}")
        return jsonify({'error': 'Failed to import models'}), 500

# ============================================================================
# ADVANCED MODEL OPERATIONS
# ============================================================================

@models_bp.route('/sync', methods=['POST'])
@require_auth()
@rate_limit("10 per hour")
def sync_models():
    """Sync models with provider APIs - Frontend: POST /api/models/sync"""
    try:
        model_manager = get_ai_model_manager()
        if not model_manager:
            return jsonify({'error': 'Model manager not initialized'}), 503
        
        data = request.get_json() or {}
        provider = data.get('provider')  # Sync specific provider or all
        
        results = model_manager.sync_models(provider=provider)
        
        return jsonify({
            'status': 'success',
            'results': results,
            'sync_time': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error syncing models: {e}")
        return jsonify({'error': 'Failed to sync models'}), 500

@models_bp.route('/validate', methods=['POST'])
@require_auth()
@rate_limit("20 per hour")
def validate_models():
    """Validate model configurations - Frontend: POST /api/models/validate"""
    try:
        model_manager = get_ai_model_manager()
        if not model_manager:
            return jsonify({'error': 'Model manager not initialized'}), 503
        
        data = request.get_json() or {}
        model_ids = data.get('model_ids')  # Validate specific models or all
        
        validation_results = model_manager.validate_models(model_ids=model_ids)
        
        return jsonify({
            'status': 'success',
            'validation_results': validation_results,
            'validation_time': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error validating models: {e}")
        return jsonify({'error': 'Failed to validate models'}), 500

@models_bp.route('/reset', methods=['POST'])
@require_auth()
@rate_limit("3 per hour")
def reset_models():
    """Reset models to default configuration - Frontend: POST /api/models/reset"""
    try:
        model_manager = get_ai_model_manager()
        if not model_manager:
            return jsonify({'error': 'Model manager not initialized'}), 503
        
        data = request.get_json() or {}
        confirm = data.get('confirm', False)
        
        if not confirm:
            return jsonify({'error': 'Reset operation requires confirmation'}), 400
        
        results = model_manager.reset_to_defaults()
        
        return jsonify({
            'status': 'success',
            'results': results,
            'message': 'Models reset to default configuration',
            'reset_time': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error resetting models: {e}")
        return jsonify({'error': 'Failed to reset models'}), 500

# ============================================================================
# STREAMING AND REAL-TIME OPERATIONS
# ============================================================================

@models_bp.route('/generate', methods=['POST'])
@require_auth()
@rate_limit("100 per hour")
@validate_json(['model_id', 'query'])
def generate_response():
    """Generate response from model - Frontend: POST /api/models/generate"""
    try:
        model_manager = get_ai_model_manager()
        if not model_manager:
            return jsonify({'error': 'Model manager not initialized'}), 503
        
        data = request.get_json()
        model_id = data['model_id']
        query = data['query']
        system_message = data.get('system_message', '')
        stream = data.get('stream', False)
        
        # Use async generate_response method
        import asyncio
        
        async def async_generate():
            return await model_manager.generate_response(
                model_id=model_id,
                query=query,
                system_message=system_message,
                user_id=session.get('user_id', 'anonymous'),
                stream=stream
            )
        
        # Run async function
        result = asyncio.run(async_generate())
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return jsonify({'error': 'Failed to generate response'}), 500

@models_bp.route('/stream/<model_id>', methods=['POST'])
@require_auth()
@rate_limit("100 per hour")
def stream_response(model_id):
    """Stream response from model - Frontend: POST /api/models/stream/{id}"""
    try:
        model_manager = get_ai_model_manager()
        if not model_manager:
            return jsonify({'error': 'Model manager not initialized'}), 503
        
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No request data provided'}), 400
        
        query = data.get('query', '')
        system_message = data.get('system_message', '')
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        import asyncio
        
        async def async_stream():
            return await model_manager.generate_response(
                model_id=model_id,
                query=query,
                system_message=system_message,
                user_id=session.get('user_id', 'anonymous'),
                stream=True
            )
        
        result = asyncio.run(async_stream())
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error streaming response: {e}")
        return jsonify({'error': 'Failed to stream response'}), 500

# ============================================================================
# HEALTH AND MONITORING
# ============================================================================

@models_bp.route('/health', methods=['GET'])
@rate_limit("200 per minute")
def models_health_check():
    """Health check for models system - Frontend: GET /api/models/health"""
    try:
        model_manager = get_ai_model_manager()
        if not model_manager:
            return jsonify({
                'status': 'unhealthy',
                'error': 'Model manager not initialized'
            }), 503
        
        # Basic health check
        models = model_manager.get_all_models()
        active_model = model_manager.get_active_model()
        
        health_status = {
            'overall_status': 'healthy',
            'total_models': len(models),
            'active_model': active_model.id if active_model else None,
            'model_manager_initialized': True,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        return jsonify({
            'status': 'success',
            'health': health_status
        })
        
    except Exception as e:
        logger.error(f"Error in health check: {e}")
        return jsonify({
            'status': 'unhealthy',
            'error': 'Health check failed',
            'timestamp': datetime.utcnow().isoformat()
        }), 503

# ============================================================================
# SIMPLIFIED BATCH OPERATIONS
# ============================================================================

@models_bp.route('/batch/status', methods=['POST'])
@require_auth()
@rate_limit("30 per hour")
@validate_json(['model_ids'])
def batch_check_status():
    """Check status of multiple models - Frontend: POST /api/models/batch/status"""
    try:
        model_manager = get_ai_model_manager()
        if not model_manager:
            return jsonify({'error': 'Model manager not initialized'}), 503
        
        data = request.get_json()
        model_ids = data['model_ids']
        
        results = {}
        for model_id in model_ids:
            model = model_manager.get_model(model_id)
            if model:
                results[model_id] = {
                    'available': True,
                    'name': model.name,
                    'provider': model.provider.value if hasattr(model.provider, 'value') else str(model.provider),
                    'is_active': getattr(model, 'is_active', False)
                }
            else:
                results[model_id] = {
                    'available': False,
                    'error': 'Model not found'
                }
        
        return jsonify({
            'status': 'success',
            'results': results,
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in batch status check: {e}")
        return jsonify({'error': 'Failed to check model status'}), 500

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

@models_bp.route('/validate/<model_id>', methods=['POST'])
@require_auth()
@rate_limit("20 per hour")
def validate_model(model_id):
    """Validate model configuration - Frontend: POST /api/models/validate/{id}"""
    try:
        model_manager = get_ai_model_manager()
        if not model_manager:
            return jsonify({'error': 'Model manager not initialized'}), 503
        
        model = model_manager.get_model(model_id)
        if not model:
            return jsonify({'error': 'Model not found'}), 404
        
        # Basic validation
        validation_result = {
            'model_id': model_id,
            'valid': True,
            'issues': [],
            'details': {
                'has_endpoint': bool(getattr(model, 'endpoint', None)),
                'has_api_key': bool(getattr(model, 'api_key_env', None)),
                'provider_valid': bool(getattr(model, 'provider', None))
            }
        }
        
        # Check for issues
        if not getattr(model, 'endpoint', None):
            validation_result['issues'].append('Missing endpoint')
            validation_result['valid'] = False
        
        if not getattr(model, 'api_key_env', None) and model.provider.value != 'ollama':
            validation_result['issues'].append('Missing API key configuration')
            validation_result['valid'] = False
        
        return jsonify({
            'status': 'success',
            'validation': validation_result
        })
        
    except Exception as e:
        logger.error(f"Error validating model {model_id}: {e}")
        return jsonify({'error': 'Failed to validate model'}), 500

@models_bp.route('/info', methods=['GET'])
@rate_limit("100 per minute")
def get_system_info():
    """Get system information - Frontend: GET /api/models/info"""
    try:
        model_manager = get_ai_model_manager()
        if not model_manager:
            return jsonify({'error': 'Model manager not initialized'}), 503
        
        models = model_manager.get_all_models()
        active_model = model_manager.get_active_model()
        
        # Count models by provider
        provider_counts = {}
        for model in models:
            provider = model.provider.value if hasattr(model.provider, 'value') else str(model.provider)
            provider_counts[provider] = provider_counts.get(provider, 0) + 1
        
        system_info = {
            'total_models': len(models),
            'active_model': {
                'id': active_model.id,
                'name': active_model.name,
                'provider': active_model.provider.value if hasattr(active_model.provider, 'value') else str(active_model.provider)
            } if active_model else None,
            'provider_distribution': provider_counts,
            'system_status': 'operational',
            'manager_type': type(model_manager).__name__,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        return jsonify({
            'status': 'success',
            'info': system_info
        })
        
    except Exception as e:
        logger.error(f"Error getting system info: {e}")
        return jsonify({'error': 'Failed to get system information'}), 500