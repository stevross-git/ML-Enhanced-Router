"""
Model Management Routes
API endpoints for managing AI models and configurations
"""

from flask import Blueprint, request, jsonify, current_app
from datetime import datetime

from ..services.ai_models import get_ai_model_manager
from ..utils.decorators import rate_limit, require_auth, validate_json
from ..utils.validators import validate_model_config
from ..utils.exceptions import ValidationError, ServiceError, ModelError

# Create blueprint
models_bp = Blueprint('models', __name__)

@models_bp.route('/', methods=['GET'])
@rate_limit("200 per minute")
def get_models():
    """Get all AI models"""
    try:
        ai_manager = get_ai_model_manager()
        if not ai_manager:
            return jsonify({'error': 'AI model manager not available'}), 503
        
        models = ai_manager.get_all_models()
        models_data = []
        
        for model in models:
            models_data.append({
                'id': model.id,
                'name': model.name,
                'provider': model.provider.value if hasattr(model.provider, 'value') else str(model.provider),
                'model_name': model.model_name,
                'endpoint': model.endpoint,
                'is_active': model.is_active,
                'categories': model.categories,
                'max_tokens': model.max_tokens,
                'temperature': model.temperature,
                'created_at': model.created_at.isoformat() if model.created_at else None,
                'last_used': model.last_used.isoformat() if model.last_used else None,
                'usage_count': model.usage_count,
                'avg_response_time': model.avg_response_time,
                'total_tokens': model.total_tokens,
                'total_cost': model.total_cost
            })
        
        return jsonify({
            'models': models_data,
            'total': len(models_data),
            'active_count': len([m for m in models_data if m['is_active']])
        })
        
    except Exception as e:
        current_app.logger.error(f"Error getting models: {e}")
        return jsonify({'error': 'Failed to retrieve models'}), 500

@models_bp.route('/', methods=['POST'])
@rate_limit("50 per minute")
@require_auth(roles=['admin'])
@validate_json(['name', 'provider', 'model_name'])
def create_model():
    """Create a new AI model"""
    try:
        ai_manager = get_ai_model_manager()
        if not ai_manager:
            return jsonify({'error': 'AI model manager not available'}), 503
        
        data = request.get_json()
        
        # Validate model configuration
        validation_errors = validate_model_config(data)
        if validation_errors:
            return jsonify({
                'error': 'Validation failed',
                'details': validation_errors
            }), 400
        
        # Create model
        model = ai_manager.create_model(
            name=data['name'],
            provider=data['provider'],
            model_name=data['model_name'],
            endpoint=data.get('endpoint'),
            api_key_env=data.get('api_key_env'),
            categories=data.get('categories', []),
            max_tokens=data.get('max_tokens', 4000),
            temperature=data.get('temperature', 0.7),
            custom_headers=data.get('custom_headers', {}),
            description=data.get('description', ''),
            is_active=data.get('is_active', False)
        )
        
        return jsonify({
            'message': 'Model created successfully',
            'model': {
                'id': model.id,
                'name': model.name,
                'provider': model.provider.value if hasattr(model.provider, 'value') else str(model.provider),
                'is_active': model.is_active
            }
        }), 201
        
    except ValidationError as e:
        return jsonify({'error': str(e)}), 400
    except ModelError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        current_app.logger.error(f"Error creating model: {e}")
        return jsonify({'error': 'Failed to create model'}), 500

@models_bp.route('/<model_id>', methods=['GET'])
@rate_limit("200 per minute")
def get_model(model_id):
    """Get specific model details"""
    try:
        ai_manager = get_ai_model_manager()
        if not ai_manager:
            return jsonify({'error': 'AI model manager not available'}), 503
        
        model = ai_manager.get_model(model_id)
        if not model:
            return jsonify({'error': 'Model not found'}), 404
        
        return jsonify({
            'model': {
                'id': model.id,
                'name': model.name,
                'provider': model.provider.value if hasattr(model.provider, 'value') else str(model.provider),
                'model_name': model.model_name,
                'endpoint': model.endpoint,
                'is_active': model.is_active,
                'categories': model.categories,
                'max_tokens': model.max_tokens,
                'temperature': model.temperature,
                'description': model.description,
                'created_at': model.created_at.isoformat() if model.created_at else None,
                'updated_at': model.updated_at.isoformat() if model.updated_at else None,
                'last_used': model.last_used.isoformat() if model.last_used else None,
                'usage_count': model.usage_count,
                'avg_response_time': model.avg_response_time,
                'total_tokens': model.total_tokens,
                'total_cost': model.total_cost,
                'custom_headers': model.custom_headers,
                'metadata': model.metadata
            }
        })
        
    except Exception as e:
        current_app.logger.error(f"Error getting model {model_id}: {e}")
        return jsonify({'error': 'Failed to retrieve model'}), 500

@models_bp.route('/<model_id>', methods=['PUT'])
@rate_limit("50 per minute")
@require_auth(roles=['admin'])
@validate_json()
def update_model(model_id):
    """Update model configuration"""
    try:
        ai_manager = get_ai_model_manager()
        if not ai_manager:
            return jsonify({'error': 'AI model manager not available'}), 503
        
        data = request.get_json()
        
        # Update model
        success = ai_manager.update_model(model_id, data)
        if not success:
            return jsonify({'error': 'Model not found or update failed'}), 404
        
        return jsonify({'message': 'Model updated successfully'})
        
    except ValidationError as e:
        return jsonify({'error': str(e)}), 400
    except ModelError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        current_app.logger.error(f"Error updating model {model_id}: {e}")
        return jsonify({'error': 'Failed to update model'}), 500

@models_bp.route('/<model_id>', methods=['DELETE'])
@rate_limit("30 per minute")
@require_auth(roles=['admin'])
def delete_model(model_id):
    """Delete a model"""
    try:
        ai_manager = get_ai_model_manager()
        if not ai_manager:
            return jsonify({'error': 'AI model manager not available'}), 503
        
        success = ai_manager.delete_model(model_id)
        if not success:
            return jsonify({'error': 'Model not found or deletion failed'}), 404
        
        return jsonify({'message': 'Model deleted successfully'})
        
    except Exception as e:
        current_app.logger.error(f"Error deleting model {model_id}: {e}")
        return jsonify({'error': 'Failed to delete model'}), 500

@models_bp.route('/<model_id>/activate', methods=['POST'])
@rate_limit("100 per minute")
@require_auth(roles=['admin', 'user'])
def activate_model(model_id):
    """Activate a model"""
    try:
        ai_manager = get_ai_model_manager()
        if not ai_manager:
            return jsonify({'error': 'AI model manager not available'}), 503
        
        success = ai_manager.activate_model(model_id)
        if not success:
            return jsonify({'error': 'Model not found or activation failed'}), 404
        
        return jsonify({'message': f'Model {model_id} activated successfully'})
        
    except Exception as e:
        current_app.logger.error(f"Error activating model {model_id}: {e}")
        return jsonify({'error': 'Failed to activate model'}), 500

@models_bp.route('/<model_id>/deactivate', methods=['POST'])
@rate_limit("100 per minute")
@require_auth(roles=['admin', 'user'])
def deactivate_model(model_id):
    """Deactivate a model"""
    try:
        ai_manager = get_ai_model_manager()
        if not ai_manager:
            return jsonify({'error': 'AI model manager not available'}), 503
        
        success = ai_manager.deactivate_model(model_id)
        if not success:
            return jsonify({'error': 'Model not found or deactivation failed'}), 404
        
        return jsonify({'message': f'Model {model_id} deactivated successfully'})
        
    except Exception as e:
        current_app.logger.error(f"Error deactivating model {model_id}: {e}")
        return jsonify({'error': 'Failed to deactivate model'}), 500

@models_bp.route('/<model_id>/test', methods=['POST'])
@rate_limit("20 per minute")
@require_auth(roles=['admin'])
@validate_json(['query'])
def test_model(model_id):
    """Test a model with a query"""
    try:
        ai_manager = get_ai_model_manager()
        if not ai_manager:
            return jsonify({'error': 'AI model manager not available'}), 503
        
        data = request.get_json()
        query = data['query']
        
        if not query or not query.strip():
            return jsonify({'error': 'Query cannot be empty'}), 400
        
        # Test the model
        result = ai_manager.test_model(model_id, query)
        
        return jsonify({
            'test_result': result,
            'model_id': model_id,
            'query': query,
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except ModelError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        current_app.logger.error(f"Error testing model {model_id}: {e}")
        return jsonify({'error': 'Model test failed'}), 500

@models_bp.route('/stats', methods=['GET'])
@rate_limit("100 per minute")
def get_model_stats():
    """Get aggregate model statistics"""
    try:
        ai_manager = get_ai_model_manager()
        if not ai_manager:
            return jsonify({'error': 'AI model manager not available'}), 503
        
        stats = ai_manager.get_model_stats()
        
        return jsonify({
            'stats': stats,
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        current_app.logger.error(f"Error getting model stats: {e}")
        return jsonify({'error': 'Failed to retrieve model statistics'}), 500

@models_bp.route('/providers', methods=['GET'])
@rate_limit("100 per minute")
def get_providers():
    """Get list of supported providers"""
    try:
        # Import here to avoid circular imports
        from ..services.ai_models import AIProvider
        
        providers = []
        for provider in AIProvider:
            providers.append({
                'id': provider.value,
                'name': provider.name.replace('_', ' ').title(),
                'description': _get_provider_description(provider)
            })
        
        return jsonify({
            'providers': providers,
            'total': len(providers)
        })
        
    except Exception as e:
        current_app.logger.error(f"Error getting providers: {e}")
        return jsonify({'error': 'Failed to retrieve providers'}), 500

@models_bp.route('/categories', methods=['GET'])
@rate_limit("100 per minute")
def get_categories():
    """Get list of available model categories"""
    try:
        categories = [
            {
                'id': 'general',
                'name': 'General Purpose',
                'description': 'General conversation and assistance'
            },
            {
                'id': 'coding',
                'name': 'Code Generation',
                'description': 'Programming and code-related tasks'
            },
            {
                'id': 'analysis',
                'name': 'Data Analysis',
                'description': 'Data analysis and interpretation'
            },
            {
                'id': 'writing',
                'name': 'Content Writing',
                'description': 'Creative and technical writing'
            },
            {
                'id': 'math',
                'name': 'Mathematics',
                'description': 'Mathematical calculations and reasoning'
            },
            {
                'id': 'research',
                'name': 'Research',
                'description': 'Research and information gathering'
            },
            {
                'id': 'translation',
                'name': 'Language Translation',
                'description': 'Translation between languages'
            },
            {
                'id': 'summarization',
                'name': 'Text Summarization',
                'description': 'Document and text summarization'
            }
        ]
        
        return jsonify({
            'categories': categories,
            'total': len(categories)
        })
        
    except Exception as e:
        current_app.logger.error(f"Error getting categories: {e}")
        return jsonify({'error': 'Failed to retrieve categories'}), 500

def _get_provider_description(provider):
    """Get description for AI provider"""
    descriptions = {
        'openai': 'OpenAI GPT models (GPT-3.5, GPT-4)',
        'anthropic': 'Anthropic Claude models',
        'google': 'Google Gemini and PaLM models',
        'microsoft': 'Microsoft Azure OpenAI Service',
        'xai': 'xAI Grok models',
        'perplexity': 'Perplexity AI models',
        'cohere': 'Cohere language models',
        'huggingface': 'Hugging Face hosted models',
        'ollama': 'Local Ollama models',
        'custom': 'Custom API endpoints'
    }
    
    return descriptions.get(provider.value, 'AI model provider')