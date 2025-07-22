"""
AI Models Blueprint - Complete Route Implementation
Handles all AI model management and interaction endpoints
"""

import json
import logging
import asyncio
import traceback
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

from flask import Blueprint, request, jsonify, current_app, Response
from werkzeug.exceptions import BadRequest, NotFound, InternalServerError

from ..services.ai_models import (
    get_ai_model_manager,
    get_comprehensive_manager, 
    is_comprehensive_system_available,
    get_model_count
)
from ..utils.decorators import rate_limit, validate_json, require_auth
from ..utils.exceptions import ValidationError, ServiceError

logger = logging.getLogger(__name__)
models_bp = Blueprint('models', __name__, url_prefix='/api/models')

# Define enums locally for blueprint use
class AIProvider(Enum):
    """Supported AI providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    OLLAMA = "ollama"
    CUSTOM = "custom"
    XAI = "xai"
    PERPLEXITY = "perplexity"
    COHERE = "cohere"
    HUGGINGFACE = "huggingface"
    MISTRAL = "mistral"
    AZURE = "azure"
    AWS_BEDROCK = "aws_bedrock"
    REPLICATE = "replicate"
    TOGETHER = "together"
    GROQ = "groq"
    DEEPSEEK = "deepseek"
    CEREBRAS = "cerebras"
    FIREWORKS = "fireworks"
    ANYSCALE = "anyscale"
    RUNPOD = "runpod"
    ELEVENLABS = "elevenlabs"

class ModelCapability(Enum):
    """Model capabilities for multi-modal AI"""
    TEXT_GENERATION = "text_generation"
    IMAGE_ANALYSIS = "image_analysis"
    IMAGE_GENERATION = "image_generation"
    AUDIO_TRANSCRIPTION = "audio_transcription"
    AUDIO_GENERATION = "audio_generation"
    VIDEO_ANALYSIS = "video_analysis"
    VIDEO_GENERATION = "video_generation"
    DOCUMENT_ANALYSIS = "document_analysis"
    CODE_GENERATION = "code_generation"
    EMBEDDING = "embedding"
    FUNCTION_CALLING = "function_calling"
    REASONING = "reasoning"
    MULTIMODAL = "multimodal"

# ================================
# MODEL MANAGEMENT ROUTES
# ================================

@models_bp.route('/', methods=['GET'])
def list_all_models():
    """Get all available AI models"""
    try:
        manager = get_ai_model_manager()
        if not manager:
            return jsonify({
                'status': 'error',
                'message': 'AI model manager not available'
            }), 500
        
        models = manager.get_all_models()
        
        # Convert models to JSON-serializable format
        models_data = []
        for model in models:
            model_data = {
                'id': getattr(model, 'id', 'unknown'),
                'name': getattr(model, 'name', 'Unknown'),
                'provider': getattr(model, 'provider', 'unknown'),
                'model_name': getattr(model, 'model_name', ''),
                'endpoint': getattr(model, 'endpoint', ''),
                'is_active': getattr(model, 'is_active', False),
                'max_tokens': getattr(model, 'max_tokens', 4096),
                'temperature': getattr(model, 'temperature', 0.7),
                'cost_per_1k_tokens': getattr(model, 'cost_per_1k_tokens', 0.0),
                'context_window': getattr(model, 'context_window', 4096),
                'supports_streaming': getattr(model, 'supports_streaming', True),
                'supports_vision': getattr(model, 'supports_vision', False),
                'supports_functions': getattr(model, 'supports_functions', False),
                'model_type': getattr(model, 'model_type', 'llm'),
                'deployment_type': getattr(model, 'deployment_type', 'cloud'),
                'capabilities': getattr(model, 'capabilities', [])
            }
            
            # Handle provider enum conversion
            provider = model_data['provider']
            if hasattr(provider, 'value'):
                model_data['provider'] = provider.value
            else:
                model_data['provider'] = str(provider)
            
            # Handle capabilities list conversion
            capabilities = model_data['capabilities']
            if isinstance(capabilities, list):
                model_data['capabilities'] = [
                    cap.value if hasattr(cap, 'value') else str(cap) 
                    for cap in capabilities
                ]
            
            models_data.append(model_data)
        
        return jsonify({
            'status': 'success',
            'models': models_data,
            'count': len(models_data),
            'comprehensive_available': is_comprehensive_system_available()
        })
        
    except Exception as e:
        logger.error(f"Error getting all models: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            'status': 'error',
            'message': f'Failed to get models: {str(e)}'
        }), 500

@models_bp.route('/<model_id>', methods=['GET'])
def get_single_model(model_id: str):
    """Get details for a specific model"""
    try:
        manager = get_ai_model_manager()
        if not manager:
            return jsonify({
                'status': 'error',
                'message': 'AI model manager not available'
            }), 500
        
        model = manager.get_model(model_id)
        if not model:
            return jsonify({
                'status': 'error',
                'message': f'Model {model_id} not found'
            }), 404
        
        model_data = {
            'id': getattr(model, 'id', model_id),
            'name': getattr(model, 'name', 'Unknown'),
            'provider': getattr(model, 'provider', 'unknown'),
            'model_name': getattr(model, 'model_name', ''),
            'endpoint': getattr(model, 'endpoint', ''),
            'is_active': getattr(model, 'is_active', False),
            'max_tokens': getattr(model, 'max_tokens', 4096),
            'temperature': getattr(model, 'temperature', 0.7),
            'top_p': getattr(model, 'top_p', 0.9),
            'cost_per_1k_tokens': getattr(model, 'cost_per_1k_tokens', 0.0),
            'context_window': getattr(model, 'context_window', 4096),
            'supports_streaming': getattr(model, 'supports_streaming', True),
            'supports_system_message': getattr(model, 'supports_system_message', True),
            'supports_vision': getattr(model, 'supports_vision', False),
            'supports_audio': getattr(model, 'supports_audio', False),
            'supports_functions': getattr(model, 'supports_functions', False),
            'model_type': getattr(model, 'model_type', 'llm'),
            'deployment_type': getattr(model, 'deployment_type', 'cloud'),
            'input_modalities': getattr(model, 'input_modalities', ['text']),
            'output_modalities': getattr(model, 'output_modalities', ['text']),
            'custom_headers': getattr(model, 'custom_headers', {}),
            'capabilities': getattr(model, 'capabilities', [])
        }
        
        # Handle provider enum conversion
        provider = model_data['provider']
        if hasattr(provider, 'value'):
            model_data['provider'] = provider.value
        else:
            model_data['provider'] = str(provider)
        
        # Handle capabilities list conversion
        capabilities = model_data['capabilities']
        if isinstance(capabilities, list):
            model_data['capabilities'] = [
                cap.value if hasattr(cap, 'value') else str(cap) 
                for cap in capabilities
            ]
        
        return jsonify({
            'status': 'success',
            'model': model_data
        })
        
    except Exception as e:
        logger.error(f"Error getting model {model_id}: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Failed to get model: {str(e)}'
        }), 500

@models_bp.route('/active', methods=['GET'])
def get_currently_active_model():
    """Get the currently active model"""
    try:
        manager = get_ai_model_manager()
        if not manager:
            return jsonify({
                'status': 'error',
                'message': 'AI model manager not available'
            }), 500
        
        model = manager.get_active_model()
        if not model:
            return jsonify({
                'status': 'error',
                'message': 'No active model found'
            }), 404
        
        provider = getattr(model, 'provider', 'unknown')
        if hasattr(provider, 'value'):
            provider = provider.value
        else:
            provider = str(provider)
        
        model_data = {
            'id': getattr(model, 'id', 'unknown'),
            'name': getattr(model, 'name', 'Unknown'),
            'provider': provider,
            'model_name': getattr(model, 'model_name', ''),
            'is_active': True
        }
        
        return jsonify({
            'status': 'success',
            'active_model': model_data
        })
        
    except Exception as e:
        logger.error(f"Error getting active model: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Failed to get active model: {str(e)}'
        }), 500

@models_bp.route('/activate', methods=['POST'])
def activate_specific_model():
    """Activate a specific model"""
    try:
        data = request.get_json()
        model_id = data.get('model_id')
        
        if not model_id:
            return jsonify({
                'status': 'error',
                'message': 'model_id is required'
            }), 400
        
        manager = get_ai_model_manager()
        if not manager:
            return jsonify({
                'status': 'error',
                'message': 'AI model manager not available'
            }), 500
        
        success = manager.set_active_model(model_id)
        
        if success:
            return jsonify({
                'status': 'success',
                'message': f'Model {model_id} activated successfully',
                'active_model_id': model_id
            })
        else:
            return jsonify({
                'status': 'error',
                'message': f'Failed to activate model {model_id}'
            }), 400
            
    except Exception as e:
        logger.error(f"Error activating model: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Failed to activate model: {str(e)}'
        }), 500

@models_bp.route('/add-custom', methods=['POST'])
def create_custom_model():
    """Add a custom model"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['model_id', 'name', 'endpoint']
        for field in required_fields:
            if not data.get(field):
                return jsonify({
                    'status': 'error',
                    'message': f'{field} is required'
                }), 400
        
        manager = get_ai_model_manager()
        if not manager:
            return jsonify({
                'status': 'error',
                'message': 'AI model manager not available'
            }), 500
        
        # Add the custom model
        model = manager.add_custom_model(
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
                'message': f'Custom model {data["model_id"]} added successfully',
                'model_id': getattr(model, 'id', data['model_id'])
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Failed to add custom model'
            }), 500
            
    except Exception as e:
        logger.error(f"Error adding custom model: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Failed to add custom model: {str(e)}'
        }), 500

@models_bp.route('/<model_id>', methods=['DELETE'])
def delete_model_by_id(model_id: str):
    """Remove a model"""
    try:
        manager = get_ai_model_manager()
        if not manager:
            return jsonify({
                'status': 'error',
                'message': 'AI model manager not available'
            }), 500
        
        success = manager.remove_model(model_id)
        
        if success:
            return jsonify({
                'status': 'success',
                'message': f'Model {model_id} removed successfully'
            })
        else:
            return jsonify({
                'status': 'error',
                'message': f'Model {model_id} not found or could not be removed'
            }), 404
            
    except Exception as e:
        logger.error(f"Error removing model {model_id}: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Failed to remove model: {str(e)}'
        }), 500

# ================================
# AI INTERACTION ROUTES
# ================================

@models_bp.route('/chat', methods=['POST'])
def process_chat_request():
    """Generate AI response using specified model"""
    try:
        data = request.get_json()
        
        # Validate required fields
        if not data.get('query'):
            return jsonify({
                'status': 'error',
                'message': 'query is required'
            }), 400
        
        model_id = data.get('model_id')
        query = data.get('query')
        system_message = data.get('system_message')
        user_id = data.get('user_id', 'anonymous')
        stream = data.get('stream', False)
        
        manager = get_ai_model_manager()
        if not manager:
            return jsonify({
                'status': 'error',
                'message': 'AI model manager not available'
            }), 500
        
        # If no model specified, use active model
        if not model_id:
            active_model = manager.get_active_model()
            if not active_model:
                return jsonify({
                    'status': 'error',
                    'message': 'No model specified and no active model found'
                }), 400
            model_id = getattr(active_model, 'id', 'unknown')
        
        # Generate response
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(
                manager.generate_response(
                    model_id=model_id,
                    query=query,
                    system_message=system_message,
                    user_id=user_id,
                    stream=stream
                )
            )
        finally:
            loop.close()
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            'status': 'error',
            'message': f'Failed to generate response: {str(e)}'
        }), 500

@models_bp.route('/chat/stream', methods=['POST'])
def process_streaming_chat():
    """Generate streaming AI response"""
    try:
        data = request.get_json()
        
        if not data.get('query'):
            return jsonify({
                'status': 'error',
                'message': 'query is required'
            }), 400
        
        model_id = data.get('model_id')
        query = data.get('query')
        system_message = data.get('system_message')
        user_id = data.get('user_id', 'anonymous')
        
        manager = get_ai_model_manager()
        if not manager:
            return jsonify({
                'status': 'error',
                'message': 'AI model manager not available'
            }), 500
        
        # If no model specified, use active model
        if not model_id:
            active_model = manager.get_active_model()
            if not active_model:
                return jsonify({
                    'status': 'error',
                    'message': 'No model specified and no active model found'
                }), 400
            model_id = getattr(active_model, 'id', 'unknown')
        
        def generate_streaming_response():
            """Generator function for streaming response"""
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                try:
                    result = loop.run_until_complete(
                        manager.generate_response(
                            model_id=model_id,
                            query=query,
                            system_message=system_message,
                            user_id=user_id,
                            stream=True
                        )
                    )
                    
                    # Send the complete response
                    yield f"data: {json.dumps(result)}\n\n"
                    yield "data: [DONE]\n\n"
                    
                finally:
                    loop.close()
                    
            except Exception as e:
                error_response = {
                    'status': 'error',
                    'message': f'Streaming error: {str(e)}'
                }
                yield f"data: {json.dumps(error_response)}\n\n"
        
        return Response(
            generate_streaming_response(),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Cache-Control'
            }
        )
        
    except Exception as e:
        logger.error(f"Error in stream endpoint: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Failed to start streaming: {str(e)}'
        }), 500

@models_bp.route('/test/<model_id>', methods=['POST'])
def test_model_connectivity(model_id: str):
    """Test a model's connectivity and response"""
    try:
        data = request.get_json() or {}
        test_query = data.get('test_query', 'Hello, how are you?')
        
        manager = get_ai_model_manager()
        if not manager:
            return jsonify({
                'status': 'error',
                'message': 'AI model manager not available'
            }), 500
        
        # Check if model exists
        model = manager.get_model(model_id)
        if not model:
            return jsonify({
                'status': 'error',
                'message': f'Model {model_id} not found'
            }), 404
        
        # Test the model
        result = manager.test_model(model_id, test_query)
        
        return jsonify({
            'status': 'success',
            'test_result': result,
            'model_id': model_id,
            'test_query': test_query
        })
        
    except Exception as e:
        logger.error(f"Error testing model {model_id}: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Failed to test model: {str(e)}'
        }), 500

# ================================
# STATISTICS AND INFORMATION ROUTES
# ================================

@models_bp.route('/stats', methods=['GET'])
def get_model_statistics():
    """Get AI models statistics"""
    try:
        manager = get_ai_model_manager()
        if not manager:
            return jsonify({
                'status': 'error',
                'message': 'AI model manager not available'
            }), 500
        
        stats = manager.get_model_statistics()
        
        return jsonify({
            'status': 'success',
            'statistics': stats,
            'comprehensive_available': is_comprehensive_system_available(),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting statistics: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Failed to get statistics: {str(e)}'
        }), 500

@models_bp.route('/providers', methods=['GET'])
def list_available_providers():
    """Get available AI providers"""
    try:
        providers = []
        for provider in AIProvider:
            providers.append({
                'id': provider.value,
                'name': provider.name,
                'value': provider.value
            })
        
        return jsonify({
            'status': 'success',
            'providers': providers
        })
        
    except Exception as e:
        logger.error(f"Error getting providers: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Failed to get providers: {str(e)}'
        }), 500

@models_bp.route('/capabilities', methods=['GET'])
def list_available_capabilities():
    """Get available model capabilities"""
    try:
        capabilities = []
        for capability in ModelCapability:
            capabilities.append({
                'id': capability.value,
                'name': capability.name,
                'value': capability.value
            })
        
        return jsonify({
            'status': 'success',
            'capabilities': capabilities
        })
        
    except Exception as e:
        logger.error(f"Error getting capabilities: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Failed to get capabilities: {str(e)}'
        }), 500

@models_bp.route('/by-capability/<capability>', methods=['GET'])
def filter_models_by_capability(capability: str):
    """Get models that support a specific capability"""
    try:
        # Validate capability
        try:
            cap = ModelCapability(capability)
        except ValueError:
            return jsonify({
                'status': 'error',
                'message': f'Invalid capability: {capability}'
            }), 400
        
        manager = get_ai_model_manager()
        if not manager:
            return jsonify({
                'status': 'error',
                'message': 'AI model manager not available'
            }), 500
        
        models = manager.get_models_by_capability(cap)
        
        models_data = []
        for model in models:
            provider = getattr(model, 'provider', 'unknown')
            if hasattr(provider, 'value'):
                provider = provider.value
            else:
                provider = str(provider)
                
            model_data = {
                'id': getattr(model, 'id', 'unknown'),
                'name': getattr(model, 'name', 'Unknown'),
                'provider': provider,
                'is_active': getattr(model, 'is_active', False),
                'model_type': getattr(model, 'model_type', 'llm')
            }
            models_data.append(model_data)
        
        return jsonify({
            'status': 'success',
            'capability': capability,
            'models': models_data,
            'count': len(models_data)
        })
        
    except Exception as e:
        logger.error(f"Error getting models by capability {capability}: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Failed to get models by capability: {str(e)}'
        }), 500

@models_bp.route('/by-provider/<provider>', methods=['GET'])
def filter_models_by_provider(provider: str):
    """Get models from a specific provider"""
    try:
        # Validate provider
        try:
            prov = AIProvider(provider)
        except ValueError:
            return jsonify({
                'status': 'error',
                'message': f'Invalid provider: {provider}'
            }), 400
        
        manager = get_ai_model_manager()
        if not manager:
            return jsonify({
                'status': 'error',
                'message': 'AI model manager not available'
            }), 500
        
        models = manager.get_models_by_provider(prov)
        
        models_data = []
        for model in models:
            model_data = {
                'id': getattr(model, 'id', 'unknown'),
                'name': getattr(model, 'name', 'Unknown'),
                'model_name': getattr(model, 'model_name', ''),
                'is_active': getattr(model, 'is_active', False),
                'model_type': getattr(model, 'model_type', 'llm'),
                'cost_per_1k_tokens': getattr(model, 'cost_per_1k_tokens', 0.0)
            }
            models_data.append(model_data)
        
        return jsonify({
            'status': 'success',
            'provider': provider,
            'models': models_data,
            'count': len(models_data)
        })
        
    except Exception as e:
        logger.error(f"Error getting models by provider {provider}: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Failed to get models by provider: {str(e)}'
        }), 500

# ================================
# HEALTH AND STATUS ROUTES
# ================================

@models_bp.route('/health', methods=['GET'])
def check_service_health():
    """Health check endpoint"""
    try:
        manager = get_ai_model_manager()
        if not manager:
            return jsonify({
                'status': 'unhealthy',
                'message': 'AI model manager not available'
            }), 503
        
        models = manager.get_all_models()
        active_model = manager.get_active_model()
        
        return jsonify({
            'status': 'healthy',
            'total_models': len(models),
            'active_model': getattr(active_model, 'id', None) if active_model else None,
            'comprehensive_available': is_comprehensive_system_available(),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return jsonify({
            'status': 'unhealthy',
            'message': f'Health check failed: {str(e)}'
        }), 503

# ================================
# ERROR HANDLERS
# ================================

@models_bp.errorhandler(ValidationError)
def handle_validation_error_models(e):
    """Handle validation errors"""
    return jsonify({
        'status': 'error',
        'message': f'Validation error: {str(e)}'
    }), 400

@models_bp.errorhandler(ServiceError)
def handle_service_error_models(e):
    """Handle service errors"""
    return jsonify({
        'status': 'error',
        'message': f'Service error: {str(e)}'
    }), 500

@models_bp.errorhandler(BadRequest)
def handle_bad_request_models(e):
    """Handle bad request errors"""
    return jsonify({
        'status': 'error',
        'message': 'Bad request',
        'details': str(e)
    }), 400

@models_bp.errorhandler(NotFound)
def handle_not_found_models(e):
    """Handle not found errors"""
    return jsonify({
        'status': 'error',
        'message': 'Resource not found',
        'details': str(e)
    }), 404

@models_bp.errorhandler(InternalServerError)
def handle_internal_error_models(e):
    """Handle internal server errors"""
    logger.error(f"Internal server error: {str(e)}")
    return jsonify({
        'status': 'error',
        'message': 'Internal server error'
    }), 500

# ================================
# UTILITY ROUTES
# ================================

@models_bp.route('/count', methods=['GET'])
def get_total_models_count():
    """Get total count of models"""
    try:
        count = get_model_count()
        return jsonify({
            'status': 'success',
            'count': count
        })
        
    except Exception as e:
        logger.error(f"Error getting model count: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Failed to get model count: {str(e)}'
        }), 500

@models_bp.route('/info', methods=['GET'])
def get_models_system_info():
    """Get AI model system information"""
    try:
        return jsonify({
            'status': 'success',
            'info': {
                'comprehensive_system_available': is_comprehensive_system_available(),
                'total_models': get_model_count(),
                'supported_providers': [p.value for p in AIProvider],
                'supported_capabilities': [c.value for c in ModelCapability],
                'version': '1.0.0',
                'timestamp': datetime.now().isoformat()
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting system info: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Failed to get system info: {str(e)}'
        }), 500

@models_bp.route('/detailed', methods=['GET'])
def get_detailed_models_info():
    """Get detailed models information for frontend"""
    try:
        manager = get_ai_model_manager()
        if not manager:
            return jsonify({
                'status': 'error',
                'message': 'AI model manager not available'
            }), 500
        
        models = manager.get_all_models()
        active_model = manager.get_active_model()
        
        # Convert models to detailed format
        detailed_models = []
        for model in models:
            provider = getattr(model, 'provider', 'unknown')
            if hasattr(provider, 'value'):
                provider = provider.value
            else:
                provider = str(provider)
            
            capabilities = getattr(model, 'capabilities', [])
            if isinstance(capabilities, list):
                capabilities = [
                    cap.value if hasattr(cap, 'value') else str(cap) 
                    for cap in capabilities
                ]
            
            model_detail = {
                'id': getattr(model, 'id', 'unknown'),
                'name': getattr(model, 'name', 'Unknown'),
                'provider': provider,
                'model_name': getattr(model, 'model_name', ''),
                'endpoint': getattr(model, 'endpoint', ''),
                'is_active': getattr(model, 'is_active', False),
                'max_tokens': getattr(model, 'max_tokens', 4096),
                'temperature': getattr(model, 'temperature', 0.7),
                'top_p': getattr(model, 'top_p', 0.9),
                'cost_per_1k_tokens': getattr(model, 'cost_per_1k_tokens', 0.0),
                'context_window': getattr(model, 'context_window', 4096),
                'supports_streaming': getattr(model, 'supports_streaming', True),
                'supports_system_message': getattr(model, 'supports_system_message', True),
                'supports_vision': getattr(model, 'supports_vision', False),
                'supports_audio': getattr(model, 'supports_audio', False),
                'supports_functions': getattr(model, 'supports_functions', False),
                'model_type': getattr(model, 'model_type', 'llm'),
                'deployment_type': getattr(model, 'deployment_type', 'cloud'),
                'input_modalities': getattr(model, 'input_modalities', ['text']),
                'output_modalities': getattr(model, 'output_modalities', ['text']),
                'capabilities': capabilities,
                'api_key_env': getattr(model, 'api_key_env', ''),
                'custom_headers': getattr(model, 'custom_headers', {})
            }
            detailed_models.append(model_detail)
        
        # Get active model info
        active_model_info = None
        if active_model:
            provider = getattr(active_model, 'provider', 'unknown')
            if hasattr(provider, 'value'):
                provider = provider.value
            else:
                provider = str(provider)
                
            active_model_info = {
                'id': getattr(active_model, 'id', 'unknown'),
                'name': getattr(active_model, 'name', 'Unknown'),
                'provider': provider,
                'model_name': getattr(active_model, 'model_name', ''),
                'is_active': True
            }
        
        return jsonify({
            'status': 'success',
            'models': detailed_models,
            'active_model': active_model_info,
            'count': len(detailed_models),
            'comprehensive_available': is_comprehensive_system_available(),
            'system_info': {
                'supported_providers': [p.value for p in AIProvider],
                'supported_capabilities': [c.value for c in ModelCapability],
                'version': '1.0.0'
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting detailed models info: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Failed to get detailed models info: {str(e)}'
        }), 500