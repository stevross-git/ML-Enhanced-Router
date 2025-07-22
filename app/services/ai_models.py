"""
AI Model Management Service - Integrated Version
Bridges the comprehensive AI model system with Flask clean architecture
"""

import logging
import os
import sys
from typing import Optional, List, Dict, Any

from flask import current_app

# Import the comprehensive AI model system
try:
    # Add current directory to path to import the comprehensive system
    current_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    from ai_models import AIModelManager as ComprehensiveAIManager, AIProvider, AIModel
    COMPREHENSIVE_AI_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("✅ Comprehensive AI model system loaded")
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"⚠️ Comprehensive AI system not available: {e}")
    COMPREHENSIVE_AI_AVAILABLE = False
    # Fallback imports
    from enum import Enum
    
    class AIProvider(Enum):
        OPENAI = "openai"
        ANTHROPIC = "anthropic"
        GOOGLE = "google"
        OLLAMA = "ollama"
        CUSTOM = "custom"

from ..extensions import db
from ..models.ai_model import MLModelRegistry, ModelConfiguration
from ..utils.exceptions import ModelError, ServiceError

class AIModelManager:
    """
    Integrated AI Model Management Service
    Provides a clean interface that works with both systems
    """
    
    def __init__(self):
        self.comprehensive_manager = None
        self.models_cache = {}
        self.active_model_id = None
        self.initialized = False
        
    def initialize(self):
        """Initialize the AI model manager"""
        try:
            if COMPREHENSIVE_AI_AVAILABLE:
                # Use the comprehensive system
                self.comprehensive_manager = ComprehensiveAIManager()
                self.comprehensive_manager.initialize_default_models()
                current_app.logger.info("✅ Using comprehensive AI model system")
            else:
                # Fallback to database-only system
                self._load_models_from_db()
                self._initialize_default_models()
                current_app.logger.info("✅ Using database-only AI model system")
            
            self.initialized = True
            current_app.logger.info("✅ AI Model Manager initialized successfully")
            
        except Exception as e:
            current_app.logger.error(f"❌ AI Model Manager initialization failed: {e}")
            raise ServiceError(f"Failed to initialize AI Model Manager: {e}")
    
    def get_all_models(self) -> List[Any]:
        """Get all available models"""
        try:
            if self.comprehensive_manager:
                return self.comprehensive_manager.get_models()
            else:
                return list(self.models_cache.values())
        except Exception as e:
            current_app.logger.error(f"Error getting all models: {e}")
            return []
    
    def get_model(self, model_id: str) -> Optional[Any]:
        """Get a specific model by ID"""
        try:
            if self.comprehensive_manager:
                return self.comprehensive_manager.get_model(model_id)
            else:
                return self.models_cache.get(model_id)
        except Exception as e:
            current_app.logger.error(f"Error getting model {model_id}: {e}")
            return None
    
    def get_active_model(self) -> Optional[Any]:
        """Get the currently active model"""
        try:
            if self.comprehensive_manager:
                # Check if there's a set active model
                if self.comprehensive_manager.active_model_id:
                    return self.comprehensive_manager.get_model(self.comprehensive_manager.active_model_id)
                
                # Otherwise find first active model
                for model in self.comprehensive_manager.get_models():
                    if getattr(model, 'is_active', False):
                        return model
                
                # If no active model, return first available
                models = self.comprehensive_manager.get_models()
                if models:
                    return models[0]
            else:
                if self.active_model_id:
                    return self.models_cache.get(self.active_model_id)
                
                # Find first active model
                for model in self.models_cache.values():
                    if getattr(model, 'is_active', False):
                        return model
                
                # Return first available
                if self.models_cache:
                    return list(self.models_cache.values())[0]
            
            return None
            
        except Exception as e:
            current_app.logger.error(f"Error getting active model: {e}")
            return None
    
    def set_active_model(self, model_id: str) -> bool:
        """Set the active model"""
        try:
            if self.comprehensive_manager:
                success = self.comprehensive_manager.set_active_model(model_id)
                if success:
                    current_app.logger.info(f"Set active model to: {model_id}")
                return success
            else:
                if model_id in self.models_cache:
                    # Deactivate all models
                    for model in self.models_cache.values():
                        model.is_active = False
                    
                    # Activate selected model
                    self.models_cache[model_id].is_active = True
                    self.active_model_id = model_id
                    
                    current_app.logger.info(f"Set active model to: {model_id}")
                    return True
            
            return False
            
        except Exception as e:
            current_app.logger.error(f"Error setting active model {model_id}: {e}")
            return False
    
    def add_custom_model(self, model_id: str, name: str, endpoint: str, 
                        api_key_env: str = '', model_name: str = '',
                        max_tokens: int = 4096, temperature: float = 0.7,
                        custom_headers: Dict = None) -> Optional[Any]:
        """Add a custom model"""
        try:
            if self.comprehensive_manager:
                return self.comprehensive_manager.add_custom_model(
                    model_id=model_id,
                    name=name,
                    endpoint=endpoint,
                    api_key_env=api_key_env,
                    model_name=model_name,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    custom_headers=custom_headers or {}
                )
            else:
                # Create a simple model object for database-only mode
                model_data = {
                    'id': model_id,
                    'name': name,
                    'provider': AIProvider.CUSTOM,
                    'model_name': model_name or model_id,
                    'endpoint': endpoint,
                    'api_key_env': api_key_env,
                    'max_tokens': max_tokens,
                    'temperature': temperature,
                    'custom_headers': custom_headers or {},
                    'is_active': False,
                    'deployment_type': 'cloud' if 'localhost' not in endpoint else 'local'
                }
                
                model = self._create_model_object(model_data)
                self.models_cache[model_id] = model
                
                # Save to database
                self._save_model_to_db(model)
                
                current_app.logger.info(f"Added custom model: {model_id}")
                return model
            
        except Exception as e:
            current_app.logger.error(f"Failed to add custom model: {e}")
            return None
    
    def remove_model(self, model_id: str) -> bool:
        """Remove a model"""
        try:
            if self.comprehensive_manager:
                return self.comprehensive_manager.remove_model(model_id)
            else:
                if model_id in self.models_cache:
                    # Remove from memory
                    del self.models_cache[model_id]
                    
                    # Remove from database
                    self._remove_model_from_db(model_id)
                    
                    current_app.logger.info(f"Removed model: {model_id}")
                    return True
                
                return False
            
        except Exception as e:
            current_app.logger.error(f"Failed to remove model: {e}")
            return False
    
    async def generate_response(self, model_id: str, query: str, system_message: str = None, 
                              user_id: str = "anonymous", stream: bool = False) -> Dict[str, Any]:
        """Generate AI response using the comprehensive system if available"""
        try:
            if self.comprehensive_manager:
                return await self.comprehensive_manager.generate_response(
                    model_id=model_id,
                    query=query,
                    system_message=system_message,
                    user_id=user_id,
                    stream=stream
                )
            else:
                # Basic fallback implementation
                model = self.get_model(model_id)
                if not model:
                    return {"error": f"Model '{model_id}' not found", "status": "error"}
                
                return {
                    "response": f"[Mock response from {model.name}] {query}",
                    "status": "success",
                    "model": model_id,
                    "response_time": 0.1,
                    "usage": {"tokens": len(query.split())},
                    "streaming": stream
                }
                
        except Exception as e:
            current_app.logger.error(f"Error generating response: {e}")
            return {
                "error": f"Internal error: {str(e)}",
                "status": "error",
                "model": model_id
            }
    
    def _load_models_from_db(self):
        """Load models from database (fallback mode)"""
        try:
            db_models = db.session.query(MLModelRegistry).all()
            
            for db_model in db_models:
                model = self._convert_db_model(db_model)
                self.models_cache[db_model.id] = model
            
            current_app.logger.info(f"Loaded {len(db_models)} models from database")
            
        except Exception as e:
            current_app.logger.error(f"Failed to load models from database: {e}")
            self.models_cache = {}
    
    def _convert_db_model(self, db_model: MLModelRegistry):
        """Convert database model to internal format"""
        model_data = {
            'id': db_model.id,
            'name': db_model.name,
            'provider': AIProvider(db_model.provider),
            'model_name': db_model.model_id,
            'endpoint': db_model.config.get('endpoint', ''),
            'api_key_env': db_model.config.get('api_key_env', ''),
            'max_tokens': db_model.config.get('max_tokens', 4096),
            'temperature': db_model.config.get('temperature', 0.7),
            'top_p': db_model.config.get('top_p', 1.0),
            'context_window': db_model.config.get('context_window', 4096),
            'cost_per_1k_tokens': db_model.cost_per_token or 0.0,
            'is_active': db_model.is_active,
            'supports_streaming': db_model.config.get('supports_streaming', False),
            'supports_system_message': db_model.config.get('supports_system_message', True),
            'supports_vision': db_model.config.get('supports_vision', False),
            'supports_audio': db_model.config.get('supports_audio', False),
            'supports_functions': db_model.config.get('supports_functions', False),
            'model_type': db_model.model_type or 'text',
            'deployment_type': db_model.config.get('deployment_type', 'cloud'),
            'input_modalities': db_model.config.get('input_modalities', ['text']),
            'output_modalities': db_model.config.get('output_modalities', ['text']),
            'capabilities': db_model.capabilities or []
        }
        
        return self._create_model_object(model_data)
    
    def _create_model_object(self, model_data: Dict) -> Any:
        """Create model object from data"""
        class SimpleAIModel:
            def __init__(self, data):
                for key, value in data.items():
                    setattr(self, key, value)
        
        # Set defaults
        model_data.setdefault('temperature', 0.7)
        model_data.setdefault('top_p', 1.0)
        model_data.setdefault('is_active', False)
        model_data.setdefault('supports_streaming', False)
        model_data.setdefault('supports_system_message', True)
        model_data.setdefault('supports_vision', False)
        model_data.setdefault('supports_audio', False)
        model_data.setdefault('supports_functions', False)
        model_data.setdefault('model_type', 'text')
        model_data.setdefault('deployment_type', 'cloud')
        model_data.setdefault('input_modalities', ['text'])
        model_data.setdefault('output_modalities', ['text'])
        model_data.setdefault('capabilities', [])
        
        return SimpleAIModel(model_data)
    
    def _initialize_default_models(self):
        """Initialize default AI models if none exist (fallback mode)"""
        if self.models_cache:
            return
        
        default_models = [
            {
                'id': 'gpt-4o',
                'name': 'GPT-4o',
                'provider': AIProvider.OPENAI,
                'model_name': 'gpt-4o',
                'endpoint': 'https://api.openai.com/v1/chat/completions',
                'api_key_env': 'OPENAI_API_KEY',
                'max_tokens': 4096,
                'context_window': 128000,
                'cost_per_1k_tokens': 0.005,
                'supports_streaming': True,
                'supports_vision': True
            },
            {
                'id': 'ollama-llama3.2',
                'name': 'Llama 3.2 (Local)',
                'provider': AIProvider.OLLAMA,
                'model_name': 'llama3.2:3b',
                'endpoint': 'http://localhost:11434/api/generate',
                'api_key_env': '',
                'max_tokens': 4096,
                'context_window': 131072,
                'cost_per_1k_tokens': 0.0,
                'deployment_type': 'local',
                'supports_streaming': True
            }
        ]
        
        for model_data in default_models:
            model = self._create_model_object(model_data)
            self.models_cache[model_data['id']] = model
        
        current_app.logger.info(f"Initialized {len(default_models)} default models")
    
    def _save_model_to_db(self, model):
        """Save model to database (fallback mode)"""
        try:
            model_data = {
                'id': model.id,
                'name': model.name,
                'provider': model.provider.value if hasattr(model.provider, 'value') else str(model.provider),
                'model_id': model.model_name,
                'config': {
                    'endpoint': getattr(model, 'endpoint', ''),
                    'api_key_env': getattr(model, 'api_key_env', ''),
                    'max_tokens': getattr(model, 'max_tokens', 4096),
                    'temperature': getattr(model, 'temperature', 0.7),
                    'top_p': getattr(model, 'top_p', 1.0),
                    'context_window': getattr(model, 'context_window', 4096),
                    'supports_streaming': getattr(model, 'supports_streaming', False),
                    'supports_system_message': getattr(model, 'supports_system_message', True),
                    'supports_vision': getattr(model, 'supports_vision', False),
                    'supports_audio': getattr(model, 'supports_audio', False),
                    'supports_functions': getattr(model, 'supports_functions', False),
                    'deployment_type': getattr(model, 'deployment_type', 'cloud'),
                    'input_modalities': getattr(model, 'input_modalities', ['text']),
                    'output_modalities': getattr(model, 'output_modalities', ['text']),
                    'custom_headers': getattr(model, 'custom_headers', {})
                },
                'cost_per_token': getattr(model, 'cost_per_1k_tokens', 0.0),
                'is_active': getattr(model, 'is_active', False),
                'model_type': getattr(model, 'model_type', 'text'),
                'capabilities': getattr(model, 'capabilities', [])
            }
            
            # Check if model already exists
            existing_model = db.session.query(MLModelRegistry).filter_by(id=model.id).first()
            
            if existing_model:
                # Update existing model
                for key, value in model_data.items():
                    if key not in ['id']:
                        setattr(existing_model, key, value)
            else:
                # Create new model
                db_model = MLModelRegistry(**model_data)
                db.session.add(db_model)
            
            db.session.commit()
            
        except Exception as e:
            db.session.rollback()
            current_app.logger.error(f"Failed to save model to database: {e}")
    
    def _remove_model_from_db(self, model_id: str):
        """Remove model from database (fallback mode)"""
        try:
            db_model = db.session.query(MLModelRegistry).filter_by(id=model_id).first()
            if db_model:
                db.session.delete(db_model)
                db.session.commit()
        except Exception as e:
            db.session.rollback()
            current_app.logger.error(f"Failed to remove model from database: {e}")


# Singleton instance
_ai_model_manager = None

def get_ai_model_manager() -> AIModelManager:
    """Get singleton AI model manager instance"""
    global _ai_model_manager
    if _ai_model_manager is None:
        _ai_model_manager = AIModelManager()
        if current_app:
            try:
                _ai_model_manager.initialize()
            except Exception as e:
                current_app.logger.error(f"Failed to initialize AI model manager: {e}")
    return _ai_model_manager

def init_ai_model_manager(app):
    """Initialize AI model manager with Flask app"""
    with app.app_context():
        service = get_ai_model_manager()
        return service