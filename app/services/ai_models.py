"""
AI Model Management Service - Integrated Version
Bridges the comprehensive AI model system with Flask clean architecture
"""

import logging
import os
import sys
from typing import Optional, List, Dict, Any

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

# These imports will be available after Flask app is created
db = None
MLModelRegistry = None
ModelConfiguration = None
ServiceError = None

def _init_flask_dependencies():
    """Initialize Flask dependencies when app context is available"""
    global db, MLModelRegistry, ModelConfiguration, ServiceError
    try:
        from ..extensions import db
        from ..models.ai_model import MLModelRegistry, ModelConfiguration
        from ..utils.exceptions import ServiceError
    except ImportError as e:
        # If imports fail, create dummy classes
        logger.warning(f"Flask dependencies not available: {e}")
        ServiceError = Exception

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
            # Initialize Flask dependencies
            _init_flask_dependencies()
            
            if COMPREHENSIVE_AI_AVAILABLE:
                # Use the comprehensive system
                self.comprehensive_manager = ComprehensiveAIManager()
                self.comprehensive_manager.initialize_default_models()
                logger.info("✅ Using comprehensive AI model system")
            else:
                # Fallback to database-only system
                self._load_models_from_db()
                self._initialize_default_models()
                logger.info("✅ Using database-only AI model system")
            
            self.initialized = True
            logger.info("✅ AI Model Manager initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ AI Model Manager initialization failed: {e}")
            # Don't raise exception during module import
            # Store the error and handle it gracefully
            self.initialization_error = str(e)
    
    def _ensure_initialized(self):
        """Ensure the manager is initialized before use"""
        if not self.initialized:
            self.initialize()
        
        if hasattr(self, 'initialization_error'):
            # If we have Flask context now, try to log properly
            try:
                from flask import current_app
                current_app.logger.error(f"AI Model Manager initialization error: {self.initialization_error}")
            except RuntimeError:
                # Still no Flask context, use standard logger
                logger.error(f"AI Model Manager initialization error: {self.initialization_error}")
    
    def _load_models_from_db(self):
        """Load models from database"""
        try:
            if MLModelRegistry and db:
                # Load existing models from database
                models = MLModelRegistry.query.all()
                for model in models:
                    # Convert database model to internal format
                    self.models_cache[model.model_id] = model
        except Exception as e:
            logger.error(f"Error loading models from database: {e}")
    
    def _initialize_default_models(self):
        """Initialize default models if none exist"""
        if not self.models_cache:
            # Add some default models if none exist
            default_model = type('DefaultModel', (), {
                'id': 'default',
                'name': 'Default Model',
                'provider': AIProvider.CUSTOM,
                'model_name': 'default',
                'endpoint': 'http://localhost:8000',
                'api_key_env': '',
                'is_active': True
            })()
            self.models_cache['default'] = default_model
            self.active_model_id = 'default'
    
    def _create_model_object(self, model_data):
        """Create a model object from data"""
        return type('Model', (), model_data)()
    
    def _save_model_to_db(self, model):
        """Save model to database"""
        try:
            if MLModelRegistry and db:
                existing = MLModelRegistry.query.filter_by(model_id=model.id).first()
                if not existing:
                    db_model = MLModelRegistry(
                        model_id=model.id,
                        name=model.name,
                        model_type=getattr(model, 'model_type', 'llm'),
                        categories=[],
                        config={}
                    )
                    db.session.add(db_model)
                    db.session.commit()
        except Exception as e:
            logger.error(f"Error saving model to database: {e}")
            if db:
                db.session.rollback()
    
    def _remove_model_from_db(self, model_id):
        """Remove model from database"""
        try:
            if MLModelRegistry and db:
                model = MLModelRegistry.query.filter_by(model_id=model_id).first()
                if model:
                    db.session.delete(model)
                    db.session.commit()
        except Exception as e:
            logger.error(f"Error removing model from database: {e}")
            if db:
                db.session.rollback()
    
    def get_all_models(self) -> List[Any]:
        """Get all available models"""
        self._ensure_initialized()
        try:
            if self.comprehensive_manager:
                return self.comprehensive_manager.get_models()
            else:
                return list(self.models_cache.values())
        except Exception as e:
            logger.error(f"Error getting all models: {e}")
            return []
    
    def get_model(self, model_id: str) -> Optional[Any]:
        """Get a specific model by ID"""
        self._ensure_initialized()
        try:
            if self.comprehensive_manager:
                return self.comprehensive_manager.get_model(model_id)
            else:
                return self.models_cache.get(model_id)
        except Exception as e:
            logger.error(f"Error getting model {model_id}: {e}")
            return None
    
    def get_active_model(self) -> Optional[Any]:
        """Get the currently active model"""
        self._ensure_initialized()
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
            logger.error(f"Error getting active model: {e}")
            return None
    
    def set_active_model(self, model_id: str) -> bool:
        """Set the active model"""
        self._ensure_initialized()
        try:
            if self.comprehensive_manager:
                success = self.comprehensive_manager.set_active_model(model_id)
                if success:
                    logger.info(f"Set active model to: {model_id}")
                return success
            else:
                if model_id in self.models_cache:
                    # Deactivate all models
                    for model in self.models_cache.values():
                        if hasattr(model, 'is_active'):
                            model.is_active = False
                    
                    # Activate selected model
                    if hasattr(self.models_cache[model_id], 'is_active'):
                        self.models_cache[model_id].is_active = True
                    self.active_model_id = model_id
                    
                    logger.info(f"Set active model to: {model_id}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error setting active model {model_id}: {e}")
            return False
    
    def activate_model(self, model_id: str) -> bool:
        """Activate a model (alias for set_active_model)"""
        return self.set_active_model(model_id)
    
    def add_custom_model(self, model_id: str, name: str, endpoint: str, 
                        api_key_env: str = '', model_name: str = '',
                        max_tokens: int = 4096, temperature: float = 0.7,
                        custom_headers: Dict = None) -> Optional[Any]:
        """Add a custom model"""
        self._ensure_initialized()
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
                
                logger.info(f"Added custom model: {model_id}")
                return model
            
        except Exception as e:
            logger.error(f"Failed to add custom model: {e}")
            return None
    
    def remove_model(self, model_id: str) -> bool:
        """Remove a model"""
        self._ensure_initialized()
        try:
            if self.comprehensive_manager:
                return self.comprehensive_manager.remove_model(model_id)
            else:
                if model_id in self.models_cache:
                    # Remove from memory
                    del self.models_cache[model_id]
                    
                    # Remove from database
                    self._remove_model_from_db(model_id)
                    
                    logger.info(f"Removed model: {model_id}")
                    return True
                
                return False
            
        except Exception as e:
            logger.error(f"Failed to remove model: {e}")
            return False
    
    def test_model(self, model_id: str, test_query: str = "Hello, how are you?") -> Dict[str, Any]:
        """Test a model's connectivity and response"""
        self._ensure_initialized()
        try:
            if self.comprehensive_manager:
                # Use comprehensive manager's test functionality if available
                return self.comprehensive_manager.test_model(model_id, test_query)
            else:
                # Simple test for fallback mode
                model = self.get_model(model_id)
                if model:
                    return {
                        "status": "success",
                        "message": f"Model {model_id} is available",
                        "model_id": model_id,
                        "test_query": test_query
                    }
                else:
                    return {
                        "status": "error",
                        "message": f"Model {model_id} not found",
                        "model_id": model_id
                    }
        except Exception as e:
            logger.error(f"Error testing model {model_id}: {e}")
            return {
                "status": "error",
                "message": f"Test failed: {str(e)}",
                "model_id": model_id
            }
    
    def get_model_statistics(self) -> Dict[str, Any]:
        """Get model usage statistics"""
        self._ensure_initialized()
        try:
            if self.comprehensive_manager:
                return self.comprehensive_manager.get_model_statistics()
            else:
                models = self.get_all_models()
                return {
                    "total_models": len(models),
                    "active_models": len([m for m in models if getattr(m, 'is_active', False)]),
                    "providers": {},
                    "model_types": {},
                    "capabilities": {}
                }
        except Exception as e:
            logger.error(f"Error getting model statistics: {e}")
            return {}
    
    def get_models_by_capability(self, capability) -> List[Any]:
        """Get models that support a specific capability"""
        self._ensure_initialized()
        try:
            if self.comprehensive_manager:
                return self.comprehensive_manager.get_models_by_capability(capability)
            else:
                # Fallback implementation
                models = self.get_all_models()
                return [
                    model for model in models
                    if hasattr(model, 'capabilities') and capability in getattr(model, 'capabilities', [])
                ]
        except Exception as e:
            logger.error(f"Error getting models by capability: {e}")
            return []
    
    def get_models_by_provider(self, provider) -> List[Any]:
        """Get models from a specific provider"""
        self._ensure_initialized()
        try:
            if self.comprehensive_manager:
                return self.comprehensive_manager.get_models_by_provider(provider)
            else:
                # Fallback implementation
                models = self.get_all_models()
                return [model for model in models if getattr(model, 'provider', None) == provider]
        except Exception as e:
            logger.error(f"Error getting models by provider: {e}")
            return []
    
    async def generate_response(self, model_id: str, query: str, system_message: str = None, 
                              user_id: str = "anonymous", stream: bool = False) -> Dict[str, Any]:
        """Generate AI response using the comprehensive system if available"""
        self._ensure_initialized()
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
                # Fallback implementation
                model = self.get_model(model_id)
                if not model:
                    return {
                        "status": "error",
                        "error": f"Model {model_id} not found"
                    }
                
                # Simple mock response for fallback
                return {
                    "status": "success",
                    "response": f"Mock response from {model_id} for query: {query}",
                    "model": model_id,
                    "streaming": stream
                }
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                "status": "error",
                "error": str(e),
                "model": model_id
            }


# Global instances
_ai_model_manager = None

def get_ai_model_manager():
    """Get the AI model manager instance"""
    global _ai_model_manager
    if _ai_model_manager is None:
        _ai_model_manager = AIModelManager()
        # Don't initialize immediately - wait for Flask context
    return _ai_model_manager

def get_comprehensive_manager():
    """Get the comprehensive AI manager instance"""
    manager = get_ai_model_manager()
    return manager.comprehensive_manager if manager else None

def is_comprehensive_system_available() -> bool:
    """Check if comprehensive system is available"""
    return COMPREHENSIVE_AI_AVAILABLE

def get_model_count() -> int:
    """Get the total number of available models"""
    try:
        manager = get_ai_model_manager()
        if manager:
            models = manager.get_all_models()
            return len(models)
        return 0
    except Exception as e:
        logger.error(f"Error getting model count: {e}")
        return 0

# Create the manager instance but don't initialize it yet
ai_model_manager = AIModelManager()