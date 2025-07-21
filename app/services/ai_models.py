"""
AI Model Management Service
Handles AI model operations, testing, and response generation
"""

import asyncio
import aiohttp
import json
from datetime import datetime
from typing import Optional, Dict, List, Any
from enum import Enum

from flask import current_app
from ..models import MLModelRegistry, ModelConfiguration
from ..extensions import db
from ..utils.exceptions import ModelError, ServiceError

class AIProvider(Enum):
    """AI model providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    MICROSOFT = "microsoft"
    XAI = "xai"
    PERPLEXITY = "perplexity"
    COHERE = "cohere"
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"
    CUSTOM = "custom"

class AIModelManager:
    """
    AI Model Management Service
    Handles model registration, configuration, and response generation
    """
    
    def __init__(self):
        self.models = {}
        self.active_model_id = None
        self.initialized = False
        
    def initialize(self):
        """Initialize the AI model manager"""
        try:
            # Load models from database
            self._load_models_from_db()
            
            # Initialize default models
            self._initialize_default_models()
            
            self.initialized = True
            current_app.logger.info("✅ AI Model Manager initialized")
            
        except Exception as e:
            current_app.logger.error(f"❌ AI Model Manager initialization failed: {e}")
            raise ServiceError(f"Failed to initialize AI Model Manager: {e}")
    
    def _load_models_from_db(self):
        """Load models from database"""
        try:
            db_models = db.session.query(MLModelRegistry).all()
            
            for db_model in db_models:
                self.models[db_model.id] = self._convert_db_model(db_model)
            
            current_app.logger.info(f"Loaded {len(db_models)} models from database")
            
        except Exception as e:
            current_app.logger.error(f"Failed to load models from database: {e}")
            self.models = {}
    
    def _convert_db_model(self, db_model: MLModelRegistry) -> Dict:
        """Convert database model to internal format"""
        return {
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
    
    def _initialize_default_models(self):
        """Initialize default AI models if none exist"""
        if self.models:
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
                'id': 'claude-sonnet-4',
                'name': 'Claude Sonnet 4',
                'provider': AIProvider.ANTHROPIC,
                'model_name': 'claude-sonnet-4-20250514',
                'endpoint': 'https://api.anthropic.com/v1/messages',
                'api_key_env': 'ANTHROPIC_API_KEY',
                'max_tokens': 4096,
                'context_window': 200000,
                'cost_per_1k_tokens': 0.003,
                'supports_streaming': True
            },
            {
                'id': 'gemini-flash',
                'name': 'Gemini 2.5 Flash',
                'provider': AIProvider.GOOGLE,
                'model_name': 'gemini-2.5-flash',
                'endpoint': 'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent',
                'api_key_env': 'GEMINI_API_KEY',
                'max_tokens': 8192,
                'context_window': 1000000,
                'cost_per_1k_tokens': 0.001,
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
            self.models[model_data['id']] = self._create_model_object(model_data)
        
        current_app.logger.info(f"Initialized {len(default_models)} default models")
    
    def _create_model_object(self, model_data: Dict) -> object:
        """Create model object from data"""
        class AIModel:
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
        
        return AIModel(model_data)
    
    def get_all_models(self) -> List[object]:
        """Get all registered models"""
        return list(self.models.values())
    
    def get_model(self, model_id: str) -> Optional[object]:
        """Get a specific model by ID"""
        return self.models.get(model_id)
    
    def get_active_model(self) -> Optional[object]:
        """Get the currently active model"""
        if self.active_model_id:
            return self.models.get(self.active_model_id)
        
        # Return first active model if no specific active model set
        for model in self.models.values():
            if model.is_active:
                return model
        
        return None
    
    def set_active_model(self, model_id: str) -> bool:
        """Set the active model"""
        if model_id in self.models:
            # Deactivate all models
            for model in self.models.values():
                model.is_active = False
            
            # Activate selected model
            self.models[model_id].is_active = True
            self.active_model_id = model_id
            
            current_app.logger.info(f"Set active model to: {model_id}")
            return True
        
        return False
    
    def add_custom_model(self, model_id: str, name: str, endpoint: str, 
                        api_key_env: str = '', model_name: str = '',
                        max_tokens: int = 4096, temperature: float = 0.7,
                        custom_headers: Dict = None) -> Optional[object]:
        """Add a custom model"""
        try:
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
            self.models[model_id] = model
            
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
            if model_id in self.models:
                # Remove from memory
                del self.models[model_id]
                
                