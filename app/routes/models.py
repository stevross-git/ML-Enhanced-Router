"""
AI Model Integration Service
Supports major AI providers, local Ollama, and custom endpoints
"""
import traceback
import sys

import json
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Union
import aiohttp
import requests
from datetime import datetime
from ai_cache import get_cache_manager
from flask import Blueprint, request, jsonify, current_app, Response
from ..services.ai_models import (
    get_ai_model_manager, 
    get_comprehensive_manager,
    is_comprehensive_system_available,
    get_model_count
)
from ..utils.decorators import rate_limit, validate_json, require_auth
from ..utils.exceptions import ValidationError, ServiceError

logger = logging.getLogger(__name__)
models_bp = Blueprint('models', __name__)

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

@dataclass
class AIModel:
    """AI model configuration"""
    id: str
    name: str
    provider: AIProvider
    model_name: str
    endpoint: str
    api_key_env: str
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.9
    supports_streaming: bool = True
    supports_system_message: bool = True
    cost_per_1k_tokens: float = 0.0
    context_window: int = 4096
    is_active: bool = True
    custom_headers: Dict[str, str] = field(default_factory=dict)
    capabilities: List[ModelCapability] = field(default_factory=list)
    supports_vision: bool = False
    supports_audio: bool = False
    supports_video: bool = False
    supports_functions: bool = False
    model_type: str = "llm"  # llm, embedding, image, audio, video, multimodal
    input_modalities: List[str] = field(default_factory=list)
    output_modalities: List[str] = field(default_factory=list)
    deployment_type: str = "cloud"  # cloud, local, hybrid

class AIModelManager:
    """Manages AI models and handles requests to different providers"""
    
    def __init__(self):
        self.models = self._initialize_models()
        self.active_model_id = None
        self.cache_manager = get_cache_manager()
        
    def _initialize_models(self) -> Dict[str, AIModel]:
        """Initialize all available AI models"""
        models = [
            # OpenAI Models
            AIModel(
                id="gpt-4o",
                name="GPT-4o",
                provider=AIProvider.OPENAI,
                model_name="gpt-4o",
                endpoint="https://api.openai.com/v1/chat/completions",
                api_key_env="OPENAI_API_KEY",
                max_tokens=4096,
                context_window=128000,
                cost_per_1k_tokens=0.005,
                capabilities=[
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.IMAGE_ANALYSIS,
                    ModelCapability.REASONING,
                    ModelCapability.CODE_GENERATION,
                    ModelCapability.FUNCTION_CALLING,
                    ModelCapability.MULTIMODAL
                ],
                supports_vision=True,
                supports_functions=True,
                model_type="multimodal",
                input_modalities=["text", "image"],
                output_modalities=["text"]
            ),
            AIModel(
                id="gpt-4o-mini",
                name="GPT-4o Mini",
                provider=AIProvider.OPENAI,
                model_name="gpt-4o-mini",
                endpoint="https://api.openai.com/v1/chat/completions",
                api_key_env="OPENAI_API_KEY",
                max_tokens=16384,
                context_window=128000,
                cost_per_1k_tokens=0.00015,
                capabilities=[
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.IMAGE_ANALYSIS,
                    ModelCapability.REASONING,
                    ModelCapability.CODE_GENERATION,
                    ModelCapability.FUNCTION_CALLING,
                    ModelCapability.MULTIMODAL
                ],
                supports_vision=True,
                supports_functions=True,
                model_type="multimodal",
                input_modalities=["text", "image"],
                output_modalities=["text"]
            ),
            AIModel(
                id="o1-preview",
                name="O1 Preview",
                provider=AIProvider.OPENAI,
                model_name="o1-preview",
                endpoint="https://api.openai.com/v1/chat/completions",
                api_key_env="OPENAI_API_KEY",
                max_tokens=32768,
                context_window=128000,
                cost_per_1k_tokens=0.015,
                capabilities=[
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.REASONING,
                    ModelCapability.CODE_GENERATION
                ],
                supports_functions=False,
                model_type="llm",
                input_modalities=["text"],
                output_modalities=["text"]
            ),
            AIModel(
                id="o1-mini",
                name="O1 Mini",
                provider=AIProvider.OPENAI,
                model_name="o1-mini",
                endpoint="https://api.openai.com/v1/chat/completions",
                api_key_env="OPENAI_API_KEY",
                max_tokens=65536,
                context_window=128000,
                cost_per_1k_tokens=0.003,
                capabilities=[
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.REASONING,
                    ModelCapability.CODE_GENERATION
                ],
                supports_functions=False,
                model_type="llm",
                input_modalities=["text"],
                output_modalities=["text"]
            ),
            AIModel(
                id="dall-e-3",
                name="DALL-E 3",
                provider=AIProvider.OPENAI,
                model_name="dall-e-3",
                endpoint="https://api.openai.com/v1/images/generations",
                api_key_env="OPENAI_API_KEY",
                max_tokens=4000,
                cost_per_1k_tokens=0.04,
                capabilities=[
                    ModelCapability.IMAGE_GENERATION
                ],
                model_type="image",
                input_modalities=["text"],
                output_modalities=["image"]
            ),
            # Anthropic Models
            AIModel(
                id="claude-3-5-sonnet-20241022",
                name="Claude 3.5 Sonnet",
                provider=AIProvider.ANTHROPIC,
                model_name="claude-3-5-sonnet-20241022",
                endpoint="https://api.anthropic.com/v1/messages",
                api_key_env="ANTHROPIC_API_KEY",
                max_tokens=8192,
                context_window=200000,
                cost_per_1k_tokens=0.003,
                capabilities=[
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.IMAGE_ANALYSIS,
                    ModelCapability.REASONING,
                    ModelCapability.CODE_GENERATION,
                    ModelCapability.MULTIMODAL
                ],
                supports_vision=True,
                model_type="multimodal",
                input_modalities=["text", "image"],
                output_modalities=["text"]
            ),
            AIModel(
                id="claude-3-5-haiku-20241022",
                name="Claude 3.5 Haiku",
                provider=AIProvider.ANTHROPIC,
                model_name="claude-3-5-haiku-20241022",
                endpoint="https://api.anthropic.com/v1/messages",
                api_key_env="ANTHROPIC_API_KEY",
                max_tokens=8192,
                context_window=200000,
                cost_per_1k_tokens=0.0008,
                capabilities=[
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.IMAGE_ANALYSIS,
                    ModelCapability.REASONING,
                    ModelCapability.CODE_GENERATION,
                    ModelCapability.MULTIMODAL
                ],
                supports_vision=True,
                model_type="multimodal",
                input_modalities=["text", "image"],
                output_modalities=["text"]
            ),
            # Google Models
            AIModel(
                id="gemini-2.0-flash-exp",
                name="Gemini 2.0 Flash (Experimental)",
                provider=AIProvider.GOOGLE,
                model_name="gemini-2.0-flash-exp",
                endpoint="https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent",
                api_key_env="GEMINI_API_KEY",
                max_tokens=8192,
                context_window=1048576,
                cost_per_1k_tokens=0.000075
            ),
            AIModel(
                id="gemini-1.5-pro",
                name="Gemini 1.5 Pro",
                provider=AIProvider.GOOGLE,
                model_name="gemini-1.5-pro",
                endpoint="https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent",
                api_key_env="GEMINI_API_KEY",
                max_tokens=8192,
                context_window=2097152,
                cost_per_1k_tokens=0.00125
            ),
            AIModel(
                id="gemini-1.5-flash",
                name="Gemini 1.5 Flash",
                provider=AIProvider.GOOGLE,
                model_name="gemini-1.5-flash",
                endpoint="https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent",
                api_key_env="GEMINI_API_KEY",
                max_tokens=8192,
                context_window=1048576,
                cost_per_1k_tokens=0.000075
            ),
            AIModel(
                id="gemini-2.5-pro",
                name="Gemini 2.5 Pro",
                provider=AIProvider.GOOGLE,
                model_name="gemini-2.5-pro",
                endpoint="https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent",
                api_key_env="GEMINI_API_KEY",
                max_tokens=8192,
                context_window=2097152,
                cost_per_1k_tokens=0.00125
            ),
            # xAI Models
            AIModel(
                id="grok-2-1212",
                name="Grok 2 (1212)",
                provider=AIProvider.XAI,
                model_name="grok-2-1212",
                endpoint="https://api.x.ai/v1/chat/completions",
                api_key_env="XAI_API_KEY",
                max_tokens=4096,
                context_window=131072,
                cost_per_1k_tokens=0.002
            ),
            AIModel(
                id="grok-2-vision-1212",
                name="Grok 2 Vision (1212)",
                provider=AIProvider.XAI,
                model_name="grok-2-vision-1212",
                endpoint="https://api.x.ai/v1/chat/completions",
                api_key_env="XAI_API_KEY",
                max_tokens=4096,
                context_window=8192,
                cost_per_1k_tokens=0.002
            ),
            # Perplexity Models
            AIModel(
                id="llama-3.1-sonar-small-128k-online",
                name="Llama 3.1 Sonar Small (Online)",
                provider=AIProvider.PERPLEXITY,
                model_name="llama-3.1-sonar-small-128k-online",
                endpoint="https://api.perplexity.ai/chat/completions",
                api_key_env="PERPLEXITY_API_KEY",
                max_tokens=4096,
                context_window=128000,
                cost_per_1k_tokens=0.0002
            ),
            # Ollama Local Models - UPDATED CONFIGURATION
            AIModel(
                id="ollama-llama3.1",
                name="Ollama Llama 3.1 (Local)",
                provider=AIProvider.OLLAMA,
                model_name="llama3.1:latest",  # FIXED: Added :latest tag
                endpoint="http://localhost:11434/api/chat",  # FIXED: Use /api/chat endpoint
                api_key_env="",
                max_tokens=4096,
                context_window=131072,
                cost_per_1k_tokens=0.0,
                supports_streaming=True,  # Ensure streaming is supported
                capabilities=[
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.REASONING
                ],
                model_type="llm",
                input_modalities=["text"],
                output_modalities=["text"],
                deployment_type="local"
            ),
            # ElevenLabs Models
            AIModel(
                id="eleven-multilingual-v2",
                name="ElevenLabs Multilingual V2",
                provider=AIProvider.ELEVENLABS,
                model_name="eleven_multilingual_v2",
                endpoint="https://api.elevenlabs.io/v1/text-to-speech",
                api_key_env="ELEVENLABS_API_KEY",
                max_tokens=5000,
                cost_per_1k_tokens=0.015,
                capabilities=[
                    ModelCapability.AUDIO_GENERATION
                ],
                model_type="audio",
                input_modalities=["text"],
                output_modalities=["audio"]
            ),
            AIModel(
                id="eleven-turbo-v2_5",
                name="ElevenLabs Turbo V2.5",
                provider=AIProvider.ELEVENLABS,
                model_name="eleven_turbo_v2_5",
                endpoint="https://api.elevenlabs.io/v1/text-to-speech",
                api_key_env="ELEVENLABS_API_KEY",
                max_tokens=5000,
                cost_per_1k_tokens=0.010,
                capabilities=[
                    ModelCapability.AUDIO_GENERATION
                ],
                model_type="audio",
                input_modalities=["text"],
                output_modalities=["audio"]
            ),
            AIModel(
                id="eleven-monolingual-v1",
                name="ElevenLabs English V1",
                provider=AIProvider.ELEVENLABS,
                model_name="eleven_monolingual_v1",
                endpoint="https://api.elevenlabs.io/v1/text-to-speech",
                api_key_env="ELEVENLABS_API_KEY",
                max_tokens=5000,
                cost_per_1k_tokens=0.012,
                capabilities=[
                    ModelCapability.AUDIO_GENERATION
                ],
                model_type="audio",
                input_modalities=["text"],
                output_modalities=["audio"]
            )
        ]
        
        return {model.id: model for model in models}
    
    def get_models(self) -> List[AIModel]:
        """Get all available models"""
        return list(self.models.values())
    
    def get_model(self, model_id: str) -> Optional[AIModel]:
        """Get a specific model by ID"""
        return self.models.get(model_id)
        
    def get_all_models(self) -> List[AIModel]:
        """Get all available models (alias for get_models)"""
        return self.get_models()

    def initialize_default_models(self):
        """Initialize default models and set first one as active"""
        if not self.active_model_id and self.models:
            # Set the first model as active by default
            first_model_id = list(self.models.keys())[0]
            self.set_active_model(first_model_id)
    
    def set_active_model(self, model_id: str) -> bool:
        """Set the active model"""
        if model_id in self.models:
            self.active_model_id = model_id
            self.models[model_id].is_active = True
            # Deactivate other models
            for mid, model in self.models.items():
                if mid != model_id:
                    model.is_active = False
            return True
        return False
    
    def add_custom_model(self, model_id: str, name: str, endpoint: str, 
                        api_key_env: str = "", model_name: str = "", 
                        max_tokens: int = 4096, temperature: float = 0.7,
                        custom_headers: Dict[str, str] = None) -> AIModel:
        """Add a custom model"""
        model = AIModel(
            id=model_id,
            name=name,
            provider=AIProvider.CUSTOM,
            model_name=model_name or model_id,
            endpoint=endpoint,
            api_key_env=api_key_env,
            max_tokens=max_tokens,
            temperature=temperature,
            custom_headers=custom_headers or {}
        )
        self.models[model_id] = model
        return model
    
    def remove_model(self, model_id: str) -> bool:
        """Remove a model"""
        if model_id in self.models:
            del self.models[model_id]
            if self.active_model_id == model_id:
                self.active_model_id = None
            return True
        return False
    
    async def generate_response(self, model_id: str, query: str, system_message: str = None, 
                              user_id: str = "anonymous", stream: bool = False) -> Dict[str, Any]:
        """
        Generate AI response with optional streaming support
        
        Args:
            model_id: ID of the model to use
            query: User query
            system_message: System message (optional)
            user_id: User identifier for caching
            stream: Whether to use streaming mode (default: False)
        """
        start_time = time.time()
        
        # Get the model
        model = self.get_model(model_id)
        if not model:
            return {"error": f"Model '{model_id}' not found", "status": "error"}
        
        if not model.is_active:
            return {"error": f"Model '{model_id}' is not active", "status": "error"}
        
        # Check cache first (only for non-streaming requests)
        if not stream:
            cached_response = self.cache_manager.get(
                query=query,
                model_id=model_id,
                system_message=system_message
            )
            if cached_response:
                logger.info(f"Cache hit for model {model_id}")
                return {
                    "response": cached_response,
                    "status": "success",
                    "model": model_id,
                    "cached": True,
                    "response_time": time.time() - start_time
                }
        
        # Get API key (skip for local models)
        api_key = None
        if model.provider != AIProvider.OLLAMA and model.api_key_env:
            api_key = os.environ.get(model.api_key_env)
            if not api_key:
                return {"error": f"API key not configured for {model.provider.value}", "status": "error"}
        
        # Route to appropriate handler with stream parameter
        try:
            if model.provider == AIProvider.OPENAI:
                result = await self._handle_openai(model, query, system_message, api_key, stream=stream)
            elif model.provider == AIProvider.ANTHROPIC:
                result = await self._handle_anthropic(model, query, system_message, api_key, stream=stream)
            elif model.provider == AIProvider.GOOGLE:
                result = await self._handle_google(model, query, system_message, api_key, stream=stream)
            elif model.provider == AIProvider.XAI:
                result = await self._handle_xai(model, query, system_message, api_key, stream=stream)
            elif model.provider == AIProvider.PERPLEXITY:
                result = await self._handle_perplexity(model, query, system_message, api_key, stream=stream)
            elif model.provider == AIProvider.OLLAMA:
                result = await self._handle_ollama(model, query, system_message, stream=stream)
            elif model.provider == AIProvider.COHERE:
                result = await self._handle_cohere(model, query, system_message, api_key, stream=stream)
            elif model.provider == AIProvider.MISTRAL:
                result = await self._handle_mistral(model, query, system_message, api_key, stream=stream)
            elif model.provider == AIProvider.ELEVENLABS:
                result = await self._handle_elevenlabs(model, query, system_message, api_key)
            elif model.provider == AIProvider.CUSTOM:
                result = await self._handle_custom(model, query, system_message, api_key, stream=stream)
            else:
                return {"error": f"Unsupported provider: {model.provider.value}", "status": "error"}
            
            # Cache successful non-streaming responses
            if not stream and result.get("status") == "success" and "response" in result:
                metadata = {
                    "user_id": user_id,
                    "provider": model.provider.value,
                    "model_name": model.model_name,
                    "timestamp": datetime.now().isoformat(),
                    "cost_per_1k_tokens": model.cost_per_1k_tokens,
                    "response_time": time.time() - start_time
                }
                self.cache_manager.set(
                    query=query,
                    model_id=model_id,
                    response=result["response"],
                    system_message=system_message,
                    metadata=metadata
                )
            
            # Add response time to result
            result["response_time"] = time.time() - start_time
            return result
            
        except Exception as e:
            logger.error(f"Error in generate_response for {model_id}: {e}")
            return {
                "error": f"Internal error: {str(e)}",
                "status": "error",
                "model": model_id,
                "response_time": time.time() - start_time
            }
    
    async def _handle_openai(self, model: AIModel, query: str, system_message: str, api_key: str, stream: bool = False) -> Dict[str, Any]:
        """Handle OpenAI API requests with optional streaming"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": query})
        
        payload = {
            "model": model.model_name,
            "messages": messages,
            "max_tokens": model.max_tokens,
            "temperature": model.temperature,
            "top_p": model.top_p,
            "stream": stream
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(model.endpoint, headers=headers, json=payload) as response:
                if response.status == 200:
                    if stream:
                        return await self._handle_openai_streaming(response, model)
                    else:
                        data = await response.json()
                        return {
                            "response": data["choices"][0]["message"]["content"],
                            "status": "success",
                            "model": model.id,
                            "usage": data.get("usage", {}),
                            "streaming": False
                        }
                else:
                    error_data = await response.json()
                    return {"error": error_data.get("error", {}).get("message", "Unknown error"), "status": "error"}
    
    async def _handle_openai_streaming(self, response, model: AIModel) -> Dict[str, Any]:
        """Handle streaming response from OpenAI API"""
        full_content = ""
        usage_stats = {}
        
        try:
            async for line in response.content:
                line = line.decode('utf-8').strip()
                if not line:
                    continue
                    
                if line.startswith('data: '):
                    line = line[6:]  # Remove 'data: ' prefix
                    
                if line == '[DONE]':
                    break
                    
                try:
                    chunk_data = json.loads(line)
                    
                    if 'choices' in chunk_data and len(chunk_data['choices']) > 0:
                        delta = chunk_data['choices'][0].get('delta', {})
                        if 'content' in delta:
                            full_content += delta['content']
                    
                    # Capture usage from final chunk
                    if 'usage' in chunk_data:
                        usage_stats = chunk_data['usage']
                        
                except json.JSONDecodeError:
                    continue
                    
            return {
                "response": full_content,
                "status": "success",
                "model": model.id,
                "usage": usage_stats,
                "streaming": True
            }
            
        except Exception as e:
            logger.error(f"Error processing OpenAI streaming response: {e}")
            return {
                "error": f"Error processing streaming response: {str(e)}",
                "status": "error",
                "model": model.id
            }
    
    async def _handle_anthropic(self, model: AIModel, query: str, system_message: str, api_key: str, stream: bool = False) -> Dict[str, Any]:
        """Handle Anthropic API requests with optional streaming"""
        headers = {
            "x-api-key": api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        
        payload = {
            "model": model.model_name,
            "max_tokens": model.max_tokens,
            "temperature": model.temperature,
            "top_p": model.top_p,
            "stream": stream
        }
        
        if system_message:
            payload["system"] = system_message
            payload["messages"] = [{"role": "user", "content": query}]
        else:
            payload["messages"] = [{"role": "user", "content": query}]
        
        async with aiohttp.ClientSession() as session:
            async with session.post(model.endpoint, headers=headers, json=payload) as response:
                if response.status == 200:
                    if stream:
                        return await self._handle_anthropic_streaming(response, model)
                    else:
                        data = await response.json()
                        return {
                            "response": data["content"][0]["text"],
                            "status": "success",
                            "model": model.id,
                            "usage": data.get("usage", {}),
                            "streaming": False
                        }
                else:
                    error_data = await response.json()
                    return {"error": error_data.get("error", {}).get("message", "Unknown error"), "status": "error"}
    
    async def _handle_anthropic_streaming(self, response, model: AIModel) -> Dict[str, Any]:
        """Handle streaming response from Anthropic API"""
        full_content = ""
        usage_stats = {}
        
        try:
            async for line in response.content:
                line = line.decode('utf-8').strip()
                if not line:
                    continue
                    
                if line.startswith('data: '):
                    line = line[6:]
                    
                try:
                    chunk_data = json.loads(line)
                    
                    if chunk_data.get('type') == 'content_block_delta':
                        delta = chunk_data.get('delta', {})
                        if 'text' in delta:
                            full_content += delta['text']
                    
                    if 'usage' in chunk_data:
                        usage_stats = chunk_data['usage']
                        
                except json.JSONDecodeError:
                    continue
                    
            return {
                "response": full_content,
                "status": "success",
                "model": model.id,
                "usage": usage_stats,
                "streaming": True
            }
            
        except Exception as e:
            logger.error(f"Error processing Anthropic streaming response: {e}")
            return {
                "error": f"Error processing streaming response: {str(e)}",
                "status": "error",
                "model": model.id
            }
    
    async def _handle_google(self, model: AIModel, query: str, system_message: str, api_key: str, stream: bool = False) -> Dict[str, Any]:
        """Handle Google API requests"""
        url = f"{model.endpoint}?key={api_key}"
        headers = {"Content-Type": "application/json"}
        
        contents = []
        if system_message:
            contents.append({"parts": [{"text": system_message}], "role": "user"})
            contents.append({"parts": [{"text": "Understood."}], "role": "model"})
        contents.append({"parts": [{"text": query}], "role": "user"})
        
        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": model.temperature,
                "topP": model.top_p,
                "maxOutputTokens": model.max_tokens
            }
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    if "candidates" in data and len(data["candidates"]) > 0:
                        content = data["candidates"][0]["content"]["parts"][0]["text"]
                        return {
                            "response": content,
                            "status": "success",
                            "model": model.id,
                            "usage": data.get("usageMetadata", {}),
                            "streaming": False
                        }
                    else:
                        return {"error": "No response generated", "status": "error"}
                else:
                    error_data = await response.json()
                    return {"error": error_data.get("error", {}).get("message", "Unknown error"), "status": "error"}
    
    async def _handle_xai(self, model: AIModel, query: str, system_message: str, api_key: str, stream: bool = False) -> Dict[str, Any]:
        """Handle xAI API requests"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": query})
        
        payload = {
            "model": model.model_name,
            "messages": messages,
            "max_tokens": model.max_tokens,
            "temperature": model.temperature,
            "top_p": model.top_p,
            "stream": stream
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(model.endpoint, headers=headers, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "response": data["choices"][0]["message"]["content"],
                        "status": "success",
                        "model": model.id,
                        "usage": data.get("usage", {}),
                        "streaming": False
                    }
                else:
                    error_data = await response.json()
                    return {"error": error_data.get("error", {}).get("message", "Unknown error"), "status": "error"}
    
    async def _handle_perplexity(self, model: AIModel, query: str, system_message: str, api_key: str, stream: bool = False) -> Dict[str, Any]:
        """Handle Perplexity API requests"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": query})
        
        payload = {
            "model": model.model_name,
            "messages": messages,
            "max_tokens": model.max_tokens,
            "temperature": model.temperature,
            "top_p": model.top_p,
            "stream": stream
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(model.endpoint, headers=headers, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "response": data["choices"][0]["message"]["content"],
                        "status": "success",
                        "model": model.id,
                        "usage": data.get("usage", {}),
                        "citations": data.get("citations", []),
                        "streaming": False
                    }
                else:
                    error_data = await response.json()
                    return {"error": error_data.get("error", {}).get("message", "Unknown error"), "status": "error"}
    
    async def _handle_ollama(self, model: AIModel, query: str, system_message: str, stream: bool = False) -> Dict[str, Any]:
        """
        Enhanced Ollama handler with comprehensive error logging and debugging
        """
        headers = {
            "Content-Type": "application/json"
        }
        
        # Build messages array
        messages = []
        if system_message and system_message.strip():
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": query})
        
        # Prepare payload
        payload = {
            "model": model.model_name,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": getattr(model, 'temperature', 0.7),
                "top_p": getattr(model, 'top_p', 0.9),
                "num_predict": getattr(model, 'max_tokens', 4096)
            }
        }
        
        logger.debug(f"Ollama request - Model: {model.model_name}, Stream: {stream}, Endpoint: {model.endpoint}")
        logger.debug(f"Ollama payload: {json.dumps(payload, indent=2)}")
        
        try:
            # Add connection verification
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as session:
                # First, check if Ollama is running
                try:
                    async with session.get(f"{model.endpoint.replace('/api/chat', '')}/api/tags") as health_check:
                        if health_check.status != 200:
                            logger.error(f"Ollama health check failed: {health_check.status}")
                            return {
                                "error": f"Ollama service unhealthy: HTTP {health_check.status}",
                                "status": "error",
                                "model": model.id
                            }
                except Exception as health_error:
                    logger.error(f"Ollama health check exception: {health_error}")
                    logger.error(f"Health check traceback: {traceback.format_exc()}")
                    return {
                        "error": f"Cannot connect to Ollama: {str(health_error)}",
                        "status": "error",
                        "model": model.id
                    }
                
                # Make the actual request
                async with session.post(model.endpoint, headers=headers, json=payload) as response:
                    logger.debug(f"Ollama response status: {response.status}")
                    logger.debug(f"Ollama response headers: {dict(response.headers)}")
                    
                    if response.status == 200:
                        if stream:
                            return await self._handle_ollama_streaming(response, model)
                        else:
                            return await self._handle_ollama_non_streaming(response, model)
                    else:
                        # Enhanced error handling
                        try:
                            error_text = await response.text()
                            logger.error(f"Ollama API error response: {error_text}")
                            try:
                                error_data = json.loads(error_text)
                                error_message = error_data.get("error", f"HTTP {response.status}")
                            except json.JSONDecodeError:
                                error_message = f"HTTP {response.status} - {error_text[:200]}"
                        except Exception as read_error:
                            logger.error(f"Failed to read error response: {read_error}")
                            error_message = f"HTTP {response.status} - Unable to read response"
                        
                        logger.error(f"Ollama API error: {error_message}")
                        return {
                            "error": f"Ollama API error: {error_message}",
                            "status": "error",
                            "model": model.id
                        }
                        
        except aiohttp.ClientConnectorError as e:
            logger.error(f"Cannot connect to Ollama at {model.endpoint}: {e}")
            logger.error(f"Connection error traceback: {traceback.format_exc()}")
            return {
                "error": f"Cannot connect to Ollama at {model.endpoint}. Make sure Ollama is running and accessible. Error: {str(e)}",
                "status": "error",
                "model": model.id
            }
        except aiohttp.ServerTimeoutError as e:
            logger.error(f"Ollama request timeout: {e}")
            logger.error(f"Timeout error traceback: {traceback.format_exc()}")
            return {
                "error": f"Request timeout. The model may be taking too long to respond. Error: {str(e)}",
                "status": "error",
                "model": model.id
            }
        except Exception as e:
            # Enhanced exception logging
            logger.error(f"Unexpected error in Ollama handler: {str(e)}")
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error(f"Exception args: {e.args}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            
            # Additional debugging info
            logger.error(f"Model info: {model.id}, {model.model_name}, {model.endpoint}")
            logger.error(f"Python version: {sys.version}")
            logger.error(f"Query length: {len(query)}")
            logger.error(f"System message length: {len(system_message) if system_message else 0}")
            
            return {
                "error": f"Unexpected error: {str(e)} (Type: {type(e).__name__})",
                "status": "error",
                "model": model.id,
                "debug_info": {
                    "exception_type": type(e).__name__,
                    "exception_args": str(e.args),
                    "model_endpoint": model.endpoint,
                    "query_length": len(query)
                }
            }
    
    async def _handle_ollama_non_streaming(self, response, model: AIModel) -> Dict[str, Any]:
        """
        Enhanced non-streaming response handler with better error logging
        """
        try:
            # Read raw response first for debugging
            raw_response = await response.text()
            logger.debug(f"Raw Ollama response: {raw_response[:500]}...")
            
            # Parse JSON
            try:
                data = json.loads(raw_response)
            except json.JSONDecodeError as json_error:
                logger.error(f"Failed to parse Ollama JSON response: {json_error}")
                logger.error(f"Raw response causing JSON error: {raw_response}")
                return {
                    "error": f"Invalid JSON response from Ollama: {str(json_error)}",
                    "status": "error",
                    "model": model.id
                }
            
            # Extract response content with validation
            if "message" in data and "content" in data["message"]:
                content = data["message"]["content"]
                if not content or not isinstance(content, str):
                    logger.warning(f"Empty or invalid content from Ollama: {content}")
                    content = "[Empty response from model]"
            else:
                logger.error(f"Unexpected Ollama response format: {data}")
                logger.error(f"Available keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
                return {
                    "error": f"Unexpected response format from Ollama. Expected 'message.content', got: {list(data.keys()) if isinstance(data, dict) else type(data)}",
                    "status": "error",
                    "model": model.id
                }
            
            # Extract usage statistics with defaults
            usage_stats = {
                "total_duration": data.get("total_duration", 0),
                "load_duration": data.get("load_duration", 0),
                "prompt_eval_count": data.get("prompt_eval_count", 0),
                "eval_count": data.get("eval_count", 0),
                "prompt_eval_duration": data.get("prompt_eval_duration", 0),
                "eval_duration": data.get("eval_duration", 0)
            }
            
            logger.debug(f"Ollama response processed successfully. Content length: {len(content)}")
            
            return {
                "response": content,
                "status": "success",
                "model": model.id,
                "usage": usage_stats,
                "streaming": False
            }
            
        except Exception as e:
            logger.error(f"Error processing non-streaming Ollama response: {str(e)}")
            logger.error(f"Processing error traceback: {traceback.format_exc()}")
            return {
                "error": f"Error processing response: {str(e)}",
                "status": "error",
                "model": model.id
            }
    
    async def _handle_ollama_streaming(self, response, model: AIModel) -> Dict[str, Any]:
        """
        Handle streaming response from Ollama API
        Collects all chunks and combines the content
        """
        full_content = ""
        usage_stats = {
            "total_duration": 0,
            "load_duration": 0,
            "prompt_eval_count": 0,
            "eval_count": 0
        }
        
        try:
            async for line in response.content:
                line = line.decode('utf-8').strip()
                if not line:
                    continue
                    
                try:
                    # Parse each JSON chunk
                    chunk_data = json.loads(line)
                    
                    # Extract content from the chunk
                    if "message" in chunk_data and "content" in chunk_data["message"]:
                        content = chunk_data["message"]["content"]
                        full_content += content
                    
                    # Update usage statistics from the final chunk
                    if chunk_data.get("done", False):
                        usage_stats.update({
                            "total_duration": chunk_data.get("total_duration", 0),
                            "load_duration": chunk_data.get("load_duration", 0),
                            "prompt_eval_count": chunk_data.get("prompt_eval_count", 0),
                            "eval_count": chunk_data.get("eval_count", 0)
                        })
                        break
                        
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse streaming chunk: {line[:100]}... Error: {e}")
                    continue
                    
            return {
                "response": full_content,
                "status": "success",
                "model": model.id,
                "usage": usage_stats,
                "streaming": True
            }
            
        except Exception as e:
            logger.error(f"Error processing streaming response: {e}")
            return {
                "error": f"Error processing streaming response: {str(e)}",
                "status": "error",
                "model": model.id
            }
    
    async def _handle_cohere(self, model: AIModel, query: str, system_message: str, api_key: str, stream: bool = False) -> Dict[str, Any]:
        """Handle Cohere API requests"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model.model_name,
            "message": query,
            "max_tokens": model.max_tokens,
            "temperature": model.temperature,
            "p": model.top_p,
            "stream": stream
        }
        
        if system_message:
            payload["preamble"] = system_message
        
        async with aiohttp.ClientSession() as session:
            async with session.post(model.endpoint, headers=headers, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "response": data["text"],
                        "status": "success",
                        "model": model.id,
                        "usage": data.get("meta", {}).get("billed_units", {}),
                        "streaming": False
                    }
                else:
                    error_data = await response.json()
                    return {"error": error_data.get("message", "Unknown error"), "status": "error"}
    
    async def _handle_mistral(self, model: AIModel, query: str, system_message: str, api_key: str, stream: bool = False) -> Dict[str, Any]:
        """Handle Mistral API requests"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": query})
        
        payload = {
            "model": model.model_name,
            "messages": messages,
            "max_tokens": model.max_tokens,
            "temperature": model.temperature,
            "top_p": model.top_p,
            "stream": stream
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(model.endpoint, headers=headers, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "response": data["choices"][0]["message"]["content"],
                        "status": "success",
                        "model": model.id,
                        "usage": data.get("usage", {}),
                        "streaming": False
                    }
                else:
                    error_data = await response.json()
                    return {"error": error_data.get("error", {}).get("message", "Unknown error"), "status": "error"}
    
    async def _handle_elevenlabs(self, model: AIModel, query: str, system_message: str, api_key: str) -> Dict[str, Any]:
        """Handle ElevenLabs API requests for text-to-speech"""
        # ElevenLabs is for audio generation, so we handle it differently
        voice_id = "21m00Tcm4TlvDq8ikWAM"  # Default voice
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": api_key
        }
        
        payload = {
            "text": query,
            "model_id": model.model_name,
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.5
            }
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status == 200:
                    audio_data = await response.read()
                    return {
                        "response": "Audio generated successfully",
                        "status": "success",
                        "model": model.id,
                        "audio_data": audio_data,
                        "content_type": "audio/mpeg"
                    }
                else:
                    error_data = await response.json()
                    return {"error": error_data.get("detail", {}).get("message", "Unknown error"), "status": "error"}
    
    async def _handle_custom(self, model: AIModel, query: str, system_message: str, api_key: str, stream: bool = False) -> Dict[str, Any]:
        """Handle custom API requests"""
        headers = {
            "Content-Type": "application/json"
        }
        
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        
        if model.custom_headers:
            headers.update(model.custom_headers)
        
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": query})
        
        payload = {
            "model": model.model_name,
            "messages": messages,
            "max_tokens": model.max_tokens,
            "temperature": model.temperature,
            "top_p": model.top_p,
            "stream": stream
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(model.endpoint, headers=headers, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    # Try to extract response in common formats
                    content = ""
                    if "choices" in data and len(data["choices"]) > 0:
                        content = data["choices"][0]["message"]["content"]
                    elif "response" in data:
                        content = data["response"]
                    elif "text" in data:
                        content = data["text"]
                    
                    return {
                        "response": content,
                        "status": "success",
                        "model": model.id,
                        "usage": data.get("usage", {}),
                        "streaming": False
                    }
                else:
                    try:
                        error_data = await response.json()
                        error_message = error_data.get("error", {}).get("message", "Unknown error")
                    except:
                        error_message = f"HTTP {response.status}"
                    return {"error": error_message, "status": "error"}


# Global model manager instance
ai_model_manager = AIModelManager()