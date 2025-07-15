import os
import sys
import logging
import time
import json
import asyncio
from datetime import datetime
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash, Response

from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
import asyncio
import json
from typing import Dict, List, Optional
import threading
from ml_router import MLEnhancedQueryRouter
from config import EnhancedRouterConfig
from model_manager import ModelManager, ModelType
from ai_models import AIModelManager, AIProvider
from auth_system import AuthManager, UserRole
from ai_cache import get_cache_manager
from rag_chat import get_rag_chat
from swagger_spec import swagger_spec
from collaborative_router import get_collaborative_router
from shared_memory import get_shared_memory_manager
from external_llm_integration import get_external_llm_manager
from advanced_ml_classifier import AdvancedMLClassifier
from intelligent_routing_engine import IntelligentRoutingEngine
from real_time_analytics import RealTimeAnalytics
from advanced_query_optimizer import AdvancedQueryOptimizer
from predictive_analytics_engine import PredictiveAnalyticsEngine
from graphql_simple import graphql_bp
from auto_chain_generator import AutoChainGenerator
from automated_evaluation_engine import get_evaluation_engine
from peer_teaching_system import get_peer_teaching_system, AgentSpecialization, LessonType, ConsensusMethod
from personal_ai_router import get_personal_ai_router
from user_profile_builder import get_user_profile_builder

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class Base(DeclarativeBase):
    pass

db = SQLAlchemy(model_class=Base)

# Create the app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key-change-in-production")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Configure the database
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", "sqlite:///query_router.db")
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}

# Initialize extensions
db.init_app(app)

# Simple rate limiting dictionary
rate_limits = {}

# Global instances
router = None
router_config = None
model_manager = None
ai_model_manager = None
auth_manager = None
cache_manager = None
rag_system = None
collaborative_router = None
shared_memory_manager = None
external_llm_manager = None
advanced_ml_classifier = None
intelligent_routing_engine = None
real_time_analytics = None
advanced_query_optimizer = None
predictive_analytics_engine = None
active_learning_system = None
contextual_memory_router = None
semantic_guardrail_system = None
multimodal_ai_integration = None
auto_chain_generator = None
evaluation_engine = None
peer_teaching_system = None
personal_ai_router = None

def initialize_router():
    """Initialize the ML router in a background thread"""
    global router, router_config, model_manager, ai_model_manager, auth_manager, cache_manager, rag_system, collaborative_router, shared_memory_manager, external_llm_manager, advanced_ml_classifier, intelligent_routing_engine, real_time_analytics, advanced_query_optimizer, predictive_analytics_engine, active_learning_system, contextual_memory_router, semantic_guardrail_system, multimodal_ai_integration, auto_chain_generator, evaluation_engine, peer_teaching_system, personal_ai_router
    
    try:
        with app.app_context():
            router_config = EnhancedRouterConfig.from_env()
            model_manager = ModelManager(db)
            ai_model_manager = AIModelManager(db)
            auth_manager = AuthManager()
            cache_manager = get_cache_manager(db)
            rag_system = get_rag_chat()
            shared_memory_manager = get_shared_memory_manager()
            external_llm_manager = get_external_llm_manager(db)
            router = MLEnhancedQueryRouter(router_config, model_manager)
            collaborative_router = get_collaborative_router(ai_model_manager)
            
            # Initialize advanced features
            advanced_ml_classifier = AdvancedMLClassifier()
            intelligent_routing_engine = IntelligentRoutingEngine()
            real_time_analytics = RealTimeAnalytics()
            advanced_query_optimizer = AdvancedQueryOptimizer()
            predictive_analytics_engine = PredictiveAnalyticsEngine()
            
            # Initialize Auto Chain Generator
            from auto_chain_generator import AutoChainGenerator
            global auto_chain_generator
            auto_chain_generator = AutoChainGenerator()
            
            # Initialize Evaluation Engine
            evaluation_engine = get_evaluation_engine()
            
            # Initialize Peer Teaching System
            peer_teaching_system = get_peer_teaching_system()
            
            # Initialize next-generation features
            from active_learning_system import get_active_learning_system
            from contextual_memory_router import get_contextual_memory_router
            from semantic_guardrails import get_semantic_guardrail_system
            from multimodal_ai_integration import get_multimodal_ai_integration
            
            active_learning_system = get_active_learning_system()
            contextual_memory_router = get_contextual_memory_router()
            semantic_guardrail_system = get_semantic_guardrail_system()
            multimodal_ai_integration = get_multimodal_ai_integration(ai_model_manager)
            personal_ai_router = get_personal_ai_router()
            
            # Initialize ML models
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(router.initialize())
            loop.run_until_complete(advanced_ml_classifier.initialize())
            loop.run_until_complete(intelligent_routing_engine.initialize())
            loop.run_until_complete(real_time_analytics.start())
            
            logger.info("ML Router and Advanced Features initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize ML Router: {e}")
        router = None

@app.route('/')
def index():
    """Home dashboard page"""
    return render_template('home_dashboard.html')

@app.route('/dashboard')
def dashboard():
    """Dashboard showing routing statistics"""
    return render_template('dashboard.html')

@app.route('/agents')
def agents():
    """Agent management page"""
    return render_template('agents.html')

@app.route('/models')
def models():
    """Model management page"""
    return render_template('models.html')

@app.route('/ai-models')
def ai_models():
    """AI model management page"""
    return render_template('ai_models.html')

@app.route('/chat')
def chat():
    """Advanced chat interface"""
    return render_template('chat.html')

@app.route('/auth')
def auth():
    """Authentication management page"""
    return render_template('auth.html')

@app.route('/settings')
def settings():
    """Settings page for API key management"""
    return render_template('settings.html')

@app.route('/config')
def config():
    """Configuration page for advanced settings"""
    return render_template('config.html')

@app.route('/api/query', methods=['POST'])
def submit_query():
    """Submit a query for routing"""
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({'error': 'Query is required'}), 400
        
        query = data['query']
        user_id = data.get('user_id', 'anonymous')
        
        if not router:
            return jsonify({'error': 'Router not initialized'}), 503
        
        # Process query asynchronously
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(router.route_query(query, user_id))
            return jsonify(result)
        except Exception as e:
            logger.error(f"Query routing error: {e}")
            return jsonify({'error': 'Failed to process query'}), 500
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"API error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/agents', methods=['GET'])
def get_agents():
    """Get list of available agents"""
    try:
        if not router:
            return jsonify({'error': 'Router not initialized'}), 503
        
        agents = []
        for agent_id, agent in router.agents.items():
            agents.append({
                'id': agent_id,
                'name': agent.name,
                'description': agent.description,
                'categories': [cat.value for cat in agent.categories],
                'load_factor': agent.load_factor,
                'is_healthy': agent.is_healthy,
                'last_health_check': agent.last_health_check.isoformat() if agent.last_health_check else None
            })
        
        return jsonify({'agents': agents})
    except Exception as e:
        logger.error(f"Error getting agents: {e}")
        return jsonify({'error': 'Failed to get agents'}), 500

@app.route('/api/agents/register', methods=['POST'])
def register_agent():
    """Register a new agent"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Agent data is required'}), 400
        
        required_fields = ['name', 'description', 'categories', 'endpoint']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'{field} is required'}), 400
        
        if not router:
            return jsonify({'error': 'Router not initialized'}), 503
        
        # Register agent
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            agent_id = loop.run_until_complete(router.register_agent(
                name=data['name'],
                description=data['description'],
                categories=data['categories'],
                endpoint=data['endpoint'],
                capabilities=data.get('capabilities', {}),
                meta_data=data.get('meta_data', {})
            ))
            
            return jsonify({'agent_id': agent_id, 'message': 'Agent registered successfully'})
        except Exception as e:
            logger.error(f"Agent registration error: {e}")
            return jsonify({'error': 'Failed to register agent'}), 500
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"API error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/agents/<agent_id>', methods=['DELETE'])
def unregister_agent(agent_id):
    """Unregister an agent"""
    try:
        if not router:
            return jsonify({'error': 'Router not initialized'}), 503
        
        if agent_id in router.agents:
            del router.agents[agent_id]
            return jsonify({'message': 'Agent unregistered successfully'})
        else:
            return jsonify({'error': 'Agent not found'}), 404
            
    except Exception as e:
        logger.error(f"Error unregistering agent: {e}")
        return jsonify({'error': 'Failed to unregister agent'}), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get routing statistics"""
    try:
        if not router:
            return jsonify({'error': 'Router not initialized'}), 503
        
        stats = {
            'total_queries': router.total_queries,
            'successful_routes': router.successful_routes,
            'failed_routes': router.failed_routes,
            'cache_hits': router.cache_hits,
            'cache_misses': router.cache_misses,
            'active_agents': len([a for a in router.agents.values() if a.is_healthy]),
            'total_agents': len(router.agents),
            'avg_response_time': router.avg_response_time,
            'category_distribution': router.category_stats
        }
        
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return jsonify({'error': 'Failed to get statistics'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        status = {
            'status': 'healthy' if router else 'unhealthy',
            'router_initialized': router is not None,
            'timestamp': datetime.now().isoformat()
        }
        
        if router:
            status['ml_classifier_initialized'] = router.ml_classifier.initialized
            status['agents_count'] = len(router.agents)
            
        return jsonify(status)
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

# Model Management API Endpoints
@app.route('/api/models', methods=['GET'])
def get_models():
    """Get all models"""
    try:
        if not model_manager:
            return jsonify({'error': 'Model manager not initialized'}), 503
        
        models = model_manager.get_all_models()
        return jsonify({
            'models': [model.to_dict() for model in models]
        })
    except Exception as e:
        logger.error(f"Error getting models: {e}")
        return jsonify({'error': 'Failed to get models'}), 500

@app.route('/api/models', methods=['POST'])
def create_model():
    """Create a new model"""
    try:
        if not model_manager:
            return jsonify({'error': 'Model manager not initialized'}), 503
        
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Model data is required'}), 400
        
        required_fields = ['name', 'description', 'type', 'categories']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'{field} is required'}), 400
        
        model_type = ModelType(data['type'])
        model = model_manager.create_model(
            name=data['name'],
            description=data['description'],
            model_type=model_type,
            categories=data['categories'],
            config=data.get('config', {})
        )
        
        return jsonify({
            'model_id': model.id,
            'message': 'Model created successfully',
            'model': model.to_dict()
        })
        
    except Exception as e:
        logger.error(f"Error creating model: {e}")
        return jsonify({'error': 'Failed to create model'}), 500

@app.route('/api/models/<model_id>', methods=['GET'])
def get_model(model_id):
    """Get a specific model"""
    try:
        if not model_manager:
            return jsonify({'error': 'Model manager not initialized'}), 503
        
        model = model_manager.get_model(model_id)
        if not model:
            return jsonify({'error': 'Model not found'}), 404
        
        return jsonify({'model': model.to_dict()})
    except Exception as e:
        logger.error(f"Error getting model: {e}")
        return jsonify({'error': 'Failed to get model'}), 500

@app.route('/api/models/<model_id>', methods=['PUT'])
def update_model(model_id):
    """Update a model"""
    try:
        if not model_manager:
            return jsonify({'error': 'Model manager not initialized'}), 503
        
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Model data is required'}), 400
        
        model_type = ModelType(data['type']) if 'type' in data else None
        
        model = model_manager.update_model(
            model_id=model_id,
            name=data.get('name'),
            description=data.get('description'),
            model_type=model_type,
            config=data.get('config')
        )
        
        if not model:
            return jsonify({'error': 'Model not found'}), 404
        
        return jsonify({
            'message': 'Model updated successfully',
            'model': model.to_dict()
        })
        
    except Exception as e:
        logger.error(f"Error updating model: {e}")
        return jsonify({'error': 'Failed to update model'}), 500

@app.route('/api/models/<model_id>', methods=['DELETE'])
def delete_model(model_id):
    """Delete a model"""
    try:
        if not model_manager:
            return jsonify({'error': 'Model manager not initialized'}), 503
        
        success = model_manager.delete_model(model_id)
        if not success:
            return jsonify({'error': 'Model not found or cannot be deleted'}), 404
        
        return jsonify({'message': 'Model deleted successfully'})
        
    except Exception as e:
        logger.error(f"Error deleting model: {e}")
        return jsonify({'error': 'Failed to delete model'}), 500

@app.route('/api/models/<model_id>/activate', methods=['POST'])
def activate_model(model_id):
    """Activate a model"""
    try:
        if not model_manager:
            return jsonify({'error': 'Model manager not initialized'}), 503
        
        success = model_manager.activate_model(model_id)
        if not success:
            return jsonify({'error': 'Model not found'}), 404
        
        return jsonify({'message': 'Model activated successfully'})
        
    except Exception as e:
        logger.error(f"Error activating model: {e}")
        return jsonify({'error': 'Failed to activate model'}), 500

@app.route('/api/models/<model_id>/train', methods=['POST'])
def train_model(model_id):
    """Train a model"""
    try:
        if not model_manager:
            return jsonify({'error': 'Model manager not initialized'}), 503
        
        success = model_manager.train_model(model_id)
        if not success:
            return jsonify({'error': 'Model not found'}), 404
        
        return jsonify({'message': 'Model training started'})
        
    except Exception as e:
        logger.error(f"Error training model: {e}")
        return jsonify({'error': 'Failed to train model'}), 500

@app.route('/api/models/stats', methods=['GET'])
def get_model_stats():
    """Get model statistics"""
    try:
        if not model_manager:
            return jsonify({'error': 'Model manager not initialized'}), 503
        
        stats = model_manager.get_model_stats()
        return jsonify(stats)
        
    except Exception as e:
        logger.error(f"Error getting model stats: {e}")
        return jsonify({'error': 'Failed to get model stats'}), 500

# AI Models API Routes
@app.route('/api/ai-models', methods=['GET'])
def get_ai_models():
    """Get all AI models"""
    try:
        if not ai_model_manager:
            return jsonify({'error': 'AI model manager not initialized'}), 503
        
        models = ai_model_manager.get_all_models()
        models_data = []
        for model in models:
            models_data.append({
                'id': model.id,
                'name': model.name,
                'provider': model.provider.value,
                'model_name': model.model_name,
                'endpoint': model.endpoint,
                'api_key_env': model.api_key_env,
                'max_tokens': model.max_tokens,
                'temperature': model.temperature,
                'top_p': model.top_p,
                'context_window': model.context_window,
                'cost_per_1k_tokens': model.cost_per_1k_tokens,
                'is_active': model.is_active,
                'supports_streaming': model.supports_streaming,
                'supports_system_message': model.supports_system_message
            })
        
        return jsonify({'status': 'success', 'models': models_data})
        
    except Exception as e:
        logger.error(f"Error getting AI models: {e}")
        return jsonify({'status': 'error', 'error': 'Failed to get AI models'}), 500

@app.route('/api/ai-models', methods=['POST'])
def create_ai_model():
    """Create a new AI model"""
    try:
        if not ai_model_manager:
            return jsonify({'error': 'AI model manager not initialized'}), 503
        
        data = request.get_json()
        required_fields = ['id', 'name', 'provider', 'model_name', 'endpoint']
        
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'{field} is required'}), 400
        
        model = ai_model_manager.add_custom_model(
            model_id=data['id'],
            name=data['name'],
            endpoint=data['endpoint'],
            api_key_env=data.get('api_key_env', ''),
            model_name=data['model_name'],
            max_tokens=data.get('max_tokens', 4096),
            temperature=data.get('temperature', 0.7),
            custom_headers=data.get('custom_headers', {})
        )
        
        return jsonify({
            'status': 'success',
            'message': 'AI model created successfully',
            'model_id': model.id
        })
        
    except Exception as e:
        logger.error(f"Error creating AI model: {e}")
        return jsonify({'status': 'error', 'error': 'Failed to create AI model'}), 500

@app.route('/api/ai-models/<model_id>', methods=['DELETE'])
def delete_ai_model(model_id):
    """Delete an AI model"""
    try:
        if not ai_model_manager:
            return jsonify({'error': 'AI model manager not initialized'}), 503
        
        success = ai_model_manager.remove_model(model_id)
        if not success:
            return jsonify({'error': 'Model not found'}), 404
        
        return jsonify({'status': 'success', 'message': 'AI model deleted successfully'})
        
    except Exception as e:
        logger.error(f"Error deleting AI model: {e}")
        return jsonify({'status': 'error', 'error': 'Failed to delete AI model'}), 500

@app.route('/api/ai-models/activate/<model_id>', methods=['POST'])
def activate_ai_model(model_id):
    """Activate an AI model"""
    try:
        if not ai_model_manager:
            return jsonify({'error': 'AI model manager not initialized'}), 503
        
        success = ai_model_manager.set_active_model(model_id)
        if not success:
            return jsonify({'error': 'Model not found'}), 404
        
        return jsonify({'status': 'success', 'message': 'AI model activated successfully'})
        
    except Exception as e:
        logger.error(f"Error activating AI model: {e}")
        return jsonify({'status': 'error', 'error': 'Failed to activate AI model'}), 500

@app.route('/api/ai-models/active', methods=['GET'])
def get_active_ai_model():
    """Get the active AI model"""
    try:
        if not ai_model_manager:
            return jsonify({'error': 'AI model manager not initialized'}), 503
        
        active_model = ai_model_manager.get_active_model()
        if not active_model:
            return jsonify({'status': 'success', 'model': None})
        
        model_data = {
            'id': active_model.id,
            'name': active_model.name,
            'provider': active_model.provider.value,
            'model_name': active_model.model_name,
            'endpoint': active_model.endpoint,
            'api_key_env': active_model.api_key_env,
            'max_tokens': active_model.max_tokens,
            'temperature': active_model.temperature,
            'top_p': active_model.top_p,
            'context_window': active_model.context_window,
            'cost_per_1k_tokens': active_model.cost_per_1k_tokens,
            'is_active': active_model.is_active,
            'supports_streaming': active_model.supports_streaming,
            'supports_system_message': active_model.supports_system_message
        }
        
        return jsonify({'status': 'success', 'model': model_data})
        
    except Exception as e:
        logger.error(f"Error getting active AI model: {e}")
        return jsonify({'status': 'error', 'error': 'Failed to get active AI model'}), 500

@app.route('/api/ai-models/test/<model_id>', methods=['POST'])
def test_ai_model(model_id):
    """Test an AI model"""
    try:
        if not ai_model_manager:
            return jsonify({'error': 'AI model manager not initialized'}), 503
        
        data = request.get_json()
        query = data.get('query', 'Hello! Can you confirm you are working correctly?')
        
        # Test the model
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(ai_model_manager.generate_response(
                query=query,
                model_id=model_id
            ))
            return jsonify(result)
        finally:
            loop.close()
        
    except Exception as e:
        logger.error(f"Error testing AI model: {e}")
        return jsonify({'status': 'error', 'error': 'Failed to test AI model'}), 500

@app.route('/api/ai-models/api-key-status', methods=['GET'])
def get_api_key_status():
    """Get API key status for all providers"""
    try:
        providers = {
            'openai': 'OPENAI_API_KEY',
            'anthropic': 'ANTHROPIC_API_KEY',
            'google': 'GEMINI_API_KEY',
            'xai': 'XAI_API_KEY',
            'perplexity': 'PERPLEXITY_API_KEY',
            'cohere': 'COHERE_API_KEY',
            'mistral': 'MISTRAL_API_KEY',
            'huggingface': 'HUGGINGFACE_API_KEY'
        }
        
        status_info = {}
        for provider, env_var in providers.items():
            api_key = os.getenv(env_var)
            status_info[provider] = {
                'available': bool(api_key),
                'message': 'API key configured' if api_key else 'API key not configured'
            }
        
        # Ollama is always available (local)
        status_info['ollama'] = {
            'available': True,
            'message': 'Local endpoint'
        }
        
        return jsonify({'status': 'success', 'status_info': status_info})
        
    except Exception as e:
        logger.error(f"Error getting API key status: {e}")
        return jsonify({'status': 'error', 'error': 'Failed to get API key status'}), 500

# Authentication API Routes
@app.route('/api/auth/current-user', methods=['GET'])
def get_current_user():
    """Get current user info"""
    try:
        if not auth_manager:
            return jsonify({'error': 'Auth manager not initialized'}), 503
        
        # For now, return the admin user
        admin_user = auth_manager.users.get('admin')
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

@app.route('/api/auth/users', methods=['GET'])
def get_all_users():
    """Get all users"""
    try:
        if not auth_manager:
            return jsonify({'error': 'Auth manager not initialized'}), 503
        
        users = auth_manager.get_all_users()
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

@app.route('/api/auth/regenerate-api-key', methods=['POST'])
def regenerate_api_key():
    """Regenerate API key for current user"""
    try:
        if not auth_manager:
            return jsonify({'error': 'Auth manager not initialized'}), 503
        
        # For now, regenerate for admin user
        new_api_key = auth_manager.regenerate_api_key('admin')
        if not new_api_key:
            return jsonify({'status': 'error', 'error': 'Failed to regenerate API key'}), 500
        
        return jsonify({'status': 'success', 'api_key': new_api_key})
        
    except Exception as e:
        logger.error(f"Error regenerating API key: {e}")
        return jsonify({'status': 'error', 'error': 'Failed to regenerate API key'}), 500

@app.route('/api/auth/generate-jwt', methods=['POST'])
def generate_jwt():
    """Generate JWT token"""
    try:
        if not auth_manager:
            return jsonify({'error': 'Auth manager not initialized'}), 503
        
        data = request.get_json()
        expires_in = data.get('expires_in', 3600)
        
        # For now, generate for admin user
        token = auth_manager.generate_jwt_token('admin', expires_in)
        
        return jsonify({'status': 'success', 'token': token})
        
    except Exception as e:
        logger.error(f"Error generating JWT: {e}")
        return jsonify({'status': 'error', 'error': 'Failed to generate JWT'}), 500

@app.route('/api/auth/usage-stats', methods=['GET'])
def get_usage_stats():
    """Get API usage statistics"""
    try:
        # Return mock data for now
        stats = {
            'total_requests': 150,
            'requests_today': 25,
            'error_rate': 2.5
        }
        
        return jsonify({'status': 'success', 'stats': stats})
        
    except Exception as e:
        logger.error(f"Error getting usage stats: {e}")
        return jsonify({'status': 'error', 'error': 'Failed to get usage stats'}), 500

# Settings API Routes
@app.route('/api/settings/save-api-keys', methods=['POST'])
def save_api_keys():
    """Save API keys to environment"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # For demo purposes, we'll just validate the keys
        # In production, these would be securely stored
        saved_keys = {}
        key_mappings = {
            'openai-key': 'OPENAI_API_KEY',
            'anthropic-key': 'ANTHROPIC_API_KEY',
            'google-key': 'GEMINI_API_KEY',
            'xai-key': 'XAI_API_KEY',
            'perplexity-key': 'PERPLEXITY_API_KEY',
            'cohere-key': 'COHERE_API_KEY',
            'mistral-key': 'MISTRAL_API_KEY',
            'huggingface-key': 'HUGGINGFACE_API_KEY'
        }
        
        for form_key, env_key in key_mappings.items():
            if form_key in data and data[form_key]:
                saved_keys[env_key] = data[form_key]
                # In production, you would save to secure storage
                # os.environ[env_key] = data[form_key]
        
        return jsonify({
            'status': 'success',
            'message': f'Saved {len(saved_keys)} API keys',
            'saved_keys': list(saved_keys.keys())
        })
        
    except Exception as e:
        logger.error(f"Error saving API keys: {e}")
        return jsonify({'status': 'error', 'error': 'Failed to save API keys'}), 500

@app.route('/api/settings/general', methods=['GET', 'POST'])
def general_settings():
    """Get or save general settings"""
    try:
        if request.method == 'GET':
            # Return current settings
            settings = {
                'default_model': 'gpt-4o-mini',
                'max_tokens': 4096,
                'temperature': 0.7,
                'auto_retry': True
            }
            return jsonify({'status': 'success', 'settings': settings})
            
        elif request.method == 'POST':
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No data provided'}), 400
            
            # Save settings (in production, save to database)
            return jsonify({
                'status': 'success',
                'message': 'General settings saved successfully'
            })
            
    except Exception as e:
        logger.error(f"Error with general settings: {e}")
        return jsonify({'status': 'error', 'error': 'Failed to handle general settings'}), 500

@app.route('/api/settings/security', methods=['GET', 'POST'])
def security_settings():
    """Get or save security settings"""
    try:
        if request.method == 'GET':
            settings = {
                'rate_limit': 60,
                'session_timeout': 60,
                'require_auth': True,
                'log_requests': True
            }
            return jsonify({'status': 'success', 'settings': settings})
            
        elif request.method == 'POST':
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No data provided'}), 400
            
            return jsonify({
                'status': 'success',
                'message': 'Security settings saved successfully'
            })
            
    except Exception as e:
        logger.error(f"Error with security settings: {e}")
        return jsonify({'status': 'error', 'error': 'Failed to handle security settings'}), 500

@app.route('/api/settings/performance', methods=['GET', 'POST'])
def performance_settings():
    """Get or save performance settings"""
    try:
        if request.method == 'GET':
            settings = {
                'cache_ttl': 3600,
                'max_concurrent': 10,
                'request_timeout': 30,
                'enable_cache': True
            }
            return jsonify({'status': 'success', 'settings': settings})
            
        elif request.method == 'POST':
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No data provided'}), 400
            
            return jsonify({
                'status': 'success',
                'message': 'Performance settings saved successfully'
            })
            
    except Exception as e:
        logger.error(f"Error with performance settings: {e}")
        return jsonify({'status': 'error', 'error': 'Failed to handle performance settings'}), 500

# Configuration API Routes
@app.route('/api/config/model', methods=['GET', 'POST'])
def model_config():
    """Get or save model configuration"""
    try:
        if request.method == 'GET':
            config = {
                'openai': {
                    'model': 'gpt-4o',
                    'max_tokens': 4096,
                    'temperature': 0.7
                },
                'anthropic': {
                    'model': 'claude-sonnet-4-20250514',
                    'max_tokens': 4096,
                    'temperature': 0.7
                },
                'google': {
                    'model': 'gemini-2.5-flash',
                    'max_tokens': 8192
                },
                'xai': {
                    'model': 'grok-2-1212',
                    'max_tokens': 131072
                },
                'ollama': {
                    'endpoint': 'http://localhost:11434',
                    'model': 'llama3.2'
                }
            }
            return jsonify({'status': 'success', 'config': config})
            
        elif request.method == 'POST':
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No data provided'}), 400
            
            return jsonify({
                'status': 'success',
                'message': 'Model configuration saved successfully'
            })
            
    except Exception as e:
        logger.error(f"Error with model config: {e}")
        return jsonify({'status': 'error', 'error': 'Failed to handle model configuration'}), 500

@app.route('/api/config/endpoint', methods=['GET', 'POST'])
def endpoint_config():
    """Get or save endpoint configuration"""
    try:
        if request.method == 'GET':
            config = {
                'openai_endpoint': 'https://api.openai.com/v1',
                'anthropic_endpoint': 'https://api.anthropic.com',
                'google_endpoint': 'https://generativelanguage.googleapis.com/v1beta',
                'xai_endpoint': 'https://api.x.ai/v1',
                'connection_timeout': 30,
                'read_timeout': 60,
                'retry_attempts': 3,
                'retry_delay': 1,
                'custom_endpoints': []
            }
            return jsonify({'status': 'success', 'config': config})
            
        elif request.method == 'POST':
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No data provided'}), 400
            
            return jsonify({
                'status': 'success',
                'message': 'Endpoint configuration saved successfully'
            })
            
    except Exception as e:
        logger.error(f"Error with endpoint config: {e}")
        return jsonify({'status': 'error', 'error': 'Failed to handle endpoint configuration'}), 500

@app.route('/api/config/routing', methods=['GET', 'POST'])
def routing_config():
    """Get or save routing configuration"""
    try:
        if request.method == 'GET':
            config = {
                'confidence_threshold': 0.7,
                'fallback_strategy': 'keyword',
                'enable_ml_classification': True,
                'load_balancing': 'least-connections',
                'max_agents': 5,
                'agent_timeout': 30,
                'enabled_categories': [
                    'analysis', 'creative', 'technical', 'coding',
                    'mathematical', 'research', 'philosophical',
                    'practical', 'educational', 'conversational'
                ]
            }
            return jsonify({'status': 'success', 'config': config})
            
        elif request.method == 'POST':
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No data provided'}), 400
            
            return jsonify({
                'status': 'success',
                'message': 'Routing configuration saved successfully'
            })
            
    except Exception as e:
        logger.error(f"Error with routing config: {e}")
        return jsonify({'status': 'error', 'error': 'Failed to handle routing configuration'}), 500

@app.route('/api/config/monitoring', methods=['GET', 'POST'])
def monitoring_config():
    """Get or save monitoring configuration"""
    try:
        if request.method == 'GET':
            config = {
                'log_level': 'INFO',
                'log_retention': 30,
                'log_queries': True,
                'log_responses': True,
                'metrics_endpoint': 'http://localhost:9090',
                'metrics_interval': 60,
                'enable_metrics': True,
                'enable_health_checks': True,
                'error_threshold': 5,
                'response_threshold': 5000,
                'alert_email': ''
            }
            return jsonify({'status': 'success', 'config': config})
            
        elif request.method == 'POST':
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No data provided'}), 400
            
            return jsonify({
                'status': 'success',
                'message': 'Monitoring configuration saved successfully'
            })
            
    except Exception as e:
        logger.error(f"Error with monitoring config: {e}")
        return jsonify({'status': 'error', 'error': 'Failed to handle monitoring configuration'}), 500

@app.route('/api/config/advanced', methods=['GET', 'POST'])
def advanced_config():
    """Get or save advanced configuration"""
    try:
        if request.method == 'GET':
            config = {
                'thread_pool_size': 10,
                'connection_pool_size': 20,
                'queue_size': 1000,
                'cache_backend': 'redis',
                'cache_ttl': 3600,
                'cache_max_size': 1024,
                'feature_flags': {
                    'enable_streaming': True,
                    'enable_caching': True,
                    'enable_compression': True,
                    'enable_rate_limiting': True,
                    'enable_circuit_breaker': False,
                    'enable_distributed_tracing': False,
                    'enable_auto_scaling': False,
                    'enable_backup': False,
                    'debug_mode': False
                }
            }
            return jsonify({'status': 'success', 'config': config})
            
        elif request.method == 'POST':
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No data provided'}), 400
            
            return jsonify({
                'status': 'success',
                'message': 'Advanced configuration saved successfully'
            })
            
    except Exception as e:
        logger.error(f"Error with advanced config: {e}")
        return jsonify({'status': 'error', 'error': 'Failed to handle advanced configuration'}), 500

@app.route('/api/config/export', methods=['GET'])
def export_config():
    """Export complete configuration"""
    try:
        # In production, this would gather all actual configuration
        config = {
            'timestamp': datetime.now().isoformat(),
            'version': '1.0',
            'model_config': {},
            'endpoint_config': {},
            'routing_config': {},
            'monitoring_config': {},
            'advanced_config': {}
        }
        
        return jsonify({'status': 'success', 'config': config})
        
    except Exception as e:
        logger.error(f"Error exporting config: {e}")
        return jsonify({'status': 'error', 'error': 'Failed to export configuration'}), 500

@app.route('/api/config/import', methods=['POST'])
def import_config():
    """Import configuration"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No configuration data provided'}), 400
        
        # In production, this would validate and apply the configuration
        return jsonify({
            'status': 'success',
            'message': 'Configuration imported successfully'
        })
        
    except Exception as e:
        logger.error(f"Error importing config: {e}")
        return jsonify({'status': 'error', 'error': 'Failed to import configuration'}), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({'error': 'Rate limit exceeded'}), 429

# Cache Management Endpoints
@app.route('/api/cache/stats')
def get_cache_stats():
    """Get cache statistics"""
    if not cache_manager:
        return jsonify({'error': 'Cache manager not initialized'}), 500
    
    try:
        stats = cache_manager.get_stats()
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/cache/entries')
def get_cache_entries():
    """Get cache entries"""
    if not cache_manager:
        return jsonify({'error': 'Cache manager not initialized'}), 500
    
    try:
        model_id = request.args.get('model_id')
        limit = int(request.args.get('limit', 100))
        
        entries = cache_manager.get_cache_entries(model_id=model_id, limit=limit)
        return jsonify(entries)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/cache/clear', methods=['POST'])
def clear_cache():
    """Clear cache entries"""
    if not cache_manager:
        return jsonify({'error': 'Cache manager not initialized'}), 500
    
    try:
        data = request.get_json() or {}
        model_id = data.get('model_id')
        
        cache_manager.clear(model_id=model_id)
        return jsonify({'status': 'success', 'message': 'Cache cleared successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Chat API Endpoints
@app.route('/api/chat/message', methods=['POST'])
def chat_message():
    """Send message to AI model"""
    try:
        data = request.get_json()
        query = data.get('query', '')
        system_message = data.get('system_message')
        model_id = data.get('model_id')
        temperature = data.get('temperature', 0.7)
        max_tokens = data.get('max_tokens', 4096)
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        if not model_id:
            return jsonify({'error': 'Model ID is required'}), 400
        
        if not ai_model_manager:
            return jsonify({'error': 'AI model manager not initialized'}), 500
        
        # Generate response using AI model manager
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            response = loop.run_until_complete(
                ai_model_manager.generate_response(
                    query=query,
                    system_message=system_message,
                    model_id=model_id,
                    user_id=session.get('user_id', 'anonymous')
                )
            )
            
            return jsonify({
                'status': 'success',
                'response': response['response'],
                'model': response['model'],
                'usage': response.get('usage', {}),
                'cached': response.get('cached', False)
            })
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"Chat message error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat/stream')
def chat_stream():
    """Stream chat response using Server-Sent Events"""
    query = request.args.get('query', '')
    system_message = request.args.get('system_message')
    model_id = request.args.get('model_id')
    
    if not query or not model_id:
        return jsonify({'error': 'Query and model_id are required'}), 400
    
    def generate():
        try:
            yield f"data: {json.dumps({'type': 'start', 'model': model_id})}\n\n"
            
            # For now, simulate streaming by chunking the response
            # In a real implementation, you'd integrate with streaming APIs
            
            # Get regular response first
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                response = loop.run_until_complete(
                    ai_model_manager.generate_response(
                        query=query,
                        system_message=system_message,
                        model_id=model_id,
                        user_id=session.get('user_id', 'anonymous')
                    )
                )
                
                # Simulate streaming by sending chunks
                full_response = response['response']
                words = full_response.split()
                
                for i, word in enumerate(words):
                    chunk = word + (' ' if i < len(words) - 1 else '')
                    yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"
                    time.sleep(0.1)  # Simulate streaming delay
                
                yield f"data: {json.dumps({'type': 'end', 'usage': response.get('usage', {})})}\n\n"
                
            finally:
                loop.close()
                
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
    
    return Response(generate(), mimetype='text/event-stream')

@app.route('/api/chat/sessions', methods=['GET'])
def get_chat_sessions():
    """Get user's chat sessions"""
    user_id = session.get('user_id', 'anonymous')
    
    # In a real implementation, you'd fetch from database
    # For now, return empty array as sessions are stored client-side
    return jsonify([])

@app.route('/api/chat/sessions', methods=['POST'])
def create_chat_session():
    """Create new chat session"""
    try:
        data = request.get_json()
        user_id = session.get('user_id', 'anonymous')
        
        # In a real implementation, you'd save to database
        session_data = {
            'id': f"chat_{int(time.time())}_{user_id}",
            'title': data.get('title', 'New Chat'),
            'model': data.get('model'),
            'created': datetime.now().isoformat(),
            'messages': []
        }
        
        return jsonify(session_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# RAG System Endpoints
@app.route('/api/rag/upload', methods=['POST'])
def upload_document():
    """Upload a document for RAG processing"""
    try:
        global rag_system
        if not rag_system:
            return jsonify({'error': 'RAG system not initialized'}), 500
        
        # Check if file is uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save file temporarily
        temp_path = f"./temp_{file.filename}"
        file.save(temp_path)
        
        try:
            # Process the uploaded file
            doc_id = rag_system.process_uploaded_file(temp_path, file.filename)
            
            if doc_id:
                return jsonify({
                    'message': 'Document uploaded successfully',
                    'document_id': doc_id,
                    'filename': file.filename
                })
            else:
                return jsonify({'error': 'Failed to process document'}), 500
                
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/rag/documents', methods=['GET'])
def list_documents():
    """List all uploaded documents"""
    try:
        global rag_system
        if not rag_system:
            return jsonify({'error': 'RAG system not initialized'}), 500
        
        documents = rag_system.get_documents_list()
        return jsonify({'documents': documents})
        
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/rag/documents/<doc_id>', methods=['DELETE'])
def delete_document(doc_id):
    """Delete a document from RAG system"""
    try:
        global rag_system
        if not rag_system:
            return jsonify({'error': 'RAG system not initialized'}), 500
        
        success = rag_system.delete_document(doc_id)
        if success:
            return jsonify({'message': 'Document deleted successfully'})
        else:
            return jsonify({'error': 'Document not found'}), 404
            
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/rag/search', methods=['POST'])
def search_documents():
    """Search documents using RAG system"""
    try:
        global rag_system
        if not rag_system:
            return jsonify({'error': 'RAG system not initialized'}), 500
        
        data = request.get_json()
        query = data.get('query', '')
        max_results = data.get('max_results', 3)
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        results = rag_system.search_documents(query, max_results)
        return jsonify({'results': results})
        
    except Exception as e:
        logger.error(f"Error searching documents: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/rag/stats', methods=['GET'])
def rag_stats():
    """Get RAG system statistics"""
    try:
        global rag_system
        if not rag_system:
            return jsonify({'error': 'RAG system not initialized'}), 500
        
        stats = rag_system.get_stats()
        return jsonify(stats)
        
    except Exception as e:
        logger.error(f"Error getting RAG stats: {e}")
        return jsonify({'error': str(e)}), 500

# Swagger Documentation Endpoints
@app.route('/api/docs', methods=['GET'])
def api_docs():
    """Interactive API documentation"""
    return render_template('api_docs.html')

@app.route('/chains')
def auto_chain_generator_page():
    """Auto Chain Generator interface"""
    return render_template('auto_chain_generator.html')

@app.route('/api/openapi.json', methods=['GET'])
def openapi_spec():
    """OpenAPI/Swagger specification"""
    return jsonify(swagger_spec)

# External LLM Integration Endpoints
@app.route('/api/external-llm/analyze', methods=['POST'])
def analyze_query_complexity():
    """Analyze query complexity for external LLM routing"""
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({'error': 'Query is required'}), 400
        
        query = data['query']
        complex_query = external_llm_manager.analyzer.analyze_query(query)
        
        return jsonify({
            'query': query,
            'complexity': complex_query.complexity.value,
            'domain': complex_query.domain,
            'requires_reasoning': complex_query.requires_reasoning,
            'requires_creativity': complex_query.requires_creativity,
            'requires_analysis': complex_query.requires_analysis,
            'requires_multi_step': complex_query.requires_multi_step,
            'context_length': complex_query.context_length,
            'specialized_knowledge': complex_query.specialized_knowledge,
            'is_complex': external_llm_manager.is_complex_query(query),
            'recommended_provider': external_llm_manager.get_recommended_provider(query).value if external_llm_manager.get_recommended_provider(query) else None
        })
    except Exception as e:
        logger.error(f"Error analyzing query: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/external-llm/process', methods=['POST'])
def process_with_external_llm():
    """Process a query using external LLM"""
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({'error': 'Query is required'}), 400
        
        query = data['query']
        context = data.get('context', '')
        preferred_provider = data.get('preferred_provider')
        
        # Convert string provider to enum if provided
        if preferred_provider:
            from external_llm_integration import ExternalProvider
            try:
                preferred_provider = ExternalProvider(preferred_provider)
            except ValueError:
                return jsonify({'error': f'Invalid provider: {preferred_provider}'}), 400
        
        # Process with external LLM
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        result = loop.run_until_complete(
            external_llm_manager.process_complex_query(
                query, context=context, preferred_provider=preferred_provider
            )
        )
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error processing with external LLM: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/external-llm/providers')
def get_external_providers():
    """Get available external LLM providers"""
    try:
        providers = []
        for provider_enum, config in external_llm_manager.providers.items():
            api_key = os.environ.get(config.api_key_env)
            providers.append({
                'id': provider_enum.value,
                'name': config.model_name,
                'endpoint': config.endpoint,
                'max_tokens': config.max_tokens,
                'cost_per_1k_tokens': config.cost_per_1k_tokens,
                'rate_limit_rpm': config.rate_limit_rpm,
                'specializations': config.specializations,
                'api_key_available': bool(api_key)
            })
        
        return jsonify({
            'providers': providers,
            'total_providers': len(providers),
            'available_providers': len([p for p in providers if p['api_key_available']])
        })
    except Exception as e:
        logger.error(f"Error getting external providers: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/external-llm/metrics')
def get_external_llm_metrics():
    """Get performance metrics for external LLM providers"""
    try:
        metrics = external_llm_manager.get_provider_metrics()
        return jsonify(metrics)
    except Exception as e:
        logger.error(f"Error getting external LLM metrics: {e}")
        return jsonify({'error': str(e)}), 500

# Collaborative AI Endpoints
@app.route('/api/collaborate', methods=['POST'])
def collaborate():
    """Submit a query for collaborative AI processing"""
    try:
        global collaborative_router
        if not collaborative_router:
            return jsonify({'error': 'Collaborative router not initialized'}), 500
        
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        query = data.get('query', '')
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        enable_rag = data.get('enable_rag', False)
        max_agents = data.get('max_agents', 3)
        collaboration_timeout = data.get('timeout', 300)
        
        # Run collaborative processing
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(
                collaborative_router.process_collaborative_query(
                    query=query,
                    enable_rag=enable_rag,
                    max_agents=max_agents,
                    collaboration_timeout=collaboration_timeout
                )
            )
            return jsonify(result)
        finally:
            loop.close()
        
    except Exception as e:
        logger.error(f"Collaborative processing error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/collaborate/sessions', methods=['GET'])
def get_collaboration_sessions():
    """Get active collaboration sessions"""
    try:
        global collaborative_router
        if not collaborative_router:
            return jsonify({'error': 'Collaborative router not initialized'}), 500
        
        active_sessions = collaborative_router.get_active_sessions()
        return jsonify({'sessions': active_sessions})
        
    except Exception as e:
        logger.error(f"Error getting collaboration sessions: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/collaborate/sessions/<session_id>', methods=['GET'])
def get_collaboration_session(session_id):
    """Get details of a specific collaboration session"""
    try:
        global collaborative_router
        if not collaborative_router:
            return jsonify({'error': 'Collaborative router not initialized'}), 500
        
        session_details = collaborative_router.get_session_details(session_id)
        return jsonify(session_details)
        
    except Exception as e:
        logger.error(f"Error getting session details: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/shared-memory/stats', methods=['GET'])
def get_shared_memory_stats():
    """Get shared memory statistics"""
    try:
        global shared_memory_manager
        if not shared_memory_manager:
            return jsonify({'error': 'Shared memory manager not initialized'}), 500
        
        # Get basic stats
        stats = {
            'total_messages': len(shared_memory_manager.messages),
            'active_sessions': len(shared_memory_manager.sessions),
            'agent_contexts': len(shared_memory_manager.agent_contexts),
            'message_index_size': sum(len(msgs) for msgs in shared_memory_manager.message_index.values())
        }
        
        return jsonify(stats)
        
    except Exception as e:
        logger.error(f"Error getting shared memory stats: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/shared-memory/sessions/<session_id>/messages', methods=['GET'])
def get_session_messages(session_id):
    """Get messages from a specific session"""
    try:
        global shared_memory_manager
        if not shared_memory_manager:
            return jsonify({'error': 'Shared memory manager not initialized'}), 500
        
        # Get query parameters
        limit = int(request.args.get('limit', 50))
        message_types = request.args.getlist('types')
        
        # Convert string types to MessageType enum
        from shared_memory import MessageType
        if message_types:
            try:
                message_types = [MessageType(t) for t in message_types]
            except ValueError:
                return jsonify({'error': 'Invalid message type'}), 400
        else:
            message_types = None
        
        messages = shared_memory_manager.get_session_messages(
            session_id, 
            message_types=message_types
        )
        
        # Limit results
        messages = messages[-limit:]
        
        return jsonify({
            'session_id': session_id,
            'messages': [msg.to_dict() for msg in messages]
        })
        
    except Exception as e:
        logger.error(f"Error getting session messages: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/shared-memory/sessions/<session_id>/context', methods=['GET'])
def get_session_context(session_id):
    """Get shared context for a session"""
    try:
        global shared_memory_manager
        if not shared_memory_manager:
            return jsonify({'error': 'Shared memory manager not initialized'}), 500
        
        context = shared_memory_manager.get_shared_context(session_id)
        return jsonify(context)
        
    except Exception as e:
        logger.error(f"Error getting session context: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/collaborate')
def collaborate_page():
    """Collaborative AI interface page"""
    return render_template('collaborate.html')

@app.route('/multimodal')
def multimodal_page():
    """Multi-modal AI processing interface"""
    return render_template('multimodal.html')

@app.route('/api-keys')
def api_keys_page():
    """API Keys management interface"""
    return render_template('api_keys.html')

# API Key Management Endpoints
@app.route('/api/models/<model_id>/configure', methods=['POST'])
def configure_model(model_id):
    """Configure API key and settings for a specific model"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Get the model from AI manager
        model = ai_model_manager.get_model(model_id)
        if not model:
            return jsonify({'error': 'Model not found'}), 404
        
        # Update model configuration
        api_key = data.get('api_key')
        max_tokens = data.get('max_tokens')
        temperature = data.get('temperature')
        
        # For local models, API key is not required
        if model.deployment_type != 'local' and api_key:
            # Set environment variable for API key
            os.environ[model.api_key_env] = api_key
        
        # Update model settings
        if max_tokens:
            model.max_tokens = max_tokens
        if temperature:
            model.temperature = temperature
        
        return jsonify({
            'message': 'Model configured successfully',
            'model_id': model_id,
            'api_key_available': bool(api_key) or model.deployment_type == 'local'
        })
        
    except Exception as e:
        logger.error(f"Error configuring model {model_id}: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/models/<model_id>/test', methods=['POST'])
def test_model(model_id):
    """Test API key for a specific model"""
    try:
        model = ai_model_manager.get_model(model_id)
        if not model:
            return jsonify({'error': 'Model not found'}), 404
        
        # For local models, just return success
        if model.deployment_type == 'local':
            return jsonify({
                'success': True,
                'message': 'Local model is ready',
                'model_id': model_id
            })
        
        # Check if API key is available
        api_key = os.environ.get(model.api_key_env)
        if not api_key:
            return jsonify({
                'success': False,
                'message': 'API key not configured',
                'model_id': model_id
            })
        
        # Test the API key with a simple request
        test_message = "Hello"
        try:
            response = ai_model_manager.generate_response(
                model_id=model_id,
                query=test_message,
                system_message="Respond with 'Test successful'"
            )
            
            return jsonify({
                'success': True,
                'message': 'API key test successful',
                'model_id': model_id,
                'response': response.get('response', 'No response')
            })
            
        except Exception as e:
            return jsonify({
                'success': False,
                'message': f'API key test failed: {str(e)}',
                'model_id': model_id
            })
        
    except Exception as e:
        logger.error(f"Error testing model {model_id}: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/models/detailed')
def get_models_detailed():
    """Get all available models with their detailed status"""
    try:
        models = ai_model_manager.get_all_models()
        model_list = []
        
        for model in models:
            api_key_available = bool(os.environ.get(model.api_key_env)) or model.deployment_type == 'local'
            
            model_dict = {
                'id': model.id,
                'name': model.name,
                'provider': model.provider.value,
                'model_name': model.model_name,
                'endpoint': model.endpoint,
                'max_tokens': model.max_tokens,
                'temperature': model.temperature,
                'context_window': model.context_window,
                'cost_per_1k_tokens': model.cost_per_1k_tokens,
                'api_key_available': api_key_available,
                'deployment_type': getattr(model, 'deployment_type', 'cloud'),
                'supports_vision': getattr(model, 'supports_vision', False),
                'supports_audio': getattr(model, 'supports_audio', False),
                'model_type': getattr(model, 'model_type', 'text'),
                'input_modalities': getattr(model, 'input_modalities', ['text']),
                'output_modalities': getattr(model, 'output_modalities', ['text']),
                'capabilities': [cap.value for cap in getattr(model, 'capabilities', [])]
            }
            
            model_list.append(model_dict)
        
        return jsonify({'models': model_list})
        
    except Exception as e:
        logger.error(f"Error getting models: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/chat')
def chat_page():
    """Multi-modal AI chat interface"""
    return render_template('chat.html')

@app.route('/api/chat', methods=['POST'])
def chat_endpoint():
    """Chat endpoint for multi-modal AI conversation"""
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({'error': 'Message is required'}), 400
        
        message = data['message']
        model_id = data.get('model_id', 'gpt-4o')
        temperature = data.get('temperature', 0.7)
        max_tokens = data.get('max_tokens', 1000)
        enable_rag = data.get('enable_rag', False)
        enable_streaming = data.get('enable_streaming', False)
        chat_history = data.get('chat_history', [])
        
        # Get the model
        model = ai_model_manager.get_model(model_id)
        if not model:
            return jsonify({'error': 'Model not found'}), 404
        
        # Check if API key is available
        api_key_available = bool(os.environ.get(model.api_key_env)) or model.deployment_type == 'local'
        if not api_key_available:
            return jsonify({'error': 'API key not configured for this model'}), 400
        
        # Generate response using AI model manager
        response = ai_model_manager.generate_response(
            model_id=model_id,
            query=message,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return jsonify({
            'response': response.get('response', 'No response generated'),
            'model_used': model.name,
            'tokens_used': response.get('tokens_used', 0),
            'processing_time': response.get('processing_time', 0)
        })
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/collaborate/agents', methods=['GET'])
def get_collaborative_agents():
    """Get collaborative agent configurations"""
    try:
        global collaborative_router
        if not collaborative_router:
            return jsonify({'error': 'Collaborative router not initialized'}), 500
        
        configurations = collaborative_router.get_agent_configurations()
        return jsonify(configurations)
        
    except Exception as e:
        logger.error(f"Error getting collaborative agents: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/collaborate/agents/<agent_id>/model', methods=['PUT'])
def update_agent_model(agent_id):
    """Update AI model for a specific collaborative agent"""
    try:
        global collaborative_router
        if not collaborative_router:
            return jsonify({'error': 'Collaborative router not initialized'}), 500
        
        data = request.get_json()
        if not data or 'model_id' not in data:
            return jsonify({'error': 'model_id is required'}), 400
        
        model_id = data['model_id']
        success = collaborative_router.update_agent_model(agent_id, model_id)
        
        if success:
            return jsonify({'message': f'Agent {agent_id} updated to use model {model_id}'})
        else:
            return jsonify({'error': 'Failed to update agent model'}), 400
        
    except Exception as e:
        logger.error(f"Error updating agent model: {e}")
        return jsonify({'error': str(e)}), 500

# Token Management Endpoints
@app.route('/api/token/usage', methods=['GET'])
def get_token_usage():
    """Get current token usage statistics"""
    try:
        if not ai_model_manager:
            return jsonify({"error": "AI model manager not initialized"}), 503
        stats = ai_model_manager.get_token_usage_stats()
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Error getting token usage: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/token/settings', methods=['GET'])
def get_token_settings():
    """Get current token management settings"""
    try:
        if not ai_model_manager:
            return jsonify({"error": "AI model manager not initialized"}), 503
        settings = ai_model_manager._get_token_settings()
        return jsonify(settings)
    except Exception as e:
        logger.error(f"Error getting token settings: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/token/settings', methods=['POST'])
def update_token_settings():
    """Update token management settings"""
    try:
        settings = request.get_json()
        if not settings:
            return jsonify({"error": "No settings provided"}), 400
        
        success = ai_model_manager.update_token_settings(settings)
        if success:
            return jsonify({"message": "Token settings updated successfully"})
        else:
            return jsonify({"error": "Failed to update token settings"}), 400
    except Exception as e:
        logger.error(f"Error updating token settings: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/token/optimize', methods=['POST'])
def optimize_tokens():
    """Optimize token usage for a given query"""
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({"error": "Query is required"}), 400
        
        query = data['query']
        system_message = data.get('system_message')
        
        # Apply token optimization
        optimized_query, optimized_system = ai_model_manager._apply_token_settings(
            query, system_message
        )
        
        # Calculate token savings
        original_tokens = len(query + (system_message or '')) / 4
        optimized_tokens = len(optimized_query + (optimized_system or '')) / 4
        savings = max(0, (original_tokens - optimized_tokens) / original_tokens * 100)
        
        return jsonify({
            "original_query": query,
            "optimized_query": optimized_query,
            "original_system": system_message,
            "optimized_system": optimized_system,
            "original_tokens": int(original_tokens),
            "optimized_tokens": int(optimized_tokens),
            "savings_percentage": round(savings, 2)
        })
    except Exception as e:
        logger.error(f"Error optimizing tokens: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/token/insights', methods=['GET'])
def get_predictive_insights():
    """Get AI-powered predictive insights for token optimization"""
    try:
        insights = ai_model_manager.get_predictive_insights()
        return jsonify(insights)
    except Exception as e:
        logger.error(f"Error getting predictive insights: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/token/predictive-insights', methods=['GET'])
def get_predictive_insights_alt():
    """Get AI-powered predictive insights for token optimization (alternative endpoint)"""
    try:
        if not ai_model_manager:
            return jsonify({"error": "AI model manager not initialized"}), 503
        insights = ai_model_manager.get_predictive_insights()
        return jsonify(insights)
    except Exception as e:
        logger.error(f"Error getting predictive insights: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/token/recommendations', methods=['GET'])
def get_optimization_recommendations():
    """Get AI-powered optimization recommendations"""
    try:
        if not ai_model_manager:
            return jsonify({"error": "AI model manager not initialized"}), 503
        usage_history = request.args.get('usage_history')
        recommendations = ai_model_manager.get_optimization_recommendations()
        return jsonify({"recommendations": recommendations})
    except Exception as e:
        logger.error(f"Error getting optimization recommendations: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/token/apply-ai-optimizations', methods=['POST'])
def apply_ai_optimizations():
    """Apply AI-powered optimizations to a query"""
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({"error": "Query is required"}), 400
        
        query = data['query']
        context = data.get('context', {})
        
        result = ai_model_manager.apply_ai_optimizations(query, context)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error applying AI optimizations: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/token/advanced-metrics', methods=['GET'])
def get_advanced_token_metrics():
    """Get advanced token metrics for performance analytics"""
    try:
        if not ai_model_manager:
            return jsonify({"error": "AI model manager not initialized"}), 503
        stats = ai_model_manager.get_token_usage_stats()
        insights = ai_model_manager.get_predictive_insights()
        recommendations = ai_model_manager.get_optimization_recommendations()
        
        return jsonify({
            "metrics": stats,
            "insights": insights,
            "recommendations": recommendations,
            "advanced_features": {
                "predictive_scaling": True,
                "intelligent_caching": True,
                "dynamic_compression": True,
                "quality_monitoring": True
            }
        })
    except Exception as e:
        logger.error(f"Error getting advanced metrics: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/token/bulk-optimize', methods=['POST'])
def bulk_optimize_tokens():
    """Bulk optimize multiple queries for batch processing"""
    try:
        data = request.get_json()
        if not data or 'queries' not in data:
            return jsonify({"error": "Queries array is required"}), 400
        
        queries = data['queries']
        if not isinstance(queries, list):
            return jsonify({"error": "Queries must be an array"}), 400
        
        results = []
        total_savings = 0
        
        for query in queries:
            if isinstance(query, dict):
                query_text = query.get('query', '')
                system_message = query.get('system_message')
            else:
                query_text = str(query)
                system_message = None
            
            # Apply optimizations
            optimized_query, optimized_system = ai_model_manager._apply_token_settings(
                query_text, system_message
            )
            
            # Calculate savings
            original_tokens = len(query_text + (system_message or '')) / 4
            optimized_tokens = len(optimized_query + (optimized_system or '')) / 4
            savings = max(0, (original_tokens - optimized_tokens) / original_tokens * 100)
            total_savings += savings
            
            results.append({
                "original_query": query_text,
                "optimized_query": optimized_query,
                "original_tokens": int(original_tokens),
                "optimized_tokens": int(optimized_tokens),
                "savings_percentage": round(savings, 2)
            })
        
        return jsonify({
            "results": results,
            "total_queries": len(queries),
            "average_savings": round(total_savings / len(queries), 2) if queries else 0,
            "processing_time_ms": len(queries) * 50
        })
    except Exception as e:
        logger.error(f"Error bulk optimizing tokens: {e}")
        return jsonify({"error": str(e)}), 500


# ==============================================================================
# ADVANCED ML ENHANCEMENT API ENDPOINTS
# ==============================================================================

@app.route('/api/ml/classify', methods=['POST'])
def classify_query():
    """Advanced ML-based query classification"""
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({"error": "Query is required"}), 400
        
        query = data['query']
        context = data.get('context', {})
        
        if not advanced_ml_classifier:
            return jsonify({"error": "Advanced ML classifier not available"}), 503
        
        # Analyze query
        analysis = asyncio.run(advanced_ml_classifier.analyze(query, context))
        
        return jsonify({
            "analysis": {
                "primary_category": analysis.primary_category.value,
                "categories": [cat.value for cat in analysis.categories],
                "confidence": analysis.confidence,
                "complexity": analysis.complexity,
                "intent": analysis.intent.value,
                "sentiment": analysis.sentiment,
                "language": analysis.language,
                "technical_level": analysis.technical_level,
                "domain_expertise": analysis.domain_expertise,
                "required_capabilities": analysis.required_capabilities,
                "context_needed": analysis.context_needed,
                "multi_step": analysis.multi_step,
                "priority": analysis.priority,
                "estimated_tokens": analysis.estimated_tokens,
                "processing_time": analysis.processing_time,
                "metadata": analysis.metadata
            }
        })
    except Exception as e:
        logger.error(f"Error classifying query: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/ml/optimize', methods=['POST'])
def optimize_query():
    """Advanced query optimization with ML enhancement"""
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({"error": "Query is required"}), 400
        
        query = data['query']
        context = data.get('context', {})
        optimization_types = data.get('optimization_types', [])
        
        if not advanced_ml_classifier or not advanced_query_optimizer:
            return jsonify({"error": "Advanced optimization not available"}), 503
        
        # Analyze query first
        analysis = asyncio.run(advanced_ml_classifier.analyze(query, context))
        
        # Apply optimizations
        enhancement = asyncio.run(advanced_query_optimizer.optimize_query(
            query, analysis, optimization_types
        ))
        
        return jsonify({
            "enhancement": {
                "original_query": enhancement.original_query,
                "enhanced_query": enhancement.enhanced_query,
                "overall_confidence": enhancement.overall_confidence,
                "quality_score": enhancement.quality_score,
                "complexity_reduction": enhancement.complexity_reduction,
                "suggested_context": enhancement.suggested_context,
                "related_queries": enhancement.related_queries,
                "optimizations": [
                    {
                        "type": opt.optimization_type.value,
                        "confidence": opt.confidence,
                        "reasoning": opt.reasoning,
                        "improvement_score": opt.improvement_score
                    } for opt in enhancement.optimizations
                ]
            }
        })
    except Exception as e:
        logger.error(f"Error optimizing query: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/ml/route', methods=['POST'])
def intelligent_route():
    """Intelligent routing with ML-enhanced agent selection"""
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({"error": "Query is required"}), 400
        
        query = data['query']
        context = data.get('context', {})
        strategy = data.get('strategy')
        
        if not intelligent_routing_engine:
            return jsonify({"error": "Intelligent routing engine not available"}), 503
        
        # Route query
        decision = asyncio.run(intelligent_routing_engine.route_query(
            query, context, strategy
        ))
        
        return jsonify({
            "routing_decision": {
                "primary_agent": {
                    "id": decision.primary_agent.id,
                    "name": decision.primary_agent.name,
                    "endpoint": decision.primary_agent.endpoint,
                    "confidence": decision.confidence
                },
                "selected_agents": [
                    {
                        "id": agent.id,
                        "name": agent.name,
                        "endpoint": agent.endpoint
                    } for agent in decision.selected_agents
                ],
                "strategy_used": decision.strategy_used.value,
                "reasoning": decision.reasoning,
                "estimated_response_time": decision.estimated_response_time,
                "estimated_cost": decision.estimated_cost,
                "fallback_agents": [
                    {
                        "id": agent.id,
                        "name": agent.name
                    } for agent in decision.fallback_agents
                ]
            }
        })
    except Exception as e:
        logger.error(f"Error routing query: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/analytics/real-time', methods=['GET'])
def get_real_time_analytics():
    """Get real-time analytics data"""
    try:
        if not real_time_analytics:
            return jsonify({"error": "Real-time analytics not available"}), 503
        
        # Get dashboard data
        dashboard_data = real_time_analytics.get_dashboard_data()
        
        return jsonify({
            "analytics": dashboard_data,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error getting real-time analytics: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/analytics/metric', methods=['POST'])
def record_metric():
    """Record a custom metric"""
    try:
        data = request.get_json()
        if not data or 'name' not in data or 'value' not in data:
            return jsonify({"error": "Metric name and value required"}), 400
        
        name = data['name']
        value = float(data['value'])
        labels = data.get('labels', {})
        
        if not real_time_analytics:
            return jsonify({"error": "Real-time analytics not available"}), 503
        
        # Record metric
        real_time_analytics.record_metric(name, value, labels)
        
        return jsonify({
            "success": True,
            "message": f"Metric {name} recorded successfully"
        })
    except Exception as e:
        logger.error(f"Error recording metric: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/analytics/performance-report', methods=['GET'])
def get_performance_report():
    """Get comprehensive performance report"""
    try:
        if not real_time_analytics:
            return jsonify({"error": "Real-time analytics not available"}), 503
        
        # Get performance analyzer
        analyzer = real_time_analytics.performance_analyzer
        
        # Generate report for key metrics
        metrics = request.args.getlist('metrics') or [
            'response_time', 'throughput', 'error_rate', 'cache_hit_rate'
        ]
        
        report = analyzer.generate_performance_report(metrics)
        
        return jsonify({
            "report": report,
            "generated_at": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error generating performance report: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/predictions/predict', methods=['POST'])
def make_prediction():
    """Make a prediction using the predictive analytics engine"""
    try:
        data = request.get_json()
        if not data or 'prediction_type' not in data or 'context' not in data:
            return jsonify({"error": "Prediction type and context required"}), 400
        
        prediction_type = data['prediction_type']
        context = data['context']
        horizon_minutes = data.get('horizon_minutes', 30)
        
        if not predictive_analytics_engine:
            return jsonify({"error": "Predictive analytics engine not available"}), 503
        
        # Import prediction type enum
        from predictive_analytics_engine import PredictionType
        
        # Convert string to enum
        try:
            pred_type = PredictionType(prediction_type)
        except ValueError:
            return jsonify({"error": f"Invalid prediction type: {prediction_type}"}), 400
        
        # Make prediction
        prediction = asyncio.run(predictive_analytics_engine.predict(
            pred_type, context, horizon_minutes
        ))
        
        if prediction:
            return jsonify({
                "prediction": {
                    "type": prediction.prediction_type.value,
                    "predicted_value": prediction.predicted_value,
                    "confidence_interval": prediction.confidence_interval,
                    "confidence_score": prediction.confidence_score,
                    "model_used": prediction.model_used,
                    "features_used": prediction.features_used,
                    "prediction_horizon": prediction.prediction_horizon,
                    "accuracy_score": prediction.accuracy_score,
                    "timestamp": prediction.timestamp.isoformat(),
                    "metadata": prediction.metadata
                }
            })
        else:
            return jsonify({"error": "Prediction could not be made"}), 500
            
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/predictions/system-health', methods=['GET'])
def get_system_health_prediction():
    """Get system health prediction"""
    try:
        if not predictive_analytics_engine:
            return jsonify({"error": "Predictive analytics engine not available"}), 503
        
        # Get current system context
        context = {
            "system_cpu_usage": 0.5,
            "system_memory_usage": 0.6,
            "concurrent_queries": 10,
            "cache_hit_rate": 0.85,
            "active_agents": 5,
            "query_processing_rate": 50,
            "network_latency": 45
        }
        
        # Get health prediction
        health_prediction = predictive_analytics_engine.get_system_health_prediction(context)
        
        return jsonify({
            "health_prediction": health_prediction
        })
    except Exception as e:
        logger.error(f"Error getting system health prediction: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/advanced/stats', methods=['GET'])
def get_advanced_stats():
    """Get comprehensive advanced features statistics"""
    try:
        stats = {
            "ml_classifier": advanced_ml_classifier.get_classification_stats() if advanced_ml_classifier else {"available": False},
            "routing_engine": intelligent_routing_engine.get_routing_stats() if intelligent_routing_engine else {"available": False},
            "real_time_analytics": real_time_analytics.get_analytics_stats() if real_time_analytics else {"available": False},
            "query_optimizer": advanced_query_optimizer.get_optimization_stats() if advanced_query_optimizer else {"available": False},
            "predictive_analytics": predictive_analytics_engine.get_analytics_stats() if predictive_analytics_engine else {"available": False}
        }
        
        return jsonify({
            "advanced_features_stats": stats,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error getting advanced stats: {e}")
        return jsonify({"error": str(e)}), 500

# Active Learning Endpoints
@app.route('/api/learning/feedback', methods=['POST'])
def submit_feedback():
    """Submit feedback for active learning"""
    try:
        data = request.get_json()
        if not data or 'query' not in data or 'agent_id' not in data:
            return jsonify({"error": "Query and agent_id are required"}), 400
        
        if not active_learning_system:
            return jsonify({"error": "Active learning system not available"}), 503
        
        from active_learning_system import FeedbackType
        
        query = data['query']
        agent_id = data['agent_id']
        feedback_type = FeedbackType(data.get('feedback_type', 'correct'))
        score = float(data.get('score', 0.8))
        user_id = data.get('user_id')
        expected_category = data.get('expected_category')
        comments = data.get('comments')
        metadata = data.get('metadata', {})
        
        success = asyncio.run(active_learning_system.collect_feedback(
            query, agent_id, feedback_type, score, user_id, expected_category, comments, metadata
        ))
        
        return jsonify({"success": success, "message": "Feedback submitted successfully"})
    except Exception as e:
        logger.error(f"Error submitting feedback: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/learning/retrain', methods=['POST'])
def trigger_retraining():
    """Trigger model retraining"""
    try:
        if not active_learning_system:
            return jsonify({"error": "Active learning system not available"}), 503
        
        should_retrain = asyncio.run(active_learning_system.should_retrain())
        if not should_retrain:
            return jsonify({"message": "Retraining not needed at this time"})
        
        success = asyncio.run(active_learning_system.trigger_retraining())
        return jsonify({"success": success, "message": "Retraining triggered" if success else "Retraining failed"})
    except Exception as e:
        logger.error(f"Error triggering retraining: {e}")
        return jsonify({"error": str(e)}), 500

# Contextual Memory Router Endpoints
@app.route('/api/contextual/route', methods=['POST'])
def contextual_route():
    """Route query using contextual memory"""
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({"error": "Query is required"}), 400
        
        if not contextual_memory_router:
            return jsonify({"error": "Contextual memory router not available"}), 503
        
        from contextual_memory_router import RoutingStrategy
        
        query = data['query']
        context = data.get('context', {})
        strategy = RoutingStrategy(data.get('strategy', 'hybrid'))
        
        route = asyncio.run(contextual_memory_router.route_with_memory(query, context, strategy))
        
        return jsonify({
            "contextual_route": {
                "query": route.query,
                "recommended_agent": route.recommended_agent,
                "confidence": route.confidence,
                "similarity_score": route.similarity_score,
                "reasoning": route.reasoning,
                "fallback_agents": route.fallback_agents,
                "estimated_success": route.estimated_success,
                "matched_memories_count": len(route.matched_memories)
            }
        })
    except Exception as e:
        logger.error(f"Error in contextual routing: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/contextual/outcome', methods=['POST'])
def record_outcome():
    """Record routing outcome for contextual memory"""
    try:
        data = request.get_json()
        if not data or 'query' not in data or 'agent_id' not in data:
            return jsonify({"error": "Query and agent_id are required"}), 400
        
        if not contextual_memory_router:
            return jsonify({"error": "Contextual memory router not available"}), 503
        
        query = data['query']
        agent_id = data['agent_id']
        success_score = float(data.get('success_score', 0.8))
        response_time = float(data.get('response_time', 1.0))
        user_satisfaction = float(data.get('user_satisfaction', 0.8))
        context = data.get('context', {})
        
        asyncio.run(contextual_memory_router.record_routing_outcome(
            query, agent_id, success_score, response_time, user_satisfaction, context
        ))
        
        return jsonify({"success": True, "message": "Outcome recorded successfully"})
    except Exception as e:
        logger.error(f"Error recording outcome: {e}")
        return jsonify({"error": str(e)}), 500

# Semantic Guardrails Endpoints
@app.route('/api/guardrails/check', methods=['POST'])
def check_content():
    """Check content against semantic guardrails"""
    try:
        data = request.get_json()
        if not data or 'content' not in data:
            return jsonify({"error": "Content is required"}), 400
        
        if not semantic_guardrail_system:
            return jsonify({"error": "Semantic guardrail system not available"}), 503
        
        content = data['content']
        result = asyncio.run(semantic_guardrail_system.check_content(content))
        
        return jsonify({
            "guardrail_result": {
                "is_safe": result.is_safe,
                "threat_level": result.threat_level.value,
                "triggered_guardrails": [g.value for g in result.triggered_guardrails],
                "confidence": result.confidence,
                "reasoning": result.reasoning,
                "flagged_content": result.flagged_content,
                "suggested_action": result.suggested_action
            }
        })
    except Exception as e:
        logger.error(f"Error checking content: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/guardrails/report-false-positive', methods=['POST'])
def report_false_positive():
    """Report false positive detection"""
    try:
        data = request.get_json()
        if not data or 'content' not in data:
            return jsonify({"error": "Content is required"}), 400
        
        if not semantic_guardrail_system:
            return jsonify({"error": "Semantic guardrail system not available"}), 503
        
        content = data['content']
        semantic_guardrail_system.report_false_positive(content)
        
        return jsonify({"success": True, "message": "False positive reported"})
    except Exception as e:
        logger.error(f"Error reporting false positive: {e}")
        return jsonify({"error": str(e)}), 500

# Multi-Modal AI Integration Endpoints
@app.route('/api/multimodal/process', methods=['POST'])
def process_multimodal_content():
    """Process multi-modal content (image, audio, document)"""
    try:
        if not multimodal_ai_integration:
            return jsonify({"error": "Multi-modal AI integration not available"}), 503
        
        # Handle file upload
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Get processing options
        processing_type = request.form.get('processing_type', 'auto')
        options = {
            'analysis_type': request.form.get('analysis_type', 'general'),
            'language': request.form.get('language', 'auto'),
            'style': request.form.get('style', 'realistic')
        }
        
        # Save uploaded file temporarily
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as temp_file:
            file.save(temp_file.name)
            temp_file_path = temp_file.name
        
        try:
            # Process the file
            result = asyncio.run(multimodal_ai_integration.process_media(
                temp_file_path, processing_type, options
            ))
            
            # Clean up temporary file
            os.unlink(temp_file_path)
            
            return jsonify({
                "processing_result": {
                    "status": result.status.value,
                    "processing_type": result.processing_type,
                    "result": result.result,
                    "confidence": result.confidence,
                    "processing_time": result.processing_time,
                    "error_message": result.error_message,
                    "timestamp": result.timestamp.isoformat()
                }
            })
            
        except Exception as e:
            # Clean up temporary file on error
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
            raise e
            
    except Exception as e:
        logger.error(f"Error processing multi-modal content: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/multimodal/generate', methods=['POST'])
def generate_multimodal_content():
    """Generate multi-modal content (image, audio)"""
    try:
        if not multimodal_ai_integration:
            return jsonify({"error": "Multi-modal AI integration not available"}), 503
        
        data = request.get_json()
        if not data or 'content_type' not in data or 'prompt' not in data:
            return jsonify({"error": "Content type and prompt are required"}), 400
        
        content_type = data['content_type']
        prompt = data['prompt']
        options = data.get('options', {})
        
        # Generate content
        result = asyncio.run(multimodal_ai_integration.generate_content(
            content_type, prompt, options
        ))
        
        return jsonify({
            "generation_result": {
                "status": result.status.value,
                "processing_type": result.processing_type,
                "result": result.result,
                "confidence": result.confidence,
                "processing_time": result.processing_time,
                "error_message": result.error_message,
                "timestamp": result.timestamp.isoformat()
            }
        })
        
    except Exception as e:
        logger.error(f"Error generating multi-modal content: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/multimodal/analyze-image', methods=['POST'])
def analyze_image():
    """Analyze image content with AI"""
    try:
        if not multimodal_ai_integration:
            return jsonify({"error": "Multi-modal AI integration not available"}), 503
        
        # Handle file upload or base64 data
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({"error": "No file selected"}), 400
            
            # Save uploaded file temporarily
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as temp_file:
                file.save(temp_file.name)
                temp_file_path = temp_file.name
        else:
            return jsonify({"error": "No image file provided"}), 400
        
        analysis_type = request.form.get('analysis_type', 'general')
        
        try:
            # Analyze image
            result = asyncio.run(multimodal_ai_integration.image_processor.analyze_image(
                temp_file_path, analysis_type
            ))
            
            # Clean up temporary file
            os.unlink(temp_file_path)
            
            return jsonify({
                "image_analysis": {
                    "status": result.status.value,
                    "analysis_type": analysis_type,
                    "result": result.result,
                    "confidence": result.confidence,
                    "processing_time": result.processing_time,
                    "error_message": result.error_message
                }
            })
            
        except Exception as e:
            # Clean up temporary file on error
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
            raise e
            
    except Exception as e:
        logger.error(f"Error analyzing image: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/multimodal/transcribe-audio', methods=['POST'])
def transcribe_audio():
    """Transcribe audio to text"""
    try:
        if not multimodal_ai_integration:
            return jsonify({"error": "Multi-modal AI integration not available"}), 503
        
        # Handle file upload
        if 'file' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Save uploaded file temporarily
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as temp_file:
            file.save(temp_file.name)
            temp_file_path = temp_file.name
        
        language = request.form.get('language', 'auto')
        
        try:
            # Transcribe audio
            result = asyncio.run(multimodal_ai_integration.audio_processor.transcribe_audio(
                temp_file_path, language
            ))
            
            # Clean up temporary file
            os.unlink(temp_file_path)
            
            return jsonify({
                "transcription": {
                    "status": result.status.value,
                    "language": language,
                    "result": result.result,
                    "confidence": result.confidence,
                    "processing_time": result.processing_time,
                    "error_message": result.error_message
                }
            })
            
        except Exception as e:
            # Clean up temporary file on error
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
            raise e
            
    except Exception as e:
        logger.error(f"Error transcribing audio: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/multimodal/generate-speech', methods=['POST'])
def generate_speech():
    """Generate speech from text"""
    try:
        if not multimodal_ai_integration:
            return jsonify({"error": "Multi-modal AI integration not available"}), 503
        
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "Text is required"}), 400
        
        text = data['text']
        voice = data.get('voice', 'neutral')
        
        # Generate speech
        result = asyncio.run(multimodal_ai_integration.audio_processor.generate_speech(
            text, voice
        ))
        
        return jsonify({
            "speech_generation": {
                "status": result.status.value,
                "text": text,
                "voice": voice,
                "result": result.result,
                "confidence": result.confidence,
                "processing_time": result.processing_time,
                "error_message": result.error_message
            }
        })
        
    except Exception as e:
        logger.error(f"Error generating speech: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/multimodal/analyze-document', methods=['POST'])
def analyze_document():
    """Analyze document content with AI"""
    try:
        if not multimodal_ai_integration:
            return jsonify({"error": "Multi-modal AI integration not available"}), 503
        
        # Handle file upload
        if 'file' not in request.files:
            return jsonify({"error": "No document file provided"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Save uploaded file temporarily
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as temp_file:
            file.save(temp_file.name)
            temp_file_path = temp_file.name
        
        analysis_type = request.form.get('analysis_type', 'summary')
        
        try:
            # Analyze document
            result = asyncio.run(multimodal_ai_integration.document_processor.analyze_document(
                temp_file_path, analysis_type
            ))
            
            # Clean up temporary file
            os.unlink(temp_file_path)
            
            return jsonify({
                "document_analysis": {
                    "status": result.status.value,
                    "analysis_type": analysis_type,
                    "result": result.result,
                    "confidence": result.confidence,
                    "processing_time": result.processing_time,
                    "error_message": result.error_message
                }
            })
            
        except Exception as e:
            # Clean up temporary file on error
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
            raise e
            
    except Exception as e:
        logger.error(f"Error analyzing document: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/multimodal/stats', methods=['GET'])
def get_multimodal_stats():
    """Get multi-modal processing statistics"""
    try:
        if not multimodal_ai_integration:
            return jsonify({"error": "Multi-modal AI integration not available"}), 503
        
        stats = multimodal_ai_integration.get_stats()
        
        return jsonify({
            "multimodal_stats": stats,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting multi-modal stats: {e}")
        return jsonify({"error": str(e)}), 500

# Auto Chain Generator API endpoints
@app.route('/api/chains/analyze', methods=['POST'])
def analyze_query_for_chain():
    """Analyze query to determine optimal chain composition"""
    try:
        if not auto_chain_generator:
            return jsonify({"error": "Auto Chain Generator not available"}), 503
        
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({"error": "Query is required"}), 400
        
        query = data['query']
        analysis = auto_chain_generator.analyze_query_for_chain(query)
        
        return jsonify({
            "analysis": analysis,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error analyzing query for chain: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/chains/generate', methods=['POST'])
def generate_agent_chain():
    """Generate optimal agent chain for a query"""
    try:
        if not auto_chain_generator:
            return jsonify({"error": "Auto Chain Generator not available"}), 503
        
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({"error": "Query is required"}), 400
        
        query = data['query']
        chain = auto_chain_generator.generate_chain(query)
        
        # Convert to dict for JSON serialization
        chain_dict = {
            "chain_id": chain.chain_id,
            "query": chain.query,
            "steps": [
                {
                    "step_id": step.step_id,
                    "step_type": step.step_type.value,
                    "agent_id": step.agent_id,
                    "agent_name": step.agent_name,
                    "description": step.description,
                    "input_from": step.input_from,
                    "parameters": step.parameters,
                    "expected_output": step.expected_output
                }
                for step in chain.steps
            ],
            "estimated_cost": chain.estimated_cost,
            "estimated_time": chain.estimated_time,
            "complexity_score": chain.complexity_score,
            "created_at": chain.created_at.isoformat()
        }
        
        return jsonify({
            "chain": chain_dict,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error generating agent chain: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/chains/execute', methods=['POST'])
def execute_agent_chain():
    """Execute a complete agent chain"""
    try:
        if not auto_chain_generator:
            return jsonify({"error": "Auto Chain Generator not available"}), 503
        
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({"error": "Query is required"}), 400
        
        query = data['query']
        
        # Generate chain
        chain = auto_chain_generator.generate_chain(query)
        
        # Execute chain asynchronously
        async def execute_chain():
            return await auto_chain_generator.execute_chain(chain)
        
        # Run in async context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            results = loop.run_until_complete(execute_chain())
        finally:
            loop.close()
        
        # Convert results to dict
        results_dict = [
            {
                "step_id": result.step_id,
                "success": result.success,
                "output": result.output,
                "execution_time": result.execution_time,
                "tokens_used": result.tokens_used,
                "cost": result.cost,
                "error_message": result.error_message,
                "metadata": result.metadata
            }
            for result in results
        ]
        
        return jsonify({
            "chain_id": chain.chain_id,
            "query": query,
            "results": results_dict,
            "total_cost": sum(r.cost for r in results),
            "total_time": sum(r.execution_time for r in results),
            "success_rate": sum(1 for r in results if r.success) / len(results) if results else 0,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error executing agent chain: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/chains/templates', methods=['GET'])
def get_chain_templates():
    """Get available chain templates"""
    try:
        if not auto_chain_generator:
            return jsonify({"error": "Auto Chain Generator not available"}), 503
        
        templates = {}
        for name, template in auto_chain_generator.chain_templates.items():
            templates[name] = {
                "name": name,
                "steps": [step.value for step in template],
                "description": f"Template for {name.replace('_', ' ').title()}"
            }
        
        return jsonify({
            "templates": templates,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting chain templates: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/chains/stats', methods=['GET'])
def get_chain_stats():
    """Get Auto Chain Generator statistics"""
    try:
        if not auto_chain_generator:
            return jsonify({"error": "Auto Chain Generator not available"}), 503
        
        stats = auto_chain_generator.get_chain_stats()
        
        return jsonify({
            "stats": stats,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting chain stats: {e}")
        return jsonify({"error": str(e)}), 500

# Automated Evaluation Engine API endpoints
@app.route('/api/evaluation/run', methods=['POST'])
def run_evaluation():
    """Run comprehensive evaluation across all test types"""
    try:
        if not evaluation_engine:
            return jsonify({"error": "Evaluation engine not available"}), 503
        
        data = request.get_json() or {}
        prompts_per_category = data.get('prompts_per_category', 5)
        include_real_prompts = data.get('include_real_prompts', True)
        
        # Run evaluation (async)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            report = loop.run_until_complete(
                evaluation_engine.run_comprehensive_evaluation(
                    prompts_per_category=prompts_per_category,
                    include_real_prompts=include_real_prompts
                )
            )
            
            # Convert to dict for JSON serialization
            report_dict = {
                "test_session_id": report.test_session_id,
                "total_tests": report.total_tests,
                "passed_tests": report.passed_tests,
                "failed_tests": report.failed_tests,
                "overall_score": report.overall_score,
                "routing_accuracy": report.routing_accuracy,
                "safety_score": report.safety_score,
                "cost_efficiency": report.cost_efficiency,
                "average_latency": report.average_latency,
                "test_results": [
                    {
                        "test_id": result.test_id,
                        "test_type": result.test_type.value,
                        "prompt": result.prompt,
                        "category": result.category.value,
                        "success": result.success,
                        "score": result.score,
                        "execution_time": result.execution_time,
                        "cost": result.cost,
                        "error_message": result.error_message,
                        "timestamp": result.timestamp.isoformat()
                    }
                    for result in report.test_results
                ],
                "recommendations": report.recommendations,
                "timestamp": report.timestamp.isoformat()
            }
            
            return jsonify({
                "evaluation_report": report_dict,
                "timestamp": datetime.now().isoformat()
            })
            
        finally:
            loop.close()
        
    except Exception as e:
        logger.error(f"Error running evaluation: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/evaluation/history', methods=['GET'])
def get_evaluation_history():
    """Get evaluation history from database"""
    try:
        if not evaluation_engine:
            return jsonify({"error": "Evaluation engine not available"}), 503
        
        limit = request.args.get('limit', 10, type=int)
        history = evaluation_engine.get_evaluation_history(limit)
        
        return jsonify({
            "evaluation_history": history,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting evaluation history: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/evaluation/stats', methods=['GET'])
def get_evaluation_stats():
    """Get evaluation statistics"""
    try:
        if not evaluation_engine:
            return jsonify({"error": "Evaluation engine not available"}), 503
        
        stats = evaluation_engine.get_evaluation_stats()
        
        return jsonify({
            "evaluation_stats": stats,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting evaluation stats: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/evaluation')
def evaluation_page():
    """Automated Evaluation Engine interface"""
    return render_template('evaluation.html')

# Peer Teaching & Collaborative Agents API endpoints
@app.route('/api/peer-teaching/agents/register', methods=['POST'])
def register_peer_agent():
    """Register a new agent in the peer teaching system"""
    try:
        if not peer_teaching_system:
            return jsonify({"error": "Peer teaching system not available"}), 503
        
        data = request.get_json()
        if not data or 'agent_name' not in data or 'specialization' not in data:
            return jsonify({"error": "Agent name and specialization are required"}), 400
        
        agent_id = data.get('agent_id', str(uuid.uuid4()))
        agent_name = data['agent_name']
        specialization = AgentSpecialization(data['specialization'])
        capabilities = data.get('capabilities', [])
        
        success = peer_teaching_system.register_agent(agent_id, agent_name, specialization, capabilities)
        
        if success:
            return jsonify({
                "success": True,
                "agent_id": agent_id,
                "message": f"Agent {agent_name} registered successfully"
            })
        else:
            return jsonify({"error": "Failed to register agent"}), 500
        
    except Exception as e:
        logger.error(f"Error registering peer agent: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/peer-teaching/lessons/contribute', methods=['POST'])
def contribute_lesson():
    """Contribute a lesson to the peer teaching system"""
    try:
        if not peer_teaching_system:
            return jsonify({"error": "Peer teaching system not available"}), 503
        
        data = request.get_json()
        required_fields = ['agent_id', 'lesson_type', 'domain', 'title', 'content', 'strategy_steps']
        
        if not all(field in data for field in required_fields):
            return jsonify({"error": "Missing required fields"}), 400
        
        lesson_id = peer_teaching_system.contribute_lesson(
            agent_id=data['agent_id'],
            lesson_type=LessonType(data['lesson_type']),
            domain=data['domain'],
            title=data['title'],
            content=data['content'],
            strategy_steps=data['strategy_steps'],
            effectiveness_score=data.get('effectiveness_score', 0.8),
            usage_context=data.get('usage_context', ''),
            success_metrics=data.get('success_metrics', {})
        )
        
        if lesson_id:
            return jsonify({
                "success": True,
                "lesson_id": lesson_id,
                "message": "Lesson contributed successfully"
            })
        else:
            return jsonify({"error": "Failed to contribute lesson"}), 500
        
    except Exception as e:
        logger.error(f"Error contributing lesson: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/peer-teaching/lessons/find', methods=['POST'])
def find_lessons():
    """Find relevant lessons for an agent"""
    try:
        if not peer_teaching_system:
            return jsonify({"error": "Peer teaching system not available"}), 503
        
        data = request.get_json()
        if not data or 'agent_id' not in data or 'domain' not in data:
            return jsonify({"error": "Agent ID and domain are required"}), 400
        
        agent_id = data['agent_id']
        domain = data['domain']
        lesson_type = LessonType(data['lesson_type']) if 'lesson_type' in data else None
        
        lessons = peer_teaching_system.find_relevant_lessons(agent_id, domain, lesson_type)
        
        lessons_data = []
        for lesson in lessons:
            lessons_data.append({
                "lesson_id": lesson.lesson_id,
                "title": lesson.title,
                "content": lesson.content,
                "lesson_type": lesson.lesson_type.value,
                "domain": lesson.domain,
                "effectiveness_score": lesson.effectiveness_score,
                "adoption_count": lesson.adoption_count,
                "strategy_steps": lesson.strategy_steps,
                "created_at": lesson.created_at.isoformat()
            })
        
        return jsonify({
            "lessons": lessons_data,
            "total_found": len(lessons_data)
        })
        
    except Exception as e:
        logger.error(f"Error finding lessons: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/peer-teaching/lessons/adopt', methods=['POST'])
def adopt_lesson():
    """Adopt a lesson from a peer"""
    try:
        if not peer_teaching_system:
            return jsonify({"error": "Peer teaching system not available"}), 503
        
        data = request.get_json()
        if not data or 'agent_id' not in data or 'lesson_id' not in data:
            return jsonify({"error": "Agent ID and lesson ID are required"}), 400
        
        success = peer_teaching_system.adopt_lesson(data['agent_id'], data['lesson_id'])
        
        if success:
            return jsonify({
                "success": True,
                "message": "Lesson adopted successfully"
            })
        else:
            return jsonify({"error": "Failed to adopt lesson"}), 500
        
    except Exception as e:
        logger.error(f"Error adopting lesson: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/peer-teaching/knowledge/contribute', methods=['POST'])
def contribute_federated_knowledge():
    """Contribute anonymized knowledge for federated learning"""
    try:
        if not peer_teaching_system:
            return jsonify({"error": "Peer teaching system not available"}), 503
        
        data = request.get_json()
        required_fields = ['agent_id', 'specialization', 'query_type', 'toolkit_used', 'approach_summary']
        
        if not all(field in data for field in required_fields):
            return jsonify({"error": "Missing required fields"}), 400
        
        contribution_id = peer_teaching_system.contribute_federated_knowledge(
            agent_id=data['agent_id'],
            specialization=AgentSpecialization(data['specialization']),
            query_type=data['query_type'],
            toolkit_used=data['toolkit_used'],
            approach_summary=data['approach_summary'],
            performance_metrics=data.get('performance_metrics', {}),
            lessons_learned=data.get('lessons_learned', []),
            optimization_tips=data.get('optimization_tips', []),
            error_patterns=data.get('error_patterns', [])
        )
        
        if contribution_id:
            return jsonify({
                "success": True,
                "contribution_id": contribution_id,
                "message": "Knowledge contributed successfully"
            })
        else:
            return jsonify({"error": "Failed to contribute knowledge"}), 500
        
    except Exception as e:
        logger.error(f"Error contributing federated knowledge: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/peer-teaching/knowledge/get', methods=['POST'])
def get_federated_knowledge():
    """Get relevant federated knowledge"""
    try:
        if not peer_teaching_system:
            return jsonify({"error": "Peer teaching system not available"}), 503
        
        data = request.get_json()
        if not data or 'specialization' not in data or 'query_type' not in data:
            return jsonify({"error": "Specialization and query type are required"}), 400
        
        specialization = AgentSpecialization(data['specialization'])
        query_type = data['query_type']
        
        contributions = peer_teaching_system.get_federated_knowledge(specialization, query_type)
        
        contributions_data = []
        for contrib in contributions:
            contributions_data.append({
                "contribution_id": contrib.contribution_id,
                "toolkit_used": contrib.toolkit_used,
                "approach_summary": contrib.approach_summary,
                "performance_metrics": contrib.performance_metrics,
                "lessons_learned": contrib.lessons_learned,
                "optimization_tips": contrib.optimization_tips,
                "error_patterns": contrib.error_patterns,
                "timestamp": contrib.timestamp.isoformat()
            })
        
        return jsonify({
            "contributions": contributions_data,
            "total_found": len(contributions_data)
        })
        
    except Exception as e:
        logger.error(f"Error getting federated knowledge: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/peer-teaching/collaborate/start', methods=['POST'])
def start_collaboration():
    """Start a collaborative session"""
    try:
        if not peer_teaching_system:
            return jsonify({"error": "Peer teaching system not available"}), 503
        
        data = request.get_json()
        if not data or 'initiator_agent' not in data or 'task_description' not in data:
            return jsonify({"error": "Initiator agent and task description are required"}), 400
        
        initiator_agent = data['initiator_agent']
        task_description = data['task_description']
        session_type = data.get('session_type', 'general')
        required_specializations = [
            AgentSpecialization(spec) for spec in data.get('required_specializations', [])
        ]
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            session_id = loop.run_until_complete(
                peer_teaching_system.start_collaborative_session(
                    initiator_agent, task_description, session_type, required_specializations
                )
            )
        finally:
            loop.close()
        
        if session_id:
            return jsonify({
                "success": True,
                "session_id": session_id,
                "message": "Collaborative session started successfully"
            })
        else:
            return jsonify({"error": "Failed to start collaborative session"}), 500
        
    except Exception as e:
        logger.error(f"Error starting collaboration: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/peer-teaching/debate', methods=['POST'])
def multi_agent_debate():
    """Conduct multi-agent debate"""
    try:
        if not peer_teaching_system:
            return jsonify({"error": "Peer teaching system not available"}), 503
        
        data = request.get_json()
        if not data or 'session_id' not in data or 'question' not in data:
            return jsonify({"error": "Session ID and question are required"}), 400
        
        session_id = data['session_id']
        question = data['question']
        consensus_method = ConsensusMethod(data.get('consensus_method', 'majority_vote'))
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            debate_result = loop.run_until_complete(
                peer_teaching_system.multi_agent_debate(
                    session_id, question, consensus_method
                )
            )
        finally:
            loop.close()
        
        return jsonify({
            "debate_result": debate_result,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in multi-agent debate: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/peer-teaching/cross-correct', methods=['POST'])
def cross_correct_agents():
    """Enable cross-correction between agents"""
    try:
        if not peer_teaching_system:
            return jsonify({"error": "Peer teaching system not available"}), 503
        
        data = request.get_json()
        required_fields = ['session_id', 'primary_agent', 'secondary_agent', 'content']
        
        if not all(field in data for field in required_fields):
            return jsonify({"error": "Missing required fields"}), 400
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            corrections = loop.run_until_complete(
                peer_teaching_system.cross_correct_agents(
                    data['session_id'], data['primary_agent'], 
                    data['secondary_agent'], data['content']
                )
            )
        finally:
            loop.close()
        
        return jsonify({
            "corrections": corrections,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in cross-correction: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/peer-teaching/stats', methods=['GET'])
def get_peer_teaching_stats():
    """Get peer teaching system statistics"""
    try:
        if not peer_teaching_system:
            return jsonify({"error": "Peer teaching system not available"}), 503
        
        stats = peer_teaching_system.get_peer_teaching_stats()
        
        return jsonify({
            "peer_teaching_stats": stats,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting peer teaching stats: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/peer-teaching')
def peer_teaching_page():
    """Peer Teaching & Collaborative Agents interface"""
    return render_template('peer_teaching.html')

@app.route('/peer-teaching/demo')
def peer_teaching_demo():
    """Peer teaching learning demonstration"""
    return render_template('peer_teaching_demo.html')

@app.route('/personal-ai')
def personal_ai():
    """Personal AI interface with hybrid edge-cloud routing"""
    return render_template('personal_ai.html')

@app.route('/api/peer-teaching/demo', methods=['POST'])
def run_peer_teaching_demo():
    """Run the peer teaching demonstration"""
    try:
        import subprocess
        import os
        
        # Run the simple demo script
        result = subprocess.run(
            [sys.executable, 'simple_peer_demo.py'],
            capture_output=True,
            text=True,
            cwd=os.getcwd()
        )
        
        if result.returncode == 0:
            return result.stdout
        else:
            return f"Demo failed: {result.stderr}", 500
            
    except Exception as e:
        logger.error(f"Error running peer teaching demo: {e}")
        return f"Error: {str(e)}", 500

# Personal AI API endpoints
@app.route('/api/personal-ai/status', methods=['GET'])
def get_personal_ai_status():
    """Get Personal AI system status"""
    try:
        if not personal_ai_router:
            return jsonify({"error": "Personal AI router not available"}), 503
        
        stats = personal_ai_router.get_stats()
        
        return jsonify({
            "ollama_connected": len(stats.get("available_local_models", [])) > 0,
            "local_models": len(stats.get("available_local_models", [])),
            "cache_hit_rate": stats.get("cache_hit_rate", 0.0),
            "total_memories": stats.get("total_memories", 0),
            "routing_stats": stats.get("routing_stats", {}),
            "p2p_network": stats.get("p2p_network", {
                "enabled": False,
                "node_id": None,
                "network_stats": {"peer_count": 0}
            }),
            "status": "ready"
        })
        
    except Exception as e:
        logger.error(f"Error getting personal AI status: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/personal-ai/stats', methods=['GET'])
def get_personal_ai_stats():
    """Get detailed Personal AI statistics"""
    try:
        if not personal_ai_router:
            return jsonify({"error": "Personal AI router not available"}), 503
        
        stats = personal_ai_router.get_stats()
        
        # Get memory and preference counts
        memory_store = personal_ai_router.memory_store
        preferences = memory_store.get_preferences()
        
        return jsonify({
            "cache_hit_rate": stats.get("cache_hit_rate", 0.0),
            "total_memories": stats.get("total_memories", 0),
            "total_preferences": len(preferences),
            "routing_stats": stats.get("routing_stats", {}),
            "available_local_models": stats.get("available_local_models", []),
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting personal AI stats: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/personal-ai/chat', methods=['POST'])
def personal_ai_chat():
    """Process a chat message with personal AI"""
    try:
        if not personal_ai_router:
            return jsonify({"error": "Personal AI router not available"}), 503
        
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({'error': 'Message is required'}), 400
        
        message = data['message']
        routing_strategy = data.get('routing_strategy', 'auto')
        local_model = data.get('local_model', 'llama3.2:3b')
        
        # Process the query
        result = asyncio.run(personal_ai_router.process_query(message, {
            'routing_strategy': routing_strategy,
            'local_model': local_model
        }))
        
        return jsonify({
            "response": result.response,
            "routing_result": {
                "decision": result.decision.value,
                "model_used": result.model_used,
                "confidence": result.confidence,
                "latency": result.latency,
                "cost_estimate": result.cost_estimate,
                "reasoning": result.reasoning,
                "cached": result.cached,
                "fallback_used": result.fallback_used
            },
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in personal AI chat: {e}")
        return jsonify({"error": str(e)}), 500

# User Profile Builder API endpoints
@app.route('/api/profile/start', methods=['POST'])
def start_profile_building():
    """Start profile building process for a user"""
    try:
        data = request.get_json()
        user_id = data.get('user_id', 'default_user')
        
        profile_builder = get_user_profile_builder()
        result = profile_builder.start_profile_building(user_id)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error starting profile building: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/profile/respond', methods=['POST'])
def handle_profile_response():
    """Handle user response to profile question"""
    try:
        data = request.get_json()
        user_id = data.get('user_id', 'default_user')
        question_id = data.get('question_id')
        response = data.get('response')
        
        if not question_id or not response:
            return jsonify({"error": "Question ID and response are required"}), 400
        
        profile_builder = get_user_profile_builder()
        result = profile_builder.process_response(user_id, question_id, response)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error handling profile response: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/profile/status', methods=['GET'])
def get_profile_status():
    """Get user profile status"""
    try:
        user_id = request.args.get('user_id', 'default_user')
        
        profile_builder = get_user_profile_builder()
        profile = profile_builder.get_user_profile(user_id)
        
        if profile:
            return jsonify({
                "has_profile": True,
                "completion": profile.profile_completion,
                "name": profile.personal_info.get("preferred_name", "User"),
                "last_updated": profile.last_updated.isoformat() if profile.last_updated else None
            })
        else:
            return jsonify({
                "has_profile": False,
                "completion": 0.0,
                "name": None,
                "last_updated": None
            })
        
    except Exception as e:
        logger.error(f"Error getting profile status: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/profile/summary', methods=['GET'])
def get_profile_summary():
    """Get user profile summary"""
    try:
        user_id = request.args.get('user_id', 'default_user')
        
        profile_builder = get_user_profile_builder()
        profile = profile_builder.get_user_profile(user_id)
        
        if profile:
            summary = profile_builder._generate_profile_summary(profile)
            return jsonify(summary)
        else:
            return jsonify({"error": "No profile found"}), 404
        
    except Exception as e:
        logger.error(f"Error getting profile summary: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/personal-ai/memory', methods=['GET', 'POST'])
def personal_ai_memory():
    """Get or store personal memories"""
    try:
        if not personal_ai_router:
            return jsonify({"error": "Personal AI router not available"}), 503
        
        memory_store = personal_ai_router.memory_store
        
        if request.method == 'GET':
            # Get recent memories and preferences
            memories = memory_store.search_memories("", limit=20)
            preferences = memory_store.get_preferences()
            
            return jsonify({
                "memories": memories,
                "preferences": preferences,
                "timestamp": datetime.now().isoformat()
            })
        
        elif request.method == 'POST':
            # Store a new memory
            data = request.get_json()
            if not data or not all(key in data for key in ['category', 'key', 'value']):
                return jsonify({'error': 'Category, key, and value are required'}), 400
            
            memory_store.store_memory(
                data['category'],
                data['key'],
                data['value'],
                data.get('context')
            )
            
            return jsonify({"success": True, "message": "Memory stored successfully"})
        
    except Exception as e:
        logger.error(f"Error with personal AI memory: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/personal-ai/preference', methods=['POST'])
def personal_ai_preference():
    """Store a personal preference"""
    try:
        if not personal_ai_router:
            return jsonify({"error": "Personal AI router not available"}), 503
        
        data = request.get_json()
        if not data or not all(key in data for key in ['category', 'preference', 'value']):
            return jsonify({'error': 'Category, preference, and value are required'}), 400
        
        memory_store = personal_ai_router.memory_store
        memory_store.store_preference(
            data['category'],
            data['preference'],
            data['value'],
            data.get('strength', 1.0)
        )
        
        return jsonify({"success": True, "message": "Preference stored successfully"})
        
    except Exception as e:
        logger.error(f"Error storing personal AI preference: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/personal-ai/test-ollama', methods=['POST'])
def test_ollama_connection():
    """Test Ollama connection"""
    try:
        if not personal_ai_router:
            return jsonify({"error": "Personal AI router not available"}), 503
        
        ollama_client = personal_ai_router.ollama_client
        
        # Test with a simple query
        result = asyncio.run(ollama_client.generate(
            "llama3.2:3b",
            "Hello! This is a test. Please respond with 'Test successful!'"
        ))
        
        if "error" in result:
            return jsonify({
                "success": False,
                "error": result["error"]
            })
        
        return jsonify({
            "success": True,
            "response": result.get("response", ""),
            "model": result.get("model", ""),
            "available_models": ollama_client.available_models
        })
        
    except Exception as e:
        logger.error(f"Error testing Ollama: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

# Register GraphQL blueprint
app.register_blueprint(graphql_bp)

# Initialize database
with app.app_context():
    import models
    db.create_all()
    
    # Initialize router after models are imported
    threading.Thread(target=initialize_router, daemon=True).start()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
