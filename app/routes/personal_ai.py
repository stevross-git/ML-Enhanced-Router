"""
Personal AI Routes
Hybrid edge-cloud routing, personal memory, and local AI capabilities
"""

from datetime import datetime
from flask import Blueprint, request, jsonify, current_app, session

from ..utils.decorators import rate_limit, validate_json
from ..utils.exceptions import ValidationError, ServiceError

# Create blueprint
personal_ai_bp = Blueprint('personal_ai', __name__)

def get_personal_ai_router():
    """Get personal AI router instance"""
    try:
        from personal_ai_router import get_personal_ai_router
        return get_personal_ai_router()
    except ImportError:
        current_app.logger.warning("Personal AI router not available")
        return None

@personal_ai_bp.route('/status', methods=['GET'])
@rate_limit("100 per minute")
def get_personal_ai_status():
    """Get Personal AI system status"""
    try:
        personal_ai_router = get_personal_ai_router()
        if not personal_ai_router:
            return jsonify({"error": "Personal AI router not available"}), 503
        
        stats = personal_ai_router.get_stats()
        
        return jsonify({
            "ollama_connected": len(stats.get("available_local_models", [])) > 0,
            "local_models": len(stats.get("available_local_models", [])),
            "available_models": stats.get("available_local_models", []),
            "cache_hit_rate": stats.get("cache_hit_rate", 0.0),
            "total_memories": stats.get("total_memories", 0),
            "routing_stats": stats.get("routing_stats", {}),
            "p2p_network": stats.get("p2p_network", {
                "enabled": False,
                "node_id": None,
                "network_stats": {"peer_count": 0}
            }),
            "edge_capabilities": stats.get("edge_capabilities", {}),
            "cloud_fallback": stats.get("cloud_fallback", True),
            "status": "ready",
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        current_app.logger.error(f"Error getting personal AI status: {e}")
        return jsonify({"error": str(e)}), 500

@personal_ai_bp.route('/query', methods=['POST'])
@rate_limit("100 per minute")
@validate_json(['query'])
def process_personal_query():
    """Process query through personal AI router"""
    try:
        personal_ai_router = get_personal_ai_router()
        if not personal_ai_router:
            return jsonify({"error": "Personal AI router not available"}), 503
        
        data = request.get_json()
        query = data['query']
        user_id = data.get('user_id', session.get('user_id', 'anonymous'))
        prefer_local = data.get('prefer_local', True)
        context = data.get('context', {})
        
        # Process through personal AI router
        result = personal_ai_router.route_query(
            query=query,
            user_id=user_id,
            prefer_local=prefer_local,
            context=context
        )
        
        return jsonify({
            "status": "success",
            "result": {
                "response": result.response,
                "routing_decision": result.routing_decision,
                "model_used": result.model_used,
                "processing_location": result.processing_location,  # "local" or "cloud"
                "confidence": result.confidence,
                "processing_time": result.processing_time,
                "cached": result.cached,
                "memory_used": result.memory_used,
                "tokens_used": result.tokens_used
            },
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        current_app.logger.error(f"Error processing personal query: {e}")
        return jsonify({"error": str(e)}), 500

@personal_ai_bp.route('/memory/store', methods=['POST'])
@rate_limit("200 per minute")
@validate_json(['content'])
def store_memory():
    """Store information in personal memory"""
    try:
        personal_ai_router = get_personal_ai_router()
        if not personal_ai_router:
            return jsonify({"error": "Personal AI router not available"}), 503
        
        data = request.get_json()
        content = data['content']
        user_id = data.get('user_id', session.get('user_id', 'anonymous'))
        memory_type = data.get('type', 'general')
        tags = data.get('tags', [])
        importance = data.get('importance', 0.5)
        
        # Store memory
        memory_id = personal_ai_router.memory_store.store_memory(
            user_id=user_id,
            content=content,
            memory_type=memory_type,
            tags=tags,
            importance=importance
        )
        
        if memory_id:
            return jsonify({
                "status": "success",
                "memory_id": memory_id,
                "message": "Memory stored successfully",
                "user_id": user_id,
                "memory_type": memory_type,
                "timestamp": datetime.now().isoformat()
            })
        else:
            return jsonify({"error": "Failed to store memory"}), 500
        
    except Exception as e:
        current_app.logger.error(f"Error storing memory: {e}")
        return jsonify({"error": str(e)}), 500

@personal_ai_bp.route('/memory/search', methods=['POST'])
@rate_limit("100 per minute")
@validate_json(['query'])
def search_memory():
    """Search personal memory"""
    try:
        personal_ai_router = get_personal_ai_router()
        if not personal_ai_router:
            return jsonify({"error": "Personal AI router not available"}), 503
        
        data = request.get_json()
        query = data['query']
        user_id = data.get('user_id', session.get('user_id', 'anonymous'))
        limit = data.get('limit', 10)
        memory_type = data.get('type')
        
        # Search memories
        memories = personal_ai_router.memory_store.search_memories(
            user_id=user_id,
            query=query,
            limit=limit,
            memory_type=memory_type
        )
        
        # Convert to dict format
        memories_data = []
        for memory in memories:
            memories_data.append({
                "memory_id": memory.memory_id,
                "content": memory.content,
                "memory_type": memory.memory_type,
                "tags": memory.tags,
                "importance": memory.importance,
                "relevance_score": memory.relevance_score,
                "created_at": memory.created_at.isoformat(),
                "last_accessed": memory.last_accessed.isoformat() if memory.last_accessed else None
            })
        
        return jsonify({
            "status": "success",
            "query": query,
            "memories": memories_data,
            "count": len(memories_data),
            "user_id": user_id,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        current_app.logger.error(f"Error searching memory: {e}")
        return jsonify({"error": str(e)}), 500

@personal_ai_bp.route('/memory/<memory_id>', methods=['GET'])
@rate_limit("100 per minute")
def get_memory(memory_id):
    """Get specific memory by ID"""
    try:
        personal_ai_router = get_personal_ai_router()
        if not personal_ai_router:
            return jsonify({"error": "Personal AI router not available"}), 503
        
        user_id = request.args.get('user_id', session.get('user_id', 'anonymous'))
        
        memory = personal_ai_router.memory_store.get_memory(memory_id, user_id)
        
        if not memory:
            return jsonify({"error": "Memory not found"}), 404
        
        return jsonify({
            "status": "success",
            "memory": {
                "memory_id": memory.memory_id,
                "content": memory.content,
                "memory_type": memory.memory_type,
                "tags": memory.tags,
                "importance": memory.importance,
                "access_count": memory.access_count,
                "created_at": memory.created_at.isoformat(),
                "last_accessed": memory.last_accessed.isoformat() if memory.last_accessed else None,
                "last_modified": memory.last_modified.isoformat() if memory.last_modified else None
            },
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        current_app.logger.error(f"Error getting memory: {e}")
        return jsonify({"error": str(e)}), 500

@personal_ai_bp.route('/memory/<memory_id>', methods=['PUT'])
@rate_limit("50 per minute")
@validate_json()
def update_memory(memory_id):
    """Update existing memory"""
    try:
        personal_ai_router = get_personal_ai_router()
        if not personal_ai_router:
            return jsonify({"error": "Personal AI router not available"}), 503
        
        data = request.get_json()
        user_id = data.get('user_id', session.get('user_id', 'anonymous'))
        
        # Get update fields
        updates = {}
        if 'content' in data:
            updates['content'] = data['content']
        if 'tags' in data:
            updates['tags'] = data['tags']
        if 'importance' in data:
            updates['importance'] = data['importance']
        if 'memory_type' in data:
            updates['memory_type'] = data['memory_type']
        
        success = personal_ai_router.memory_store.update_memory(
            memory_id=memory_id,
            user_id=user_id,
            updates=updates
        )
        
        if success:
            return jsonify({
                "status": "success",
                "message": "Memory updated successfully",
                "memory_id": memory_id,
                "updates": updates,
                "timestamp": datetime.now().isoformat()
            })
        else:
            return jsonify({"error": "Memory not found or update failed"}), 404
        
    except Exception as e:
        current_app.logger.error(f"Error updating memory: {e}")
        return jsonify({"error": str(e)}), 500

@personal_ai_bp.route('/memory/<memory_id>', methods=['DELETE'])
@rate_limit("30 per minute")
def delete_memory(memory_id):
    """Delete specific memory"""
    try:
        personal_ai_router = get_personal_ai_router()
        if not personal_ai_router:
            return jsonify({"error": "Personal AI router not available"}), 503
        
        user_id = request.args.get('user_id', session.get('user_id', 'anonymous'))
        
        success = personal_ai_router.memory_store.delete_memory(memory_id, user_id)
        
        if success:
            return jsonify({
                "status": "success",
                "message": "Memory deleted successfully",
                "memory_id": memory_id,
                "timestamp": datetime.now().isoformat()
            })
        else:
            return jsonify({"error": "Memory not found"}), 404
        
    except Exception as e:
        current_app.logger.error(f"Error deleting memory: {e}")
        return jsonify({"error": str(e)}), 500

@personal_ai_bp.route('/models/local', methods=['GET'])
@rate_limit("50 per minute")
def get_local_models():
    """Get available local models"""
    try:
        personal_ai_router = get_personal_ai_router()
        if not personal_ai_router:
            return jsonify({"error": "Personal AI router not available"}), 503
        
        models = personal_ai_router.get_local_models()
        
        return jsonify({
            "status": "success",
            "local_models": models,
            "count": len(models),
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        current_app.logger.error(f"Error getting local models: {e}")
        return jsonify({"error": str(e)}), 500

@personal_ai_bp.route('/models/switch', methods=['POST'])
@rate_limit("20 per minute")
@validate_json(['model_name'])
def switch_model():
    """Switch to a different local model"""
    try:
        personal_ai_router = get_personal_ai_router()
        if not personal_ai_router:
            return jsonify({"error": "Personal AI router not available"}), 503
        
        data = request.get_json()
        model_name = data['model_name']
        user_id = data.get('user_id', session.get('user_id', 'anonymous'))
        
        success = personal_ai_router.switch_model(model_name, user_id)
        
        if success:
            return jsonify({
                "status": "success",
                "message": f"Switched to model: {model_name}",
                "model_name": model_name,
                "user_id": user_id,
                "timestamp": datetime.now().isoformat()
            })
        else:
            return jsonify({"error": "Model not found or switch failed"}), 404
        
    except Exception as e:
        current_app.logger.error(f"Error switching model: {e}")
        return jsonify({"error": str(e)}), 500

@personal_ai_bp.route('/preferences', methods=['GET'])
@rate_limit("100 per minute")
def get_user_preferences():
    """Get user's AI preferences"""
    try:
        personal_ai_router = get_personal_ai_router()
        if not personal_ai_router:
            return jsonify({"error": "Personal AI router not available"}), 503
        
        user_id = request.args.get('user_id', session.get('user_id', 'anonymous'))
        
        preferences = personal_ai_router.get_user_preferences(user_id)
        
        return jsonify({
            "status": "success",
            "preferences": preferences,
            "user_id": user_id,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        current_app.logger.error(f"Error getting user preferences: {e}")
        return jsonify({"error": str(e)}), 500

@personal_ai_bp.route('/preferences', methods=['PUT'])
@rate_limit("20 per minute")
@validate_json()
def update_user_preferences():
    """Update user's AI preferences"""
    try:
        personal_ai_router = get_personal_ai_router()
        if not personal_ai_router:
            return jsonify({"error": "Personal AI router not available"}), 503
        
        data = request.get_json()
        user_id = data.get('user_id', session.get('user_id', 'anonymous'))
        preferences = data.get('preferences', {})
        
        success = personal_ai_router.update_user_preferences(user_id, preferences)
        
        if success:
            return jsonify({
                "status": "success",
                "message": "Preferences updated successfully",
                "user_id": user_id,
                "preferences": preferences,
                "timestamp": datetime.now().isoformat()
            })
        else:
            return jsonify({"error": "Failed to update preferences"}), 500
        
    except Exception as e:
        current_app.logger.error(f"Error updating user preferences: {e}")
        return jsonify({"error": str(e)}), 500

@personal_ai_bp.route('/p2p/status', methods=['GET'])
@rate_limit("50 per minute")
def get_p2p_status():
    """Get P2P network status"""
    try:
        personal_ai_router = get_personal_ai_router()
        if not personal_ai_router:
            return jsonify({"error": "Personal AI router not available"}), 503
        
        p2p_status = personal_ai_router.get_p2p_status()
        
        return jsonify({
            "status": "success",
            "p2p_network": p2p_status,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        current_app.logger.error(f"Error getting P2P status: {e}")
        return jsonify({"error": str(e)}), 500

@personal_ai_bp.route('/p2p/connect', methods=['POST'])
@rate_limit("10 per minute")
@validate_json(['peer_address'])
def connect_p2p_peer():
    """Connect to a P2P peer"""
    try:
        personal_ai_router = get_personal_ai_router()
        if not personal_ai_router:
            return jsonify({"error": "Personal AI router not available"}), 503
        
        data = request.get_json()
        peer_address = data['peer_address']
        
        success = personal_ai_router.connect_p2p_peer(peer_address)
        
        if success:
            return jsonify({
                "status": "success",
                "message": f"Connected to peer: {peer_address}",
                "peer_address": peer_address,
                "timestamp": datetime.now().isoformat()
            })
        else:
            return jsonify({"error": "Failed to connect to peer"}), 500
        
    except Exception as e:
        current_app.logger.error(f"Error connecting P2P peer: {e}")
        return jsonify({"error": str(e)}), 500

@personal_ai_bp.route('/analytics', methods=['GET'])
@rate_limit("50 per minute")
def get_personal_analytics():
    """Get personal AI usage analytics"""
    try:
        personal_ai_router = get_personal_ai_router()
        if not personal_ai_router:
            return jsonify({"error": "Personal AI router not available"}), 503
        
        user_id = request.args.get('user_id', session.get('user_id', 'anonymous'))
        time_range = request.args.get('range', '7d')  # 1d, 7d, 30d
        
        analytics = personal_ai_router.get_user_analytics(user_id, time_range)
        
        return jsonify({
            "status": "success",
            "analytics": analytics,
            "user_id": user_id,
            "time_range": time_range,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        current_app.logger.error(f"Error getting personal analytics: {e}")
        return jsonify({"error": str(e)}), 500
