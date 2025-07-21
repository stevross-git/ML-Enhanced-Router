"""
Core API Routes
Main query processing and routing endpoints
"""

import asyncio
from flask import Blueprint, request, jsonify, current_app, session
from datetime import datetime

from ..services.ml_router import get_ml_router
from ..services.ai_models import get_ai_model_manager
from ..services.agent_service import get_agent_service
from ..utils.decorators import rate_limit, require_auth, validate_json
from ..utils.validators import validate_query_request
from ..utils.exceptions import ValidationError, ServiceError

# Create blueprint
api_bp = Blueprint('api', __name__)

@api_bp.route('/query', methods=['POST'])
@rate_limit("100 per minute")
@validate_json(['query'])
def process_query():
    """
    Process a query through the ML router
    
    Expected JSON:
    {
        "query": "string",
        "model_id": "optional_string",
        "parameters": "optional_dict",
        "session_id": "optional_string"
    }
    """
    try:
        data = request.get_json()
        
        # Validate request
        validation_result = validate_query_request(data)
        if not validation_result.is_valid:
            return jsonify({
                'error': 'Validation failed',
                'details': validation_result.errors
            }), 400
        
        # Get services
        ml_router = get_ml_router()
        if not ml_router:
            return jsonify({'error': 'ML Router not available'}), 503
        
        # Extract parameters
        query = data['query']
        model_id = data.get('model_id')
        parameters = data.get('parameters', {})
        session_id = data.get('session_id', session.get('session_id'))
        
        # Process query
        result = asyncio.run(ml_router.process_query(
            query=query,
            model_id=model_id,
            parameters=parameters,
            session_id=session_id
        ))
        
        return jsonify(result)
        
    except ValidationError as e:
        return jsonify({'error': str(e)}), 400
    except ServiceError as e:
        return jsonify({'error': str(e)}), 503
    except Exception as e:
        current_app.logger.error(f"Query processing error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@api_bp.route('/query/stream', methods=['POST'])
@rate_limit("50 per minute")
@validate_json(['query'])
def stream_query():
    """
    Process a query with streaming response
    
    Returns Server-Sent Events (SSE) stream
    """
    try:
        data = request.get_json()
        
        # Validate request
        validation_result = validate_query_request(data)
        if not validation_result.is_valid:
            return jsonify({
                'error': 'Validation failed',
                'details': validation_result.errors
            }), 400
        
        # Get ML router
        ml_router = get_ml_router()
        if not ml_router:
            return jsonify({'error': 'ML Router not available'}), 503
        
        # Create streaming response
        def generate_stream():
            try:
                query = data['query']
                model_id = data.get('model_id')
                parameters = data.get('parameters', {})
                session_id = data.get('session_id', session.get('session_id'))
                
                # Process query with streaming
                for chunk in ml_router.stream_query(
                    query=query,
                    model_id=model_id,
                    parameters=parameters,
                    session_id=session_id
                ):
                    yield f"data: {chunk}\n\n"
                    
                yield "data: [DONE]\n\n"
                
            except Exception as e:
                current_app.logger.error(f"Streaming error: {e}")
                yield f"data: {{\"error\": \"{str(e)}\"}}\n\n"
        
        return current_app.response_class(
            generate_stream(),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'Access-Control-Allow-Origin': '*'
            }
        )
        
    except Exception as e:
        current_app.logger.error(f"Stream setup error: {e}")
        return jsonify({'error': 'Failed to setup stream'}), 500

@api_bp.route('/classify', methods=['POST'])
@rate_limit("200 per minute")
@validate_json(['query'])
def classify_query():
    """
    Classify a query without processing it
    
    Expected JSON:
    {
        "query": "string"
    }
    """
    try:
        data = request.get_json()
        query = data['query']
        
        ml_router = get_ml_router()
        if not ml_router:
            return jsonify({'error': 'ML Router not available'}), 503
        
        # Classify the query
        classification = asyncio.run(ml_router.classify_query(query))
        
        return jsonify({
            'query': query,
            'classification': classification,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        current_app.logger.error(f"Classification error: {e}")
        return jsonify({'error': 'Classification failed'}), 500

@api_bp.route('/agents', methods=['GET'])
@rate_limit("100 per minute")
def get_agents():
    """Get list of available agents"""
    try:
        agent_service = get_agent_service()
        if not agent_service:
            return jsonify({'error': 'Agent service not available'}), 503
        
        agents = agent_service.get_all_agents()
        
        return jsonify({
            'agents': [agent.to_dict() for agent in agents],
            'total': len(agents),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        current_app.logger.error(f"Agent list error: {e}")
        return jsonify({'error': 'Failed to get agents'}), 500

@api_bp.route('/agents/<agent_id>', methods=['GET'])
@rate_limit("200 per minute")
def get_agent(agent_id):
    """Get specific agent details"""
    try:
        agent_service = get_agent_service()
        if not agent_service:
            return jsonify({'error': 'Agent service not available'}), 503
        
        agent = agent_service.get_agent(agent_id)
        if not agent:
            return jsonify({'error': 'Agent not found'}), 404
        
        return jsonify({
            'agent': agent.to_dict(),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        current_app.logger.error(f"Agent details error: {e}")
        return jsonify({'error': 'Failed to get agent details'}), 500

@api_bp.route('/agents/<agent_id>/health', methods=['GET'])
@rate_limit("50 per minute")
def check_agent_health(agent_id):
    """Check agent health status"""
    try:
        agent_service = get_agent_service()
        if not agent_service:
            return jsonify({'error': 'Agent service not available'}), 503
        
        health_status = asyncio.run(agent_service.check_agent_health(agent_id))
        
        return jsonify({
            'agent_id': agent_id,
            'health': health_status,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        current_app.logger.error(f"Agent health check error: {e}")
        return jsonify({'error': 'Health check failed'}), 500

@api_bp.route('/stats', methods=['GET'])
@rate_limit("50 per minute")
def get_system_stats():
    """Get system statistics"""
    try:
        from ..models import QueryLog, AgentRegistration
        from ..extensions import db
        
        # Get basic statistics
        total_queries = db.session.query(QueryLog).count()
        successful_queries = db.session.query(QueryLog).filter_by(status='success').count()
        total_agents = db.session.query(AgentRegistration).count()
        active_agents = db.session.query(AgentRegistration).filter_by(is_active=True).count()
        
        # Calculate rates
        success_rate = (successful_queries / total_queries * 100) if total_queries > 0 else 0
        
        stats = {
            'queries': {
                'total': total_queries,
                'successful': successful_queries,
                'success_rate': round(success_rate, 2)
            },
            'agents': {
                'total': total_agents,
                'active': active_agents,
                'healthy': active_agents  # TODO: Add actual health check
            },
            'system': {
                'uptime': 'unknown',  # TODO: Track actual uptime
                'version': '1.0.0',
                'environment': current_app.config.get('FLASK_ENV', 'unknown')
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(stats)
        
    except Exception as e:
        current_app.logger.error(f"Stats error: {e}")
        return jsonify({'error': 'Failed to get statistics'}), 500

@api_bp.route('/session', methods=['POST'])
@rate_limit("10 per minute")
def create_session():
    """Create a new session"""
    try:
        import uuid
        
        session_id = str(uuid.uuid4())
        session['session_id'] = session_id
        session['created_at'] = datetime.now().isoformat()
        
        return jsonify({
            'session_id': session_id,
            'created_at': session['created_at'],
            'expires_in': 86400  # 24 hours
        })
        
    except Exception as e:
        current_app.logger.error(f"Session creation error: {e}")
        return jsonify({'error': 'Failed to create session'}), 500

@api_bp.route('/session', methods=['GET'])
@rate_limit("100 per minute")
def get_session():
    """Get current session information"""
    try:
        session_id = session.get('session_id')
        if not session_id:
            return jsonify({'error': 'No active session'}), 404
        
        return jsonify({
            'session_id': session_id,
            'created_at': session.get('created_at'),
            'active': True
        })
        
    except Exception as e:
        current_app.logger.error(f"Session info error: {e}")
        return jsonify({'error': 'Failed to get session info'}), 500

@api_bp.route('/session', methods=['DELETE'])
@rate_limit("10 per minute")
def delete_session():
    """Delete current session"""
    try:
        session_id = session.get('session_id')
        session.clear()
        
        return jsonify({
            'message': 'Session deleted',
            'session_id': session_id,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        current_app.logger.error(f"Session deletion error: {e}")
        return jsonify({'error': 'Failed to delete session'}), 500

# Error handlers for this blueprint
@api_bp.errorhandler(ValidationError)
def handle_validation_error(error):
    """Handle validation errors"""
    return jsonify({
        'error': 'Validation failed',
        'message': str(error),
        'timestamp': datetime.now().isoformat()
    }), 400

@api_bp.errorhandler(ServiceError)
def handle_service_error(error):
    """Handle service errors"""
    return jsonify({
        'error': 'Service error',
        'message': str(error),
        'timestamp': datetime.now().isoformat()
    }), 503

@api_bp.errorhandler(429)
def handle_rate_limit_error(error):
    """Handle rate limit errors"""
    return jsonify({
        'error': 'Rate limit exceeded',
        'message': 'Too many requests. Please try again later.',
        'timestamp': datetime.now().isoformat()
    }), 429
