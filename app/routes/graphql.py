"""
GraphQL Routes
GraphQL endpoint for unified API access
"""

from flask import Blueprint, request, jsonify, current_app
from datetime import datetime
import json

from ..services.ml_router import get_ml_router
from ..services.ai_models import get_ai_model_manager
from ..services.agent_service import get_agent_service
from ..utils.decorators import rate_limit, require_auth
from ..utils.exceptions import ValidationError, ServiceError

# Create blueprint
graphql_bp = Blueprint('graphql', __name__)

@graphql_bp.route('/', methods=['POST', 'GET'])
@rate_limit("200 per minute")
def graphql_endpoint():
    """Main GraphQL endpoint"""
    if request.method == 'GET':
        return graphql_playground()
    
    try:
        # Parse request data
        if request.content_type and 'application/json' in request.content_type:
            data = request.get_json()
        else:
            data = request.form.to_dict()
        
        if not data:
            return jsonify({'error': 'No query provided'}), 400
        
        # Extract query and variables
        query = data.get('query', '')
        variables = data.get('variables', {})
        operation_name = data.get('operationName')
        
        if not query:
            return jsonify({'error': 'No query provided'}), 400
        
        # Simple query parsing and execution
        result = execute_graphql_query(query, variables, operation_name)
        
        return jsonify(result)
        
    except Exception as e:
        current_app.logger.error(f"GraphQL error: {e}")
        return jsonify({
            'errors': [{'message': str(e)}]
        }), 500

def graphql_playground():
    """GraphQL Playground interface"""
    playground_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>GraphQL Playground</title>
        <link rel="stylesheet" href="//cdn.jsdelivr.net/npm/graphql-playground-react/build/static/css/index.css" />
        <link rel="shortcut icon" href="//cdn.jsdelivr.net/npm/graphql-playground-react/build/favicon.png" />
        <script src="//cdn.jsdelivr.net/npm/graphql-playground-react/build/static/js/middleware.js"></script>
    </head>
    <body>
        <div id="root">
            <style>
                body { background-color: rgb(23, 42, 58); font-family: Open Sans, sans-serif; height: 90vh; }
                #root { height: 100%; width: 100%; display: flex; align-items: center; justify-content: center; }
                .loading { font-size: 32px; font-weight: 200; color: rgba(255, 255, 255, .6); margin-left: 20px; }
                img { width: 78px; height: 78px; }
                .title { font-weight: 400; }
            </style>
            <img src="//cdn.jsdelivr.net/npm/graphql-playground-react/build/logo.png" alt="">
            <div class="loading"> Loading
                <span class="title">GraphQL Playground</span>
            </div>
        </div>
        <script>window.addEventListener('load', function (event) {
            GraphQLPlayground.init(document.getElementById('root'), {
                endpoint: '/graphql'
            })
        })</script>
    </body>
    </html>
    """
    return playground_html

def execute_graphql_query(query: str, variables: dict, operation_name: str = None):
    """Execute a GraphQL query with simple parsing"""
    try:
        # Simple query parsing for basic operations
        query_lower = query.lower().strip()
        
        # Handle introspection queries
        if '__schema' in query_lower or '__type' in query_lower:
            return handle_introspection_query(query)
        
        # Handle different query types
        if query_lower.startswith('query'):
            return handle_query(query, variables)
        elif query_lower.startswith('mutation'):
            return handle_mutation(query, variables)
        else:
            return {'errors': [{'message': 'Invalid query type'}]}
        
    except Exception as e:
        current_app.logger.error(f"GraphQL execution error: {e}")
        return {'errors': [{'message': str(e)}]}

def handle_introspection_query(query: str):
    """Handle GraphQL introspection queries"""
    # Simplified schema for introspection
    schema = {
        'data': {
            '__schema': {
                'types': [
                    {
                        'name': 'Query',
                        'kind': 'OBJECT',
                        'fields': [
                            {'name': 'models', 'type': {'name': '[Model]'}},
                            {'name': 'agents', 'type': {'name': '[Agent]'}},
                            {'name': 'stats', 'type': {'name': 'Stats'}},
                            {'name': 'health', 'type': {'name': 'Health'}}
                        ]
                    },
                    {
                        'name': 'Mutation',
                        'kind': 'OBJECT', 
                        'fields': [
                            {'name': 'processQuery', 'type': {'name': 'QueryResult'}},
                            {'name': 'activateModel', 'type': {'name': 'Boolean'}},
                            {'name': 'registerAgent', 'type': {'name': 'Boolean'}}
                        ]
                    },
                    {
                        'name': 'Model',
                        'kind': 'OBJECT',
                        'fields': [
                            {'name': 'id', 'type': {'name': 'String'}},
                            {'name': 'name', 'type': {'name': 'String'}},
                            {'name': 'provider', 'type': {'name': 'String'}},
                            {'name': 'isActive', 'type': {'name': 'Boolean'}}
                        ]
                    },
                    {
                        'name': 'Agent',
                        'kind': 'OBJECT',
                        'fields': [
                            {'name': 'id', 'type': {'name': 'String'}},
                            {'name': 'name', 'type': {'name': 'String'}},
                            {'name': 'categories', 'type': {'name': '[String]'}},
                            {'name': 'isHealthy', 'type': {'name': 'Boolean'}}
                        ]
                    }
                ]
            }
        }
    }
    
    return schema

def handle_query(query: str, variables: dict):
    """Handle GraphQL queries"""
    try:
        data = {}
        
        # Parse query for different fields
        if 'models' in query:
            data['models'] = get_models_data()
        
        if 'agents' in query:
            data['agents'] = get_agents_data()
        
        if 'stats' in query:
            data['stats'] = get_stats_data()
        
        if 'health' in query:
            data['health'] = get_health_data()
        
        return {'data': data}
        
    except Exception as e:
        return {'errors': [{'message': str(e)}]}

def handle_mutation(query: str, variables: dict):
    """Handle GraphQL mutations"""
    try:
        data = {}
        
        # Parse mutation for different operations
        if 'processQuery' in query:
            query_text = variables.get('query', '')
            if not query_text:
                return {'errors': [{'message': 'Query text is required'}]}
            
            result = process_query_mutation(query_text, variables)
            data['processQuery'] = result
        
        if 'activateModel' in query:
            model_id = variables.get('modelId', '')
            if not model_id:
                return {'errors': [{'message': 'Model ID is required'}]}
            
            result = activate_model_mutation(model_id)
            data['activateModel'] = result
        
        if 'registerAgent' in query:
            agent_data = variables.get('agent', {})
            if not agent_data:
                return {'errors': [{'message': 'Agent data is required'}]}
            
            result = register_agent_mutation(agent_data)
            data['registerAgent'] = result
        
        return {'data': data}
        
    except Exception as e:
        return {'errors': [{'message': str(e)}]}

def get_models_data():
    """Get models data for GraphQL"""
    try:
        ai_manager = get_ai_model_manager()
        if not ai_manager:
            return []
        
        models = ai_manager.get_all_models()
        return [
            {
                'id': model.id,
                'name': model.name,
                'provider': model.provider.value if hasattr(model.provider, 'value') else str(model.provider),
                'isActive': model.is_active,
                'categories': getattr(model, 'categories', []),
                'endpoint': getattr(model, 'endpoint', ''),
                'createdAt': getattr(model, 'created_at', datetime.utcnow()).isoformat()
            }
            for model in models
        ]
        
    except Exception as e:
        current_app.logger.error(f"Error getting models data: {e}")
        return []

def get_agents_data():
    """Get agents data for GraphQL"""
    try:
        agent_service = get_agent_service()
        if not agent_service:
            return []
        
        agents = agent_service.get_all_agents()
        return [
            {
                'id': agent['id'],
                'name': agent['name'],
                'description': agent.get('description', ''),
                'categories': agent.get('categories', []),
                'isHealthy': agent.get('is_healthy', False),
                'isActive': agent.get('is_active', False),
                'endpoint': agent.get('endpoint', ''),
                'lastSeen': agent.get('last_seen', datetime.utcnow()).isoformat() if agent.get('last_seen') else None
            }
            for agent in agents
        ]
        
    except Exception as e:
        current_app.logger.error(f"Error getting agents data: {e}")
        return []

def get_stats_data():
    """Get statistics data for GraphQL"""
    try:
        ml_router = get_ml_router()
        if not ml_router:
            return {
                'totalQueries': 0,
                'successfulQueries': 0,
                'failedQueries': 0,
                'avgResponseTime': 0.0,
                'cacheHitRate': 0.0,
                'activeAgents': 0
            }
        
        stats = ml_router.get_statistics()
        return {
            'totalQueries': stats.get('total_queries', 0),
            'successfulQueries': stats.get('successful_routes', 0),
            'failedQueries': stats.get('failed_routes', 0),
            'avgResponseTime': stats.get('avg_response_time', 0.0),
            'cacheHitRate': stats.get('cache_hit_rate', 0.0),
            'activeAgents': stats.get('active_agents', 0),
            'categoryDistribution': stats.get('category_distribution', {})
        }
        
    except Exception as e:
        current_app.logger.error(f"Error getting stats data: {e}")
        return {}

def get_health_data():
    """Get health data for GraphQL"""
    try:
        ml_router = get_ml_router()
        ai_manager = get_ai_model_manager()
        
        return {
            'status': 'healthy' if ml_router and ai_manager else 'unhealthy',
            'timestamp': datetime.utcnow().isoformat(),
            'services': {
                'mlRouter': bool(ml_router),
                'aiManager': bool(ai_manager),
                'database': True,  # Assume healthy if we can execute this
                'cache': True     # Assume healthy if we can execute this
            },
            'uptime': 0.0  # Would need to track application start time
        }
        
    except Exception as e:
        current_app.logger.error(f"Error getting health data: {e}")
        return {
            'status': 'error',
            'timestamp': datetime.utcnow().isoformat(),
            'services': {},
            'uptime': 0.0
        }

def process_query_mutation(query_text: str, variables: dict):
    """Process query mutation"""
    try:
        ml_router = get_ml_router()
        if not ml_router:
            return {
                'success': False,
                'error': 'ML Router not available',
                'response': None
            }
        
        # Import asyncio here to avoid issues
        import asyncio
        
        # Process the query
        result = asyncio.run(ml_router.process_query(
            query=query_text,
            model_id=variables.get('modelId'),
            parameters=variables.get('parameters', {}),
            session_id=variables.get('sessionId')
        ))
        
        return {
            'success': result.get('status') == 'success',
            'response': result.get('response'),
            'error': result.get('error_message'),
            'agentId': result.get('agent_id'),
            'agentName': result.get('agent_name'),
            'responseTime': result.get('response_time', 0.0),
            'tokensUsed': result.get('tokens_used', 0),
            'cost': result.get('cost', 0.0),
            'cacheHit': result.get('cache_hit', False)
        }
        
    except Exception as e:
        current_app.logger.error(f"Error processing query mutation: {e}")
        return {
            'success': False,
            'error': str(e),
            'response': None
        }

def activate_model_mutation(model_id: str):
    """Activate model mutation"""
    try:
        ai_manager = get_ai_model_manager()
        if not ai_manager:
            return False
        
        return ai_manager.activate_model(model_id)
        
    except Exception as e:
        current_app.logger.error(f"Error activating model: {e}")
        return False

def register_agent_mutation(agent_data: dict):
    """Register agent mutation"""
    try:
        ml_router = get_ml_router()
        if not ml_router:
            return False
        
        # Import asyncio here
        import asyncio
        
        return asyncio.run(ml_router.register_agent(agent_data))
        
    except Exception as e:
        current_app.logger.error(f"Error registering agent: {e}")
        return False

@graphql_bp.route('/schema', methods=['GET'])
@rate_limit("100 per minute")
def get_schema():
    """Get GraphQL schema definition"""
    schema_sdl = """
    type Query {
        models: [Model!]!
        agents: [Agent!]!
        stats: Stats!
        health: Health!
    }
    
    type Mutation {
        processQuery(query: String!, modelId: String, parameters: JSON, sessionId: String): QueryResult!
        activateModel(modelId: String!): Boolean!
        registerAgent(agent: AgentInput!): Boolean!
    }
    
    type Model {
        id: String!
        name: String!
        provider: String!
        isActive: Boolean!
        categories: [String!]!
        endpoint: String
        createdAt: String!
    }
    
    type Agent {
        id: String!
        name: String!
        description: String
        categories: [String!]!
        isHealthy: Boolean!
        isActive: Boolean!
        endpoint: String
        lastSeen: String
    }
    
    type Stats {
        totalQueries: Int!
        successfulQueries: Int!
        failedQueries: Int!
        avgResponseTime: Float!
        cacheHitRate: Float!
        activeAgents: Int!
        categoryDistribution: JSON
    }
    
    type Health {
        status: String!
        timestamp: String!
        services: JSON!
        uptime: Float!
    }
    
    type QueryResult {
        success: Boolean!
        response: String
        error: String
        agentId: String
        agentName: String
        responseTime: Float!
        tokensUsed: Int!
        cost: Float!
        cacheHit: Boolean!
    }
    
    input AgentInput {
        id: String!
        name: String!
        description: String
        endpoint: String!
        categories: [String!]!
        capabilities: JSON
    }
    
    scalar JSON
    """
    
    return jsonify({
        'schema': schema_sdl,
        'timestamp': datetime.utcnow().isoformat()
    })

@graphql_bp.route('/health', methods=['GET'])
@rate_limit("200 per minute")
def graphql_health():
    """GraphQL endpoint health check"""
    try:
        return jsonify({
            'status': 'healthy',
            'endpoint': 'graphql',
            'timestamp': datetime.utcnow().isoformat(),
            'features': {
                'queries': True,
                'mutations': True,
                'introspection': True,
                'playground': True
            }
        })
        
    except Exception as e:
        current_app.logger.error(f"GraphQL health check error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500