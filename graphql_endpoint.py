"""
GraphQL Agent Mesh API Endpoint
Flask integration for GraphQL schema
"""

from flask import Blueprint, request, jsonify
from graphql import graphql_sync, build_schema
from graphql_schema import schema
import json

# Create GraphQL blueprint
graphql_bp = Blueprint('graphql', __name__)

@graphql_bp.route('/graphql', methods=['POST', 'GET'])
def graphql_endpoint():
    """Main GraphQL endpoint"""
    if request.method == 'GET':
        # Return GraphiQL interface for GET requests
        return graphql_playground()
    
    try:
        # Parse request data
        if request.content_type and 'application/json' in request.content_type:
            data = request.get_json()
        else:
            data = request.form
        
        if not data:
            return jsonify({'error': 'No query provided'}), 400
        
        # Extract query and variables
        query = data.get('query', '')
        variables = data.get('variables', {})
        operation_name = data.get('operationName')
        
        if not query:
            return jsonify({'error': 'No query provided'}), 400
        
        # Execute GraphQL query
        result = graphql_sync(
            schema,
            query,
            variable_values=variables,
            operation_name=operation_name
        )
        
        # Format response
        response = {}
        if result.data:
            response['data'] = result.data
        if result.errors:
            response['errors'] = [str(error) for error in result.errors]
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@graphql_bp.route('/graphql/playground', methods=['GET'])
def graphql_playground():
    """GraphQL Playground interface"""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>GraphQL Agent Mesh API</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body { margin: 0; padding: 0; font-family: Arial, sans-serif; }
            #graphiql { height: 100vh; }
            .header { background: #1f2937; color: white; padding: 1rem; }
            .header h1 { margin: 0; }
            .examples { background: #f3f4f6; padding: 1rem; border-bottom: 1px solid #e5e7eb; }
            .example { margin: 0.5rem 0; padding: 0.5rem; background: white; border: 1px solid #d1d5db; border-radius: 4px; }
            .example h4 { margin: 0 0 0.5rem 0; color: #374151; }
            .example pre { margin: 0; font-size: 0.875rem; background: #1f2937; color: #f9fafb; padding: 0.5rem; border-radius: 4px; overflow-x: auto; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🪢 GraphQL Agent Mesh API</h1>
            <p>Unified GraphQL endpoint for intelligent AI agent routing and execution</p>
        </div>
        
        <div class="examples">
            <div class="example">
                <h4>🔍 Query Classification</h4>
                <pre>query {
  classify(query: "Explain quantum computing") {
    primaryCategory
    confidence
    complexity
    intent
    suggestedAgents
    requiredCapabilities
  }
}</pre>
            </div>
            
            <div class="example">
                <h4>🎯 Agent Routing</h4>
                <pre>query {
  route(query: "Write a Python function", strategy: "ml_optimized") {
    agentId
    agentName
    confidence
    costEstimate
    estimatedResponseTime
    reasoning
  }
}</pre>
            </div>
            
            <div class="example">
                <h4>🤖 Agent Execution</h4>
                <pre>mutation {
  executeAgent(
    agentId: "gpt-4o"
    query: "Explain machine learning"
    temperature: 0.7
    maxTokens: 1000
  ) {
    response
    executionTime
    tokensUsed
    cost
    success
  }
}</pre>
            </div>
            
            <div class="example">
                <h4>🌐 Collaborative AI</h4>
                <pre>query {
  routeCollaborative(
    query: "Analyze market trends and create a business strategy"
    maxAgents: 3
  ) {
    sessionId
    primaryResponse
    synthesis
    agentsUsed
    totalCost
  }
}</pre>
            </div>
            
            <div class="example">
                <h4>📊 System Metrics</h4>
                <pre>query {
  metrics {
    totalQueries
    activeAgents
    avgResponseTime
    successRate
    cacheHitRate
    totalCost
  }
}</pre>
            </div>
        </div>
        
        <div id="graphiql">
            <p style="text-align: center; margin-top: 2rem;">
                <a href="/graphql" style="color: #3b82f6; text-decoration: none; font-weight: bold;">
                    → Open GraphiQL Interface
                </a>
            </p>
        </div>
    </body>
    </html>
    '''

@graphql_bp.route('/graphql/schema', methods=['GET'])
def graphql_schema():
    """Get GraphQL schema definition"""
    try:
        from graphql import build_schema, get_schema_from_ast, print_schema
        
        # Get schema SDL
        schema_sdl = print_schema(schema)
        
        return jsonify({
            'schema': schema_sdl,
            'types': len(schema.type_map),
            'queries': len(schema.query_type.fields) if schema.query_type else 0,
            'mutations': len(schema.mutation_type.fields) if schema.mutation_type else 0
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@graphql_bp.route('/graphql/examples', methods=['GET'])
def graphql_examples():
    """Get example GraphQL queries"""
    examples = {
        'classification': {
            'description': 'Classify a query and get routing suggestions',
            'query': '''query {
  classify(query: "Explain quantum computing") {
    primaryCategory
    confidence
    complexity
    intent
    suggestedAgents
    requiredCapabilities
  }
}'''
        },
        'routing': {
            'description': 'Route a query to the best agent',
            'query': '''query {
  route(query: "Write a Python function", strategy: "ml_optimized") {
    agentId
    agentName
    confidence
    costEstimate
    estimatedResponseTime
    reasoning
  }
}'''
        },
        'execution': {
            'description': 'Execute a query with a specific agent',
            'query': '''mutation {
  executeAgent(
    agentId: "gpt-4o"
    query: "Explain machine learning"
    temperature: 0.7
    maxTokens: 1000
  ) {
    response
    executionTime
    tokensUsed
    cost
    success
  }
}'''
        },
        'collaborative': {
            'description': 'Use collaborative AI with multiple agents',
            'query': '''query {
  routeCollaborative(
    query: "Analyze market trends and create a business strategy"
    maxAgents: 3
  ) {
    sessionId
    primaryResponse
    synthesis
    agentsUsed
    totalCost
  }
}'''
        },
        'agents': {
            'description': 'List all available agents',
            'query': '''query {
  agents(activeOnly: true) {
    id
    name
    provider
    capabilities {
      name
      confidence
    }
    isActive
    costPer1kTokens
    supportsStreaming
    supportsMultimodal
  }
}'''
        },
        'metrics': {
            'description': 'Get system metrics and health',
            'query': '''query {
  metrics {
    totalQueries
    activeAgents
    avgResponseTime
    successRate
    cacheHitRate
    totalCost
  }
}'''
        },
        'prediction': {
            'description': 'Get predictive analytics',
            'query': '''query {
  predict(predictionType: "load_forecast", timeframe: "1h") {
    predictionType
    value
    confidence
    timeframe
  }
}'''
        }
    }
    
    return jsonify(examples)