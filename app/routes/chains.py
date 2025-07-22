"""
Auto Chain Generator Routes
Advanced AI orchestration and agent chains
"""

import asyncio
import uuid
from datetime import datetime
from flask import Blueprint, request, jsonify, current_app

from ..utils.decorators import rate_limit, validate_json
from ..utils.exceptions import ValidationError, ServiceError

# Create blueprint
chains_bp = Blueprint('chains', __name__)

def get_auto_chain_generator():
    """Get auto chain generator instance"""
    try:
        from auto_chain_generator import AutoChainGenerator
        # This would be injected via service container in production
        return AutoChainGenerator()
    except ImportError:
        current_app.logger.warning("Auto Chain Generator not available")
        return None

@chains_bp.route('/analyze', methods=['POST'])
@rate_limit("30 per minute")
@validate_json(['query'])
def analyze_query_for_chain():
    """Analyze query to determine optimal chain composition"""
    try:
        auto_chain_generator = get_auto_chain_generator()
        if not auto_chain_generator:
            return jsonify({"error": "Auto Chain Generator not available"}), 503
        
        data = request.get_json()
        query = data['query']
        
        analysis = auto_chain_generator.analyze_query_for_chain(query)
        
        return jsonify({
            "analysis": analysis,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        current_app.logger.error(f"Error analyzing query for chain: {e}")
        return jsonify({"error": str(e)}), 500

@chains_bp.route('/generate', methods=['POST'])
@rate_limit("20 per minute")
@validate_json(['query'])
def generate_agent_chain():
    """Generate optimal agent chain for a query"""
    try:
        auto_chain_generator = get_auto_chain_generator()
        if not auto_chain_generator:
            return jsonify({"error": "Auto Chain Generator not available"}), 503
        
        data = request.get_json()
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
        current_app.logger.error(f"Error generating agent chain: {e}")
        return jsonify({"error": str(e)}), 500

@chains_bp.route('/execute', methods=['POST'])
@rate_limit("10 per minute")
@validate_json(['query'])
def execute_agent_chain():
    """Execute a complete agent chain"""
    try:
        auto_chain_generator = get_auto_chain_generator()
        if not auto_chain_generator:
            return jsonify({"error": "Auto Chain Generator not available"}), 503
        
        data = request.get_json()
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
        current_app.logger.error(f"Error executing agent chain: {e}")
        return jsonify({"error": str(e)}), 500

@chains_bp.route('/templates', methods=['GET'])
@rate_limit("50 per minute")
def get_chain_templates():
    """Get available chain templates"""
    try:
        auto_chain_generator = get_auto_chain_generator()
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
        current_app.logger.error(f"Error getting chain templates: {e}")
        return jsonify({"error": str(e)}), 500

@chains_bp.route('/stats', methods=['GET'])
@rate_limit("100 per minute")
def get_chain_stats():
    """Get Auto Chain Generator statistics"""
    try:
        auto_chain_generator = get_auto_chain_generator()
        if not auto_chain_generator:
            return jsonify({"error": "Auto Chain Generator not available"}), 503
        
        stats = auto_chain_generator.get_chain_stats()
        
        return jsonify({
            "stats": stats,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        current_app.logger.error(f"Error getting chain stats: {e}")
        return jsonify({"error": str(e)}), 500

@chains_bp.route('/<chain_id>/status', methods=['GET'])
@rate_limit("100 per minute")
def get_chain_status(chain_id):
    """Get status of a specific chain execution"""
    try:
        auto_chain_generator = get_auto_chain_generator()
        if not auto_chain_generator:
            return jsonify({"error": "Auto Chain Generator not available"}), 503
        
        # Get chain status (this would be implemented in the auto_chain_generator)
        status = auto_chain_generator.get_chain_status(chain_id)
        
        if not status:
            return jsonify({"error": "Chain not found"}), 404
        
        return jsonify({
            "chain_id": chain_id,
            "status": status,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        current_app.logger.error(f"Error getting chain status: {e}")
        return jsonify({"error": str(e)}), 500

@chains_bp.route('/<chain_id>/cancel', methods=['POST'])
@rate_limit("20 per minute")
def cancel_chain_execution(chain_id):
    """Cancel a running chain execution"""
    try:
        auto_chain_generator = get_auto_chain_generator()
        if not auto_chain_generator:
            return jsonify({"error": "Auto Chain Generator not available"}), 503
        
        success = auto_chain_generator.cancel_chain(chain_id)
        
        if not success:
            return jsonify({"error": "Chain not found or already completed"}), 404
        
        return jsonify({
            "message": "Chain execution cancelled successfully",
            "chain_id": chain_id,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        current_app.logger.error(f"Error cancelling chain: {e}")
        return jsonify({"error": str(e)}), 500

@chains_bp.route('/optimize', methods=['POST'])
@rate_limit("10 per minute")
@validate_json(['chain_data'])
def optimize_chain():
    """Optimize an existing chain for better performance"""
    try:
        auto_chain_generator = get_auto_chain_generator()
        if not auto_chain_generator:
            return jsonify({"error": "Auto Chain Generator not available"}), 503
        
        data = request.get_json()
        chain_data = data['chain_data']
        optimization_goals = data.get('optimization_goals', ['cost', 'speed'])
        
        optimized_chain = auto_chain_generator.optimize_chain(chain_data, optimization_goals)
        
        # Convert to dict for JSON serialization
        optimized_dict = {
            "chain_id": optimized_chain.chain_id,
            "query": optimized_chain.query,
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
                for step in optimized_chain.steps
            ],
            "estimated_cost": optimized_chain.estimated_cost,
            "estimated_time": optimized_chain.estimated_time,
            "complexity_score": optimized_chain.complexity_score,
            "optimization_improvements": optimized_chain.optimization_improvements,
            "created_at": optimized_chain.created_at.isoformat()
        }
        
        return jsonify({
            "optimized_chain": optimized_dict,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        current_app.logger.error(f"Error optimizing chain: {e}")
        return jsonify({"error": str(e)}), 500
