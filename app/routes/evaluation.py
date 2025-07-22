"""
Automated Evaluation Engine Routes
Quality assessment and performance evaluation for AI models and agents
"""

import uuid
from datetime import datetime
from flask import Blueprint, request, jsonify, current_app

from ..utils.decorators import rate_limit, validate_json
from ..utils.exceptions import ValidationError, ServiceError

# Create blueprint
evaluation_bp = Blueprint('evaluation', __name__)

def get_evaluation_engine():
    """Get evaluation engine instance"""
    try:
        from automated_evaluation_engine import get_evaluation_engine
        return get_evaluation_engine()
    except ImportError:
        current_app.logger.warning("Evaluation engine not available")
        return None

@evaluation_bp.route('/run', methods=['POST'])
@rate_limit("10 per minute")
@validate_json(['evaluation_type'])
def run_evaluation():
    """Run automated evaluation"""
    try:
        evaluation_engine = get_evaluation_engine()
        if not evaluation_engine:
            return jsonify({"error": "Evaluation engine not available"}), 503
        
        data = request.get_json()
        evaluation_type = data['evaluation_type']
        target_id = data.get('target_id')
        parameters = data.get('parameters', {})
        
        # Validate evaluation type
        valid_types = ['model_performance', 'agent_quality', 'response_accuracy', 'system_health']
        if evaluation_type not in valid_types:
            return jsonify({"error": f"Invalid evaluation type. Must be one of: {valid_types}"}), 400
        
        # Run evaluation
        evaluation_id = str(uuid.uuid4())
        result = evaluation_engine.run_evaluation(
            evaluation_id=evaluation_id,
            evaluation_type=evaluation_type,
            target_id=target_id,
            parameters=parameters
        )
        
        return jsonify({
            "status": "success",
            "evaluation_id": evaluation_id,
            "evaluation_type": evaluation_type,
            "result": {
                "overall_score": result.overall_score,
                "detailed_metrics": result.detailed_metrics,
                "recommendations": result.recommendations,
                "execution_time": result.execution_time,
                "timestamp": result.timestamp.isoformat()
            },
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        current_app.logger.error(f"Error running evaluation: {e}")
        return jsonify({"error": str(e)}), 500

@evaluation_bp.route('/results/<evaluation_id>', methods=['GET'])
@rate_limit("100 per minute")
def get_evaluation_result(evaluation_id):
    """Get specific evaluation result"""
    try:
        evaluation_engine = get_evaluation_engine()
        if not evaluation_engine:
            return jsonify({"error": "Evaluation engine not available"}), 503
        
        result = evaluation_engine.get_evaluation_result(evaluation_id)
        
        if not result:
            return jsonify({"error": "Evaluation result not found"}), 404
        
        return jsonify({
            "evaluation_id": evaluation_id,
            "result": {
                "evaluation_type": result.evaluation_type,
                "target_id": result.target_id,
                "overall_score": result.overall_score,
                "detailed_metrics": result.detailed_metrics,
                "recommendations": result.recommendations,
                "execution_time": result.execution_time,
                "status": result.status,
                "created_at": result.created_at.isoformat(),
                "completed_at": result.completed_at.isoformat() if result.completed_at else None
            },
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        current_app.logger.error(f"Error getting evaluation result: {e}")
        return jsonify({"error": str(e)}), 500

@evaluation_bp.route('/history', methods=['GET'])
@rate_limit("50 per minute")
def get_evaluation_history():
    """Get evaluation history from database"""
    try:
        evaluation_engine = get_evaluation_engine()
        if not evaluation_engine:
            return jsonify({"error": "Evaluation engine not available"}), 503
        
        limit = request.args.get('limit', 10, type=int)
        evaluation_type = request.args.get('type')
        target_id = request.args.get('target_id')
        
        history = evaluation_engine.get_evaluation_history(
            limit=limit,
            evaluation_type=evaluation_type,
            target_id=target_id
        )
        
        return jsonify({
            "evaluation_history": history,
            "count": len(history),
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        current_app.logger.error(f"Error getting evaluation history: {e}")
        return jsonify({"error": str(e)}), 500

@evaluation_bp.route('/stats', methods=['GET'])
@rate_limit("100 per minute")
def get_evaluation_stats():
    """Get evaluation statistics"""
    try:
        evaluation_engine = get_evaluation_engine()
        if not evaluation_engine:
            return jsonify({"error": "Evaluation engine not available"}), 503
        
        stats = evaluation_engine.get_evaluation_stats()
        
        return jsonify({
            "evaluation_stats": stats,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        current_app.logger.error(f"Error getting evaluation stats: {e}")
        return jsonify({"error": str(e)}), 500

@evaluation_bp.route('/benchmark', methods=['POST'])
@rate_limit("5 per minute")
@validate_json(['benchmark_type'])
def run_benchmark():
    """Run comprehensive benchmark evaluation"""
    try:
        evaluation_engine = get_evaluation_engine()
        if not evaluation_engine:
            return jsonify({"error": "Evaluation engine not available"}), 503
        
        data = request.get_json()
        benchmark_type = data['benchmark_type']
        targets = data.get('targets', [])
        parameters = data.get('parameters', {})
        
        # Validate benchmark type
        valid_benchmarks = ['performance_comparison', 'accuracy_test', 'stress_test', 'comprehensive']
        if benchmark_type not in valid_benchmarks:
            return jsonify({"error": f"Invalid benchmark type. Must be one of: {valid_benchmarks}"}), 400
        
        # Run benchmark
        benchmark_id = str(uuid.uuid4())
        result = evaluation_engine.run_benchmark(
            benchmark_id=benchmark_id,
            benchmark_type=benchmark_type,
            targets=targets,
            parameters=parameters
        )
        
        return jsonify({
            "status": "success",
            "benchmark_id": benchmark_id,
            "benchmark_type": benchmark_type,
            "result": {
                "summary": result.summary,
                "individual_results": result.individual_results,
                "comparative_analysis": result.comparative_analysis,
                "recommendations": result.recommendations,
                "execution_time": result.execution_time
            },
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        current_app.logger.error(f"Error running benchmark: {e}")
        return jsonify({"error": str(e)}), 500

@evaluation_bp.route('/metrics', methods=['GET'])
@rate_limit("100 per minute")
def get_available_metrics():
    """Get available evaluation metrics"""
    try:
        evaluation_engine = get_evaluation_engine()
        if not evaluation_engine:
            return jsonify({"error": "Evaluation engine not available"}), 503
        
        metrics = evaluation_engine.get_available_metrics()
        
        return jsonify({
            "available_metrics": metrics,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        current_app.logger.error(f"Error getting available metrics: {e}")
        return jsonify({"error": str(e)}), 500

@evaluation_bp.route('/schedule', methods=['POST'])
@rate_limit("20 per minute")
@validate_json(['evaluation_type', 'schedule'])
def schedule_evaluation():
    """Schedule recurring evaluation"""
    try:
        evaluation_engine = get_evaluation_engine()
        if not evaluation_engine:
            return jsonify({"error": "Evaluation engine not available"}), 503
        
        data = request.get_json()
        evaluation_type = data['evaluation_type']
        schedule = data['schedule']  # e.g., "daily", "weekly", "monthly"
        target_id = data.get('target_id')
        parameters = data.get('parameters', {})
        
        # Validate schedule
        valid_schedules = ['hourly', 'daily', 'weekly', 'monthly']
        if schedule not in valid_schedules:
            return jsonify({"error": f"Invalid schedule. Must be one of: {valid_schedules}"}), 400
        
        # Schedule evaluation
        schedule_id = str(uuid.uuid4())
        success = evaluation_engine.schedule_evaluation(
            schedule_id=schedule_id,
            evaluation_type=evaluation_type,
            schedule=schedule,
            target_id=target_id,
            parameters=parameters
        )
        
        if success:
            return jsonify({
                "status": "success",
                "message": "Evaluation scheduled successfully",
                "schedule_id": schedule_id,
                "evaluation_type": evaluation_type,
                "schedule": schedule,
                "timestamp": datetime.now().isoformat()
            })
        else:
            return jsonify({"error": "Failed to schedule evaluation"}), 500
        
    except Exception as e:
        current_app.logger.error(f"Error scheduling evaluation: {e}")
        return jsonify({"error": str(e)}), 500

@evaluation_bp.route('/schedules', methods=['GET'])
@rate_limit("50 per minute")
def get_scheduled_evaluations():
    """Get list of scheduled evaluations"""
    try:
        evaluation_engine = get_evaluation_engine()
        if not evaluation_engine:
            return jsonify({"error": "Evaluation engine not available"}), 503
        
        schedules = evaluation_engine.get_scheduled_evaluations()
        
        return jsonify({
            "scheduled_evaluations": schedules,
            "count": len(schedules),
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        current_app.logger.error(f"Error getting scheduled evaluations: {e}")
        return jsonify({"error": str(e)}), 500

@evaluation_bp.route('/schedules/<schedule_id>', methods=['DELETE'])
@rate_limit("20 per minute")
def cancel_scheduled_evaluation(schedule_id):
    """Cancel a scheduled evaluation"""
    try:
        evaluation_engine = get_evaluation_engine()
        if not evaluation_engine:
            return jsonify({"error": "Evaluation engine not available"}), 503
        
        success = evaluation_engine.cancel_scheduled_evaluation(schedule_id)
        
        if success:
            return jsonify({
                "status": "success",
                "message": "Scheduled evaluation cancelled successfully",
                "schedule_id": schedule_id,
                "timestamp": datetime.now().isoformat()
            })
        else:
            return jsonify({"error": "Schedule not found or already cancelled"}), 404
        
    except Exception as e:
        current_app.logger.error(f"Error cancelling scheduled evaluation: {e}")
        return jsonify({"error": str(e)}), 500

@evaluation_bp.route('/compare', methods=['POST'])
@rate_limit("10 per minute")
@validate_json(['targets'])
def compare_targets():
    """Compare multiple targets using evaluation metrics"""
    try:
        evaluation_engine = get_evaluation_engine()
        if not evaluation_engine:
            return jsonify({"error": "Evaluation engine not available"}), 503
        
        data = request.get_json()
        targets = data['targets']
        metrics = data.get('metrics', ['performance', 'accuracy', 'efficiency'])
        parameters = data.get('parameters', {})
        
        if len(targets) < 2:
            return jsonify({"error": "At least 2 targets required for comparison"}), 400
        
        # Run comparison
        comparison_id = str(uuid.uuid4())
        result = evaluation_engine.compare_targets(
            comparison_id=comparison_id,
            targets=targets,
            metrics=metrics,
            parameters=parameters
        )
        
        return jsonify({
            "status": "success",
            "comparison_id": comparison_id,
            "result": {
                "summary": result.summary,
                "detailed_comparison": result.detailed_comparison,
                "rankings": result.rankings,
                "recommendations": result.recommendations,
                "statistical_significance": result.statistical_significance
            },
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        current_app.logger.error(f"Error comparing targets: {e}")
        return jsonify({"error": str(e)}), 500

@evaluation_bp.route('/export/<evaluation_id>', methods=['GET'])
@rate_limit("20 per minute")
def export_evaluation_report(evaluation_id):
    """Export detailed evaluation report"""
    try:
        evaluation_engine = get_evaluation_engine()
        if not evaluation_engine:
            return jsonify({"error": "Evaluation engine not available"}), 503
        
        format_type = request.args.get('format', 'json')  # json, csv, pdf
        include_details = request.args.get('details', 'true').lower() == 'true'
        
        # Generate export
        report = evaluation_engine.export_evaluation_report(
            evaluation_id=evaluation_id,
            format_type=format_type,
            include_details=include_details
        )
        
        if not report:
            return jsonify({"error": "Evaluation not found or export failed"}), 404
        
        return jsonify({
            "status": "success",
            "evaluation_id": evaluation_id,
            "format": format_type,
            "report": report,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        current_app.logger.error(f"Error exporting evaluation report: {e}")
        return jsonify({"error": str(e)}), 500