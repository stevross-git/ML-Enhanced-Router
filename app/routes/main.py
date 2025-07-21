"""
Main Web Routes
Core web interface routes for the ML Query Router
"""

from flask import Blueprint, render_template, request, jsonify, session, redirect, url_for, current_app
from datetime import datetime

from ..services.ml_router import get_ml_router
from ..services.ai_models import get_ai_model_manager
from ..utils.decorators import rate_limit, require_auth

# Create blueprint
main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def index():
    """Main application dashboard"""
    try:
        # Get basic system status
        ml_router = get_ml_router()
        ai_manager = get_ai_model_manager()
        
        context = {
            'title': 'ML Query Router',
            'router_status': 'active' if ml_router else 'inactive',
            'active_models': len(ai_manager.get_active_models()) if ai_manager else 0,
            'current_time': datetime.now().isoformat()
        }
        
        return render_template('dashboard.html', **context)
        
    except Exception as e:
        current_app.logger.error(f"Dashboard error: {e}")
        return render_template('error.html', error="Dashboard temporarily unavailable"), 500

@main_bp.route('/chat')
def chat():
    """Interactive chat interface"""
    try:
        ai_manager = get_ai_model_manager()
        available_models = ai_manager.get_available_models() if ai_manager else []
        
        context = {
            'title': 'Chat Interface',
            'available_models': [model.to_dict() for model in available_models],
            'session_id': session.get('session_id', 'anonymous')
        }
        
        return render_template('chat.html', **context)
        
    except Exception as e:
        current_app.logger.error(f"Chat interface error: {e}")
        return render_template('error.html', error="Chat temporarily unavailable"), 500

@main_bp.route('/settings')
@require_auth(optional=True)
def settings():
    """Application settings page"""
    try:
        context = {
            'title': 'Settings',
            'user_authenticated': 'user_id' in session,
            'cache_enabled': current_app.config.get('CACHE_ENABLED', True),
            'auth_enabled': current_app.config.get('AUTH_ENABLED', False)
        }
        
        return render_template('settings.html', **context)
        
    except Exception as e:
        current_app.logger.error(f"Settings page error: {e}")
        return render_template('error.html', error="Settings temporarily unavailable"), 500

@main_bp.route('/configuration')
@require_auth(roles=['admin'])
def configuration():
    """Advanced configuration page (admin only)"""
    try:
        ml_router = get_ml_router()
        ai_manager = get_ai_model_manager()
        
        context = {
            'title': 'Configuration',
            'router_config': ml_router.get_config() if ml_router else {},
            'model_count': len(ai_manager.get_all_models()) if ai_manager else 0,
            'system_status': 'healthy'
        }
        
        return render_template('configuration.html', **context)
        
    except Exception as e:
        current_app.logger.error(f"Configuration page error: {e}")
        return render_template('error.html', error="Configuration temporarily unavailable"), 500

@main_bp.route('/analytics')
@require_auth(roles=['admin', 'analyst'])
def analytics():
    """Analytics and monitoring dashboard"""
    try:
        from ..models import QueryLog, QueryMetrics
        from ..extensions import db
        
        # Get recent statistics
        recent_queries = db.session.query(QueryLog).order_by(QueryLog.created_at.desc()).limit(10).all()
        
        context = {
            'title': 'Analytics',
            'recent_queries': [q.to_dict() for q in recent_queries],
            'total_queries': db.session.query(QueryLog).count(),
            'active_sessions': len(session.keys()) if session else 0
        }
        
        return render_template('analytics.html', **context)
        
    except Exception as e:
        current_app.logger.error(f"Analytics page error: {e}")
        return render_template('error.html', error="Analytics temporarily unavailable"), 500

@main_bp.route('/docs')
def documentation():
    """API documentation page"""
    try:
        context = {
            'title': 'API Documentation',
            'api_version': '1.0',
            'endpoints_count': 50,  # Approximate count
            'swagger_url': url_for('graphql.swagger_ui')
        }
        
        return render_template('documentation.html', **context)
        
    except Exception as e:
        current_app.logger.error(f"Documentation page error: {e}")
        return render_template('error.html', error="Documentation temporarily unavailable"), 500

@main_bp.route('/health')
@rate_limit("100 per minute")
def health_check():
    """Health check endpoint for load balancers"""
    try:
        # Check critical components
        ml_router = get_ml_router()
        ai_manager = get_ai_model_manager()
        
        # Database connectivity check
        from ..extensions import db
        db.session.execute('SELECT 1')
        
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'version': '1.0.0',
            'components': {
                'ml_router': 'healthy' if ml_router else 'unhealthy',
                'ai_manager': 'healthy' if ai_manager else 'unhealthy',
                'database': 'healthy',
                'cache': 'healthy'  # TODO: Add actual cache check
            }
        }
        
        # Determine overall status
        component_statuses = list(health_status['components'].values())
        if 'unhealthy' in component_statuses:
            health_status['status'] = 'degraded'
            
        return jsonify(health_status)
        
    except Exception as e:
        current_app.logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'unhealthy',
            'timestamp': datetime.now().isoformat(),
            'error': str(e)
        }), 503

@main_bp.route('/status')
@rate_limit("50 per minute")
def system_status():
    """Detailed system status for monitoring"""
    try:
        from ..models import QueryLog, AgentRegistration
        from ..extensions import db
        
        # Get system statistics
        total_queries = db.session.query(QueryLog).count()
        recent_queries = db.session.query(QueryLog).filter(
            QueryLog.created_at >= datetime.now().replace(hour=0, minute=0, second=0)
        ).count()
        
        active_agents = db.session.query(AgentRegistration).filter_by(is_active=True).count()
        
        status_info = {
            'system': {
                'status': 'operational',
                'uptime': 'unknown',  # TODO: Track actual uptime
                'version': '1.0.0'
            },
            'statistics': {
                'total_queries': total_queries,
                'queries_today': recent_queries,
                'active_agents': active_agents,
                'cache_hit_rate': 0.0  # TODO: Get from cache service
            },
            'resources': {
                'memory_usage': 'unknown',  # TODO: Add resource monitoring
                'cpu_usage': 'unknown',
                'disk_usage': 'unknown'
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(status_info)
        
    except Exception as e:
        current_app.logger.error(f"Status check failed: {e}")
        return jsonify({'error': 'Status check failed'}), 500

# Error handlers for this blueprint
@main_bp.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return render_template('errors/404.html'), 404

@main_bp.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return render_template('errors/500.html'), 500

@main_bp.errorhandler(403)
def forbidden(error):
    """Handle 403 errors"""
    return render_template('errors/403.html'), 403
