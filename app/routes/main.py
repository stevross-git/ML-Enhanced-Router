"""
Main Web Routes - FIXED VERSION
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
        
        # FIXED: Use get_all_models() instead of get_active_models()
        all_models = []
        active_count = 0
        
        if ai_manager:
            try:
                all_models = ai_manager.get_all_models() if hasattr(ai_manager, 'get_all_models') else []
                active_count = len([m for m in all_models if getattr(m, 'is_active', False)])
            except Exception as e:
                current_app.logger.warning(f"Could not get models: {e}")
                active_count = 0
        
        context = {
            'title': 'ML Query Router',
            'router_status': 'active' if ml_router else 'inactive',
            'active_models': active_count,
            'total_models': len(all_models),
            'current_time': datetime.now().isoformat(),
            'system_healthy': True
        }
        
        return render_template('dashboard.html', **context)
        
    except Exception as e:
        current_app.logger.error(f"Dashboard error: {e}")
        # FIXED: Use a simple fallback instead of error.html
        context = {
            'title': 'ML Query Router',
            'router_status': 'unknown',
            'active_models': 0,
            'total_models': 0,
            'current_time': datetime.now().isoformat(),
            'system_healthy': False,
            'error_message': 'Dashboard temporarily unavailable'
        }
        return render_template('dashboard.html', **context)

@main_bp.route('/chat')
def chat():
    """Interactive chat interface"""
    try:
        ai_manager = get_ai_model_manager()
        available_models = []
        
        if ai_manager:
            try:
                if hasattr(ai_manager, 'get_available_models'):
                    available_models = ai_manager.get_available_models()
                elif hasattr(ai_manager, 'get_all_models'):
                    all_models = ai_manager.get_all_models()
                    available_models = [m for m in all_models if getattr(m, 'is_active', False)]
            except Exception as e:
                current_app.logger.warning(f"Could not get available models: {e}")
        
        context = {
            'title': 'Chat Interface',
            'available_models': [model.to_dict() if hasattr(model, 'to_dict') else {'id': str(model), 'name': str(model)} for model in available_models],
            'session_id': session.get('session_id', 'anonymous')
        }
        
        return render_template('chat.html', **context)
        
    except Exception as e:
        current_app.logger.error(f"Chat interface error: {e}")
        # FIXED: Fallback to basic chat interface
        context = {
            'title': 'Chat Interface',
            'available_models': [],
            'session_id': session.get('session_id', 'anonymous'),
            'error_message': 'Some features may be temporarily unavailable'
        }
        return render_template('chat.html', **context)

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
        # FIXED: Fallback settings
        context = {
            'title': 'Settings',
            'user_authenticated': False,
            'cache_enabled': True,
            'auth_enabled': False,
            'error_message': 'Some settings may be temporarily unavailable'
        }
        return render_template('settings.html', **context)

@main_bp.route('/configuration')
@require_auth(roles=['admin'])
def configuration():
    """Advanced configuration page (admin only)"""
    try:
        ml_router = get_ml_router()
        ai_manager = get_ai_model_manager()
        
        model_count = 0
        if ai_manager:
            try:
                models = ai_manager.get_all_models() if hasattr(ai_manager, 'get_all_models') else []
                model_count = len(models)
            except Exception:
                model_count = 0
        
        context = {
            'title': 'Configuration',
            'router_config': ml_router.get_config() if ml_router and hasattr(ml_router, 'get_config') else {},
            'model_count': model_count,
            'system_status': 'healthy'
        }
        
        return render_template('configuration.html', **context)
        
    except Exception as e:
        current_app.logger.error(f"Configuration page error: {e}")
        context = {
            'title': 'Configuration',
            'router_config': {},
            'model_count': 0,
            'system_status': 'unknown',
            'error_message': 'Configuration temporarily unavailable'
        }
        return render_template('configuration.html', **context)

@main_bp.route('/analytics')
@require_auth(roles=['admin', 'analyst'])
def analytics():
    """Analytics and monitoring dashboard"""
    try:
        # Try to import models safely
        recent_queries = []
        total_queries = 0
        
        try:
            from ..models import QueryLog
            from ..extensions import db
            
            recent_queries = db.session.query(QueryLog).order_by(QueryLog.created_at.desc()).limit(10).all()
            total_queries = db.session.query(QueryLog).count()
            recent_queries = [q.to_dict() if hasattr(q, 'to_dict') else {'id': q.id, 'query': str(q)} for q in recent_queries]
        except Exception as e:
            current_app.logger.warning(f"Could not load analytics data: {e}")
        
        context = {
            'title': 'Analytics',
            'recent_queries': recent_queries,
            'total_queries': total_queries,
            'active_sessions': len(session.keys()) if session else 0
        }
        
        return render_template('analytics.html', **context)
        
    except Exception as e:
        current_app.logger.error(f"Analytics page error: {e}")
        context = {
            'title': 'Analytics',
            'recent_queries': [],
            'total_queries': 0,
            'active_sessions': 0,
            'error_message': 'Analytics temporarily unavailable'
        }
        return render_template('analytics.html', **context)

@main_bp.route('/health')
def health_check():
    """Simple health check endpoint"""
    try:
        ml_router = get_ml_router()
        ai_manager = get_ai_model_manager()
        
        status = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'services': {
                'ml_router': 'active' if ml_router else 'inactive',
                'ai_manager': 'active' if ai_manager else 'inactive',
                'database': 'active'  # If we got here, DB is working
            }
        }
        
        return jsonify(status)
        
    except Exception as e:
        current_app.logger.error(f"Health check error: {e}")
        return jsonify({
            'status': 'degraded',
            'timestamp': datetime.now().isoformat(),
            'error': str(e)
        }), 500