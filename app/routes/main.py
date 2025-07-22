"""
Main Web Routes - UPDATED VERSION
Core web interface routes for the ML Query Router with advanced features
"""

from flask import Blueprint, render_template, request, jsonify, session, redirect, url_for, current_app
from datetime import datetime

from ..services.ml_router import get_ml_router
from ..services.ai_models import get_ai_model_manager
from ..utils.decorators import rate_limit, require_auth

# Create blueprint
main_bp = Blueprint('main', __name__)

# ===============================
# EXISTING CORE ROUTES
# ===============================

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
            recent_queries = QueryLog.query.order_by(QueryLog.timestamp.desc()).limit(10).all()
            total_queries = QueryLog.query.count()
        except Exception as e:
            current_app.logger.warning(f"Could not fetch query logs: {e}")
        
        context = {
            'title': 'Analytics',
            'recent_queries': [q.to_dict() if hasattr(q, 'to_dict') else {'query': str(q)} for q in recent_queries],
            'total_queries': total_queries,
            'system_uptime': '99.9%',  # Placeholder
            'average_response_time': '150ms'  # Placeholder
        }
        
        return render_template('analytics.html', **context)
        
    except Exception as e:
        current_app.logger.error(f"Analytics page error: {e}")
        context = {
            'title': 'Analytics',
            'recent_queries': [],
            'total_queries': 0,
            'system_uptime': 'Unknown',
            'average_response_time': 'Unknown',
            'error_message': 'Analytics temporarily unavailable'
        }
        return render_template('analytics.html', **context)

# ===============================
# ADDITIONAL MISSING ROUTES FROM APP.PY
# ===============================

@main_bp.route('/dashboard')
def dashboard():
    """Dashboard showing routing statistics"""
    return redirect(url_for('main.index'))  # Redirect to main dashboard

@main_bp.route('/agents')
def agents():
    """Agent management page"""
    try:
        return render_template('agents.html', title='Agent Management')
    except Exception as e:
        current_app.logger.error(f"Error rendering agents page: {e}")
        return render_template('error.html', error="Failed to load agents interface"), 500

@main_bp.route('/models')
def models():
    """Model management page"""
    try:
        return render_template('models.html', title='Model Management')
    except Exception as e:
        current_app.logger.error(f"Error rendering models page: {e}")
        return render_template('error.html', error="Failed to load models interface"), 500

@main_bp.route('/ai-models')
def ai_models():
    """AI model management page"""
    try:
        return render_template('ai_models.html', title='AI Model Management')
    except Exception as e:
        current_app.logger.error(f"Error rendering AI models page: {e}")
        return render_template('error.html', error="Failed to load AI models interface"), 500

@main_bp.route('/auth')
def auth():
    """Authentication management page"""
    try:
        return render_template('auth.html', title='Authentication')
    except Exception as e:
        current_app.logger.error(f"Error rendering auth page: {e}")
        return render_template('error.html', error="Failed to load authentication interface"), 500

@main_bp.route('/config')
def config():
    """Configuration page for advanced settings"""
    try:
        return render_template('config.html', title='Configuration')
    except Exception as e:
        current_app.logger.error(f"Error rendering config page: {e}")
        return render_template('error.html', error="Failed to load configuration interface"), 500

# ===============================
# NEW ADVANCED FEATURE ROUTES
# ===============================

@main_bp.route('/peer-teaching')
@rate_limit("50 per minute")
def peer_teaching_page():
    """Peer Teaching & Collaborative Agents interface"""
    try:
        return render_template('peer_teaching.html', title='Peer Teaching & Collaborative Agents')
    except Exception as e:
        current_app.logger.error(f"Error rendering peer teaching page: {e}")
        return render_template('error.html', error="Failed to load peer teaching interface"), 500

@main_bp.route('/peer-teaching/demo')
@rate_limit("20 per minute")
def peer_teaching_demo():
    """Peer teaching learning demonstration"""
    try:
        return render_template('peer_teaching_demo.html', title='Peer Teaching Demo')
    except Exception as e:
        current_app.logger.error(f"Error rendering peer teaching demo: {e}")
        return render_template('error.html', error="Failed to load demo interface"), 500

@main_bp.route('/personal-ai')
@rate_limit("50 per minute")
def personal_ai():
    """Personal AI interface with hybrid edge-cloud routing"""
    try:
        return render_template('personal_ai.html', title='Personal AI')
    except Exception as e:
        current_app.logger.error(f"Error rendering personal AI page: {e}")
        return render_template('error.html', error="Failed to load personal AI interface"), 500

@main_bp.route('/email-intelligence')
@rate_limit("50 per minute")
def email_intelligence_page():
    """Enhanced email intelligence page with Office 365 integration"""
    try:
        return render_template('email_intelligence_office365.html', title='Email Intelligence')
    except Exception as e:
        current_app.logger.error(f"Error rendering email intelligence page: {e}")
        # Fallback to basic email intelligence template
        try:
            return render_template('email_intelligence.html', title='Email Intelligence')
        except:
            return render_template('error.html', error="Failed to load email intelligence interface"), 500

@main_bp.route('/evaluation')
@rate_limit("50 per minute")
def evaluation_page():
    """Automated Evaluation Engine interface"""
    try:
        return render_template('evaluation.html', title='Automated Evaluation')
    except Exception as e:
        current_app.logger.error(f"Error rendering evaluation page: {e}")
        return render_template('error.html', error="Failed to load evaluation interface"), 500

@main_bp.route('/multimodal')
@rate_limit("50 per minute")
def multimodal_page():
    """Multimodal AI processing interface"""
    try:
        return render_template('multimodal.html', title='Multimodal AI')
    except Exception as e:
        current_app.logger.error(f"Error rendering multimodal page: {e}")
        return render_template('error.html', error="Failed to load multimodal interface"), 500

@main_bp.route('/chains')
@rate_limit("50 per minute")
def chains_page():
    """Auto Chain Generator interface"""
    try:
        return render_template('chains.html', title='AI Chain Generator')
    except Exception as e:
        current_app.logger.error(f"Error rendering chains page: {e}")
        return render_template('error.html', error="Failed to load chains interface"), 500

@main_bp.route('/shared-memory')
@rate_limit("50 per minute")
def shared_memory_page():
    """Shared Memory Management interface"""
    try:
        return render_template('shared_memory.html', title='Shared Memory')
    except Exception as e:
        current_app.logger.error(f"Error rendering shared memory page: {e}")
        return render_template('error.html', error="Failed to load shared memory interface"), 500

@main_bp.route('/playground')
@rate_limit("50 per minute")
def playground():
    """AI Model Playground interface"""
    try:
        return render_template('playground.html', title='AI Playground')
    except Exception as e:
        current_app.logger.error(f"Error rendering playground: {e}")
        return render_template('error.html', error="Failed to load playground interface"), 500

@main_bp.route('/monitoring')
@rate_limit("30 per minute")
def monitoring_dashboard():
    """System Monitoring Dashboard"""
    try:
        return render_template('monitoring.html', title='System Monitoring')
    except Exception as e:
        current_app.logger.error(f"Error rendering monitoring dashboard: {e}")
        return render_template('error.html', error="Failed to load monitoring interface"), 500

@main_bp.route('/docs')
@rate_limit("100 per minute")
def documentation():
    """API Documentation interface"""
    try:
        return render_template('docs/index.html', title='Documentation')
    except Exception as e:
        current_app.logger.error(f"Error rendering documentation: {e}")
        return render_template('error.html', error="Failed to load documentation"), 500

@main_bp.route('/docs/api')
@rate_limit("100 per minute")
def api_documentation():
    """API Reference Documentation"""
    try:
        return render_template('docs/api.html', title='API Documentation')
    except Exception as e:
        current_app.logger.error(f"Error rendering API docs: {e}")
        return render_template('error.html', error="Failed to load API documentation"), 500

@main_bp.route('/help')
@rate_limit("100 per minute")
def help_center():
    """Help Center and User Guides"""
    try:
        return render_template('help/index.html', title='Help Center')
    except Exception as e:
        current_app.logger.error(f"Error rendering help center: {e}")
        return render_template('error.html', error="Failed to load help center"), 500

@main_bp.route('/help/getting-started')
@rate_limit("100 per minute")
def getting_started():
    """Getting Started Guide"""
    try:
        return render_template('help/getting_started.html', title='Getting Started')
    except Exception as e:
        current_app.logger.error(f"Error rendering getting started: {e}")
        return render_template('error.html', error="Failed to load getting started guide"), 500

@main_bp.route('/help/troubleshooting')
@rate_limit("100 per minute")
def troubleshooting():
    """Troubleshooting Guide"""
    try:
        return render_template('help/troubleshooting.html', title='Troubleshooting')
    except Exception as e:
        current_app.logger.error(f"Error rendering troubleshooting: {e}")
        return render_template('error.html', error="Failed to load troubleshooting guide"), 500

@main_bp.route('/admin/system')
@rate_limit("20 per minute")
@require_auth(roles=['admin'])
def system_admin():
    """System Administration interface"""
    try:
        return render_template('admin/system.html', title='System Administration')
    except Exception as e:
        current_app.logger.error(f"Error rendering system admin page: {e}")
        return render_template('error.html', error="Failed to load admin interface"), 500
    
@main_bp.route('/favicon.ico')
def favicon():
    """Serve favicon"""
    from flask import send_from_directory
    import os
    
    try:
        return send_from_directory(
            os.path.join(current_app.root_path, 'static'),
            'favicon.ico',
            mimetype='image/vnd.microsoft.icon'
        )
    except Exception:
        # Return a simple response if favicon doesn't exist
        return '', 404