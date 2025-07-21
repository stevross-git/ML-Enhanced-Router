"""
Cache Management Routes
API endpoints for managing response caching
"""

from flask import Blueprint, request, jsonify, current_app
from datetime import datetime, timedelta

from ..services.cache_service import get_cache_manager
from ..utils.decorators import rate_limit, require_auth, validate_json
from ..utils.exceptions import ValidationError, ServiceError, CacheError

# Create blueprint
cache_bp = Blueprint('cache', __name__)

@cache_bp.route('/stats', methods=['GET'])
@rate_limit("100 per minute")
def get_cache_stats():
    """Get cache statistics"""
    try:
        cache_manager = get_cache_manager()
        if not cache_manager:
            return jsonify({'error': 'Cache manager not available'}), 503
        
        stats = cache_manager.get_stats()
        
        return jsonify({
            'stats': stats,
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        current_app.logger.error(f"Error getting cache stats: {e}")
        return jsonify({'error': 'Failed to retrieve cache statistics'}), 500

@cache_bp.route('/entries', methods=['GET'])
@rate_limit("50 per minute")
@require_auth(roles=['admin'])
def get_cache_entries():
    """Get cache entries with pagination"""
    try:
        cache_manager = get_cache_manager()
        if not cache_manager:
            return jsonify({'error': 'Cache manager not available'}), 503
        
        # Get query parameters
        page = request.args.get('page', 1, type=int)
        per_page = min(request.args.get('per_page', 20, type=int), 100)
        model_id = request.args.get('model_id')
        status = request.args.get('status')  # valid, expired, all
        
        entries = cache_manager.get_entries(
            page=page,
            per_page=per_page,
            model_id=model_id,
            status=status
        )
        
        return jsonify({
            'entries': entries['entries'],
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total': entries['total'],
                'pages': entries['pages']
            }
        })
        
    except Exception as e:
        current_app.logger.error(f"Error getting cache entries: {e}")
        return jsonify({'error': 'Failed to retrieve cache entries'}), 500

@cache_bp.route('/entries/<entry_id>', methods=['GET'])
@rate_limit("100 per minute")
@require_auth(roles=['admin'])
def get_cache_entry(entry_id):
    """Get specific cache entry"""
    try:
        cache_manager = get_cache_manager()
        if not cache_manager:
            return jsonify({'error': 'Cache manager not available'}), 503
        
        entry = cache_manager.get_entry(entry_id)
        if not entry:
            return jsonify({'error': 'Cache entry not found'}), 404
        
        return jsonify({'entry': entry})
        
    except Exception as e:
        current_app.logger.error(f"Error getting cache entry {entry_id}: {e}")
        return jsonify({'error': 'Failed to retrieve cache entry'}), 500

@cache_bp.route('/entries/<entry_id>', methods=['DELETE'])
@rate_limit("50 per minute")
@require_auth(roles=['admin'])
def delete_cache_entry(entry_id):
    """Delete specific cache entry"""
    try:
        cache_manager = get_cache_manager()
        if not cache_manager:
            return jsonify({'error': 'Cache manager not available'}), 503
        
        success = cache_manager.delete_entry(entry_id)
        if not success:
            return jsonify({'error': 'Cache entry not found'}), 404
        
        return jsonify({'message': 'Cache entry deleted successfully'})
        
    except Exception as e:
        current_app.logger.error(f"Error deleting cache entry {entry_id}: {e}")
        return jsonify({'error': 'Failed to delete cache entry'}), 500

@cache_bp.route('/clear', methods=['POST'])
@rate_limit("10 per minute")
@require_auth(roles=['admin'])
def clear_cache():
    """Clear cache based on filters"""
    try:
        cache_manager = get_cache_manager()
        if not cache_manager:
            return jsonify({'error': 'Cache manager not available'}), 503
        
        data = request.get_json() or {}
        
        # Clear options
        model_id = data.get('model_id')  # Clear cache for specific model
        expired_only = data.get('expired_only', False)  # Clear only expired entries
        older_than_hours = data.get('older_than_hours')  # Clear entries older than X hours
        confirm = data.get('confirm', False)  # Confirmation flag
        
        if not confirm:
            return jsonify({
                'error': 'Confirmation required',
                'message': 'Set "confirm": true to proceed with cache clearing'
            }), 400
        
        # Perform cache clearing
        result = cache_manager.clear_cache(
            model_id=model_id,
            expired_only=expired_only,
            older_than_hours=older_than_hours
        )
        
        return jsonify({
            'message': 'Cache cleared successfully',
            'cleared_count': result['cleared_count'],
            'remaining_count': result['remaining_count']
        })
        
    except Exception as e:
        current_app.logger.error(f"Error clearing cache: {e}")
        return jsonify({'error': 'Failed to clear cache'}), 500

@cache_bp.route('/cleanup', methods=['POST'])
@rate_limit("5 per minute")
@require_auth(roles=['admin'])
def cleanup_cache():
    """Run cache cleanup (remove expired entries)"""
    try:
        cache_manager = get_cache_manager()
        if not cache_manager:
            return jsonify({'error': 'Cache manager not available'}), 503
        
        result = cache_manager.cleanup_expired()
        
        return jsonify({
            'message': 'Cache cleanup completed',
            'removed_count': result['removed_count'],
            'remaining_count': result['remaining_count']
        })
        
    except Exception as e:
        current_app.logger.error(f"Error during cache cleanup: {e}")
        return jsonify({'error': 'Cache cleanup failed'}), 500

@cache_bp.route('/warmup', methods=['POST'])
@rate_limit("5 per minute")
@require_auth(roles=['admin'])
@validate_json(['queries'])
def warmup_cache():
    """Warm up cache with common queries"""
    try:
        cache_manager = get_cache_manager()
        if not cache_manager:
            return jsonify({'error': 'Cache manager not available'}), 503
        
        data = request.get_json()
        queries = data['queries']
        model_id = data.get('model_id')
        
        if not isinstance(queries, list):
            return jsonify({'error': 'Queries must be a list'}), 400
        
        if len(queries) > 50:
            return jsonify({'error': 'Maximum 50 queries allowed for warmup'}), 400
        
        # Start cache warmup
        result = cache_manager.warmup_cache(queries, model_id)
        
        return jsonify({
            'message': 'Cache warmup initiated',
            'queued_count': result['queued_count'],
            'job_id': result.get('job_id')
        })
        
    except ValidationError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        current_app.logger.error(f"Error during cache warmup: {e}")
        return jsonify({'error': 'Cache warmup failed'}), 500

@cache_bp.route('/hit-rate', methods=['GET'])
@rate_limit("100 per minute")
def get_hit_rate():
    """Get cache hit rate statistics"""
    try:
        cache_manager = get_cache_manager()
        if not cache_manager:
            return jsonify({'error': 'Cache manager not available'}), 503
        
        # Get time period from query params
        hours = request.args.get('hours', 24, type=int)
        model_id = request.args.get('model_id')
        
        hit_rate_data = cache_manager.get_hit_rate(hours=hours, model_id=model_id)
        
        return jsonify({
            'hit_rate_data': hit_rate_data,
            'period_hours': hours,
            'model_id': model_id,
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        current_app.logger.error(f"Error getting hit rate: {e}")
        return jsonify({'error': 'Failed to retrieve hit rate data'}), 500

@cache_bp.route('/size', methods=['GET'])
@rate_limit("100 per minute")
def get_cache_size():
    """Get cache size information"""
    try:
        cache_manager = get_cache_manager()
        if not cache_manager:
            return jsonify({'error': 'Cache manager not available'}), 503
        
        size_info = cache_manager.get_size_info()
        
        return jsonify({
            'size_info': size_info,
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        current_app.logger.error(f"Error getting cache size: {e}")
        return jsonify({'error': 'Failed to retrieve cache size information'}), 500

@cache_bp.route('/config', methods=['GET'])
@rate_limit("50 per minute")
@require_auth(roles=['admin'])
def get_cache_config():
    """Get cache configuration"""
    try:
        cache_manager = get_cache_manager()
        if not cache_manager:
            return jsonify({'error': 'Cache manager not available'}), 503
        
        config = cache_manager.get_config()
        
        return jsonify({
            'config': config,
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        current_app.logger.error(f"Error getting cache config: {e}")
        return jsonify({'error': 'Failed to retrieve cache configuration'}), 500

@cache_bp.route('/config', methods=['PUT'])
@rate_limit("20 per minute")
@require_auth(roles=['admin'])
@validate_json()
def update_cache_config():
    """Update cache configuration"""
    try:
        cache_manager = get_cache_manager()
        if not cache_manager:
            return jsonify({'error': 'Cache manager not available'}), 503
        
        data = request.get_json()
        
        # Validate configuration
        valid_keys = [
            'default_ttl', 'max_entries', 'cleanup_interval',
            'hit_rate_window', 'memory_threshold'
        ]
        
        config_updates = {}
        for key, value in data.items():
            if key in valid_keys:
                config_updates[key] = value
            else:
                return jsonify({'error': f'Invalid configuration key: {key}'}), 400
        
        if not config_updates:
            return jsonify({'error': 'No valid configuration updates provided'}), 400
        
        # Update configuration
        success = cache_manager.update_config(config_updates)
        if not success:
            return jsonify({'error': 'Failed to update configuration'}), 500
        
        return jsonify({
            'message': 'Cache configuration updated successfully',
            'updated_keys': list(config_updates.keys())
        })
        
    except ValidationError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        current_app.logger.error(f"Error updating cache config: {e}")
        return jsonify({'error': 'Failed to update cache configuration'}), 500

@cache_bp.route('/models', methods=['GET'])
@rate_limit("100 per minute")
def get_cached_models():
    """Get list of models with cached responses"""
    try:
        cache_manager = get_cache_manager()
        if not cache_manager:
            return jsonify({'error': 'Cache manager not available'}), 503
        
        models = cache_manager.get_cached_models()
        
        return jsonify({
            'models': models,
            'total': len(models),
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        current_app.logger.error(f"Error getting cached models: {e}")
        return jsonify({'error': 'Failed to retrieve cached models'}), 500

@cache_bp.route('/health', methods=['GET'])
@rate_limit("200 per minute")
def cache_health():
    """Get cache health status"""
    try:
        cache_manager = get_cache_manager()
        if not cache_manager:
            return jsonify({
                'status': 'unavailable',
                'message': 'Cache manager not available'
            }), 503
        
        health = cache_manager.get_health()
        
        status_code = 200 if health['status'] == 'healthy' else 503
        
        return jsonify(health), status_code
        
    except Exception as e:
        current_app.logger.error(f"Error getting cache health: {e}")
        return jsonify({
            'status': 'error',
            'message': 'Failed to check cache health'
        }), 500