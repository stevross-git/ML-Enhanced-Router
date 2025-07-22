"""
Shared Memory Routes
Cross-session data sharing and collaborative memory management
"""

from datetime import datetime
from flask import Blueprint, request, jsonify, current_app

from ..utils.decorators import rate_limit, validate_json
from ..utils.exceptions import ValidationError, ServiceError

# Create blueprint
shared_memory_bp = Blueprint('shared_memory', __name__)

def get_shared_memory_manager():
    """Get shared memory manager instance"""
    try:
        from shared_memory import get_shared_memory_manager
        return get_shared_memory_manager()
    except ImportError:
        current_app.logger.warning("Shared memory manager not available")
        return None

@shared_memory_bp.route('/sessions', methods=['POST'])
@rate_limit("50 per minute")
@validate_json(['session_id'])
def create_session():
    """Create a new shared memory session"""
    try:
        shared_memory_manager = get_shared_memory_manager()
        if not shared_memory_manager:
            return jsonify({'error': 'Shared memory manager not initialized'}), 500
        
        data = request.get_json()
        session_id = data['session_id']
        metadata = data.get('metadata', {})
        
        success = shared_memory_manager.create_session(session_id, metadata)
        
        if success:
            return jsonify({
                'status': 'success',
                'session_id': session_id,
                'message': 'Session created successfully',
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({'error': 'Failed to create session'}), 500
        
    except Exception as e:
        current_app.logger.error(f"Error creating session: {e}")
        return jsonify({'error': str(e)}), 500

@shared_memory_bp.route('/sessions/<session_id>/messages', methods=['POST'])
@rate_limit("200 per minute")
@validate_json(['content', 'message_type'])
def add_message():
    """Add a message to shared memory session"""
    try:
        shared_memory_manager = get_shared_memory_manager()
        if not shared_memory_manager:
            return jsonify({'error': 'Shared memory manager not initialized'}), 500
        
        data = request.get_json()
        session_id = request.view_args['session_id']
        content = data['content']
        message_type = data['message_type']
        sender_id = data.get('sender_id', 'anonymous')
        metadata = data.get('metadata', {})
        
        # Validate message type
        from shared_memory import MessageType
        try:
            msg_type = MessageType(message_type)
        except ValueError:
            valid_types = [mt.value for mt in MessageType]
            return jsonify({
                'error': f'Invalid message type. Must be one of: {valid_types}'
            }), 400
        
        message_id = shared_memory_manager.add_message(
            session_id=session_id,
            content=content,
            message_type=msg_type,
            sender_id=sender_id,
            metadata=metadata
        )
        
        if message_id:
            return jsonify({
                'status': 'success',
                'message_id': message_id,
                'session_id': session_id,
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({'error': 'Failed to add message'}), 500
        
    except Exception as e:
        current_app.logger.error(f"Error adding message: {e}")
        return jsonify({'error': str(e)}), 500

@shared_memory_bp.route('/sessions/<session_id>/messages', methods=['GET'])
@rate_limit("100 per minute")
def get_session_messages(session_id):
    """Get messages from a specific session"""
    try:
        shared_memory_manager = get_shared_memory_manager()
        if not shared_memory_manager:
            return jsonify({'error': 'Shared memory manager not initialized'}), 500
        
        # Get query parameters
        limit = int(request.args.get('limit', 50))
        message_types = request.args.getlist('types')
        since_timestamp = request.args.get('since', type=float)
        
        # Convert string types to MessageType enum
        from shared_memory import MessageType
        if message_types:
            try:
                message_types = [MessageType(t) for t in message_types]
            except ValueError:
                return jsonify({'error': 'Invalid message type'}), 400
        else:
            message_types = None
        
        messages = shared_memory_manager.get_session_messages(
            session_id=session_id,
            message_types=message_types,
            since_timestamp=since_timestamp
        )
        
        # Limit results
        messages = messages[-limit:]
        
        return jsonify({
            'session_id': session_id,
            'messages': [msg.to_dict() for msg in messages],
            'count': len(messages),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        current_app.logger.error(f"Error getting session messages: {e}")
        return jsonify({'error': str(e)}), 500

@shared_memory_bp.route('/sessions/<session_id>/context', methods=['GET'])
@rate_limit("100 per minute")
def get_session_context(session_id):
    """Get shared context for a session"""
    try:
        shared_memory_manager = get_shared_memory_manager()
        if not shared_memory_manager:
            return jsonify({'error': 'Shared memory manager not initialized'}), 500
        
        context = shared_memory_manager.get_shared_context(session_id)
        
        return jsonify({
            'session_id': session_id,
            'context': context,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        current_app.logger.error(f"Error getting session context: {e}")
        return jsonify({'error': str(e)}), 500

@shared_memory_bp.route('/sessions/<session_id>/context', methods=['PUT'])
@rate_limit("50 per minute")
@validate_json(['context'])
def update_session_context(session_id):
    """Update shared context for a session"""
    try:
        shared_memory_manager = get_shared_memory_manager()
        if not shared_memory_manager:
            return jsonify({'error': 'Shared memory manager not initialized'}), 500
        
        data = request.get_json()
        context = data['context']
        merge = data.get('merge', True)
        
        success = shared_memory_manager.update_shared_context(
            session_id=session_id,
            context=context,
            merge=merge
        )
        
        if success:
            return jsonify({
                'status': 'success',
                'session_id': session_id,
                'message': 'Context updated successfully',
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({'error': 'Failed to update context'}), 500
        
    except Exception as e:
        current_app.logger.error(f"Error updating session context: {e}")
        return jsonify({'error': str(e)}), 500

@shared_memory_bp.route('/sessions', methods=['GET'])
@rate_limit("50 per minute")
def get_active_sessions():
    """Get list of active sessions"""
    try:
        shared_memory_manager = get_shared_memory_manager()
        if not shared_memory_manager:
            return jsonify({'error': 'Shared memory manager not initialized'}), 500
        
        limit = request.args.get('limit', 20, type=int)
        
        sessions = shared_memory_manager.get_active_sessions(limit)
        
        return jsonify({
            'active_sessions': sessions,
            'count': len(sessions),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        current_app.logger.error(f"Error getting active sessions: {e}")
        return jsonify({'error': str(e)}), 500

@shared_memory_bp.route('/sessions/<session_id>', methods=['DELETE'])
@rate_limit("20 per minute")
def delete_session(session_id):
    """Delete a shared memory session"""
    try:
        shared_memory_manager = get_shared_memory_manager()
        if not shared_memory_manager:
            return jsonify({'error': 'Shared memory manager not initialized'}), 500
        
        success = shared_memory_manager.delete_session(session_id)
        
        if success:
            return jsonify({
                'status': 'success',
                'session_id': session_id,
                'message': 'Session deleted successfully',
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({'error': 'Session not found'}), 404
        
    except Exception as e:
        current_app.logger.error(f"Error deleting session: {e}")
        return jsonify({'error': str(e)}), 500

@shared_memory_bp.route('/sessions/<session_id>/participants', methods=['POST'])
@rate_limit("50 per minute")
@validate_json(['participant_id'])
def add_participant(session_id):
    """Add a participant to a session"""
    try:
        shared_memory_manager = get_shared_memory_manager()
        if not shared_memory_manager:
            return jsonify({'error': 'Shared memory manager not initialized'}), 500
        
        data = request.get_json()
        participant_id = data['participant_id']
        role = data.get('role', 'participant')
        permissions = data.get('permissions', ['read', 'write'])
        
        success = shared_memory_manager.add_participant(
            session_id=session_id,
            participant_id=participant_id,
            role=role,
            permissions=permissions
        )
        
        if success:
            return jsonify({
                'status': 'success',
                'session_id': session_id,
                'participant_id': participant_id,
                'role': role,
                'permissions': permissions,
                'message': 'Participant added successfully',
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({'error': 'Failed to add participant'}), 500
        
    except Exception as e:
        current_app.logger.error(f"Error adding participant: {e}")
        return jsonify({'error': str(e)}), 500

@shared_memory_bp.route('/sessions/<session_id>/participants', methods=['GET'])
@rate_limit("100 per minute")
def get_session_participants(session_id):
    """Get participants of a session"""
    try:
        shared_memory_manager = get_shared_memory_manager()
        if not shared_memory_manager:
            return jsonify({'error': 'Shared memory manager not initialized'}), 500
        
        participants = shared_memory_manager.get_session_participants(session_id)
        
        return jsonify({
            'session_id': session_id,
            'participants': participants,
            'count': len(participants),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        current_app.logger.error(f"Error getting session participants: {e}")
        return jsonify({'error': str(e)}), 500

@shared_memory_bp.route('/sessions/<session_id>/participants/<participant_id>', methods=['DELETE'])
@rate_limit("30 per minute")
def remove_participant(session_id, participant_id):
    """Remove a participant from a session"""
    try:
        shared_memory_manager = get_shared_memory_manager()
        if not shared_memory_manager:
            return jsonify({'error': 'Shared memory manager not initialized'}), 500
        
        success = shared_memory_manager.remove_participant(session_id, participant_id)
        
        if success:
            return jsonify({
                'status': 'success',
                'session_id': session_id,
                'participant_id': participant_id,
                'message': 'Participant removed successfully',
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({'error': 'Participant not found'}), 404
        
    except Exception as e:
        current_app.logger.error(f"Error removing participant: {e}")
        return jsonify({'error': str(e)}), 500

@shared_memory_bp.route('/sessions/<session_id>/search', methods=['POST'])
@rate_limit("50 per minute")
@validate_json(['query'])
def search_session_messages(session_id):
    """Search messages within a session"""
    try:
        shared_memory_manager = get_shared_memory_manager()
        if not shared_memory_manager:
            return jsonify({'error': 'Shared memory manager not initialized'}), 500
        
        data = request.get_json()
        query = data['query']
        limit = data.get('limit', 20)
        message_types = data.get('message_types')
        
        # Convert string types to MessageType enum
        from shared_memory import MessageType
        if message_types:
            try:
                message_types = [MessageType(t) for t in message_types]
            except ValueError:
                return jsonify({'error': 'Invalid message type'}), 400
        
        results = shared_memory_manager.search_messages(
            session_id=session_id,
            query=query,
            limit=limit,
            message_types=message_types
        )
        
        return jsonify({
            'session_id': session_id,
            'query': query,
            'results': [result.to_dict() for result in results],
            'count': len(results),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        current_app.logger.error(f"Error searching session messages: {e}")
        return jsonify({'error': str(e)}), 500

@shared_memory_bp.route('/sessions/<session_id>/export', methods=['GET'])
@rate_limit("10 per minute")
def export_session(session_id):
    """Export session data"""
    try:
        shared_memory_manager = get_shared_memory_manager()
        if not shared_memory_manager:
            return jsonify({'error': 'Shared memory manager not initialized'}), 500
        
        format_type = request.args.get('format', 'json')  # json, csv, txt
        include_metadata = request.args.get('metadata', 'true').lower() == 'true'
        
        export_data = shared_memory_manager.export_session(
            session_id=session_id,
            format_type=format_type,
            include_metadata=include_metadata
        )
        
        if export_data:
            return jsonify({
                'status': 'success',
                'session_id': session_id,
                'format': format_type,
                'data': export_data,
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({'error': 'Session not found or export failed'}), 404
        
    except Exception as e:
        current_app.logger.error(f"Error exporting session: {e}")
        return jsonify({'error': str(e)}), 500

@shared_memory_bp.route('/sessions/<session_id>/summary', methods=['GET'])
@rate_limit("50 per minute")
def get_session_summary(session_id):
    """Get session summary and statistics"""
    try:
        shared_memory_manager = get_shared_memory_manager()
        if not shared_memory_manager:
            return jsonify({'error': 'Shared memory manager not initialized'}), 500
        
        summary = shared_memory_manager.get_session_summary(session_id)
        
        if summary:
            return jsonify({
                'session_id': session_id,
                'summary': summary,
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({'error': 'Session not found'}), 404
        
    except Exception as e:
        current_app.logger.error(f"Error getting session summary: {e}")
        return jsonify({'error': str(e)}), 500

@shared_memory_bp.route('/sessions/<session_id>/clone', methods=['POST'])
@rate_limit("10 per minute")
@validate_json(['new_session_id'])
def clone_session(session_id):
    """Clone an existing session"""
    try:
        shared_memory_manager = get_shared_memory_manager()
        if not shared_memory_manager:
            return jsonify({'error': 'Shared memory manager not initialized'}), 500
        
        data = request.get_json()
        new_session_id = data['new_session_id']
        include_messages = data.get('include_messages', True)
        include_context = data.get('include_context', True)
        
        success = shared_memory_manager.clone_session(
            source_session_id=session_id,
            new_session_id=new_session_id,
            include_messages=include_messages,
            include_context=include_context
        )
        
        if success:
            return jsonify({
                'status': 'success',
                'source_session_id': session_id,
                'new_session_id': new_session_id,
                'message': 'Session cloned successfully',
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({'error': 'Failed to clone session'}), 500
        
    except Exception as e:
        current_app.logger.error(f"Error cloning session: {e}")
        return jsonify({'error': str(e)}), 500

@shared_memory_bp.route('/stats', methods=['GET'])
@rate_limit("50 per minute")
def get_shared_memory_stats():
    """Get shared memory system statistics"""
    try:
        shared_memory_manager = get_shared_memory_manager()
        if not shared_memory_manager:
            return jsonify({'error': 'Shared memory manager not initialized'}), 500
        
        stats = shared_memory_manager.get_stats()
        
        return jsonify({
            'shared_memory_stats': stats,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        current_app.logger.error(f"Error getting shared memory stats: {e}")
        return jsonify({'error': str(e)}), 500

@shared_memory_bp.route('/cleanup', methods=['POST'])
@rate_limit("5 per minute")
def cleanup_expired_sessions():
    """Clean up expired sessions and messages"""
    try:
        shared_memory_manager = get_shared_memory_manager()
        if not shared_memory_manager:
            return jsonify({'error': 'Shared memory manager not initialized'}), 500
        
        max_age_hours = request.json.get('max_age_hours', 24) if request.json else 24
        
        cleaned_count = shared_memory_manager.cleanup_expired_sessions(max_age_hours)
        
        return jsonify({
            'status': 'success',
            'cleaned_sessions': cleaned_count,
            'max_age_hours': max_age_hours,
            'message': f'Cleaned up {cleaned_count} expired sessions',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        current_app.logger.error(f"Error cleaning up sessions: {e}")
        return jsonify({'error': str(e)}), 500