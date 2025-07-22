"""
Streaming Routes
Server-Sent Events (SSE) for real-time updates
"""

import json
import time
import uuid
from flask import Blueprint, Response, request, session, current_app

from ..utils.decorators import rate_limit
from ..utils.exceptions import ValidationError

# Create blueprint
streaming_bp = Blueprint('streaming', __name__)

# Global variable to track active SSE connections
active_sse_connections = {}

@streaming_bp.route('/api/stream')
@rate_limit("10 per minute")
def stream_updates():
    """
    Server-Sent Events endpoint for real-time updates
    """
    session_id = request.args.get('session_id', session.get('session_id'))
    
    if not session_id:
        return Response("Session ID required", status=400)
    
    def event_stream():
        """Generator function for SSE events"""
        try:
            # Import here to avoid circular imports
            from ..services import get_shared_memory_manager
            shared_memory_manager = get_shared_memory_manager()
            
            if not shared_memory_manager:
                yield f"data: {json.dumps({'error': 'Shared memory not initialized'})}\n\n"
                return
            
            # Send initial session data
            try:
                context = shared_memory_manager.get_shared_context(session_id)
                yield f"data: {json.dumps({'type': 'initial', 'context': context})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
                return
            
            # Store this connection for updates
            connection_id = str(uuid.uuid4())
            if session_id not in active_sse_connections:
                active_sse_connections[session_id] = {}
            active_sse_connections[session_id][connection_id] = True
            
            try:
                last_message_count = 0
                
                while active_sse_connections.get(session_id, {}).get(connection_id, False):
                    try:
                        # Check for new messages
                        messages = shared_memory_manager.get_session_messages(session_id)
                        current_count = len(messages)
                        
                        if current_count > last_message_count:
                            # Send new messages
                            new_messages = messages[last_message_count:]
                            for message in new_messages:
                                message_data = {
                                    'type': 'message',
                                    'message': message.to_dict()
                                }
                                yield f"data: {json.dumps(message_data)}\n\n"
                            
                            last_message_count = current_count
                        
                        # Send heartbeat every 10 seconds
                        yield f"data: {json.dumps({'type': 'heartbeat', 'timestamp': time.time()})}\n\n"
                        
                        time.sleep(2)  # Check for updates every 2 seconds
                        
                    except Exception as e:
                        current_app.logger.error(f"Error in SSE stream: {e}")
                        yield f"data: {json.dumps({'error': str(e)})}\n\n"
                        break
                        
            finally:
                # Clean up connection
                if session_id in active_sse_connections and connection_id in active_sse_connections[session_id]:
                    del active_sse_connections[session_id][connection_id]
                    if not active_sse_connections[session_id]:
                        del active_sse_connections[session_id]
        
        except Exception as e:
            current_app.logger.error(f"Stream setup error: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return Response(
        event_stream(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'Access-Control-Allow-Origin': '*'
        }
    )

@streaming_bp.route('/api/shared-memory/sessions/<session_id>/stream')
@rate_limit("10 per minute")
def stream_session_updates(session_id):
    """Server-Sent Events endpoint for specific session updates"""
    
    def event_stream():
        """Generator function for SSE events"""
        try:
            from ..services import get_shared_memory_manager
            shared_memory_manager = get_shared_memory_manager()
            
            if not shared_memory_manager:
                yield f"data: {json.dumps({'error': 'Shared memory not initialized'})}\n\n"
                return
            
            # Send initial session data
            try:
                context = shared_memory_manager.get_shared_context(session_id)
                yield f"data: {json.dumps({'type': 'initial', 'context': context})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
                return
            
            # Store this connection for updates
            connection_id = str(uuid.uuid4())
            if session_id not in active_sse_connections:
                active_sse_connections[session_id] = {}
            active_sse_connections[session_id][connection_id] = True
            
            try:
                last_message_count = 0
                
                while active_sse_connections.get(session_id, {}).get(connection_id, False):
                    try:
                        # Check for new messages
                        messages = shared_memory_manager.get_session_messages(session_id)
                        current_count = len(messages)
                        
                        if current_count > last_message_count:
                            # Send new messages
                            new_messages = messages[last_message_count:]
                            for message in new_messages:
                                message_data = {
                                    'type': 'message',
                                    'message': message.to_dict()
                                }
                                yield f"data: {json.dumps(message_data)}\n\n"
                            
                            last_message_count = current_count
                        
                        # Send heartbeat every 10 seconds
                        yield f"data: {json.dumps({'type': 'heartbeat', 'timestamp': time.time()})}\n\n"
                        
                        time.sleep(2)  # Check for updates every 2 seconds
                        
                    except Exception as e:
                        current_app.logger.error(f"Error in SSE stream: {e}")
                        yield f"data: {json.dumps({'error': str(e)})}\n\n"
                        break
                        
            finally:
                # Clean up connection
                if session_id in active_sse_connections and connection_id in active_sse_connections[session_id]:
                    del active_sse_connections[session_id][connection_id]
                    if not active_sse_connections[session_id]:
                        del active_sse_connections[session_id]
                        
        except Exception as e:
            current_app.logger.error(f"Session stream setup error: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return Response(
        event_stream(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'Access-Control-Allow-Origin': '*'
        }
    )

@streaming_bp.route('/api/chat/stream')
@rate_limit("50 per minute")
def chat_stream_get():
    """Stream chat response using Server-Sent Events"""
    query = request.args.get('query', '')
    system_message = request.args.get('system_message')
    model_id = request.args.get('model_id')
    
    if not query or not model_id:
        return Response("Query and model_id are required", status=400)
    
    def generate():
        try:
            yield f"data: {json.dumps({'type': 'start', 'model': model_id})}\n\n"
            
            # Import AI model manager
            from ..services import get_ai_model_manager
            ai_model_manager = get_ai_model_manager()
            
            if not ai_model_manager:
                yield f"data: {json.dumps({'type': 'error', 'error': 'AI model manager not available'})}\n\n"
                return
            
            # For now, simulate streaming by chunking the response
            # In a real implementation, you'd integrate with streaming APIs
            import asyncio
            
            # Get regular response first
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                response = loop.run_until_complete(
                    ai_model_manager.generate_response(
                        query=query,
                        system_message=system_message,
                        model_id=model_id,
                        user_id=session.get('user_id', 'anonymous')
                    )
                )
                
                # Stream the response in chunks
                full_response = response['response']
                words = full_response.split()
                
                for i, word in enumerate(words):
                    chunk_data = {
                        'type': 'chunk',
                        'content': word + ' ',
                        'index': i,
                        'finished': False
                    }
                    yield f"data: {json.dumps(chunk_data)}\n\n"
                    time.sleep(0.05)  # Simulate streaming delay
                
                # Send completion
                completion_data = {
                    'type': 'completion',
                    'model': response['model'],
                    'usage': response.get('usage', {}),
                    'cached': response.get('cached', False),
                    'finished': True
                }
                yield f"data: {json.dumps(completion_data)}\n\n"
                
            finally:
                loop.close()
                
        except Exception as e:
            current_app.logger.error(f"Chat streaming error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
    
    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'Access-Control-Allow-Origin': '*'
        }
    )

@streaming_bp.route('/api/shared-memory/sessions/<session_id>/messages/latest')
@rate_limit("100 per minute")
def get_latest_messages(session_id):
    """Get the latest messages from a session (for polling fallback)"""
    try:
        from ..services import get_shared_memory_manager
        shared_memory_manager = get_shared_memory_manager()
        
        if not shared_memory_manager:
            return {'error': 'Shared memory manager not initialized'}, 500
        
        # Get query parameters
        since = request.args.get('since', type=float)
        limit = int(request.args.get('limit', 10))
        
        messages = shared_memory_manager.get_session_messages(session_id)
        
        # Filter by timestamp if provided
        if since:
            messages = [msg for msg in messages if msg.timestamp > since]
        
        # Limit results
        messages = messages[-limit:]
        
        return {
            'session_id': session_id,
            'messages': [msg.to_dict() for msg in messages],
            'timestamp': time.time()
        }
        
    except Exception as e:
        current_app.logger.error(f"Error getting latest messages: {e}")
        return {'error': str(e)}, 500
