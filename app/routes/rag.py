"""
RAG System Routes
API endpoints for document management and retrieval-augmented generation
"""

from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
from datetime import datetime
import os

from ..services.rag_service import get_rag_service
from ..utils.decorators import rate_limit, require_auth, validate_json
from ..utils.validators import validate_file_upload
from ..utils.exceptions import ValidationError, ServiceError, RAGError

# Create blueprint
rag_bp = Blueprint('rag', __name__)

@rag_bp.route('/upload', methods=['POST'])
@rate_limit("20 per minute")
@require_auth(optional=True)
def upload_document():
    """Upload a document to the RAG system"""
    try:
        rag_service = get_rag_service()
        if not rag_service:
            return jsonify({'error': 'RAG service not available'}), 503
        
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Validate file
        file_data = {
            'filename': file.filename,
            'size': len(file.read())
        }
        file.seek(0)  # Reset file pointer
        
        validation_errors = validate_file_upload(file_data)
        if validation_errors:
            return jsonify({
                'error': 'File validation failed',
                'details': validation_errors
            }), 400
        
        # Get additional metadata
        description = request.form.get('description', '')
        tags = request.form.get('tags', '').split(',') if request.form.get('tags') else []
        
        # Upload and process document
        result = rag_service.upload_document(
            file=file,
            description=description,
            tags=[tag.strip() for tag in tags if tag.strip()],
            user_id=request.form.get('user_id', 'anonymous')
        )
        
        return jsonify({
            'message': 'Document uploaded successfully',
            'document': result
        }), 201
        
    except ValidationError as e:
        return jsonify({'error': str(e)}), 400
    except RAGError as e:
        return jsonify({'error': str(e)}), 500
    except Exception as e:
        current_app.logger.error(f"Document upload error: {e}")
        return jsonify({'error': 'Document upload failed'}), 500

@rag_bp.route('/documents', methods=['GET'])
@rate_limit("100 per minute")
def list_documents():
    """List all documents in the RAG system"""
    try:
        rag_service = get_rag_service()
        if not rag_service:
            return jsonify({'error': 'RAG service not available'}), 503
        
        # Get query parameters
        page = request.args.get('page', 1, type=int)
        per_page = min(request.args.get('per_page', 20, type=int), 100)
        search = request.args.get('search', '')
        file_type = request.args.get('file_type', '')
        tags = request.args.get('tags', '').split(',') if request.args.get('tags') else []
        
        documents = rag_service.list_documents(
            page=page,
            per_page=per_page,
            search=search,
            file_type=file_type,
            tags=[tag.strip() for tag in tags if tag.strip()]
        )
        
        return jsonify({
            'documents': documents['documents'],
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total': documents['total'],
                'pages': documents['pages']
            }
        })
        
    except Exception as e:
        current_app.logger.error(f"Error listing documents: {e}")
        return jsonify({'error': 'Failed to retrieve documents'}), 500

@rag_bp.route('/documents/<document_id>', methods=['GET'])
@rate_limit("200 per minute")
def get_document(document_id):
    """Get specific document details"""
    try:
        rag_service = get_rag_service()
        if not rag_service:
            return jsonify({'error': 'RAG service not available'}), 503
        
        document = rag_service.get_document(document_id)
        if not document:
            return jsonify({'error': 'Document not found'}), 404
        
        return jsonify({'document': document})
        
    except Exception as e:
        current_app.logger.error(f"Error getting document {document_id}: {e}")
        return jsonify({'error': 'Failed to retrieve document'}), 500

@rag_bp.route('/documents/<document_id>', methods=['DELETE'])
@rate_limit("30 per minute")
@require_auth(roles=['admin', 'user'])
def delete_document(document_id):
    """Delete a document from the RAG system"""
    try:
        rag_service = get_rag_service()
        if not rag_service:
            return jsonify({'error': 'RAG service not available'}), 503
        
        success = rag_service.delete_document(document_id)
        if not success:
            return jsonify({'error': 'Document not found'}), 404
        
        return jsonify({'message': 'Document deleted successfully'})
        
    except Exception as e:
        current_app.logger.error(f"Error deleting document {document_id}: {e}")
        return jsonify({'error': 'Failed to delete document'}), 500

@rag_bp.route('/search', methods=['POST'])
@rate_limit("100 per minute")
@validate_json(['query'])
def search_documents():
    """Search documents using semantic search"""
    try:
        rag_service = get_rag_service()
        if not rag_service:
            return jsonify({'error': 'RAG service not available'}), 503
        
        data = request.get_json()
        query = data['query']
        
        if not query.strip():
            return jsonify({'error': 'Query cannot be empty'}), 400
        
        # Search parameters
        max_results = min(data.get('max_results', 5), 20)
        min_similarity = data.get('min_similarity', 0.7)
        file_types = data.get('file_types', [])
        tags = data.get('tags', [])
        
        results = rag_service.search_documents(
            query=query,
            max_results=max_results,
            min_similarity=min_similarity,
            file_types=file_types,
            tags=tags
        )
        
        return jsonify({
            'query': query,
            'results': results,
            'total_found': len(results)
        })
        
    except ValidationError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        current_app.logger.error(f"Document search error: {e}")
        return jsonify({'error': 'Document search failed'}), 500

@rag_bp.route('/query', methods=['POST'])
@rate_limit("50 per minute")
@validate_json(['query'])
def rag_query():
    """Perform RAG-enhanced query"""
    try:
        rag_service = get_rag_service()
        if not rag_service:
            return jsonify({'error': 'RAG service not available'}), 503
        
        data = request.get_json()
        query = data['query']
        
        if not query.strip():
            return jsonify({'error': 'Query cannot be empty'}), 400
        
        # RAG parameters
        max_context_docs = min(data.get('max_context_docs', 3), 10)
        model_id = data.get('model_id')
        include_sources = data.get('include_sources', True)
        
        result = rag_service.query_with_context(
            query=query,
            max_context_docs=max_context_docs,
            model_id=model_id,
            include_sources=include_sources
        )
        
        return jsonify(result)
        
    except ValidationError as e:
        return jsonify({'error': str(e)}), 400
    except RAGError as e:
        return jsonify({'error': str(e)}), 500
    except Exception as e:
        current_app.logger.error(f"RAG query error: {e}")
        return jsonify({'error': 'RAG query failed'}), 500

@rag_bp.route('/chunks/<document_id>', methods=['GET'])
@rate_limit("100 per minute")
def get_document_chunks(document_id):
    """Get chunks for a specific document"""
    try:
        rag_service = get_rag_service()
        if not rag_service:
            return jsonify({'error': 'RAG service not available'}), 503
        
        # Get pagination parameters
        page = request.args.get('page', 1, type=int)
        per_page = min(request.args.get('per_page', 20, type=int), 100)
        
        chunks = rag_service.get_document_chunks(
            document_id=document_id,
            page=page,
            per_page=per_page
        )
        
        if chunks is None:
            return jsonify({'error': 'Document not found'}), 404
        
        return jsonify({
            'document_id': document_id,
            'chunks': chunks['chunks'],
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total': chunks['total'],
                'pages': chunks['pages']
            }
        })
        
    except Exception as e:
        current_app.logger.error(f"Error getting document chunks: {e}")
        return jsonify({'error': 'Failed to retrieve document chunks'}), 500

@rag_bp.route('/reindex', methods=['POST'])
@rate_limit("5 per minute")
@require_auth(roles=['admin'])
def reindex_documents():
    """Reindex all documents in the RAG system"""
    try:
        rag_service = get_rag_service()
        if not rag_service:
            return jsonify({'error': 'RAG service not available'}), 503
        
        data = request.get_json() or {}
        force = data.get('force', False)
        
        if not force:
            return jsonify({
                'error': 'Confirmation required',
                'message': 'Set "force": true to proceed with reindexing'
            }), 400
        
        # Start reindexing process
        result = rag_service.reindex_documents()
        
        return jsonify({
            'message': 'Document reindexing started',
            'job_id': result.get('job_id'),
            'estimated_time': result.get('estimated_time')
        })
        
    except Exception as e:
        current_app.logger.error(f"Error starting reindexing: {e}")
        return jsonify({'error': 'Failed to start reindexing'}), 500

@rag_bp.route('/stats', methods=['GET'])
@rate_limit("100 per minute")
def get_rag_stats():
    """Get RAG system statistics"""
    try:
        rag_service = get_rag_service()
        if not rag_service:
            return jsonify({'error': 'RAG service not available'}), 503
        
        stats = rag_service.get_statistics()
        
        return jsonify({
            'stats': stats,
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        current_app.logger.error(f"Error getting RAG stats: {e}")
        return jsonify({'error': 'Failed to retrieve RAG statistics'}), 500

@rag_bp.route('/health', methods=['GET'])
@rate_limit("200 per minute")
def rag_health():
    """Get RAG system health status"""
    try:
        rag_service = get_rag_service()
        if not rag_service:
            return jsonify({
                'status': 'unavailable',
                'message': 'RAG service not available'
            }), 503
        
        health = rag_service.get_health()
        
        status_code = 200 if health['status'] == 'healthy' else 503
        
        return jsonify(health), status_code
        
    except Exception as e:
        current_app.logger.error(f"Error getting RAG health: {e}")
        return jsonify({
            'status': 'error',
            'message': 'Failed to check RAG health'
        }), 500

@rag_bp.route('/config', methods=['GET'])
@rate_limit("50 per minute")
@require_auth(roles=['admin'])
def get_rag_config():
    """Get RAG system configuration"""
    try:
        rag_service = get_rag_service()
        if not rag_service:
            return jsonify({'error': 'RAG service not available'}), 503
        
        config = rag_service.get_config()
        
        return jsonify({
            'config': config,
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        current_app.logger.error(f"Error getting RAG config: {e}")
        return jsonify({'error': 'Failed to retrieve RAG configuration'}), 500

@rag_bp.route('/config', methods=['PUT'])
@rate_limit("10 per minute")
@require_auth(roles=['admin'])
@validate_json()
def update_rag_config():
    """Update RAG system configuration"""
    try:
        rag_service = get_rag_service()
        if not rag_service:
            return jsonify({'error': 'RAG service not available'}), 503
        
        data = request.get_json()
        
        # Validate configuration keys
        valid_keys = [
            'chunk_size', 'chunk_overlap', 'max_context_docs',
            'similarity_threshold', 'embedding_model', 'vector_store_config'
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
        success = rag_service.update_config(config_updates)
        if not success:
            return jsonify({'error': 'Failed to update configuration'}), 500
        
        return jsonify({
            'message': 'RAG configuration updated successfully',
            'updated_keys': list(config_updates.keys())
        })
        
    except ValidationError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        current_app.logger.error(f"Error updating RAG config: {e}")
        return jsonify({'error': 'Failed to update RAG configuration'}), 500