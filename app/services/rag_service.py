"""
RAG (Retrieval-Augmented Generation) Service
Handles document indexing, vector search, and RAG query processing
"""

import os
import json
import hashlib
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from flask import current_app
import numpy as np
from sqlalchemy import text

from app.extensions import db
from app.models.rag import RAGDocument, RAGChunk, RAGQuery, RAGResult
from app.utils.exceptions import RAGError, ValidationError, ServiceError


class RAGService:
    """Service for handling RAG operations"""
    
    def __init__(self):
        self.embeddings_model = None
        self.vector_store = None
        self.chunk_size = 1000
        self.chunk_overlap = 200
        self.similarity_threshold = 0.7
        self.max_results = 10
        self.initialized = False
    
    def initialize(self, app):
        """Initialize the RAG service with app configuration"""
        try:
            self.chunk_size = app.config.get('RAG_CHUNK_SIZE', 1000)
            self.chunk_overlap = app.config.get('RAG_CHUNK_OVERLAP', 200)
            self.similarity_threshold = app.config.get('RAG_SIMILARITY_THRESHOLD', 0.7)
            self.max_results = app.config.get('RAG_MAX_RESULTS', 10)
            
            self._init_embeddings_model(app)
            
            self._init_vector_store(app)
            
            self.initialized = True
            current_app.logger.info("RAG service initialized successfully")
            
        except Exception as e:
            current_app.logger.error(f"RAG service initialization failed: {e}")
            self.initialized = False
    
    def index_document(self, document_path: str, metadata: Dict[str, Any] = None) -> str:
        """
        Index a document for RAG retrieval
        
        Args:
            document_path: Path to the document file
            metadata: Additional metadata for the document
            
        Returns:
            Document ID
            
        Raises:
            RAGError: If indexing fails
            ValidationError: If document is invalid
        """
        if not self.initialized:
            raise RAGError("RAG service not initialized")
        
        try:
            if not os.path.exists(document_path):
                raise ValidationError(f"Document not found: {document_path}")
            
            content = self._read_document(document_path)
            if not content.strip():
                raise ValidationError("Document is empty")
            
            doc_hash = hashlib.sha256(content.encode()).hexdigest()
            
            existing_doc = RAGDocument.query.filter_by(content_hash=doc_hash).first()
            if existing_doc:
                current_app.logger.info(f"Document already indexed: {existing_doc.id}")
                return existing_doc.id
            
            document = RAGDocument(
                filename=os.path.basename(document_path),
                file_path=document_path,
                content=content,
                content_hash=doc_hash,
                metadata=metadata or {},
                created_at=datetime.utcnow(),
                is_active=True
            )
            
            db.session.add(document)
            db.session.flush()  # Get document ID
            
            chunks = self._chunk_document(content)
            
            for i, chunk_text in enumerate(chunks):
                embeddings = self._generate_embeddings(chunk_text)
                
                chunk = RAGChunk(
                    document_id=document.id,
                    chunk_index=i,
                    content=chunk_text,
                    embeddings=embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings,
                    created_at=datetime.utcnow(),
                    is_active=True
                )
                
                db.session.add(chunk)
            
            db.session.commit()
            
            current_app.logger.info(f"Document indexed: {document.id} with {len(chunks)} chunks")
            return document.id
            
        except Exception as e:
            db.session.rollback()
            current_app.logger.error(f"Document indexing error: {e}")
            if isinstance(e, (RAGError, ValidationError)):
                raise
            raise RAGError(f"Failed to index document: {str(e)}")
    
    def search_documents(self, query: str, limit: int = None) -> List[Dict[str, Any]]:
        """
        Search for relevant documents using vector similarity
        
        Args:
            query: Search query
            limit: Maximum number of results (optional)
            
        Returns:
            List of relevant document chunks with scores
            
        Raises:
            RAGError: If search fails
        """
        if not self.initialized:
            raise RAGError("RAG service not initialized")
        
        try:
            limit = limit or self.max_results
            
            query_embeddings = self._generate_embeddings(query)
            
            similar_chunks = self._vector_search(query_embeddings, limit)
            
            results = []
            for chunk, score in similar_chunks:
                if score >= self.similarity_threshold:
                    results.append({
                        'chunk_id': chunk.id,
                        'document_id': chunk.document_id,
                        'content': chunk.content,
                        'score': float(score),
                        'document_filename': chunk.document.filename,
                        'document_metadata': chunk.document.metadata,
                        'chunk_index': chunk.chunk_index
                    })
            
            self._log_search(query, len(results))
            
            return results
            
        except Exception as e:
            current_app.logger.error(f"Document search error: {e}")
            if isinstance(e, RAGError):
                raise
            raise RAGError(f"Search failed: {str(e)}")
    
    def process_rag_query(self, query: str, context_limit: int = 5) -> Dict[str, Any]:
        """
        Process a RAG query with context retrieval
        
        Args:
            query: User query
            context_limit: Maximum number of context chunks
            
        Returns:
            Dict containing query results and context
        """
        if not self.initialized:
            raise RAGError("RAG service not initialized")
        
        try:
            context_chunks = self.search_documents(query, context_limit)
            
            context_text = self._prepare_context(context_chunks)
            
            rag_query = RAGQuery(
                query=query,
                context_chunks=len(context_chunks),
                created_at=datetime.utcnow()
            )
            
            db.session.add(rag_query)
            db.session.flush()
            
            response = self._generate_response(query, context_text)
            
            rag_result = RAGResult(
                query_id=rag_query.id,
                response=response,
                context_used=context_text,
                confidence_score=self._calculate_confidence(context_chunks),
                created_at=datetime.utcnow()
            )
            
            db.session.add(rag_result)
            db.session.commit()
            
            return {
                'query_id': rag_query.id,
                'query': query,
                'response': response,
                'context_chunks': context_chunks,
                'confidence_score': rag_result.confidence_score,
                'timestamp': rag_result.created_at.isoformat()
            }
            
        except Exception as e:
            db.session.rollback()
            current_app.logger.error(f"RAG query processing error: {e}")
            if isinstance(e, RAGError):
                raise
            raise RAGError(f"RAG query failed: {str(e)}")
    
    def get_document_stats(self) -> Dict[str, Any]:
        """
        Get RAG document statistics
        
        Returns:
            Dict containing document and chunk statistics
        """
        try:
            stats = {
                'total_documents': RAGDocument.query.filter_by(is_active=True).count(),
                'total_chunks': RAGChunk.query.filter_by(is_active=True).count(),
                'total_queries': RAGQuery.query.count(),
                'avg_chunks_per_document': 0,
                'recent_queries': 0
            }
            
            if stats['total_documents'] > 0:
                stats['avg_chunks_per_document'] = stats['total_chunks'] / stats['total_documents']
            
            yesterday = datetime.utcnow() - timedelta(days=1)
            stats['recent_queries'] = RAGQuery.query.filter(
                RAGQuery.created_at >= yesterday
            ).count()
            
            return stats
            
        except Exception as e:
            current_app.logger.error(f"RAG stats error: {e}")
            return {'error': str(e)}
    
    def delete_document(self, document_id: str) -> bool:
        """
        Delete a document and its chunks
        
        Args:
            document_id: Document ID to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            document = RAGDocument.query.get(document_id)
            if not document:
                return False
            
            document.is_active = False
            
            RAGChunk.query.filter_by(document_id=document_id).update({
                'is_active': False
            })
            
            db.session.commit()
            
            current_app.logger.info(f"Document deleted: {document_id}")
            return True
            
        except Exception as e:
            db.session.rollback()
            current_app.logger.error(f"Document deletion error: {e}")
            return False
    
    def _init_embeddings_model(self, app):
        """Initialize embeddings model (placeholder)"""
        self.embeddings_model = "placeholder"
        current_app.logger.info("Embeddings model initialized (placeholder)")
    
    def _init_vector_store(self, app):
        """Initialize vector store (placeholder)"""
        self.vector_store = "placeholder"
        current_app.logger.info("Vector store initialized (placeholder)")
    
    def _read_document(self, document_path: str) -> str:
        """Read document content"""
        try:
            with open(document_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            with open(document_path, 'r', encoding='latin-1') as f:
                return f.read()
    
    def _chunk_document(self, content: str) -> List[str]:
        """Split document into chunks"""
        chunks = []
        words = content.split()
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = ' '.join(chunk_words)
            chunks.append(chunk_text)
        
        return chunks
    
    def _generate_embeddings(self, text: str) -> List[float]:
        """Generate embeddings for text (placeholder)"""
        import random
        return [random.random() for _ in range(384)]  # Typical embedding dimension
    
    def _vector_search(self, query_embeddings: List[float], limit: int) -> List[Tuple[RAGChunk, float]]:
        """Search for similar vectors (placeholder)"""
        chunks = RAGChunk.query.filter_by(is_active=True).limit(limit).all()
        results = []
        
        for chunk in chunks:
            score = random.uniform(0.5, 1.0)
            results.append((chunk, score))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    
    def _prepare_context(self, context_chunks: List[Dict[str, Any]]) -> str:
        """Prepare context text from chunks"""
        context_parts = []
        for chunk in context_chunks:
            context_parts.append(f"[Document: {chunk['document_filename']}]\n{chunk['content']}")
        
        return "\n\n".join(context_parts)
    
    def _generate_response(self, query: str, context: str) -> str:
        """Generate response using query and context (placeholder)"""
        return f"Based on the provided context, here's a response to your query: {query}"
    
    def _calculate_confidence(self, context_chunks: List[Dict[str, Any]]) -> float:
        """Calculate confidence score based on context quality"""
        if not context_chunks:
            return 0.0
        
        avg_score = sum(chunk['score'] for chunk in context_chunks) / len(context_chunks)
        return min(avg_score, 1.0)
    
    def _log_search(self, query: str, results_count: int):
        """Log search for analytics"""
        try:
            current_app.logger.info(f"RAG search: query='{query}', results={results_count}")
        except Exception as e:
            current_app.logger.warning(f"Search logging error: {e}")


# Singleton instance
_rag_service = None

def get_rag_service() -> RAGService:
    """Get singleton RAG service instance"""
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
        if current_app:
            _rag_service.initialize(current_app)
    return _rag_service

def init_rag_service(app):
    """Initialize RAG service with Flask app"""
    service = get_rag_service()
    service.initialize(app)
    return service
