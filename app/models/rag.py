"""
RAG (Retrieval-Augmented Generation) System Models
Models for document storage, chunking, and vector search
"""

from datetime import datetime
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import Integer, String, Float, DateTime, Boolean, Text, JSON, ForeignKey, LargeBinary
import hashlib

from .base import Base, TimestampMixin, generate_id

class Document(Base, TimestampMixin):
    """Document storage for RAG system"""
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_id)
    
    # Document identification
    title: Mapped[str] = mapped_column(String(500), nullable=False, index=True)
    filename: Mapped[str | None] = mapped_column(String(255), nullable=True)
    file_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    
    # Content information
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)  # pdf, docx, txt, etc.
    file_size: Mapped[int] = mapped_column(Integer, nullable=False)
    
    # Processing status
    status: Mapped[str] = mapped_column(String(20), default='uploaded', index=True)  # uploaded, processing, processed, failed
    processing_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    
    # Document metadata
    language: Mapped[str | None] = mapped_column(String(10), nullable=True)
    author: Mapped[str | None] = mapped_column(String(200), nullable=True)
    source_url: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    
    # Processing configuration
    chunk_size: Mapped[int] = mapped_column(Integer, default=1000)
    chunk_overlap: Mapped[int] = mapped_column(Integer, default=100)
    
    # Statistics
    total_chunks: Mapped[int] = mapped_column(Integer, default=0)
    word_count: Mapped[int] = mapped_column(Integer, default=0)
    character_count: Mapped[int] = mapped_column(Integer, default=0)
    
    # Usage tracking
    query_count: Mapped[int] = mapped_column(Integer, default=0)
    last_queried: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    
    # Additional metadata - FIXED: renamed to avoid SQLAlchemy conflict
    tags: Mapped[list | None] = mapped_column(JSON, nullable=True)
    document_metadata: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    
    # Relationships
    chunks: Mapped[list["DocumentChunk"]] = relationship(
        "DocumentChunk", back_populates="document", cascade="all, delete-orphan"
    )
    
    def __repr__(self):
        return f"<Document {self.title[:50]}... - {self.total_chunks} chunks>"
    
    @classmethod
    def generate_file_hash(cls, content: str) -> str:
        """Generate hash for file content"""
        return hashlib.sha256(content.encode()).hexdigest()
    
    def update_processing_status(self, status: str, error: str = None):
        """Update document processing status"""
        self.status = status
        self.processing_error = error
        self.updated_at = datetime.utcnow()
    
    def record_query(self):
        """Record that this document was queried"""
        self.query_count += 1
        self.last_queried = datetime.utcnow()
    
    def to_dict(self):
        """Convert to dictionary representation"""
        return {
            'id': self.id,
            'title': self.title,
            'filename': self.filename,
            'content_type': self.content_type,
            'file_size': self.file_size,
            'status': self.status,
            'language': self.language,
            'author': self.author,
            'total_chunks': self.total_chunks,
            'word_count': self.word_count,
            'character_count': self.character_count,
            'query_count': self.query_count,
            'last_queried': self.last_queried.isoformat() if self.last_queried else None,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'tags': self.tags,
            'metadata': self.document_metadata  # Return as 'metadata' for API compatibility
        }

class DocumentChunk(Base, TimestampMixin):
    """Document chunks for vector search"""
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_id)
    
    # Document reference
    document_id: Mapped[str] = mapped_column(String(36), ForeignKey('document.id'), nullable=False, index=True)
    document: Mapped["Document"] = relationship("Document", back_populates="chunks")
    
    # Chunk identification
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    chunk_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    
    # Chunk content
    content: Mapped[str] = mapped_column(Text, nullable=False)
    
    # Position information
    start_char: Mapped[int] = mapped_column(Integer, nullable=False)
    end_char: Mapped[int] = mapped_column(Integer, nullable=False)
    word_count: Mapped[int] = mapped_column(Integer, nullable=False)
    
    # Vector information (stored as references to external vector DB)
    vector_id: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)
    embedding_model: Mapped[str | None] = mapped_column(String(100), nullable=True)
    
    # Semantic information
    section_title: Mapped[str | None] = mapped_column(String(500), nullable=True)
    keywords: Mapped[list | None] = mapped_column(JSON, nullable=True)
    
    # Usage tracking
    retrieval_count: Mapped[int] = mapped_column(Integer, default=0)
    last_retrieved: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    relevance_scores: Mapped[list | None] = mapped_column(JSON, nullable=True)  # Historical relevance scores
    
    # Additional metadata - FIXED: renamed to avoid SQLAlchemy conflict
    chunk_metadata: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    
    def __repr__(self):
        return f"<DocumentChunk {self.document_id}:{self.chunk_index} - {self.retrieval_count} retrievals>"
    
    @classmethod
    def generate_chunk_hash(cls, content: str, document_id: str, chunk_index: int) -> str:
        """Generate hash for chunk content"""
        hash_content = f"{document_id}:{chunk_index}:{content}"
        return hashlib.sha256(hash_content.encode()).hexdigest()
    
    def record_retrieval(self, relevance_score: float = None):
        """Record that this chunk was retrieved"""
        self.retrieval_count += 1
        self.last_retrieved = datetime.utcnow()
        
        if relevance_score is not None:
            if self.relevance_scores is None:
                self.relevance_scores = []
            self.relevance_scores.append({
                'score': relevance_score,
                'timestamp': datetime.utcnow().isoformat()
            })
            # Keep only last 100 scores
            if len(self.relevance_scores) > 100:
                self.relevance_scores = self.relevance_scores[-100:]
    
    @property
    def average_relevance_score(self) -> float:
        """Calculate average relevance score"""
        if not self.relevance_scores:
            return 0.0
        scores = [item['score'] for item in self.relevance_scores]
        return sum(scores) / len(scores)
    
    def to_dict(self):
        """Convert to dictionary representation"""
        return {
            'id': self.id,
            'document_id': self.document_id,
            'chunk_index': self.chunk_index,
            'content': self.content[:200] + '...' if len(self.content) > 200 else self.content,
            'start_char': self.start_char,
            'end_char': self.end_char,
            'word_count': self.word_count,
            'section_title': self.section_title,
            'keywords': self.keywords,
            'retrieval_count': self.retrieval_count,
            'last_retrieved': self.last_retrieved.isoformat() if self.last_retrieved else None,
            'average_relevance_score': self.average_relevance_score,
            'created_at': self.created_at.isoformat(),
            'metadata': self.chunk_metadata  # Return as 'metadata' for API compatibility
        }

class RAGQuery(Base, TimestampMixin):
    """RAG query tracking and results"""
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_id)
    
    # Query identification
    query_text: Mapped[str] = mapped_column(Text, nullable=False)
    query_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    
    # User information
    user_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    session_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    
    # Search parameters
    search_type: Mapped[str] = mapped_column(String(20), default='similarity')  # similarity, keyword, hybrid
    max_results: Mapped[int] = mapped_column(Integer, default=5)
    similarity_threshold: Mapped[float] = mapped_column(Float, default=0.7)
    
    # Search results
    results_found: Mapped[int] = mapped_column(Integer, default=0)
    results_used: Mapped[int] = mapped_column(Integer, default=0)
    best_match_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    retrieved_chunks: Mapped[list | None] = mapped_column(JSON, nullable=True)
    
    # Performance metrics
    search_time: Mapped[float | None] = mapped_column(Float, nullable=True)  # seconds
    
    # Context generation
    context_used: Mapped[str | None] = mapped_column(Text, nullable=True)
    context_length: Mapped[int] = mapped_column(Integer, default=0)
    
    # Additional metadata - FIXED: renamed to avoid SQLAlchemy conflict
    rag_metadata: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    
    def __repr__(self):
        return f"<RAGQuery {self.query_text[:50]}... - {self.results_found} results>"
    
    @classmethod
    def generate_query_hash(cls, query: str) -> str:
        """Generate hash for query text"""
        return hashlib.sha256(query.strip().lower().encode()).hexdigest()[:16]
    
    def add_retrieved_chunk(self, chunk_id: str, score: float, rank: int):
        """Add a retrieved chunk to the results"""
        if self.retrieved_chunks is None:
            self.retrieved_chunks = []
        
        self.retrieved_chunks.append({
            'chunk_id': chunk_id,
            'score': score,
            'rank': rank,
            'timestamp': datetime.utcnow().isoformat()
        })
    
    def to_dict(self):
        """Convert to dictionary representation"""
        return {
            'id': self.id,
            'query_text': self.query_text,
            'search_type': self.search_type,
            'max_results': self.max_results,
            'results_found': self.results_found,
            'results_used': self.results_used,
            'best_match_score': self.best_match_score,
            'search_time': self.search_time,
            'context_used': self.context_used,
            'context_length': self.context_length,
            'created_at': self.created_at.isoformat(),
            'retrieved_chunks': self.retrieved_chunks,
            'metadata': self.rag_metadata  # Return as 'metadata' for API compatibility
        }

# Create indexes for performance
from sqlalchemy import Index

Index('idx_document_status_created', Document.status, Document.created_at)
Index('idx_document_type_created', Document.content_type, Document.created_at)
Index('idx_chunk_document_index', DocumentChunk.document_id, DocumentChunk.chunk_index)
Index('idx_chunk_retrieval_count', DocumentChunk.retrieval_count.desc())
Index('idx_rag_query_user_created', RAGQuery.user_id, RAGQuery.created_at)
Index('idx_rag_query_hash_created', RAGQuery.query_hash, RAGQuery.created_at)