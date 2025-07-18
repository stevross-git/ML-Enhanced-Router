"""
Database Models for ML Enhanced Router
Fixed to remove circular imports, use proper SQLAlchemy 2.x syntax, and avoid reserved words
"""

from datetime import datetime
from sqlalchemy.orm import Mapped, mapped_column, DeclarativeBase, relationship
from sqlalchemy import Integer, String, Float, DateTime, Boolean, Text, JSON, ForeignKey
import hashlib
import uuid


class Base(DeclarativeBase):
    """Base class for all models"""
    pass


class QueryLog(Base):
    """Log of all queries processed by the router"""
    __tablename__ = "query_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    user_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    category: Mapped[str] = mapped_column(String(32), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    agent_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    agent_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    response_time: Mapped[float | None] = mapped_column(Float, nullable=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    meta_data: Mapped[dict | None] = mapped_column(JSON, nullable=True)


class AgentRegistration(Base):
    """Registry of all agents"""
    __tablename__ = "agent_registrations"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    endpoint: Mapped[str] = mapped_column(String(256), nullable=False)
    categories: Mapped[dict] = mapped_column(JSON, nullable=False)
    capabilities: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    meta_data: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    last_seen: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class RouterMetrics(Base):
    """Performance metrics for the router"""
    __tablename__ = "router_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    metric_name: Mapped[str] = mapped_column(String(64), nullable=False)
    metric_value: Mapped[float] = mapped_column(Float, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    meta_data: Mapped[dict | None] = mapped_column(JSON, nullable=True)


class MLModelRegistry(Base):
    """Registry of ML models"""
    __tablename__ = "ml_model_registry"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    model_type: Mapped[str] = mapped_column(String(32), nullable=False)
    categories: Mapped[dict] = mapped_column(JSON, nullable=False)
    config: Mapped[dict] = mapped_column(JSON, nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="inactive")
    accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    version: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class AICacheEntry(Base):
    """AI response cache entries"""
    __tablename__ = "ai_cache_entries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    cache_key: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    response: Mapped[str] = mapped_column(Text, nullable=False)
    model_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    system_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    meta_data: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    hit_count: Mapped[int] = mapped_column(Integer, default=0)
    last_accessed: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    
    @classmethod
    def generate_cache_key(cls, query: str, model_id: str, system_message: str = None) -> str:
        """Generate cache key from query parameters"""
        key_data = f"{query}|{model_id}|{system_message or ''}"
        return hashlib.sha256(key_data.encode()).hexdigest()


class AICacheStats(Base):
    """AI cache statistics"""
    __tablename__ = "ai_cache_stats"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    date: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    cache_hits: Mapped[int] = mapped_column(Integer, default=0)
    cache_misses: Mapped[int] = mapped_column(Integer, default=0)
    total_requests: Mapped[int] = mapped_column(Integer, default=0)
    hit_rate: Mapped[float] = mapped_column(Float, default=0.0)
    avg_response_time: Mapped[float] = mapped_column(Float, default=0.0)
    model_usage: Mapped[dict | None] = mapped_column(JSON, nullable=True)


class ChatSession(Base):
    """Chat session model for storing chat conversations"""
    __tablename__ = "chat_sessions"

    id: Mapped[str] = mapped_column(String(100), primary_key=True, default=lambda: str(uuid.uuid4()))
    title: Mapped[str] = mapped_column(String(200), nullable=False, default="New Chat")
    user_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    model_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    system_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    message_count: Mapped[int] = mapped_column(Integer, default=0)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    
    # Relationship to messages
    messages: Mapped[list["ChatMessage"]] = relationship('ChatMessage', back_populates='session', cascade='all, delete-orphan')
    
    def __repr__(self):
        return f'<ChatSession {self.id}: {self.title}>'


class ChatMessage(Base):
    """Chat message model for storing individual messages"""
    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[str] = mapped_column(String(100), ForeignKey("chat_sessions.id"), nullable=False)
    role: Mapped[str] = mapped_column(String(20), nullable=False)  # 'user', 'assistant', 'system'
    content: Mapped[str] = mapped_column(Text, nullable=False)
    model_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    system_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
    usage_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cached: Mapped[bool] = mapped_column(Boolean, default=False)
    attachments: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    
    # Relationship to session
    session: Mapped["ChatSession"] = relationship('ChatSession', back_populates='messages')
    
    def __repr__(self):
        return f'<ChatMessage {self.id}: {self.role} - {self.content[:50]}...>'


# RAG System Models
class Document(Base):
    """Document model for storing uploaded documents"""
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    original_name: Mapped[str] = mapped_column(String(255), nullable=False)
    file_type: Mapped[str] = mapped_column(String(50), nullable=False)
    file_size: Mapped[int] = mapped_column(Integer, nullable=False)
    file_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    content: Mapped[str | None] = mapped_column(Text, nullable=True)
    document_metadata: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    uploaded_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    uploaded_by: Mapped[str | None] = mapped_column(String(100), nullable=True)
    is_processed: Mapped[bool] = mapped_column(Boolean, default=False)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    
    # Relationship to chunks
    chunks: Mapped[list["DocumentChunk"]] = relationship('DocumentChunk', back_populates='document', cascade='all, delete-orphan')
    
    def __repr__(self):
        return f'<Document {self.id}: {self.original_name}>'


class DocumentChunk(Base):
    """Document chunk model for storing document segments"""
    __tablename__ = "document_chunks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id: Mapped[str] = mapped_column(String(36), ForeignKey("documents.id"), nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_size: Mapped[int] = mapped_column(Integer, nullable=False)
    embedding_id: Mapped[str | None] = mapped_column(String(64), nullable=True)  # ChromaDB ID
    chunk_metadata: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationship to document
    document: Mapped["Document"] = relationship('Document', back_populates='chunks')
    
    def __repr__(self):
        return f'<DocumentChunk {self.id}: Doc {self.document_id}, Chunk {self.chunk_index}>'


class RAGQuery(Base):
    """RAG query model for storing search queries and results"""
    __tablename__ = "rag_queries"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    query: Mapped[str] = mapped_column(Text, nullable=False)
    user_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    results_found: Mapped[int] = mapped_column(Integer, default=0)
    max_results: Mapped[int] = mapped_column(Integer, default=5)
    query_time: Mapped[float] = mapped_column(Float, nullable=False)
    query_metadata: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    
    def __repr__(self):
        return f'<RAGQuery {self.id}: {self.query[:50]}...>'


# Enterprise Features Models
class CrossPersonaMemoryLink(Base):
    """Cross-persona memory linkage model"""
    __tablename__ = "cross_persona_memory_links"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    source_persona_id: Mapped[str] = mapped_column(String(100), nullable=False)
    target_persona_id: Mapped[str] = mapped_column(String(100), nullable=False)
    memory_content: Mapped[str] = mapped_column(Text, nullable=False)
    compatibility_score: Mapped[float] = mapped_column(Float, nullable=False)
    bridge_type: Mapped[str] = mapped_column(String(50), nullable=False)  # 'skill_transfer', 'context_bridge', etc.
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    last_accessed: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    access_count: Mapped[int] = mapped_column(Integer, default=0)
    link_metadata: Mapped[dict | None] = mapped_column(JSON, nullable=True)


class CognitiveDebugSession(Base):
    """Cognitive debugging session model"""
    __tablename__ = "cognitive_debug_sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(String(100), nullable=False)
    session_name: Mapped[str] = mapped_column(String(200), nullable=False)
    start_time: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    end_time: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    total_decisions: Mapped[int] = mapped_column(Integer, default=0)
    avg_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    session_metadata: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)


class CognitiveDebugDecision(Base):
    """Individual cognitive debugging decision model"""
    __tablename__ = "cognitive_debug_decisions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id: Mapped[str] = mapped_column(String(36), ForeignKey("cognitive_debug_sessions.id"), nullable=False)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    decision_type: Mapped[str] = mapped_column(String(50), nullable=False)  # 'routing', 'model_selection', etc.
    chosen_option: Mapped[str] = mapped_column(String(200), nullable=False)
    confidence_score: Mapped[float] = mapped_column(Float, nullable=False)
    alternative_options: Mapped[dict] = mapped_column(JSON, nullable=False)
    reasoning: Mapped[str] = mapped_column(Text, nullable=False)
    execution_time: Mapped[float] = mapped_column(Float, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    decision_metadata: Mapped[dict | None] = mapped_column(JSON, nullable=True)


class TemporalMemory(Base):
    """Temporal memory weighting model"""
    __tablename__ = "temporal_memory"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(String(100), nullable=False)
    memory_content: Mapped[str] = mapped_column(Text, nullable=False)
    memory_type: Mapped[str] = mapped_column(String(50), nullable=False)  # 'preference', 'fact', 'context', etc.
    base_importance: Mapped[float] = mapped_column(Float, nullable=False)
    current_weight: Mapped[float] = mapped_column(Float, nullable=False)
    decay_rate: Mapped[float] = mapped_column(Float, default=0.1)
    boost_factor: Mapped[float] = mapped_column(Float, default=1.0)
    last_accessed: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    access_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    memory_metadata: Mapped[dict | None] = mapped_column(JSON, nullable=True)


# External API Models
class ExternalAPICall(Base):
    """External API call logging model"""
    __tablename__ = "external_api_calls"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    provider: Mapped[str] = mapped_column(String(50), nullable=False)
    model_id: Mapped[str] = mapped_column(String(100), nullable=False)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    response: Mapped[str | None] = mapped_column(Text, nullable=True)
    status_code: Mapped[int] = mapped_column(Integer, nullable=False)
    response_time: Mapped[float] = mapped_column(Float, nullable=False)
    tokens_used: Mapped[int] = mapped_column(Integer, default=0)
    cost: Mapped[float] = mapped_column(Float, default=0.0)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    call_metadata: Mapped[dict | None] = mapped_column(JSON, nullable=True)


# User Management Models
class User(Base):
    """User model for authentication and profile management"""
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    username: Mapped[str] = mapped_column(String(80), unique=True, nullable=False)
    email: Mapped[str] = mapped_column(String(120), unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    first_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    last_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    role: Mapped[str] = mapped_column(String(20), default="user")  # 'admin', 'user', 'premium'
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    last_login: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    profile_metadata: Mapped[dict | None] = mapped_column(JSON, nullable=True)


class APIKey(Base):
    """API key management model"""
    __tablename__ = "api_keys"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), nullable=False)
    provider: Mapped[str] = mapped_column(String(50), nullable=False)  # 'openai', 'anthropic', etc.
    key_name: Mapped[str] = mapped_column(String(100), nullable=False)
    encrypted_key: Mapped[str] = mapped_column(Text, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    last_used: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    usage_count: Mapped[int] = mapped_column(Integer, default=0)
    key_metadata: Mapped[dict | None] = mapped_column(JSON, nullable=True)