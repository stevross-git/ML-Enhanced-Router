"""
Unit tests for models
"""

import pytest
from datetime import datetime

from app.models.query import QueryLog
from app.models.agent import Agent, AgentCapability
from app.models.user import User
from app.models.auth import UserSession
from app.models.cache import AICacheEntry
from app.models.rag import Document, DocumentChunk


class TestQueryModel:
    """Test cases for Query model"""
    
    def test_query_log_creation(self, db_session):
        """Test creating a query log entry"""
        query_log = QueryLog(
            query="Test query",
            user_id="test-user",
            response="Test response",
            response_time=1.5,
            created_at=datetime.utcnow()
        )
        
        db_session.add(query_log)
        db_session.commit()
        
        assert query_log.id is not None
        assert query_log.query == "Test query"
        assert query_log.response_time == 1.5


class TestAgentModel:
    """Test cases for Agent model"""
    
    def test_agent_creation(self, db_session):
        """Test creating an agent"""
        agent = Agent(
            id="test-agent-id",
            name="Test Agent",
            type="llm",
            endpoint="http://localhost:8000",
            status="active",
            created_at=datetime.utcnow(),
            is_active=True
        )
        
        db_session.add(agent)
        db_session.commit()
        
        assert agent.id == "test-agent-id"
        assert agent.name == "Test Agent"
        assert agent.is_active is True
    
    def test_agent_capability_relationship(self, db_session):
        """Test agent-capability relationship"""
        agent = Agent(
            id="test-agent-id",
            name="Test Agent",
            type="llm",
            endpoint="http://localhost:8000",
            status="active",
            created_at=datetime.utcnow(),
            is_active=True
        )
        
        capability = AgentCapability(
            agent_id="test-agent-id",
            capability="general",
            confidence_score=0.9,
            created_at=datetime.utcnow(),
            is_active=True
        )
        
        db_session.add(agent)
        db_session.add(capability)
        db_session.commit()
        
        assert capability.agent_id == agent.id


class TestUserModel:
    """Test cases for User model"""
    
    def test_user_creation(self, db_session):
        """Test creating a user"""
        user = User(
            username="testuser",
            email="test@example.com",
            password_hash="hashed_password",
            role="user",
            created_at=datetime.utcnow(),
            is_active=True
        )
        
        db_session.add(user)
        db_session.commit()
        
        assert user.id is not None
        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert user.is_active is True


class TestCacheModel:
    """Test cases for Cache model"""
    
    def test_cache_entry_creation(self, db_session):
        """Test creating a cache entry"""
        cache_entry = AICacheEntry(
            cache_key="test:key",
            query_hash="abc123",
            model_id="test-model",
            provider="test-provider",
            query_text="test query",
            response_data={"response": "test"},
            ttl_seconds=3600,
            expires_at=datetime.utcnow()
        )
        
        db_session.add(cache_entry)
        db_session.commit()
        
        assert cache_entry.id is not None
        assert cache_entry.cache_key == "test:key"
        assert cache_entry.response_data == {"response": "test"}


class TestRAGModel:
    """Test cases for RAG model"""
    
    def test_rag_document_creation(self, db_session):
        """Test creating a RAG document"""
        document = Document(
            title="test.txt",
            filename="test.txt",
            file_hash="abc123",
            content="Test document content",
            content_type="txt",
            file_size=100,
            document_metadata={"test": True},
            word_count=3,
            character_count=20
        )
        
        db_session.add(document)
        db_session.commit()
        
        assert document.id is not None
        assert document.filename == "test.txt"
        assert document.content == "Test document content"
    
    def test_rag_chunk_relationship(self, db_session):
        """Test RAG document-chunk relationship"""
        document = Document(
            title="test.txt",
            filename="test.txt",
            file_hash="abc123",
            content="Test document content",
            content_type="txt",
            file_size=100,
            document_metadata={"test": True},
            word_count=3,
            character_count=20
        )
        
        db_session.add(document)
        db_session.flush()
        
        chunk = DocumentChunk(
            document_id=document.id,
            chunk_index=0,
            content="Test chunk content",
            chunk_hash="chunk123",
            start_char=0,
            end_char=18,
            word_count=3
        )
        
        db_session.add(chunk)
        db_session.commit()
        
        assert chunk.document_id == document.id
        assert chunk.chunk_index == 0
