"""
Unit tests for service layer
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from app.services.auth_service import AuthService, get_auth_service
from app.services.cache_service import CacheService, get_cache_manager
from app.services.rag_service import RAGService, get_rag_service
from app.services.agent_service import AgentService, get_agent_service
from app.services.email_service import EmailService, get_email_service
from app.utils.exceptions import AuthenticationError, ValidationError, EmailError


class TestAuthService:
    """Test cases for AuthService"""
    
    def test_auth_service_initialization(self, app):
        """Test auth service initialization"""
        with app.app_context():
            service = AuthService()
            service.initialize(app)
            assert service.jwt_secret is not None
            assert service.jwt_algorithm == 'HS256'
    
    def test_get_auth_service_singleton(self, app):
        """Test auth service singleton pattern"""
        with app.app_context():
            service1 = get_auth_service()
            service2 = get_auth_service()
            assert service1 is service2
    
    def test_password_hashing(self, app):
        """Test password hashing and verification"""
        with app.app_context():
            service = AuthService()
            service.initialize(app)
            
            password = "testpassword123"
            hashed = service._hash_password(password)
            
            assert hashed != password
            assert service._verify_password(password, hashed)
            assert not service._verify_password("wrongpassword", hashed)
    
    def test_jwt_token_generation(self, app):
        """Test JWT token generation and validation"""
        with app.app_context():
            service = AuthService()
            service.initialize(app)
            
            user_id = 123
            token = service.generate_jwt_token(user_id)
            
            assert token is not None
            assert isinstance(token, str)


class TestCacheService:
    """Test cases for CacheService"""
    
    def test_cache_service_initialization(self, app, mock_redis):
        """Test cache service initialization"""
        with app.app_context():
            service = CacheService()
            service.initialize(app)
            assert service.enabled
            assert service.default_ttl == 3600
    
    def test_cache_set_get(self, app, mock_redis):
        """Test cache set and get operations"""
        with app.app_context():
            service = CacheService()
            service.initialize(app)
            
            key = "test_key"
            value = {"test": "data"}
            
            result = service.set(key, value)
            assert result is True
            
            retrieved = service.get(key)
            assert retrieved == value
    
    def test_cache_delete(self, app, mock_redis):
        """Test cache delete operation"""
        with app.app_context():
            service = CacheService()
            service.initialize(app)
            
            key = "test_key"
            value = "test_value"
            
            service.set(key, value)
            assert service.get(key) == value
            
            service.delete(key)
            assert service.get(key) is None
    
    def test_cache_decorator(self, app, mock_redis):
        """Test cache decorator functionality"""
        with app.app_context():
            service = CacheService()
            service.initialize(app)
            
            call_count = 0
            
            @service.cached(ttl=60)
            def expensive_function(x):
                nonlocal call_count
                call_count += 1
                return x * 2
            
            result1 = expensive_function(5)
            result2 = expensive_function(5)
            
            assert result1 == 10
            assert result2 == 10
            assert call_count == 1


class TestRAGService:
    """Test cases for RAGService"""
    
    def test_rag_service_initialization(self, app):
        """Test RAG service initialization"""
        with app.app_context():
            service = RAGService()
            service.initialize(app)
            assert service.chunk_size == 1000
            assert service.chunk_overlap == 200
    
    def test_document_chunking(self, app):
        """Test document chunking functionality"""
        with app.app_context():
            service = RAGService()
            service.initialize(app)
            
            content = " ".join(["word"] * 1500)
            chunks = service._chunk_document(content)
            
            assert len(chunks) > 1
            assert all(len(chunk.split()) <= service.chunk_size for chunk in chunks)
    
    def test_embeddings_generation(self, app):
        """Test embeddings generation"""
        with app.app_context():
            service = RAGService()
            service.initialize(app)
            
            text = "This is a test document for embeddings."
            embeddings = service._generate_embeddings(text)
            
            assert isinstance(embeddings, list)
            assert len(embeddings) > 0
            assert all(isinstance(x, float) for x in embeddings)


class TestAgentService:
    """Test cases for AgentService"""
    
    def test_agent_service_initialization(self, app):
        """Test agent service initialization"""
        with app.app_context():
            service = AgentService()
            service.initialize(app)
            assert service.load_balancing_strategy == "round_robin"
            assert service.max_concurrent_sessions == 10
    
    def test_query_capability_analysis(self, app):
        """Test query capability analysis"""
        with app.app_context():
            service = AgentService()
            service.initialize(app)
            
            query = "Please search for information about machine learning"
            capabilities = service._analyze_query_capabilities(query)
            
            assert "search" in capabilities
    
    def test_routing_score_calculation(self, app):
        """Test routing score calculation"""
        with app.app_context():
            service = AgentService()
            service.initialize(app)
            
            agent = {
                'confidence_score': 0.9,
                'average_response_time': 1.0,
                'success_rate': 95.0,
                'active_sessions': 2,
                'max_sessions': 10
            }
            
            score = service._calculate_routing_score(agent, "test query")
            
            assert 0 <= score <= 1
            assert isinstance(score, float)


class TestEmailService:
    """Test cases for EmailService"""
    
    def test_email_service_initialization(self, app):
        """Test email service initialization"""
        with app.app_context():
            service = EmailService()
            service.initialize(app)
            assert service.use_tls is True
            assert service.smtp_port == 587
    
    def test_email_validation(self, app):
        """Test email address validation"""
        with app.app_context():
            service = EmailService()
            service.initialize(app)
            
            assert service._is_valid_email("test@example.com")
            assert service._is_valid_email("user.name+tag@domain.co.uk")
            assert not service._is_valid_email("invalid-email")
            assert not service._is_valid_email("@domain.com")
            assert not service._is_valid_email("user@")
    
    def test_sentiment_analysis(self, app):
        """Test sentiment analysis"""
        with app.app_context():
            service = EmailService()
            service.initialize(app)
            
            positive_text = "This is great and excellent work!"
            negative_text = "This is terrible and awful."
            neutral_text = "This is a normal message."
            
            positive_result = service._analyze_sentiment(positive_text)
            negative_result = service._analyze_sentiment(negative_text)
            neutral_result = service._analyze_sentiment(neutral_text)
            
            assert positive_result['sentiment'] == 'positive'
            assert negative_result['sentiment'] == 'negative'
            assert neutral_result['sentiment'] == 'neutral'
    
    def test_entity_extraction(self, app):
        """Test entity extraction"""
        with app.app_context():
            service = EmailService()
            service.initialize(app)
            
            text = "Contact me at test@example.com or call 555-123-4567"
            entities = service._extract_entities(text)
            
            email_entities = [e for e in entities if e['type'] == 'email']
            phone_entities = [e for e in entities if e['type'] == 'phone']
            
            assert len(email_entities) > 0
            assert email_entities[0]['value'] == 'test@example.com'
    
    def test_training_consent_processing(self, app):
        """Test training consent processing"""
        with app.app_context():
            service = EmailService()
            service.initialize(app)
            
            consent_text = "I consent to use my data for training the model"
            no_consent_text = "This is just a regular email"
            
            consent_result = service.process_training_consent(consent_text, "user@example.com")
            no_consent_result = service.process_training_consent(no_consent_text, "user@example.com")
            
            assert consent_result['consent_given'] is True
            assert consent_result['consent_type'] == 'training'
            assert no_consent_result['consent_given'] is False
