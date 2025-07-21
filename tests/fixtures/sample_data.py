"""
Sample data fixtures for testing
"""

from datetime import datetime, timedelta


def sample_user_data():
    """Sample user data for testing"""
    return {
        'username': 'testuser',
        'email': 'test@example.com',
        'password': 'testpassword123',
        'role': 'user',
        'is_active': True,
        'created_at': datetime.utcnow()
    }


def sample_admin_user_data():
    """Sample admin user data for testing"""
    return {
        'username': 'admin',
        'email': 'admin@example.com',
        'password': 'adminpassword123',
        'role': 'admin',
        'is_active': True,
        'created_at': datetime.utcnow()
    }


def sample_agent_data():
    """Sample agent data for testing"""
    return {
        'name': 'Test Agent',
        'type': 'llm',
        'endpoint': 'http://localhost:8000/api/v1',
        'capabilities': ['general', 'analysis', 'generation'],
        'description': 'Test agent for unit testing',
        'version': '1.0.0',
        'metadata': {
            'test': True,
            'model': 'gpt-3.5-turbo',
            'provider': 'openai'
        },
        'max_concurrent_sessions': 5,
        'confidence_scores': {
            'general': 0.9,
            'analysis': 0.8,
            'generation': 0.85
        }
    }


def sample_query_data():
    """Sample query data for testing"""
    return {
        'query': 'What is the weather like today?',
        'context': {
            'user_id': 'test-user',
            'session_id': 'test-session',
            'timestamp': datetime.utcnow().isoformat()
        },
        'capabilities': ['general', 'search'],
        'metadata': {
            'source': 'web_interface',
            'priority': 'normal'
        }
    }


def sample_rag_document_data():
    """Sample RAG document data for testing"""
    return {
        'filename': 'test_document.txt',
        'file_path': '/tmp/test_document.txt',
        'content': '''
        This is a test document for RAG testing.
        It contains multiple paragraphs with different topics.
        
        The first topic is about machine learning and artificial intelligence.
        Machine learning is a subset of AI that focuses on algorithms that can learn from data.
        
        The second topic is about natural language processing.
        NLP is a field that combines linguistics and computer science.
        
        The third topic is about information retrieval.
        Information retrieval systems help users find relevant information from large collections.
        ''',
        'metadata': {
            'author': 'Test Author',
            'category': 'technical',
            'tags': ['ml', 'ai', 'nlp', 'ir'],
            'created_date': '2024-01-01'
        }
    }


def sample_email_data():
    """Sample email data for testing"""
    return {
        'to_addresses': ['recipient@example.com'],
        'subject': 'Test Email Subject',
        'body': '''
        Dear Recipient,
        
        This is a test email for the email service testing.
        Please ignore this message as it is generated for testing purposes.
        
        Best regards,
        Test System
        ''',
        'html_body': '''
        <html>
        <body>
        <h2>Test Email</h2>
        <p>Dear Recipient,</p>
        <p>This is a test email for the email service testing.</p>
        <p>Please ignore this message as it is generated for testing purposes.</p>
        <p>Best regards,<br>Test System</p>
        </body>
        </html>
        ''',
        'from_address': 'test@example.com'
    }


def sample_cache_data():
    """Sample cache data for testing"""
    return [
        {
            'key': 'user:123:profile',
            'value': {'name': 'John Doe', 'email': 'john@example.com'},
            'ttl': 3600
        },
        {
            'key': 'query:abc123:result',
            'value': {'response': 'Test response', 'confidence': 0.9},
            'ttl': 1800
        },
        {
            'key': 'model:gpt-3.5:config',
            'value': {'temperature': 0.7, 'max_tokens': 1000},
            'ttl': 7200
        }
    ]


def sample_training_consent_email():
    """Sample training consent email for testing"""
    return {
        'sender_email': 'user@example.com',
        'subject': 'Training Data Consent',
        'content': '''
        Dear ML Router Team,
        
        I hereby give my consent for you to use my query data and interactions
        for training and improving your machine learning models.
        
        I understand that this data will be used to enhance the system's
        performance and provide better responses to users.
        
        Please confirm receipt of this consent.
        
        Best regards,
        John Doe
        '''
    }


def sample_ml_router_config():
    """Sample ML router configuration for testing"""
    return {
        'default_model': 'gpt-3.5-turbo',
        'fallback_model': 'gpt-3.5-turbo-instruct',
        'max_tokens': 1000,
        'temperature': 0.7,
        'timeout': 30,
        'retry_attempts': 3,
        'load_balancing': 'round_robin',
        'cache_enabled': True,
        'cache_ttl': 3600,
        'rate_limiting': {
            'enabled': True,
            'requests_per_minute': 60,
            'burst_limit': 10
        }
    }


def sample_agent_metrics():
    """Sample agent metrics for testing"""
    return {
        'total_requests': 1000,
        'successful_requests': 950,
        'failed_requests': 50,
        'average_response_time': 1.5,
        'success_rate': 95.0,
        'last_24h_requests': 100,
        'peak_concurrent_sessions': 8,
        'uptime_percentage': 99.5
    }
