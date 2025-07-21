"""
Test configuration and fixtures for ML-Enhanced-Router
"""

import os
import pytest
import tempfile
from flask import Flask
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app import create_app
from app.extensions import db
from config.testing import TestingConfig


@pytest.fixture(scope='session')
def app():
    """Create application for the tests."""
    app = create_app('testing')
    
    with app.app_context():
        db.create_all()
        yield app
        db.drop_all()


@pytest.fixture(scope='function')
def client(app):
    """Create a test client for the Flask application."""
    return app.test_client()


@pytest.fixture(scope='function')
def runner(app):
    """Create a test runner for the Flask application's Click commands."""
    return app.test_cli_runner()


@pytest.fixture(scope='function')
def db_session(app):
    """Create a database session for testing."""
    with app.app_context():
        connection = db.engine.connect()
        transaction = connection.begin()
        
        options = dict(bind=connection, binds={})
        session = db.create_scoped_session(options=options)
        
        db.session = session
        
        yield session
        
        transaction.rollback()
        connection.close()
        session.remove()


@pytest.fixture
def auth_headers():
    """Create authentication headers for testing."""
    return {
        'Authorization': 'Bearer test-token',
        'Content-Type': 'application/json'
    }


@pytest.fixture
def sample_user_data():
    """Sample user data for testing."""
    return {
        'username': 'testuser',
        'email': 'test@example.com',
        'password': 'testpassword123',
        'role': 'user'
    }


@pytest.fixture
def sample_agent_data():
    """Sample agent data for testing."""
    return {
        'name': 'Test Agent',
        'type': 'llm',
        'endpoint': 'http://localhost:8000/api/v1',
        'capabilities': ['general', 'analysis'],
        'description': 'Test agent for unit testing',
        'version': '1.0.0',
        'metadata': {'test': True}
    }


@pytest.fixture
def sample_query_data():
    """Sample query data for testing."""
    return {
        'query': 'What is the weather like today?',
        'context': {'user_id': 'test-user'},
        'capabilities': ['general']
    }


@pytest.fixture
def mock_redis(monkeypatch):
    """Mock Redis for testing."""
    class MockRedis:
        def __init__(self):
            self.data = {}
        
        def get(self, key):
            return self.data.get(key)
        
        def set(self, key, value):
            self.data[key] = value
        
        def setex(self, key, ttl, value):
            self.data[key] = value
        
        def delete(self, key):
            self.data.pop(key, None)
        
        def keys(self, pattern):
            return [k for k in self.data.keys() if pattern.replace('*', '') in k]
        
        def ping(self):
            return True
        
        def info(self):
            return {'used_memory': 1024}
    
    mock_redis_instance = MockRedis()
    monkeypatch.setattr('redis.Redis', lambda **kwargs: mock_redis_instance)
    monkeypatch.setattr('redis.from_url', lambda url: mock_redis_instance)
    return mock_redis_instance


@pytest.fixture
def mock_email_server(monkeypatch):
    """Mock email server for testing."""
    class MockSMTP:
        def __init__(self, server, port):
            self.server = server
            self.port = port
        
        def __enter__(self):
            return self
        
        def __exit__(self, *args):
            pass
        
        def starttls(self):
            pass
        
        def login(self, username, password):
            pass
        
        def send_message(self, msg):
            pass
    
    monkeypatch.setattr('smtplib.SMTP', MockSMTP)


@pytest.fixture
def temp_file():
    """Create a temporary file for testing."""
    fd, path = tempfile.mkstemp()
    try:
        with os.fdopen(fd, 'w') as tmp:
            tmp.write('Test file content for RAG testing.')
        yield path
    finally:
        os.unlink(path)
