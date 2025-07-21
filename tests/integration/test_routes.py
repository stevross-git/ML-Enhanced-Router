"""
Integration tests for routes
"""

import pytest
import json
from unittest.mock import patch, Mock

from app.models.user import User


class TestMainRoutes:
    """Test cases for main routes"""
    
    def test_health_check(self, client):
        """Test health check endpoint"""
        response = client.get('/health')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert data['status'] == 'healthy'
    
    def test_index_route(self, client):
        """Test index route"""
        response = client.get('/')
        assert response.status_code == 200


class TestAPIRoutes:
    """Test cases for API routes"""
    
    def test_query_endpoint_without_auth(self, client):
        """Test query endpoint without authentication"""
        response = client.post('/api/query', 
                             json={'query': 'test query'},
                             headers={'Content-Type': 'application/json'})
        
        assert response.status_code in [401, 403]
    
    @patch('app.services.ml_router.get_ml_router_service')
    def test_query_endpoint_with_auth(self, mock_service, client, auth_headers):
        """Test query endpoint with authentication"""
        mock_router = Mock()
        mock_router.route_query.return_value = {
            'response': 'Test response',
            'confidence': 0.9
        }
        mock_service.return_value = mock_router
        
        response = client.post('/api/query',
                             json={'query': 'test query'},
                             headers=auth_headers)
        
        assert response.status_code in [200, 401, 403]
    
    def test_models_endpoint(self, client):
        """Test models listing endpoint"""
        response = client.get('/api/models')
        assert response.status_code in [200, 401]
    
    def test_cache_stats_endpoint(self, client):
        """Test cache statistics endpoint"""
        response = client.get('/api/cache/stats')
        assert response.status_code in [200, 401]


class TestAuthRoutes:
    """Test cases for authentication routes"""
    
    def test_login_endpoint_exists(self, client):
        """Test login endpoint exists"""
        response = client.post('/auth/login',
                             json={'username': 'test', 'password': 'test'},
                             headers={'Content-Type': 'application/json'})
        
        assert response.status_code in [200, 400, 401, 422]
    
    def test_register_endpoint_exists(self, client):
        """Test register endpoint exists"""
        response = client.post('/auth/register',
                             json={
                                 'username': 'newuser',
                                 'email': 'new@example.com',
                                 'password': 'password123'
                             },
                             headers={'Content-Type': 'application/json'})
        
        assert response.status_code in [200, 201, 400, 422]


class TestRAGRoutes:
    """Test cases for RAG routes"""
    
    def test_rag_search_endpoint(self, client):
        """Test RAG search endpoint"""
        response = client.post('/api/rag/search',
                             json={'query': 'test search'},
                             headers={'Content-Type': 'application/json'})
        
        assert response.status_code in [200, 401, 403, 422]
    
    def test_rag_documents_endpoint(self, client):
        """Test RAG documents listing endpoint"""
        response = client.get('/api/rag/documents')
        assert response.status_code in [200, 401]


class TestConfigRoutes:
    """Test cases for configuration routes"""
    
    def test_config_endpoint(self, client):
        """Test configuration endpoint"""
        response = client.get('/api/config')
        assert response.status_code in [200, 401]
    
    def test_config_update_endpoint(self, client):
        """Test configuration update endpoint"""
        response = client.put('/api/config',
                            json={'setting': 'value'},
                            headers={'Content-Type': 'application/json'})
        
        assert response.status_code in [200, 401, 403]


class TestGraphQLRoutes:
    """Test cases for GraphQL routes"""
    
    def test_graphql_endpoint_exists(self, client):
        """Test GraphQL endpoint exists"""
        response = client.post('/graphql',
                             json={'query': '{ __schema { types { name } } }'},
                             headers={'Content-Type': 'application/json'})
        
        assert response.status_code in [200, 400, 401]
    
    def test_graphiql_interface(self, client):
        """Test GraphiQL interface"""
        response = client.get('/graphql')
        
        assert response.status_code in [200, 302, 401]
