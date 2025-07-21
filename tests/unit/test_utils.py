"""
Unit tests for utility functions
"""

import pytest
from unittest.mock import Mock, patch

from app.utils.decorators import rate_limit, require_auth
from app.utils.validators import validate_email, validate_query
from app.utils.exceptions import ValidationError, AuthenticationError
from app.utils.helpers import format_response, parse_query_params


class TestDecorators:
    """Test cases for decorators"""
    
    def test_rate_limit_decorator(self, app, client):
        """Test rate limiting decorator"""
        with app.app_context():
            @rate_limit(limit=2, per=60)
            def test_endpoint():
                return "success"
            
            assert callable(test_endpoint)
    
    def test_require_auth_decorator(self, app):
        """Test authentication requirement decorator"""
        with app.app_context():
            @require_auth
            def test_endpoint():
                return "success"
            
            assert callable(test_endpoint)


class TestValidators:
    """Test cases for validators"""
    
    def test_validate_email(self):
        """Test email validation"""
        assert validate_email("test@example.com") is True
        assert validate_email("user.name+tag@domain.co.uk") is True
        assert validate_email("invalid-email") is False
        assert validate_email("@domain.com") is False
        assert validate_email("user@") is False
    
    def test_validate_query(self):
        """Test query validation"""
        assert validate_query("What is the weather?") is True
        assert validate_query("") is False
        assert validate_query(None) is False
        assert validate_query("a" * 10000) is False


class TestHelpers:
    """Test cases for helper functions"""
    
    def test_format_response(self):
        """Test response formatting"""
        data = {"message": "success"}
        formatted = format_response(data, status="success")
        
        assert formatted["status"] == "success"
        assert formatted["data"] == data
        assert "timestamp" in formatted
    
    def test_parse_query_params(self):
        """Test query parameter parsing"""
        params = "limit=10&offset=20&sort=name"
        parsed = parse_query_params(params)
        
        assert parsed["limit"] == "10"
        assert parsed["offset"] == "20"
        assert parsed["sort"] == "name"


class TestExceptions:
    """Test cases for custom exceptions"""
    
    def test_validation_error(self):
        """Test ValidationError exception"""
        with pytest.raises(ValidationError) as exc_info:
            raise ValidationError("Invalid input")
        
        assert str(exc_info.value) == "Invalid input"
    
    def test_authentication_error(self):
        """Test AuthenticationError exception"""
        with pytest.raises(AuthenticationError) as exc_info:
            raise AuthenticationError("Authentication failed")
        
        assert str(exc_info.value) == "Authentication failed"
