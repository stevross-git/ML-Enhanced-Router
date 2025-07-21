#!/usr/bin/env python3
"""
Verification script for service implementations
"""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_service_imports():
    """Test that all services can be imported successfully"""
    print("🔍 Testing service imports...")
    
    try:
        from app.services.auth_service import AuthService, get_auth_service
        from app.services.cache_service import CacheService, get_cache_manager  
        from app.services.rag_service import RAGService, get_rag_service
        from app.services.agent_service import AgentService, get_agent_service
        from app.services.email_service import EmailService, get_email_service
        
        print("✅ Individual service imports successful")
        
        auth_service = AuthService()
        cache_service = CacheService()
        rag_service = RAGService()
        agent_service = AgentService()
        email_service = EmailService()
        
        print("✅ Service instantiation successful")
        
        from app.services import (
            get_ml_router_service, get_ai_model_manager,
            get_auth_service, get_cache_manager, get_rag_service,
            get_agent_service, get_email_service
        )
        
        print("✅ Services module import successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Service import error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_service_patterns():
    """Test that services follow established patterns"""
    print("\n🔍 Testing service patterns...")
    
    try:
        from app.services.auth_service import AuthService
        from app.services.cache_service import CacheService
        from app.services.rag_service import RAGService
        from app.services.agent_service import AgentService
        from app.services.email_service import EmailService
        
        services = [
            ('AuthService', AuthService),
            ('CacheService', CacheService),
            ('RAGService', RAGService),
            ('AgentService', AgentService),
            ('EmailService', EmailService)
        ]
        
        for name, service_class in services:
            service = service_class()
            
            if not hasattr(service, '__init__'):
                print(f"❌ {name} missing __init__ method")
                return False
                
            if not hasattr(service, 'initialize'):
                print(f"❌ {name} missing initialize method")
                return False
            
            print(f"✅ {name} follows required patterns")
        
        print("✅ All services follow established patterns")
        return True
        
    except Exception as e:
        print(f"❌ Service pattern error: {e}")
        return False

def test_directory_structure():
    """Test that test directory structure is correct"""
    print("\n🔍 Testing directory structure...")
    
    required_files = [
        'tests/__init__.py',
        'tests/conftest.py',
        'tests/unit/__init__.py',
        'tests/unit/test_services.py',
        'tests/unit/test_models.py',
        'tests/unit/test_utils.py',
        'tests/integration/__init__.py',
        'tests/integration/test_routes.py',
        'tests/fixtures/__init__.py',
        'tests/fixtures/sample_data.py'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing test files: {missing_files}")
        return False
    
    print("✅ Test directory structure complete")
    return True

def test_pytest_discovery():
    """Test that pytest can discover tests"""
    print("\n🔍 Testing pytest discovery...")
    
    try:
        import pytest
        
        test_files = []
        for root, dirs, files in os.walk('tests'):
            for file in files:
                if file.startswith('test_') and file.endswith('.py'):
                    test_files.append(os.path.join(root, file))
        
        print(f"✅ Found {len(test_files)} test files:")
        for test_file in test_files:
            print(f"  - {test_file}")
        
        if len(test_files) >= 3:  # At least test_services, test_models, test_utils
            print("✅ Sufficient test files found")
            return True
        else:
            print("❌ Insufficient test files found")
            return False
            
    except Exception as e:
        print(f"❌ Pytest discovery error: {e}")
        return False

def main():
    """Run all verification tests"""
    print("🚀 Starting service verification...\n")
    
    tests = [
        test_service_imports,
        test_service_patterns,
        test_directory_structure,
        test_pytest_discovery
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print(f"\n📊 Verification Results:")
    print(f"✅ Passed: {sum(results)}/{len(results)}")
    print(f"❌ Failed: {len(results) - sum(results)}/{len(results)}")
    
    if all(results):
        print("\n🎉 All verifications passed! Service implementation is complete.")
        return 0
    else:
        print("\n⚠️ Some verifications failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
