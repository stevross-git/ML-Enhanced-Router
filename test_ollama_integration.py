#!/usr/bin/env python3
"""
Test script for Ollama integration with streaming support
Run this script to test both streaming and non-streaming modes
"""

import asyncio
import json
import requests
import time
from typing import Dict, Any

# Configuration
BASE_URL = "http://localhost:5000"  # Your Flask app URL
OLLAMA_MODEL_ID = "ollama-llama3.1"

class OllamaIntegrationTester:
    """Test class for Ollama integration"""
    
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()
    
    def test_model_availability(self) -> bool:
        """Test if Ollama model is available"""
        try:
            response = self.session.get(f"{self.base_url}/api/ai-models")
            if response.status_code == 200:
                models = response.json().get('models', [])
                ollama_model = next((m for m in models if m['id'] == OLLAMA_MODEL_ID), None)
                if ollama_model:
                    print(f"✅ Ollama model found: {ollama_model['name']}")
                    print(f"   Model name: {ollama_model['model_name']}")
                    print(f"   Endpoint: {ollama_model['endpoint']}")
                    print(f"   Supports streaming: {ollama_model['supports_streaming']}")
                    return True
                else:
                    print(f"❌ Ollama model {OLLAMA_MODEL_ID} not found")
                    return False
            else:
                print(f"❌ Failed to get models: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Error checking model availability: {e}")
            return False
    
    def test_non_streaming(self) -> Dict[str, Any]:
        """Test non-streaming mode (for testing endpoint)"""
        print("\n🧪 Testing Non-Streaming Mode...")
        
        payload = {
            "query": "Hello! Can you confirm you are working correctly? Please respond with 'Test successful' if everything is working.",
            "system_message": "You are a helpful assistant. Respond briefly and clearly."
        }
        
        start_time = time.time()
        
        try:
            response = self.session.post(
                f"{self.base_url}/api/ai-models/test/{OLLAMA_MODEL_ID}",
                json=payload,
                timeout=30
            )
            
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Non-streaming test successful!")
                print(f"   Status: {data['status']}")
                print(f"   Response: {data['response'][:100]}...")
                print(f"   Response time: {response_time:.2f}s")
                print(f"   Model: {data['model']}")
                print(f"   Streaming: {data.get('streaming', 'N/A')}")
                print(f"   Cached: {data.get('cached', 'N/A')}")
                if 'usage' in data:
                    print(f"   Usage: {data['usage']}")
                return data
            else:
                error_data = response.json()
                print(f"❌ Non-streaming test failed: {error_data.get('error', 'Unknown error')}")
                return {"status": "error", "error": error_data.get('error')}
                
        except Exception as e:
            print(f"❌ Non-streaming test error: {e}")
            return {"status": "error", "error": str(e)}
    
    def test_streaming_via_chat(self) -> Dict[str, Any]:
        """Test streaming mode via chat endpoint"""
        print("\n🚀 Testing Streaming Mode via Chat...")
        
        payload = {
            "message": "Write a short story about a robot learning to paint. Keep it under 200 words.",
            "model_id": OLLAMA_MODEL_ID,
            "system_message": "You are a creative writing assistant. Write engaging short stories.",
            "enable_streaming": True
        }
        
        start_time = time.time()
        
        try:
            response = self.session.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=60
            )
            
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Streaming test successful!")
                print(f"   Status: {data['status']}")
                print(f"   Response length: {len(data['response'])} characters")
                print(f"   Response preview: {data['response'][:150]}...")
                print(f"   Response time: {response_time:.2f}s")
                print(f"   Model: {data['model']}")
                print(f"   Streaming: {data.get('streaming', 'N/A')}")
                print(f"   Cached: {data.get('cached', 'N/A')}")
                if 'usage' in data:
                    print(f"   Usage: {data['usage']}")
                return data
            else:
                error_data = response.json()
                print(f"❌ Streaming test failed: {error_data.get('error', 'Unknown error')}")
                return {"status": "error", "error": error_data.get('error')}
                
        except Exception as e:
            print(f"❌ Streaming test error: {e}")
            return {"status": "error", "error": str(e)}
    
    def test_non_streaming_via_chat(self) -> Dict[str, Any]:
        """Test non-streaming mode via chat endpoint"""
        print("\n📝 Testing Non-Streaming Mode via Chat...")
        
        payload = {
            "message": "What is the capital of France?",
            "model_id": OLLAMA_MODEL_ID,
            "enable_streaming": False
        }
        
        start_time = time.time()
        
        try:
            response = self.session.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=30
            )
            
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Non-streaming chat test successful!")
                print(f"   Status: {data['status']}")
                print(f"   Response: {data['response']}")
                print(f"   Response time: {response_time:.2f}s")
                print(f"   Model: {data['model']}")
                print(f"   Streaming: {data.get('streaming', 'N/A')}")
                print(f"   Cached: {data.get('cached', 'N/A')}")
                return data
            else:
                error_data = response.json()
                print(f"❌ Non-streaming chat test failed: {error_data.get('error', 'Unknown error')}")
                return {"status": "error", "error": error_data.get('error')}
                
        except Exception as e:
            print(f"❌ Non-streaming chat test error: {e}")
            return {"status": "error", "error": str(e)}
    
    def test_streaming_specific_endpoint(self) -> Dict[str, Any]:
        """Test the streaming-specific test endpoint"""
        print("\n⚡ Testing Streaming-Specific Endpoint...")
        
        payload = {
            "query": "Count from 1 to 5 and explain each number.",
            "system_message": "You are a helpful teacher. Explain things clearly."
        }
        
        start_time = time.time()
        
        try:
            response = self.session.post(
                f"{self.base_url}/api/ai-models/test-streaming/{OLLAMA_MODEL_ID}",
                json=payload,
                timeout=30
            )
            
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Streaming-specific test successful!")
                print(f"   Status: {data['status']}")
                print(f"   Response: {data['response'][:200]}...")
                print(f"   Response time: {response_time:.2f}s")
                print(f"   Model: {data['model']}")
                print(f"   Test type: {data.get('test_type', 'N/A')}")
                print(f"   Streaming: {data.get('streaming', 'N/A')}")
                return data
            else:
                error_data = response.json()
                print(f"❌ Streaming-specific test failed: {error_data.get('error', 'Unknown error')}")
                return {"status": "error", "error": error_data.get('error')}
                
        except Exception as e:
            print(f"❌ Streaming-specific test error: {e}")
            return {"status": "error", "error": str(e)}
    
    def test_model_capabilities(self) -> Dict[str, Any]:
        """Test model capabilities endpoint"""
        print("\n🔍 Testing Model Capabilities...")
        
        try:
            response = self.session.get(
                f"{self.base_url}/api/ai-models/{OLLAMA_MODEL_ID}/capabilities"
            )
            
            if response.status_code == 200:
                data = response.json()
                capabilities = data['capabilities']
                print(f"✅ Model capabilities retrieved!")
                print(f"   Name: {capabilities['name']}")
                print(f"   Provider: {capabilities['provider']}")
                print(f"   Model Type: {capabilities['model_type']}")
                print(f"   Deployment Type: {capabilities['deployment_type']}")
                print(f"   Supports Streaming: {capabilities['supports_streaming']}")
                print(f"   Supports System Message: {capabilities['supports_system_message']}")
                print(f"   Context Window: {capabilities['context_window']}")
                print(f"   Max Tokens: {capabilities['max_tokens']}")
                print(f"   Input Modalities: {capabilities['input_modalities']}")
                print(f"   Output Modalities: {capabilities['output_modalities']}")
                print(f"   Capabilities: {capabilities['capabilities']}")
                return data
            else:
                error_data = response.json()
                print(f"❌ Capabilities test failed: {error_data.get('error', 'Unknown error')}")
                return {"status": "error", "error": error_data.get('error')}
                
        except Exception as e:
            print(f"❌ Capabilities test error: {e}")
            return {"status": "error", "error": str(e)}
    
    def test_error_handling(self) -> None:
        """Test error handling scenarios"""
        print("\n🚨 Testing Error Handling...")
        
        # Test with invalid model
        print("   Testing invalid model...")
        try:
            response = self.session.post(
                f"{self.base_url}/api/ai-models/test/invalid-model",
                json={"query": "Hello"},
                timeout=10
            )
            if response.status_code == 404:
                print("   ✅ Invalid model properly rejected")
            else:
                print(f"   ❌ Unexpected response for invalid model: {response.status_code}")
        except Exception as e:
            print(f"   ❌ Error testing invalid model: {e}")
        
        # Test with empty query
        print("   Testing empty query...")
        try:
            response = self.session.post(
                f"{self.base_url}/api/chat",
                json={},
                timeout=10
            )
            if response.status_code == 400:
                print("   ✅ Empty query properly rejected")
            else:
                print(f"   ❌ Unexpected response for empty query: {response.status_code}")
        except Exception as e:
            print(f"   ❌ Error testing empty query: {e}")
    
    def run_all_tests(self) -> None:
        """Run all tests"""
        print("🔧 Ollama Integration Test Suite")
        print("=" * 50)
        
        # Check if model is available
        if not self.test_model_availability():
            print("\n❌ Cannot proceed with tests - Ollama model not available")
            return
        
        # Run all tests
        tests = [
            self.test_model_capabilities,
            self.test_non_streaming,
            self.test_non_streaming_via_chat,
            self.test_streaming_via_chat,
            self.test_streaming_specific_endpoint,
            self.test_error_handling
        ]
        
        results = []
        for test in tests:
            try:
                result = test()
                if result and isinstance(result, dict):
                    results.append(result)
            except Exception as e:
                print(f"❌ Test {test.__name__} failed with exception: {e}")
        
        # Summary
        print("\n" + "=" * 50)
        print("📊 Test Summary")
        print("=" * 50)
        
        successful_tests = sum(1 for r in results if r.get('status') == 'success')
        total_tests = len([t for t in tests if not t.__name__ == 'test_error_handling'])  # Error handling doesn't return results
        
        print(f"✅ Successful tests: {successful_tests}/{total_tests}")
        
        if successful_tests == total_tests:
            print("🎉 All tests passed! Ollama integration is working correctly.")
        else:
            print("⚠️  Some tests failed. Check the output above for details.")
        
        # Performance summary
        response_times = [r.get('response_time', 0) for r in results if 'response_time' in r]
        if response_times:
            avg_time = sum(response_times) / len(response_times)
            print(f"⏱️  Average response time: {avg_time:.2f}s")


def main():
    """Main function to run tests"""
    tester = OllamaIntegrationTester()
    tester.run_all_tests()


# Direct test functions for manual testing
def test_ollama_direct():
    """Direct test of Ollama API (bypassing Flask app)"""
    print("🔧 Direct Ollama API Test")
    print("=" * 30)
    
    import aiohttp
    
    async def direct_test():
        url = "http://localhost:11434/api/chat"
        headers = {"Content-Type": "application/json"}
        
        payload = {
            "model": "llama3.1:latest",
            "messages": [
                {"role": "user", "content": "Hello! Please respond with 'Direct test successful'."}
            ],
            "stream": False
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        print(f"✅ Direct Ollama test successful!")
                        print(f"   Response: {data.get('message', {}).get('content', 'No content')}")
                        print(f"   Model: {data.get('model', 'Unknown')}")
                        print(f"   Done: {data.get('done', 'Unknown')}")
                    else:
                        print(f"❌ Direct Ollama test failed: HTTP {response.status}")
                        error_data = await response.text()
                        print(f"   Error: {error_data}")
        except Exception as e:
            print(f"❌ Direct Ollama test error: {e}")
    
    asyncio.run(direct_test())


if __name__ == "__main__":
    print("Choose test mode:")
    print("1. Full integration test (via Flask app)")
    print("2. Direct Ollama API test")
    print("3. Both")
    
    choice = input("Enter choice (1/2/3): ").strip()
    
    if choice == "1":
        main()
    elif choice == "2":
        test_ollama_direct()
    elif choice == "3":
        test_ollama_direct()
        print("\n" + "="*60 + "\n")
        main()
    else:
        print("Invalid choice. Running full integration test...")
        main()