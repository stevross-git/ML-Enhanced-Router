#!/usr/bin/env python3
"""
Simple Security Fixes Verification
"""

import os

def check_fix(file_path, search_strings, fix_name):
    """Check if a fix is present in a file"""
    if not os.path.exists(file_path):
        print(f"[SKIP] {fix_name}: File {file_path} not found")
        return False
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if isinstance(search_strings, str):
            search_strings = [search_strings]
        
        found = all(s in content for s in search_strings)
        
        if found:
            print(f"[PASS] {fix_name}")
            return True
        else:
            print(f"[FAIL] {fix_name}")
            return False
            
    except Exception as e:
        print(f"[ERROR] {fix_name}: {e}")
        return False

def main():
    print("Security Fixes Verification")
    print("=" * 40)
    
    tests = [
        (
            "app/utils/decorators.py",
            "Authentication is required but disabled in configuration",
            "Authentication Bypass Fix"
        ),
        (
            "config/base.py", 
            ["must be set in production", "sys.exit(1)"],
            "Secret Key Enforcement"
        ),
        (
            "app/services/cache_service.py",
            ["escaped_pattern", "escape='\\\\"],
            "SQL Injection Prevention"
        ),
        (
            "app/middleware/security.py",
            "Content-Security-Policy",
            "CSP Headers Middleware"
        ),
        (
            "app/utils/exceptions.py",
            "generic_messages",
            "Error Sanitization"
        ),
        (
            "app/utils/validators.py",
            ["weak_patterns", "enforce_password_policy"],
            "Enhanced Password Policy"
        ),
        (
            "Dockerfile",
            "USER appuser",
            "Docker Non-root User"
        ),
        (
            ".dockerignore",
            ".env",
            "Docker Ignore File"
        )
    ]
    
    passed = 0
    total = len(tests)
    
    for file_path, search_strings, fix_name in tests:
        if check_fix(file_path, search_strings, fix_name):
            passed += 1
    
    print("\n" + "=" * 40)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("All security fixes verified successfully!")
    elif passed >= total * 0.8:
        print("Most security fixes verified successfully!")
    else:
        print("Some security fixes need attention.")

if __name__ == "__main__":
    main()