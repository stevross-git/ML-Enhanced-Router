#!/usr/bin/env python3
"""
Production Key Generator for MLEnhancedRouter
Generates cryptographically secure keys for production deployment
"""

import secrets
import string
import os
from datetime import datetime

def generate_secure_key(length=64, prefix=""):
    """Generate a cryptographically secure key"""
    alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
    key = ''.join(secrets.choice(alphabet) for _ in range(length))
    return f"{prefix}{key}" if prefix else key

def generate_database_password(length=48):
    """Generate a secure database password (no special chars that might conflict)"""
    alphabet = string.ascii_letters + string.digits
    return ''.join(secrets.choice(alphabet) for _ in range(length))

def generate_production_keys():
    """Generate all required production keys"""
    print("MLEnhancedRouter Production Key Generator")
    print("=" * 60)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    keys = {
        # Core Flask secrets (64 chars for maximum security)
        "SECRET_KEY": generate_secure_key(64, "flask_secret_"),
        "JWT_SECRET_KEY": generate_secure_key(64, "jwt_secret_"),
        "SESSION_SECRET": generate_secure_key(64, "session_secret_"),
        
        # Database credentials
        "POSTGRES_USER": "ml_router_prod_" + secrets.token_hex(8),
        "POSTGRES_PASSWORD": generate_database_password(48),
        "POSTGRES_DB": "ml_router_production",
        
        # Admin passwords
        "PGADMIN_PASSWORD": generate_database_password(32),
        
        # Email password placeholder
        "MAIL_PASSWORD": generate_database_password(32),
        
        # OAuth secrets (placeholders - you need to get these from providers)
        "OFFICE365_CLIENT_ID": "your_azure_app_client_id_from_portal",
        "OFFICE365_CLIENT_SECRET": "your_azure_app_client_secret_from_portal",
    }
    
    print("\\nGENERATED PRODUCTION KEYS:")
    print("=" * 60)
    
    for key, value in keys.items():
        if "PASSWORD" in key or "SECRET" in key:
            # Show first 8 chars for verification, rest as asterisks
            display_value = value[:8] + "*" * (len(value) - 8)
            print(f"{key}={display_value}")
        else:
            print(f"{key}={value}")
    
    # Generate .env file content
    env_content = f"""# ========================================
# MLEnhancedRouter Production Environment Configuration
# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# Security Level: Production Ready - DO NOT COMMIT TO VERSION CONTROL
# ========================================

# ========================================
# Flask Application Configuration
# ========================================
FLASK_ENV=production
FLASK_DEBUG=False

# ========================================
# Security Keys (Generated: {datetime.now().strftime('%Y-%m-%d')})
# ========================================
SECRET_KEY={keys['SECRET_KEY']}
JWT_SECRET_KEY={keys['JWT_SECRET_KEY']}
SESSION_SECRET={keys['SESSION_SECRET']}

# ========================================
# Database Configuration
# ========================================
POSTGRES_DB={keys['POSTGRES_DB']}
POSTGRES_USER={keys['POSTGRES_USER']}
POSTGRES_PASSWORD={keys['POSTGRES_PASSWORD']}
DATABASE_URL=postgresql://${{POSTGRES_USER}}:${{POSTGRES_PASSWORD}}@db:5432/${{POSTGRES_DB}}

# ========================================
# Admin Panel Configuration
# ========================================
PGADMIN_DEFAULT_EMAIL=admin@yourdomain.com
PGADMIN_DEFAULT_PASSWORD={keys['PGADMIN_PASSWORD']}
PGADMIN_PASSWORD=${{PGADMIN_DEFAULT_PASSWORD}}

# ========================================
# Security & CORS Configuration
# CRITICAL: Update with your actual production domains
# ========================================
CORS_ORIGINS=["https://yourdomain.com","https://api.yourdomain.com","https://admin.yourdomain.com"]
AUTH_ENABLED=true

# ========================================
# OAuth Configuration (Update with your production values)
# ========================================
OFFICE365_CLIENT_ID={keys['OFFICE365_CLIENT_ID']}
OFFICE365_CLIENT_SECRET={keys['OFFICE365_CLIENT_SECRET']}
OFFICE365_REDIRECT_URI=https://yourdomain.com/auth/office365/callback

# ========================================
# AI Provider API Keys
# CRITICAL: Add your production API keys here
# ========================================
OPENAI_API_KEY=your_openai_production_key_here
ANTHROPIC_API_KEY=your_anthropic_production_key_here
GOOGLE_API_KEY=your_google_production_key_here
XAI_API_KEY=your_xai_production_key_here
PERPLEXITY_API_KEY=your_perplexity_production_key_here
COHERE_API_KEY=your_cohere_production_key_here
MISTRAL_API_KEY=your_mistral_production_key_here
HUGGINGFACE_API_KEY=your_huggingface_production_key_here

# ========================================
# Caching & Performance
# ========================================
REDIS_URL=redis://redis:6379/0
CACHE_TTL=3600
RATE_LIMIT_STORAGE_URL=redis://redis:6379/1

# ========================================
# ML Router Configuration
# ========================================
ROUTER_CACHE_TTL=3600
ROUTER_CACHE_SIZE=10000
ROUTER_MAX_CONCURRENT=50
ROUTER_RATE_LIMIT=1000
ROUTER_GLOBAL_RATE_LIMIT=5000
ROUTER_USE_ML=true
ROUTER_AUTH_ENABLED=true
ROUTER_CONFIDENCE_THRESHOLD=0.7
ROUTER_CONSENSUS_THRESHOLD=0.8
ROUTER_ML_FALLBACK_THRESHOLD=0.7
ROUTER_MAX_AGENTS=5
ROUTER_MAX_RETRIES=3
ROUTER_RETRY_DELAY=1.0
ROUTER_LOAD_PENALTY=0.2
ROUTER_AGENT_TIMEOUT=10.0
ROUTER_USE_REDIS=true
ROUTER_ML_MODEL_PATH=./models/query_classifier
ROUTER_SIMILARITY_MODEL=all-MiniLM-L6-v2
ROUTER_SERVICE_REGISTRY=http://localhost:8500/v1/agent/services
ROUTER_DISCOVERY_INTERVAL=30
ROUTER_METRICS_PORT=9090

# ========================================
# Logging Configuration
# ========================================
LOG_LEVEL=INFO
LOG_FORMAT=json

# ========================================
# SSL/TLS Configuration
# ========================================
PREFERRED_URL_SCHEME=https
# SSL_CERT_PATH=/etc/ssl/certs/ml_router.crt
# SSL_KEY_PATH=/etc/ssl/private/ml_router.key

# ========================================
# Email Configuration
# CRITICAL: Update with your production SMTP settings
# ========================================
MAIL_SERVER=smtp.yourdomain.com
MAIL_PORT=587
MAIL_USE_TLS=true
MAIL_USERNAME=noreply@yourdomain.com
MAIL_PASSWORD={keys['MAIL_PASSWORD']}

# ========================================
# API Rate Limiting
# ========================================
RATE_LIMIT_DEFAULT=100 per hour
RATE_LIMIT_API=30 per minute
RATE_LIMIT_AUTH=5 per minute
RATE_LIMIT_ENABLED=true

# ========================================
# Feature Flags
# ========================================
AUTH_ENABLED=true
ML_ROUTER_ENABLED=true
CACHE_ENABLED=true
RAG_ENABLED=true

# ========================================
# ML Model Configuration
# ========================================
MODEL_CACHE_SIZE=1000
MODEL_TIMEOUT=30
MAX_QUERY_LENGTH=10000

# ========================================
# Monitoring & Health Checks
# ========================================
HEALTH_CHECK_TIMEOUT=5
METRICS_ENABLED=true

# ========================================
# Background Services
# ========================================
CELERY_BROKER_URL=redis://redis:6379/2
CELERY_RESULT_BACKEND=redis://redis:6379/3
"""
    
    # Write to file
    with open('.env.production.secure', 'w') as f:
        f.write(env_content)
    
    print("\\n" + "=" * 60)
    print("PRODUCTION ENVIRONMENT FILE CREATED!")
    print("=" * 60)
    print("File: .env.production.secure")
    print("Contains: Cryptographically secure keys and configuration")
    print()
    print("NEXT STEPS:")
    print("1. Review and customize .env.production.secure")
    print("2. Add your AI provider API keys")
    print("3. Update domain names (replace 'yourdomain.com')")
    print("4. Set OAuth client ID/secret from Azure Portal")
    print("5. Configure SMTP settings for email")
    print("6. Copy to .env for deployment: cp .env.production.secure .env")
    print()
    print("SECURITY WARNINGS:")
    print("* NEVER commit .env files to version control")
    print("* Store production keys in secure password manager")
    print("* Rotate keys regularly (quarterly recommended)")
    print("* Use different keys for staging and production")
    print()
    print("Key Security Summary:")
    print(f"* Secret keys: {len([k for k in keys.keys() if 'SECRET' in k or 'PASSWORD' in k])} generated")
    print("* Key length: 32-64 characters (cryptographically secure)")
    print("* Character set: Letters, numbers, and special chars")
    print("* Generated using: Python secrets module (CSPRNG)")
    
    return keys

if __name__ == "__main__":
    try:
        keys = generate_production_keys()
        print("\\nProduction key generation completed successfully!")
    except Exception as e:
        print(f"\\nError generating keys: {e}")
        exit(1)