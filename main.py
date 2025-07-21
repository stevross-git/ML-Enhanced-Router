#!/usr/bin/env python3
"""
ML Query Router - Application Entry Point
Clean entry point following application factory pattern
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import application factory
from app import create_app

def main():
    """Main entry point for the application"""
    # Get environment configuration
    env = os.environ.get('FLASK_ENV', 'development')
    
    # Create Flask application
    app = create_app(config_name=env)
    
    # Get server configuration
    host = os.environ.get('HOST', '0.0.0.0')
    port = int(os.environ.get('PORT', 5000))
    debug = env == 'development'
    
    print(f"🚀 Starting ML Query Router")
    print(f"📍 Environment: {env}")
    print(f"🌐 Server: http://{host}:{port}")
    print(f"🔧 Debug mode: {debug}")
    
    # Run the application
    app.run(
        host=host,
        port=port,
        debug=debug,
        threaded=True
    )

if __name__ == '__main__':
    main()
