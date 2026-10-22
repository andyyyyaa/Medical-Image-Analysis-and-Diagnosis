#!/usr/bin/env python3
"""
Production startup script
Address flash message and login session issues
"""

import os
import sys
from app import app

if __name__ == '__main__':
    # Set environment variables
    os.environ['FLASK_ENV'] = 'production'
    
    # Read configuration
    host = os.environ.get('HOST', '0.0.0.0')
    port = int(os.environ.get('PORT', 5000))
    
    print("=== Production startup configuration ===")
    print(f"Host: {host}")
    print(f"Port: {port}")
    print(f"Flask environment: {os.environ.get('FLASK_ENV')}")
    print(f"Secret Key: {app.config.get('SECRET_KEY', 'Not Set')[:20]}...")
    print(f"Session configuration:")
    print(f"  - SECURE: {app.config.get('SESSION_COOKIE_SECURE')}")
    print(f"  - HTTPONLY: {app.config.get('SESSION_COOKIE_HTTPONLY')}")
    print(f"  - SAMESITE: {app.config.get('SESSION_COOKIE_SAMESITE')}")
    print(f"  - LIFETIME: {app.config.get('PERMANENT_SESSION_LIFETIME')} seconds")
    
    try:
        print("\nStarting the server...")
        app.run(
            host=host,
            port=port,
            debug=False,
            threaded=True
        )
    except KeyboardInterrupt:
        print("\nServer stopped")
    except Exception as e:
        print(f"Startup failed: {e}")
        sys.exit(1) 
