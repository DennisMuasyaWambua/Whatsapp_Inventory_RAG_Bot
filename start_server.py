#!/usr/bin/env python3

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Start the Django server with gunicorn using inline configuration."""
    
    # Get the project root directory
    project_root = Path(__file__).parent.absolute()
    os.chdir(project_root)
    
    # Set Django settings module
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'whatsapp_inventory_bot.settings')
    
    # Gunicorn command with inline configuration
    gunicorn_cmd = [
        'gunicorn',
        'whatsapp_inventory_bot.wsgi:application',
        '--bind', '0.0.0.0:8000',
        '--workers', '1',
        '--worker-class', 'sync',
        '--worker-connections', '1000',
        '--timeout', '120',
        '--keepalive', '2',
        '--max-requests', '100',
        '--max-requests-jitter', '10',
        '--preload',  # Changed from --no-preload for better memory management
        '--log-level', 'info',
        '--access-logfile', '-',  # Log to stdout
        '--error-logfile', '-',   # Log to stderr
        '--capture-output',
        '--enable-stdio-inheritance',
        '--worker-tmp-dir', '/dev/shm',
        '--graceful-timeout', '30',
    ]
    
    # Add memory limits on Linux
    try:
        import resource
        # Set soft memory limit to 768MB
        resource.setrlimit(resource.RLIMIT_AS, (768 * 1024 * 1024, 1024 * 1024 * 1024))
        print("Memory limits applied: 768MB soft, 1GB hard")
    except (ImportError, OSError) as e:
        print(f"Could not set memory limits: {e}")
    
    print("Starting Django server with gunicorn...")
    print(f"Command: {' '.join(gunicorn_cmd)}")
    print("Server will be available at http://0.0.0.0:8000")
    
    try:
        # Execute gunicorn
        subprocess.run(gunicorn_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error starting server: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nShutting down server...")
        sys.exit(0)

if __name__ == '__main__':
    main()