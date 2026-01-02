"""
Flower - Celery Monitoring Tool

This script starts Flower, a web-based tool for monitoring and administrating
Celery clusters.

Usage:
    # Development (no auth)
    python start_flower.py

    # With basic authentication
    python start_flower.py --auth

    # Custom port and address
    python start_flower.py --port=5555 --address=0.0.0.0

    # Production mode with auth
    python start_flower.py --auth --basic_auth=admin:secret

Environment Variables:
    FLOWER_BASIC_AUTH: Basic auth credentials (format: user:password)
    FLOWER_PORT: Port to run on (default: 5555)
    FLOWER_DEBUG: Enable debug mode (default: False)
"""

import sys
import os
import argparse
from pathlib import Path
from app.celery_app import celery_app
from app.core.config import settings

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Start Flower monitoring dashboard')
    parser.add_argument('--port', type=int, default=5555, help='Port to run Flower on')
    parser.add_argument('--address', default='0.0.0.0', help='Address to bind to')
    parser.add_argument('--auth', action='store_true', help='Enable authentication')
    parser.add_argument('--basic_auth', help='Basic auth credentials (user:password)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--max_tasks', type=int, default=10000, help='Max tasks to keep')
    parser.add_argument('--db', default='flower.db', help='Database file path')
    parser.add_argument('--url_prefix', default='flower', help='URL prefix')
    parser.add_argument('--no-persistent', action='store_true', help='Disable persistence')

    return parser.parse_known_args()

def get_auth_credentials():
    """Get authentication credentials from env or config."""
    # Check environment variable first
    env_auth = os.getenv('FLOWER_BASIC_AUTH')
    if env_auth:
        return env_auth

    # Default credentials for development
    return 'admin:admin123'

if __name__ == '__main__':
    # Import flower
    try:
        from flower.command import FlowerCommand
    except ImportError:
        print("=" * 60)
        print("ERROR: Flower is not installed.")
        print("=" * 60)
        print("Install it with:")
        print("  pip install flower")
        print("\nOr install all monitoring dependencies:")
        print("  pip install -r requirements.monitoring.txt")
        print("=" * 60)
        sys.exit(1)

    # Parse arguments
    args, extra_args = parse_args()

    # Get port from environment or args
    port = int(os.getenv('FLOWER_PORT', args.port))
    debug = os.getenv('FLOWER_DEBUG', 'false').lower() == 'true' or args.debug

    # Build Flower arguments
    flower_args = [
        'flower',
        f'--broker={celery_app.conf.broker_url}',
        f'--port={port}',
        f'--address={args.address}',
        f'--url_prefix={args.url_prefix}',
        f'--max_tasks={args.max_tasks}',
    ]

    # Add persistence
    if not args.no_persistent:
        flower_args.append('--persistent=True')
        flower_args.append(f'--db={args.db}')

    # Add authentication if enabled
    auth_enabled = False
    if args.auth or args.basic_auth or os.getenv('FLOWER_BASIC_AUTH'):
        auth_creds = args.basic_auth or get_auth_credentials()
        flower_args.append(f'--basic_auth={auth_creds}')
        auth_enabled = True

    # Add debug mode
    if debug:
        flower_args.append('--debug=True')

    # Add any extra command line arguments
    flower_args.extend(extra_args)

    # Print startup information
    print("=" * 70)
    print("🌸 Starting Flower - Celery Monitoring Dashboard")
    print("=" * 70)
    print(f"📡 Broker:       {celery_app.conf.broker_url}")
    print(f"💾 Backend:      {celery_app.conf.result_backend}")
    print(f"🌐 URL:          http://localhost:{port}/{args.url_prefix}")
    print(f"🗄️  Database:     {args.db if not args.no_persistent else 'In-memory (no persistence)'}")
    print(f"📊 Max Tasks:    {args.max_tasks}")
    print(f"🔒 Auth:         {'Enabled' if auth_enabled else 'Disabled (development only)'}")
    if debug:
        print(f"🐛 Debug:        Enabled")
    print("=" * 70)

    if auth_enabled and args.basic_auth == 'admin:admin123':
        print("⚠️  WARNING: Using default credentials!")
        print("   For production, set custom credentials:")
        print("   python start_flower.py --basic_auth=myuser:mypassword")
        print("   or set FLOWER_BASIC_AUTH environment variable")
        print("=" * 70)

    if not auth_enabled:
        print("⚠️  WARNING: Authentication is DISABLED!")
        print("   Anyone can access the dashboard and control workers.")
        print("   Enable auth with: python start_flower.py --auth")
        print("=" * 70)

    print("\n✨ Flower is starting... Press Ctrl+C to stop\n")

    try:
        # Start Flower
        flower = FlowerCommand(app=celery_app)
        flower.execute_from_commandline(flower_args)
    except KeyboardInterrupt:
        print("\n\n🛑 Flower stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error starting Flower: {e}")
        sys.exit(1)
