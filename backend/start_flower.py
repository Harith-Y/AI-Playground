"""
Flower - Celery Monitoring Tool

This script starts Flower, a web-based tool for monitoring and administrating
Celery clusters.

Usage:
    python start_flower.py

Or with custom options:
    python start_flower.py --port=5555 --address=0.0.0.0
"""

import sys
import os
from app.celery_app import celery_app

if __name__ == '__main__':
    # Import flower
    try:
        from flower.command import FlowerCommand
    except ImportError:
        print("ERROR: Flower is not installed.")
        print("Install it with: pip install flower")
        sys.exit(1)

    # Default Flower arguments
    default_args = [
        'flower',
        '--broker=' + celery_app.conf.broker_url,
        '--port=5555',
        '--address=0.0.0.0',
        '--url_prefix=flower',
        '--persistent=True',
        '--db=flower.db',
        '--max_tasks=10000',
    ]

    # Merge with command line arguments
    args = default_args + sys.argv[1:]

    print("=" * 60)
    print("Starting Flower - Celery Monitoring")
    print("=" * 60)
    print(f"Broker: {celery_app.conf.broker_url}")
    print(f"Backend: {celery_app.conf.result_backend}")
    print(f"Access URL: http://localhost:5555/flower")
    print("=" * 60)

    # Start Flower
    flower = FlowerCommand(app=celery_app)
    flower.execute_from_commandline(args)
