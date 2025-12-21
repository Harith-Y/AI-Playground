"""
Celery worker entry point
Run with: celery -A celery_worker.celery_app worker --loglevel=info
"""
from app.celery_app import celery_app

# This makes the celery_app accessible when running: celery -A celery_worker worker
__all__ = ["celery_app"]
