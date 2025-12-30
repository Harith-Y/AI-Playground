"""
Celery worker entry point
Run with: celery -A celery_worker.celery_app worker --loglevel=info
"""
from app.celery_app import celery_app
from app.monitoring.celery_metrics import setup_celery_metrics
from app.monitoring.resource_monitor import start_global_monitor
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Setup monitoring
try:
    setup_celery_metrics(celery_app)
    logger.info("Celery metrics monitoring initialized")
    
    # Start resource monitoring (every 30 seconds)
    start_global_monitor(interval=30.0, process_type='celery_worker')
    logger.info("Resource monitoring started")
    
except Exception as e:
    logger.error(f"Failed to initialize monitoring: {e}", exc_info=True)

# This makes the celery_app accessible when running: celery -A celery_worker worker
__all__ = ["celery_app"]
