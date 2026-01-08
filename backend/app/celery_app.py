"""
Celery application instance
"""
from celery import Celery
from app.core.config import settings

# Create Celery instance
celery_app = Celery(
    "aiplayground",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=[
        "app.tasks.preprocessing_tasks",
        "app.tasks.training_tasks",
        "app.tasks.tuning_tasks"
    ]
)

# Celery configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 hour
    task_soft_time_limit=3000,  # 50 minutes
    worker_send_task_events=True,
    task_send_sent_event=True,
    # Redis backend settings for stability
    broker_connection_retry_on_startup=True,
    broker_transport_options={
        "visibility_timeout": 3600,
        "health_check_interval": 10,
        "socket_keepalive": True,
    },
    result_backend_transport_options={
        "health_check_interval": 10,
        "socket_keepalive": True,
        "retry_policy": {
            "timeout": 5.0
        }
    }
)
