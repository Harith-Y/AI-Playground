"""
Celery Events Camera

Captures and stores Celery task events for historical analysis and monitoring.
Events are stored in the database for querying and dashboards.

Usage:
    celery -A celery_worker.celery_app events --camera=app.monitoring.celery_camera.TaskCamera
"""

from datetime import datetime
from typing import Dict, Any, Optional
from celery.events.snapshot import Polaroid
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

Base = declarative_base()


class TaskEvent(Base):
    """Model for storing task events."""
    __tablename__ = 'celery_task_events'

    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(String(255), index=True, nullable=False)
    task_name = Column(String(255), index=True)
    state = Column(String(50), index=True)
    worker = Column(String(255))
    timestamp = Column(DateTime, index=True, nullable=False)
    runtime = Column(Float)
    retries = Column(Integer, default=0)
    exception = Column(Text)
    traceback = Column(Text)
    args = Column(JSON)
    kwargs = Column(JSON)
    result = Column(JSON)
    metadata = Column(JSON)

    def __repr__(self):
        return f"<TaskEvent(task_id='{self.task_id}', state='{self.state}', timestamp='{self.timestamp}')>"


class WorkerEvent(Base):
    """Model for storing worker events."""
    __tablename__ = 'celery_worker_events'

    id = Column(Integer, primary_key=True, autoincrement=True)
    worker_name = Column(String(255), index=True, nullable=False)
    event_type = Column(String(50), index=True)
    timestamp = Column(DateTime, index=True, nullable=False)
    load_avg = Column(JSON)
    processed = Column(Integer)
    active = Column(Integer)
    metadata = Column(JSON)

    def __repr__(self):
        return f"<WorkerEvent(worker='{self.worker_name}', event='{self.event_type}', timestamp='{self.timestamp}')>"


class TaskCamera(Polaroid):
    """
    Camera that captures task events and stores them in database.

    This runs as a separate process that listens to Celery events
    and persists them for historical analysis.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Create database engine and session
        self.engine = create_engine(
            settings.DATABASE_URL,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10
        )

        # Create tables if they don't exist
        Base.metadata.create_all(self.engine)

        # Create session factory
        self.SessionLocal = sessionmaker(
            bind=self.engine,
            autocommit=False,
            autoflush=False
        )

        logger.info("TaskCamera initialized and connected to database")

    def on_shutter(self, state):
        """
        Called periodically (default every second) to capture state snapshot.

        Args:
            state: Current state from event receiver
        """
        db: Session = self.SessionLocal()

        try:
            # Process task events
            for task_id, task in state.tasks.items():
                self._store_task_event(db, task_id, task)

            # Process worker events
            for worker_name, worker in state.workers.items():
                self._store_worker_event(db, worker_name, worker)

            db.commit()

        except Exception as e:
            logger.error(f"Error in camera snapshot: {e}", exc_info=True)
            db.rollback()

        finally:
            db.close()

    def _store_task_event(self, db: Session, task_id: str, task: Any):
        """
        Store or update task event in database.

        Args:
            db: Database session
            task_id: Task ID
            task: Task state object
        """
        try:
            # Check if event already exists
            event = db.query(TaskEvent).filter(
                TaskEvent.task_id == task_id
            ).first()

            task_data = {
                'task_id': task_id,
                'task_name': task.name,
                'state': task.state,
                'worker': task.worker.hostname if task.worker else None,
                'timestamp': datetime.fromtimestamp(task.timestamp) if task.timestamp else datetime.utcnow(),
                'runtime': task.runtime,
                'retries': task.retries or 0,
                'exception': task.exception if hasattr(task, 'exception') else None,
                'traceback': task.traceback if hasattr(task, 'traceback') else None,
                'args': task.args if hasattr(task, 'args') else None,
                'kwargs': task.kwargs if hasattr(task, 'kwargs') else None,
                'result': task.result if hasattr(task, 'result') else None,
                'metadata': {
                    'eta': str(task.eta) if task.eta else None,
                    'expires': str(task.expires) if task.expires else None,
                    'clock': task.clock,
                }
            }

            if event:
                # Update existing event
                for key, value in task_data.items():
                    setattr(event, key, value)
            else:
                # Create new event
                event = TaskEvent(**task_data)
                db.add(event)

        except Exception as e:
            logger.error(f"Error storing task event {task_id}: {e}", exc_info=True)

    def _store_worker_event(self, db: Session, worker_name: str, worker: Any):
        """
        Store worker event in database.

        Args:
            db: Database session
            worker_name: Worker name
            worker: Worker state object
        """
        try:
            event_data = {
                'worker_name': worker_name,
                'event_type': 'heartbeat',
                'timestamp': datetime.fromtimestamp(worker.timestamp) if worker.timestamp else datetime.utcnow(),
                'load_avg': worker.loadavg if hasattr(worker, 'loadavg') else None,
                'processed': worker.processed if hasattr(worker, 'processed') else 0,
                'active': len(worker.active) if hasattr(worker, 'active') else 0,
                'metadata': {
                    'status': worker.status_string if hasattr(worker, 'status_string') else None,
                    'clock': worker.clock if hasattr(worker, 'clock') else None,
                }
            }

            event = WorkerEvent(**event_data)
            db.add(event)

        except Exception as e:
            logger.error(f"Error storing worker event {worker_name}: {e}", exc_info=True)

    def cleanup(self):
        """Cleanup old events from database."""
        db: Session = self.SessionLocal()

        try:
            # Delete task events older than 30 days
            cutoff_date = datetime.utcnow() - timedelta(days=30)

            deleted_tasks = db.query(TaskEvent).filter(
                TaskEvent.timestamp < cutoff_date
            ).delete()

            deleted_workers = db.query(WorkerEvent).filter(
                WorkerEvent.timestamp < cutoff_date
            ).delete()

            db.commit()

            logger.info(
                f"Cleanup complete: deleted {deleted_tasks} task events "
                f"and {deleted_workers} worker events older than 30 days"
            )

        except Exception as e:
            logger.error(f"Error during cleanup: {e}", exc_info=True)
            db.rollback()

        finally:
            db.close()


# Export for use
__all__ = ['TaskCamera', 'TaskEvent', 'WorkerEvent']
