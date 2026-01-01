"""
Lightweight task handler for Render Free Tier
Uses FastAPI BackgroundTasks instead of Celery to save memory
"""
import asyncio
from typing import Callable, Any, Dict
from datetime import datetime
import uuid
from enum import Enum

class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

# In-memory task storage (use Redis in production)
_tasks: Dict[str, Dict[str, Any]] = {}

class TaskManager:
    """Manages background tasks without Celery"""
    
    @staticmethod
    def create_task(task_name: str) -> str:
        """Create a new task and return task_id"""
        task_id = str(uuid.uuid4())
        _tasks[task_id] = {
            "id": task_id,
            "name": task_name,
            "status": TaskStatus.PENDING,
            "created_at": datetime.utcnow().isoformat(),
            "started_at": None,
            "completed_at": None,
            "result": None,
            "error": None,
            "progress": 0
        }
        return task_id
    
    @staticmethod
    def get_task(task_id: str) -> Dict[str, Any]:
        """Get task status and result"""
        return _tasks.get(task_id, {
            "id": task_id,
            "status": TaskStatus.FAILED,
            "error": "Task not found"
        })
    
    @staticmethod
    def update_task(task_id: str, **kwargs):
        """Update task information"""
        if task_id in _tasks:
            _tasks[task_id].update(kwargs)
    
    @staticmethod
    async def run_task(task_id: str, func: Callable, *args, **kwargs):
        """Run a task in the background"""
        try:
            # Update status to running
            TaskManager.update_task(
                task_id,
                status=TaskStatus.RUNNING,
                started_at=datetime.utcnow().isoformat()
            )
            
            # Run the function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = await asyncio.to_thread(func, *args, **kwargs)
            
            # Update status to completed
            TaskManager.update_task(
                task_id,
                status=TaskStatus.COMPLETED,
                completed_at=datetime.utcnow().isoformat(),
                result=result,
                progress=100
            )
            
        except Exception as e:
            # Update status to failed
            TaskManager.update_task(
                task_id,
                status=TaskStatus.FAILED,
                completed_at=datetime.utcnow().isoformat(),
                error=str(e)
            )
    
    @staticmethod
    def cleanup_old_tasks(max_age_hours: int = 24):
        """Remove old completed tasks to free memory"""
        from datetime import timedelta
        cutoff = datetime.utcnow() - timedelta(hours=max_age_hours)
        
        to_remove = []
        for task_id, task in _tasks.items():
            if task.get("completed_at"):
                completed = datetime.fromisoformat(task["completed_at"])
                if completed < cutoff:
                    to_remove.append(task_id)
        
        for task_id in to_remove:
            del _tasks[task_id]

# Helper function for training tasks
async def run_training_task(task_id: str, training_func: Callable, *args, **kwargs):
    """
    Wrapper for training tasks with progress updates
    """
    try:
        TaskManager.update_task(task_id, status=TaskStatus.RUNNING, progress=10)
        
        # Run training
        result = await asyncio.to_thread(training_func, *args, **kwargs)
        
        TaskManager.update_task(
            task_id,
            status=TaskStatus.COMPLETED,
            completed_at=datetime.utcnow().isoformat(),
            result=result,
            progress=100
        )
        
    except Exception as e:
        TaskManager.update_task(
            task_id,
            status=TaskStatus.FAILED,
            completed_at=datetime.utcnow().isoformat(),
            error=str(e)
        )

# Periodic cleanup task
async def periodic_cleanup():
    """Run cleanup every hour"""
    while True:
        await asyncio.sleep(3600)  # 1 hour
        TaskManager.cleanup_old_tasks()
