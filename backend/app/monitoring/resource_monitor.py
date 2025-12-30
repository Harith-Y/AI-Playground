"""
Resource Monitoring Utilities

Monitors system resources (CPU, memory, disk, network) for performance tracking.
"""

import time
import psutil
import threading
from typing import Dict, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime

from .metrics import (
    memory_usage_gauge,
    cpu_usage_gauge,
    disk_io_counter,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ResourceSnapshot:
    """Snapshot of system resources at a point in time."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_usage_percent: float
    disk_read_mb: float
    disk_write_mb: float
    network_sent_mb: float
    network_recv_mb: float
    process_count: int
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


class ResourceMonitor:
    """
    Monitor system resources and track performance metrics.
    
    Can run in background thread to continuously monitor resources.
    """
    
    def __init__(
        self,
        interval: float = 30.0,
        process_type: str = 'celery_worker',
        callback: Optional[Callable[[ResourceSnapshot], None]] = None
    ):
        """
        Initialize resource monitor.
        
        Args:
            interval: Monitoring interval in seconds
            process_type: Type of process being monitored
            callback: Optional callback function for each snapshot
        """
        self.interval = interval
        self.process_type = process_type
        self.callback = callback
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_disk_io = None
        self._last_network_io = None
    
    def start(self):
        """Start monitoring in background thread."""
        if self._running:
            logger.warning("Resource monitor already running")
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.info(f"Resource monitor started (interval={self.interval}s)")
    
    def stop(self):
        """Stop monitoring."""
        if not self._running:
            return
        
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        logger.info("Resource monitor stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self._running:
            try:
                snapshot = self.capture_snapshot()
                self._update_metrics(snapshot)
                
                if self.callback:
                    self.callback(snapshot)
                
            except Exception as e:
                logger.error(f"Error in resource monitoring: {e}", exc_info=True)
            
            time.sleep(self.interval)
    
    def capture_snapshot(self) -> ResourceSnapshot:
        """
        Capture current resource usage snapshot.
        
        Returns:
            ResourceSnapshot with current metrics
        """
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1.0)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used_mb = memory.used / (1024 * 1024)
        memory_available_mb = memory.available / (1024 * 1024)
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_usage_percent = disk.percent
        
        # Disk I/O
        disk_io = psutil.disk_io_counters()
        if disk_io:
            disk_read_mb = disk_io.read_bytes / (1024 * 1024)
            disk_write_mb = disk_io.write_bytes / (1024 * 1024)
        else:
            disk_read_mb = 0.0
            disk_write_mb = 0.0
        
        # Network I/O
        network_io = psutil.net_io_counters()
        if network_io:
            network_sent_mb = network_io.bytes_sent / (1024 * 1024)
            network_recv_mb = network_io.bytes_recv / (1024 * 1024)
        else:
            network_sent_mb = 0.0
            network_recv_mb = 0.0
        
        # Process count
        process_count = len(psutil.pids())
        
        return ResourceSnapshot(
            timestamp=datetime.utcnow(),
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_used_mb=memory_used_mb,
            memory_available_mb=memory_available_mb,
            disk_usage_percent=disk_usage_percent,
            disk_read_mb=disk_read_mb,
            disk_write_mb=disk_write_mb,
            network_sent_mb=network_sent_mb,
            network_recv_mb=network_recv_mb,
            process_count=process_count
        )
    
    def _update_metrics(self, snapshot: ResourceSnapshot):
        """
        Update Prometheus metrics with snapshot data.
        
        Args:
            snapshot: Resource snapshot
        """
        # Update CPU gauge
        cpu_usage_gauge.labels(
            process_type=self.process_type
        ).set(snapshot.cpu_percent)
        
        # Update memory gauge
        memory_usage_gauge.labels(
            process_type=self.process_type
        ).set(snapshot.memory_used_mb * 1024 * 1024)  # Convert to bytes
        
        # Update disk I/O counters (track deltas)
        if self._last_disk_io:
            disk_read_delta = snapshot.disk_read_mb - self._last_disk_io[0]
            disk_write_delta = snapshot.disk_write_mb - self._last_disk_io[1]
            
            if disk_read_delta > 0:
                disk_io_counter.labels(
                    operation='read',
                    process_type=self.process_type
                ).inc(disk_read_delta * 1024 * 1024)
            
            if disk_write_delta > 0:
                disk_io_counter.labels(
                    operation='write',
                    process_type=self.process_type
                ).inc(disk_write_delta * 1024 * 1024)
        
        self._last_disk_io = (snapshot.disk_read_mb, snapshot.disk_write_mb)
        
        # Log warnings for high resource usage
        if snapshot.cpu_percent > 90:
            logger.warning(
                f"High CPU usage: {snapshot.cpu_percent:.1f}%",
                extra={
                    'event': 'high_cpu_usage',
                    'cpu_percent': snapshot.cpu_percent
                }
            )
        
        if snapshot.memory_percent > 90:
            logger.warning(
                f"High memory usage: {snapshot.memory_percent:.1f}%",
                extra={
                    'event': 'high_memory_usage',
                    'memory_percent': snapshot.memory_percent,
                    'memory_used_mb': snapshot.memory_used_mb
                }
            )
        
        if snapshot.disk_usage_percent > 90:
            logger.warning(
                f"High disk usage: {snapshot.disk_usage_percent:.1f}%",
                extra={
                    'event': 'high_disk_usage',
                    'disk_usage_percent': snapshot.disk_usage_percent
                }
            )
    
    def get_current_usage(self) -> Dict:
        """
        Get current resource usage as dictionary.
        
        Returns:
            Dictionary with current resource metrics
        """
        snapshot = self.capture_snapshot()
        return snapshot.to_dict()
    
    @staticmethod
    def check_resource_availability(
        min_memory_mb: float = 1000,
        min_disk_gb: float = 5.0
    ) -> tuple[bool, str]:
        """
        Check if sufficient resources are available.
        
        Args:
            min_memory_mb: Minimum required memory in MB
            min_disk_gb: Minimum required disk space in GB
            
        Returns:
            Tuple of (is_available, message)
        """
        # Check memory
        memory = psutil.virtual_memory()
        available_memory_mb = memory.available / (1024 * 1024)
        
        if available_memory_mb < min_memory_mb:
            return False, f"Insufficient memory: {available_memory_mb:.0f}MB available, {min_memory_mb:.0f}MB required"
        
        # Check disk space
        disk = psutil.disk_usage('/')
        available_disk_gb = disk.free / (1024 * 1024 * 1024)
        
        if available_disk_gb < min_disk_gb:
            return False, f"Insufficient disk space: {available_disk_gb:.1f}GB available, {min_disk_gb:.1f}GB required"
        
        return True, "Resources available"
    
    @staticmethod
    def get_process_info(pid: Optional[int] = None) -> Dict:
        """
        Get information about a specific process.
        
        Args:
            pid: Process ID (None = current process)
            
        Returns:
            Dictionary with process information
        """
        try:
            process = psutil.Process(pid)
            
            with process.oneshot():
                return {
                    'pid': process.pid,
                    'name': process.name(),
                    'status': process.status(),
                    'cpu_percent': process.cpu_percent(interval=0.1),
                    'memory_mb': process.memory_info().rss / (1024 * 1024),
                    'memory_percent': process.memory_percent(),
                    'num_threads': process.num_threads(),
                    'create_time': datetime.fromtimestamp(process.create_time()).isoformat(),
                }
        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            logger.error(f"Failed to get process info: {e}")
            return {}


# Global resource monitor instance
_global_monitor: Optional[ResourceMonitor] = None


def start_global_monitor(interval: float = 30.0, process_type: str = 'celery_worker'):
    """
    Start global resource monitor.
    
    Args:
        interval: Monitoring interval in seconds
        process_type: Type of process being monitored
    """
    global _global_monitor
    
    if _global_monitor is not None:
        logger.warning("Global resource monitor already started")
        return
    
    _global_monitor = ResourceMonitor(interval=interval, process_type=process_type)
    _global_monitor.start()


def stop_global_monitor():
    """Stop global resource monitor."""
    global _global_monitor
    
    if _global_monitor is not None:
        _global_monitor.stop()
        _global_monitor = None


def get_global_monitor() -> Optional[ResourceMonitor]:
    """Get global resource monitor instance."""
    return _global_monitor
