"""
Metrics Endpoint

Exposes Prometheus metrics for monitoring.
"""

from fastapi import APIRouter, Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from app.monitoring.metrics import metrics_registry
from app.monitoring.resource_monitor import ResourceMonitor

router = APIRouter()


@router.get("/metrics", include_in_schema=False)
async def metrics():
    """
    Prometheus metrics endpoint.
    
    Returns metrics in Prometheus text format.
    """
    return Response(
        content=generate_latest(metrics_registry),
        media_type=CONTENT_TYPE_LATEST
    )


@router.get("/health/resources")
async def resource_health():
    """
    Get current resource usage for health checks.
    
    Returns:
        Dictionary with current resource metrics
    """
    monitor = ResourceMonitor()
    snapshot = monitor.capture_snapshot()
    
    return {
        "status": "healthy",
        "resources": snapshot.to_dict(),
        "warnings": _check_resource_warnings(snapshot)
    }


def _check_resource_warnings(snapshot) -> list:
    """Check for resource warnings."""
    warnings = []
    
    if snapshot.cpu_percent > 80:
        warnings.append(f"High CPU usage: {snapshot.cpu_percent:.1f}%")
    
    if snapshot.memory_percent > 80:
        warnings.append(f"High memory usage: {snapshot.memory_percent:.1f}%")
    
    if snapshot.disk_usage_percent > 80:
        warnings.append(f"High disk usage: {snapshot.disk_usage_percent:.1f}%")
    
    return warnings
