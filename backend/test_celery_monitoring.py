"""
Test script for Celery monitoring functionality

Run this script to verify that monitoring is working correctly:
    python test_celery_monitoring.py
"""

import requests
import time
from app.celery_app import celery_app
from app.monitoring.celery_metrics import get_celery_metrics_summary


def test_monitoring_setup():
    """Test that monitoring is properly set up."""
    print("=" * 60)
    print("Testing Celery Monitoring Setup")
    print("=" * 60)

    # Test 1: Check if metrics are defined
    print("\n1. Checking if metrics are defined...")
    try:
        from app.monitoring.celery_metrics import (
            celery_worker_pool_size,
            celery_queue_length,
            celery_task_failure_rate,
            celery_task_retry_counter,
        )
        print("   ✓ All new metrics are defined")
    except ImportError as e:
        print(f"   ✗ Failed to import metrics: {e}")
        return False

    # Test 2: Check if functions are available
    print("\n2. Checking monitoring functions...")
    try:
        from app.monitoring.celery_metrics import (
            update_queue_length,
            update_worker_stats,
            get_celery_metrics_summary,
        )
        print("   ✓ All monitoring functions are available")
    except ImportError as e:
        print(f"   ✗ Failed to import functions: {e}")
        return False

    # Test 3: Check API endpoints (requires server to be running)
    print("\n3. Testing API endpoints...")
    base_url = "http://localhost:8000"

    endpoints = [
        "/api/v1/celery/health",
        "/api/v1/celery/tasks/status",
        "/api/v1/celery/workers/status",
        "/api/v1/celery/queues/status",
        "/api/v1/celery/metrics/summary",
    ]

    print("   Note: This test requires the FastAPI server to be running")
    print("   Start with: uvicorn app.main:app --reload")

    for endpoint in endpoints:
        url = base_url + endpoint
        try:
            response = requests.get(url, timeout=2)
            if response.status_code == 200:
                print(f"   ✓ {endpoint} - OK")
            else:
                print(f"   ✗ {endpoint} - Status {response.status_code}")
        except requests.ConnectionError:
            print(f"   ⚠ {endpoint} - Server not running (skip)")
        except Exception as e:
            print(f"   ✗ {endpoint} - Error: {e}")

    # Test 4: Check metrics summary function
    print("\n4. Testing metrics summary function...")
    try:
        summary = get_celery_metrics_summary(celery_app)
        print(f"   ✓ Metrics summary retrieved")
        print(f"      Workers: {summary.get('workers', {}).get('total', 0)}")
        print(f"      Active tasks: {summary.get('tasks', {}).get('active', 0)}")
    except Exception as e:
        print(f"   ✗ Failed to get metrics summary: {e}")

    # Test 5: Check Prometheus metrics endpoint
    print("\n5. Testing Prometheus metrics endpoint...")
    try:
        response = requests.get(f"{base_url}/metrics", timeout=2)
        if response.status_code == 200:
            # Check for new metrics
            content = response.text
            new_metrics = [
                'celery_worker_pool_size',
                'celery_queue_length',
                'celery_task_failure_rate',
            ]

            found_metrics = []
            for metric in new_metrics:
                if metric in content:
                    found_metrics.append(metric)

            if found_metrics:
                print(f"   ✓ Prometheus metrics endpoint OK")
                print(f"      Found {len(found_metrics)}/{len(new_metrics)} new metrics")
            else:
                print(f"   ⚠ Metrics endpoint OK but new metrics not found")
                print(f"      This is normal if no tasks have run yet")
        else:
            print(f"   ✗ Metrics endpoint returned {response.status_code}")
    except requests.ConnectionError:
        print(f"   ⚠ Server not running (skip)")
    except Exception as e:
        print(f"   ✗ Error: {e}")

    print("\n" + "=" * 60)
    print("Monitoring Test Complete")
    print("=" * 60)
    print("\nNext Steps:")
    print("1. Start FastAPI server: uvicorn app.main:app --reload")
    print("2. Start Celery worker: celery -A celery_worker.celery_app worker")
    print("3. (Optional) Start Flower: python start_flower.py")
    print("4. Submit some tasks and monitor them!")
    print("\nMonitoring URLs:")
    print("- API: http://localhost:8000/api/v1/celery/tasks/status")
    print("- Metrics: http://localhost:8000/metrics")
    print("- Flower: http://localhost:5555/flower")


def test_task_submission():
    """Test submitting a task and monitoring it."""
    print("\n" + "=" * 60)
    print("Testing Task Submission and Monitoring")
    print("=" * 60)

    base_url = "http://localhost:8000"

    # Check if server is running
    try:
        requests.get(f"{base_url}/api/v1/celery/health", timeout=2)
    except requests.ConnectionError:
        print("\n✗ Server not running. Start it first:")
        print("  uvicorn app.main:app --reload")
        return

    print("\nThis is a placeholder for task submission test.")
    print("To test:")
    print("1. Submit a training task via the API")
    print("2. Monitor it with: GET /api/v1/celery/tasks/{task_id}/status")
    print("3. Watch metrics at: GET /api/v1/celery/metrics/summary")


if __name__ == "__main__":
    test_monitoring_setup()

    # Uncomment to test task submission
    # test_task_submission()
