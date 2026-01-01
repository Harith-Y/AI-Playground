#!/usr/bin/env python3
"""
Generate a test execution report for ML pipeline integration tests.

This script runs the integration tests and generates a detailed HTML report
with test results, coverage, and performance metrics.
"""

import subprocess
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any


def run_tests_with_report():
    """Run integration tests and generate report"""
    
    print("=" * 80)
    print("ML PIPELINE INTEGRATION TEST REPORT GENERATOR")
    print("=" * 80)
    print()
    
    # Create reports directory
    reports_dir = Path("test_reports")
    reports_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_name = f"integration_test_report_{timestamp}"
    
    # Test execution configuration
    test_config = {
        "timestamp": timestamp,
        "test_suites": [
            {
                "name": "Fast Integration Tests",
                "markers": "integration and not slow",
                "timeout": 300
            },
            {
                "name": "Full Integration Tests",
                "markers": "integration",
                "timeout": 1800
            }
        ]
    }
    
    print("Select test suite:")
    print("  1. Fast tests (skip slow tests)")
    print("  2. Full test suite")
    print()
    
    choice = input("Your choice (1 or 2): ").strip()
    
    if choice == "1":
        suite = test_config["test_suites"][0]
    else:
        suite = test_config["test_suites"][1]
    
    print(f"\n[Running] {suite['name']}...")
    print("-" * 80)
    print()
    
    # Build pytest command
    cmd = [
        "pytest",
        "-v",
        "-m", suite["markers"],
        "tests/integration/",
        f"--junit-xml={reports_dir}/{report_name}.xml",
        f"--html={reports_dir}/{report_name}.html",
        "--self-contained-html",
        "--cov=app.ml_engine",
        "--cov=app.api.v1.endpoints",
        f"--cov-report=html:{reports_dir}/{report_name}_coverage",
        "--cov-report=term-missing",
        "--durations=10",
        f"--timeout={suite['timeout']}"
    ]
    
    # Run tests
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=False)
    duration = time.time() - start_time
    
    print()
    print("-" * 80)
    print()
    
    # Generate summary
    print("[Report] Generating test report...")
    
    report_summary = {
        "suite_name": suite["name"],
        "timestamp": timestamp,
        "duration_seconds": round(duration, 2),
        "exit_code": result.returncode,
        "status": "PASSED" if result.returncode == 0 else "FAILED",
        "reports": {
            "html": f"{report_name}.html",
            "xml": f"{report_name}.xml",
            "coverage": f"{report_name}_coverage/index.html"
        }
    }
    
    # Save summary
    summary_file = reports_dir / f"{report_name}_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(report_summary, f, indent=2)
    
    # Print summary
    print()
    print("=" * 80)
    print("TEST EXECUTION SUMMARY")
    print("=" * 80)
    print(f"Suite:     {report_summary['suite_name']}")
    print(f"Status:    {report_summary['status']}")
    print(f"Duration:  {report_summary['duration_seconds']}s")
    print(f"Timestamp: {report_summary['timestamp']}")
    print()
    print("Reports:")
    print(f"  HTML:     {reports_dir}/{report_summary['reports']['html']}")
    print(f"  XML:      {reports_dir}/{report_summary['reports']['xml']}")
    print(f"  Coverage: {reports_dir}/{report_summary['reports']['coverage']}")
    print()
    
    # Open reports
    if sys.platform == "win32":
        open_cmd = "start"
    elif sys.platform == "darwin":
        open_cmd = "open"
    else:
        open_cmd = "xdg-open"
    
    print("Open reports in browser? (y/n): ", end="")
    if input().strip().lower() == 'y':
        subprocess.run([open_cmd, f"{reports_dir}/{report_summary['reports']['html']}"])
        subprocess.run([open_cmd, f"{reports_dir}/{report_summary['reports']['coverage']}"])
    
    print("=" * 80)
    
    return result.returncode


if __name__ == "__main__":
    sys.exit(run_tests_with_report())
