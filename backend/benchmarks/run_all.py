"""
Run complete benchmarking suite.

This script:
1. Prepares all benchmark datasets
2. Runs all benchmarks
3. Generates reports
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from prepare_datasets import main as prepare_datasets
from run_benchmarks import main as run_benchmarks


def main():
    """Run complete benchmarking suite."""
    print("\n" + "=" * 70)
    print("AI-Playground Complete Benchmark Suite")
    print("=" * 70)
    print()

    # Step 1: Prepare datasets
    print("STEP 1: Preparing benchmark datasets...")
    print("-" * 70)
    try:
        prepare_datasets()
        print("\n✓ Dataset preparation complete!")
    except Exception as e:
        print(f"\n✗ Dataset preparation failed: {e}")
        return 1

    # Step 2: Run benchmarks
    print("\n" + "=" * 70)
    print("STEP 2: Running benchmarks...")
    print("-" * 70)
    try:
        run_benchmarks()
        print("\n✓ Benchmarking complete!")
    except Exception as e:
        print(f"\n✗ Benchmarking failed: {e}")
        return 1

    print("\n" + "=" * 70)
    print("✓ Complete benchmark suite finished successfully!")
    print("=" * 70)
    print()
    print("Results saved to: backend/benchmarks/results/")
    print("  - benchmark_results.json")
    print("  - benchmark_results.csv")
    print("  - BENCHMARK_REPORT.md")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
