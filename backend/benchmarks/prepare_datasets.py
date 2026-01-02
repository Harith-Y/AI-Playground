"""
Prepare benchmark datasets for testing ML pipeline.

This script downloads and prepares standard ML datasets:
- Iris (classification)
- Diabetes (regression)
- Wine Quality (classification)
- Boston Housing (regression) - using California Housing as alternative
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn import datasets
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_datasets_dir():
    """Create benchmarks/datasets directory."""
    datasets_dir = Path(__file__).parent / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    return datasets_dir


def prepare_iris(output_dir: Path):
    """
    Prepare Iris dataset.

    Dataset info:
    - Samples: 150
    - Features: 4 (sepal length, sepal width, petal length, petal width)
    - Target: 3 classes (setosa, versicolor, virginica)
    - Task: Multi-class classification
    """
    logger.info("Preparing Iris dataset...")

    iris = datasets.load_iris(as_frame=True)
    df = iris.frame

    # Rename target column
    df = df.rename(columns={'target': 'species'})

    # Map target to class names
    df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

    output_path = output_dir / "iris.csv"
    df.to_csv(output_path, index=False)

    logger.info(f"✓ Iris dataset saved: {output_path}")
    logger.info(f"  Shape: {df.shape}, Target: species (3 classes)")

    return output_path


def prepare_diabetes(output_dir: Path):
    """
    Prepare Diabetes dataset.

    Dataset info:
    - Samples: 442
    - Features: 10 (age, sex, bmi, bp, s1-s6)
    - Target: Continuous (disease progression)
    - Task: Regression
    """
    logger.info("Preparing Diabetes dataset...")

    diabetes = datasets.load_diabetes(as_frame=True)
    df = diabetes.frame

    # Rename target column
    df = df.rename(columns={'target': 'progression'})

    output_path = output_dir / "diabetes.csv"
    df.to_csv(output_path, index=False)

    logger.info(f"✓ Diabetes dataset saved: {output_path}")
    logger.info(f"  Shape: {df.shape}, Target: progression (regression)")

    return output_path


def prepare_california_housing(output_dir: Path):
    """
    Prepare California Housing dataset (alternative to Boston Housing).

    Dataset info:
    - Samples: 20,640
    - Features: 8 (MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude)
    - Target: Continuous (median house value)
    - Task: Regression

    Note: Boston Housing dataset is deprecated due to ethical concerns.
    California Housing is the recommended alternative.
    """
    logger.info("Preparing California Housing dataset...")

    housing = datasets.fetch_california_housing(as_frame=True)
    df = housing.frame

    # Rename target column
    df = df.rename(columns={'MedHouseVal': 'median_house_value'})

    output_path = output_dir / "california_housing.csv"
    df.to_csv(output_path, index=False)

    logger.info(f"✓ California Housing dataset saved: {output_path}")
    logger.info(f"  Shape: {df.shape}, Target: median_house_value (regression)")

    return output_path


def prepare_wine(output_dir: Path):
    """
    Prepare Wine dataset.

    Dataset info:
    - Samples: 178
    - Features: 13 (chemical analysis features)
    - Target: 3 classes (wine cultivars)
    - Task: Multi-class classification
    """
    logger.info("Preparing Wine dataset...")

    wine = datasets.load_wine(as_frame=True)
    df = wine.frame

    # Rename target column
    df = df.rename(columns={'target': 'cultivar'})

    output_path = output_dir / "wine.csv"
    df.to_csv(output_path, index=False)

    logger.info(f"✓ Wine dataset saved: {output_path}")
    logger.info(f"  Shape: {df.shape}, Target: cultivar (3 classes)")

    return output_path


def prepare_breast_cancer(output_dir: Path):
    """
    Prepare Breast Cancer dataset.

    Dataset info:
    - Samples: 569
    - Features: 30 (computed from digitized image)
    - Target: 2 classes (malignant, benign)
    - Task: Binary classification
    """
    logger.info("Preparing Breast Cancer dataset...")

    cancer = datasets.load_breast_cancer(as_frame=True)
    df = cancer.frame

    # Rename target column
    df = df.rename(columns={'target': 'diagnosis'})

    # Map target to class names
    df['diagnosis'] = df['diagnosis'].map({0: 'malignant', 1: 'benign'})

    output_path = output_dir / "breast_cancer.csv"
    df.to_csv(output_path, index=False)

    logger.info(f"✓ Breast Cancer dataset saved: {output_path}")
    logger.info(f"  Shape: {df.shape}, Target: diagnosis (2 classes)")

    return output_path


def prepare_digits(output_dir: Path):
    """
    Prepare Digits dataset.

    Dataset info:
    - Samples: 1,797
    - Features: 64 (8x8 pixel values)
    - Target: 10 classes (digits 0-9)
    - Task: Multi-class classification
    """
    logger.info("Preparing Digits dataset...")

    digits = datasets.load_digits(as_frame=True)
    df = digits.frame

    # Rename target column
    df = df.rename(columns={'target': 'digit'})

    output_path = output_dir / "digits.csv"
    df.to_csv(output_path, index=False)

    logger.info(f"✓ Digits dataset saved: {output_path}")
    logger.info(f"  Shape: {df.shape}, Target: digit (10 classes)")

    return output_path


def create_dataset_info(output_dir: Path):
    """Create a README with dataset information."""
    readme_content = """# Benchmark Datasets

This directory contains standard ML datasets for benchmarking the AI-Playground pipeline.

## Datasets

### Classification Datasets

1. **Iris** (iris.csv)
   - Samples: 150
   - Features: 4
   - Target: species (3 classes: setosa, versicolor, virginica)
   - Use case: Multi-class classification
   - Difficulty: Easy

2. **Wine** (wine.csv)
   - Samples: 178
   - Features: 13
   - Target: cultivar (3 classes)
   - Use case: Multi-class classification
   - Difficulty: Easy

3. **Breast Cancer** (breast_cancer.csv)
   - Samples: 569
   - Features: 30
   - Target: diagnosis (2 classes: malignant, benign)
   - Use case: Binary classification
   - Difficulty: Medium

4. **Digits** (digits.csv)
   - Samples: 1,797
   - Features: 64
   - Target: digit (10 classes: 0-9)
   - Use case: Multi-class classification
   - Difficulty: Medium

### Regression Datasets

1. **Diabetes** (diabetes.csv)
   - Samples: 442
   - Features: 10
   - Target: progression (disease progression after one year)
   - Use case: Regression
   - Difficulty: Easy

2. **California Housing** (california_housing.csv)
   - Samples: 20,640
   - Features: 8
   - Target: median_house_value
   - Use case: Regression
   - Difficulty: Medium
   - Note: This is the recommended alternative to the deprecated Boston Housing dataset

## Usage

These datasets are used for:
- Testing the ML training pipeline
- Benchmarking model performance
- Validating memory optimization
- Testing preprocessing operations
- Performance profiling

## Data Sources

All datasets are from scikit-learn's built-in datasets:
- https://scikit-learn.org/stable/datasets/toy_dataset.html
- https://scikit-learn.org/stable/datasets/real_world.html
"""

    readme_path = output_dir / "README.md"
    readme_path.write_text(readme_content)
    logger.info(f"✓ Dataset info saved: {readme_path}")


def main():
    """Prepare all benchmark datasets."""
    logger.info("=" * 60)
    logger.info("Preparing benchmark datasets")
    logger.info("=" * 60)

    # Create output directory
    output_dir = create_datasets_dir()
    logger.info(f"Output directory: {output_dir}")
    logger.info("")

    # Prepare all datasets
    datasets_prepared = []

    try:
        datasets_prepared.append(prepare_iris(output_dir))
        datasets_prepared.append(prepare_diabetes(output_dir))

        # Try California Housing, skip if download fails
        try:
            datasets_prepared.append(prepare_california_housing(output_dir))
        except Exception as e:
            logger.warning(f"Skipping California Housing dataset (download failed): {e}")

        datasets_prepared.append(prepare_wine(output_dir))
        datasets_prepared.append(prepare_breast_cancer(output_dir))
        datasets_prepared.append(prepare_digits(output_dir))

        # Create dataset info
        create_dataset_info(output_dir)

        logger.info("")
        logger.info("=" * 60)
        logger.info(f"✓ Successfully prepared {len(datasets_prepared)} datasets")
        logger.info("=" * 60)

        for path in datasets_prepared:
            logger.info(f"  - {path.name}")

        return datasets_prepared

    except Exception as e:
        logger.error(f"Error preparing datasets: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
