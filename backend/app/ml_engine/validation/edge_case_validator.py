"""
Edge Case Validator for ML Pipeline

Handles critical edge cases including:
- Tiny datasets (< 20 samples)
- High cardinality categorical features (> 100 unique values)
- Extreme class imbalance (> 100:1 ratio)
- Single-sample classes
- All-null or constant columns
- Insufficient samples for stratification

Prevents crashes and provides actionable recommendations.
"""

from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum

from app.utils.logger import get_logger

logger = get_logger(__name__)


class Severity(Enum):
    """Severity levels for edge case issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class EdgeCaseIssue:
    """Represents an edge case issue found in the data."""
    severity: Severity
    category: str
    message: str
    details: Dict[str, Any]
    recommendation: str
    auto_fixable: bool = False
    fix_applied: bool = False


class EdgeCaseValidator:
    """
    Validates data for edge cases that could cause crashes or poor model performance.

    Checks for:
    - Dataset size issues (tiny datasets, insufficient samples)
    - High cardinality categorical features
    - Class imbalance problems
    - Null value issues
    - Constant features
    - Stratification feasibility

    Example:
        >>> validator = EdgeCaseValidator()
        >>> issues = validator.validate_dataset(df, target_column='label')
        >>> if validator.has_critical_issues(issues):
        ...     raise ValueError("Critical data issues found")
    """

    # Thresholds
    MIN_SAMPLES_WARNING = 20
    MIN_SAMPLES_CRITICAL = 10
    MIN_SAMPLES_PER_CLASS_WARNING = 10
    MIN_SAMPLES_PER_CLASS_CRITICAL = 5
    HIGH_CARDINALITY_WARNING = 50
    HIGH_CARDINALITY_CRITICAL = 100
    MAX_CARDINALITY_RATIO_WARNING = 0.5  # 50% unique values
    MAX_CARDINALITY_RATIO_CRITICAL = 0.8  # 80% unique values
    CLASS_IMBALANCE_WARNING = 10  # 10:1 ratio
    CLASS_IMBALANCE_CRITICAL = 100  # 100:1 ratio
    MIN_SAMPLES_FOR_SMOTE = 6  # Minimum for k_neighbors=5
    NULL_RATIO_WARNING = 0.5  # 50% null
    NULL_RATIO_CRITICAL = 0.9  # 90% null

    def __init__(self):
        """Initialize edge case validator."""
        logger.debug("EdgeCaseValidator initialized")

    def validate_dataset(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        task_type: Optional[str] = None,
        test_size: float = 0.2,
        use_stratify: bool = False,
        use_oversampling: bool = False,
        encoding_columns: Optional[List[str]] = None
    ) -> List[EdgeCaseIssue]:
        """
        Validate dataset for edge cases.

        Args:
            df: DataFrame to validate
            target_column: Name of target column (if supervised)
            task_type: 'classification', 'regression', or 'clustering'
            test_size: Proportion of data for test set
            use_stratify: Whether stratified splitting will be used
            use_oversampling: Whether oversampling (SMOTE) will be used
            encoding_columns: Columns to be one-hot encoded

        Returns:
            List of EdgeCaseIssue objects
        """
        issues = []

        logger.info(f"Validating dataset: {df.shape[0]} rows, {df.shape[1]} columns")

        # 1. Dataset size validation
        issues.extend(self._validate_dataset_size(df, test_size))

        # 2. Target column validation (if supervised)
        if target_column and target_column in df.columns:
            issues.extend(self._validate_target_column(
                df, target_column, task_type, use_stratify, use_oversampling
            ))

        # 3. Feature validation
        feature_cols = [col for col in df.columns if col != target_column]
        issues.extend(self._validate_features(df[feature_cols], encoding_columns))

        # 4. Null value validation
        issues.extend(self._validate_null_values(df))

        # 5. Constant feature validation
        issues.extend(self._validate_constant_features(df))

        # 6. Train/test split feasibility
        if target_column and use_stratify:
            issues.extend(self._validate_stratification_feasibility(
                df, target_column, test_size
            ))

        logger.info(f"Validation complete: {len(issues)} issues found")
        return issues

    def _validate_dataset_size(
        self,
        df: pd.DataFrame,
        test_size: float
    ) -> List[EdgeCaseIssue]:
        """Validate dataset has sufficient samples."""
        issues = []
        n_samples = len(df)
        n_train = int(n_samples * (1 - test_size))
        n_test = n_samples - n_train

        # Critical: Too few total samples
        if n_samples < self.MIN_SAMPLES_CRITICAL:
            issues.append(EdgeCaseIssue(
                severity=Severity.CRITICAL,
                category="dataset_size",
                message=f"Dataset has only {n_samples} samples (minimum: {self.MIN_SAMPLES_CRITICAL})",
                details={
                    "n_samples": n_samples,
                    "min_required": self.MIN_SAMPLES_CRITICAL
                },
                recommendation=(
                    "Dataset is too small for reliable machine learning. "
                    "Recommendations:\n"
                    "1. Collect more data (target: 100+ samples)\n"
                    "2. Use cross-validation instead of train/test split\n"
                    "3. Consider simpler models (linear/logistic regression)\n"
                    "4. Use domain knowledge instead of ML if data collection is not possible"
                )
            ))

        # Warning: Small dataset
        elif n_samples < self.MIN_SAMPLES_WARNING:
            issues.append(EdgeCaseIssue(
                severity=Severity.WARNING,
                category="dataset_size",
                message=f"Small dataset with {n_samples} samples (recommended: {self.MIN_SAMPLES_WARNING}+)",
                details={
                    "n_samples": n_samples,
                    "n_train": n_train,
                    "n_test": n_test,
                    "recommended_min": self.MIN_SAMPLES_WARNING
                },
                recommendation=(
                    "Small dataset may lead to overfitting. Recommendations:\n"
                    "1. Use k-fold cross-validation (k=5 or k=10)\n"
                    "2. Use simpler models to avoid overfitting\n"
                    "3. Apply regularization (L1/L2)\n"
                    "4. Consider collecting more data if possible"
                )
            ))

        # Warning: Tiny test set
        if n_test < 5 and n_samples >= self.MIN_SAMPLES_CRITICAL:
            issues.append(EdgeCaseIssue(
                severity=Severity.WARNING,
                category="split_size",
                message=f"Test set would have only {n_test} samples (test_size={test_size})",
                details={
                    "n_test": n_test,
                    "test_size": test_size,
                    "n_samples": n_samples
                },
                recommendation=(
                    f"Test set too small for reliable evaluation. Recommendations:\n"
                    f"1. Increase test_size to {max(0.3, 5/n_samples):.2f} "
                    f"to get at least 5 test samples\n"
                    "2. Use cross-validation instead of single train/test split"
                )
            ))

        return issues

    def _validate_target_column(
        self,
        df: pd.DataFrame,
        target_column: str,
        task_type: Optional[str],
        use_stratify: bool,
        use_oversampling: bool
    ) -> List[EdgeCaseIssue]:
        """Validate target column for classification/regression tasks."""
        issues = []
        y = df[target_column]

        # Infer task type if not provided
        if task_type is None:
            task_type = 'classification' if y.dtype == 'object' or y.nunique() < 20 else 'regression'

        # Classification-specific checks
        if task_type == 'classification':
            value_counts = y.value_counts()
            n_classes = len(value_counts)

            # Critical: Single class
            if n_classes == 1:
                issues.append(EdgeCaseIssue(
                    severity=Severity.CRITICAL,
                    category="target_classes",
                    message="Target has only 1 unique class - cannot train classifier",
                    details={
                        "n_classes": 1,
                        "class": value_counts.index[0]
                    },
                    recommendation=(
                        "Cannot perform classification with single class. Options:\n"
                        "1. Check if data filtering removed variation\n"
                        "2. Verify target column is correct\n"
                        "3. Collect data with multiple classes"
                    )
                ))

            # Check class sizes
            min_class_size = value_counts.min()
            max_class_size = value_counts.max()

            # Critical: Class with < 2 samples (can't split)
            if min_class_size < 2:
                classes_with_one = value_counts[value_counts < 2].index.tolist()
                issues.append(EdgeCaseIssue(
                    severity=Severity.CRITICAL,
                    category="class_size",
                    message=f"Classes with only 1 sample: {classes_with_one}",
                    details={
                        "single_sample_classes": classes_with_one,
                        "value_counts": value_counts.to_dict()
                    },
                    recommendation=(
                        "Classes with single samples cannot be split into train/test. Options:\n"
                        "1. Remove single-sample classes (lose those predictions)\n"
                        "2. Collect more samples for minority classes\n"
                        "3. Combine rare classes into 'Other' category\n"
                        "4. Disable stratification (not recommended)"
                    ),
                    auto_fixable=True
                ))

            # Warning: Very small minority class
            elif min_class_size < self.MIN_SAMPLES_PER_CLASS_CRITICAL:
                small_classes = value_counts[value_counts < self.MIN_SAMPLES_PER_CLASS_CRITICAL]
                issues.append(EdgeCaseIssue(
                    severity=Severity.WARNING,
                    category="class_size",
                    message=f"Classes with < {self.MIN_SAMPLES_PER_CLASS_CRITICAL} samples: {list(small_classes.index)}",
                    details={
                        "small_classes": small_classes.to_dict(),
                        "min_class_size": min_class_size
                    },
                    recommendation=(
                        "Very small minority classes may cause issues. Recommendations:\n"
                        "1. Collect more samples for minority classes\n"
                        "2. Use cross-validation instead of single split\n"
                        "3. Consider class merging if semantically appropriate\n"
                        "4. Disable stratification if necessary (may impact performance)"
                    )
                ))

            # Class imbalance detection
            if max_class_size > 0:
                imbalance_ratio = max_class_size / min_class_size

                # Critical: Extreme imbalance
                if imbalance_ratio > self.CLASS_IMBALANCE_CRITICAL:
                    issues.append(EdgeCaseIssue(
                        severity=Severity.CRITICAL,
                        category="class_imbalance",
                        message=f"Extreme class imbalance: {imbalance_ratio:.1f}:1 ratio",
                        details={
                            "imbalance_ratio": imbalance_ratio,
                            "majority_class": value_counts.index[0],
                            "majority_count": max_class_size,
                            "minority_class": value_counts.index[-1],
                            "minority_count": min_class_size,
                            "value_counts": value_counts.to_dict()
                        },
                        recommendation=(
                            f"Extreme imbalance ({imbalance_ratio:.0f}:1) will cause poor minority class performance. "
                            "Critical actions:\n"
                            "1. Use stratified sampling (REQUIRED)\n"
                            "2. Apply class weights in model training\n"
                            "3. Use appropriate metrics (F1, ROC-AUC, not accuracy)\n"
                            "4. Consider oversampling (SMOTE) if minority class has enough samples\n"
                            "5. Consider undersampling majority class\n"
                            "6. Collect more minority class samples if possible"
                        )
                    ))

                # Warning: Moderate imbalance
                elif imbalance_ratio > self.CLASS_IMBALANCE_WARNING:
                    issues.append(EdgeCaseIssue(
                        severity=Severity.WARNING,
                        category="class_imbalance",
                        message=f"Class imbalance detected: {imbalance_ratio:.1f}:1 ratio",
                        details={
                            "imbalance_ratio": imbalance_ratio,
                            "value_counts": value_counts.to_dict()
                        },
                        recommendation=(
                            "Moderate imbalance may affect model performance. Recommendations:\n"
                            "1. Use stratified sampling\n"
                            "2. Consider class weights\n"
                            "3. Use balanced metrics (F1 score, not just accuracy)\n"
                            "4. Consider resampling techniques if appropriate"
                        )
                    ))

            # SMOTE feasibility check
            if use_oversampling and min_class_size < self.MIN_SAMPLES_FOR_SMOTE:
                issues.append(EdgeCaseIssue(
                    severity=Severity.ERROR,
                    category="oversampling",
                    message=f"Minority class too small for SMOTE: {min_class_size} samples (minimum: {self.MIN_SAMPLES_FOR_SMOTE})",
                    details={
                        "min_class_size": min_class_size,
                        "min_required_for_smote": self.MIN_SAMPLES_FOR_SMOTE,
                        "k_neighbors": 5
                    },
                    recommendation=(
                        "SMOTE requires at least 6 samples in minority class (for k_neighbors=5). Options:\n"
                        "1. Disable SMOTE/oversampling\n"
                        "2. Collect more minority class samples\n"
                        "3. Use class weights instead of oversampling\n"
                        "4. Use RandomOverSampler (duplicates samples) instead of SMOTE"
                    )
                ))

        return issues

    def _validate_features(
        self,
        df: pd.DataFrame,
        encoding_columns: Optional[List[str]] = None
    ) -> List[EdgeCaseIssue]:
        """Validate features for high cardinality and other issues."""
        issues = []

        # Detect categorical columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        if encoding_columns is None:
            encoding_columns = categorical_cols

        # Check each column to be encoded
        for col in encoding_columns:
            if col not in df.columns:
                continue

            n_unique = df[col].nunique()
            n_samples = len(df)
            cardinality_ratio = n_unique / n_samples if n_samples > 0 else 0

            # Critical: Extremely high cardinality
            if n_unique > self.HIGH_CARDINALITY_CRITICAL:
                issues.append(EdgeCaseIssue(
                    severity=Severity.CRITICAL,
                    category="high_cardinality",
                    message=f"Column '{col}' has extremely high cardinality: {n_unique} unique values",
                    details={
                        "column": col,
                        "n_unique": n_unique,
                        "n_samples": n_samples,
                        "cardinality_ratio": cardinality_ratio,
                        "estimated_features_after_encoding": n_unique - 1  # if drop_first=True
                    },
                    recommendation=(
                        f"One-hot encoding '{col}' would create {n_unique-1} features (feature explosion). "
                        "Critical actions:\n"
                        "1. DO NOT one-hot encode this column\n"
                        "2. Use target encoding or frequency encoding instead\n"
                        "3. Group rare categories into 'Other' (keep top 10-20 categories)\n"
                        "4. Use embeddings if using neural networks\n"
                        "5. Consider if this column should be a feature at all (e.g., IDs)"
                    )
                ))

            # Warning: High cardinality
            elif n_unique > self.HIGH_CARDINALITY_WARNING:
                issues.append(EdgeCaseIssue(
                    severity=Severity.WARNING,
                    category="high_cardinality",
                    message=f"Column '{col}' has high cardinality: {n_unique} unique values",
                    details={
                        "column": col,
                        "n_unique": n_unique,
                        "cardinality_ratio": cardinality_ratio
                    },
                    recommendation=(
                        f"One-hot encoding '{col}' will create {n_unique-1} features. Recommendations:\n"
                        "1. Consider grouping rare categories (keep top 20-30)\n"
                        "2. Use alternative encoding (target, frequency, ordinal)\n"
                        "3. Increase regularization if using one-hot encoding\n"
                        "4. Ensure this column provides predictive value"
                    ),
                    auto_fixable=True
                ))

            # Warning: High cardinality ratio (mostly unique values)
            if cardinality_ratio > self.MAX_CARDINALITY_RATIO_CRITICAL:
                issues.append(EdgeCaseIssue(
                    severity=Severity.WARNING,
                    category="cardinality_ratio",
                    message=f"Column '{col}' has {cardinality_ratio*100:.1f}% unique values (likely an ID)",
                    details={
                        "column": col,
                        "cardinality_ratio": cardinality_ratio,
                        "n_unique": n_unique
                    },
                    recommendation=(
                        "Column appears to be an identifier (mostly unique values). Recommendations:\n"
                        "1. Remove this column from features (likely not predictive)\n"
                        "2. If it encodes information, extract meaningful features\n"
                        "3. Verify this is not a data leak (e.g., transaction ID)"
                    )
                ))

        return issues

    def _validate_null_values(self, df: pd.DataFrame) -> List[EdgeCaseIssue]:
        """Validate null value patterns."""
        issues = []

        for col in df.columns:
            null_count = df[col].isna().sum()
            null_ratio = null_count / len(df) if len(df) > 0 else 0

            # Critical: Mostly null
            if null_ratio > self.NULL_RATIO_CRITICAL:
                issues.append(EdgeCaseIssue(
                    severity=Severity.CRITICAL,
                    category="null_values",
                    message=f"Column '{col}' is {null_ratio*100:.1f}% null ({null_count}/{len(df)} rows)",
                    details={
                        "column": col,
                        "null_count": null_count,
                        "null_ratio": null_ratio
                    },
                    recommendation=(
                        f"Column '{col}' is mostly missing data. Recommendations:\n"
                        "1. Remove this column (not enough information)\n"
                        "2. Investigate why data is missing (MCAR, MAR, MNAR?)\n"
                        "3. Consider if missingness itself is informative (create 'is_missing' feature)"
                    ),
                    auto_fixable=True
                ))

            # Warning: Many nulls
            elif null_ratio > self.NULL_RATIO_WARNING:
                issues.append(EdgeCaseIssue(
                    severity=Severity.WARNING,
                    category="null_values",
                    message=f"Column '{col}' has {null_ratio*100:.1f}% null values",
                    details={
                        "column": col,
                        "null_count": null_count,
                        "null_ratio": null_ratio
                    },
                    recommendation=(
                        "High proportion of missing values. Recommendations:\n"
                        "1. Investigate missing data pattern\n"
                        "2. Use appropriate imputation strategy\n"
                        "3. Consider creating 'is_missing' indicator feature\n"
                        "4. Document imputation approach for interpretability"
                    )
                ))

        return issues

    def _validate_constant_features(self, df: pd.DataFrame) -> List[EdgeCaseIssue]:
        """Validate for constant or near-constant features."""
        issues = []

        for col in df.columns:
            # Skip if all null
            if df[col].isna().all():
                continue

            n_unique = df[col].nunique()

            # Critical: Constant feature
            if n_unique == 1:
                issues.append(EdgeCaseIssue(
                    severity=Severity.ERROR,
                    category="constant_feature",
                    message=f"Column '{col}' has only 1 unique value (constant feature)",
                    details={
                        "column": col,
                        "value": df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else None
                    },
                    recommendation=(
                        "Constant features provide no information for prediction. Actions:\n"
                        "1. Remove this column from features\n"
                        "2. Investigate if this indicates a data quality issue\n"
                        "3. Check if data filtering removed variation"
                    ),
                    auto_fixable=True
                ))

            # Warning: Near-constant feature
            elif n_unique == 2:
                value_counts = df[col].value_counts()
                if len(value_counts) > 1:
                    ratio = value_counts.iloc[0] / len(df)
                    if ratio > 0.99:
                        issues.append(EdgeCaseIssue(
                            severity=Severity.WARNING,
                            category="near_constant_feature",
                            message=f"Column '{col}' is {ratio*100:.1f}% one value (near-constant)",
                            details={
                                "column": col,
                                "value_counts": value_counts.to_dict(),
                                "dominant_ratio": ratio
                            },
                            recommendation=(
                                "Near-constant features have minimal predictive power. Consider:\n"
                                "1. Remove this feature if not domain-critical\n"
                                "2. Investigate if rare values are informative\n"
                                "3. Document if retained for interpretability"
                            )
                        ))

        return issues

    def _validate_stratification_feasibility(
        self,
        df: pd.DataFrame,
        target_column: str,
        test_size: float
    ) -> List[EdgeCaseIssue]:
        """Validate that stratified splitting is feasible."""
        issues = []

        value_counts = df[target_column].value_counts()
        n_test_per_class = (value_counts * test_size).astype(int)

        # Find classes that would have 0 samples in test set
        classes_with_zero_test = n_test_per_class[n_test_per_class == 0].index.tolist()

        if classes_with_zero_test:
            issues.append(EdgeCaseIssue(
                severity=Severity.ERROR,
                category="stratification",
                message=f"Stratification impossible: {len(classes_with_zero_test)} classes would have 0 test samples",
                details={
                    "classes_with_zero_test": classes_with_zero_test,
                    "test_size": test_size,
                    "class_counts": value_counts.to_dict()
                },
                recommendation=(
                    "Stratified split impossible with current settings. Options:\n"
                    f"1. Increase test_size to {(1.0 / value_counts.min()):.2f} or higher\n"
                    "2. Disable stratification (use_stratify=False)\n"
                    "3. Remove or merge classes with few samples\n"
                    "4. Use cross-validation instead of single split"
                )
            ))

        # Find classes with only 1 test sample
        classes_with_one_test = n_test_per_class[n_test_per_class == 1].index.tolist()

        if classes_with_one_test:
            issues.append(EdgeCaseIssue(
                severity=Severity.WARNING,
                category="stratification",
                message=f"{len(classes_with_one_test)} classes would have only 1 test sample",
                details={
                    "classes_with_one_test": classes_with_one_test,
                    "test_size": test_size
                },
                recommendation=(
                    "Single test samples provide unreliable evaluation. Recommendations:\n"
                    "1. Use k-fold cross-validation for more robust evaluation\n"
                    "2. Increase test_size if possible\n"
                    "3. Combine rare classes if semantically appropriate"
                )
            ))

        return issues

    def has_critical_issues(self, issues: List[EdgeCaseIssue]) -> bool:
        """Check if any issues are critical."""
        return any(issue.severity == Severity.CRITICAL for issue in issues)

    def has_errors(self, issues: List[EdgeCaseIssue]) -> bool:
        """Check if any issues are errors or critical."""
        return any(issue.severity in [Severity.ERROR, Severity.CRITICAL] for issue in issues)

    def get_issues_by_severity(
        self,
        issues: List[EdgeCaseIssue]
    ) -> Dict[Severity, List[EdgeCaseIssue]]:
        """Group issues by severity."""
        result = {severity: [] for severity in Severity}
        for issue in issues:
            result[issue.severity].append(issue)
        return result

    def format_report(self, issues: List[EdgeCaseIssue]) -> str:
        """Format validation report as human-readable string."""
        if not issues:
            return "✓ No edge case issues found"

        by_severity = self.get_issues_by_severity(issues)

        report = []
        report.append("=" * 80)
        report.append("EDGE CASE VALIDATION REPORT")
        report.append("=" * 80)
        report.append(f"\nTotal issues found: {len(issues)}")

        for severity in [Severity.CRITICAL, Severity.ERROR, Severity.WARNING, Severity.INFO]:
            severity_issues = by_severity[severity]
            if not severity_issues:
                continue

            icon = {"CRITICAL": "🔴", "ERROR": "❌", "WARNING": "⚠️", "INFO": "ℹ️"}[severity.name]
            report.append(f"\n{icon} {severity.name}: {len(severity_issues)} issue(s)")
            report.append("-" * 80)

            for i, issue in enumerate(severity_issues, 1):
                report.append(f"\n{i}. [{issue.category}] {issue.message}")
                report.append(f"   Details: {issue.details}")
                report.append(f"   Recommendation: {issue.recommendation}")
                if issue.auto_fixable:
                    report.append(f"   ⚡ Auto-fixable: Yes")

        report.append("\n" + "=" * 80)
        return "\n".join(report)


def validate_for_training(
    df: pd.DataFrame,
    target_column: Optional[str] = None,
    task_type: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None
) -> Tuple[bool, List[EdgeCaseIssue]]:
    """
    Convenience function to validate dataset before training.

    Args:
        df: DataFrame to validate
        target_column: Target column name
        task_type: Type of ML task
        config: Training configuration

    Returns:
        Tuple of (is_valid, issues_list)
    """
    validator = EdgeCaseValidator()

    # Extract config parameters
    test_size = config.get('test_size', 0.2) if config else 0.2
    use_stratify = config.get('use_stratify', False) if config else False
    use_oversampling = config.get('use_oversampling', False) if config else False
    encoding_columns = config.get('encoding_columns') if config else None

    issues = validator.validate_dataset(
        df=df,
        target_column=target_column,
        task_type=task_type,
        test_size=test_size,
        use_stratify=use_stratify,
        use_oversampling=use_oversampling,
        encoding_columns=encoding_columns
    )

    # Log report
    logger.info(f"\n{validator.format_report(issues)}")

    # Check if valid (no critical or error issues)
    is_valid = not validator.has_errors(issues)

    return is_valid, issues
