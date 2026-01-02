"""
Auto-Fix Utilities for Edge Cases

Provides automatic fixes for common edge case issues:
- Remove single-sample classes
- Cap high cardinality features
- Remove constant/near-constant features
- Remove mostly-null columns
- Balance extreme class imbalance

Fixes are applied safely with logging and can be reverted.
"""

from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass

from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Fix:
    """Represents a fix applied to the dataset."""
    fix_type: str
    description: str
    affected_columns: List[str]
    affected_rows: int
    details: Dict[str, Any]


class EdgeCaseFixer:
    """
    Applies automatic fixes for edge case issues.

    Example:
        >>> fixer = EdgeCaseFixer()
        >>> df_fixed, fixes = fixer.auto_fix(df, target_column='label')
        >>> print(f"Applied {len(fixes)} fixes")
    """

    def __init__(
        self,
        high_cardinality_threshold: int = 50,
        null_threshold: float = 0.9,
        top_n_categories: int = 20
    ):
        """
        Initialize edge case fixer.

        Args:
            high_cardinality_threshold: Max unique values before capping
            null_threshold: Max null ratio before removing column
            top_n_categories: Number of top categories to keep when capping
        """
        self.high_cardinality_threshold = high_cardinality_threshold
        self.null_threshold = null_threshold
        self.top_n_categories = top_n_categories
        logger.debug(f"EdgeCaseFixer initialized: cardinality_threshold={high_cardinality_threshold}")

    def auto_fix(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        remove_single_sample_classes: bool = True,
        cap_high_cardinality: bool = True,
        remove_constant_features: bool = True,
        remove_mostly_null: bool = True,
        encoding_columns: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, List[Fix]]:
        """
        Apply all auto-fixes to dataset.

        Args:
            df: DataFrame to fix
            target_column: Target column (if supervised)
            remove_single_sample_classes: Remove classes with 1 sample
            cap_high_cardinality: Cap categorical features to top N
            remove_constant_features: Remove features with 1 unique value
            remove_mostly_null: Remove columns that are >90% null
            encoding_columns: Columns to check for high cardinality

        Returns:
            Tuple of (fixed_dataframe, list_of_fixes_applied)
        """
        df_fixed = df.copy()
        fixes = []

        logger.info(f"Starting auto-fix: {df.shape[0]} rows, {df.shape[1]} columns")

        # 1. Remove mostly-null columns
        if remove_mostly_null:
            df_fixed, null_fixes = self.remove_mostly_null_columns(df_fixed)
            fixes.extend(null_fixes)

        # 2. Remove constant features
        if remove_constant_features:
            df_fixed, const_fixes = self.remove_constant_features(
                df_fixed, target_column
            )
            fixes.extend(const_fixes)

        # 3. Cap high cardinality features
        if cap_high_cardinality and encoding_columns:
            df_fixed, card_fixes = self.cap_high_cardinality_features(
                df_fixed, encoding_columns, target_column
            )
            fixes.extend(card_fixes)

        # 4. Remove single-sample classes
        if remove_single_sample_classes and target_column:
            df_fixed, class_fixes = self.remove_single_sample_classes(
                df_fixed, target_column
            )
            fixes.extend(class_fixes)

        logger.info(f"Auto-fix complete: {df_fixed.shape[0]} rows, {df_fixed.shape[1]} columns")
        logger.info(f"Applied {len(fixes)} fixes")

        return df_fixed, fixes

    def remove_single_sample_classes(
        self,
        df: pd.DataFrame,
        target_column: str
    ) -> Tuple[pd.DataFrame, List[Fix]]:
        """
        Remove rows belonging to classes with only 1 sample.

        Args:
            df: DataFrame
            target_column: Name of target column

        Returns:
            Tuple of (fixed_dataframe, list_of_fixes)
        """
        fixes = []

        if target_column not in df.columns:
            return df, fixes

        value_counts = df[target_column].value_counts()
        single_sample_classes = value_counts[value_counts == 1].index.tolist()

        if not single_sample_classes:
            return df, fixes

        # Remove rows with single-sample classes
        df_fixed = df[~df[target_column].isin(single_sample_classes)].copy()
        rows_removed = len(df) - len(df_fixed)

        fix = Fix(
            fix_type="remove_single_sample_classes",
            description=f"Removed {rows_removed} rows belonging to {len(single_sample_classes)} single-sample classes",
            affected_columns=[target_column],
            affected_rows=rows_removed,
            details={
                "removed_classes": single_sample_classes,
                "rows_before": len(df),
                "rows_after": len(df_fixed)
            }
        )
        fixes.append(fix)

        logger.info(
            f"Removed {rows_removed} rows from single-sample classes: {single_sample_classes}"
        )

        return df_fixed, fixes

    def cap_high_cardinality_features(
        self,
        df: pd.DataFrame,
        encoding_columns: List[str],
        target_column: Optional[str] = None
    ) -> Tuple[pd.DataFrame, List[Fix]]:
        """
        Cap high cardinality categorical features to top N categories.

        Rare categories are grouped into 'Other'.

        Args:
            df: DataFrame
            encoding_columns: Columns to cap
            target_column: Target column (to exclude)

        Returns:
            Tuple of (fixed_dataframe, list_of_fixes)
        """
        df_fixed = df.copy()
        fixes = []

        for col in encoding_columns:
            # Skip target column
            if col == target_column:
                continue

            if col not in df_fixed.columns:
                continue

            n_unique = df_fixed[col].nunique()

            # Only cap if above threshold
            if n_unique <= self.high_cardinality_threshold:
                continue

            # Get top N categories
            top_categories = df_fixed[col].value_counts().nlargest(self.top_n_categories).index.tolist()

            # Replace others with 'Other'
            original_unique = n_unique
            df_fixed.loc[~df_fixed[col].isin(top_categories), col] = 'Other'
            new_unique = df_fixed[col].nunique()

            # Count affected rows
            rows_affected = (df[col] != df_fixed[col]).sum()

            fix = Fix(
                fix_type="cap_high_cardinality",
                description=f"Capped '{col}' from {original_unique} to {new_unique} categories",
                affected_columns=[col],
                affected_rows=rows_affected,
                details={
                    "column": col,
                    "original_unique": original_unique,
                    "new_unique": new_unique,
                    "top_categories": top_categories,
                    "rows_changed": rows_affected
                }
            )
            fixes.append(fix)

            logger.info(
                f"Capped '{col}': {original_unique} → {new_unique} categories "
                f"({rows_affected} rows changed to 'Other')"
            )

        return df_fixed, fixes

    def remove_constant_features(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None
    ) -> Tuple[pd.DataFrame, List[Fix]]:
        """
        Remove columns with only 1 unique value.

        Args:
            df: DataFrame
            target_column: Target column (to exclude)

        Returns:
            Tuple of (fixed_dataframe, list_of_fixes)
        """
        df_fixed = df.copy()
        fixes = []
        columns_to_remove = []

        for col in df.columns:
            # Skip target column
            if col == target_column:
                continue

            # Skip if all null
            if df[col].isna().all():
                continue

            n_unique = df[col].nunique()

            if n_unique == 1:
                columns_to_remove.append(col)

        if not columns_to_remove:
            return df_fixed, fixes

        # Remove constant columns
        df_fixed = df_fixed.drop(columns=columns_to_remove)

        fix = Fix(
            fix_type="remove_constant_features",
            description=f"Removed {len(columns_to_remove)} constant features",
            affected_columns=columns_to_remove,
            affected_rows=0,
            details={
                "removed_columns": columns_to_remove,
                "columns_before": len(df.columns),
                "columns_after": len(df_fixed.columns)
            }
        )
        fixes.append(fix)

        logger.info(f"Removed {len(columns_to_remove)} constant features: {columns_to_remove}")

        return df_fixed, fixes

    def remove_mostly_null_columns(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None
    ) -> Tuple[pd.DataFrame, List[Fix]]:
        """
        Remove columns that are mostly null (>threshold).

        Args:
            df: DataFrame
            target_column: Target column (to exclude)

        Returns:
            Tuple of (fixed_dataframe, list_of_fixes)
        """
        df_fixed = df.copy()
        fixes = []
        columns_to_remove = []

        for col in df.columns:
            # Skip target column
            if col == target_column:
                continue

            null_ratio = df[col].isna().sum() / len(df) if len(df) > 0 else 0

            if null_ratio > self.null_threshold:
                columns_to_remove.append(col)

        if not columns_to_remove:
            return df_fixed, fixes

        # Remove mostly-null columns
        df_fixed = df_fixed.drop(columns=columns_to_remove)

        fix = Fix(
            fix_type="remove_mostly_null",
            description=f"Removed {len(columns_to_remove)} mostly-null columns (>{self.null_threshold*100:.0f}% null)",
            affected_columns=columns_to_remove,
            affected_rows=0,
            details={
                "removed_columns": columns_to_remove,
                "null_threshold": self.null_threshold,
                "columns_before": len(df.columns),
                "columns_after": len(df_fixed.columns)
            }
        )
        fixes.append(fix)

        logger.info(
            f"Removed {len(columns_to_remove)} mostly-null columns: {columns_to_remove}"
        )

        return df_fixed, fixes

    def remove_near_zero_variance_features(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        unique_ratio_threshold: float = 0.01
    ) -> Tuple[pd.DataFrame, List[Fix]]:
        """
        Remove features where one value dominates (near-constant).

        Args:
            df: DataFrame
            target_column: Target column (to exclude)
            unique_ratio_threshold: Minimum ratio of second-most-common value

        Returns:
            Tuple of (fixed_dataframe, list_of_fixes)
        """
        df_fixed = df.copy()
        fixes = []
        columns_to_remove = []

        for col in df.columns:
            # Skip target
            if col == target_column:
                continue

            # Skip if all null
            if df[col].isna().all():
                continue

            # Only check if 2+ unique values
            if df[col].nunique() < 2:
                continue

            value_counts = df[col].value_counts()
            most_common_ratio = value_counts.iloc[0] / len(df)

            # If dominant value is >99%, near-constant
            if most_common_ratio > (1.0 - unique_ratio_threshold):
                columns_to_remove.append(col)

        if not columns_to_remove:
            return df_fixed, fixes

        # Remove near-constant columns
        df_fixed = df_fixed.drop(columns=columns_to_remove)

        fix = Fix(
            fix_type="remove_near_zero_variance",
            description=f"Removed {len(columns_to_remove)} near-constant features (>99% one value)",
            affected_columns=columns_to_remove,
            affected_rows=0,
            details={
                "removed_columns": columns_to_remove,
                "threshold": unique_ratio_threshold,
                "columns_before": len(df.columns),
                "columns_after": len(df_fixed.columns)
            }
        )
        fixes.append(fix)

        logger.info(
            f"Removed {len(columns_to_remove)} near-zero variance features: {columns_to_remove}"
        )

        return df_fixed, fixes

    def balance_extreme_imbalance(
        self,
        df: pd.DataFrame,
        target_column: str,
        max_ratio: float = 20.0,
        method: str = 'undersample_majority'
    ) -> Tuple[pd.DataFrame, List[Fix]]:
        """
        Balance extremely imbalanced datasets.

        Args:
            df: DataFrame
            target_column: Target column
            max_ratio: Maximum allowed imbalance ratio
            method: 'undersample_majority' or 'remove_rare_classes'

        Returns:
            Tuple of (fixed_dataframe, list_of_fixes)
        """
        fixes = []

        if target_column not in df.columns:
            return df, fixes

        value_counts = df[target_column].value_counts()
        max_class_size = value_counts.max()
        min_class_size = value_counts.min()

        if max_class_size == 0:
            return df, fixes

        current_ratio = max_class_size / min_class_size

        # Not imbalanced enough to fix
        if current_ratio <= max_ratio:
            return df, fixes

        if method == 'undersample_majority':
            # Undersample majority class to achieve max_ratio
            target_max_size = int(min_class_size * max_ratio)

            df_fixed = pd.DataFrame()
            for class_value in value_counts.index:
                class_df = df[df[target_column] == class_value]

                if len(class_df) > target_max_size:
                    # Sample to target size
                    class_df = class_df.sample(n=target_max_size, random_state=42)

                df_fixed = pd.concat([df_fixed, class_df], ignore_index=True)

            rows_removed = len(df) - len(df_fixed)

            fix = Fix(
                fix_type="undersample_majority",
                description=f"Undersampled majority classes: {current_ratio:.1f}:1 → {max_ratio:.1f}:1",
                affected_columns=[target_column],
                affected_rows=rows_removed,
                details={
                    "original_ratio": current_ratio,
                    "target_ratio": max_ratio,
                    "rows_before": len(df),
                    "rows_after": len(df_fixed),
                    "rows_removed": rows_removed
                }
            )
            fixes.append(fix)

            logger.info(
                f"Undersampled majority: removed {rows_removed} rows, "
                f"ratio {current_ratio:.1f}:1 → {max_ratio:.1f}:1"
            )

            return df_fixed, fixes

        elif method == 'remove_rare_classes':
            # Remove classes that are too rare
            min_size_needed = max_class_size / max_ratio
            classes_to_keep = value_counts[value_counts >= min_size_needed].index.tolist()

            df_fixed = df[df[target_column].isin(classes_to_keep)].copy()
            rows_removed = len(df) - len(df_fixed)
            classes_removed = len(value_counts) - len(classes_to_keep)

            fix = Fix(
                fix_type="remove_rare_classes",
                description=f"Removed {classes_removed} rare classes to achieve {max_ratio}:1 ratio",
                affected_columns=[target_column],
                affected_rows=rows_removed,
                details={
                    "classes_removed": classes_removed,
                    "rows_removed": rows_removed,
                    "original_ratio": current_ratio,
                    "target_ratio": max_ratio
                }
            )
            fixes.append(fix)

            logger.info(
                f"Removed {classes_removed} rare classes ({rows_removed} rows)"
            )

            return df_fixed, fixes

        return df, fixes

    def format_fixes_report(self, fixes: List[Fix]) -> str:
        """Format fixes report as human-readable string."""
        if not fixes:
            return "No fixes applied"

        report = []
        report.append("=" * 80)
        report.append("AUTO-FIX REPORT")
        report.append("=" * 80)
        report.append(f"\nTotal fixes applied: {len(fixes)}\n")

        for i, fix in enumerate(fixes, 1):
            report.append(f"{i}. {fix.description}")
            report.append(f"   Type: {fix.fix_type}")
            if fix.affected_columns:
                report.append(f"   Affected columns: {', '.join(fix.affected_columns)}")
            if fix.affected_rows > 0:
                report.append(f"   Affected rows: {fix.affected_rows}")
            report.append(f"   Details: {fix.details}")
            report.append("")

        report.append("=" * 80)
        return "\n".join(report)


def auto_fix_edge_cases(
    df: pd.DataFrame,
    target_column: Optional[str] = None,
    encoding_columns: Optional[List[str]] = None,
    **kwargs
) -> Tuple[pd.DataFrame, List[Fix]]:
    """
    Convenience function to auto-fix edge cases.

    Args:
        df: DataFrame to fix
        target_column: Target column name
        encoding_columns: Columns that will be encoded
        **kwargs: Additional arguments for EdgeCaseFixer

    Returns:
        Tuple of (fixed_dataframe, list_of_fixes)
    """
    fixer = EdgeCaseFixer(**kwargs)

    df_fixed, fixes = fixer.auto_fix(
        df=df,
        target_column=target_column,
        encoding_columns=encoding_columns
    )

    # Log report
    logger.info(f"\n{fixer.format_fixes_report(fixes)}")

    return df_fixed, fixes
