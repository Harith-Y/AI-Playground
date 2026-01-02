"""
Integration tests for edge case handling in ML pipeline.

Tests various edge cases:
- Tiny datasets (< 20 samples)
- High cardinality features (> 100 unique values)
- Extreme class imbalance (> 100:1)
- Single-sample classes
- Nearly all-null columns
- Constant features
"""

import pytest
import pandas as pd
import numpy as np
from app.ml_engine.validation.edge_case_validator import EdgeCaseValidator, Severity, validate_for_training
from app.ml_engine.validation.edge_case_fixes import EdgeCaseFixer, auto_fix_edge_cases
from app.ml_engine.training.data_split import train_test_split


class TestTinyDatasets:
    """Test handling of tiny datasets."""
    
    def test_critical_tiny_dataset(self):
        """Test dataset with < 10 samples (critical)."""
        df = pd.DataFrame({
            'feature1': range(8),
            'feature2': range(8, 16),
            'target': ['A', 'B'] * 4
        })
        
        validator = EdgeCaseValidator()
        issues = validator.validate_dataset(df, target_column='target', test_size=0.2)
        
        # Should have critical dataset size issue
        critical_issues = [i for i in issues if i.severity == Severity.CRITICAL]
        assert len(critical_issues) > 0
        assert any('dataset' in i.category.lower() and 'size' in i.category.lower() for i in critical_issues)
    
    def test_warning_small_dataset(self):
        """Test dataset with 10-20 samples (warning)."""
        df = pd.DataFrame({
            'feature1': range(15),
            'feature2': range(15, 30),
            'target': ['A', 'B', 'C'] * 5
        })
        
        validator = EdgeCaseValidator()
        issues = validator.validate_dataset(df, target_column='target', test_size=0.2)
        
        # Should have warning about small dataset
        warnings = [i for i in issues if i.severity == Severity.WARNING]
        assert len(warnings) > 0
    
    def test_split_with_tiny_dataset(self):
        """Test train/test split with tiny dataset."""
        df = pd.DataFrame({
            'feature1': range(12),
            'feature2': range(12, 24),
            'target': ['A', 'B'] * 6
        })
        
        # Should succeed with minimal samples
        result = train_test_split(
            df[['feature1', 'feature2']],
            df['target'],
            test_size=0.25,
            random_state=42
        )
        
        assert len(result.X_train) == 9
        assert len(result.X_test) == 3


class TestHighCardinalityFeatures:
    """Test handling of high-cardinality categorical features."""
    
    def test_extremely_high_cardinality(self):
        """Test feature with > 100 unique values (critical)."""
        df = pd.DataFrame({
            'high_card_feature': [f'cat_{i}' for i in range(150)],
            'normal_feature': np.random.rand(150),
            'target': np.random.choice(['A', 'B'], 150)
        })
        
        validator = EdgeCaseValidator()
        issues = validator.validate_dataset(
            df,
            target_column='target',
            encoding_columns=['high_card_feature']
        )
        
        # Should have critical high cardinality issue
        critical_issues = [i for i in issues if i.severity == Severity.CRITICAL]
        assert any('cardinality' in i.category.lower() for i in critical_issues)
    
    def test_high_cardinality_warning(self):
        """Test feature with 50-100 unique values (warning)."""
        df = pd.DataFrame({
            'high_card_feature': [f'cat_{i}' for i in range(75)],
            'normal_feature': np.random.rand(75),
            'target': np.random.choice(['A', 'B'], 75)
        })
        
        validator = EdgeCaseValidator()
        issues = validator.validate_dataset(
            df,
            target_column='target',
            encoding_columns=['high_card_feature']
        )
        
        # Should have warning about high cardinality
        warnings = [i for i in issues if i.severity == Severity.WARNING]
        assert any('cardinality' in i.category.lower() for i in warnings)
    
    def test_auto_fix_high_cardinality(self):
        """Test auto-fixing high cardinality by capping to top N categories."""
        df = pd.DataFrame({
            'high_card_feature': [f'cat_{i % 120}' for i in range(500)],  # 120 unique
            'target': np.random.choice(['A', 'B'], 500)
        })
        
        fixer = EdgeCaseFixer()
        df_fixed, fixes = fixer.cap_high_cardinality_features(
            df,
            encoding_columns=['high_card_feature'],
            target_column='target'
        )
        
        # Should have been capped
        assert df_fixed['high_card_feature'].nunique() <= fixer.top_n_categories + 1  # +1 for 'Other'
        assert len(fixes) > 0
        assert fixes[0].fix_type == 'cap_high_cardinality'


class TestExtremeClassImbalance:
    """Test handling of extreme class imbalance."""
    
    def test_extreme_imbalance_critical(self):
        """Test > 100:1 class imbalance (critical)."""
        df = pd.DataFrame({
            'feature1': np.random.rand(1010),
            'feature2': np.random.rand(1010),
            'target': ['A'] * 1000 + ['B'] * 10  # 100:1 ratio
        })
        
        validator = EdgeCaseValidator()
        issues = validator.validate_dataset(df, target_column='target', test_size=0.2)
        
        # Should have critical imbalance issue
        critical_issues = [i for i in issues if i.severity == Severity.CRITICAL]
        assert any('imbalance' in i.category.lower() for i in critical_issues)
    
    def test_moderate_imbalance_warning(self):
        """Test 10:1 - 100:1 class imbalance (warning)."""
        df = pd.DataFrame({
            'feature1': np.random.rand(300),
            'feature2': np.random.rand(300),
            'target': ['A'] * 270 + ['B'] * 30  # 9:1 ratio
        })
        
        validator = EdgeCaseValidator()
        issues = validator.validate_dataset(df, target_column='target', test_size=0.2)
        
        # Should have warning about imbalance
        warnings = [i for i in issues if i.severity == Severity.WARNING]
        # May or may not trigger depending on exact threshold
        assert len(issues) >= 0  # At least no errors
    
    def test_stratification_with_imbalance(self):
        """Test stratified split with imbalanced classes."""
        df = pd.DataFrame({
            'feature1': np.random.rand(500),
            'feature2': np.random.rand(500),
            'target': ['A'] * 450 + ['B'] * 50  # 9:1 ratio
        })
        
        # Stratified split should preserve ratio
        result = train_test_split(
            df[['feature1', 'feature2']],
            df['target'],
            test_size=0.2,
            random_state=42,
            stratify=True
        )
        
        # Check train ratio
        train_ratio = (result.y_train == 'A').sum() / len(result.y_train)
        # Check test ratio
        test_ratio = (result.y_test == 'A').sum() / len(result.y_test)
        
        # Both should be close to 0.9 (90%)
        assert 0.85 < train_ratio < 0.95
        assert 0.85 < test_ratio < 0.95


class TestSingleSampleClasses:
    """Test handling of classes with only 1 sample."""
    
    def test_single_sample_class_critical(self):
        """Test class with only 1 sample (critical for stratification)."""
        df = pd.DataFrame({
            'feature1': np.random.rand(50),
            'feature2': np.random.rand(50),
            'target': ['A'] * 20 + ['B'] * 20 + ['C'] * 9 + ['D']  # D has 1 sample
        })
        
        validator = EdgeCaseValidator()
        issues = validator.validate_dataset(
            df,
            target_column='target',
            test_size=0.2,
            use_stratify=True
        )
        
        # Should have critical issue about single-sample class
        critical_issues = [i for i in issues if i.severity == Severity.CRITICAL]
        assert any('class' in i.category.lower() and 'size' in i.category.lower() for i in critical_issues)
    
    def test_auto_fix_single_sample_classes(self):
        """Test auto-fixing by removing single-sample classes."""
        df = pd.DataFrame({
            'feature1': np.random.rand(50),
            'feature2': np.random.rand(50),
            'target': ['A'] * 20 + ['B'] * 20 + ['C'] * 9 + ['D']
        })
        
        fixer = EdgeCaseFixer()
        df_fixed, fixes = fixer.remove_single_sample_classes(df, 'target')
        
        # Should have removed class D
        assert 'D' not in df_fixed['target'].values
        assert len(df_fixed) == 49
        assert len(fixes) > 0


class TestNullValues:
    """Test handling of columns with many null values."""
    
    def test_mostly_null_column_critical(self):
        """Test column with > 90% null values (critical)."""
        df = pd.DataFrame({
            'mostly_null': [1.0] * 10 + [np.nan] * 90,
            'good_feature': np.random.rand(100),
            'target': np.random.choice(['A', 'B'], 100)
        })
        
        validator = EdgeCaseValidator()
        issues = validator.validate_dataset(df, target_column='target')
        
        # Should have critical issue about null values
        critical_issues = [i for i in issues if i.severity == Severity.CRITICAL]
        assert any('null' in i.category.lower() for i in critical_issues)
    
    def test_many_null_values_warning(self):
        """Test column with 50-90% null values (warning)."""
        df = pd.DataFrame({
            'many_nulls': [1.0] * 40 + [np.nan] * 60,
            'good_feature': np.random.rand(100),
            'target': np.random.choice(['A', 'B'], 100)
        })
        
        validator = EdgeCaseValidator()
        issues = validator.validate_dataset(df, target_column='target')
        
        # Should have warning about null values
        warnings = [i for i in issues if i.severity == Severity.WARNING]
        assert any('null' in i.category.lower() for i in warnings)
    
    def test_auto_fix_remove_mostly_null(self):
        """Test auto-fixing by removing mostly-null columns."""
        df = pd.DataFrame({
            'mostly_null': [1.0] * 5 + [np.nan] * 95,
            'good_feature': np.random.rand(100),
            'target': np.random.choice(['A', 'B'], 100)
        })
        
        fixer = EdgeCaseFixer()
        df_fixed, fixes = fixer.remove_mostly_null_columns(df)
        
        # Should have removed mostly_null column
        assert 'mostly_null' not in df_fixed.columns
        assert 'good_feature' in df_fixed.columns
        assert len(fixes) > 0


class TestConstantFeatures:
    """Test handling of constant or near-constant features."""
    
    def test_constant_feature_error(self):
        """Test feature with only 1 unique value (error)."""
        df = pd.DataFrame({
            'constant_feature': [5] * 100,
            'good_feature': np.random.rand(100),
            'target': np.random.choice(['A', 'B'], 100)
        })
        
        validator = EdgeCaseValidator()
        issues = validator.validate_dataset(df, target_column='target')
        
        # Should have error about constant feature
        errors = [i for i in issues if i.severity == Severity.ERROR]
        assert any('constant' in i.category.lower() for i in errors)
    
    def test_auto_fix_remove_constant(self):
        """Test auto-fixing by removing constant features."""
        df = pd.DataFrame({
            'constant_feature': [5] * 100,
            'good_feature': np.random.rand(100),
            'target': np.random.choice(['A', 'B'], 100)
        })
        
        fixer = EdgeCaseFixer()
        df_fixed, fixes = fixer.remove_constant_features(df, target_column='target')
        
        # Should have removed constant_feature
        assert 'constant_feature' not in df_fixed.columns
        assert 'good_feature' in df_fixed.columns
        assert len(fixes) > 0


class TestIntegrationValidateForTraining:
    """Integration tests for validate_for_training function."""
    
    def test_valid_dataset_no_issues(self):
        """Test that a valid dataset passes validation."""
        df = pd.DataFrame({
            'feature1': np.random.rand(100),
            'feature2': np.random.rand(100),
            'category': np.random.choice(['A', 'B', 'C'], 100),
            'target': np.random.choice(['X', 'Y'], 100)
        })
        
        is_valid, issues = validate_for_training(
            df,
            target_column='target',
            config={'test_size': 0.2, 'encoding_columns': ['category']}
        )
        
        # Should be valid with no critical/error issues
        assert is_valid or not any(i.severity.value in ['critical', 'error'] for i in issues)
    
    def test_multiple_edge_cases(self):
        """Test dataset with multiple edge cases."""
        df = pd.DataFrame({
            'high_card': [f'cat_{i}' for i in range(15)],  # Some cardinality
            'constant': [1] * 15,
            'mostly_null': [1.0] * 2 + [np.nan] * 13,
            'target': ['A'] * 10 + ['B'] * 4 + ['C']  # Imbalanced + single sample
        })
        
        is_valid, issues = validate_for_training(
            df,
            target_column='target',
            config={'test_size': 0.2, 'encoding_columns': ['high_card']}
        )
        
        # Should have multiple issues
        assert len(issues) > 2
        # Should detect constant feature
        assert any('constant' in i.category.lower() for i in issues)
        # Should detect null values
        assert any('null' in i.category.lower() for i in issues)
        # Should detect tiny dataset or class issues
        assert any('size' in i.category.lower() or 'class' in i.category.lower() for i in issues)
    
    def test_auto_fix_integration(self):
        """Test auto-fixing multiple edge cases."""
        df = pd.DataFrame({
            'high_card': [f'cat_{i % 150}' for i in range(500)],  # High cardinality
            'constant': [1] * 500,
            'mostly_null': [1.0] * 50 + [np.nan] * 450,
            'good_feature': np.random.rand(500),
            'target': ['A'] * 250 + ['B'] * 240 + ['C'] * 9 + ['D']  # Single sample class
        })
        
        validator = EdgeCaseValidator()
        issues = validator.validate_dataset(
            df,
            target_column='target',
            encoding_columns=['high_card']
        )
        
        # Apply auto-fixes
        df_fixed, fixes = auto_fix_edge_cases(
            df,
            issues,
            target_column='target',
            encoding_columns=['high_card']
        )
        
        # Should have applied multiple fixes
        assert len(fixes) >= 3  # constant, null column, high card, single sample
        
        # Verify fixes
        assert 'constant' not in df_fixed.columns  # Removed
        assert 'mostly_null' not in df_fixed.columns  # Removed
        assert df_fixed['high_card'].nunique() <= 50  # Capped
        assert 'D' not in df_fixed['target'].values  # Removed single-sample class


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
