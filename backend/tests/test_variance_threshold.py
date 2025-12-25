"""
Unit tests for VarianceThreshold feature selection.
"""

import pytest
import pandas as pd
import numpy as np
from app.ml_engine.feature_selection.variance_threshold import VarianceThreshold


class TestVarianceThresholdBasic:
    """Basic functionality tests for VarianceThreshold"""

    def test_initialization(self):
        """Test VarianceThreshold initialization"""
        selector = VarianceThreshold(threshold=0.5)
        assert selector.threshold == 0.5
        assert selector.columns is None
        assert not selector.fitted

    def test_remove_constant_features(self):
        """Test removal of constant features (variance = 0)"""
        # Create dataset with one constant feature
        df = pd.DataFrame({
            'constant': [1, 1, 1, 1, 1],
            'variable': [1, 2, 3, 4, 5],
            'target': [0, 1, 0, 1, 0]
        })

        selector = VarianceThreshold(threshold=0.0)
        df_transformed = selector.fit_transform(df)

        # Constant feature should be removed
        assert 'constant' not in df_transformed.columns
        assert 'variable' in df_transformed.columns
        assert 'target' in df_transformed.columns
        assert len(selector.get_removed_features()) == 1
        assert 'constant' in selector.get_removed_features()

    def test_remove_low_variance_features(self):
        """Test removal of low-variance features"""
        # Create dataset with varying variances
        df = pd.DataFrame({
            'very_low_var': [1, 1, 1, 1, 2],  # variance ≈ 0.16
            'low_var': [1, 1, 2, 2, 3],        # variance ≈ 0.64
            'high_var': [1, 5, 10, 15, 20],   # variance ≈ 57.5
        })

        # Remove features with variance < 0.5
        selector = VarianceThreshold(threshold=0.5)
        df_transformed = selector.fit_transform(df)

        # Only high_var and low_var should remain
        assert 'very_low_var' not in df_transformed.columns
        assert 'low_var' in df_transformed.columns
        assert 'high_var' in df_transformed.columns

    def test_fit_transform_separate(self):
        """Test separate fit and transform calls"""
        df_train = pd.DataFrame({
            'constant': [1, 1, 1],
            'variable': [1, 2, 3]
        })

        df_test = pd.DataFrame({
            'constant': [1, 1],
            'variable': [4, 5]
        })

        selector = VarianceThreshold(threshold=0.0)

        # Fit on training data
        selector.fit(df_train)
        assert selector.fitted
        assert 'constant' in selector.get_removed_features()

        # Transform test data
        df_test_transformed = selector.transform(df_test)
        assert 'constant' not in df_test_transformed.columns
        assert 'variable' in df_test_transformed.columns


class TestVarianceThresholdDataTypes:
    """Test VarianceThreshold with different data types"""

    def test_numpy_array_input(self):
        """Test with NumPy array input"""
        X = np.array([
            [1, 1, 5],
            [1, 2, 6],
            [1, 3, 7],
            [1, 4, 8]
        ])

        selector = VarianceThreshold(threshold=0.0)
        X_transformed = selector.fit_transform(X)

        # Should remove first column (constant)
        assert X_transformed.shape == (4, 2)
        assert isinstance(X_transformed, np.ndarray)

    def test_dataframe_with_mixed_types(self):
        """Test with DataFrame containing mixed data types"""
        df = pd.DataFrame({
            'numeric_const': [1, 1, 1, 1],
            'numeric_var': [1, 2, 3, 4],
            'string': ['a', 'b', 'c', 'd'],
            'category': pd.Categorical(['x', 'y', 'x', 'y'])
        })

        selector = VarianceThreshold(threshold=0.0)
        df_transformed = selector.fit_transform(df)

        # Only numeric_const should be removed
        assert 'numeric_const' not in df_transformed.columns
        assert 'numeric_var' in df_transformed.columns
        # Non-numeric columns should pass through
        assert 'string' in df_transformed.columns
        assert 'category' in df_transformed.columns

    def test_specific_columns(self):
        """Test variance threshold on specific columns only"""
        df = pd.DataFrame({
            'const_1': [1, 1, 1],
            'const_2': [2, 2, 2],
            'var_1': [1, 2, 3],
            'var_2': [4, 5, 6]
        })

        # Apply threshold only to const_1 and var_1
        selector = VarianceThreshold(threshold=0.0, columns=['const_1', 'var_1'])
        df_transformed = selector.fit_transform(df)

        # const_1 should be removed
        assert 'const_1' not in df_transformed.columns
        assert 'var_1' in df_transformed.columns
        # const_2 should NOT be removed (not in columns list)
        assert 'const_2' in df_transformed.columns
        assert 'var_2' in df_transformed.columns


class TestVarianceThresholdEdgeCases:
    """Test edge cases and error handling"""

    def test_all_features_removed(self):
        """Test when all features are below threshold"""
        df = pd.DataFrame({
            'a': [1, 1, 1],
            'b': [2, 2, 2],
            'c': [3, 3, 3]
        })

        selector = VarianceThreshold(threshold=0.0)
        df_transformed = selector.fit_transform(df)

        # All features have zero variance, so all removed
        assert df_transformed.shape[1] == 0

    def test_no_features_removed(self):
        """Test when no features are below threshold"""
        df = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [7, 8, 9]
        })

        selector = VarianceThreshold(threshold=0.0)
        df_transformed = selector.fit_transform(df)

        # All features have variance > 0
        assert df_transformed.shape[1] == 3
        assert len(selector.get_removed_features()) == 0

    def test_transform_before_fit_error(self):
        """Test that transform before fit raises error"""
        df = pd.DataFrame({'a': [1, 2, 3]})
        selector = VarianceThreshold()

        with pytest.raises(RuntimeError, match="must be fitted"):
            selector.transform(df)

    def test_negative_threshold_error(self):
        """Test that negative threshold raises error"""
        df = pd.DataFrame({'a': [1, 2, 3]})
        selector = VarianceThreshold(threshold=-0.5)

        with pytest.raises(ValueError, match="Threshold must be non-negative"):
            selector.fit(df)

    def test_no_numeric_columns_error(self):
        """Test error when no numeric columns present"""
        df = pd.DataFrame({
            'string': ['a', 'b', 'c'],
            'category': pd.Categorical(['x', 'y', 'z'])
        })

        selector = VarianceThreshold()

        with pytest.raises(ValueError, match="No numeric columns found"):
            selector.fit(df)

    def test_missing_columns_warning(self):
        """Test handling of missing columns in transform"""
        df_train = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [1, 1, 1],
            'c': [4, 5, 6]
        })

        df_test = pd.DataFrame({
            'a': [7, 8],
            'c': [9, 10]
            # 'b' is missing
        })

        selector = VarianceThreshold(threshold=0.0)
        selector.fit(df_train)

        # Should handle missing column gracefully
        df_transformed = selector.transform(df_test)
        assert 'a' in df_transformed.columns
        assert 'c' in df_transformed.columns


class TestVarianceThresholdMethods:
    """Test utility methods"""

    def test_get_feature_variances(self):
        """Test getting feature variance values"""
        df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [1, 1, 1, 1, 1],
            'c': [2, 4, 6, 8, 10]
        })

        selector = VarianceThreshold()
        selector.fit(df)

        variances = selector.get_feature_variances()
        assert variances is not None
        assert len(variances) == 3
        assert variances['a'] > 0
        assert variances['b'] == 0
        assert variances['c'] > variances['a']

    def test_get_support_mask(self):
        """Test getting boolean mask of selected features"""
        df = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [1, 1, 1],
            'c': [4, 5, 6]
        })

        selector = VarianceThreshold(threshold=0.0)
        selector.fit(df)

        mask = selector.get_support(indices=False)
        assert len(mask) == 3
        assert mask[0] == True   # 'a' kept
        assert mask[1] == False  # 'b' removed
        assert mask[2] == True   # 'c' kept

    def test_get_support_indices(self):
        """Test getting indices of selected features"""
        df = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [1, 1, 1],
            'c': [4, 5, 6],
            'd': [2, 2, 2]
        })

        selector = VarianceThreshold(threshold=0.0)
        selector.fit(df)

        indices = selector.get_support(indices=True)
        assert isinstance(indices, list)
        assert 0 in indices  # 'a'
        assert 2 in indices  # 'c'
        assert 1 not in indices  # 'b' removed
        assert 3 not in indices  # 'd' removed

    def test_get_selected_and_removed_features(self):
        """Test getting lists of selected and removed features"""
        df = pd.DataFrame({
            'keep_1': [1, 2, 3],
            'remove_1': [1, 1, 1],
            'keep_2': [4, 5, 6],
            'remove_2': [2, 2, 2]
        })

        selector = VarianceThreshold(threshold=0.0)
        selector.fit(df)

        selected = selector.get_selected_features()
        removed = selector.get_removed_features()

        assert 'keep_1' in selected
        assert 'keep_2' in selected
        assert 'remove_1' in removed
        assert 'remove_2' in removed


class TestVarianceThresholdSerialization:
    """Test serialization and deserialization"""

    def test_to_dict(self):
        """Test serialization to dictionary"""
        df = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [1, 1, 1]
        })

        selector = VarianceThreshold(threshold=0.5, name="test_selector")
        selector.fit(df)

        config = selector.to_dict()

        assert config['name'] == 'test_selector'
        assert config['fitted'] is True
        assert config['params']['threshold'] == 0.5
        assert 'variances' in config
        assert 'selected_features' in config
        assert 'removed_features' in config

    def test_from_dict(self):
        """Test deserialization from dictionary"""
        config = {
            'name': 'restored_selector',
            'fitted': True,
            'params': {'threshold': 0.3, 'columns': None},
            'variances': {'a': 2.5, 'b': 0.1, 'c': 1.0},
            'selected_features': ['a', 'c'],
            'removed_features': ['b']
        }

        selector = VarianceThreshold.from_dict(config)

        assert selector.name == 'restored_selector'
        assert selector.threshold == 0.3
        assert selector.fitted is True
        assert selector.selected_features_ == ['a', 'c']
        assert selector.removed_features_ == ['b']

    def test_save_load(self, tmp_path):
        """Test saving and loading fitted selector"""
        df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [1, 1, 1, 1, 1],
            'c': [10, 20, 30, 40, 50]
        })

        selector = VarianceThreshold(threshold=0.0, name="saved_selector")
        selector.fit(df)

        # Save
        save_path = tmp_path / "variance_selector.pkl"
        selector.save(save_path)

        # Load
        loaded_selector = VarianceThreshold.load(save_path)

        assert loaded_selector.name == "saved_selector"
        assert loaded_selector.fitted
        assert loaded_selector.threshold == 0.0
        assert loaded_selector.get_removed_features() == ['b']


class TestVarianceThresholdPractical:
    """Practical usage scenarios"""

    def test_binary_features(self):
        """Test with binary (0/1) features"""
        # Binary feature variance = p * (1-p)
        # When p=0.9, variance = 0.9 * 0.1 = 0.09
        df = pd.DataFrame({
            'almost_all_zeros': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # p=0.1, var=0.09
            'balanced': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],           # p=0.5, var=0.25
            'almost_all_ones': [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],   # p=0.9, var=0.09
        })

        # Remove features with variance < 0.16 (implies >80% same value)
        selector = VarianceThreshold(threshold=0.16)
        df_transformed = selector.fit_transform(df)

        # Only balanced feature should remain
        assert 'balanced' in df_transformed.columns
        assert 'almost_all_zeros' not in df_transformed.columns
        assert 'almost_all_ones' not in df_transformed.columns

    def test_large_dataset_performance(self):
        """Test with larger dataset"""
        # Create dataset with 1000 samples and 50 features
        np.random.seed(42)
        n_samples = 1000
        n_features = 50

        # Mix of high and low variance features
        data = {}
        for i in range(n_features):
            if i < 10:
                # Low variance features (constant or near-constant)
                data[f'low_var_{i}'] = np.random.choice([1, 2], size=n_samples, p=[0.95, 0.05])
            else:
                # High variance features
                data[f'high_var_{i}'] = np.random.randn(n_samples) * 10 + 50

        df = pd.DataFrame(data)

        selector = VarianceThreshold(threshold=1.0)
        df_transformed = selector.fit_transform(df)

        # Should remove most of the low variance features
        assert df_transformed.shape[1] < df.shape[1]
        assert len(selector.get_removed_features()) >= 10

    def test_pipeline_integration(self):
        """Test that selector can be used in a pipeline"""
        from app.ml_engine.preprocessing.pipeline import PreprocessingPipeline
        from app.ml_engine.preprocessing.scaler import StandardScaler

        df = pd.DataFrame({
            'constant': [1, 1, 1, 1],
            'feature1': [10, 20, 30, 40],
            'feature2': [5, 10, 15, 20]
        })

        # Create pipeline: remove constant features, then scale
        pipeline = PreprocessingPipeline(steps=[
            VarianceThreshold(threshold=0.0, name="variance_filter"),
            StandardScaler(name="scaler")
        ])

        df_transformed = pipeline.fit_transform(df)

        # Constant should be removed
        assert 'constant' not in df_transformed.columns
        # Features should be scaled (mean ≈ 0, std ≈ 1)
        assert abs(df_transformed['feature1'].mean()) < 0.1
        assert abs(df_transformed['feature1'].std() - 1.0) < 0.1
