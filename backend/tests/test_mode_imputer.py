"""
Unit tests for ModeImputer.
"""

import pytest
import pandas as pd
import numpy as np
from app.ml_engine.preprocessing.imputer import ModeImputer


class TestModeImputerBasic:
    """Basic functionality tests for ModeImputer"""

    def test_initialization(self):
        """Test ModeImputer initialization"""
        imputer = ModeImputer()
        assert imputer.modes == {}
        assert not imputer.fitted

        imputer_with_cols = ModeImputer(columns=['col1', 'col2'])
        assert imputer_with_cols.params['columns'] == ['col1', 'col2']

    def test_fit_categorical_columns(self):
        """Test fitting on categorical columns"""
        df = pd.DataFrame({
            'color': ['red', 'blue', 'red', None, 'red'],
            'size': ['S', 'M', None, 'M', 'M']
        })

        imputer = ModeImputer()
        imputer.fit(df)

        assert imputer.fitted
        assert imputer.modes['color'] == 'red'  # Most frequent
        assert imputer.modes['size'] == 'M'     # Most frequent

    def test_fit_numeric_columns(self):
        """Test fitting on numeric columns"""
        df = pd.DataFrame({
            'count': [1, 2, 2, None, 2],
            'rating': [5, None, 3, 3, 3]
        })

        imputer = ModeImputer()
        imputer.fit(df)

        assert imputer.modes['count'] == 2
        assert imputer.modes['rating'] == 3

    def test_fit_mixed_types(self):
        """Test fitting on mixed data types"""
        df = pd.DataFrame({
            'category': ['A', 'B', 'A', None, 'A'],
            'number': [1, 2, 1, None, 1],
            'binary': [True, False, True, None, True]
        })

        imputer = ModeImputer()
        imputer.fit(df)

        assert imputer.modes['category'] == 'A'
        assert imputer.modes['number'] == 1
        assert imputer.modes['binary'] == True

    def test_transform_basic(self):
        """Test basic transformation"""
        df_train = pd.DataFrame({
            'color': ['red', 'blue', 'red', 'red'],
            'size': ['S', 'M', 'M', 'M']
        })

        df_test = pd.DataFrame({
            'color': ['blue', None, 'green', None],
            'size': [None, 'S', None, 'L']
        })

        imputer = ModeImputer()
        imputer.fit(df_train)
        df_result = imputer.transform(df_test)

        # Check missing values are filled
        assert df_result['color'].isna().sum() == 0
        assert df_result['size'].isna().sum() == 0

        # Check filled with correct mode
        assert df_result.loc[1, 'color'] == 'red'  # Filled with mode
        assert df_result.loc[0, 'size'] == 'M'     # Filled with mode

    def test_fit_transform(self):
        """Test fit_transform combined method"""
        df = pd.DataFrame({
            'category': ['A', 'B', 'A', None, 'A'],
            'value': [10, 20, 10, None, 10]
        })

        imputer = ModeImputer()
        df_result = imputer.fit_transform(df)

        assert imputer.fitted
        assert df_result['category'].isna().sum() == 0
        assert df_result['value'].isna().sum() == 0
        assert df_result.loc[3, 'category'] == 'A'
        assert df_result.loc[3, 'value'] == 10


class TestModeImputerSpecificColumns:
    """Test ModeImputer with specific column selection"""

    def test_specific_columns_only(self):
        """Test imputing only specified columns"""
        df = pd.DataFrame({
            'col1': ['A', 'B', 'A', None, 'A'],
            'col2': ['X', 'Y', 'X', None, 'X'],
            'col3': [1, 2, 1, None, 1]
        })

        # Only impute col1 and col3
        imputer = ModeImputer(columns=['col1', 'col3'])
        imputer.fit(df)

        assert 'col1' in imputer.modes
        assert 'col3' in imputer.modes
        assert 'col2' not in imputer.modes  # Not imputed

        df_result = imputer.transform(df)

        # col1 and col3 should be imputed
        assert df_result['col1'].isna().sum() == 0
        assert df_result['col3'].isna().sum() == 0

        # col2 should still have missing value
        assert df_result['col2'].isna().sum() == 1

    def test_columns_without_missing_values(self):
        """Test with columns that have no missing values"""
        df = pd.DataFrame({
            'complete': ['A', 'B', 'A', 'A'],
            'incomplete': ['X', None, 'X', 'X']
        })

        # When columns=None, only fit columns with missing values
        imputer = ModeImputer()
        imputer.fit(df)

        # Should only learn mode for incomplete column
        assert 'incomplete' in imputer.modes
        assert 'complete' not in imputer.modes

    def test_invalid_column_names(self):
        """Test error handling for invalid column names"""
        df = pd.DataFrame({
            'col1': ['A', 'B', 'A'],
            'col2': [1, 2, 1]
        })

        imputer = ModeImputer(columns=['col1', 'nonexistent'])

        with pytest.raises(ValueError, match="Columns not found"):
            imputer.fit(df)


class TestModeImputerEdgeCases:
    """Test edge cases and special scenarios"""

    def test_all_values_missing(self):
        """Test column with all missing values"""
        df = pd.DataFrame({
            'all_nan': [None, None, None],
            'normal': ['A', 'B', 'A']
        })

        imputer = ModeImputer()
        imputer.fit(df)

        # all_nan should have None as mode (can't compute)
        assert imputer.modes['all_nan'] is None
        assert imputer.modes['normal'] == 'A'

        # Transform should not fill all_nan column
        df_result = imputer.transform(df)
        assert df_result['all_nan'].isna().all()
        assert df_result['normal'].isna().sum() == 0

    def test_no_missing_values(self):
        """Test with dataset having no missing values"""
        df = pd.DataFrame({
            'col1': ['A', 'B', 'A'],
            'col2': [1, 2, 1]
        })

        imputer = ModeImputer()
        imputer.fit(df)

        # No columns should be learned (no missing values)
        assert len(imputer.modes) == 0

        df_result = imputer.transform(df)
        pd.testing.assert_frame_equal(df, df_result)

    def test_multimodal_distribution(self):
        """Test with columns having multiple modes"""
        df = pd.DataFrame({
            # 'A' and 'B' both appear twice (tie)
            'multimodal': ['A', 'B', 'A', 'B', None]
        })

        imputer = ModeImputer()
        imputer.fit(df)

        # Should use first mode (pandas default behavior)
        # This will be either 'A' or 'B' depending on pandas version
        assert imputer.modes['multimodal'] in ['A', 'B']

        df_result = imputer.transform(df)
        assert df_result['multimodal'].isna().sum() == 0

    def test_single_value_column(self):
        """Test column with only one unique value"""
        df = pd.DataFrame({
            'single': ['A', 'A', 'A', None, 'A']
        })

        imputer = ModeImputer()
        imputer.fit(df)

        assert imputer.modes['single'] == 'A'

        df_result = imputer.transform(df)
        assert (df_result['single'] == 'A').all()

    def test_empty_dataframe(self):
        """Test with empty DataFrame"""
        df = pd.DataFrame()

        imputer = ModeImputer()
        imputer.fit(df)

        assert len(imputer.modes) == 0
        assert imputer.fitted


class TestModeImputerDataTypes:
    """Test ModeImputer with various data types"""

    def test_string_columns(self):
        """Test with string columns"""
        df = pd.DataFrame({
            'text': ['hello', 'world', 'hello', None, 'hello']
        })

        imputer = ModeImputer()
        df_result = imputer.fit_transform(df)

        assert df_result['text'].isna().sum() == 0
        assert df_result.loc[3, 'text'] == 'hello'

    def test_boolean_columns(self):
        """Test with boolean columns"""
        df = pd.DataFrame({
            'flag': [True, False, True, None, True]
        })

        imputer = ModeImputer()
        df_result = imputer.fit_transform(df)

        assert df_result['flag'].isna().sum() == 0
        assert df_result.loc[3, 'flag'] == True

    def test_integer_columns(self):
        """Test with integer columns"""
        df = pd.DataFrame({
            'count': [1, 2, 1, None, 1]
        })

        imputer = ModeImputer()
        df_result = imputer.fit_transform(df)

        assert df_result['count'].isna().sum() == 0
        assert df_result.loc[3, 'count'] == 1

    def test_float_columns(self):
        """Test with float columns"""
        df = pd.DataFrame({
            'price': [1.5, 2.5, 1.5, None, 1.5]
        })

        imputer = ModeImputer()
        df_result = imputer.fit_transform(df)

        assert df_result['price'].isna().sum() == 0
        assert df_result.loc[3, 'price'] == 1.5

    def test_datetime_columns(self):
        """Test with datetime columns"""
        df = pd.DataFrame({
            'date': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-01', None, '2024-01-01'])
        })

        imputer = ModeImputer()
        df_result = imputer.fit_transform(df)

        assert df_result['date'].isna().sum() == 0
        assert df_result.loc[3, 'date'] == pd.Timestamp('2024-01-01')

    def test_category_dtype(self):
        """Test with pandas Categorical dtype"""
        df = pd.DataFrame({
            'cat': pd.Categorical(['low', 'high', 'low', None, 'low'])
        })

        imputer = ModeImputer()
        df_result = imputer.fit_transform(df)

        assert df_result['cat'].isna().sum() == 0
        assert df_result.loc[3, 'cat'] == 'low'


class TestModeImputerErrorHandling:
    """Test error handling"""

    def test_transform_before_fit(self):
        """Test that transform before fit raises error"""
        df = pd.DataFrame({'col': ['A', None, 'B']})
        imputer = ModeImputer()

        with pytest.raises(RuntimeError, match="must be fitted"):
            imputer.transform(df)

    def test_non_dataframe_input(self):
        """Test that non-DataFrame input raises error"""
        arr = np.array([[1, 2], [3, 4]])
        imputer = ModeImputer()

        with pytest.raises(TypeError, match="expects a pandas DataFrame"):
            imputer.fit(arr)

        imputer_fitted = ModeImputer()
        df = pd.DataFrame({'col': ['A', 'B']})
        imputer_fitted.fit(df)

        with pytest.raises(TypeError, match="expects a pandas DataFrame"):
            imputer_fitted.transform(arr)


class TestModeImputerUtilityMethods:
    """Test utility methods"""

    def test_get_modes_before_fit(self):
        """Test get_modes before fitting"""
        imputer = ModeImputer()
        assert imputer.get_modes() is None

    def test_get_modes_after_fit(self):
        """Test get_modes after fitting"""
        df = pd.DataFrame({
            'col1': ['A', 'B', 'A', None],
            'col2': [1, 2, 1, None]
        })

        imputer = ModeImputer()
        imputer.fit(df)

        modes = imputer.get_modes()
        assert modes is not None
        assert modes['col1'] == 'A'
        assert modes['col2'] == 1

        # Should return a copy
        modes['col1'] = 'Z'
        assert imputer.modes['col1'] == 'A'  # Original unchanged

    def test_repr(self):
        """Test string representation"""
        imputer = ModeImputer(columns=['col1'], name="test_imputer")
        repr_str = repr(imputer)
        assert 'ModeImputer' in repr_str or 'test_imputer' in repr_str


class TestModeImputerPractical:
    """Practical use case tests"""

    def test_survey_data(self):
        """Test with realistic survey data"""
        df = pd.DataFrame({
            'gender': ['M', 'F', 'M', None, 'M', 'F'],
            'age_group': ['18-25', '26-35', '18-25', '18-25', None, '26-35'],
            'satisfaction': ['High', 'Medium', 'High', None, 'High', 'Medium'],
            'rating': [5, 4, 5, None, 5, 4]
        })

        imputer = ModeImputer()
        df_result = imputer.fit_transform(df)

        # All missing values should be filled
        assert df_result.isna().sum().sum() == 0

        # Check modes are reasonable
        assert df_result.loc[3, 'gender'] == 'M'  # Most common
        assert df_result.loc[4, 'age_group'] == '18-25'
        assert df_result.loc[3, 'satisfaction'] == 'High'
        assert df_result.loc[3, 'rating'] == 5

    def test_ecommerce_data(self):
        """Test with e-commerce data"""
        df = pd.DataFrame({
            'product_category': ['Electronics', 'Clothing', 'Electronics', None, 'Electronics'],
            'payment_method': ['Credit', 'PayPal', 'Credit', 'Credit', None],
            'shipping': ['Standard', None, 'Standard', 'Standard', 'Standard']
        })

        imputer = ModeImputer()
        df_result = imputer.fit_transform(df)

        assert df_result['product_category'].isna().sum() == 0
        assert df_result['payment_method'].isna().sum() == 0
        assert df_result['shipping'].isna().sum() == 0

        assert df_result.loc[3, 'product_category'] == 'Electronics'
        assert df_result.loc[4, 'payment_method'] == 'Credit'
        assert df_result.loc[1, 'shipping'] == 'Standard'

    def test_separate_train_test(self):
        """Test with separate train and test sets"""
        df_train = pd.DataFrame({
            'category': ['A', 'B', 'A', 'A', 'B'],
            'type': ['X', 'Y', 'X', 'X', 'Y']
        })

        df_test = pd.DataFrame({
            'category': ['B', None, 'C', None],
            'type': [None, 'Z', None, 'X']
        })

        imputer = ModeImputer()
        imputer.fit(df_train)  # Fit only on training data

        df_test_result = imputer.transform(df_test)

        # Missing values filled with training data modes
        assert df_test_result.loc[1, 'category'] == 'A'  # Mode from train
        assert df_test_result.loc[0, 'type'] == 'X'      # Mode from train

    def test_pipeline_compatibility(self):
        """Test that ModeImputer works in preprocessing pipeline"""
        from app.ml_engine.preprocessing.pipeline import PreprocessingPipeline
        from app.ml_engine.preprocessing.encoder import LabelEncoder

        df = pd.DataFrame({
            'category': ['A', 'B', 'A', None, 'B'],
            'type': ['X', None, 'Y', 'X', 'X']
        })

        # Pipeline: impute then encode
        pipeline = PreprocessingPipeline(steps=[
            ModeImputer(name="mode_imputer"),
            LabelEncoder(name="label_encoder")
        ])

        df_result = pipeline.fit_transform(df)

        # Should have no missing values and be encoded
        assert df_result.isna().sum().sum() == 0
