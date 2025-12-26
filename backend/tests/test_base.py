"""
Tests for PreprocessingStep base class.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import pickle
from pathlib import Path
from app.ml_engine.preprocessing.base import PreprocessingStep


class ConcretePreprocessingStep(PreprocessingStep):
    """Concrete implementation for testing the abstract base class."""

    def __init__(self, name=None, **params):
        super().__init__(name=name, **params)
        self.transform_count = 0

    def fit(self, X, y=None):
        """Simple fit that just marks as fitted."""
        self.fitted = True
        return self

    def transform(self, X):
        """Simple transform that returns input unchanged."""
        self._check_fitted()
        self.transform_count += 1
        return X


class TestPreprocessingStepInitialization:
    """Test initialization and basic attributes."""

    def test_default_initialization(self):
        """Test initialization with default name."""
        step = ConcretePreprocessingStep()
        assert step.name == "ConcretePreprocessingStep"
        assert not step.fitted
        assert step.params == {}

    def test_custom_name(self):
        """Test initialization with custom name."""
        step = ConcretePreprocessingStep(name="MyStep")
        assert step.name == "MyStep"
        assert not step.fitted

    def test_params_storage(self):
        """Test that parameters are stored correctly."""
        step = ConcretePreprocessingStep(threshold=0.5, method="mean")
        assert step.params == {"threshold": 0.5, "method": "mean"}

    def test_multiple_instances_independent(self):
        """Test that multiple instances have independent state."""
        step1 = ConcretePreprocessingStep(value=1)
        step2 = ConcretePreprocessingStep(value=2)
        assert step1.params["value"] == 1
        assert step2.params["value"] == 2


class TestPreprocessingStepFitTransform:
    """Test fit, transform, and fit_transform methods."""

    def test_fit_returns_self(self):
        """Test that fit returns self for method chaining."""
        step = ConcretePreprocessingStep()
        result = step.fit(pd.DataFrame())
        assert result is step

    def test_fit_sets_fitted_flag(self):
        """Test that fit sets the fitted flag."""
        step = ConcretePreprocessingStep()
        assert not step.fitted
        step.fit(pd.DataFrame())
        assert step.fitted

    def test_transform_requires_fit(self):
        """Test that transform raises error if not fitted."""
        step = ConcretePreprocessingStep()
        with pytest.raises(RuntimeError, match="must be fitted before transform"):
            step.transform(pd.DataFrame())

    def test_transform_after_fit(self):
        """Test that transform works after fitting."""
        df = pd.DataFrame({'x': [1, 2, 3]})
        step = ConcretePreprocessingStep()
        step.fit(df)
        result = step.transform(df)
        assert isinstance(result, pd.DataFrame)

    def test_fit_transform_convenience(self):
        """Test fit_transform convenience method."""
        df = pd.DataFrame({'x': [1, 2, 3]})
        step = ConcretePreprocessingStep()
        result = step.fit_transform(df)
        assert step.fitted
        assert step.transform_count == 1
        assert isinstance(result, pd.DataFrame)

    def test_fit_transform_with_y(self):
        """Test fit_transform with y parameter."""
        df = pd.DataFrame({'x': [1, 2, 3]})
        y = pd.Series([0, 1, 0])
        step = ConcretePreprocessingStep()
        result = step.fit_transform(df, y)
        assert step.fitted
        assert isinstance(result, pd.DataFrame)

    def test_multiple_transforms(self):
        """Test that transform can be called multiple times."""
        df = pd.DataFrame({'x': [1, 2, 3]})
        step = ConcretePreprocessingStep()
        step.fit(df)
        step.transform(df)
        step.transform(df)
        assert step.transform_count == 2


class TestPreprocessingStepParameters:
    """Test parameter getting and setting."""

    def test_get_params(self):
        """Test get_params returns copy of parameters."""
        step = ConcretePreprocessingStep(a=1, b=2)
        params = step.get_params()
        assert params == {"a": 1, "b": 2}

        # Verify it's a copy
        params["a"] = 99
        assert step.params["a"] == 1

    def test_set_params_updates(self):
        """Test set_params updates parameters."""
        step = ConcretePreprocessingStep(a=1, b=2)
        result = step.set_params(a=10, c=3)
        assert step.params == {"a": 10, "b": 2, "c": 3}
        assert result is step  # Check method chaining

    def test_set_params_partial_update(self):
        """Test set_params with partial update."""
        step = ConcretePreprocessingStep(a=1, b=2)
        step.set_params(b=20)
        assert step.params == {"a": 1, "b": 20}

    def test_get_params_empty(self):
        """Test get_params with no parameters."""
        step = ConcretePreprocessingStep()
        assert step.get_params() == {}


class TestPreprocessingStepSerialization:
    """Test serialization methods (to_dict, from_dict)."""

    def test_to_dict(self):
        """Test to_dict serialization."""
        step = ConcretePreprocessingStep(name="MyStep", threshold=0.5)
        step.fit(pd.DataFrame())

        config = step.to_dict()
        assert config["class"] == "ConcretePreprocessingStep"
        assert config["name"] == "MyStep"
        assert config["fitted"] == True
        assert config["params"] == {"threshold": 0.5}

    def test_to_dict_not_fitted(self):
        """Test to_dict when not fitted."""
        step = ConcretePreprocessingStep(name="Test")
        config = step.to_dict()
        assert config["fitted"] == False

    def test_from_dict(self):
        """Test from_dict deserialization."""
        config = {
            "class": "ConcretePreprocessingStep",
            "name": "RestoredStep",
            "fitted": False,
            "params": {"value": 42}
        }
        step = ConcretePreprocessingStep.from_dict(config)
        assert step.name == "RestoredStep"
        assert step.params == {"value": 42}
        assert not step.fitted

    def test_from_dict_minimal(self):
        """Test from_dict with minimal config."""
        config = {"class": "ConcretePreprocessingStep"}
        step = ConcretePreprocessingStep.from_dict(config)
        assert step.name == "ConcretePreprocessingStep"
        assert step.params == {}

    def test_round_trip_dict(self):
        """Test to_dict -> from_dict round trip."""
        original = ConcretePreprocessingStep(name="Original", a=1, b=2)
        config = original.to_dict()
        restored = ConcretePreprocessingStep.from_dict(config)

        assert restored.name == original.name
        assert restored.params == original.params
        assert restored.fitted == original.fitted


class TestPreprocessingStepPersistence:
    """Test save and load methods."""

    def test_save_and_load(self):
        """Test saving and loading a preprocessing step."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "step.pkl"

            # Create and save
            original = ConcretePreprocessingStep(name="Saved", value=123)
            original.fit(pd.DataFrame())
            original.transform_count = 5
            original.save(path)

            # Load and verify
            loaded = ConcretePreprocessingStep.load(path)
            assert loaded.name == "Saved"
            assert loaded.params["value"] == 123
            assert loaded.fitted == True
            assert loaded.transform_count == 5

    def test_save_creates_directory(self):
        """Test that save creates parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "dir" / "step.pkl"

            step = ConcretePreprocessingStep()
            step.save(path)
            assert path.exists()

    def test_save_unfitted_step(self):
        """Test saving unfitted step."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "unfitted.pkl"

            original = ConcretePreprocessingStep(value=99)
            original.save(path)

            loaded = ConcretePreprocessingStep.load(path)
            assert not loaded.fitted
            assert loaded.params["value"] == 99

    def test_save_with_string_path(self):
        """Test save with string path instead of Path object."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "step.pkl")

            step = ConcretePreprocessingStep()
            step.save(path)
            loaded = ConcretePreprocessingStep.load(path)
            assert loaded is not None


class TestPreprocessingStepStringRepresentation:
    """Test __repr__ and __str__ methods."""

    def test_repr_not_fitted(self):
        """Test repr for unfitted step."""
        step = ConcretePreprocessingStep(name="Test", a=1)
        repr_str = repr(step)
        assert "Test" in repr_str
        assert "not fitted" in repr_str
        assert "a" in repr_str or "1" in repr_str

    def test_repr_fitted(self):
        """Test repr for fitted step."""
        step = ConcretePreprocessingStep(name="Fitted")
        step.fit(pd.DataFrame())
        repr_str = repr(step)
        assert "Fitted" in repr_str
        assert "fitted" in repr_str

    def test_str_equals_repr(self):
        """Test that str() and repr() return same value."""
        step = ConcretePreprocessingStep()
        assert str(step) == repr(step)

    def test_repr_with_empty_params(self):
        """Test repr with no parameters."""
        step = ConcretePreprocessingStep()
        repr_str = repr(step)
        assert "ConcretePreprocessingStep" in repr_str
        assert "not fitted" in repr_str


class TestPreprocessingStepCheckFitted:
    """Test _check_fitted internal method."""

    def test_check_fitted_raises_when_not_fitted(self):
        """Test _check_fitted raises error when not fitted."""
        step = ConcretePreprocessingStep()
        with pytest.raises(RuntimeError, match="must be fitted"):
            step._check_fitted()

    def test_check_fitted_passes_when_fitted(self):
        """Test _check_fitted does not raise when fitted."""
        step = ConcretePreprocessingStep()
        step.fitted = True
        step._check_fitted()  # Should not raise

    def test_check_fitted_custom_name(self):
        """Test _check_fitted error message includes custom name."""
        step = ConcretePreprocessingStep(name="CustomName")
        try:
            step._check_fitted()
            assert False, "Should have raised RuntimeError"
        except RuntimeError as e:
            assert "CustomName" in str(e)


class TestPreprocessingStepEdgeCases:
    """Test edge cases and error conditions."""

    def test_cannot_instantiate_abstract_class(self):
        """Test that PreprocessingStep cannot be instantiated directly."""
        with pytest.raises(TypeError):
            PreprocessingStep()

    def test_fit_multiple_times(self):
        """Test that fit can be called multiple times."""
        step = ConcretePreprocessingStep()
        df = pd.DataFrame({'x': [1, 2, 3]})

        step.fit(df)
        assert step.fitted

        step.fit(df)  # Fit again
        assert step.fitted

    def test_params_persist_after_fit(self):
        """Test that parameters persist after fitting."""
        step = ConcretePreprocessingStep(value=42)
        step.fit(pd.DataFrame())
        assert step.params["value"] == 42

    def test_name_defaults_to_class_name(self):
        """Test that name defaults to class name when not provided."""
        step = ConcretePreprocessingStep()
        assert step.name == "ConcretePreprocessingStep"

    def test_empty_dataframe_handling(self):
        """Test handling of empty DataFrame."""
        step = ConcretePreprocessingStep()
        df = pd.DataFrame()
        step.fit(df)
        result = step.transform(df)
        assert isinstance(result, pd.DataFrame)
