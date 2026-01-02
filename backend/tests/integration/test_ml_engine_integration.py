"""
Integration Tests for ML Engine Components

Tests the core ML engine modules including preprocessing, training,
evaluation, and serialization.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any
import pickle
import json

from app.ml_engine.preprocessing.pipeline import Pipeline
from app.ml_engine.preprocessing.imputer import MeanImputer, MedianImputer, ModeImputer
from app.ml_engine.preprocessing.scaler import StandardScaler, MinMaxScaler
from app.ml_engine.preprocessing.encoder import OneHotEncoder, LabelEncoder
from app.ml_engine.preprocessing.cleaner import IQROutlierDetector, ZScoreOutlierDetector
from app.ml_engine.feature_selection.variance_threshold import VarianceThreshold
from app.ml_engine.feature_selection.correlation_selector import CorrelationSelector
from app.ml_engine.models.classification import RandomForestClassifierWrapper
from app.ml_engine.models.regression import LinearRegressionWrapper
from app.ml_engine.training.trainer import train_model
from app.ml_engine.evaluation.classification_metrics import ClassificationMetrics
from app.ml_engine.evaluation.regression_metrics import RegressionMetrics
from app.ml_engine.utils.serialization import ModelSerializer, save_model, load_model


@pytest.mark.integration
class TestPreprocessingPipeline:
    """Test complete preprocessing pipeline workflow"""

    def test_classification_preprocessing_pipeline(self):
        """Test preprocessing pipeline for classification data"""
        
        # Create sample data with various issues
        np.random.seed(42)
        df = pd.DataFrame({
            'age': [25, np.nan, 35, 100, 28, 32, np.nan, 40],
            'income': [50000, 60000, np.nan, 75000, 55000, 62000, 58000, 150000],
            'category': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'B'],
            'score': [0.5, 0.7, 0.6, 0.8, 0.55, 0.65, 0.75, 0.9],
            'target': [0, 1, 0, 1, 0, 0, 1, 1]
        })
        
        # Build preprocessing pipeline
        pipeline = Pipeline(steps=[
            MeanImputer(columns=['age', 'income']),
            IQROutlierDetector(threshold=1.5, columns=['age', 'income']),
            StandardScaler(columns=['age', 'income', 'score']),
            OneHotEncoder(columns=['category'])
        ])
        
        # Fit and transform
        X = df.drop('target', axis=1)
        y = df['target']
        
        pipeline.fit(X, y)
        X_transformed = pipeline.transform(X)
        
        # Verify pipeline
        assert pipeline.fitted
        assert len(pipeline.steps) == 4
        assert X_transformed is not None
        assert len(X_transformed) <= len(X)  # May remove outliers
        
        # Verify no missing values
        if isinstance(X_transformed, pd.DataFrame):
            assert X_transformed.isnull().sum().sum() == 0
        
        print(f"✓ Classification preprocessing pipeline: {X.shape} -> {X_transformed.shape}")

    def test_regression_preprocessing_pipeline(self):
        """Test preprocessing pipeline for regression data"""
        
        np.random.seed(42)
        df = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100) * 10 + 50,
            'feature3': np.random.choice(['low', 'medium', 'high'], 100),
            'target': np.random.randn(100) * 5 + 100
        })
        
        # Add missing values
        df.loc[df.sample(5).index, 'feature1'] = np.nan
        df.loc[df.sample(5).index, 'feature2'] = np.nan
        
        # Build pipeline
        pipeline = Pipeline(
            name="Regression Preprocessing",
            steps=[
                MedianImputer(columns=['feature1', 'feature2']),
                ZScoreOutlierDetector(threshold=3.0),
                MinMaxScaler(columns=['feature1', 'feature2']),
                LabelEncoder(columns=['feature3'])
            ]
        )
        
        X = df.drop('target', axis=1)
        y = df['target']
        
        pipeline.fit(X, y)
        X_transformed = pipeline.transform(X)
        
        assert pipeline.fitted
        assert X_transformed.shape[1] >= X.shape[1]  # May add columns from encoding
        
        # Verify scaling (values should be in [0, 1] for MinMaxScaler)
        if isinstance(X_transformed, pd.DataFrame):
            scaled_cols = [col for col in X_transformed.columns if 'feature' in str(col)]
            if scaled_cols:
                assert X_transformed[scaled_cols[0]].max() <= 1.0
                assert X_transformed[scaled_cols[0]].min() >= 0.0
        
        print(f"✓ Regression preprocessing pipeline: {X.shape} -> {X_transformed.shape}")

    def test_pipeline_serialization(self, tmp_path: Path):
        """Test pipeline serialization and deserialization"""
        
        # Create and fit pipeline
        df = pd.DataFrame({
            'x1': np.random.randn(50),
            'x2': np.random.randn(50),
            'x3': np.random.choice(['A', 'B', 'C'], 50),
            'y': np.random.randint(0, 2, 50)
        })
        
        pipeline = Pipeline(steps=[
            MeanImputer(columns=['x1', 'x2']),
            StandardScaler(columns=['x1', 'x2']),
            OneHotEncoder(columns=['x3'])
        ])
        
        X = df[['x1', 'x2', 'x3']]
        y = df['y']
        
        pipeline.fit(X, y)
        X_transformed = pipeline.transform(X)
        
        # Save pipeline
        pipeline_file = tmp_path / "test_pipeline.pkl"
        with open(pipeline_file, 'wb') as f:
            pickle.dump(pipeline, f)
        
        # Load pipeline
        with open(pipeline_file, 'rb') as f:
            loaded_pipeline = pickle.load(f)
        
        # Verify loaded pipeline
        assert loaded_pipeline.fitted
        assert len(loaded_pipeline.steps) == len(pipeline.steps)
        
        # Transform with loaded pipeline
        X_loaded_transformed = loaded_pipeline.transform(X)
        
        # Results should be identical
        if isinstance(X_transformed, pd.DataFrame) and isinstance(X_loaded_transformed, pd.DataFrame):
            pd.testing.assert_frame_equal(X_transformed, X_loaded_transformed)
        
        print(f"✓ Pipeline serialization successful")

    def test_pipeline_inverse_transform(self):
        """Test pipeline inverse transform capability"""
        
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [10, 20, 30, 40, 50]
        })
        
        pipeline = Pipeline(steps=[
            StandardScaler(columns=['feature1', 'feature2'])
        ])
        
        pipeline.fit(df)
        X_transformed = pipeline.transform(df)
        
        # Try inverse transform
        try:
            X_inverse = pipeline.inverse_transform(X_transformed)
            
            # Should be close to original
            if isinstance(X_inverse, pd.DataFrame):
                np.testing.assert_array_almost_equal(
                    df[['feature1', 'feature2']].values,
                    X_inverse[['feature1', 'feature2']].values,
                    decimal=5
                )
            
            print(f"✓ Pipeline inverse transform successful")
        except (AttributeError, NotImplementedError):
            pytest.skip("Inverse transform not implemented")


@pytest.mark.integration
class TestFeatureSelection:
    """Test feature selection methods"""

    def test_variance_threshold_selection(self):
        """Test variance threshold feature selection"""
        
        # Create data with low and high variance features
        np.random.seed(42)
        df = pd.DataFrame({
            'low_var': [1, 1, 1, 1, 1, 1, 1, 1],  # Very low variance
            'medium_var': [1, 2, 1, 2, 1, 2, 1, 2],
            'high_var': np.random.randn(8) * 10
        })
        
        selector = VarianceThresholdSelector(threshold=0.1)
        selector.fit(df)
        df_selected = selector.transform(df)
        
        # Low variance feature should be removed
        assert 'low_var' not in df_selected.columns
        assert 'high_var' in df_selected.columns
        
        print(f"✓ Variance threshold selection: {df.shape[1]} -> {df_selected.shape[1]} features")

    def test_correlation_selection(self):
        """Test correlation-based feature selection"""
        
        np.random.seed(42)
        n_samples = 100
        
        # Create correlated features
        x1 = np.random.randn(n_samples)
        x2 = x1 + np.random.randn(n_samples) * 0.1  # Highly correlated with x1
        x3 = np.random.randn(n_samples)  # Independent
        
        df = pd.DataFrame({
            'x1': x1,
            'x2': x2,
            'x3': x3,
            'target': x1 + x3 + np.random.randn(n_samples) * 0.5
        })
        
        selector = CorrelationSelector(
            target_column='target',
            threshold=0.5,
            method='pearson'
        )
        
        X = df.drop('target', axis=1)
        y = df['target']
        
        selector.fit(X, y)
        X_selected = selector.transform(X)
        
        # Should select features correlated with target
        assert X_selected.shape[1] <= X.shape[1]
        assert X_selected.shape[0] == X.shape[0]
        
        print(f"✓ Correlation selection: {X.shape[1]} -> {X_selected.shape[1]} features")


@pytest.mark.integration
class TestModelTraining:
    """Test model training workflow"""

    def test_classification_model_training(self):
        """Test training classification model"""
        
        # Create classification dataset
        from sklearn.datasets import make_classification
        
        X, y = make_classification(
            n_samples=200,
            n_features=10,
            n_informative=5,
            n_redundant=2,
            random_state=42
        )
        
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
        df['target'] = y
        
        # Train model
        model = RandomForestClassifierWrapper(
            n_estimators=50,
            max_depth=5,
            random_state=42
        )
        
        X_train = df.drop('target', axis=1)
        y_train = df['target']
        
        model.fit(X_train, y_train)
        
        # Make predictions
        predictions = model.predict(X_train)
        probabilities = model.predict_proba(X_train)
        
        assert len(predictions) == len(X_train)
        assert probabilities.shape == (len(X_train), 2)
        
        # Get feature importance
        importance = model.get_feature_importance()
        assert len(importance) == X_train.shape[1]
        
        print(f"✓ Classification model trained: {X_train.shape} samples")

    def test_regression_model_training(self):
        """Test training regression model"""
        
        # Create regression dataset
        from sklearn.datasets import make_regression
        
        X, y = make_regression(
            n_samples=200,
            n_features=8,
            n_informative=5,
            noise=10,
            random_state=42
        )
        
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(8)])
        df['target'] = y
        
        # Train model
        model = LinearRegressionWrapper(fit_intercept=True)
        
        X_train = df.drop('target', axis=1)
        y_train = df['target']
        
        model.fit(X_train, y_train)
        
        # Make predictions
        predictions = model.predict(X_train)
        
        assert len(predictions) == len(X_train)
        
        # Get coefficients
        coefficients = model.get_coefficients()
        assert len(coefficients) == X_train.shape[1]
        
        print(f"✓ Regression model trained: {X_train.shape} samples")

    def test_model_serialization(self, tmp_path: Path):
        """Test model serialization and loading"""
        
        from sklearn.datasets import make_classification
        
        X, y = make_classification(
            n_samples=100,
            n_features=5,
            random_state=42
        )
        
        df = pd.DataFrame(X, columns=[f'f{i}' for i in range(5)])
        df['target'] = y
        
        # Train and save model
        model = RandomForestClassifierWrapper(n_estimators=10, random_state=42)
        model.fit(df.drop('target', axis=1), df['target'])
        
        model_file = tmp_path / "test_model.pkl"
        save_model(model, model_file)
        
        # Load model
        loaded_model = load_model(model_file)
        
        # Compare predictions
        original_pred = model.predict(df.drop('target', axis=1))
        loaded_pred = loaded_model.predict(df.drop('target', axis=1))
        
        np.testing.assert_array_equal(original_pred, loaded_pred)
        
        print(f"✓ Model serialization successful")


@pytest.mark.integration
class TestModelEvaluation:
    """Test model evaluation metrics"""

    def test_classification_metrics(self):
        """Test classification evaluation metrics"""
        
        # Create sample predictions
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1, 0, 0, 1, 1, 0, 1])
        y_proba = np.array([
            [0.9, 0.1], [0.2, 0.8], [0.8, 0.2], [0.3, 0.7],
            [0.85, 0.15], [0.6, 0.4], [0.4, 0.6], [0.2, 0.8],
            [0.9, 0.1], [0.1, 0.9]
        ])
        
        evaluator = ClassificationMetrics()
        metrics = evaluator.compute_metrics(y_true, y_pred, y_proba)
        
        # Verify all metrics computed
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        
        # Verify confusion matrix
        confusion_matrix = evaluator.compute_confusion_matrix(y_true, y_pred)
        assert confusion_matrix.shape == (2, 2)
        
        # Verify ROC curve
        roc_data = evaluator.compute_roc_curve(y_true, y_proba[:, 1])
        assert 'fpr' in roc_data
        assert 'tpr' in roc_data
        assert 'auc' in roc_data
        
        print(f"✓ Classification metrics: Accuracy={metrics['accuracy']:.3f}")

    def test_regression_metrics(self):
        """Test regression evaluation metrics"""
        
        # Create sample predictions
        y_true = np.array([1.5, 2.3, 3.1, 4.8, 5.2, 6.1, 7.0, 8.2, 9.5, 10.1])
        y_pred = np.array([1.6, 2.1, 3.3, 4.7, 5.5, 6.0, 6.8, 8.5, 9.3, 10.2])
        
        evaluator = RegressionMetrics()
        metrics = evaluator.compute_metrics(y_true, y_pred)
        
        # Verify all metrics computed
        assert 'mse' in metrics
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'r2' in metrics
        
        # Verify values are reasonable
        assert metrics['mse'] >= 0
        assert metrics['rmse'] >= 0
        assert metrics['mae'] >= 0
        assert metrics['r2'] <= 1.0
        
        print(f"✓ Regression metrics: RMSE={metrics['rmse']:.3f}, R²={metrics['r2']:.3f}")

    def test_cross_validation_evaluation(self):
        """Test model evaluation with cross-validation"""
        
        from sklearn.datasets import make_classification
        from sklearn.model_selection import cross_val_score
        
        X, y = make_classification(
            n_samples=100,
            n_features=10,
            random_state=42
        )
        
        model = RandomForestClassifierWrapper(n_estimators=20, random_state=42)
        
        # Perform cross-validation
        scores = cross_val_score(
            model.model,  # Use underlying sklearn model
            X, y,
            cv=5,
            scoring='accuracy'
        )
        
        assert len(scores) == 5
        assert all(0 <= score <= 1 for score in scores)
        
        print(f"✓ Cross-validation: {scores.mean():.3f} ± {scores.std():.3f}")


@pytest.mark.integration
class TestCompleteMLWorkflow:
    """Test complete ML workflow from data to model"""

    def test_classification_workflow(self, tmp_path: Path):
        """Test complete classification workflow"""
        
        print("\n" + "=" * 60)
        print("COMPLETE CLASSIFICATION WORKFLOW TEST")
        print("=" * 60)
        
        # Step 1: Generate data
        print("\n[1/6] Generating classification dataset...")
        from sklearn.datasets import make_classification
        
        X, y = make_classification(
            n_samples=500,
            n_features=15,
            n_informative=10,
            n_redundant=3,
            n_classes=2,
            random_state=42
        )
        
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(15)])
        df['target'] = y
        
        # Add missing values and categorical features
        df.loc[df.sample(20).index, 'feature_0'] = np.nan
        df['category'] = np.random.choice(['A', 'B', 'C'], size=len(df))
        
        print(f"✓ Dataset created: {df.shape}")
        
        # Step 2: Preprocessing
        print("\n[2/6] Building preprocessing pipeline...")
        preprocessing_pipeline = Pipeline(
            name="Classification Preprocessing",
            steps=[
                MeanImputer(columns=['feature_0']),
                StandardScaler(columns=[f'feature_{i}' for i in range(15)]),
                OneHotEncoder(columns=['category'])
            ]
        )
        
        X = df.drop('target', axis=1)
        y = df['target']
        
        preprocessing_pipeline.fit(X, y)
        X_processed = preprocessing_pipeline.transform(X)
        
        print(f"✓ Data preprocessed: {X.shape} -> {X_processed.shape}")
        
        # Step 3: Feature selection
        print("\n[3/6] Performing feature selection...")
        selector = VarianceThresholdSelector(threshold=0.01)
        selector.fit(X_processed)
        X_selected = selector.transform(X_processed)
        
        print(f"✓ Features selected: {X_processed.shape[1]} -> {X_selected.shape[1]}")
        
        # Step 4: Split data
        print("\n[4/6] Splitting train/test data...")
        from sklearn.model_selection import train_test_split
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )
        
        print(f"✓ Train: {X_train.shape}, Test: {X_test.shape}")
        
        # Step 5: Train model
        print("\n[5/6] Training model...")
        model = RandomForestClassifierWrapper(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        print(f"✓ Model trained")
        
        # Step 6: Evaluate
        print("\n[6/6] Evaluating model...")
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        
        evaluator = ClassificationMetrics()
        metrics = evaluator.compute_metrics(y_test, y_pred, y_proba)
        
        print(f"✓ Evaluation complete:")
        print(f"   - Accuracy: {metrics['accuracy']:.3f}")
        print(f"   - Precision: {metrics['precision']:.3f}")
        print(f"   - Recall: {metrics['recall']:.3f}")
        print(f"   - F1 Score: {metrics['f1_score']:.3f}")
        
        # Save artifacts
        print("\n[Saving] Saving pipeline and model...")
        pipeline_file = tmp_path / "preprocessing_pipeline.pkl"
        model_file = tmp_path / "trained_model.pkl"
        
        with open(pipeline_file, 'wb') as f:
            pickle.dump(preprocessing_pipeline, f)
        
        save_model(model, model_file)
        
        print(f"✓ Artifacts saved to {tmp_path}")
        
        # Verify we can load and use saved artifacts
        print("\n[Verification] Testing saved artifacts...")
        with open(pipeline_file, 'rb') as f:
            loaded_pipeline = pickle.load(f)
        
        loaded_model = load_model(model_file)
        
        # Process new data and predict
        X_new_processed = loaded_pipeline.transform(X.head(5))
        X_new_selected = selector.transform(X_new_processed)
        new_predictions = loaded_model.predict(X_new_selected)
        
        assert len(new_predictions) == 5
        
        print(f"✓ Loaded artifacts working correctly")
        print("\n" + "=" * 60)
        print("WORKFLOW TEST COMPLETED SUCCESSFULLY")
        print("=" * 60)

    def test_regression_workflow(self, tmp_path: Path):
        """Test complete regression workflow"""
        
        print("\n" + "=" * 60)
        print("COMPLETE REGRESSION WORKFLOW TEST")
        print("=" * 60)
        
        # Step 1: Generate data
        print("\n[1/5] Generating regression dataset...")
        from sklearn.datasets import make_regression
        
        X, y = make_regression(
            n_samples=400,
            n_features=12,
            n_informative=8,
            noise=15,
            random_state=42
        )
        
        df = pd.DataFrame(X, columns=[f'x{i}' for i in range(12)])
        df['target'] = y
        
        print(f"✓ Dataset created: {df.shape}")
        
        # Step 2: Preprocessing
        print("\n[2/5] Preprocessing data...")
        preprocessing_pipeline = Pipeline(steps=[
            MinMaxScaler(columns=[f'x{i}' for i in range(12)])
        ])
        
        X = df.drop('target', axis=1)
        y = df['target']
        
        preprocessing_pipeline.fit(X, y)
        X_processed = preprocessing_pipeline.transform(X)
        
        print(f"✓ Data preprocessed")
        
        # Step 3: Train/test split
        print("\n[3/5] Splitting data...")
        from sklearn.model_selection import train_test_split
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y,
            test_size=0.2,
            random_state=42
        )
        
        print(f"✓ Train: {X_train.shape}, Test: {X_test.shape}")
        
        # Step 4: Train model
        print("\n[4/5] Training regression model...")
        model = LinearRegressionWrapper(fit_intercept=True)
        model.fit(X_train, y_train)
        
        print(f"✓ Model trained")
        
        # Step 5: Evaluate
        print("\n[5/5] Evaluating model...")
        y_pred = model.predict(X_test)
        
        evaluator = RegressionMetrics()
        metrics = evaluator.compute_metrics(y_test, y_pred)
        
        print(f"✓ Evaluation complete:")
        print(f"   - RMSE: {metrics['rmse']:.2f}")
        print(f"   - MAE: {metrics['mae']:.2f}")
        print(f"   - R²: {metrics['r2']:.3f}")
        
        print("\n" + "=" * 60)
        print("WORKFLOW TEST COMPLETED SUCCESSFULLY")
        print("=" * 60)


@pytest.mark.integration
@pytest.mark.slow
class TestMLPipelineStressTests:
    """Stress tests for ML pipeline"""

    def test_pipeline_with_many_steps(self):
        """Test pipeline with many preprocessing steps"""
        
        df = pd.DataFrame({
            'x1': np.random.randn(100),
            'x2': np.random.randn(100),
            'x3': np.random.choice(['A', 'B'], 100),
            'target': np.random.randint(0, 2, 100)
        })
        
        # Create pipeline with many steps
        pipeline = Pipeline(steps=[
            MeanImputer(),
            StandardScaler(columns=['x1', 'x2']),
            MinMaxScaler(columns=['x1']),
            OneHotEncoder(columns=['x3']),
            VarianceThresholdSelector(threshold=0.0)
        ])
        
        X = df.drop('target', axis=1)
        y = df['target']
        
        pipeline.fit(X, y)
        X_transformed = pipeline.transform(X)
        
        assert pipeline.fitted
        assert len(pipeline.step_statistics) == len(pipeline.steps)
        
        print(f"✓ Pipeline with {len(pipeline.steps)} steps completed")

    def test_high_dimensional_data(self):
        """Test pipeline with high-dimensional data"""
        
        # Create high-dimensional dataset
        n_samples = 200
        n_features = 100
        
        X = np.random.randn(n_samples, n_features)
        y = np.random.randint(0, 2, n_samples)
        
        df = pd.DataFrame(X, columns=[f'f{i}' for i in range(n_features)])
        
        # Apply feature selection
        selector = VarianceThresholdSelector(threshold=0.5)
        selector.fit(df)
        df_selected = selector.transform(df)
        
        print(f"✓ High-dimensional data: {df.shape} -> {df_selected.shape}")
        assert df_selected.shape[1] < df.shape[1]
