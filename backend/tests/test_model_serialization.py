"""
Tests for Model Serialization Service

This module tests the ModelSerializationService functionality including:
- Model saving with metadata
- Model loading and deserialization
- Model information retrieval
- Model deletion
- Experiment cleanup
- Model package export
"""

import pytest
import json
import shutil
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

from app.services.storage_service import ModelSerializationService, get_model_serialization_service
from app.ml_engine.models import create_model
from app.ml_engine.models.base import ModelConfig


@pytest.fixture
def temp_storage_dir(tmp_path):
    """Create a temporary storage directory."""
    storage_dir = tmp_path / "test_models"
    storage_dir.mkdir()
    yield storage_dir
    # Cleanup
    if storage_dir.exists():
        shutil.rmtree(storage_dir)


@pytest.fixture
def serialization_service(temp_storage_dir):
    """Create a ModelSerializationService instance with temp directory."""
    return ModelSerializationService(base_dir=str(temp_storage_dir))


@pytest.fixture
def trained_model():
    """Create and train a simple model for testing."""
    # Create sample data
    X = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'feature3': np.random.randn(100)
    })
    y = pd.Series(np.random.randint(0, 2, 100), name='target')
    
    # Create and train model
