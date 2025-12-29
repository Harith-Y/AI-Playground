"""
Test configuration for ML engine tests (without full app dependencies).
"""

import pytest
import tempfile
from pathlib import Path


@pytest.fixture
def tmp_path():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)
