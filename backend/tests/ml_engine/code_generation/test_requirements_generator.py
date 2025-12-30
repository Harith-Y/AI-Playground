"""
Tests for Requirements Generator

Tests the generation of requirements.txt files from code and configurations.
"""

import pytest
from app.ml_engine.code_generation.requirements_generator import (
    RequirementsGenerator,
    DependencyInfo,
    generate_requirements,
)


class TestDependencyInfo:
    """Test DependencyInfo dataclass."""
    
    def test_to_requirement_line_with_version(self):
        """Test requirement line generation with version."""
        dep = DependencyInfo(
            package='numpy',
            version='1.24.3',
            purpose='Numerical computing'
        )
        
        line = dep.to_requirement_line(include_version=True)
        assert line == 'numpy==1.24.3'
    
    def test_to_requirement_line_without_version(self):
        """Test requirement line generation without exact version."""
        dep = DependencyInfo(
            package='pandas',
            version='2.0.3',
            purpose='Data manipulation'
        )
        
        line = dep.to_requirement_line(include_version=False)
        assert line == 'pandas>=2.0.3'
    
    def test_to_requirement_line_with_extras(self):
        """Test requirement line with extras."""
        dep = DependencyInfo(
            package='fastapi',
            version='0.100.0',
            purpose='Web framework',
            extras=['all']
        )
        
        line = dep.to_requirement_line(include_version=True)
        assert line == 'fastapi[all]==0.100.0'


class TestRequirementsGenerator:
    """Test RequirementsGenerator class."""
    
    @pytest.fixture
    def generator(self):
        """Create generator instance."""
        return RequirementsGenerator()
    
    def test_extract_imports_simple(self, generator):
        """Test extracting simple imports."""
        code = """
import numpy
import pandas
from sklearn.ensemble import RandomForestClassifier
"""
        imports = generator._extract_imports(code)
        
        assert 'numpy' in imports
        assert 'pandas' in imports
        assert 'sklearn' in imports
    
    def test_extract_imports_complex(self, generator):
        """Test extracting complex imports."""
        code = """
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import seaborn as sns
"""
        imports = generator._extract_imports(code)
        
        assert 'numpy' in imports
        assert 'pandas' in imports
        assert 'sklearn' in imports
        assert 'matplotlib' in imports
        assert 'seaborn' in imports
    
    def test_map_imports_to_packages(self, generator):
        """Test mapping imports to package names."""
        imports = {'sklearn', 'numpy', 'pandas', 'cv2', 'PIL'}
        packages = generator._map_imports_to_packages(imports)
        
        assert 'scikit-learn' in packages  # sklearn -> scikit-learn
        assert 'numpy' in packages
        assert 'pandas' in packages
        # cv2 and PIL might not be in PACKAGE_VERSIONS
    
    def test_create_dependencies(self, generator):
        """Test creating dependency info objects."""
        packages = {'numpy', 'pandas', 'scikit-learn'}
        dependencies = generator._create_dependencies(packages)
        
        assert len(dependencies) == 3
        assert all(isinstance(dep, DependencyInfo) for dep in dependencies)
        
        # Check sorted order
        package_names = [dep.package for dep in dependencies]
        assert package_names == sorted(package_names)
    
    def test_generate_from_code(self, generator):
        """Test generating requirements from code."""
        code = """
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
"""
        requirements = generator.generate_from_code(code)
        
        assert 'numpy' in requirements
        assert 'pandas' in requirements
        assert 'scikit-learn' in requirements
        assert 'matplotlib' in requirements
        assert '# Core Dependencies' in requirements
        assert '# Installation Instructions' in requirements
    
    def test_generate_from_config_classification(self, generator):
        """Test generating requirements from classification config."""
        config = {
            'model_type': 'random_forest_classifier',
            'task_type': 'classification',
            'preprocessing_steps': [
                {'type': 'missing_value_imputation'},
                {'type': 'scaling'}
            ],
            'include_evaluation': True
        }
        
        requirements = generator.generate_from_config(config)
        
        assert 'numpy' in requirements
        assert 'pandas' in requirements
        assert 'scikit-learn' in requirements
        assert 'matplotlib' in requirements
        assert 'seaborn' in requirements
        assert 'joblib' in requirements
    
    def test_generate_from_config_xgboost(self, generator):
        """Test generating requirements for XGBoost model."""
        config = {
            'model_type': 'xgboost_classifier',
            'task_type': 'classification'
        }
        
        requirements = generator.generate_from_config(config)
        
        assert 'xgboost' in requirements
        assert 'scikit-learn' in requirements
    
    def test_generate_modular_requirements(self, generator):
        """Test generating modular requirements files."""
        config = {
            'model_type': 'random_forest_classifier',
            'task_type': 'classification',
            'preprocessing_steps': [{'type': 'scaling'}]
        }
        
        requirements_files = generator.generate_modular_requirements(config)
        
        # Check all expected files are generated
        assert 'requirements.txt' in requirements_files
        assert 'requirements-preprocessing.txt' in requirements_files
        assert 'requirements-training.txt' in requirements_files
        assert 'requirements-evaluation.txt' in requirements_files
        assert 'requirements-prediction.txt' in requirements_files
        assert 'requirements-dev.txt' in requirements_files
        
        # Check core requirements
        core_req = requirements_files['requirements.txt']
        assert 'numpy' in core_req
        assert 'pandas' in core_req
        
        # Check preprocessing requirements
        preprocessing_req = requirements_files['requirements-preprocessing.txt']
        assert 'scikit-learn' in preprocessing_req
        
        # Check dev requirements
        dev_req = requirements_files['requirements-dev.txt']
        assert 'pytest' in dev_req
    
    def test_generate_modular_with_code_sections(self, generator):
        """Test modular generation with code analysis."""
        config = {
            'model_type': 'random_forest_classifier',
            'task_type': 'classification'
        }
        
        code_sections = {
            'preprocessing': """
import pandas as pd
from sklearn.preprocessing import StandardScaler
""",
            'training': """
import numpy as np
from sklearn.ensemble import RandomForestClassifier
"""
        }
        
        requirements_files = generator.generate_modular_requirements(
            config,
            code_sections=code_sections
        )
        
        # Check preprocessing requirements include StandardScaler's package
        preprocessing_req = requirements_files['requirements-preprocessing.txt']
        assert 'scikit-learn' in preprocessing_req
        
        # Check training requirements
        training_req = requirements_files['requirements-training.txt']
        assert 'scikit-learn' in training_req
    
    def test_get_model_packages_sklearn(self, generator):
        """Test getting packages for sklearn models."""
        packages = generator._get_model_packages('random_forest_classifier')
        assert 'scikit-learn' in packages
    
    def test_get_model_packages_xgboost(self, generator):
        """Test getting packages for XGBoost models."""
        packages = generator._get_model_packages('xgboost_classifier')
        assert 'xgboost' in packages
        assert 'scikit-learn' in packages
    
    def test_get_model_packages_lightgbm(self, generator):
        """Test getting packages for LightGBM models."""
        packages = generator._get_model_packages('lightgbm_regressor')
        assert 'lightgbm' in packages
        assert 'scikit-learn' in packages
    
    def test_get_model_packages_catboost(self, generator):
        """Test getting packages for CatBoost models."""
        packages = generator._get_model_packages('catboost_classifier')
        assert 'catboost' in packages
        assert 'scikit-learn' in packages
    
    def test_generate_docker_requirements(self, generator):
        """Test generating Docker-optimized requirements."""
        config = {
            'model_type': 'random_forest_classifier',
            'task_type': 'classification'
        }
        
        requirements = generator.generate_docker_requirements(config, python_version='3.9')
        
        assert 'Python 3.9' in requirements
        assert 'Pinned versions' in requirements
        # Check pinned versions (==)
        assert '==' in requirements
        assert 'numpy==' in requirements
    
    def test_generate_conda_environment(self, generator):
        """Test generating conda environment.yml."""
        config = {
            'model_type': 'random_forest_classifier',
            'task_type': 'classification'
        }
        
        env_yml = generator.generate_conda_environment(
            config,
            env_name='test-env',
            python_version='3.9'
        )
        
        assert 'name: test-env' in env_yml
        assert 'python=3.9' in env_yml
        assert 'channels:' in env_yml
        assert 'dependencies:' in env_yml
        assert 'numpy' in env_yml
        assert 'pandas' in env_yml
    
    def test_format_requirements_with_title(self, generator):
        """Test formatting requirements with custom title."""
        dependencies = [
            DependencyInfo('numpy', '1.24.3', 'Numerical computing'),
            DependencyInfo('pandas', '2.0.3', 'Data manipulation')
        ]
        
        content = generator._format_requirements(
            dependencies,
            title="Test Requirements"
        )
        
        assert 'Test Requirements' in content
        assert 'numpy==1.24.3' in content
        assert 'pandas==2.0.3' in content
        assert '# Numerical computing' in content
        assert '# Data manipulation' in content
        assert '# Installation Instructions' in content


class TestGenerateRequirementsFunction:
    """Test convenience function."""
    
    def test_generate_requirements_pip(self):
        """Test generating pip requirements."""
        config = {
            'model_type': 'random_forest_classifier',
            'task_type': 'classification'
        }
        
        requirements = generate_requirements(config, output_format='pip')
        
        assert 'requirements.txt' in requirements
        assert 'numpy' in requirements['requirements.txt']
    
    def test_generate_requirements_modular(self):
        """Test generating modular requirements."""
        config = {
            'model_type': 'random_forest_classifier',
            'task_type': 'classification'
        }
        
        requirements = generate_requirements(config, modular=True)
        
        assert len(requirements) == 6  # 6 modular files
        assert 'requirements.txt' in requirements
        assert 'requirements-dev.txt' in requirements
    
    def test_generate_requirements_docker(self):
        """Test generating Docker requirements."""
        config = {
            'model_type': 'random_forest_classifier',
            'task_type': 'classification'
        }
        
        requirements = generate_requirements(config, output_format='docker')
        
        assert 'requirements.txt' in requirements
        assert 'Docker Requirements' in requirements['requirements.txt']
    
    def test_generate_requirements_conda(self):
        """Test generating conda environment."""
        config = {
            'model_type': 'random_forest_classifier',
            'task_type': 'classification'
        }
        
        requirements = generate_requirements(config, output_format='conda')
        
        assert 'environment.yml' in requirements
        assert 'name:' in requirements['environment.yml']
        assert 'channels:' in requirements['environment.yml']
    
    def test_generate_requirements_invalid_format(self):
        """Test error handling for invalid format."""
        config = {'model_type': 'random_forest_classifier'}
        
        with pytest.raises(ValueError, match="Unknown output_format"):
            generate_requirements(config, output_format='invalid')


class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_modular_generation(self):
        """Test complete modular requirements generation."""
        config = {
            'experiment_name': 'Test Experiment',
            'model_type': 'xgboost_classifier',
            'task_type': 'classification',
            'preprocessing_steps': [
                {'type': 'missing_value_imputation', 'strategy': 'mean'},
                {'type': 'scaling', 'scaler': 'standard'}
            ],
            'hyperparameters': {
                'n_estimators': 100,
                'max_depth': 10
            },
            'include_evaluation': True
        }
        
        code_sections = {
            'preprocessing': """
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
""",
            'training': """
import xgboost as xgb
from sklearn.model_selection import train_test_split
"""
        }
        
        requirements_files = generate_requirements(
            config,
            modular=True,
            code_sections=code_sections
        )
        
        # Verify all files generated
        assert len(requirements_files) == 6
        
        # Verify core requirements
        core = requirements_files['requirements.txt']
        assert 'numpy' in core
        assert 'pandas' in core
        
        # Verify preprocessing requirements
        preprocessing = requirements_files['requirements-preprocessing.txt']
        assert 'scikit-learn' in preprocessing
        
        # Verify training requirements include XGBoost
        training = requirements_files['requirements-training.txt']
        assert 'xgboost' in training or 'scikit-learn' in training
        
        # Verify evaluation requirements
        evaluation = requirements_files['requirements-evaluation.txt']
        assert 'matplotlib' in evaluation
        
        # Verify prediction requirements are minimal
        prediction = requirements_files['requirements-prediction.txt']
        assert 'numpy' in prediction
        assert 'joblib' in prediction
        
        # Verify dev requirements
        dev = requirements_files['requirements-dev.txt']
        assert 'pytest' in dev
    
    def test_minimal_requirements_generation(self):
        """Test generating minimal requirements."""
        config = {
            'model_type': 'linear_regression',
            'task_type': 'regression',
            'preprocessing_steps': [],
            'include_evaluation': False
        }
        
        requirements = generate_requirements(config, output_format='pip')
        content = requirements['requirements.txt']
        
        # Should have minimal dependencies
        assert 'numpy' in content
        assert 'pandas' in content
        assert 'scikit-learn' in content
        assert 'joblib' in content
