"""
Requirements.txt Generator

Generates minimal, modular requirements.txt files based on actual code usage.
Analyzes generated code to determine exact dependencies needed.

Based on: ML-TO-DO.md > ML-67
"""

from typing import Dict, List, Set, Any, Optional
from dataclasses import dataclass, field
import re
from datetime import datetime
from app.utils.logger import get_logger

logger = get_logger("requirements_generator")


@dataclass
class DependencyInfo:
    """Information about a Python package dependency."""
    package: str
    version: str
    purpose: str
    optional: bool = False
    extras: List[str] = field(default_factory=list)
    
    def to_requirement_line(self, include_version: bool = True) -> str:
        """Convert to requirements.txt line format."""
        if include_version:
            base = f"{self.package}=={self.version}"
        else:
            base = f"{self.package}>={self.version}"
        
        if self.extras:
            base = f"{self.package}[{','.join(self.extras)}]=={self.version}"
        
        return base


class RequirementsGenerator:
    """
    Generator for requirements.txt files.
    
    Analyzes generated code to determine minimal dependencies needed.
    Creates modular requirements files for different use cases.
    
    Example:
        >>> generator = RequirementsGenerator()
        >>> code = "import pandas as pd\\nimport sklearn..."
        >>> requirements = generator.generate_from_code(code)
    """
    
    # Core ML/Data Science packages with versions
    PACKAGE_VERSIONS = {
        # Core data science
        'numpy': '1.24.3',
        'pandas': '2.0.3',
        'scipy': '1.11.1',
        
        # Machine learning
        'scikit-learn': '1.3.0',
        'xgboost': '1.7.6',
        'lightgbm': '4.0.0',
        'catboost': '1.2',
        
        # Deep learning
        'tensorflow': '2.13.0',
        'torch': '2.0.1',
        'keras': '2.13.1',
        
        # Visualization
        'matplotlib': '3.7.2',
        'seaborn': '0.12.2',
        'plotly': '5.15.0',
        
        # Utilities
        'joblib': '1.3.1',
        'tqdm': '4.65.0',
        'pyyaml': '6.0.1',
        'python-dotenv': '1.0.0',
        
        # API/Web
        'fastapi': '0.100.0',
        'uvicorn': '0.23.1',
        'pydantic': '2.1.1',
        'requests': '2.31.0',
        
        # Database
        'sqlalchemy': '2.0.19',
        'psycopg2-binary': '2.9.6',
        'redis': '4.6.0',
        
        # Testing
        'pytest': '7.4.0',
        'pytest-cov': '4.1.0',
    }
    
    # Import to package mappings
    IMPORT_TO_PACKAGE = {
        'sklearn': 'scikit-learn',
        'cv2': 'opencv-python',
        'PIL': 'Pillow',
        'yaml': 'pyyaml',
        'dotenv': 'python-dotenv',
    }
    
    # Package purposes/descriptions
    PACKAGE_PURPOSES = {
        'numpy': 'Numerical computing',
        'pandas': 'Data manipulation and analysis',
        'scipy': 'Scientific computing',
        'scikit-learn': 'Machine learning algorithms',
        'xgboost': 'Gradient boosting',
        'lightgbm': 'Light gradient boosting',
        'catboost': 'Categorical boosting',
        'tensorflow': 'Deep learning framework',
        'torch': 'PyTorch deep learning',
        'keras': 'High-level neural networks API',
        'matplotlib': 'Data visualization',
        'seaborn': 'Statistical data visualization',
        'plotly': 'Interactive visualizations',
        'joblib': 'Model serialization',
        'tqdm': 'Progress bars',
        'pyyaml': 'YAML configuration files',
        'python-dotenv': 'Environment variable management',
        'fastapi': 'Web API framework',
        'uvicorn': 'ASGI server',
        'pydantic': 'Data validation',
        'requests': 'HTTP library',
        'sqlalchemy': 'SQL toolkit and ORM',
        'psycopg2-binary': 'PostgreSQL adapter',
        'redis': 'Redis client',
        'pytest': 'Testing framework',
        'pytest-cov': 'Test coverage',
    }
    
    def __init__(self):
        """Initialize requirements generator."""
        logger.debug("Initialized RequirementsGenerator")
    
    def generate_from_code(
        self,
        code: str,
        include_optional: bool = False,
        include_dev: bool = False
    ) -> str:
        """
        Generate requirements.txt from code analysis.
        
        Args:
            code: Python code to analyze
            include_optional: Include optional dependencies
            include_dev: Include development dependencies
        
        Returns:
            requirements.txt content as string
        """
        logger.info("Analyzing code for dependencies...")
        
        # Extract imports from code
        imports = self._extract_imports(code)
        
        # Map imports to packages
        packages = self._map_imports_to_packages(imports)
        
        # Create dependency info objects
        dependencies = self._create_dependencies(packages)
        
        # Generate requirements content
        content = self._format_requirements(
            dependencies,
            include_optional=include_optional,
            include_dev=include_dev
        )
        
        logger.info(f"Generated requirements with {len(dependencies)} packages")
        return content
    
    def generate_from_config(
        self,
        config: Dict[str, Any],
        include_optional: bool = False,
        include_dev: bool = False
    ) -> str:
        """
        Generate requirements.txt from experiment configuration.
        
        Args:
            config: Experiment configuration dictionary
            include_optional: Include optional dependencies
            include_dev: Include development dependencies
        
        Returns:
            requirements.txt content as string
        """
        logger.info("Generating requirements from configuration...")
        
        # Determine required packages from config
        packages = self._determine_packages_from_config(config)
        
        # Create dependency info objects
        dependencies = self._create_dependencies(packages)
        
        # Generate requirements content
        content = self._format_requirements(
            dependencies,
            include_optional=include_optional,
            include_dev=include_dev
        )
        
        logger.info(f"Generated requirements with {len(dependencies)} packages")
        return content
    
    def generate_modular_requirements(
        self,
        config: Dict[str, Any],
        code_sections: Optional[Dict[str, str]] = None
    ) -> Dict[str, str]:
        """
        Generate modular requirements files for different components.
        
        Creates separate requirements files for:
        - Core dependencies (requirements.txt)
        - Preprocessing (requirements-preprocessing.txt)
        - Training (requirements-training.txt)
        - Evaluation (requirements-evaluation.txt)
        - Prediction (requirements-prediction.txt)
        - Development (requirements-dev.txt)
        
        Args:
            config: Experiment configuration
            code_sections: Optional dict of code sections to analyze
        
        Returns:
            Dictionary mapping filename to content
        """
        logger.info("Generating modular requirements files...")
        
        requirements_files = {}
        
        # Core requirements (always needed)
        core_packages = {'numpy', 'pandas'}
        core_deps = self._create_dependencies(core_packages)
        requirements_files['requirements.txt'] = self._format_requirements(
            core_deps,
            title="Core Dependencies"
        )
        
        # Preprocessing requirements
        if code_sections and 'preprocessing' in code_sections:
            preprocessing_imports = self._extract_imports(code_sections['preprocessing'])
            preprocessing_packages = self._map_imports_to_packages(preprocessing_imports)
            preprocessing_deps = self._create_dependencies(preprocessing_packages)
            requirements_files['requirements-preprocessing.txt'] = self._format_requirements(
                preprocessing_deps,
                title="Preprocessing Dependencies"
            )
        else:
            # Default preprocessing packages
            preprocessing_packages = {'scikit-learn'}
            preprocessing_deps = self._create_dependencies(preprocessing_packages)
            requirements_files['requirements-preprocessing.txt'] = self._format_requirements(
                preprocessing_deps,
                title="Preprocessing Dependencies"
            )
        
        # Training requirements
        if code_sections and 'training' in code_sections:
            training_imports = self._extract_imports(code_sections['training'])
            training_packages = self._map_imports_to_packages(training_imports)
        else:
            # Determine from model type
            model_type = config.get('model_type', '')
            training_packages = self._get_model_packages(model_type)
        
        training_deps = self._create_dependencies(training_packages)
        requirements_files['requirements-training.txt'] = self._format_requirements(
            training_deps,
            title="Training Dependencies"
        )
        
        # Evaluation requirements
        evaluation_packages = {'scikit-learn', 'matplotlib', 'seaborn'}
        evaluation_deps = self._create_dependencies(evaluation_packages)
        requirements_files['requirements-evaluation.txt'] = self._format_requirements(
            evaluation_deps,
            title="Evaluation Dependencies"
        )
        
        # Prediction requirements (minimal)
        prediction_packages = {'numpy', 'pandas', 'joblib'}
        prediction_packages.update(self._get_model_packages(config.get('model_type', '')))
        prediction_deps = self._create_dependencies(prediction_packages)
        requirements_files['requirements-prediction.txt'] = self._format_requirements(
            prediction_deps,
            title="Prediction/Inference Dependencies"
        )
        
        # Development requirements
        dev_packages = {'pytest', 'pytest-cov', 'black', 'flake8', 'mypy'}
        dev_deps = self._create_dependencies(dev_packages)
        requirements_files['requirements-dev.txt'] = self._format_requirements(
            dev_deps,
            title="Development Dependencies"
        )
        
        logger.info(f"Generated {len(requirements_files)} modular requirements files")
        return requirements_files
    
    def _extract_imports(self, code: str) -> Set[str]:
        """
        Extract import statements from code.
        
        Args:
            code: Python code
        
        Returns:
            Set of imported module names
        """
        imports = set()
        
        # Match "import module" and "from module import ..."
        import_patterns = [
            r'^import\s+(\w+)',
            r'^from\s+(\w+)',
        ]
        
        for line in code.split('\n'):
            line = line.strip()
            for pattern in import_patterns:
                match = re.match(pattern, line)
                if match:
                    module = match.group(1)
                    imports.add(module)
        
        return imports
    
    def _map_imports_to_packages(self, imports: Set[str]) -> Set[str]:
        """
        Map import names to package names.
        
        Args:
            imports: Set of import names
        
        Returns:
            Set of package names
        """
        packages = set()
        
        for imp in imports:
            # Check if there's a mapping
            package = self.IMPORT_TO_PACKAGE.get(imp, imp)
            
            # Only include known packages
            if package in self.PACKAGE_VERSIONS:
                packages.add(package)
        
        return packages
    
    def _create_dependencies(self, packages: Set[str]) -> List[DependencyInfo]:
        """
        Create DependencyInfo objects for packages.
        
        Args:
            packages: Set of package names
        
        Returns:
            List of DependencyInfo objects
        """
        dependencies = []
        
        for package in sorted(packages):
            version = self.PACKAGE_VERSIONS.get(package, '1.0.0')
            purpose = self.PACKAGE_PURPOSES.get(package, 'Required dependency')
            
            dep = DependencyInfo(
                package=package,
                version=version,
                purpose=purpose
            )
            dependencies.append(dep)
        
        return dependencies
    
    def _format_requirements(
        self,
        dependencies: List[DependencyInfo],
        include_optional: bool = False,
        include_dev: bool = False,
        title: Optional[str] = None
    ) -> str:
        """
        Format dependencies as requirements.txt content.
        
        Args:
            dependencies: List of DependencyInfo objects
            include_optional: Include optional dependencies
            include_dev: Include development dependencies
            title: Optional title for the requirements file
        
        Returns:
            Formatted requirements.txt content
        """
        lines = []
        
        # Header
        lines.append("# " + "=" * 78)
        if title:
            lines.append(f"# {title}")
        else:
            lines.append("# Python Requirements")
        lines.append("# Auto-generated by AI-Playground")
        lines.append(f"# Generated: {datetime.now().isoformat()}")
        lines.append("# " + "=" * 78)
        lines.append("")
        
        # Core dependencies
        core_deps = [d for d in dependencies if not d.optional]
        if core_deps:
            lines.append("# Core Dependencies")
            lines.append("# " + "-" * 78)
            for dep in core_deps:
                lines.append(f"# {dep.purpose}")
                lines.append(dep.to_requirement_line())
                lines.append("")
        
        # Optional dependencies
        if include_optional:
            optional_deps = [d for d in dependencies if d.optional]
            if optional_deps:
                lines.append("# Optional Dependencies")
                lines.append("# " + "-" * 78)
                for dep in optional_deps:
                    lines.append(f"# {dep.purpose}")
                    lines.append(dep.to_requirement_line())
                    lines.append("")
        
        # Installation instructions
        lines.append("# " + "=" * 78)
        lines.append("# Installation Instructions")
        lines.append("# " + "=" * 78)
        lines.append("#")
        lines.append("# Install all dependencies:")
        lines.append("#   pip install -r requirements.txt")
        lines.append("#")
        lines.append("# Install with specific Python version:")
        lines.append("#   python3.9 -m pip install -r requirements.txt")
        lines.append("#")
        lines.append("# Create virtual environment first (recommended):")
        lines.append("#   python -m venv venv")
        lines.append("#   source venv/bin/activate  # On Windows: venv\\Scripts\\activate")
        lines.append("#   pip install -r requirements.txt")
        lines.append("")
        
        return '\n'.join(lines)
    
    def _determine_packages_from_config(self, config: Dict[str, Any]) -> Set[str]:
        """
        Determine required packages from configuration.
        
        Args:
            config: Experiment configuration
        
        Returns:
            Set of required package names
        """
        packages = {'numpy', 'pandas'}  # Always needed
        
        # Add model-specific packages
        model_type = config.get('model_type', '')
        packages.update(self._get_model_packages(model_type))
        
        # Add preprocessing packages
        preprocessing_steps = config.get('preprocessing_steps', [])
        if preprocessing_steps:
            packages.add('scikit-learn')
        
        # Add evaluation packages
        if config.get('include_evaluation', True):
            packages.update({'matplotlib', 'seaborn'})
        
        # Add serialization
        packages.add('joblib')
        
        return packages
    
    def _get_model_packages(self, model_type: str) -> Set[str]:
        """
        Get required packages for a specific model type.
        
        Args:
            model_type: Model type identifier
        
        Returns:
            Set of required packages
        """
        packages = {'scikit-learn'}  # Base ML library
        
        # XGBoost models
        if 'xgb' in model_type.lower() or 'xgboost' in model_type.lower():
            packages.add('xgboost')
        
        # LightGBM models
        if 'lgb' in model_type.lower() or 'lightgbm' in model_type.lower():
            packages.add('lightgbm')
        
        # CatBoost models
        if 'catboost' in model_type.lower():
            packages.add('catboost')
        
        # Deep learning models
        if 'neural' in model_type.lower() or 'deep' in model_type.lower():
            packages.update({'tensorflow', 'keras'})
        
        return packages
    
    def generate_docker_requirements(
        self,
        config: Dict[str, Any],
        python_version: str = "3.9"
    ) -> str:
        """
        Generate requirements optimized for Docker containers.
        
        Uses pinned versions for reproducibility.
        
        Args:
            config: Experiment configuration
            python_version: Python version for container
        
        Returns:
            Docker-optimized requirements.txt content
        """
        packages = self._determine_packages_from_config(config)
        dependencies = self._create_dependencies(packages)
        
        lines = []
        lines.append(f"# Docker Requirements (Python {python_version})")
        lines.append(f"# Generated: {datetime.now().isoformat()}")
        lines.append("# Pinned versions for reproducibility")
        lines.append("")
        
        for dep in dependencies:
            lines.append(dep.to_requirement_line(include_version=True))
        
        return '\n'.join(lines)
    
    def generate_conda_environment(
        self,
        config: Dict[str, Any],
        env_name: str = "ml-env",
        python_version: str = "3.9"
    ) -> str:
        """
        Generate conda environment.yml file.
        
        Args:
            config: Experiment configuration
            env_name: Name for conda environment
            python_version: Python version
        
        Returns:
            environment.yml content
        """
        packages = self._determine_packages_from_config(config)
        
        lines = []
        lines.append(f"name: {env_name}")
        lines.append("channels:")
        lines.append("  - conda-forge")
        lines.append("  - defaults")
        lines.append("dependencies:")
        lines.append(f"  - python={python_version}")
        
        for package in sorted(packages):
            version = self.PACKAGE_VERSIONS.get(package, '')
            if version:
                lines.append(f"  - {package}={version}")
            else:
                lines.append(f"  - {package}")
        
        return '\n'.join(lines)


def generate_requirements(
    config: Dict[str, Any],
    output_format: str = 'pip',
    modular: bool = False,
    code_sections: Optional[Dict[str, str]] = None
) -> Dict[str, str]:
    """
    Convenience function to generate requirements files.
    
    Args:
        config: Experiment configuration
        output_format: 'pip', 'docker', or 'conda'
        modular: Generate modular requirements files
        code_sections: Optional code sections for analysis
    
    Returns:
        Dictionary mapping filename to content
    
    Example:
        >>> config = {'model_type': 'random_forest_classifier', ...}
        >>> requirements = generate_requirements(config, modular=True)
        >>> for filename, content in requirements.items():
        ...     with open(filename, 'w') as f:
        ...         f.write(content)
    """
    generator = RequirementsGenerator()
    
    if modular:
        return generator.generate_modular_requirements(config, code_sections)
    
    if output_format == 'pip':
        content = generator.generate_from_config(config)
        return {'requirements.txt': content}
    
    elif output_format == 'docker':
        content = generator.generate_docker_requirements(config)
        return {'requirements.txt': content}
    
    elif output_format == 'conda':
        content = generator.generate_conda_environment(config)
        return {'environment.yml': content}
    
    else:
        raise ValueError(f"Unknown output_format: {output_format}")
