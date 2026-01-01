"""
Experiment Configuration Serialization Service.

This service handles serialization and deserialization of complete experiment
configurations including preprocessing steps, model configurations, and results.
Enables experiment reproducibility and sharing.
"""

import json
from typing import Dict, Any, Optional, List
from uuid import UUID
from datetime import datetime
from pathlib import Path
from sqlalchemy.orm import Session

from app.models.experiment import Experiment
from app.models.preprocessing_step import PreprocessingStep
from app.models.model_run import ModelRun
from app.models.dataset import Dataset
from app.utils.logger import get_logger

logger = get_logger("experiment_config")


class ExperimentConfigSerializer:
    """
    Serializes and deserializes experiment configurations.
    
    Handles complete experiment state including:
    - Dataset information
    - Preprocessing pipeline configuration
    - Model configurations and hyperparameters
    - Training results and metrics
    - Metadata and timestamps
    """
    
    VERSION = "1.0.0"
    
    def __init__(self, db: Session):
        """
        Initialize the serializer.
        
        Args:
            db: Database session
        """
        self.db = db
    
    def serialize_experiment(
        self,
        experiment_id: UUID,
        include_results: bool = True,
        include_artifacts: bool = False
    ) -> Dict[str, Any]:
        """
        Serialize a complete experiment configuration.
        
        Args:
            experiment_id: ID of the experiment to serialize
            include_results: Whether to include training results and metrics
            include_artifacts: Whether to include paths to model artifacts
        
        Returns:
            Dictionary containing complete experiment configuration
        
        Raises:
            ValueError: If experiment not found
        """
        # Fetch experiment
        experiment = self.db.query(Experiment).filter(
            Experiment.id == experiment_id
        ).first()
        
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        logger.info(f"Serializing experiment {experiment_id}")
        
        # Build configuration
        config = {
            "version": self.VERSION,
            "experiment": self._serialize_experiment_info(experiment),
            "dataset": self._serialize_dataset_info(experiment.dataset),
            "preprocessing": self._serialize_preprocessing_steps(experiment.dataset_id),
            "models": self._serialize_model_runs(
                experiment_id,
                include_results=include_results,
                include_artifacts=include_artifacts
            ),
            "metadata": {
                "serialized_at": datetime.utcnow().isoformat(),
                "serializer_version": self.VERSION
            }
        }
        
        logger.info(f"Experiment {experiment_id} serialized successfully")
        
        return config
    
    def _serialize_experiment_info(self, experiment: Experiment) -> Dict[str, Any]:
        """Serialize basic experiment information."""
        return {
            "id": str(experiment.id),
            "name": experiment.name,
            "status": experiment.status.value if hasattr(experiment.status, 'value') else experiment.status,
            "created_at": experiment.created_at.isoformat() if experiment.created_at else None,
            "user_id": str(experiment.user_id),
            "dataset_id": str(experiment.dataset_id)
        }
    
    def _serialize_dataset_info(self, dataset: Dataset) -> Dict[str, Any]:
        """Serialize dataset information."""
        return {
            "id": str(dataset.id),
            "name": dataset.name,
            "file_path": dataset.file_path,
            "shape": dataset.shape,
            "dtypes": dataset.dtypes,
            "missing_values": dataset.missing_values,
            "uploaded_at": dataset.uploaded_at.isoformat() if dataset.uploaded_at else None
        }
    
    def _serialize_preprocessing_steps(self, dataset_id: UUID) -> List[Dict[str, Any]]:
        """Serialize preprocessing pipeline configuration."""
        steps = self.db.query(PreprocessingStep).filter(
            PreprocessingStep.dataset_id == dataset_id,
            PreprocessingStep.is_active == True
        ).order_by(PreprocessingStep.order).all()
        
        return [
            {
                "id": str(step.id),
                "step_type": step.step_type,
                "parameters": step.parameters,
                "column_name": step.column_name,
                "order": step.order
            }
            for step in steps
        ]
    
    def _serialize_model_runs(
        self,
        experiment_id: UUID,
        include_results: bool = True,
        include_artifacts: bool = False
    ) -> List[Dict[str, Any]]:
        """Serialize model run configurations and results."""
        model_runs = self.db.query(ModelRun).filter(
            ModelRun.experiment_id == experiment_id
        ).all()
        
        serialized_runs = []
        
        for run in model_runs:
            run_config = {
                "id": str(run.id),
                "model_type": run.model_type,
                "hyperparameters": run.hyperparameters,
                "status": run.status,
                "created_at": run.created_at.isoformat() if run.created_at else None
            }
            
            if include_results:
                run_config.update({
                    "metrics": run.metrics,
                    "training_time": run.training_time,
                    "run_metadata": run.run_metadata
                })
            
            if include_artifacts:
                run_config["model_artifact_path"] = run.model_artifact_path
            
            serialized_runs.append(run_config)
        
        return serialized_runs
    
    def save_to_file(
        self,
        experiment_id: UUID,
        file_path: Path,
        include_results: bool = True,
        include_artifacts: bool = False,
        pretty: bool = True
    ) -> Path:
        """
        Serialize experiment and save to JSON file.
        
        Args:
            experiment_id: ID of the experiment
            file_path: Path to save the configuration
            include_results: Whether to include results
            include_artifacts: Whether to include artifact paths
            pretty: Whether to format JSON with indentation
        
        Returns:
            Path to the saved file
        """
        config = self.serialize_experiment(
            experiment_id,
            include_results=include_results,
            include_artifacts=include_artifacts
        )
        
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w') as f:
            if pretty:
                json.dump(config, f, indent=2, default=str)
            else:
                json.dump(config, f, default=str)
        
        logger.info(f"Experiment configuration saved to {file_path}")
        
        return file_path
    
    def load_from_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Load experiment configuration from JSON file.
        
        Args:
            file_path: Path to the configuration file
        
        Returns:
            Experiment configuration dictionary
        
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        with open(file_path, 'r') as f:
            config = json.load(f)
        
        # Validate version
        if config.get("version") != self.VERSION:
            logger.warning(
                f"Configuration version mismatch: {config.get('version')} != {self.VERSION}"
            )
        
        logger.info(f"Experiment configuration loaded from {file_path}")
        
        return config
    
    def compare_experiments(
        self,
        experiment_id_1: UUID,
        experiment_id_2: UUID
    ) -> Dict[str, Any]:
        """
        Compare two experiment configurations.
        
        Args:
            experiment_id_1: First experiment ID
            experiment_id_2: Second experiment ID
        
        Returns:
            Dictionary with comparison results
        """
        config1 = self.serialize_experiment(experiment_id_1, include_results=True)
        config2 = self.serialize_experiment(experiment_id_2, include_results=True)
        
        comparison = {
            "experiments": {
                "experiment_1": config1["experiment"]["name"],
                "experiment_2": config2["experiment"]["name"]
            },
            "differences": {
                "preprocessing": self._compare_preprocessing(
                    config1["preprocessing"],
                    config2["preprocessing"]
                ),
                "models": self._compare_models(
                    config1["models"],
                    config2["models"]
                )
            },
            "summary": {
                "same_preprocessing": self._is_same_preprocessing(
                    config1["preprocessing"],
                    config2["preprocessing"]
                ),
                "same_models": self._is_same_models(
                    config1["models"],
                    config2["models"]
                )
            }
        }
        
        return comparison
    
    def _compare_preprocessing(
        self,
        steps1: List[Dict[str, Any]],
        steps2: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compare preprocessing steps between two experiments."""
        return {
            "num_steps": {
                "experiment_1": len(steps1),
                "experiment_2": len(steps2)
            },
            "step_types_1": [s["step_type"] for s in steps1],
            "step_types_2": [s["step_type"] for s in steps2],
            "identical": steps1 == steps2
        }
    
    def _compare_models(
        self,
        models1: List[Dict[str, Any]],
        models2: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compare model configurations between two experiments."""
        return {
            "num_models": {
                "experiment_1": len(models1),
                "experiment_2": len(models2)
            },
            "model_types_1": [m["model_type"] for m in models1],
            "model_types_2": [m["model_type"] for m in models2]
        }
    
    def _is_same_preprocessing(
        self,
        steps1: List[Dict[str, Any]],
        steps2: List[Dict[str, Any]]
    ) -> bool:
        """Check if preprocessing configurations are identical."""
        if len(steps1) != len(steps2):
            return False
        
        for s1, s2 in zip(steps1, steps2):
            if (s1["step_type"] != s2["step_type"] or
                s1["parameters"] != s2["parameters"]):
                return False
        
        return True
    
    def _is_same_models(
        self,
        models1: List[Dict[str, Any]],
        models2: List[Dict[str, Any]]
    ) -> bool:
        """Check if model configurations are identical."""
        if len(models1) != len(models2):
            return False
        
        types1 = sorted([m["model_type"] for m in models1])
        types2 = sorted([m["model_type"] for m in models2])
        
        return types1 == types2
    
    def export_for_reproduction(
        self,
        experiment_id: UUID,
        output_dir: Path
    ) -> Dict[str, Path]:
        """
        Export complete experiment configuration for reproduction.
        
        Creates a package with:
        - experiment_config.json: Full configuration
        - preprocessing_config.json: Preprocessing steps only
        - model_configs.json: Model configurations only
        - README.md: Instructions for reproduction
        
        Args:
            experiment_id: ID of the experiment
            output_dir: Directory to save export files
        
        Returns:
            Dictionary mapping file types to paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get full configuration
        config = self.serialize_experiment(
            experiment_id,
            include_results=True,
            include_artifacts=False
        )
        
        # Save full configuration
        full_config_path = output_dir / "experiment_config.json"
        with open(full_config_path, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        
        # Save preprocessing configuration
        preprocessing_path = output_dir / "preprocessing_config.json"
        with open(preprocessing_path, 'w') as f:
            json.dump(config["preprocessing"], f, indent=2, default=str)
        
        # Save model configurations
        models_path = output_dir / "model_configs.json"
        with open(models_path, 'w') as f:
            json.dump(config["models"], f, indent=2, default=str)
        
        # Create README
        readme_path = output_dir / "README.md"
        self._create_reproduction_readme(config, readme_path)
        
        logger.info(f"Experiment {experiment_id} exported to {output_dir}")
        
        return {
            "full_config": full_config_path,
            "preprocessing": preprocessing_path,
            "models": models_path,
            "readme": readme_path
        }
    
    def _create_reproduction_readme(
        self,
        config: Dict[str, Any],
        readme_path: Path
    ) -> None:
        """Create README with reproduction instructions."""
        experiment_name = config["experiment"]["name"]
        num_steps = len(config["preprocessing"])
        num_models = len(config["models"])
        
        readme_content = f"""# Experiment Reproduction Guide

## Experiment: {experiment_name}

### Overview
- **Experiment ID**: {config["experiment"]["id"]}
- **Created**: {config["experiment"]["created_at"]}
- **Status**: {config["experiment"]["status"]}
- **Preprocessing Steps**: {num_steps}
- **Models Trained**: {num_models}

### Dataset Information
- **Name**: {config["dataset"]["name"]}
- **Shape**: {config["dataset"]["shape"]}
- **Data Types**: {len(config["dataset"]["dtypes"])} columns

### Preprocessing Pipeline

The following preprocessing steps were applied in order:

"""
        
        for i, step in enumerate(config["preprocessing"], 1):
            readme_content += f"{i}. **{step['step_type']}**\n"
            if step["parameters"]:
                readme_content += f"   - Parameters: {json.dumps(step['parameters'], indent=6)}\n"
            if step["column_name"]:
                readme_content += f"   - Column: {step['column_name']}\n"
            readme_content += "\n"
        
        readme_content += """### Models

The following models were trained:

"""
        
        for i, model in enumerate(config["models"], 1):
            readme_content += f"{i}. **{model['model_type']}**\n"
            if model.get("hyperparameters"):
                readme_content += f"   - Hyperparameters: {json.dumps(model['hyperparameters'], indent=6)}\n"
            if model.get("metrics"):
                readme_content += f"   - Metrics: {json.dumps(model['metrics'], indent=6)}\n"
            readme_content += "\n"
        
        readme_content += """### Reproduction Steps

1. **Load the dataset**:
   ```python
   import pandas as pd
   df = pd.read_csv('your_dataset.csv')
   ```

2. **Apply preprocessing**:
   ```python
   from app.ml_engine.preprocessing.pipeline import Pipeline
   
   # Load preprocessing configuration
   with open('preprocessing_config.json', 'r') as f:
       preprocessing_config = json.load(f)
   
   # Reconstruct and apply pipeline
   pipeline = Pipeline.from_config(preprocessing_config)
   X_transformed = pipeline.fit_transform(df)
   ```

3. **Train models**:
   ```python
   from app.ml_engine.models.registry import ModelFactory
   
   # Load model configurations
   with open('model_configs.json', 'r') as f:
       model_configs = json.load(f)
   
   # Train each model
   for model_config in model_configs:
       model = ModelFactory.create_model(
           model_config['model_type'],
           **model_config['hyperparameters']
       )
       model.fit(X_train, y_train)
   ```

### Files in This Package

- `experiment_config.json`: Complete experiment configuration
- `preprocessing_config.json`: Preprocessing pipeline configuration
- `model_configs.json`: Model configurations and hyperparameters
- `README.md`: This file

### Notes

- Ensure you have the same dataset or a compatible dataset
- Install required dependencies: `pip install -r requirements.txt`
- Results may vary slightly due to random initialization

---

Generated by AI-Playground Experiment Configuration Serializer v{config["metadata"]["serializer_version"]}
"""
        
        with open(readme_path, 'w') as f:
            f.write(readme_content)


# Convenience functions

def serialize_experiment_to_file(
    db: Session,
    experiment_id: UUID,
    file_path: Path,
    include_results: bool = True
) -> Path:
    """
    Convenience function to serialize experiment to file.
    
    Args:
        db: Database session
        experiment_id: Experiment ID
        file_path: Output file path
        include_results: Whether to include results
    
    Returns:
        Path to saved file
    """
    serializer = ExperimentConfigSerializer(db)
    return serializer.save_to_file(
        experiment_id,
        file_path,
        include_results=include_results
    )


def load_experiment_from_file(
    db: Session,
    file_path: Path
) -> Dict[str, Any]:
    """
    Convenience function to load experiment from file.
    
    Args:
        db: Database session
        file_path: Configuration file path
    
    Returns:
        Experiment configuration
    """
    serializer = ExperimentConfigSerializer(db)
    return serializer.load_from_file(file_path)


def export_experiment_package(
    db: Session,
    experiment_id: UUID,
    output_dir: Path
) -> Dict[str, Path]:
    """
    Convenience function to export complete experiment package.
    
    Args:
        db: Database session
        experiment_id: Experiment ID
        output_dir: Output directory
    
    Returns:
        Dictionary of exported file paths
    """
    serializer = ExperimentConfigSerializer(db)
    return serializer.export_for_reproduction(experiment_id, output_dir)
