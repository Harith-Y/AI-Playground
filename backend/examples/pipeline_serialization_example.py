"""
Pipeline Serialization Examples

This file demonstrates various ways to serialize and deserialize
preprocessing pipelines for reuse across different environments.

Key use cases:
1. Save fitted pipelines for production deployment
2. Share pipeline configurations across team members
3. Version control pipeline configurations
4. Export pipelines in different formats
5. Manage pipeline registry for organization
"""

from pathlib import Path
import pandas as pd
import numpy as np

from app.ml_engine.preprocessing.pipeline import Pipeline
from app.ml_engine.preprocessing.imputer import MeanImputer, MedianImputer
from app.ml_engine.preprocessing.scaler import StandardScaler, MinMaxScaler
from app.ml_engine.preprocessing.encoder import OneHotEncoder, LabelEncoder
from app.ml_engine.preprocessing.serializer import (
    PipelineSerializer,
    PipelineRegistry,
    save_pipeline,
    load_pipeline,
)


def example_1_basic_save_load():
    """
    Example 1: Basic Pipeline Save and Load

    This shows the simplest way to save and load a pipeline using pickle format.
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Pipeline Save and Load")
    print("="*70)

    # Create sample data
    df = pd.DataFrame({
        "age": [25, 30, np.nan, 40, 35, 28],
        "salary": [50000, 60000, 55000, np.nan, 70000, 58000],
        "department": ["Sales", "Engineering", "Sales", "HR", "Engineering", "Sales"]
    })

    print("\nOriginal Data:")
    print(df)

    # Create and fit pipeline
    pipeline = Pipeline(
        steps=[
            MeanImputer(columns=["age", "salary"]),
            StandardScaler(columns=["age", "salary"]),
            OneHotEncoder(columns=["department"])
        ],
        name="BasicPipeline"
    )

    print("\nFitting pipeline...")
    pipeline.fit(df)

    # Save pipeline (fitted)
    save_path = Path("data/pipelines/basic_pipeline.pkl")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving pipeline to {save_path}...")
    file_info = save_pipeline(pipeline, save_path, format="pickle")

    print(f"✓ Saved successfully!")
    print(f"  - Size: {file_info['size_bytes']:,} bytes")
    print(f"  - Format: {file_info['format']}")
    print(f"  - Checksum: {file_info['checksum'][:16]}...")

    # Load pipeline
    print(f"\nLoading pipeline from {save_path}...")
    loaded_pipeline = load_pipeline(save_path)

    print(f"✓ Loaded successfully!")
    print(f"  - Name: {loaded_pipeline.name}")
    print(f"  - Steps: {len(loaded_pipeline.steps)}")
    print(f"  - Fitted: {loaded_pipeline.fitted}")

    # Use loaded pipeline
    df_new = pd.DataFrame({
        "age": [32, np.nan, 45],
        "salary": [62000, 65000, np.nan],
        "department": ["Sales", "HR", "Engineering"]
    })

    print("\nNew data to transform:")
    print(df_new)

    transformed = loaded_pipeline.transform(df_new)
    print("\nTransformed data:")
    print(transformed.head())

    print("\n✓ Pipeline successfully saved, loaded, and used!")


def example_2_configuration_export():
    """
    Example 2: Export and Import Pipeline Configuration

    This shows how to save just the configuration (no fitted parameters)
    in JSON format for version control and sharing.
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Configuration Export and Import")
    print("="*70)

    # Create pipeline (unfitted)
    pipeline = Pipeline(
        steps=[
            MedianImputer(columns=["age", "income"]),
            MinMaxScaler(columns=["age", "income"]),
            LabelEncoder(columns=["category"])
        ],
        name="ConfigurablePipeline"
    )

    print(f"\nOriginal Pipeline:")
    print(f"  - Name: {pipeline.name}")
    print(f"  - Steps: {pipeline.get_step_names()}")

    # Save configuration as JSON
    serializer = PipelineSerializer(default_format="json")
    config_path = Path("data/configs/pipeline_config.json")
    config_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving configuration to {config_path}...")
    file_info = serializer.save(pipeline, config_path)

    print(f"✓ Configuration saved!")
    print(f"  - Size: {file_info['size_bytes']:,} bytes")
    print(f"  - Format: {file_info['format']}")

    # Read the JSON to show it's human-readable
    with open(config_path, 'r') as f:
        content = f.read()

    print("\nConfiguration content (first 300 chars):")
    print(content[:300] + "...")

    # Load configuration and reconstruct pipeline
    print(f"\nLoading configuration from {config_path}...")
    loaded_data = serializer.load(config_path)

    # Reconstruct pipeline from config
    if isinstance(loaded_data, dict) and "pipeline" in loaded_data:
        config = loaded_data["pipeline"]
    else:
        config = loaded_data

    reconstructed_pipeline = Pipeline.from_dict(config)

    print(f"✓ Pipeline reconstructed from configuration!")
    print(f"  - Name: {reconstructed_pipeline.name}")
    print(f"  - Steps: {reconstructed_pipeline.get_step_names()}")
    print(f"  - Fitted: {reconstructed_pipeline.fitted}")

    print("\n✓ Configuration successfully exported and imported!")


def example_3_compression():
    """
    Example 3: Using Compression

    Demonstrates different compression algorithms for reducing file size.
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Pipeline Compression")
    print("="*70)

    # Create a larger pipeline
    pipeline = Pipeline(
        steps=[
            MeanImputer(columns=[f"feature_{i}" for i in range(20)]),
            StandardScaler(columns=[f"feature_{i}" for i in range(20)]),
        ],
        name="LargePipeline"
    )

    # Fit with sample data
    df = pd.DataFrame({
        f"feature_{i}": np.random.randn(1000) for i in range(20)
    })
    pipeline.fit(df)

    # Test different compression methods
    compressions = ["none", "gzip", "bz2", "lzma"]
    results = {}

    for compression in compressions:
        serializer = PipelineSerializer(compression=compression)
        save_path = Path(f"data/pipelines/compressed_{compression}.pkl")

        file_info = serializer.save(pipeline, save_path)
        results[compression] = file_info

        print(f"\n{compression.upper():8} - {file_info['size_bytes']:>8,} bytes")

    # Compare compression ratios
    uncompressed_size = results["none"]["size_bytes"]
    print("\nCompression Ratios:")
    for compression, info in results.items():
        ratio = (1 - info["size_bytes"] / uncompressed_size) * 100
        print(f"  {compression:8} - {ratio:5.1f}% reduction")

    print("\n✓ Compression comparison complete!")


def example_4_multiple_formats():
    """
    Example 4: Multiple Serialization Formats

    Shows how to save pipelines in different formats for different use cases.
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Multiple Serialization Formats")
    print("="*70)

    # Create and fit pipeline
    df = pd.DataFrame({
        "age": [25, 30, 35, 40],
        "income": [50000, 60000, 70000, 80000]
    })

    pipeline = Pipeline(
        steps=[StandardScaler(columns=["age", "income"])],
        name="MultiFormatPipeline"
    )
    pipeline.fit(df)

    formats = {
        "pickle": "Full state, binary format (recommended for production)",
        "json": "Configuration only, human-readable (good for version control)",
    }

    # Try joblib and yaml if available
    try:
        import joblib
        formats["joblib"] = "Optimized binary format (good for large numpy arrays)"
    except ImportError:
        pass

    try:
        import yaml
        formats["yaml"] = "Configuration only, human-readable (alternative to JSON)"
    except ImportError:
        pass

    print("\nSaving pipeline in multiple formats:")

    for format_name, description in formats.items():
        serializer = PipelineSerializer(default_format=format_name)

        ext_map = {
            "pickle": "pkl",
            "json": "json",
            "joblib": "joblib",
            "yaml": "yml"
        }

        save_path = Path(f"data/pipelines/multi_format.{ext_map[format_name]}")
        file_info = serializer.save(pipeline, save_path)

        print(f"\n{format_name.upper():10} - {file_info['size_bytes']:>8,} bytes")
        print(f"           {description}")

    print("\n✓ Multiple formats demonstrated!")


def example_5_pipeline_registry():
    """
    Example 5: Using Pipeline Registry

    Demonstrates how to use the registry to organize and manage multiple pipelines.
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: Pipeline Registry Management")
    print("="*70)

    # Initialize registry
    registry_path = Path("data/pipeline_registry.json")
    registry = PipelineRegistry(registry_path)

    print(f"Registry initialized at: {registry_path}")

    # Create and save multiple pipelines
    pipelines_data = [
        {
            "name": "Classification_v1",
            "steps": [MeanImputer(columns=["age"]), StandardScaler(columns=["age"])],
            "tags": ["classification", "v1"],
            "description": "Basic classification preprocessing pipeline"
        },
        {
            "name": "Regression_v1",
            "steps": [MedianImputer(columns=["salary"]), MinMaxScaler(columns=["salary"])],
            "tags": ["regression", "v1"],
            "description": "Basic regression preprocessing pipeline"
        },
        {
            "name": "Classification_v2",
            "steps": [MeanImputer(columns=["age"]), StandardScaler(columns=["age"]), OneHotEncoder(columns=["category"])],
            "tags": ["classification", "v2", "production"],
            "description": "Enhanced classification pipeline with encoding"
        },
    ]

    print("\nRegistering pipelines:")
    for pipeline_data in pipelines_data:
        pipeline = Pipeline(steps=pipeline_data["steps"], name=pipeline_data["name"])

        # Save pipeline
        save_path = Path(f"data/pipelines/{pipeline_data['name']}.pkl")
        file_info = save_pipeline(pipeline, save_path)

        # Register in catalog
        pipeline_id = pipeline_data["name"].lower().replace("_", "-")
        registry.register(
            pipeline_id,
            file_info,
            tags=pipeline_data["tags"],
            description=pipeline_data["description"]
        )

        print(f"  ✓ {pipeline_data['name']}")

    # List all pipelines
    print("\nAll registered pipelines:")
    all_pipelines = registry.list()
    for pid, info in all_pipelines.items():
        print(f"  - {pid}: {info['description']}")

    # Filter by tags
    print("\nPipelines tagged 'classification':")
    classification_pipelines = registry.list(tags=["classification"])
    for pid, info in classification_pipelines.items():
        print(f"  - {pid}: {', '.join(info['tags'])}")

    # Search by keyword
    print("\nSearch for 'production':")
    results = registry.search("production")
    for pid, info in results.items():
        print(f"  - {pid}: {info['description']}")

    # Get specific pipeline info
    print("\nGet info for 'classification-v2':")
    info = registry.get("classification-v2")
    if info:
        print(f"  Path: {info['path']}")
        print(f"  Format: {info['format']}")
        print(f"  Size: {info['size_bytes']:,} bytes")
        print(f"  Tags: {', '.join(info['tags'])}")

    print("\n✓ Registry management demonstrated!")


def example_6_production_workflow():
    """
    Example 6: Production Workflow

    Shows a complete workflow from development to production deployment.
    """
    print("\n" + "="*70)
    print("EXAMPLE 6: Production Deployment Workflow")
    print("="*70)

    # Step 1: Development - Create and test pipeline
    print("\n[DEV] Creating and testing pipeline...")
    df_train = pd.DataFrame({
        "age": [25, 30, np.nan, 40, 35],
        "salary": [50000, 60000, 55000, np.nan, 70000],
        "dept": ["Sales", "Eng", "Sales", "HR", "Eng"]
    })

    pipeline = Pipeline(
        steps=[
            MeanImputer(columns=["age", "salary"]),
            StandardScaler(columns=["age", "salary"]),
            OneHotEncoder(columns=["dept"])
        ],
        name="ProductionPipeline_v1.0.0"
    )

    pipeline.fit(df_train)
    print("  ✓ Pipeline trained on development data")

    # Step 2: Save configuration for version control
    print("\n[DEV] Saving configuration for version control...")
    config_path = Path("data/configs/production_pipeline_v1.0.0.json")
    serializer = PipelineSerializer(default_format="json")
    serializer.save(pipeline, config_path)
    print(f"  ✓ Configuration saved to {config_path}")
    print("  → Commit this file to Git")

    # Step 3: Save fitted pipeline for deployment
    print("\n[DEV] Saving fitted pipeline for deployment...")
    model_path = Path("data/models/production_pipeline_v1.0.0.pkl.gz")
    serializer = PipelineSerializer(default_format="pickle", compression="gzip")
    file_info = serializer.save(
        pipeline,
        model_path,
        metadata={
            "version": "1.0.0",
            "trained_on": "2024-01-15",
            "dataset_size": len(df_train),
            "author": "Data Science Team"
        }
    )
    print(f"  ✓ Fitted pipeline saved to {model_path}")
    print(f"  → Deploy this file to production servers")
    print(f"  → Checksum: {file_info['checksum'][:16]}... (for verification)")

    # Step 4: Register in pipeline registry
    print("\n[OPS] Registering in production registry...")
    registry = PipelineRegistry(Path("data/production_registry.json"))
    registry.register(
        "production-pipeline-v1.0.0",
        file_info,
        tags=["production", "v1.0.0", "classification"],
        description="Production classification pipeline v1.0.0"
    )
    print("  ✓ Pipeline registered in production registry")

    # Step 5: Production - Load and use
    print("\n[PROD] Loading pipeline in production...")
    prod_pipeline = load_pipeline(model_path)
    print(f"  ✓ Pipeline loaded: {prod_pipeline.name}")

    # Transform new data
    df_new = pd.DataFrame({
        "age": [28, np.nan],
        "salary": [58000, 65000],
        "dept": ["Sales", "Eng"]
    })

    print("\n[PROD] Transforming new data...")
    result = prod_pipeline.transform(df_new)
    print("  ✓ Data transformed successfully")
    print(f"  → Shape: {result.shape}")

    # Step 6: Verification
    print("\n[OPS] Verifying deployment...")
    print(f"  ✓ Pipeline name: {prod_pipeline.name}")
    print(f"  ✓ Pipeline fitted: {prod_pipeline.fitted}")
    print(f"  ✓ Number of steps: {len(prod_pipeline.steps)}")
    print(f"  ✓ Checksum matches: {file_info['checksum'][:16]}...")

    print("\n✓ Production deployment workflow complete!")


def example_7_versioning_and_rollback():
    """
    Example 7: Versioning and Rollback

    Demonstrates version management and rollback capabilities.
    """
    print("\n" + "="*70)
    print("EXAMPLE 7: Versioning and Rollback")
    print("="*70)

    # Create multiple versions
    versions = ["v1.0.0", "v1.1.0", "v2.0.0"]
    registry = PipelineRegistry(Path("data/version_registry.json"))

    print("\nCreating multiple versions:")
    for version in versions:
        pipeline = Pipeline(
            steps=[StandardScaler(columns=["age"])],
            name=f"Pipeline_{version}"
        )

        save_path = Path(f"data/pipelines/pipeline_{version}.pkl")
        file_info = save_pipeline(pipeline, save_path)

        registry.register(
            f"pipeline-{version}",
            file_info,
            tags=["versioned", version],
            description=f"Pipeline version {version}"
        )

        print(f"  ✓ {version} saved and registered")

    # List all versions
    print("\nAll versions:")
    all_versions = registry.list(tags=["versioned"])
    for pid in sorted(all_versions.keys()):
        print(f"  - {pid}")

    # Rollback scenario
    print("\nSimulating rollback from v2.0.0 to v1.1.0...")

    # Get v1.1.0 from registry
    v1_1_info = registry.get("pipeline-v1.1.0")
    if v1_1_info:
        print(f"  Loading {v1_1_info['path']}...")
        rollback_pipeline = load_pipeline(v1_1_info["path"])
        print(f"  ✓ Rolled back to {rollback_pipeline.name}")

    print("\n✓ Versioning and rollback demonstrated!")


def run_all_examples():
    """Run all examples."""
    print("\n" + "="*70)
    print("PIPELINE SERIALIZATION EXAMPLES")
    print("="*70)

    examples = [
        ("Basic Save and Load", example_1_basic_save_load),
        ("Configuration Export", example_2_configuration_export),
        ("Compression", example_3_compression),
        ("Multiple Formats", example_4_multiple_formats),
        ("Pipeline Registry", example_5_pipeline_registry),
        ("Production Workflow", example_6_production_workflow),
        ("Versioning and Rollback", example_7_versioning_and_rollback),
    ]

    for name, example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"\n❌ Error in {name}: {str(e)}")

    print("\n" + "="*70)
    print("ALL EXAMPLES COMPLETED")
    print("="*70)


if __name__ == "__main__":
    run_all_examples()
