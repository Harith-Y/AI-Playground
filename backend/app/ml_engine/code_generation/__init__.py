"""
Code Generation Package

Provides tools for generating production-ready Python code from ML experiments.

Modules:
    templates: Jinja2 templates for code generation
    preprocessing_generator: Generate preprocessing code
    training_generator: Generate model training code
    evaluation_generator: Generate model evaluation code
    prediction_generator: Generate prediction/inference code
    generator: Main code generation logic [Coming in ML-67]
"""

from app.ml_engine.code_generation.templates import (
    TEMPLATES,
    get_template,
    create_notebook_cell,
    NOTEBOOK_TEMPLATE,
)

from app.ml_engine.code_generation.preprocessing_generator import (
    PreprocessingCodeGenerator,
    generate_preprocessing_code,
)

from app.ml_engine.code_generation.training_generator import (
    TrainingCodeGenerator,
    generate_training_code,
)

from app.ml_engine.code_generation.evaluation_generator import (
    EvaluationCodeGenerator,
    generate_evaluation_code,
)

from app.ml_engine.code_generation.prediction_generator import (
    PredictionCodeGenerator,
    generate_prediction_code,
)

__all__ = [
    "TEMPLATES",
    "get_template",
    "create_notebook_cell",
    "NOTEBOOK_TEMPLATE",
    "PreprocessingCodeGenerator",
    "generate_preprocessing_code",
    "TrainingCodeGenerator",
    "generate_training_code",
    "EvaluationCodeGenerator",
    "generate_evaluation_code",
    "PredictionCodeGenerator",
    "generate_prediction_code",
]
