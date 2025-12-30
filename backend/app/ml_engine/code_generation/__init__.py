"""
Code Generation Package

Provides tools for generating production-ready Python code from ML experiments.

Modules:
    templates: Jinja2 templates for code generation
    generator: Main code generation logic [Coming in ML-63-67]
"""

from app.ml_engine.code_generation.templates import (
    TEMPLATES,
    get_template,
    create_notebook_cell,
    NOTEBOOK_TEMPLATE,
)

__all__ = [
    "TEMPLATES",
    "get_template",
    "create_notebook_cell",
    "NOTEBOOK_TEMPLATE",
]
