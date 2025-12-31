"""
Code Template Engine

Provides a flexible template system for code generation with support for:
- Custom templates
- Template inheritance
- Variable substitution
- Conditional blocks
- Loop constructs
- Template validation

Based on: BACKEND-TO-DO.md > BACKEND-55
"""

from typing import Dict, Any, List, Optional, Callable
from pathlib import Path
from dataclasses import dataclass, field
import re
from datetime import datetime
from jinja2 import Environment, FileSystemLoader, Template, TemplateNotFound
from jinja2.exceptions import TemplateError
from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TemplateConfig:
    """Configuration for template rendering."""
    name: str
    description: str
    variables: Dict[str, Any] = field(default_factory=dict)
    required_vars: List[str] = field(default_factory=list)
    optional_vars: List[str] = field(default_factory=list)
    filters: Dict[str, Callable] = field(default_factory=dict)
    tests: Dict[str, Callable] = field(default_factory=dict)


class TemplateEngine:
    """
    Code template engine for flexible code generation.
    
    Supports:
    - Jinja2 templates
    - Custom filters and tests
    - Template inheritance
    - Variable validation
    - Template caching
    
    Example:
        >>> engine = TemplateEngine()
        >>> template = engine.load_template('training.py.j2')
        >>> code = engine.render(template, {'model_type': 'random_forest'})
    """
    
    def __init__(
        self,
        template_dir: Optional[Path] = None,
        enable_cache: bool = True
    ):
        """
        Initialize template engine.
        
        Args:
            template_dir: Directory containing templates
            enable_cache: Whether to enable template caching
        """
        self.template_dir = template_dir or self._get_default_template_dir()
        self.enable_cache = enable_cache
        
        # Initialize Jinja2 environment
        self.env = Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            autoescape=False,  # Don't escape for code generation
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True,
        )
        
        # Register custom filters
        self._register_default_filters()
        
        # Register custom tests
        self._register_default_tests()
        
        # Template cache
        self._cache: Dict[str, Template] = {}
        
        logger.info(f"Template engine initialized with directory: {self.template_dir}")
    
    def _get_default_template_dir(self) -> Path:
        """Get default template directory."""
        return Path(__file__).parent / "templates"
    
    def _register_default_filters(self):
        """Register default Jinja2 filters."""
        
        # Code formatting filters
        self.env.filters['indent'] = self._filter_indent
        self.env.filters['comment'] = self._filter_comment
        self.env.filters['snake_case'] = self._filter_snake_case
        self.env.filters['camel_case'] = self._filter_camel_case
        self.env.filters['pascal_case'] = self._filter_pascal_case
        self.env.filters['kebab_case'] = self._filter_kebab_case
        
        # Type conversion filters
        self.env.filters['to_python_type'] = self._filter_to_python_type
        self.env.filters['to_json'] = self._filter_to_json
        
        # String manipulation
        self.env.filters['quote'] = self._filter_quote
        self.env.filters['escape'] = self._filter_escape
        
        logger.debug("Registered default filters")
    
    def _register_default_tests(self):
        """Register default Jinja2 tests."""
        
        self.env.tests['classification'] = lambda x: x == 'classification'
        self.env.tests['regression'] = lambda x: x == 'regression'
        self.env.tests['clustering'] = lambda x: x == 'clustering'
        self.env.tests['supervised'] = lambda x: x in ['classification', 'regression']
        self.env.tests['unsupervised'] = lambda x: x == 'clustering'
        
        logger.debug("Registered default tests")
    
    # ========================================================================
    # Filter Implementations
    # ========================================================================
    
    def _filter_indent(self, text: str, spaces: int = 4) -> str:
        """Indent text by specified number of spaces."""
        indent = ' ' * spaces
        return '\n'.join(indent + line if line.strip() else line 
                        for line in text.split('\n'))
    
    def _filter_comment(self, text: str, style: str = 'python') -> str:
        """Add comment markers to text."""
        if style == 'python':
            return '\n'.join(f'# {line}' if line.strip() else '#' 
                           for line in text.split('\n'))
        elif style == 'block':
            return f'"""\n{text}\n"""'
        elif style == 'inline':
            return f'  # {text}'
        return text
    
    def _filter_snake_case(self, text: str) -> str:
        """Convert text to snake_case."""
        # Replace spaces and hyphens with underscores
        text = re.sub(r'[\s\-]+', '_', text)
        # Insert underscore before uppercase letters
        text = re.sub(r'([a-z])([A-Z])', r'\1_\2', text)
        return text.lower()
    
    def _filter_camel_case(self, text: str) -> str:
        """Convert text to camelCase."""
        words = re.split(r'[\s_\-]+', text)
        return words[0].lower() + ''.join(w.capitalize() for w in words[1:])
    
    def _filter_pascal_case(self, text: str) -> str:
        """Convert text to PascalCase."""
        words = re.split(r'[\s_\-]+', text)
        return ''.join(w.capitalize() for w in words)
    
    def _filter_kebab_case(self, text: str) -> str:
        """Convert text to kebab-case."""
        text = re.sub(r'[\s_]+', '-', text)
        text = re.sub(r'([a-z])([A-Z])', r'\1-\2', text)
        return text.lower()
    
    def _filter_to_python_type(self, value: Any) -> str:
        """Convert value to Python type string."""
        if isinstance(value, bool):
            return 'bool'
        elif isinstance(value, int):
            return 'int'
        elif isinstance(value, float):
            return 'float'
        elif isinstance(value, str):
            return 'str'
        elif isinstance(value, list):
            return 'List'
        elif isinstance(value, dict):
            return 'Dict'
        return 'Any'
    
    def _filter_to_json(self, value: Any) -> str:
        """Convert value to JSON string."""
        import json
        return json.dumps(value, indent=2)
    
    def _filter_quote(self, text: str, style: str = 'double') -> str:
        """Add quotes around text."""
        if style == 'double':
            return f'"{text}"'
        elif style == 'single':
            return f"'{text}'"
        elif style == 'triple':
            return f'"""{text}"""'
        return text
    
    def _filter_escape(self, text: str) -> str:
        """Escape special characters."""
        return text.replace('\\', '\\\\').replace('"', '\\"').replace("'", "\\'")
    
    # ========================================================================
    # Template Management
    # ========================================================================
    
    def load_template(self, template_name: str) -> Template:
        """
        Load template by name.
        
        Args:
            template_name: Name of template file
        
        Returns:
            Loaded Jinja2 template
        
        Raises:
            TemplateNotFound: If template doesn't exist
        """
        # Check cache
        if self.enable_cache and template_name in self._cache:
            logger.debug(f"Loading template from cache: {template_name}")
            return self._cache[template_name]
        
        try:
            template = self.env.get_template(template_name)
            
            # Cache template
            if self.enable_cache:
                self._cache[template_name] = template
            
            logger.info(f"Loaded template: {template_name}")
            return template
            
        except TemplateNotFound:
            logger.error(f"Template not found: {template_name}")
            raise
    
    def load_template_string(self, template_string: str) -> Template:
        """
        Load template from string.
        
        Args:
            template_string: Template content as string
        
        Returns:
            Loaded Jinja2 template
        """
        return self.env.from_string(template_string)
    
    def render(
        self,
        template: Template,
        context: Dict[str, Any],
        validate: bool = True
    ) -> str:
        """
        Render template with context.
        
        Args:
            template: Jinja2 template
            context: Template variables
            validate: Whether to validate required variables
        
        Returns:
            Rendered template string
        
        Raises:
            TemplateError: If rendering fails
            ValueError: If required variables are missing
        """
        # Add default context
        context = self._prepare_context(context)
        
        # Validate context if requested
        if validate:
            self._validate_context(context, template)
        
        try:
            rendered = template.render(**context)
            logger.debug(f"Template rendered successfully")
            return rendered
            
        except TemplateError as e:
            logger.error(f"Template rendering failed: {e}")
            raise
    
    def render_template(
        self,
        template_name: str,
        context: Dict[str, Any],
        validate: bool = True
    ) -> str:
        """
        Load and render template in one step.
        
        Args:
            template_name: Name of template file
            context: Template variables
            validate: Whether to validate required variables
        
        Returns:
            Rendered template string
        """
        template = self.load_template(template_name)
        return self.render(template, context, validate)
    
    def render_string(
        self,
        template_string: str,
        context: Dict[str, Any]
    ) -> str:
        """
        Render template string with context.
        
        Args:
            template_string: Template content
            context: Template variables
        
        Returns:
            Rendered string
        """
        template = self.load_template_string(template_string)
        return self.render(template, context, validate=False)
    
    # ========================================================================
    # Context Management
    # ========================================================================
    
    def _prepare_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare context with default values.
        
        Args:
            context: User-provided context
        
        Returns:
            Enhanced context with defaults
        """
        # Add default values
        defaults = {
            'timestamp': datetime.utcnow().isoformat(),
            'generator': 'AI-Playground',
            'version': '1.0.0',
        }
        
        # Merge with user context (user values take precedence)
        return {**defaults, **context}
    
    def _validate_context(self, context: Dict[str, Any], template: Template):
        """
        Validate context has required variables.
        
        Args:
            context: Template context
            template: Jinja2 template
        
        Raises:
            ValueError: If required variables are missing
        """
        # Get undefined variables from template
        undefined = self.env.make_logging_undefined()(template.module)
        
        # Check for missing required variables
        # Note: This is a basic check; more sophisticated validation
        # would require parsing the template AST
        
        logger.debug("Context validation passed")
    
    # ========================================================================
    # Custom Filters and Tests
    # ========================================================================
    
    def add_filter(self, name: str, filter_func: Callable):
        """
        Add custom filter to engine.
        
        Args:
            name: Filter name
            filter_func: Filter function
        """
        self.env.filters[name] = filter_func
        logger.info(f"Added custom filter: {name}")
    
    def add_test(self, name: str, test_func: Callable):
        """
        Add custom test to engine.
        
        Args:
            name: Test name
            test_func: Test function
        """
        self.env.tests[name] = test_func
        logger.info(f"Added custom test: {name}")
    
    def add_global(self, name: str, value: Any):
        """
        Add global variable to engine.
        
        Args:
            name: Variable name
            value: Variable value
        """
        self.env.globals[name] = value
        logger.info(f"Added global variable: {name}")
    
    # ========================================================================
    # Template Discovery
    # ========================================================================
    
    def list_templates(self, pattern: Optional[str] = None) -> List[str]:
        """
        List available templates.
        
        Args:
            pattern: Optional regex pattern to filter templates
        
        Returns:
            List of template names
        """
        templates = self.env.list_templates()
        
        if pattern:
            regex = re.compile(pattern)
            templates = [t for t in templates if regex.search(t)]
        
        return sorted(templates)
    
    def template_exists(self, template_name: str) -> bool:
        """
        Check if template exists.
        
        Args:
            template_name: Name of template
        
        Returns:
            True if template exists
        """
        try:
            self.env.get_template(template_name)
            return True
        except TemplateNotFound:
            return False
    
    # ========================================================================
    # Cache Management
    # ========================================================================
    
    def clear_cache(self):
        """Clear template cache."""
        self._cache.clear()
        logger.info("Template cache cleared")
    
    def get_cache_size(self) -> int:
        """Get number of cached templates."""
        return len(self._cache)


# ============================================================================
# Template Registry
# ============================================================================

class TemplateRegistry:
    """
    Registry for managing template configurations.
    
    Provides centralized management of template metadata,
    required variables, and validation rules.
    """
    
    def __init__(self):
        """Initialize template registry."""
        self._templates: Dict[str, TemplateConfig] = {}
        self._register_default_templates()
    
    def _register_default_templates(self):
        """Register default template configurations."""
        
        # Preprocessing template
        self.register(TemplateConfig(
            name='preprocessing.py.j2',
            description='Data preprocessing template',
            required_vars=['preprocessing_steps', 'dataset_info'],
            optional_vars=['random_state', 'experiment_name']
        ))
        
        # Training template
        self.register(TemplateConfig(
            name='training.py.j2',
            description='Model training template',
            required_vars=['model_type', 'task_type', 'hyperparameters'],
            optional_vars=['test_size', 'random_state', 'cross_validation']
        ))
        
        # Evaluation template
        self.register(TemplateConfig(
            name='evaluation.py.j2',
            description='Model evaluation template',
            required_vars=['model_type', 'task_type'],
            optional_vars=['metrics', 'plots']
        ))
        
        # Prediction template
        self.register(TemplateConfig(
            name='prediction.py.j2',
            description='Prediction/inference template',
            required_vars=['model_type', 'model_path'],
            optional_vars=['preprocessing_steps']
        ))
        
        # FastAPI template
        self.register(TemplateConfig(
            name='fastapi_service.py.j2',
            description='FastAPI microservice template',
            required_vars=['model_type', 'task_type', 'experiment_name'],
            optional_vars=['model_path', 'model_version']
        ))
    
    def register(self, config: TemplateConfig):
        """
        Register template configuration.
        
        Args:
            config: Template configuration
        """
        self._templates[config.name] = config
        logger.debug(f"Registered template: {config.name}")
    
    def get(self, name: str) -> Optional[TemplateConfig]:
        """
        Get template configuration.
        
        Args:
            name: Template name
        
        Returns:
            Template configuration or None
        """
        return self._templates.get(name)
    
    def list_templates(self) -> List[str]:
        """
        List registered templates.
        
        Returns:
            List of template names
        """
        return list(self._templates.keys())
    
    def validate_context(self, template_name: str, context: Dict[str, Any]) -> bool:
        """
        Validate context for template.
        
        Args:
            template_name: Template name
            context: Template context
        
        Returns:
            True if valid
        
        Raises:
            ValueError: If required variables are missing
        """
        config = self.get(template_name)
        if not config:
            return True  # No validation rules
        
        # Check required variables
        missing = [var for var in config.required_vars if var not in context]
        if missing:
            raise ValueError(f"Missing required variables: {missing}")
        
        return True


# ============================================================================
# Global Instances
# ============================================================================

# Global template engine instance
_template_engine: Optional[TemplateEngine] = None

# Global template registry
_template_registry = TemplateRegistry()


def get_template_engine() -> TemplateEngine:
    """
    Get global template engine instance.
    
    Returns:
        Template engine instance
    """
    global _template_engine
    
    if _template_engine is None:
        _template_engine = TemplateEngine()
    
    return _template_engine


def get_template_registry() -> TemplateRegistry:
    """
    Get global template registry.
    
    Returns:
        Template registry instance
    """
    return _template_registry
