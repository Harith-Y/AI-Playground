"""
Tests for Template Engine

Tests the code template engine functionality.
"""

import pytest
from pathlib import Path
from jinja2 import TemplateNotFound, TemplateError

from app.ml_engine.code_generation.template_engine import (
    TemplateEngine,
    TemplateRegistry,
    TemplateConfig,
    get_template_engine,
    get_template_registry,
)


class TestTemplateEngine:
    """Test TemplateEngine class."""
    
    @pytest.fixture
    def engine(self, tmp_path):
        """Create template engine with temporary directory."""
        template_dir = tmp_path / "templates"
        template_dir.mkdir()
        return TemplateEngine(template_dir=template_dir, enable_cache=True)
    
    def test_initialization(self, engine):
        """Test engine initialization."""
        assert engine is not None
        assert engine.env is not None
        assert engine.enable_cache is True
    
    def test_load_template_string(self, engine):
        """Test loading template from string."""
        template_string = "Hello {{ name }}!"
        template = engine.load_template_string(template_string)
        
        assert template is not None
        result = template.render(name="World")
        assert result == "Hello World!"
    
    def test_render_string(self, engine):
        """Test rendering template string."""
        template_string = "Hello {{ name }}!"
        result = engine.render_string(template_string, {"name": "World"})
        
        assert result == "Hello World!"
    
    def test_prepare_context(self, engine):
        """Test context preparation with defaults."""
        context = {"model_type": "random_forest"}
        prepared = engine._prepare_context(context)
        
        assert "model_type" in prepared
        assert "timestamp" in prepared
        assert "generator" in prepared
        assert "version" in prepared
        assert prepared["generator"] == "AI-Playground"
    
    def test_filter_indent(self, engine):
        """Test indent filter."""
        text = "line1\nline2\nline3"
        result = engine._filter_indent(text, spaces=2)
        
        assert result == "  line1\n  line2\n  line3"
    
    def test_filter_comment_python(self, engine):
        """Test comment filter with Python style."""
        text = "This is a comment\nAnother line"
        result = engine._filter_comment(text, style='python')
        
        assert result == "# This is a comment\n# Another line"
    
    def test_filter_snake_case(self, engine):
        """Test snake_case filter."""
        assert engine._filter_snake_case("HelloWorld") == "hello_world"
        assert engine._filter_snake_case("hello world") == "hello_world"
        assert engine._filter_snake_case("hello-world") == "hello_world"
    
    def test_filter_camel_case(self, engine):
        """Test camelCase filter."""
        assert engine._filter_camel_case("hello_world") == "helloWorld"
        assert engine._filter_camel_case("hello world") == "helloWorld"
        assert engine._filter_camel_case("hello-world") == "helloWorld"
    
    def test_filter_pascal_case(self, engine):
        """Test PascalCase filter."""
        assert engine._filter_pascal_case("hello_world") == "HelloWorld"
        assert engine._filter_pascal_case("hello world") == "HelloWorld"
        assert engine._filter_pascal_case("hello-world") == "HelloWorld"
    
    def test_filter_kebab_case(self, engine):
        """Test kebab-case filter."""
        assert engine._filter_kebab_case("HelloWorld") == "hello-world"
        assert engine._filter_kebab_case("hello_world") == "hello-world"
        assert engine._filter_kebab_case("hello world") == "hello-world"
    
    def test_filter_to_python_type(self, engine):
        """Test to_python_type filter."""
        assert engine._filter_to_python_type(True) == "bool"
        assert engine._filter_to_python_type(42) == "int"
        assert engine._filter_to_python_type(3.14) == "float"
        assert engine._filter_to_python_type("hello") == "str"
        assert engine._filter_to_python_type([1, 2, 3]) == "List"
        assert engine._filter_to_python_type({"key": "value"}) == "Dict"
    
    def test_filter_quote(self, engine):
        """Test quote filter."""
        assert engine._filter_quote("hello", style='double') == '"hello"'
        assert engine._filter_quote("hello", style='single') == "'hello'"
        assert engine._filter_quote("hello", style='triple') == '"""hello"""'
    
    def test_filter_escape(self, engine):
        """Test escape filter."""
        text = 'He said "Hello"'
        result = engine._filter_escape(text)
        assert '\\"' in result
    
    def test_add_custom_filter(self, engine):
        """Test adding custom filter."""
        def uppercase_filter(text):
            return text.upper()
        
        engine.add_filter('uppercase', uppercase_filter)
        
        template = engine.load_template_string("{{ name | uppercase }}")
        result = template.render(name="hello")
        
        assert result == "HELLO"
    
    def test_add_custom_test(self, engine):
        """Test adding custom test."""
        def is_even(n):
            return n % 2 == 0
        
        engine.add_test('even', is_even)
        
        template = engine.load_template_string("{% if number is even %}even{% else %}odd{% endif %}")
        result = template.render(number=4)
        
        assert result == "even"
    
    def test_add_global(self, engine):
        """Test adding global variable."""
        engine.add_global('pi', 3.14159)
        
        template = engine.load_template_string("Pi is {{ pi }}")
        result = template.render()
        
        assert "3.14159" in result
    
    def test_cache_functionality(self, engine, tmp_path):
        """Test template caching."""
        # Create a template file
        template_file = engine.template_dir / "test.j2"
        template_file.write_text("Hello {{ name }}!")
        
        # Load template (should cache)
        template1 = engine.load_template("test.j2")
        assert engine.get_cache_size() == 1
        
        # Load again (should use cache)
        template2 = engine.load_template("test.j2")
        assert template1 is template2
        assert engine.get_cache_size() == 1
        
        # Clear cache
        engine.clear_cache()
        assert engine.get_cache_size() == 0
    
    def test_template_not_found(self, engine):
        """Test handling of missing template."""
        with pytest.raises(TemplateNotFound):
            engine.load_template("nonexistent.j2")
    
    def test_template_exists(self, engine, tmp_path):
        """Test template existence check."""
        # Create a template file
        template_file = engine.template_dir / "exists.j2"
        template_file.write_text("Content")
        
        assert engine.template_exists("exists.j2") is True
        assert engine.template_exists("nonexistent.j2") is False
    
    def test_list_templates(self, engine, tmp_path):
        """Test listing templates."""
        # Create template files
        (engine.template_dir / "template1.j2").write_text("Content 1")
        (engine.template_dir / "template2.j2").write_text("Content 2")
        (engine.template_dir / "other.txt").write_text("Other")
        
        # List all templates
        templates = engine.list_templates()
        assert len(templates) >= 2
        assert "template1.j2" in templates
        assert "template2.j2" in templates
        
        # List with pattern
        j2_templates = engine.list_templates(pattern=r'\.j2$')
        assert all(t.endswith('.j2') for t in j2_templates)


class TestTemplateConfig:
    """Test TemplateConfig dataclass."""
    
    def test_creation(self):
        """Test creating template config."""
        config = TemplateConfig(
            name="test.j2",
            description="Test template",
            required_vars=["var1", "var2"],
            optional_vars=["var3"]
        )
        
        assert config.name == "test.j2"
        assert config.description == "Test template"
        assert len(config.required_vars) == 2
        assert len(config.optional_vars) == 1


class TestTemplateRegistry:
    """Test TemplateRegistry class."""
    
    @pytest.fixture
    def registry(self):
        """Create template registry."""
        return TemplateRegistry()
    
    def test_initialization(self, registry):
        """Test registry initialization."""
        assert registry is not None
        # Should have default templates registered
        templates = registry.list_templates()
        assert len(templates) > 0
    
    def test_register_template(self, registry):
        """Test registering template."""
        config = TemplateConfig(
            name="custom.j2",
            description="Custom template",
            required_vars=["var1"]
        )
        
        registry.register(config)
        
        retrieved = registry.get("custom.j2")
        assert retrieved is not None
        assert retrieved.name == "custom.j2"
        assert retrieved.description == "Custom template"
    
    def test_get_nonexistent_template(self, registry):
        """Test getting nonexistent template."""
        result = registry.get("nonexistent.j2")
        assert result is None
    
    def test_list_templates(self, registry):
        """Test listing templates."""
        templates = registry.list_templates()
        assert isinstance(templates, list)
        assert len(templates) > 0
    
    def test_validate_context_success(self, registry):
        """Test successful context validation."""
        config = TemplateConfig(
            name="test.j2",
            description="Test",
            required_vars=["var1", "var2"]
        )
        registry.register(config)
        
        context = {"var1": "value1", "var2": "value2", "var3": "value3"}
        
        # Should not raise
        assert registry.validate_context("test.j2", context) is True
    
    def test_validate_context_missing_vars(self, registry):
        """Test context validation with missing variables."""
        config = TemplateConfig(
            name="test.j2",
            description="Test",
            required_vars=["var1", "var2"]
        )
        registry.register(config)
        
        context = {"var1": "value1"}  # Missing var2
        
        with pytest.raises(ValueError, match="Missing required variables"):
            registry.validate_context("test.j2", context)
    
    def test_validate_context_no_config(self, registry):
        """Test validation with no config (should pass)."""
        context = {"var1": "value1"}
        
        # Should not raise for unregistered template
        assert registry.validate_context("nonexistent.j2", context) is True


class TestGlobalInstances:
    """Test global instance functions."""
    
    def test_get_template_engine(self):
        """Test getting global template engine."""
        engine1 = get_template_engine()
        engine2 = get_template_engine()
        
        assert engine1 is not None
        assert engine1 is engine2  # Should be same instance
    
    def test_get_template_registry(self):
        """Test getting global template registry."""
        registry1 = get_template_registry()
        registry2 = get_template_registry()
        
        assert registry1 is not None
        assert registry1 is registry2  # Should be same instance


class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_rendering(self, tmp_path):
        """Test complete template rendering workflow."""
        # Create engine
        template_dir = tmp_path / "templates"
        template_dir.mkdir()
        engine = TemplateEngine(template_dir=template_dir)
        
        # Create template file
        template_content = """
# {{ experiment_name }}
# Generated: {{ timestamp }}

def train_model():
    model_type = "{{ model_type }}"
    task_type = "{{ task_type }}"
    
    {% if task_type is classification %}
    # Classification task
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier()
    {% elif task_type is regression %}
    # Regression task
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor()
    {% endif %}
    
    return model
"""
        template_file = template_dir / "training.j2"
        template_file.write_text(template_content)
        
        # Render template
        context = {
            "experiment_name": "Test Experiment",
            "model_type": "random_forest_classifier",
            "task_type": "classification"
        }
        
        result = engine.render_template("training.j2", context, validate=False)
        
        # Verify output
        assert "Test Experiment" in result
        assert "random_forest_classifier" in result
        assert "Classification task" in result
        assert "RandomForestClassifier" in result
        assert "Regression task" not in result
    
    def test_template_with_filters(self, tmp_path):
        """Test template with custom filters."""
        template_dir = tmp_path / "templates"
        template_dir.mkdir()
        engine = TemplateEngine(template_dir=template_dir)
        
        template_content = """
class {{ class_name | pascal_case }}:
    def {{ method_name | snake_case }}(self):
        pass
"""
        template_file = template_dir / "class.j2"
        template_file.write_text(template_content)
        
        context = {
            "class_name": "my_model_trainer",
            "method_name": "TrainModel"
        }
        
        result = engine.render_template("class.j2", context, validate=False)
        
        assert "class MyModelTrainer:" in result
        assert "def train_model(self):" in result
