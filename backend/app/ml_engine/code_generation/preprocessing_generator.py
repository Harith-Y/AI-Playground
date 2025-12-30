"""
Preprocessing Code Generator

Generates Python code for data preprocessing steps from experiment configurations.
Uses Jinja2 templates to create production-ready preprocessing pipelines.

Based on: ML-TO-DO.md > ML-63
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
from app.ml_engine.code_generation.templates import get_template
from app.utils.logger import get_logger

logger = get_logger("preprocessing_generator")


class PreprocessingCodeGenerator:
    """
    Generator for preprocessing code.
    
    Converts preprocessing step configurations into executable Python code.
    
    Example:
        >>> generator = PreprocessingCodeGenerator()
        >>> config = {
        ...     'preprocessing_steps': [
        ...         {'type': 'missing_value_imputation', 'strategy': 'mean', ...}
        ...     ]
        ... }
        >>> code = generator.generate(config)
    """
    
    def __init__(self):
        """Initialize preprocessing code generator."""
        logger.debug("Initialized PreprocessingCodeGenerator")
    
    def generate(
        self,
        preprocessing_config: Dict[str, Any],
        include_imports: bool = True,
        include_data_loading: bool = True
    ) -> str:
        """
        Generate complete preprocessing code.
        
        Args:
            preprocessing_config: Configuration dictionary with preprocessing steps
            include_imports: Whether to include import statements
            include_data_loading: Whether to include data loading code
        
        Returns:
            Generated Python code as string
        """
        logger.info("Generating preprocessing code...")
        
        # Extract configuration
        steps = preprocessing_config.get('preprocessing_steps', [])
        dataset_info = preprocessing_config.get('dataset_info', {})
        
        # Prepare context for templates
        context = self._prepare_context(preprocessing_config)
        
        # Generate code sections
        code_sections = []
        
        if include_imports:
            imports_code = self._generate_imports(context)
            code_sections.append(imports_code)
        
        if include_data_loading:
            data_loading_code = self._generate_data_loading(context)
            code_sections.append(data_loading_code)
        
        # Generate preprocessing code
        preprocessing_code = self._generate_preprocessing(context)
        code_sections.append(preprocessing_code)
        
        # Combine all sections
        complete_code = '\n\n'.join(code_sections)
        
        logger.info(f"Generated preprocessing code with {len(steps)} steps")
        return complete_code
    
    def _prepare_context(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare template context from configuration.
        
        Args:
            config: Preprocessing configuration
        
        Returns:
            Context dictionary for templates
        """
        # Extract dataset info
        dataset_info = config.get('dataset_info', {})
        
        # Extract preprocessing steps and transform them
        raw_steps = config.get('preprocessing_steps', [])
        transformed_steps = self._transform_steps(raw_steps)
        
        context = {
            'timestamp': datetime.now().isoformat(),
            'experiment_name': config.get('experiment_name', 'ML Experiment'),
            'dataset_path': dataset_info.get('file_path', 'data.csv'),
            'file_format': dataset_info.get('file_format', 'csv'),
            'preprocessing_steps': transformed_steps,
            'random_state': config.get('random_state', 42),
            'task_type': config.get('task_type', 'classification'),
        }
        
        return context
    
    def _transform_steps(self, steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Transform preprocessing steps into template-friendly format.
        
        Args:
            steps: List of preprocessing step configurations
        
        Returns:
            Transformed steps for templates
        """
        transformed = []
        
        for step in steps:
            step_type = step.get('step_type') or step.get('type')
            
            if step_type == 'missing_value_imputation':
                transformed.append(self._transform_imputation_step(step))
            
            elif step_type == 'outlier_detection':
                transformed.append(self._transform_outlier_step(step))
            
            elif step_type == 'scaling':
                transformed.append(self._transform_scaling_step(step))
            
            elif step_type == 'encoding':
                transformed.append(self._transform_encoding_step(step))
            
            elif step_type == 'feature_selection':
                transformed.append(self._transform_feature_selection_step(step))
            
            else:
                logger.warning(f"Unknown step type: {step_type}")
        
        return transformed
    
    def _transform_imputation_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Transform missing value imputation step."""
        params = step.get('parameters', {})
        
        return {
            'type': 'missing_value_imputation',
            'name': step.get('name', 'Missing Value Imputation'),
            'description': f"Impute missing values using {params.get('strategy', 'mean')} strategy",
            'strategy': params.get('strategy', 'mean'),
            'columns': params.get('columns', []),
        }
    
    def _transform_outlier_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Transform outlier detection step."""
        params = step.get('parameters', {})
        
        return {
            'type': 'outlier_detection',
            'name': step.get('name', 'Outlier Detection'),
            'description': f"Detect outliers using {params.get('method', 'iqr')} method",
            'method': params.get('method', 'iqr'),
            'threshold': params.get('threshold', 1.5),
            'action': params.get('action', 'clip'),
            'columns': params.get('columns', []),
        }
    
    def _transform_scaling_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Transform scaling step."""
        params = step.get('parameters', {})
        
        return {
            'type': 'scaling',
            'name': step.get('name', 'Feature Scaling'),
            'description': f"Scale features using {params.get('scaler', 'standard')} scaler",
            'scaler': params.get('scaler', 'standard'),
            'columns': params.get('columns', []),
        }
    
    def _transform_encoding_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Transform encoding step."""
        params = step.get('parameters', {})
        
        return {
            'type': 'encoding',
            'name': step.get('name', 'Categorical Encoding'),
            'description': f"Encode categorical features using {params.get('encoder', 'onehot')} encoding",
            'encoder': params.get('encoder', 'onehot'),
            'columns': params.get('columns', []),
        }
    
    def _transform_feature_selection_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Transform feature selection step."""
        params = step.get('parameters', {})
        
        return {
            'type': 'feature_selection',
            'name': step.get('name', 'Feature Selection'),
            'description': f"Select features using {params.get('method', 'variance_threshold')} method",
            'method': params.get('method', 'variance_threshold'),
            'threshold': params.get('threshold', 0.0),
            'columns': params.get('columns', []),
            'target': params.get('target_column'),
        }
    
    def _generate_imports(self, context: Dict[str, Any]) -> str:
        """Generate import statements."""
        template = get_template('imports')
        
        # Add preprocessing-specific context
        context['preprocessing_steps'] = context.get('preprocessing_steps', [])
        context['include_evaluation'] = False  # Not needed for preprocessing only
        context['model_type'] = None
        context['model_imports'] = ''
        
        return template.render(context)
    
    def _generate_data_loading(self, context: Dict[str, Any]) -> str:
        """Generate data loading code."""
        template = get_template('data_loading')
        return template.render(context)
    
    def _generate_preprocessing(self, context: Dict[str, Any]) -> str:
        """Generate preprocessing pipeline code."""
        template = get_template('preprocessing')
        return template.render(context)
    
    def generate_preprocessing_function(
        self,
        preprocessing_config: Dict[str, Any],
        function_name: str = 'preprocess_data'
    ) -> str:
        """
        Generate standalone preprocessing function.
        
        Args:
            preprocessing_config: Configuration dictionary
            function_name: Name for the preprocessing function
        
        Returns:
            Python function code as string
        """
        context = self._prepare_context(preprocessing_config)
        context['function_name'] = function_name
        
        template = get_template('preprocessing')
        return template.render(context)
    
    def generate_preprocessing_class(
        self,
        preprocessing_config: Dict[str, Any],
        class_name: str = 'DataPreprocessor'
    ) -> str:
        """
        Generate preprocessing as a class (sklearn-style).
        
        Args:
            preprocessing_config: Configuration dictionary
            class_name: Name for the preprocessing class
        
        Returns:
            Python class code as string
        """
        context = self._prepare_context(preprocessing_config)
        steps = context.get('preprocessing_steps', [])
        
        # Generate class code
        code = f'''
class {class_name}:
    """
    Data preprocessing pipeline.
    
    Auto-generated from AI-Playground experiment.
    Generated: {context['timestamp']}
    """
    
    def __init__(self):
        """Initialize preprocessing pipeline."""
        self.fitted = False
        self.transformers = {{}}
    
    def fit(self, df: pd.DataFrame) -> '{class_name}':
        """
        Fit preprocessing transformers on data.
        
        Args:
            df: Input DataFrame
        
        Returns:
            Self (for method chaining)
        """
        df = df.copy()
        
'''
        
        # Add fit logic for each step
        for i, step in enumerate(steps, 1):
            code += f"        # Step {i}: {step['name']}\n"
            
            if step['type'] == 'missing_value_imputation':
                code += f'''        from sklearn.impute import SimpleImputer
        self.transformers['imputer_{i}'] = SimpleImputer(strategy='{step['strategy']}')
        self.transformers['imputer_{i}'].fit(df[{step['columns']}])
        
'''
            
            elif step['type'] == 'scaling':
                scaler_map = {
                    'standard': 'StandardScaler',
                    'minmax': 'MinMaxScaler',
                    'robust': 'RobustScaler'
                }
                scaler_class = scaler_map.get(step['scaler'], 'StandardScaler')
                code += f'''        from sklearn.preprocessing import {scaler_class}
        self.transformers['scaler_{i}'] = {scaler_class}()
        self.transformers['scaler_{i}'].fit(df[{step['columns']}])
        
'''
            
            elif step['type'] == 'encoding':
                if step['encoder'] == 'onehot':
                    code += f'''        from sklearn.preprocessing import OneHotEncoder
        self.transformers['encoder_{i}'] = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.transformers['encoder_{i}'].fit(df[{step['columns']}])
        
'''
                elif step['encoder'] == 'label':
                    code += f'''        from sklearn.preprocessing import LabelEncoder
        self.transformers['encoder_{i}'] = {{}}
        for col in {step['columns']}:
            encoder = LabelEncoder()
            encoder.fit(df[col].astype(str))
            self.transformers['encoder_{i}'][col] = encoder
        
'''
        
        code += '''        self.fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted transformers.
        
        Args:
            df: Input DataFrame
        
        Returns:
            Transformed DataFrame
        """
        if not self.fitted:
            raise RuntimeError("Preprocessor must be fitted before transform")
        
        df = df.copy()
        
'''
        
        # Add transform logic for each step
        for i, step in enumerate(steps, 1):
            code += f"        # Step {i}: {step['name']}\n"
            
            if step['type'] == 'missing_value_imputation':
                code += f'''        df[{step['columns']}] = self.transformers['imputer_{i}'].transform(df[{step['columns']}])
        
'''
            
            elif step['type'] == 'scaling':
                code += f'''        df[{step['columns']}] = self.transformers['scaler_{i}'].transform(df[{step['columns']}])
        
'''
            
            elif step['type'] == 'encoding':
                if step['encoder'] == 'onehot':
                    code += f'''        encoded = self.transformers['encoder_{i}'].transform(df[{step['columns']}])
        encoded_df = pd.DataFrame(
            encoded,
            columns=self.transformers['encoder_{i}'].get_feature_names_out({step['columns']}),
            index=df.index
        )
        df = df.drop(columns={step['columns']})
        df = pd.concat([df, encoded_df], axis=1)
        
'''
                elif step['encoder'] == 'label':
                    code += f'''        for col in {step['columns']}:
            df[col] = self.transformers['encoder_{i}'][col].transform(df[col].astype(str))
        
'''
        
        code += '''        return df
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform in one step.
        
        Args:
            df: Input DataFrame
        
        Returns:
            Transformed DataFrame
        """
        return self.fit(df).transform(df)
'''
        
        return code


def generate_preprocessing_code(
    preprocessing_config: Dict[str, Any],
    output_format: str = 'script',
    **kwargs
) -> str:
    """
    Convenience function to generate preprocessing code.
    
    Args:
        preprocessing_config: Configuration dictionary
        output_format: 'script', 'function', or 'class'
        **kwargs: Additional arguments for generator
    
    Returns:
        Generated Python code
    
    Example:
        >>> config = {
        ...     'preprocessing_steps': [...],
        ...     'dataset_info': {...}
        ... }
        >>> code = generate_preprocessing_code(config, output_format='script')
    """
    generator = PreprocessingCodeGenerator()
    
    if output_format == 'script':
        return generator.generate(preprocessing_config, **kwargs)
    elif output_format == 'function':
        return generator.generate_preprocessing_function(preprocessing_config, **kwargs)
    elif output_format == 'class':
        return generator.generate_preprocessing_class(preprocessing_config, **kwargs)
    else:
        raise ValueError(f"Invalid output_format: {output_format}. Choose 'script', 'function', or 'class'")
