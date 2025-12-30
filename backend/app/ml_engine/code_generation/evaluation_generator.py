"""
Evaluation Code Generator

Generates Python code for model evaluation from experiment configurations.
Uses templates to create production-ready evaluation pipelines.

Based on: ML-TO-DO.md > ML-65
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
from app.utils.logger import get_logger

logger = get_logger("evaluation_generator")


class EvaluationCodeGenerator:
    """
    Generator for model evaluation code.
    
    Converts evaluation configurations into executable Python code.
    
    Example:
        >>> generator = EvaluationCodeGenerator()
        >>> config = {
        ...     'task_type': 'classification',
        ...     'metrics': ['accuracy', 'precision', 'recall', 'f1']
        ... }
        >>> code = generator.generate(config)
    """
    
    def __init__(self):
        """Initialize evaluation code generator."""
        logger.debug("Initialized EvaluationCodeGenerator")
    
    def generate(
        self,
        evaluation_config: Dict[str, Any],
        output_format: str = 'script',
        include_imports: bool = True
    ) -> str:
        """
        Generate model evaluation code.
        
        Args:
            evaluation_config: Configuration dictionary with evaluation settings
            output_format: Output format ('script', 'function', 'module')
            include_imports: Whether to include import statements
        
        Returns:
            Generated Python code as string
        """
        logger.info(f"Generating evaluation code in '{output_format}' format...")
        
        # Prepare context
        context = self._prepare_context(evaluation_config)
        
        # Generate based on format
        if output_format == 'script':
            code = self._generate_script(context, include_imports)
        elif output_format == 'function':
            code = self._generate_function(context)
        elif output_format == 'module':
            code = self._generate_module(context, include_imports)
        else:
            raise ValueError(f"Unknown output format: {output_format}")
        
        logger.info(f"Generated evaluation code for task: {context['task_type']}")
        return code
    
    def _prepare_context(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare template context from configuration.
        
        Args:
            config: Evaluation configuration
        
        Returns:
            Context dictionary for templates
        """
        task_type = config.get('task_type', 'classification')
        metrics = config.get('metrics', [])
        
        # Default metrics based on task type
        if not metrics:
            if task_type == 'classification':
                metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
            elif task_type == 'regression':
                metrics = ['mae', 'mse', 'rmse', 'r2', 'mape']
            elif task_type == 'clustering':
                metrics = ['silhouette', 'calinski_harabasz', 'davies_bouldin']
        
        context = {
            'timestamp': datetime.now().isoformat(),
            'experiment_name': config.get('experiment_name', 'ML Experiment'),
            'task_type': task_type,
            'metrics': metrics,
            'random_state': config.get('random_state', 42),
            'include_confusion_matrix': config.get('include_confusion_matrix', task_type == 'classification'),
            'include_roc_curve': config.get('include_roc_curve', task_type == 'classification'),
            'include_pr_curve': config.get('include_pr_curve', task_type == 'classification'),
            'include_feature_importance': config.get('include_feature_importance', False),
            'include_residual_analysis': config.get('include_residual_analysis', task_type == 'regression'),
            'save_results': config.get('save_results', True),
            'results_path': config.get('results_path', 'evaluation_results.json'),
        }
        
        return context
    
    def _generate_script(self, context: Dict[str, Any], include_imports: bool) -> str:
        """Generate complete evaluation script."""
        sections = []
        
        # Imports
        if include_imports:
            imports = self._generate_imports(context)
            sections.append(imports)
        
        # Evaluation function
        evaluation = self._generate_evaluation_code(context)
        sections.append(evaluation)
        
        # Visualization functions
        if context['include_confusion_matrix'] or context['include_roc_curve']:
            viz = self._generate_visualization_code(context)
            sections.append(viz)
        
        # Save results function
        if context['save_results']:
            save = self._generate_save_results(context)
            sections.append(save)
        
        # Main execution
        main = self._generate_main_execution(context)
        sections.append(main)
        
        return '\n\n'.join(sections)
    
    def _generate_function(self, context: Dict[str, Any]) -> str:
        """Generate evaluation function."""
        return self._generate_evaluation_code(context)
    
    def _generate_module(self, context: Dict[str, Any], include_imports: bool) -> str:
        """Generate modular evaluation code."""
        sections = []
        
        # Module docstring
        docstring = f'''"""
Evaluation Module - {context['experiment_name']}

Auto-generated by AI-Playground
Generated: {context['timestamp']}
Task Type: {context['task_type']}

This module can be imported and used in other scripts:
    from evaluate import evaluate_model, save_results
"""
'''
        sections.append(docstring)
        
        # Imports
        if include_imports:
            imports = self._generate_imports(context)
            sections.append(imports)
        
        # Configuration
        config = f"""
# Configuration
RANDOM_STATE = {context['random_state']}
RESULTS_PATH = '{context['results_path']}'
"""
        sections.append(config)
        
        # Evaluation function
        evaluation = self._generate_evaluation_code(context)
        sections.append(evaluation)
        
        # Additional functions
        if context['include_confusion_matrix'] or context['include_roc_curve']:
            viz = self._generate_visualization_code(context)
            sections.append(viz)
        
        if context['save_results']:
            save = self._generate_save_results(context)
            sections.append(save)
        
        # Main block
        main = '''
if __name__ == '__main__':
    """
    Example usage when run as a script.
    """
    print("=" * 80)
    print("Evaluation Module - {experiment_name}")
    print("=" * 80)
    
    # Example: Load your model and data here
    # model = load_model('model.pkl')
    # X_test, y_test = load_test_data()
    
    # Evaluate model
    # results = evaluate_model(model, X_test, y_test)
    
    # Save results
    # save_results(results, RESULTS_PATH)
    
    print("\\nTo use this module in another script:")
    print("  from evaluate import evaluate_model, save_results")
'''.format(experiment_name=context['experiment_name'])
        sections.append(main)
        
        return '\n\n'.join(sections)
    
    def _generate_imports(self, context: Dict[str, Any]) -> str:
        """Generate import statements."""
        task_type = context['task_type']
        
        imports = f"""# Auto-generated by AI-Playground
# Generated: {context['timestamp']}
# Experiment: {context['experiment_name']}

import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
"""
        
        # Task-specific imports
        if task_type == 'classification':
            imports += """
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)
"""
        elif task_type == 'regression':
            imports += """
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
    explained_variance_score
)
"""
        elif task_type == 'clustering':
            imports += """
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)
"""
        
        # Optional imports
        if context['include_confusion_matrix'] or context['include_roc_curve']:
            imports += """
import matplotlib.pyplot as plt
import seaborn as sns
"""
        
        imports += """
import warnings
warnings.filterwarnings('ignore')
"""
        
        return imports
    
    def _generate_evaluation_code(self, context: Dict[str, Any]) -> str:
        """Generate evaluation function code."""
        task_type = context['task_type']
        
        if task_type == 'classification':
            return self._generate_classification_evaluation(context)
        elif task_type == 'regression':
            return self._generate_regression_evaluation(context)
        elif task_type == 'clustering':
            return self._generate_clustering_evaluation(context)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    def _generate_classification_evaluation(self, context: Dict[str, Any]) -> str:
        """Generate classification evaluation code."""
        metrics = context['metrics']
        
        code = """# ============================================================================
# CLASSIFICATION EVALUATION
# ============================================================================

def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_pred: Optional[np.ndarray] = None,
    y_proba: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    \"\"\"
    Evaluate classification model performance.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: True labels
        y_pred: Predicted labels (optional, will be computed if not provided)
        y_proba: Predicted probabilities (optional, for AUC/ROC)
    
    Returns:
        Dictionary containing evaluation metrics
    \"\"\"
    print("Evaluating classification model...")
    print(f"Test samples: {len(y_test)}")
    
    # Get predictions if not provided
    if y_pred is None:
        y_pred = model.predict(X_test)
    
    # Get probabilities if available and not provided
    if y_proba is None and hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)
    
    results = {
        'task_type': 'classification',
        'n_samples': len(y_test),
        'n_classes': len(np.unique(y_test)),
        'metrics': {}
    }
    
"""
        
        # Add metric calculations
        if 'accuracy' in metrics:
            code += """    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    results['metrics']['accuracy'] = float(accuracy)
    print(f"Accuracy: {accuracy:.4f}")
    
"""
        
        if 'precision' in metrics:
            code += """    # Precision
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    results['metrics']['precision'] = float(precision)
    print(f"Precision: {precision:.4f}")
    
"""
        
        if 'recall' in metrics:
            code += """    # Recall
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    results['metrics']['recall'] = float(recall)
    print(f"Recall: {recall:.4f}")
    
"""
        
        if 'f1' in metrics:
            code += """    # F1 Score
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    results['metrics']['f1_score'] = float(f1)
    print(f"F1 Score: {f1:.4f}")
    
"""
        
        if 'roc_auc' in metrics:
            code += """    # ROC AUC (if probabilities available)
    if y_proba is not None:
        try:
            if results['n_classes'] == 2:
                # Binary classification
                roc_auc = roc_auc_score(y_test, y_proba[:, 1])
            else:
                # Multi-class classification
                roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
            results['metrics']['roc_auc'] = float(roc_auc)
            print(f"ROC AUC: {roc_auc:.4f}")
        except Exception as e:
            print(f"Could not compute ROC AUC: {e}")
    
"""
        
        # Confusion matrix
        if context['include_confusion_matrix']:
            code += """    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    results['confusion_matrix'] = cm.tolist()
    print(f"\\nConfusion Matrix:\\n{cm}")
    
"""
        
        # Classification report
        code += """    # Classification Report
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    results['classification_report'] = report
    
    print(f"\\nEvaluation complete!")
    return results
"""
        
        return code
    
    def _generate_regression_evaluation(self, context: Dict[str, Any]) -> str:
        """Generate regression evaluation code."""
        metrics = context['metrics']
        
        code = """# ============================================================================
# REGRESSION EVALUATION
# ============================================================================

def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_pred: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    \"\"\"
    Evaluate regression model performance.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: True values
        y_pred: Predicted values (optional, will be computed if not provided)
    
    Returns:
        Dictionary containing evaluation metrics
    \"\"\"
    print("Evaluating regression model...")
    print(f"Test samples: {len(y_test)}")
    
    # Get predictions if not provided
    if y_pred is None:
        y_pred = model.predict(X_test)
    
    results = {
        'task_type': 'regression',
        'n_samples': len(y_test),
        'metrics': {}
    }
    
"""
        
        # Add metric calculations
        if 'mae' in metrics:
            code += """    # Mean Absolute Error
    mae = mean_absolute_error(y_test, y_pred)
    results['metrics']['mae'] = float(mae)
    print(f"MAE: {mae:.4f}")
    
"""
        
        if 'mse' in metrics:
            code += """    # Mean Squared Error
    mse = mean_squared_error(y_test, y_pred)
    results['metrics']['mse'] = float(mse)
    print(f"MSE: {mse:.4f}")
    
"""
        
        if 'rmse' in metrics:
            code += """    # Root Mean Squared Error
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    results['metrics']['rmse'] = float(rmse)
    print(f"RMSE: {rmse:.4f}")
    
"""
        
        if 'r2' in metrics:
            code += """    # R² Score
    r2 = r2_score(y_test, y_pred)
    results['metrics']['r2'] = float(r2)
    print(f"R² Score: {r2:.4f}")
    
"""
        
        if 'mape' in metrics:
            code += """    # Mean Absolute Percentage Error
    try:
        mape = mean_absolute_percentage_error(y_test, y_pred)
        results['metrics']['mape'] = float(mape)
        print(f"MAPE: {mape:.4f}")
    except Exception as e:
        print(f"Could not compute MAPE: {e}")
    
"""
        
        # Residual analysis
        if context['include_residual_analysis']:
            code += """    # Residual Analysis
    residuals = y_test - y_pred
    results['residuals'] = {
        'mean': float(np.mean(residuals)),
        'std': float(np.std(residuals)),
        'min': float(np.min(residuals)),
        'max': float(np.max(residuals))
    }
    print(f"\\nResidual Statistics:")
    print(f"  Mean: {results['residuals']['mean']:.4f}")
    print(f"  Std: {results['residuals']['std']:.4f}")
    
"""
        
        code += """    print(f"\\nEvaluation complete!")
    return results
"""
        
        return code
    
    def _generate_clustering_evaluation(self, context: Dict[str, Any]) -> str:
        """Generate clustering evaluation code."""
        metrics = context['metrics']
        
        code = """# ============================================================================
# CLUSTERING EVALUATION
# ============================================================================

def evaluate_model(
    model,
    X_test: np.ndarray,
    labels: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    \"\"\"
    Evaluate clustering model performance.
    
    Args:
        model: Trained clustering model
        X_test: Test features
        labels: Cluster labels (optional, will be computed if not provided)
    
    Returns:
        Dictionary containing evaluation metrics
    \"\"\"
    print("Evaluating clustering model...")
    print(f"Test samples: {len(X_test)}")
    
    # Get cluster labels if not provided
    if labels is None:
        if hasattr(model, 'labels_'):
            labels = model.labels_
        elif hasattr(model, 'predict'):
            labels = model.predict(X_test)
        else:
            raise ValueError("Could not get cluster labels from model")
    
    results = {
        'task_type': 'clustering',
        'n_samples': len(X_test),
        'n_clusters': len(np.unique(labels[labels != -1])),  # Exclude noise points
        'metrics': {}
    }
    
"""
        
        # Add metric calculations
        if 'silhouette' in metrics:
            code += """    # Silhouette Score
    try:
        # Filter out noise points (label -1) for DBSCAN
        mask = labels != -1
        if np.sum(mask) > 1 and len(np.unique(labels[mask])) > 1:
            silhouette = silhouette_score(X_test[mask], labels[mask])
            results['metrics']['silhouette_score'] = float(silhouette)
            print(f"Silhouette Score: {silhouette:.4f}")
        else:
            print("Not enough valid clusters for silhouette score")
    except Exception as e:
        print(f"Could not compute silhouette score: {e}")
    
"""
        
        if 'calinski_harabasz' in metrics:
            code += """    # Calinski-Harabasz Score
    try:
        mask = labels != -1
        if np.sum(mask) > 1 and len(np.unique(labels[mask])) > 1:
            ch_score = calinski_harabasz_score(X_test[mask], labels[mask])
            results['metrics']['calinski_harabasz_score'] = float(ch_score)
            print(f"Calinski-Harabasz Score: {ch_score:.4f}")
    except Exception as e:
        print(f"Could not compute Calinski-Harabasz score: {e}")
    
"""
        
        if 'davies_bouldin' in metrics:
            code += """    # Davies-Bouldin Score
    try:
        mask = labels != -1
        if np.sum(mask) > 1 and len(np.unique(labels[mask])) > 1:
            db_score = davies_bouldin_score(X_test[mask], labels[mask])
            results['metrics']['davies_bouldin_score'] = float(db_score)
            print(f"Davies-Bouldin Score: {db_score:.4f}")
    except Exception as e:
        print(f"Could not compute Davies-Bouldin score: {e}")
    
"""
        
        # Cluster sizes
        code += """    # Cluster Sizes
    unique, counts = np.unique(labels, return_counts=True)
    cluster_sizes = dict(zip(unique.tolist(), counts.tolist()))
    results['cluster_sizes'] = cluster_sizes
    print(f"\\nCluster Sizes: {cluster_sizes}")
    
    print(f"\\nEvaluation complete!")
    return results
"""
        
        return code
    
    def _generate_visualization_code(self, context: Dict[str, Any]) -> str:
        """Generate visualization functions."""
        code = """# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================
"""
        
        if context['include_confusion_matrix']:
            code += """
def plot_confusion_matrix(cm: np.ndarray, save_path: Optional[str] = None):
    \"\"\"
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix
        save_path: Path to save plot (optional)
    \"\"\"
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()
    
    plt.close()
"""
        
        if context['include_roc_curve']:
            code += """
def plot_roc_curve(y_test: np.ndarray, y_proba: np.ndarray, save_path: Optional[str] = None):
    \"\"\"
    Plot ROC curve for binary classification.
    
    Args:
        y_test: True labels
        y_proba: Predicted probabilities
        save_path: Path to save plot (optional)
    \"\"\"
    fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
    roc_auc = roc_auc_score(y_test, y_proba[:, 1])
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to {save_path}")
    else:
        plt.show()
    
    plt.close()
"""
        
        return code
    
    def _generate_save_results(self, context: Dict[str, Any]) -> str:
        """Generate save results function."""
        code = f"""# ============================================================================
# SAVE RESULTS
# ============================================================================

def save_results(results: Dict[str, Any], filepath: str = '{context['results_path']}'):
    \"\"\"
    Save evaluation results to JSON file.
    
    Args:
        results: Evaluation results dictionary
        filepath: Path to save results
    \"\"\"
    # Ensure directory exists
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    # Save to JSON
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\\nResults saved to {{filepath}}")
"""
        return code
    
    def _generate_main_execution(self, context: Dict[str, Any]) -> str:
        """Generate main execution block."""
        code = f"""# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    \"\"\"
    Example usage of evaluation functions.
    \"\"\"
    print("=" * 80)
    print("Model Evaluation - {context['experiment_name']}")
    print("=" * 80)
    
    # Example: Load your model and test data
    # import pickle
    # with open('model.pkl', 'rb') as f:
    #     model = pickle.load(f)
    # 
    # X_test = ...  # Load test features
    # y_test = ...  # Load test labels
    
    # Evaluate model
    # results = evaluate_model(model, X_test, y_test)
    
    # Save results
    # save_results(results, '{context['results_path']}')
    
"""
        
        if context['include_confusion_matrix']:
            code += """    # Plot confusion matrix
    # if 'confusion_matrix' in results:
    #     plot_confusion_matrix(np.array(results['confusion_matrix']), 'confusion_matrix.png')
    
"""
        
        if context['include_roc_curve']:
            code += """    # Plot ROC curve (for binary classification)
    # if hasattr(model, 'predict_proba'):
    #     y_proba = model.predict_proba(X_test)
    #     plot_roc_curve(y_test, y_proba, 'roc_curve.png')
    
"""
        
        code += """    print("\\nEvaluation script ready!")
    print("Uncomment the example code above to run evaluation.")
"""
        
        return code


def generate_evaluation_code(
    evaluation_config: Dict[str, Any],
    output_format: str = 'script',
    include_imports: bool = True
) -> str:
    """
    Convenience function to generate evaluation code.
    
    Args:
        evaluation_config: Configuration dictionary with evaluation settings
        output_format: Output format ('script', 'function', 'module')
        include_imports: Whether to include import statements
    
    Returns:
        Generated Python code as string
    
    Example:
        >>> config = {
        ...     'task_type': 'classification',
        ...     'metrics': ['accuracy', 'precision', 'recall', 'f1'],
        ...     'include_confusion_matrix': True
        ... }
        >>> code = generate_evaluation_code(config)
    """
    generator = EvaluationCodeGenerator()
    return generator.generate(evaluation_config, output_format, include_imports)
