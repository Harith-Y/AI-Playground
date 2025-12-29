"""
Celery tasks for preprocessing operations

This module contains asynchronous tasks for executing preprocessing pipelines.
Long-running preprocessing operations are handled here to prevent API timeouts.
"""

import uuid
import time
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import numpy as np
from celery import Task
from sqlalchemy.orm import Session

from app.celery_app import celery_app
from app.db.session import SessionLocal
from app.models.preprocessing_step import PreprocessingStep
from app.models.dataset import Dataset
from app.ml_engine.preprocessing.imputer import MeanImputer, MedianImputer
from app.ml_engine.preprocessing.scaler import StandardScaler, MinMaxScaler, RobustScaler
from app.ml_engine.preprocessing.cleaner import IQROutlierDetector, ZScoreOutlierDetector
from app.core.logging_config import get_preprocessing_logger, log_preprocessing_metrics


class PreprocessingTask(Task):
    """Base task for preprocessing operations with progress tracking and logging"""

    def on_success(self, retval, task_id, args, kwargs):
        """Called when task completes successfully"""
        logger = get_preprocessing_logger(task_id=task_id)
        logger.info(
            f"Preprocessing task completed successfully",
            extra={
                'event': 'task_success',
                'duration_seconds': retval.get('total_execution_time', 0),
                'steps_applied': retval.get('steps_applied', 0),
                'rows_processed': retval.get('transformed_shape', [0])[0]
            }
        )

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Called when task fails"""
        logger = get_preprocessing_logger(task_id=task_id)
        logger.error(
            f"Preprocessing task failed: {exc}",
            extra={
                'event': 'task_failure',
                'error_type': type(exc).__name__,
                'error_message': str(exc),
                'traceback': str(einfo)
            },
            exc_info=True
        )


@celery_app.task(
    base=PreprocessingTask,
    bind=True,
    name="app.tasks.preprocessing_tasks.apply_preprocessing_pipeline"
)
def apply_preprocessing_pipeline(
    self,
    dataset_id: str,
    user_id: str,
    save_output: bool = True,
    output_name: str = None
) -> Dict[str, Any]:
    """
    Apply preprocessing pipeline to a dataset asynchronously.

    Args:
        self: Task instance (bound)
        dataset_id: UUID of the dataset to preprocess
        user_id: UUID of the user owning the dataset
        save_output: Whether to save the transformed dataset
        output_name: Name for the output dataset

    Returns:
        Dictionary with processing results
    """
    # Setup logging with context
    logger = get_preprocessing_logger(
        dataset_id=dataset_id,
        task_id=self.request.id,
        user_id=user_id
    )

    logger.info(
        f"Starting preprocessing pipeline",
        extra={
            'event': 'pipeline_start',
            'save_output': save_output,
            'output_name': output_name
        }
    )

    task_start_time = time.time()
    db: Session = SessionLocal()

    try:
        # Update task state to STARTED
        self.update_state(
            state='STARTED',
            meta={'status': 'Loading dataset...', 'progress': 0}
        )

        logger.debug("Task state updated to STARTED")

        # Verify dataset exists and belongs to user
        dataset = db.query(Dataset).filter(
            Dataset.id == dataset_id,
            Dataset.user_id == user_id
        ).first()

        if not dataset:
            logger.error(
                f"Dataset not found or access denied",
                extra={'event': 'dataset_not_found'}
            )
            raise ValueError(f"Dataset {dataset_id} not found or access denied")

        logger.info(
            f"Dataset found: {dataset.name}",
            extra={
                'event': 'dataset_loaded',
                'dataset_name': dataset.name,
                'file_path': dataset.file_path
            }
        )

        # Get all preprocessing steps
        steps = db.query(PreprocessingStep).filter(
            PreprocessingStep.dataset_id == dataset_id
        ).order_by(PreprocessingStep.order).all()

        if not steps:
            logger.error(
                "No preprocessing steps configured",
                extra={'event': 'no_steps_configured'}
            )
            raise ValueError("No preprocessing steps configured for this dataset")

        total_steps = len(steps)
        logger.info(
            f"Found {total_steps} preprocessing steps to apply",
            extra={
                'event': 'steps_loaded',
                'total_steps': total_steps,
                'step_types': [s.step_type for s in steps]
            }
        )

        # Load dataset
        self.update_state(
            state='PROGRESS',
            meta={'status': f'Loading dataset: {dataset.name}', 'progress': 5}
        )

        logger.debug(f"Loading data from {dataset.file_path}")
        load_start = time.time()
        df = pd.read_csv(dataset.file_path)
        load_time = time.time() - load_start

        original_shape = list(df.shape)

        logger.info(
            f"Dataset loaded successfully",
            extra={
                'event': 'data_loaded',
                'shape': original_shape,
                'rows': df.shape[0],
                'columns': df.shape[1],
                'load_time_seconds': round(load_time, 3),
                'memory_mb': round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2)
            }
        )

        # Initialize statistics
        stats = {
            "rows_before": df.shape[0],
            "columns_before": df.shape[1],
            "missing_values_filled": 0,
            "outliers_removed": 0,
            "features_scaled": 0,
            "features_encoded": 0,
            "features_removed": 0,
        }

        # Apply each preprocessing step
        for idx, step in enumerate(steps, 1):
            progress = 10 + int((idx / total_steps) * 70)  # 10-80%

            logger.info(
                f"Starting step {idx}/{total_steps}: {step.step_type}",
                extra={
                    'event': 'step_start',
                    'step_number': idx,
                    'step_type': step.step_type,
                    'step_parameters': step.parameters,
                    'column_name': step.column_name,
                    'progress': progress
                }
            )

            self.update_state(
                state='PROGRESS',
                meta={
                    'status': f'Applying step {idx}/{total_steps}: {step.step_type}',
                    'progress': progress,
                    'current_step': step.step_type
                }
            )

            step_start_time = time.time()
            rows_before = df.shape[0]
            cols_before = df.shape[1]

            df, step_stats = _apply_single_step(df, step, logger)

            step_execution_time = time.time() - step_start_time
            rows_after = df.shape[0]
            cols_after = df.shape[1]

            # Log step metrics
            log_preprocessing_metrics(
                logger=logger.logger,  # Get underlying logger from adapter
                dataset_id=dataset_id,
                task_id=self.request.id,
                step_type=step.step_type,
                execution_time=step_execution_time,
                rows_before=rows_before,
                rows_after=rows_after,
                cols_before=cols_before,
                cols_after=cols_after,
                **step_stats
            )

            logger.info(
                f"Step {idx}/{total_steps} completed: {step.step_type}",
                extra={
                    'event': 'step_complete',
                    'step_number': idx,
                    'step_type': step.step_type,
                    'execution_time_seconds': round(step_execution_time, 3),
                    'shape_before': [rows_before, cols_before],
                    'shape_after': [rows_after, cols_after]
                }
            )

            # Update statistics
            for key, value in step_stats.items():
                if key in stats:
                    stats[key] += value

        transformed_shape = list(df.shape)
        stats["rows_after"] = df.shape[0]
        stats["columns_after"] = df.shape[1]

        logger.info(
            f"All steps applied successfully",
            extra={
                'event': 'all_steps_complete',
                'total_steps': total_steps,
                'shape_before': original_shape,
                'shape_after': transformed_shape,
                'statistics': stats
            }
        )

        # Save transformed dataset
        self.update_state(
            state='PROGRESS',
            meta={'status': 'Saving transformed dataset...', 'progress': 85}
        )

        output_dataset_id = None
        if save_output:
            logger.debug(f"Saving transformed dataset")
            save_start = time.time()

            output_name = output_name or f"{dataset.name}_preprocessed"
            output_path = _save_transformed_dataset(df, output_name, dataset.file_path, logger)

            save_time = time.time() - save_start

            # Create new dataset record
            new_dataset = Dataset(
                id=uuid.uuid4(),
                user_id=user_id,
                name=output_name,
                file_path=output_path,
                rows=df.shape[0],
                cols=df.shape[1],
                dtypes={col: str(df[col].dtype) for col in df.columns},
                missing_values={col: int(df[col].isna().sum()) for col in df.columns}
            )
            db.add(new_dataset)
            db.commit()
            db.refresh(new_dataset)
            output_dataset_id = str(new_dataset.id)

            logger.info(
                f"Transformed dataset saved",
                extra={
                    'event': 'dataset_saved',
                    'output_name': output_name,
                    'output_path': output_path,
                    'output_dataset_id': output_dataset_id,
                    'save_time_seconds': round(save_time, 3),
                    'file_size_mb': round(Path(output_path).stat().st_size / 1024 / 1024, 2)
                }
            )

        # Generate preview
        self.update_state(
            state='PROGRESS',
            meta={'status': 'Generating preview...', 'progress': 95}
        )

        preview = df.head(5).to_dict('records')

        total_execution_time = time.time() - task_start_time

        # Task complete
        result = {
            'success': True,
            'message': f'Successfully applied {total_steps} preprocessing steps',
            'steps_applied': total_steps,
            'original_shape': original_shape,
            'transformed_shape': transformed_shape,
            'output_dataset_id': output_dataset_id,
            'preview': preview,
            'statistics': stats,
            'progress': 100,
            'total_execution_time': round(total_execution_time, 3)
        }

        logger.info(
            f"Preprocessing pipeline completed successfully",
            extra={
                'event': 'pipeline_complete',
                'total_execution_time_seconds': round(total_execution_time, 3),
                'total_steps': total_steps,
                'output_dataset_id': output_dataset_id,
                **stats
            }
        )

        return result

    except Exception as e:
        # Task failed
        logger.error(
            f"Preprocessing pipeline failed: {e}",
            extra={
                'event': 'pipeline_failure',
                'error_type': type(e).__name__,
                'error_message': str(e)
            },
            exc_info=True
        )

        self.update_state(
            state='FAILURE',
            meta={'error': str(e), 'progress': 0}
        )
        raise

    finally:
        db.close()
        logger.debug("Database session closed")


def _apply_single_step(df: pd.DataFrame, step: PreprocessingStep, logger) -> tuple:
    """
    Apply a single preprocessing step to the dataframe.

    Args:
        df: Input dataframe
        step: Preprocessing step configuration
        logger: Logger instance for step execution logging

    Returns:
        Tuple of (transformed_df, statistics_dict)
    """
    step_type = step.step_type
    params = step.parameters or {}
    column_name = step.column_name
    stats = {}

    logger.debug(
        f"Applying {step_type}",
        extra={
            'step_type': step_type,
            'parameters': params,
            'column': column_name
        }
    )

    if step_type == "missing_value_imputation":
        strategy = params.get("strategy", "mean")

        if strategy == "mean":
            imputer = MeanImputer(columns=[column_name] if column_name else None)
        elif strategy == "median":
            imputer = MedianImputer(columns=[column_name] if column_name else None)
        elif strategy in ["mode", "constant", "ffill", "bfill"]:
            if column_name:
                missing_count = df[column_name].isna().sum()
                if strategy == "mode":
                    df[column_name] = df[column_name].fillna(
                        df[column_name].mode()[0] if not df[column_name].mode().empty else 0
                    )
                elif strategy == "constant":
                    df[column_name] = df[column_name].fillna(params.get("fill_value", 0))
                elif strategy == "ffill":
                    df[column_name] = df[column_name].fillna(method='ffill')
                elif strategy == "bfill":
                    df[column_name] = df[column_name].fillna(method='bfill')
                stats["missing_values_filled"] = int(missing_count)
            return df, stats
        else:
            raise ValueError(f"Unknown imputation strategy: {strategy}")

        missing_before = df[column_name].isna().sum() if column_name else df.isna().sum().sum()
        df = imputer.fit_transform(df)
        stats["missing_values_filled"] = int(missing_before)

    elif step_type == "scaling":
        method = params.get("method", "standard")

        if method == "standard":
            scaler = StandardScaler(columns=[column_name] if column_name else None)
        elif method == "minmax":
            scaler = MinMaxScaler(
                columns=[column_name] if column_name else None,
                feature_range=(
                    params.get("feature_range_min", 0),
                    params.get("feature_range_max", 1)
                )
            )
        elif method == "robust":
            scaler = RobustScaler(columns=[column_name] if column_name else None)
        else:
            raise ValueError(f"Unknown scaling method: {method}")

        df = scaler.fit_transform(df)
        stats["features_scaled"] = 1 if column_name else len(df.select_dtypes(include=[np.number]).columns)

    elif step_type == "outlier_detection":
        method = params.get("method", "iqr")
        threshold = params.get("threshold", 1.5)
        action = params.get("action", "flag")

        if method == "iqr":
            detector = IQROutlierDetector(
                columns=[column_name] if column_name else None,
                threshold=threshold
            )
        elif method == "zscore":
            detector = ZScoreOutlierDetector(
                columns=[column_name] if column_name else None,
                threshold=threshold
            )
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")

        rows_before = len(df)
        outlier_mask = detector.fit_transform(df)

        if action == "remove":
            df = df[~outlier_mask.any(axis=1)]
            stats["outliers_removed"] = rows_before - len(df)
        elif action == "cap":
            for col in outlier_mask.columns:
                if outlier_mask[col].any():
                    q1 = df[col].quantile(0.25)
                    q3 = df[col].quantile(0.75)
                    iqr = q3 - q1
                    lower = q1 - threshold * iqr
                    upper = q3 + threshold * iqr
                    df[col] = df[col].clip(lower, upper)

    elif step_type == "encoding":
        method = params.get("method", "onehot")
        if column_name and method == "label":
            df[column_name] = pd.factorize(df[column_name])[0]
            stats["features_encoded"] = 1
        elif column_name and method == "onehot":
            dummies = pd.get_dummies(df[column_name], prefix=column_name)
            df = pd.concat([df.drop(column_name, axis=1), dummies], axis=1)
            stats["features_encoded"] = len(dummies.columns)

    elif step_type == "feature_selection":
        method = params.get("method", "variance_threshold")
        threshold = params.get("threshold", 0.01)

        if method == "variance_threshold":
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            variances = df[numeric_cols].var()
            low_variance_cols = variances[variances < threshold].index.tolist()
            df = df.drop(columns=low_variance_cols)
            stats["features_removed"] = len(low_variance_cols)

    elif step_type == "transformation":
        method = params.get("method", "log")

        if column_name and method == "log":
            df[column_name] = np.log1p(df[column_name])
        elif column_name and method == "sqrt":
            df[column_name] = np.sqrt(df[column_name])

    return df, stats


def _save_transformed_dataset(df: pd.DataFrame, name: str, original_path: str, logger) -> str:
    """
    Save the transformed dataset to disk.

    Args:
        df: Transformed dataframe
        name: Name for the output file
        original_path: Path to the original dataset
        logger: Logger instance

    Returns:
        Path to the saved file
    """
    output_dir = Path(original_path).parent / "preprocessed"
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / f"{name}.csv"

    logger.debug(f"Saving to {output_path}")
    df.to_csv(output_path, index=False)
    logger.debug(f"File saved successfully: {output_path}")

    return str(output_path)
