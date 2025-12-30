"""
Model Comparison Service

Provides logic for comparing multiple model runs across various metrics
and generating recommendations for model selection.
"""

from typing import List, Dict, Any, Optional, Tuple
from uuid import UUID
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import and_
import numpy as np
import uuid as uuid_lib

from app.models.model_run import ModelRun
from app.schemas.model import (
    CompareModelsRequest,
    ModelComparisonItem,
    ModelComparisonResponse,
    MetricStatistics,
    ModelRankingRequest,
    ModelRankingResponse,
    RankedModel
)
from app.utils.logger import get_logger

logger = get_logger(__name__)


class ModelComparisonService:
    """Service for comparing and ranking multiple model runs."""
    
    def __init__(self, db: Session):
        """
        Initialize the comparison service.
        
        Args:
            db: Database session
        """
        self.db = db
    
    def compare_models(
        self,
        request: CompareModelsRequest,
        user_id: Optional[str] = None
    ) -> ModelComparisonResponse:
        """
        Compare multiple model runs and generate comprehensive comparison report.
        
        Args:
            request: Comparison request with model run IDs and parameters
            user_id: Optional user ID for permission checking
        
        Returns:
            ModelComparisonResponse with ranked models and statistics
        
        Raises:
            ValueError: If models not found, have different task types, or insufficient data
        """
        logger.info(f"Comparing {len(request.model_run_ids)} models")
        
        # Fetch model runs
        model_runs = self._fetch_model_runs(request.model_run_ids, user_id)
        
        # Validate all models are comparable
        self._validate_comparability(model_runs)
        
        # Detect task type from models
        task_type = self._detect_task_type(model_runs)
        
        # Determine comparison metrics
        comparison_metrics = request.comparison_metrics
        if comparison_metrics is None:
            comparison_metrics = self._auto_detect_metrics(model_runs, task_type)
        
        # Determine ranking criteria
        ranking_criteria = request.ranking_criteria
        if ranking_criteria is None:
            ranking_criteria = self._get_default_ranking_metric(task_type)
        
        # Build comparison items
        comparison_items = self._build_comparison_items(
            model_runs,
            comparison_metrics,
            ranking_criteria
        )
        
        # Calculate metric statistics
        metric_stats = self._calculate_metric_statistics(
            comparison_items,
            comparison_metrics
        )
        
        # Get best model
        best_model = max(
            comparison_items,
            key=lambda x: x.ranking_score if x.ranking_score is not None else float('-inf')
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            comparison_items,
            best_model,
            task_type,
            ranking_criteria
        )
        
        comparison_id = f"comp-{uuid_lib.uuid4().hex[:12]}"
        
        return ModelComparisonResponse(
            comparison_id=comparison_id,
            task_type=task_type,
            total_models=len(comparison_items),
            compared_models=comparison_items,
            best_model=best_model,
            metric_statistics=metric_stats,
            ranking_criteria=ranking_criteria,
            recommendations=recommendations,
            timestamp=datetime.utcnow().isoformat()
        )
    
    def rank_models(
        self,
        request: ModelRankingRequest,
        user_id: Optional[str] = None
    ) -> ModelRankingResponse:
        """
        Rank models using custom weighted criteria.
        
        Args:
            request: Ranking request with model IDs and weights
            user_id: Optional user ID for permission checking
        
        Returns:
            ModelRankingResponse with ranked models
        
        Raises:
            ValueError: If weights don't sum to 1.0 or models not found
        """
        logger.info(f"Ranking {len(request.model_run_ids)} models with custom weights")
        
        # Validate weights sum to 1.0
        weight_sum = sum(request.ranking_weights.values())
        if not np.isclose(weight_sum, 1.0, atol=1e-6):
            raise ValueError(f"Ranking weights must sum to 1.0, got {weight_sum}")
        
        # Fetch model runs
        model_runs = self._fetch_model_runs(request.model_run_ids, user_id)
        
        # Calculate composite scores
        ranked_models = self._calculate_composite_scores(
            model_runs,
            request.ranking_weights,
            request.higher_is_better
        )
        
        # Sort by composite score (descending)
        ranked_models.sort(key=lambda x: x.composite_score, reverse=True)
        
        # Assign ranks
        for idx, model in enumerate(ranked_models, start=1):
            model.rank = idx
        
        best_model = ranked_models[0] if ranked_models else None
        
        # Calculate score range
        scores = [m.composite_score for m in ranked_models]
        score_range = {
            "min": min(scores) if scores else 0.0,
            "max": max(scores) if scores else 0.0,
            "spread": max(scores) - min(scores) if scores else 0.0
        }
        
        ranking_id = f"rank-{uuid_lib.uuid4().hex[:12]}"
        
        return ModelRankingResponse(
            ranking_id=ranking_id,
            ranked_models=ranked_models,
            ranking_weights=request.ranking_weights,
            best_model=best_model,
            score_range=score_range,
            timestamp=datetime.utcnow().isoformat()
        )
    
    def _fetch_model_runs(
        self,
        model_run_ids: List[UUID],
        user_id: Optional[str] = None
    ) -> List[ModelRun]:
        """
        Fetch model runs from database.
        
        Args:
            model_run_ids: List of model run UUIDs
            user_id: Optional user ID for filtering
        
        Returns:
            List of ModelRun objects
        
        Raises:
            ValueError: If models not found
        """
        query = self.db.query(ModelRun).filter(
            ModelRun.id.in_(model_run_ids)
        )
        
        model_runs = query.all()
        
        if len(model_runs) != len(model_run_ids):
            found_ids = {str(mr.id) for mr in model_runs}
            missing_ids = [str(mid) for mid in model_run_ids if str(mid) not in found_ids]
            raise ValueError(f"Model runs not found: {missing_ids}")
        
        logger.info(f"Fetched {len(model_runs)} model runs for comparison")
        return model_runs
    
    def _validate_comparability(self, model_runs: List[ModelRun]) -> None:
        """
        Validate that models can be compared.
        
        Args:
            model_runs: List of model runs
        
        Raises:
            ValueError: If models are not comparable
        """
        if len(model_runs) < 2:
            raise ValueError("At least 2 models required for comparison")
        
        # Check all have completed successfully
        incomplete = [mr for mr in model_runs if mr.status != "completed"]
        if incomplete:
            incomplete_ids = [str(mr.id) for mr in incomplete]
            raise ValueError(
                f"All models must have status 'completed'. "
                f"Incomplete models: {incomplete_ids}"
            )
        
        # Check all have metrics
        no_metrics = [mr for mr in model_runs if not mr.metrics]
        if no_metrics:
            no_metrics_ids = [str(mr.id) for mr in no_metrics]
            raise ValueError(
                f"All models must have metrics. "
                f"Models without metrics: {no_metrics_ids}"
            )
    
    def _detect_task_type(self, model_runs: List[ModelRun]) -> str:
        """
        Detect task type from model runs.
        
        Args:
            model_runs: List of model runs
        
        Returns:
            Task type string (classification, regression, clustering)
        """
        # Check metrics to infer task type
        first_metrics = model_runs[0].metrics
        
        classification_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        regression_metrics = ['mse', 'rmse', 'mae', 'r2', 'r2_score']
        
        has_classification = any(m in first_metrics for m in classification_metrics)
        has_regression = any(m in first_metrics for m in regression_metrics)
        
        if has_classification:
            return "classification"
        elif has_regression:
            return "regression"
        else:
            return "unknown"
    
    def _auto_detect_metrics(
        self,
        model_runs: List[ModelRun],
        task_type: str
    ) -> List[str]:
        """
        Auto-detect available metrics for comparison.
        
        Args:
            model_runs: List of model runs
            task_type: Detected task type
        
        Returns:
            List of metric names
        """
        # Find common metrics across all models
        all_metrics = [set(mr.metrics.keys()) for mr in model_runs]
        common_metrics = set.intersection(*all_metrics) if all_metrics else set()
        
        # Filter to relevant metrics based on task type
        if task_type == "classification":
            preferred = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        elif task_type == "regression":
            preferred = ['r2_score', 'rmse', 'mae', 'mse']
        else:
            preferred = []
        
        # Return common metrics in preferred order
        metrics = [m for m in preferred if m in common_metrics]
        
        # Add any other common metrics not in preferred list
        other_metrics = sorted(common_metrics - set(metrics))
        metrics.extend(other_metrics)
        
        logger.info(f"Auto-detected metrics: {metrics}")
        return metrics
    
    def _get_default_ranking_metric(self, task_type: str) -> str:
        """
        Get default ranking metric for task type.
        
        Args:
            task_type: Task type
        
        Returns:
            Default metric name
        """
        defaults = {
            "classification": "f1_score",
            "regression": "r2_score",
            "unknown": "accuracy"
        }
        return defaults.get(task_type, "accuracy")
    
    def _build_comparison_items(
        self,
        model_runs: List[ModelRun],
        comparison_metrics: List[str],
        ranking_criteria: str
    ) -> List[ModelComparisonItem]:
        """
        Build comparison items from model runs.
        
        Args:
            model_runs: List of model runs
            comparison_metrics: Metrics to compare
            ranking_criteria: Primary metric for ranking
        
        Returns:
            List of ModelComparisonItem objects
        """
        items = []
        
        for mr in model_runs:
            # Get ranking score
            ranking_score = mr.metrics.get(ranking_criteria)
            
            item = ModelComparisonItem(
                model_run_id=str(mr.id),
                model_type=mr.model_type,
                experiment_id=str(mr.experiment_id),
                status=mr.status,
                metrics=mr.metrics,
                hyperparameters=mr.hyperparameters or {},
                training_time=mr.training_time,
                created_at=mr.created_at.isoformat() if mr.created_at else None,
                rank=None,  # Will be assigned later
                ranking_score=ranking_score
            )
            items.append(item)
        
        # Sort by ranking score (descending)
        items.sort(
            key=lambda x: x.ranking_score if x.ranking_score is not None else float('-inf'),
            reverse=True
        )
        
        # Assign ranks
        for idx, item in enumerate(items, start=1):
            item.rank = idx
        
        return items
    
    def _calculate_metric_statistics(
        self,
        comparison_items: List[ModelComparisonItem],
        comparison_metrics: List[str]
    ) -> List[MetricStatistics]:
        """
        Calculate statistics for each metric.
        
        Args:
            comparison_items: List of comparison items
            comparison_metrics: Metrics to analyze
        
        Returns:
            List of MetricStatistics objects
        """
        stats = []
        
        for metric in comparison_metrics:
            values = []
            model_ids = []
            
            for item in comparison_items:
                if metric in item.metrics and item.metrics[metric] is not None:
                    values.append(item.metrics[metric])
                    model_ids.append(item.model_run_id)
            
            if not values:
                continue
            
            values_array = np.array(values)
            
            best_idx = np.argmax(values_array)
            worst_idx = np.argmin(values_array)
            
            stat = MetricStatistics(
                metric_name=metric,
                mean=float(np.mean(values_array)),
                std=float(np.std(values_array)),
                min=float(np.min(values_array)),
                max=float(np.max(values_array)),
                best_model_id=model_ids[best_idx],
                worst_model_id=model_ids[worst_idx]
            )
            stats.append(stat)
        
        return stats
    
    def _generate_recommendations(
        self,
        comparison_items: List[ModelComparisonItem],
        best_model: ModelComparisonItem,
        task_type: str,
        ranking_criteria: str
    ) -> List[str]:
        """
        Generate recommendations based on comparison results.
        
        Args:
            comparison_items: List of comparison items
            best_model: Best performing model
            task_type: Task type
            ranking_criteria: Ranking metric used
        
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        # Best model recommendation
        best_score = best_model.ranking_score
        recommendations.append(
            f"{best_model.model_type} achieved the best {ranking_criteria} "
            f"of {best_score:.4f}"
        )
        
        # Check if there's a clear winner
        if len(comparison_items) >= 2:
            second_best = comparison_items[1]
            score_diff = best_score - (second_best.ranking_score or 0)
            
            if score_diff < 0.01:
                recommendations.append(
                    f"The top 2 models have very similar performance "
                    f"(difference: {score_diff:.4f}). Consider ensembling them."
                )
            elif score_diff > 0.1:
                recommendations.append(
                    f"{best_model.model_type} significantly outperforms other models "
                    f"by {score_diff:.4f}. Use it with confidence."
                )
        
        # Training time considerations
        if best_model.training_time is not None:
            fast_models = [
                m for m in comparison_items
                if m.training_time and m.training_time < best_model.training_time * 0.5
            ]
            
            if fast_models:
                fastest = min(fast_models, key=lambda x: x.training_time or float('inf'))
                fastest_score = fastest.ranking_score or 0
                score_drop = best_score - fastest_score
                
                if score_drop < 0.05:
                    recommendations.append(
                        f"Consider {fastest.model_type} for faster training "
                        f"({fastest.training_time:.1f}s vs {best_model.training_time:.1f}s) "
                        f"with minimal performance drop ({score_drop:.4f})"
                    )
        
        # Hyperparameter insights
        if len(comparison_items) >= 3:
            recommendations.append(
                f"Compared {len(comparison_items)} models. "
                "Consider hyperparameter tuning on the top model for further improvements."
            )
        
        return recommendations
    
    def _calculate_composite_scores(
        self,
        model_runs: List[ModelRun],
        ranking_weights: Dict[str, float],
        higher_is_better: Optional[Dict[str, bool]] = None
    ) -> List[RankedModel]:
        """
        Calculate composite scores using weighted metrics.
        
        Args:
            model_runs: List of model runs
            ranking_weights: Metric weights
            higher_is_better: Optional dict indicating if higher is better for each metric
        
        Returns:
            List of RankedModel objects
        """
        if higher_is_better is None:
            # Default: assume higher is better for most metrics
            higher_is_better = {metric: True for metric in ranking_weights.keys()}
        
        ranked_models = []
        
        # Normalize metrics across all models for fair weighting
        metric_ranges = self._calculate_metric_ranges(model_runs, ranking_weights.keys())
        
        for mr in model_runs:
            individual_scores = {}
            weighted_contributions = {}
            composite_score = 0.0
            
            for metric, weight in ranking_weights.items():
                if metric not in mr.metrics or mr.metrics[metric] is None:
                    raise ValueError(
                        f"Metric '{metric}' not found in model {mr.id}"
                    )
                
                raw_value = mr.metrics[metric]
                individual_scores[metric] = raw_value
                
                # Normalize to 0-1 range
                metric_min, metric_max = metric_ranges[metric]
                if metric_max - metric_min > 1e-10:
                    normalized = (raw_value - metric_min) / (metric_max - metric_min)
                else:
                    normalized = 1.0
                
                # Invert if lower is better
                if not higher_is_better.get(metric, True):
                    normalized = 1.0 - normalized
                
                contribution = normalized * weight
                weighted_contributions[metric] = contribution
                composite_score += contribution
            
            ranked_model = RankedModel(
                model_run_id=str(mr.id),
                model_type=mr.model_type,
                rank=0,  # Will be assigned later
                composite_score=composite_score,
                individual_scores=individual_scores,
                weighted_contributions=weighted_contributions
            )
            ranked_models.append(ranked_model)
        
        return ranked_models
    
    def _calculate_metric_ranges(
        self,
        model_runs: List[ModelRun],
        metrics: Any  # keys() from dict
    ) -> Dict[str, Tuple[float, float]]:
        """
        Calculate min/max ranges for metrics across models.
        
        Args:
            model_runs: List of model runs
            metrics: Metric names
        
        Returns:
            Dict of metric -> (min, max) tuples
        """
        ranges = {}
        
        for metric in metrics:
            values = [
                mr.metrics[metric]
                for mr in model_runs
                if metric in mr.metrics and mr.metrics[metric] is not None
            ]
            
            if values:
                ranges[metric] = (min(values), max(values))
            else:
                ranges[metric] = (0.0, 1.0)
        
        return ranges
