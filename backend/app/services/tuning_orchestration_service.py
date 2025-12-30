"""
Tuning Orchestration Service.

Provides high-level orchestration of hyperparameter tuning workflows including:
- Progressive refinement (grid → random → bayesian)
- Multi-model parallel tuning
- Adaptive search strategies
- Workflow management and state tracking
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from uuid import UUID, uuid4
from sqlalchemy.orm import Session
from celery import group, chain, chord
from celery.result import AsyncResult, GroupResult

from app.models.model_run import ModelRun
from app.models.tuning_run import TuningRun, TuningStatus
from app.models.experiment import Experiment
from app.tasks.tuning_tasks import tune_hyperparameters
from app.utils.logger import get_logger

logger = get_logger(__name__)


class ProgressiveSearchConfig:
    """Configuration for progressive search workflow."""
    
    def __init__(
        self,
        initial_method: str = "grid_search",
        intermediate_method: str = "random_search",
        final_method: str = "bayesian",
        initial_param_grid: Optional[Dict[str, List]] = None,
        refinement_factor: float = 0.3,
        cv_folds: int = 5,
        scoring_metric: Optional[str] = None,
        n_iter_random: int = 50,
        n_iter_bayesian: int = 30
    ):
        """
        Initialize progressive search configuration.
        
        Args:
            initial_method: First search method (default: grid_search)
            intermediate_method: Second search method (default: random_search)
            final_method: Final search method (default: bayesian)
            initial_param_grid: Initial parameter space
            refinement_factor: Factor to narrow parameter space (0.0-1.0)
            cv_folds: Number of cross-validation folds
            scoring_metric: Metric to optimize
            n_iter_random: Iterations for random search
            n_iter_bayesian: Iterations for Bayesian search
        """
        self.initial_method = initial_method
        self.intermediate_method = intermediate_method
        self.final_method = final_method
        self.initial_param_grid = initial_param_grid
        self.refinement_factor = refinement_factor
        self.cv_folds = cv_folds
        self.scoring_metric = scoring_metric
        self.n_iter_random = n_iter_random
        self.n_iter_bayesian = n_iter_bayesian


class MultiModelConfig:
    """Configuration for multi-model comparison."""
    
    def __init__(
        self,
        model_run_ids: List[UUID],
        tuning_method: str = "bayesian",
        param_grids: Optional[Dict[UUID, Dict[str, List]]] = None,
        cv_folds: int = 5,
        scoring_metric: Optional[str] = None,
        n_iter: int = 30,
        parallel: bool = True
    ):
        """
        Initialize multi-model comparison configuration.
        
        Args:
            model_run_ids: List of model run IDs to tune
            tuning_method: Tuning method to use for all models
            param_grids: Parameter grids per model (optional)
            cv_folds: Number of cross-validation folds
            scoring_metric: Metric to optimize
            n_iter: Iterations for random/bayesian search
            parallel: Whether to run tuning in parallel
        """
        self.model_run_ids = model_run_ids
        self.tuning_method = tuning_method
        self.param_grids = param_grids or {}
        self.cv_folds = cv_folds
        self.scoring_metric = scoring_metric
        self.n_iter = n_iter
        self.parallel = parallel


class TuningOrchestrationService:
    """Service for orchestrating complex tuning workflows."""
    
    def __init__(self, db: Session):
        """
        Initialize orchestration service.
        
        Args:
            db: Database session
        """
        self.db = db
        self.logger = get_logger(__name__)
    
    def progressive_search(
        self,
        model_run_id: UUID,
        config: ProgressiveSearchConfig,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute progressive search workflow: grid → random → bayesian.
        
        This workflow starts with a broad grid search to identify promising
        regions, then uses random search for exploration, and finally
        applies Bayesian optimization for fine-tuning.
        
        Args:
            model_run_id: UUID of model run to tune
            config: Progressive search configuration
            user_id: User ID for logging
        
        Returns:
            Dict containing orchestration details and task IDs
        
        Workflow:
            1. Grid search with initial parameter space
            2. Extract top parameters from grid search
            3. Refine parameter space around best results
            4. Random search in refined space
            5. Further refine around best random search results
            6. Bayesian optimization for final tuning
        """
        self.logger.info(
            f"Starting progressive search for model_run_id={model_run_id}",
            extra={
                'event': 'progressive_search_start',
                'model_run_id': str(model_run_id),
                'user_id': user_id
            }
        )
        
        # 1. Validate model run
        model_run = self.db.query(ModelRun).filter(
            ModelRun.id == model_run_id
        ).first()
        
        if not model_run:
            raise ValueError(f"ModelRun {model_run_id} not found")
        
        if model_run.status != "completed":
            raise ValueError(
                f"Model run must be completed. Current status: {model_run.status}"
            )
        
        orchestration_id = uuid4()
        
        # 2. Create tuning runs for each stage
        stages = []
        
        # Stage 1: Grid Search
        grid_tuning_run = TuningRun(
            id=uuid4(),
            model_run_id=model_run_id,
            tuning_method=config.initial_method,
            status=TuningStatus.PENDING,
            created_at=datetime.utcnow()
        )
        self.db.add(grid_tuning_run)
        stages.append({
            'stage': 'grid_search',
            'tuning_run_id': grid_tuning_run.id,
            'method': config.initial_method
        })
        
        # Stage 2: Random Search
        random_tuning_run = TuningRun(
            id=uuid4(),
            model_run_id=model_run_id,
            tuning_method=config.intermediate_method,
            status=TuningStatus.PENDING,
            created_at=datetime.utcnow()
        )
        self.db.add(random_tuning_run)
        stages.append({
            'stage': 'random_search',
            'tuning_run_id': random_tuning_run.id,
            'method': config.intermediate_method
        })
        
        # Stage 3: Bayesian Optimization
        bayesian_tuning_run = TuningRun(
            id=uuid4(),
            model_run_id=model_run_id,
            tuning_method=config.final_method,
            status=TuningStatus.PENDING,
            created_at=datetime.utcnow()
        )
        self.db.add(bayesian_tuning_run)
        stages.append({
            'stage': 'bayesian_optimization',
            'tuning_run_id': bayesian_tuning_run.id,
            'method': config.final_method
        })
        
        self.db.commit()
        
        # 3. Queue grid search task
        grid_task = tune_hyperparameters.apply_async(
            kwargs={
                'tuning_run_id': str(grid_tuning_run.id),
                'model_run_id': str(model_run_id),
                'tuning_method': config.initial_method,
                'param_grid': config.initial_param_grid,
                'cv_folds': config.cv_folds,
                'scoring_metric': config.scoring_metric,
                'user_id': user_id
            }
        )
        
        grid_tuning_run.status = TuningStatus.RUNNING
        if not grid_tuning_run.results:
            grid_tuning_run.results = {}
        grid_tuning_run.results['task_id'] = grid_task.id
        grid_tuning_run.results['orchestration_id'] = str(orchestration_id)
        grid_tuning_run.results['stage'] = 'grid_search'
        grid_tuning_run.results['next_stage'] = 'random_search'
        
        from sqlalchemy.orm.attributes import flag_modified
        flag_modified(grid_tuning_run, 'results')
        self.db.commit()
        
        self.logger.info(
            f"Progressive search orchestration created",
            extra={
                'event': 'progressive_search_created',
                'orchestration_id': str(orchestration_id),
                'grid_task_id': grid_task.id,
                'stages': len(stages)
            }
        )
        
        return {
            'orchestration_id': str(orchestration_id),
            'model_run_id': str(model_run_id),
            'workflow': 'progressive_search',
            'stages': stages,
            'grid_search': {
                'tuning_run_id': str(grid_tuning_run.id),
                'task_id': grid_task.id,
                'status': 'RUNNING'
            },
            'random_search': {
                'tuning_run_id': str(random_tuning_run.id),
                'status': 'PENDING'
            },
            'bayesian_optimization': {
                'tuning_run_id': str(bayesian_tuning_run.id),
                'status': 'PENDING'
            },
            'message': 'Progressive search workflow initiated. Grid search is running.'
        }
    
    def trigger_next_stage(
        self,
        orchestration_id: str,
        completed_tuning_run_id: UUID,
        user_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Trigger the next stage of progressive search based on completed stage.
        
        This method should be called when a stage completes to automatically
        start the next stage with refined parameters.
        
        Args:
            orchestration_id: ID of the orchestration workflow
            completed_tuning_run_id: ID of the just-completed tuning run
            user_id: User ID for logging
        
        Returns:
            Dict with next stage details, or None if workflow is complete
        """
        # Get completed tuning run
        completed_run = self.db.query(TuningRun).filter(
            TuningRun.id == completed_tuning_run_id
        ).first()
        
        if not completed_run:
            raise ValueError(f"TuningRun {completed_tuning_run_id} not found")
        
        if completed_run.status != TuningStatus.COMPLETED:
            raise ValueError(
                f"TuningRun must be completed. Current status: {completed_run.status}"
            )
        
        # Get orchestration details from results
        if not completed_run.results or 'next_stage' not in completed_run.results:
            self.logger.info(
                f"No next stage found, workflow complete",
                extra={'orchestration_id': orchestration_id}
            )
            return None
        
        next_stage = completed_run.results['next_stage']
        model_run_id = completed_run.model_run_id
        
        # Refine parameter grid based on best results
        refined_param_grid = self._refine_param_grid(
            completed_run.best_params,
            completed_run.results.get('all_results', [])
        )
        
        # Find next stage tuning run
        if next_stage == 'random_search':
            next_tuning_run = self.db.query(TuningRun).filter(
                TuningRun.model_run_id == model_run_id,
                TuningRun.tuning_method == 'random_search',
                TuningRun.status == TuningStatus.PENDING
            ).first()
            
            if next_tuning_run:
                # Queue random search
                task = tune_hyperparameters.apply_async(
                    kwargs={
                        'tuning_run_id': str(next_tuning_run.id),
                        'model_run_id': str(model_run_id),
                        'tuning_method': 'random_search',
                        'param_grid': refined_param_grid,
                        'cv_folds': completed_run.results.get('cv_folds', 5),
                        'scoring_metric': completed_run.results.get('scoring_metric'),
                        'n_iter': 50,
                        'user_id': user_id
                    }
                )
                
                next_tuning_run.status = TuningStatus.RUNNING
                if not next_tuning_run.results:
                    next_tuning_run.results = {}
                next_tuning_run.results['task_id'] = task.id
                next_tuning_run.results['orchestration_id'] = orchestration_id
                next_tuning_run.results['stage'] = 'random_search'
                next_tuning_run.results['next_stage'] = 'bayesian_optimization'
                next_tuning_run.results['refined_from'] = str(completed_tuning_run_id)
                
                from sqlalchemy.orm.attributes import flag_modified
                flag_modified(next_tuning_run, 'results')
                self.db.commit()
                
                return {
                    'stage': 'random_search',
                    'tuning_run_id': str(next_tuning_run.id),
                    'task_id': task.id,
                    'status': 'RUNNING'
                }
        
        elif next_stage == 'bayesian_optimization':
            next_tuning_run = self.db.query(TuningRun).filter(
                TuningRun.model_run_id == model_run_id,
                TuningRun.tuning_method == 'bayesian',
                TuningRun.status == TuningStatus.PENDING
            ).first()
            
            if next_tuning_run:
                # Queue Bayesian optimization
                task = tune_hyperparameters.apply_async(
                    kwargs={
                        'tuning_run_id': str(next_tuning_run.id),
                        'model_run_id': str(model_run_id),
                        'tuning_method': 'bayesian',
                        'param_grid': refined_param_grid,
                        'cv_folds': completed_run.results.get('cv_folds', 5),
                        'scoring_metric': completed_run.results.get('scoring_metric'),
                        'n_iter': 30,
                        'user_id': user_id
                    }
                )
                
                next_tuning_run.status = TuningStatus.RUNNING
                if not next_tuning_run.results:
                    next_tuning_run.results = {}
                next_tuning_run.results['task_id'] = task.id
                next_tuning_run.results['orchestration_id'] = orchestration_id
                next_tuning_run.results['stage'] = 'bayesian_optimization'
                next_tuning_run.results['refined_from'] = str(completed_tuning_run_id)
                
                from sqlalchemy.orm.attributes import flag_modified
                flag_modified(next_tuning_run, 'results')
                self.db.commit()
                
                return {
                    'stage': 'bayesian_optimization',
                    'tuning_run_id': str(next_tuning_run.id),
                    'task_id': task.id,
                    'status': 'RUNNING'
                }
        
        return None
    
    def _refine_param_grid(
        self,
        best_params: Dict[str, Any],
        top_results: List[Dict[str, Any]],
        refinement_factor: float = 0.3
    ) -> Dict[str, List]:
        """
        Refine parameter grid based on best results.
        
        Creates a narrower parameter space around the best parameters
        found in previous search.
        
        Args:
            best_params: Best parameters from previous search
            top_results: Top N results from previous search
            refinement_factor: How much to narrow the search (0.0-1.0)
        
        Returns:
            Refined parameter grid
        """
        refined_grid = {}
        
        for param, value in best_params.items():
            # Numeric parameters
            if isinstance(value, (int, float)):
                # Find range from top results
                values = [
                    r.get('params', {}).get(param, value)
                    for r in top_results
                    if param in r.get('params', {})
                ]
                
                if values:
                    min_val = min(values)
                    max_val = max(values)
                    
                    # Narrow range
                    range_size = max_val - min_val
                    new_min = max(min_val, value - range_size * refinement_factor)
                    new_max = min(max_val, value + range_size * refinement_factor)
                    
                    if isinstance(value, int):
                        refined_grid[param] = list(range(
                            int(new_min),
                            int(new_max) + 1
                        ))
                    else:
                        refined_grid[param] = [
                            new_min,
                            (new_min + new_max) / 2,
                            new_max
                        ]
            
            # Categorical parameters
            else:
                # Keep best value and similar values from top results
                values = set([
                    r.get('params', {}).get(param)
                    for r in top_results
                    if param in r.get('params', {})
                ])
                values.add(value)
                refined_grid[param] = list(values)
        
        return refined_grid
    
    def multi_model_comparison(
        self,
        config: MultiModelConfig,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute parallel tuning across multiple models.
        
        Tunes multiple models simultaneously and provides comparison
        of their performance after tuning.
        
        Args:
            config: Multi-model comparison configuration
            user_id: User ID for logging
        
        Returns:
            Dict containing orchestration details and task IDs
        """
        self.logger.info(
            f"Starting multi-model comparison for {len(config.model_run_ids)} models",
            extra={
                'event': 'multi_model_comparison_start',
                'n_models': len(config.model_run_ids),
                'user_id': user_id
            }
        )
        
        # Validate all model runs
        for model_run_id in config.model_run_ids:
            model_run = self.db.query(ModelRun).filter(
                ModelRun.id == model_run_id
            ).first()
            
            if not model_run:
                raise ValueError(f"ModelRun {model_run_id} not found")
            
            if model_run.status != "completed":
                raise ValueError(
                    f"Model run {model_run_id} must be completed. "
                    f"Current status: {model_run.status}"
                )
        
        orchestration_id = uuid4()
        tuning_tasks = []
        tuning_runs_info = []
        
        # Create tuning runs for each model
        for model_run_id in config.model_run_ids:
            tuning_run = TuningRun(
                id=uuid4(),
                model_run_id=model_run_id,
                tuning_method=config.tuning_method,
                status=TuningStatus.PENDING,
                created_at=datetime.utcnow()
            )
            self.db.add(tuning_run)
            
            # Get model-specific param grid or None
            param_grid = config.param_grids.get(model_run_id)
            
            # Create task signature
            task_kwargs = {
                'tuning_run_id': str(tuning_run.id),
                'model_run_id': str(model_run_id),
                'tuning_method': config.tuning_method,
                'param_grid': param_grid,
                'cv_folds': config.cv_folds,
                'scoring_metric': config.scoring_metric,
                'n_iter': config.n_iter,
                'user_id': user_id
            }
            
            tuning_runs_info.append({
                'model_run_id': str(model_run_id),
                'tuning_run_id': str(tuning_run.id)
            })
            
            if config.parallel:
                # Add to group for parallel execution
                tuning_tasks.append(
                    tune_hyperparameters.signature(kwargs=task_kwargs)
                )
            else:
                # Execute immediately for sequential
                task = tune_hyperparameters.apply_async(kwargs=task_kwargs)
                tuning_run.status = TuningStatus.RUNNING
                if not tuning_run.results:
                    tuning_run.results = {}
                tuning_run.results['task_id'] = task.id
                tuning_run.results['orchestration_id'] = str(orchestration_id)
                
                from sqlalchemy.orm.attributes import flag_modified
                flag_modified(tuning_run, 'results')
        
        self.db.commit()
        
        # Execute parallel group if configured
        group_task_id = None
        if config.parallel and tuning_tasks:
            job = group(tuning_tasks)
            group_result = job.apply_async()
            group_task_id = group_result.id
            
            # Update all tuning runs with group task ID
            for info in tuning_runs_info:
                tuning_run = self.db.query(TuningRun).filter(
                    TuningRun.id == UUID(info['tuning_run_id'])
                ).first()
                
                if tuning_run:
                    tuning_run.status = TuningStatus.RUNNING
                    if not tuning_run.results:
                        tuning_run.results = {}
                    tuning_run.results['orchestration_id'] = str(orchestration_id)
                    tuning_run.results['group_task_id'] = group_task_id
                    
                    from sqlalchemy.orm.attributes import flag_modified
                    flag_modified(tuning_run, 'results')
            
            self.db.commit()
        
        self.logger.info(
            f"Multi-model comparison orchestration created",
            extra={
                'event': 'multi_model_comparison_created',
                'orchestration_id': str(orchestration_id),
                'n_models': len(config.model_run_ids),
                'parallel': config.parallel,
                'group_task_id': group_task_id
            }
        )
        
        return {
            'orchestration_id': str(orchestration_id),
            'workflow': 'multi_model_comparison',
            'n_models': len(config.model_run_ids),
            'tuning_method': config.tuning_method,
            'parallel': config.parallel,
            'group_task_id': group_task_id,
            'tuning_runs': tuning_runs_info,
            'message': f"Multi-model comparison initiated for {len(config.model_run_ids)} models"
        }
    
    def get_orchestration_status(
        self,
        orchestration_id: str
    ) -> Dict[str, Any]:
        """
        Get status of an orchestration workflow.
        
        Args:
            orchestration_id: ID of the orchestration
        
        Returns:
            Dict with orchestration status and progress
        """
        # Find all tuning runs with this orchestration_id
        tuning_runs = self.db.query(TuningRun).filter(
            TuningRun.results.contains({'orchestration_id': orchestration_id})
        ).all()
        
        if not tuning_runs:
            raise ValueError(f"No orchestration found with id {orchestration_id}")
        
        # Determine workflow type
        workflow_type = None
        if tuning_runs[0].results:
            if 'stage' in tuning_runs[0].results:
                workflow_type = 'progressive_search'
            elif 'group_task_id' in tuning_runs[0].results:
                workflow_type = 'multi_model_comparison'
        
        # Aggregate status
        statuses = {
            'completed': 0,
            'running': 0,
            'failed': 0,
            'pending': 0
        }
        
        stages_info = []
        for run in tuning_runs:
            status_key = run.status.value.lower()
            statuses[status_key] = statuses.get(status_key, 0) + 1
            
            stage_info = {
                'tuning_run_id': str(run.id),
                'model_run_id': str(run.model_run_id),
                'method': run.tuning_method,
                'status': run.status.value
            }
            
            if run.results:
                if 'stage' in run.results:
                    stage_info['stage'] = run.results['stage']
                if 'task_id' in run.results:
                    stage_info['task_id'] = run.results['task_id']
                if run.best_params:
                    stage_info['best_score'] = run.results.get('best_score')
            
            stages_info.append(stage_info)
        
        # Overall status
        total = len(tuning_runs)
        if statuses['failed'] > 0:
            overall_status = 'FAILED'
        elif statuses['completed'] == total:
            overall_status = 'COMPLETED'
        elif statuses['running'] > 0:
            overall_status = 'RUNNING'
        else:
            overall_status = 'PENDING'
        
        progress = {
            'completed': statuses['completed'],
            'total': total,
            'percentage': round((statuses['completed'] / total) * 100, 2)
        }
        
        return {
            'orchestration_id': orchestration_id,
            'workflow_type': workflow_type,
            'overall_status': overall_status,
            'progress': progress,
            'statuses': statuses,
            'stages': stages_info
        }
    
    def get_best_model_from_comparison(
        self,
        orchestration_id: str
    ) -> Dict[str, Any]:
        """
        Get the best performing model from a multi-model comparison.
        
        Args:
            orchestration_id: ID of the multi-model comparison orchestration
        
        Returns:
            Dict with best model details
        """
        # Find all completed tuning runs with this orchestration_id
        tuning_runs = self.db.query(TuningRun).filter(
            TuningRun.results.contains({'orchestration_id': orchestration_id}),
            TuningRun.status == TuningStatus.COMPLETED
        ).all()
        
        if not tuning_runs:
            raise ValueError(
                f"No completed tuning runs found for orchestration {orchestration_id}"
            )
        
        # Find best based on score
        best_run = None
        best_score = float('-inf')
        
        for run in tuning_runs:
            if run.results and 'best_score' in run.results:
                score = run.results['best_score']
                if score > best_score:
                    best_score = score
                    best_run = run
        
        if not best_run:
            raise ValueError("No scores found in completed tuning runs")
        
        # Get model run details
        model_run = self.db.query(ModelRun).filter(
            ModelRun.id == best_run.model_run_id
        ).first()
        
        return {
            'orchestration_id': orchestration_id,
            'best_model': {
                'model_run_id': str(best_run.model_run_id),
                'tuning_run_id': str(best_run.id),
                'model_type': model_run.model_type if model_run else None,
                'best_score': best_score,
                'best_params': best_run.best_params,
                'tuning_method': best_run.tuning_method
            },
            'all_models': [
                {
                    'model_run_id': str(run.model_run_id),
                    'tuning_run_id': str(run.id),
                    'score': run.results.get('best_score') if run.results else None,
                    'method': run.tuning_method
                }
                for run in tuning_runs
            ]
        }
