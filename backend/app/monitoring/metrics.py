"""
Prometheus Metrics Definitions

Defines all Prometheus metrics for monitoring evaluation and tuning performance.
"""

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    Summary,
    CollectorRegistry,
    REGISTRY,
)

# Use default registry
metrics_registry = REGISTRY

# ============================================================================
# Task Metrics
# ============================================================================

# Task execution duration
task_duration_histogram = Histogram(
    'celery_task_duration_seconds',
    'Duration of Celery task execution',
    ['task_name', 'status'],
    buckets=(1, 5, 10, 30, 60, 120, 300, 600, 1800, 3600, 7200)  # Up to 2 hours
)

# Task counter
task_counter = Counter(
    'celery_task_total',
    'Total number of Celery tasks',
    ['task_name', 'status']
)

# Active tasks
active_tasks_gauge = Gauge(
    'celery_active_tasks',
    'Number of currently active tasks',
    ['task_name']
)

# Task queue length
task_queue_length_gauge = Gauge(
    'celery_queue_length',
    'Number of tasks waiting in queue',
    ['queue_name']
)

# ============================================================================
# Resource Metrics
# ============================================================================

# Memory usage
memory_usage_gauge = Gauge(
    'process_memory_usage_bytes',
    'Memory usage in bytes',
    ['process_type']
)

# CPU usage
cpu_usage_gauge = Gauge(
    'process_cpu_usage_percent',
    'CPU usage percentage',
    ['process_type']
)

# Disk I/O
disk_io_counter = Counter(
    'disk_io_bytes_total',
    'Total disk I/O in bytes',
    ['operation', 'process_type']
)

# ============================================================================
# Model Training Metrics
# ============================================================================

# Training duration
training_duration_histogram = Histogram(
    'model_training_duration_seconds',
    'Duration of model training',
    ['model_type', 'task_type'],
    buckets=(10, 30, 60, 120, 300, 600, 1800, 3600, 7200)
)

# Training samples
training_samples_histogram = Histogram(
    'model_training_samples',
    'Number of samples used in training',
    ['model_type'],
    buckets=(100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000)
)

# Training features
training_features_histogram = Histogram(
    'model_training_features',
    'Number of features used in training',
    ['model_type'],
    buckets=(5, 10, 20, 50, 100, 200, 500, 1000)
)

# Training success/failure
training_result_counter = Counter(
    'model_training_result_total',
    'Training results',
    ['model_type', 'result']
)

# ============================================================================
# Hyperparameter Tuning Metrics
# ============================================================================

# Tuning duration
tuning_duration_histogram = Histogram(
    'hyperparameter_tuning_duration_seconds',
    'Duration of hyperparameter tuning',
    ['model_type', 'tuning_method'],
    buckets=(60, 300, 600, 1800, 3600, 7200, 14400, 28800)  # Up to 8 hours
)

# Tuning iterations
tuning_iterations_histogram = Histogram(
    'hyperparameter_tuning_iterations',
    'Number of tuning iterations',
    ['tuning_method'],
    buckets=(5, 10, 20, 50, 100, 200, 500, 1000)
)

# Tuning result
tuning_result_counter = Counter(
    'hyperparameter_tuning_result_total',
    'Tuning results',
    ['model_type', 'tuning_method', 'result']
)

# Best score achieved
tuning_best_score_histogram = Histogram(
    'hyperparameter_tuning_best_score',
    'Best score achieved during tuning',
    ['model_type', 'tuning_method'],
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0)
)

# ============================================================================
# Evaluation Metrics
# ============================================================================

# Evaluation duration
evaluation_duration_histogram = Histogram(
    'model_evaluation_duration_seconds',
    'Duration of model evaluation',
    ['model_type', 'task_type'],
    buckets=(0.1, 0.5, 1, 5, 10, 30, 60, 120)
)

# Evaluation requests
evaluation_request_counter = Counter(
    'model_evaluation_requests_total',
    'Total evaluation requests',
    ['model_type', 'endpoint']
)

# Cache hits/misses
cache_counter = Counter(
    'evaluation_cache_total',
    'Cache hits and misses',
    ['cache_type', 'result']
)

# ============================================================================
# API Endpoint Metrics
# ============================================================================

# Request duration
api_request_duration_histogram = Histogram(
    'api_request_duration_seconds',
    'API request duration',
    ['method', 'endpoint', 'status_code'],
    buckets=(0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 30)
)

# Request counter
api_request_counter = Counter(
    'api_requests_total',
    'Total API requests',
    ['method', 'endpoint', 'status_code']
)

# Active requests
api_active_requests_gauge = Gauge(
    'api_active_requests',
    'Number of active API requests',
    ['method', 'endpoint']
)

# Request size
api_request_size_histogram = Histogram(
    'api_request_size_bytes',
    'API request size in bytes',
    ['method', 'endpoint'],
    buckets=(100, 1000, 10000, 100000, 1000000, 10000000, 100000000)
)

# Response size
api_response_size_histogram = Histogram(
    'api_response_size_bytes',
    'API response size in bytes',
    ['method', 'endpoint'],
    buckets=(100, 1000, 10000, 100000, 1000000, 10000000, 100000000)
)

# ============================================================================
# Database Metrics
# ============================================================================

# Query duration
db_query_duration_histogram = Histogram(
    'database_query_duration_seconds',
    'Database query duration',
    ['operation', 'table'],
    buckets=(0.001, 0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10)
)

# Query counter
db_query_counter = Counter(
    'database_queries_total',
    'Total database queries',
    ['operation', 'table']
)

# Connection pool
db_connection_pool_gauge = Gauge(
    'database_connection_pool_size',
    'Database connection pool size',
    ['state']  # active, idle, total
)

# ============================================================================
# Preprocessing Metrics
# ============================================================================

# Preprocessing duration
preprocessing_duration_histogram = Histogram(
    'preprocessing_duration_seconds',
    'Duration of preprocessing operations',
    ['step_type'],
    buckets=(0.1, 0.5, 1, 5, 10, 30, 60, 120, 300)
)

# Preprocessing steps
preprocessing_steps_counter = Counter(
    'preprocessing_steps_total',
    'Total preprocessing steps executed',
    ['step_type', 'result']
)

# ============================================================================
# Error Metrics
# ============================================================================

# Error counter
error_counter = Counter(
    'application_errors_total',
    'Total application errors',
    ['error_type', 'component']
)

# Exception counter
exception_counter = Counter(
    'application_exceptions_total',
    'Total exceptions raised',
    ['exception_type', 'component']
)

# ============================================================================
# Business Metrics
# ============================================================================

# Models trained
models_trained_counter = Counter(
    'models_trained_total',
    'Total models trained',
    ['model_type', 'task_type']
)

# Datasets uploaded
datasets_uploaded_counter = Counter(
    'datasets_uploaded_total',
    'Total datasets uploaded',
    ['file_type']
)

# Experiments created
experiments_created_counter = Counter(
    'experiments_created_total',
    'Total experiments created'
)

# ============================================================================
# Summary Metrics (for percentiles)
# ============================================================================

# Training time summary
training_time_summary = Summary(
    'model_training_time_summary',
    'Summary of model training times',
    ['model_type']
)

# Tuning time summary
tuning_time_summary = Summary(
    'hyperparameter_tuning_time_summary',
    'Summary of hyperparameter tuning times',
    ['tuning_method']
)

# API latency summary
api_latency_summary = Summary(
    'api_latency_summary',
    'Summary of API latencies',
    ['endpoint']
)
