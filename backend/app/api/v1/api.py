# API v1 router aggregator

from fastapi import APIRouter
from app.api.v1.endpoints import (
    auth,
    datasets,
    preprocessing,
    models,
    tuning,
    tuning_orchestration,
    metrics,
    code_generation,
    experiment_config
)

api_router = APIRouter()

# Authentication endpoints (no auth required for these)
api_router.include_router(
    auth.router,
    prefix="/auth",
    tags=["authentication"]
)

api_router.include_router(
    datasets.router,
    prefix="/datasets",
    tags=["datasets"]
)

api_router.include_router(
    preprocessing.router,
    prefix="/preprocessing",
    tags=["preprocessing"]
)

api_router.include_router(
    models.router,
    prefix="/models",
    tags=["models"]
)

api_router.include_router(
    tuning.router,
    prefix="/tuning",
    tags=["tuning"]
)

api_router.include_router(
    tuning_orchestration.router,
    prefix="/tuning-orchestration",
    tags=["tuning-orchestration"]
)

api_router.include_router(
    code_generation.router,
    prefix="/code-generation",
    tags=["code-generation"]
)

api_router.include_router(
    experiment_config.router,
    prefix="/experiments",
    tags=["experiments"]
)

# Metrics endpoint (no prefix, no tags - for Prometheus)
api_router.include_router(
    metrics.router,
    tags=["monitoring"]
)
