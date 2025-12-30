# API v1 router aggregator

from fastapi import APIRouter
from app.api.v1.endpoints import datasets, preprocessing, models, tuning, tuning_orchestration

api_router = APIRouter()

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
