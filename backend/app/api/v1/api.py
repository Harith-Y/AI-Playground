# API v1 router aggregator

from fastapi import APIRouter
from app.api.v1.endpoints import datasets

api_router = APIRouter()

api_router.include_router(
    datasets.router,
    prefix="/datasets",
    tags=["datasets"]
)
