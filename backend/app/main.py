from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from time import perf_counter
from uuid import uuid4

from app.core.config import settings
from app.core.exceptions import register_exception_handlers
from app.core.api_docs import get_openapi_config, get_custom_openapi_schema, API_TAGS
from app.utils.logger import get_logger, set_correlation_id
from app.api.v1.api import api_router
from app.monitoring.middleware import PerformanceMonitoringMiddleware


def create_app() -> FastAPI:
	# Get OpenAPI configuration
	openapi_config = get_openapi_config()
	
	app = FastAPI(
		title=openapi_config["title"],
		version=openapi_config["version"],
		description=openapi_config["description"],
		openapi_tags=openapi_config["openapi_tags"],
		contact=openapi_config["contact"],
		license_info=openapi_config["license_info"],
		terms_of_service=openapi_config["terms_of_service"],
		docs_url="/docs" if settings.DEBUG else None,
		redoc_url="/redoc" if settings.DEBUG else None,
		openapi_url="/openapi.json" if settings.DEBUG else None,
	)
	
	# Set custom OpenAPI schema
	app.openapi = lambda: get_custom_openapi_schema(app)

	logger = get_logger("backend")

	# Add performance monitoring middleware
	app.add_middleware(PerformanceMonitoringMiddleware)
	logger.info("Performance monitoring middleware added")

	# Request/Response logging middleware with correlation ID
	@app.middleware("http")
	async def logging_middleware(request: Request, call_next):
		correlation_id = request.headers.get("X-Request-ID", str(uuid4()))
		request.state.correlation_id = correlation_id
		set_correlation_id(correlation_id)

		start = perf_counter()
		logger.info(f"Request start method={request.method} path={request.url.path}")

		try:
			response = await call_next(request)
		finally:
			duration_ms = (perf_counter() - start) * 1000.0
			status = getattr(response, "status_code", 0)
			logger.info(
				f"Request end method={request.method} path={request.url.path} status={status} duration_ms={duration_ms:.2f}"
			)
			# Clear correlation id for non-request contexts
			set_correlation_id(None)

		# Echo correlation id in response for tracing
		response.headers["X-Request-ID"] = correlation_id
		return response

	# Register exception handlers
	register_exception_handlers(app, logger)

	# Register API v1 router
	app.include_router(api_router, prefix="/api/v1")

	# Basic health endpoint
	@app.get(
		"/health",
		tags=["health"],
		summary="Health Check",
		description="Check if the API is running and healthy",
		responses={
			200: {
				"description": "API is healthy",
				"content": {
					"application/json": {
						"example": {
							"status": "ok",
							"version": "1.0.0"
						}
					}
				}
			}
		}
	)
	async def health():
		"""
		Health check endpoint.
		
		Returns the current status and version of the API.
		Use this endpoint for:
		- Monitoring and alerting
		- Load balancer health checks
		- Deployment verification
		"""
		return {"status": "ok", "version": settings.VERSION}

	return app


app = create_app()
