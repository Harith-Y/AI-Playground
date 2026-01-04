from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from time import perf_counter
from uuid import uuid4

from app.core.config import settings
from app.core.exceptions import register_exception_handlers
from app.core.api_docs import get_openapi_config, get_custom_openapi_schema, API_TAGS
from app.utils.logger import get_logger, set_correlation_id
from app.api.v1.api import api_router
from app.monitoring.middleware import PerformanceMonitoringMiddleware
from app.middleware import create_rate_limiter


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

	# Add CORS middleware for frontend access
	origins = [origin.strip() for origin in settings.ALLOWED_ORIGINS.split(",")]
	app.add_middleware(
		CORSMiddleware,
		allow_origins=origins,
		allow_origin_regex=r"https://.*\.vercel\.app",
		allow_credentials=True,
		allow_methods=["*"],
		allow_headers=["*"],
	)
	logger.info(f"CORS middleware added with origins: {origins} and Vercel regex")

	# Add rate limiting middleware
	if settings.ENABLE_RATE_LIMITING:
		rate_limiter = create_rate_limiter()
		app.add_middleware(rate_limiter)
		logger.info("Rate limiting middleware added")

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
	
	# Database migration status endpoint
	@app.get(
		"/health/migrations",
		tags=["health"],
		summary="Migration Status",
		description="Check database migration status",
		responses={
			200: {
				"description": "Migration status",
				"content": {
					"application/json": {
						"example": {
							"current_revision": "abc123",
							"head_revision": "abc123",
							"is_up_to_date": True,
							"pending_count": 0
						}
					}
				}
			}
		}
	)
	async def migration_status():
		"""
		Database migration status endpoint.
		
		Returns:
		- current_revision: Current database revision
		- head_revision: Latest available revision
		- is_up_to_date: Whether database is current
		- pending_count: Number of pending migrations
		"""
		from app.db.migration_manager import check_migration_status
		return check_migration_status()

	@app.post(
		"/health/migrations/run",
		tags=["health"],
		summary="Run Migrations",
		description="Manually trigger database migrations",
		responses={
			200: {
				"description": "Migration result",
				"content": {
					"application/json": {
						"example": {
							"success": True,
							"message": "Migrations applied successfully"
						}
					}
				}
			},
			500: {
				"description": "Migration failed",
				"content": {
					"application/json": {
						"example": {
							"success": False,
							"error": "Migration error details"
						}
					}
				}
			}
		}
	)
	async def run_migrations():
		"""
		Manually trigger database migrations.
		Useful if automatic migrations failed on startup.
		"""
		from app.db.migration_manager import run_migrations_on_startup
		from fastapi import HTTPException
		
		try:
			success = run_migrations_on_startup(
				auto_upgrade=True,
				wait_for_db=False, # DB should be up if app is running
				fail_on_error=True
			)
			if success:
				return {"success": True, "message": "Migrations applied successfully"}
			else:
				raise HTTPException(status_code=500, detail="Migrations failed (check logs)")
		except Exception as e:
			import traceback
			return JSONResponse(
				status_code=500,
				content={
					"success": False, 
					"error": str(e),
					"traceback": traceback.format_exc()
				}
			)

	return app


app = create_app()
