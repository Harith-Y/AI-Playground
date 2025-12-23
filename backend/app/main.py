from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from time import perf_counter
from uuid import uuid4

from app.core.config import settings
from app.core.exceptions import register_exception_handlers
from app.utils.logger import get_logger, set_correlation_id


def create_app() -> FastAPI:
	app = FastAPI(
		title=settings.PROJECT_NAME,
		version=settings.VERSION,
		docs_url="/docs" if settings.DEBUG else None,
		redoc_url="/redoc" if settings.DEBUG else None,
	)

	logger = get_logger("backend")

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

	# Basic health endpoint
	@app.get("/health")
	async def health():
		return {"status": "ok", "version": settings.VERSION}

	return app


app = create_app()

