from typing import Optional, Any, Dict
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi import HTTPException
from pydantic import BaseModel
import logging


class AppError(Exception):
	"""Generic application error carrying status and optional code."""

	def __init__(self, message: str, status_code: int = 400, error_code: Optional[str] = None):
		super().__init__(message)
		self.message = message
		self.status_code = status_code
		self.error_code = error_code


class ErrorResponse(BaseModel):
	detail: str
	error_code: Optional[str] = None
	trace_id: Optional[str] = None
	context: Optional[Dict[str, Any]] = None


def register_exception_handlers(app: FastAPI, logger: logging.Logger) -> None:
	"""Attach global exception handlers to the FastAPI app."""

	@app.exception_handler(HTTPException)
	async def http_exception_handler(request: Request, exc: HTTPException):
		trace_id = getattr(request.state, "correlation_id", None)
		level = logging.WARNING if 400 <= exc.status_code < 500 else logging.ERROR
		logger.log(level, f"HTTPException status={exc.status_code} path={request.url.path} msg={exc.detail}")
		return JSONResponse(
			status_code=exc.status_code,
			content=ErrorResponse(detail=exc.detail, trace_id=trace_id).model_dump(),
		)

	@app.exception_handler(RequestValidationError)
	async def validation_exception_handler(request: Request, exc: RequestValidationError):
		trace_id = getattr(request.state, "correlation_id", None)
		logger.warning(f"ValidationError path={request.url.path} errors={exc.errors()}")
		return JSONResponse(
			status_code=422,
			content=ErrorResponse(detail="Validation error", trace_id=trace_id, context={"errors": exc.errors()}).model_dump(),
		)

	@app.exception_handler(AppError)
	async def app_error_handler(request: Request, exc: AppError):
		trace_id = getattr(request.state, "correlation_id", None)
		logger.warning(f"AppError status={exc.status_code} code={exc.error_code} path={request.url.path} msg={exc.message}")
		return JSONResponse(
			status_code=exc.status_code,
			content=ErrorResponse(detail=exc.message, error_code=exc.error_code, trace_id=trace_id).model_dump(),
		)

	@app.exception_handler(Exception)
	async def unhandled_exception_handler(request: Request, exc: Exception):
		trace_id = getattr(request.state, "correlation_id", None)
		logger.error(f"UnhandledException path={request.url.path}", exc_info=exc)
		return JSONResponse(
			status_code=500,
			content=ErrorResponse(detail="Internal Server Error", trace_id=trace_id).model_dump(),
		)

