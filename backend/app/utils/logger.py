import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional
from contextvars import ContextVar

try:
	# Settings are optional to avoid import cycles during early boot
	from app.core.config import settings
except Exception:
	settings = None  # Fallback if settings are not yet available


# Context variable to store correlation IDs across a request lifecycle
correlation_id_ctx: ContextVar[Optional[str]] = ContextVar("correlation_id", default=None)


class CorrelationIdFilter(logging.Filter):
	def filter(self, record: logging.LogRecord) -> bool:
		# Attach correlation_id to the log record if present
		cid = correlation_id_ctx.get()
		record.correlation_id = cid if cid else "-"
		return True


class KeyValueFormatter(logging.Formatter):
	"""Simple key=value formatter for structured logs without extra deps."""

	def format(self, record: logging.LogRecord) -> str:
		# Base fields
		parts = [
			f"time={self.formatTime(record, datefmt='%Y-%m-%dT%H:%M:%S')}",
			f"level={record.levelname}",
			f"name={record.name}",
			f"cid={getattr(record, 'correlation_id', '-')}",
		]

		# Location info for easier debugging
		parts.append(f"module={record.module}")
		parts.append(f"func={record.funcName}")
		parts.append(f"line={record.lineno}")

		# Message
		msg = super().format(record)
		parts.append(f"msg={msg}")
		return " " .join(parts)


def _get_log_level() -> int:
	level_str = None
	if settings and hasattr(settings, "LOG_LEVEL"):
		level_str = str(getattr(settings, "LOG_LEVEL"))
	else:
		level_str = os.getenv("LOG_LEVEL", "INFO")

	return getattr(logging, level_str.upper(), logging.INFO)


def _get_log_dir() -> Path:
	# Default to backend/logs directory
	base_dir = Path(__file__).resolve().parents[2]
	default_dir = base_dir / "logs"

	log_dir = None
	if settings and hasattr(settings, "LOG_DIR"):
		log_dir = Path(getattr(settings, "LOG_DIR"))
	else:
		log_dir = Path(os.getenv("LOG_DIR", str(default_dir)))

	log_dir.mkdir(parents=True, exist_ok=True)
	return log_dir


_LOGGER_INITIALIZED = False


def init_logger() -> None:
	global _LOGGER_INITIALIZED
	if _LOGGER_INITIALIZED:
		return

	level = _get_log_level()
	log_dir = _get_log_dir()

	# Root logger configuration
	root_logger = logging.getLogger()
	root_logger.setLevel(level)

	# Common formatter and filter
	kv_formatter = KeyValueFormatter("%(message)s")
	cid_filter = CorrelationIdFilter()

	# Console handler
	console_handler = logging.StreamHandler()
	console_handler.setLevel(level)
	console_handler.setFormatter(kv_formatter)
	console_handler.addFilter(cid_filter)
	root_logger.addHandler(console_handler)

	# Rotating file handler
	file_path = log_dir / "backend.log"
	file_handler = RotatingFileHandler(
		filename=str(file_path), maxBytes=10 * 1024 * 1024, backupCount=5
	)
	file_handler.setLevel(level)
	file_handler.setFormatter(kv_formatter)
	file_handler.addFilter(cid_filter)
	root_logger.addHandler(file_handler)

	# Align uvicorn/fastapi loggers with our setup
	for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "fastapi"):
		lg = logging.getLogger(name)
		lg.setLevel(level)
		lg.propagate = True

	_LOGGER_INITIALIZED = True


def get_logger(name: Optional[str] = None) -> logging.Logger:
	"""Get a configured logger. Call `init_logger()` on first use."""
	init_logger()
	return logging.getLogger(name if name else "app")


def set_correlation_id(correlation_id: Optional[str]) -> None:
	"""Set correlation id for current context (used by middleware)."""
	correlation_id_ctx.set(correlation_id)

