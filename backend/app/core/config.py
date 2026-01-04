"""
Application configuration settings
Automatically loads from .env file in the backend directory
"""
from pydantic_settings import BaseSettings
from typing import Optional
import os
from pathlib import Path

# Get the backend directory path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
ENV_FILE = BASE_DIR / ".env"


class Settings(BaseSettings):
    # API Settings
    API_V1_PREFIX: str = "/api/v1"
    PROJECT_NAME: str = "AI-Playground"
    VERSION: str = "1.0.0"
    
    # Database (defaults shown - will be overridden by .env)
    DATABASE_URL: str = "postgresql://user:pass@localhost:5432/aiplayground"
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # Celery
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/0"
    
    # Security
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # File Storage
    UPLOAD_DIR: str = str(BASE_DIR / "uploads")
    MAX_UPLOAD_SIZE: int = 104857600  # 100MB
    
    # Environment
    ENVIRONMENT: str = "development"
    DEBUG: bool = True

    # Logging
    LOG_LEVEL: str = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    LOG_DIR: str = "./backend/logs"  # Relative to project root by default
    
    # CORS (for frontend access)
    ALLOWED_ORIGINS: str = "http://localhost:5173,http://localhost:3000"
    
    # Rate Limiting
    ENABLE_RATE_LIMITING: bool = False
    RATE_LIMIT_PER_MINUTE: int = 60
    RATE_LIMIT_PER_HOUR: int = 1000
    
    # Cloudflare R2 Storage (S3-compatible)
    R2_ACCOUNT_ID: Optional[str] = None
    R2_ACCESS_KEY_ID: Optional[str] = None
    R2_SECRET_ACCESS_KEY: Optional[str] = None
    R2_BUCKET_NAME: str = "aiplayground-storage"
    R2_PUBLIC_URL: Optional[str] = None
    USE_R2_STORAGE: bool = False  # Set to True to use R2 instead of local filesystem
    
    class Config:
        env_file = str(ENV_FILE)
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "ignore"  # Ignore extra fields in .env


# Initialize settings - will automatically load from .env
settings = Settings()

# Verify .env was loaded (optional - for debugging)
if not ENV_FILE.exists():
    print(f"Warning: .env file not found at {ENV_FILE}")
    print(f"Using default configuration values")
else:
    print(f"Configuration loaded from {ENV_FILE}")
