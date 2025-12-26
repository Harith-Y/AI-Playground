-- PostgreSQL initialization script for AI-Playground
-- This script runs automatically when the container is first created

-- Create database (if not exists - usually created by POSTGRES_DB env var)
-- SELECT 'CREATE DATABASE aiplayground' WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'aiplayground')\gexec

-- Connect to the database
\c aiplayground;

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";  -- For text search optimization

-- Create schema for better organization (optional)
-- CREATE SCHEMA IF NOT EXISTS ml_engine;

-- Grant privileges to the application user
GRANT ALL PRIVILEGES ON DATABASE aiplayground TO aiplayground;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO aiplayground;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO aiplayground;

-- Set default privileges for future tables
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO aiplayground;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO aiplayground;

-- Create initial indexes for common queries (optional - Alembic will handle this)
-- These will be created by Alembic migrations, but we can add any custom setup here

-- Log successful initialization
DO $$
BEGIN
    RAISE NOTICE 'AI-Playground database initialized successfully';
END $$;

-- Display database info
SELECT version();
SELECT current_database();
SELECT current_user;
