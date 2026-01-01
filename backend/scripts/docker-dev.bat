@echo off
REM Development Docker Helper Script for AI-Playground Backend (Windows)

set COMPOSE_FILE=docker-compose.dev.yml
set PROJECT_NAME=aiplayground

if "%1"=="" goto help
if "%1"=="help" goto help
if "%1"=="start" goto start
if "%1"=="stop" goto stop
if "%1"=="restart" goto restart
if "%1"=="rebuild" goto rebuild
if "%1"=="logs" goto logs
if "%1"=="shell" goto shell
if "%1"=="migrate" goto migrate
if "%1"=="migrate-create" goto migrate-create
if "%1"=="db-shell" goto db-shell
if "%1"=="db-backup" goto db-backup
if "%1"=="db-restore" goto db-restore
if "%1"=="status" goto status
if "%1"=="clean" goto clean
if "%1"=="clean-all" goto clean-all
if "%1"=="test" goto test
if "%1"=="lint" goto lint
if "%1"=="format" goto format
if "%1"=="init" goto init
goto help

:start
echo Starting development environment...
docker-compose -f %COMPOSE_FILE% up -d
echo.
echo [SUCCESS] Services started!
echo Backend: http://localhost:8000
echo API Docs: http://localhost:8000/docs
echo Flower: http://localhost:5555
echo pgAdmin: http://localhost:5050
goto end

:stop
echo Stopping development environment...
docker-compose -f %COMPOSE_FILE% down
echo [SUCCESS] Services stopped!
goto end

:restart
echo Restarting development environment...
docker-compose -f %COMPOSE_FILE% restart
echo [SUCCESS] Services restarted!
goto end

:rebuild
echo Rebuilding and restarting services...
docker-compose -f %COMPOSE_FILE% up -d --build
echo [SUCCESS] Services rebuilt and restarted!
goto end

:logs
set SERVICE=%2
if "%SERVICE%"=="" set SERVICE=backend
echo Showing logs for %SERVICE% (Ctrl+C to exit)...
docker-compose -f %COMPOSE_FILE% logs -f %SERVICE%
goto end

:shell
set SERVICE=%2
if "%SERVICE%"=="" set SERVICE=backend
echo Opening shell in %SERVICE%...
docker-compose -f %COMPOSE_FILE% exec %SERVICE% bash
goto end

:migrate
echo Running database migrations...
docker-compose -f %COMPOSE_FILE% exec backend alembic upgrade head
echo [SUCCESS] Migrations completed!
goto end

:migrate-create
if "%2"=="" (
    echo [ERROR] Please provide migration message
    echo Usage: %0 migrate-create "migration message"
    goto end
)
echo Creating new migration: %2
docker-compose -f %COMPOSE_FILE% exec backend alembic revision --autogenerate -m %2
echo [SUCCESS] Migration created!
goto end

:db-shell
echo Opening PostgreSQL shell...
docker-compose -f %COMPOSE_FILE% exec postgres psql -U aiplayground -d aiplayground
goto end

:db-backup
for /f "tokens=2-4 delims=/ " %%a in ('date /t') do (set mydate=%%c%%a%%b)
for /f "tokens=1-2 delims=/:" %%a in ('time /t') do (set mytime=%%a%%b)
set BACKUP_FILE=backup_%mydate%_%mytime%.sql
echo Creating database backup: %BACKUP_FILE%
docker-compose -f %COMPOSE_FILE% exec -T postgres pg_dump -U aiplayground aiplayground > %BACKUP_FILE%
echo [SUCCESS] Backup created: %BACKUP_FILE%
goto end

:db-restore
if "%2"=="" (
    echo [ERROR] Please provide backup file
    echo Usage: %0 db-restore backup.sql
    goto end
)
echo Restoring database from: %2
docker-compose -f %COMPOSE_FILE% exec -T postgres psql -U aiplayground aiplayground < %2
echo [SUCCESS] Database restored!
goto end

:status
echo Service status:
docker-compose -f %COMPOSE_FILE% ps
goto end

:clean
echo Cleaning up containers and images...
docker-compose -f %COMPOSE_FILE% down --rmi local
echo [SUCCESS] Cleanup complete!
goto end

:clean-all
set /p CONFIRM="This will remove all containers, volumes, and data. Continue? (y/N): "
if /i "%CONFIRM%"=="y" (
    echo Removing all containers, volumes, and images...
    docker-compose -f %COMPOSE_FILE% down -v --rmi all
    echo [SUCCESS] Full cleanup complete!
) else (
    echo Cancelled
)
goto end

:test
echo Running tests...
docker-compose -f %COMPOSE_FILE% exec backend pytest %2 %3 %4 %5 %6 %7 %8 %9
goto end

:lint
echo Running linters...
docker-compose -f %COMPOSE_FILE% exec backend black app --check
docker-compose -f %COMPOSE_FILE% exec backend flake8 app
docker-compose -f %COMPOSE_FILE% exec backend mypy app
echo [SUCCESS] Linting complete!
goto end

:format
echo Formatting code...
docker-compose -f %COMPOSE_FILE% exec backend black app
echo [SUCCESS] Code formatted!
goto end

:init
echo Initializing development environment...

REM Check if .env exists
if not exist "..\\.env" (
    echo Creating .env from template...
    copy "..\.env.docker" "..\\.env"
    echo [SUCCESS] .env created
)

REM Start services
echo Starting services...
docker-compose -f %COMPOSE_FILE% up -d

REM Wait for database
echo Waiting for database...
timeout /t 10 /nobreak > nul

REM Run migrations
echo Running migrations...
docker-compose -f %COMPOSE_FILE% exec backend alembic upgrade head

REM Initialize database
echo Initializing database...
docker-compose -f %COMPOSE_FILE% exec backend python init_db.py

echo.
echo [SUCCESS] Development environment initialized!
echo Backend: http://localhost:8000
echo API Docs: http://localhost:8000/docs
goto end

:help
echo AI-Playground Backend - Development Docker Helper (Windows)
echo.
echo Usage: %0 ^<command^> [options]
echo.
echo Commands:
echo   init              Initialize development environment
echo   start             Start all services
echo   stop              Stop all services
echo   restart           Restart all services
echo   rebuild           Rebuild and restart services
echo   status            Show service status
echo.
echo   logs [service]    Show logs (default: backend)
echo   shell [service]   Open shell in service (default: backend)
echo.
echo   migrate           Run database migrations
echo   migrate-create "msg"  Create new migration
echo.
echo   db-shell          Open PostgreSQL shell
echo   db-backup         Create database backup
echo   db-restore file   Restore database from backup
echo.
echo   test [args]       Run tests
echo   lint              Run linters
echo   format            Format code with black
echo.
echo   clean             Clean up containers and images
echo   clean-all         Remove everything including volumes
echo.
echo   help              Show this help message
echo.
echo Examples:
echo   %0 init                          # First time setup
echo   %0 start                         # Start services
echo   %0 logs backend                  # View backend logs
echo   %0 migrate-create "add users"    # Create migration
echo   %0 test tests/test_api.py        # Run specific test

:end
