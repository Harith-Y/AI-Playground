@echo off
REM AI-Playground Docker Quick Start Script for Windows
REM This script helps you quickly start the Docker environment

setlocal enabledelayedexpansion

echo.
echo ========================================
echo   AI-Playground Docker Quick Start
echo ========================================
echo.

REM Check if Docker is installed
docker --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker is not installed. Please install Docker Desktop first.
    echo Download from: https://www.docker.com/products/docker-desktop
    pause
    exit /b 1
)

REM Check if Docker Compose is installed
docker-compose --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker Compose is not installed.
    pause
    exit /b 1
)

echo [OK] Docker and Docker Compose are installed
echo.

REM Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker is not running. Please start Docker Desktop.
    pause
    exit /b 1
)

echo [OK] Docker is running
echo.

:menu
cls
echo.
echo ========================================
echo   AI-Playground Docker Quick Start
echo ========================================
echo.
echo What would you like to do?
echo.
echo 1. Build all services
echo 2. Start all services
echo 3. Build and start all services
echo 4. Stop all services
echo 5. Stop and remove all data (clean slate)
echo 6. View service status
echo 7. View logs
echo 8. Run database migrations
echo 9. Test services
echo 0. Exit
echo.

set /p choice="Enter your choice: "

if "%choice%"=="1" goto build
if "%choice%"=="2" goto start
if "%choice%"=="3" goto build_and_start
if "%choice%"=="4" goto stop
if "%choice%"=="5" goto clean
if "%choice%"=="6" goto status
if "%choice%"=="7" goto logs
if "%choice%"=="8" goto migrate
if "%choice%"=="9" goto test
if "%choice%"=="0" goto exit
echo Invalid choice. Please try again.
pause
goto menu

:build
echo.
echo Building Docker images...
docker-compose build
echo.
echo [OK] Build complete!
echo.
pause
goto menu

:start
echo.
echo Starting services...
docker-compose up -d
echo.
echo [OK] Services started!
echo.
echo Access points:
echo   - Frontend:  http://localhost:3000
echo   - Backend:   http://localhost:8000
echo   - API Docs:  http://localhost:8000/docs
echo.
pause
goto menu

:build_and_start
echo.
echo Building Docker images...
docker-compose build
echo.
echo Starting services...
docker-compose up -d
echo.
echo [OK] Services started!
echo.
echo Access points:
echo   - Frontend:  http://localhost:3000
echo   - Backend:   http://localhost:8000
echo   - API Docs:  http://localhost:8000/docs
echo.
pause
goto menu

:stop
echo.
echo Stopping services...
docker-compose down
echo.
echo [OK] Services stopped!
echo.
pause
goto menu

:clean
echo.
echo [WARNING] This will delete all data (database, uploads, etc.)
set /p confirm="Are you sure? (yes/no): "
if /i "%confirm%"=="yes" (
    echo.
    echo Removing all services and volumes...
    docker-compose down -v
    echo.
    echo [OK] Clean slate complete!
) else (
    echo Cancelled.
)
echo.
pause
goto menu

:status
echo.
echo Service Status:
echo.
docker-compose ps
echo.
pause
goto menu

:logs
echo.
echo Which service logs would you like to view?
echo 1. All services
echo 2. Backend
echo 3. Frontend
echo 4. Celery Worker
echo 5. PostgreSQL
echo 6. Redis
echo.
set /p log_choice="Enter choice: "

if "%log_choice%"=="1" docker-compose logs -f
if "%log_choice%"=="2" docker-compose logs -f backend
if "%log_choice%"=="3" docker-compose logs -f frontend
if "%log_choice%"=="4" docker-compose logs -f celery-worker
if "%log_choice%"=="5" docker-compose logs -f postgres
if "%log_choice%"=="6" docker-compose logs -f redis

goto menu

:migrate
echo.
echo Running database migrations...
docker-compose exec backend alembic upgrade head
echo.
echo [OK] Migrations complete!
echo.
pause
goto menu

:test
echo.
echo Testing services...
echo.

REM Test backend health
echo Testing Backend health check...
curl -f http://localhost:8000/health >nul 2>&1
if errorlevel 1 (
    echo [FAIL] Backend health check failed
) else (
    echo [OK] Backend is healthy
)

REM Test frontend
echo Testing Frontend...
curl -f http://localhost:3000 >nul 2>&1
if errorlevel 1 (
    echo [FAIL] Frontend check failed
) else (
    echo [OK] Frontend is accessible
)

REM Test PostgreSQL
echo Testing PostgreSQL...
docker-compose exec -T postgres pg_isready -U aiplayground >nul 2>&1
if errorlevel 1 (
    echo [FAIL] PostgreSQL check failed
) else (
    echo [OK] PostgreSQL is ready
)

REM Test Redis
echo Testing Redis...
docker-compose exec -T redis redis-cli ping >nul 2>&1
if errorlevel 1 (
    echo [FAIL] Redis check failed
) else (
    echo [OK] Redis is ready
)

echo.
pause
goto menu

:exit
echo.
echo Goodbye!
exit /b 0
