@echo off
REM ML Platform Alerting Setup Script for Windows
REM This script helps you set up the alerting system

setlocal enabledelayedexpansion

echo ==========================================
echo ML Platform Alerting System Setup
echo ==========================================
echo.

REM Check if .env exists
if not exist "docker\alerting\.env" (
    echo Creating .env file from template...
    copy "docker\alerting\.env.example" "docker\alerting\.env" >nul
    echo [32m✓[0m Created .env file
    echo.
    echo [33m⚠️  Please edit docker\alerting\.env with your configuration:[0m
    echo    - SMTP credentials for email alerts
    echo    - Slack webhook URL ^(optional^)
    echo    - PagerDuty service key ^(optional^)
    echo    - Alert recipient emails
    echo.
    pause
) else (
    echo [32m✓[0m .env file already exists
)

echo.
echo Checking Docker...
docker --version >nul 2>&1
if errorlevel 1 (
    echo [31m❌ Docker is not installed[0m
    echo Please install Docker Desktop from https://www.docker.com/products/docker-desktop
    pause
    exit /b 1
)
echo [32m✓[0m Docker is installed

echo.
echo Checking Docker Compose...
docker-compose --version >nul 2>&1
if errorlevel 1 (
    echo [31m❌ Docker Compose is not installed[0m
    echo Please install Docker Compose
    pause
    exit /b 1
)
echo [32m✓[0m Docker Compose is installed

echo.
echo Creating ml-platform network if it doesn't exist...
docker network create ml-platform >nul 2>&1
if errorlevel 1 (
    echo [32m✓[0m Network already exists
) else (
    echo [32m✓[0m Network created
)

echo.
echo Starting alerting services...
docker-compose -f docker\docker-compose.alerting.yml up -d

echo.
echo Waiting for services to be healthy...
timeout /t 10 /nobreak >nul

echo.
echo Checking service health...

REM Check Prometheus
curl -s http://localhost:9090/-/healthy >nul 2>&1
if errorlevel 1 (
    echo [33m⚠️  Prometheus health check failed[0m
) else (
    echo [32m✓[0m Prometheus is healthy
)

REM Check AlertManager
curl -s http://localhost:9093/-/healthy >nul 2>&1
if errorlevel 1 (
    echo [33m⚠️  AlertManager health check failed[0m
) else (
    echo [32m✓[0m AlertManager is healthy
)

REM Check Grafana
curl -s http://localhost:3001/api/health >nul 2>&1
if errorlevel 1 (
    echo [33m⚠️  Grafana health check failed[0m
) else (
    echo [32m✓[0m Grafana is healthy
)

echo.
echo ==========================================
echo Setup Complete!
echo ==========================================
echo.
echo Access the services:
echo   • Prometheus:    http://localhost:9090
echo   • AlertManager:  http://localhost:9093
echo   • Grafana:       http://localhost:3001
echo.
echo Default Grafana credentials:
echo   • Username: admin
echo   • Password: admin ^(change in production!^)
echo.
echo To view logs:
echo   docker-compose -f docker\docker-compose.alerting.yml logs -f
echo.
echo To stop services:
echo   docker-compose -f docker\docker-compose.alerting.yml down
echo.
echo For more information, see docker\alerting\README.md
echo.

pause
