@echo off
REM Database Migration Script for Windows
REM
REM Usage:
REM   migrate.bat                    # Run migrations
REM   migrate.bat status             # Show migration status
REM   migrate.bat downgrade -1       # Rollback one migration
REM   migrate.bat history            # Show migration history
REM

setlocal enabledelayedexpansion

REM Get script directory
set "SCRIPT_DIR=%~dp0"
set "BACKEND_DIR=%SCRIPT_DIR%.."

REM Configuration from environment or defaults
if not defined MIGRATION_WAIT_TIMEOUT set MIGRATION_WAIT_TIMEOUT=60
if not defined MIGRATION_FAIL_ON_ERROR set MIGRATION_FAIL_ON_ERROR=true
if not defined MIGRATION_DRY_RUN set MIGRATION_DRY_RUN=false

REM Change to backend directory
cd /d "%BACKEND_DIR%"

REM Activate virtual environment if it exists
if exist "venv\Scripts\activate.bat" (
    echo [INFO] Activating virtual environment...
    call venv\Scripts\activate.bat
)

REM Default command is 'auto' (run migrations)
set "COMMAND=%~1"
if "%COMMAND%"=="" set "COMMAND=auto"

if /i "%COMMAND%"=="auto" goto :run_migrations
if /i "%COMMAND%"=="upgrade" goto :run_migrations
if /i "%COMMAND%"=="status" goto :show_status
if /i "%COMMAND%"=="downgrade" goto :downgrade
if /i "%COMMAND%"=="history" goto :show_history
if /i "%COMMAND%"=="create" goto :create_migration
if /i "%COMMAND%"=="help" goto :show_help
if /i "%COMMAND%"=="-h" goto :show_help
if /i "%COMMAND%"=="--help" goto :show_help

echo [ERROR] Unknown command: %COMMAND%
echo Run '%~nx0 help' for usage information
exit /b 1

:run_migrations
echo [INFO] Running database migrations...
python scripts\run_migrations.py
goto :end

:show_status
echo [INFO] Checking migration status...
python -m app.db.migration_manager status
goto :end

:downgrade
set "REVISION=%~2"
if "%REVISION%"=="" set "REVISION=-1"
echo [WARN] Downgrading database to revision: %REVISION%
python -m app.db.migration_manager downgrade --revision "%REVISION%"
goto :end

:show_history
echo [INFO] Showing migration history...
python -m app.db.migration_manager history
goto :end

:create_migration
set "MESSAGE=%~2"
if "%MESSAGE%"=="" (
    echo [ERROR] Please provide a migration message
    echo Usage: %~nx0 create "migration message"
    exit /b 1
)
echo [INFO] Creating new migration: %MESSAGE%
cd /d "%BACKEND_DIR%"
alembic revision --autogenerate -m "%MESSAGE%"
goto :end

:show_help
echo Database Migration Script
echo.
echo Usage:
echo   %~nx0                      Run migrations ^(default^)
echo   %~nx0 status              Show migration status
echo   %~nx0 upgrade             Run migrations ^(alias for auto^)
echo   %~nx0 downgrade [rev]     Rollback migrations ^(default: -1^)
echo   %~nx0 history             Show migration history
echo   %~nx0 create "message"    Create new migration
echo   %~nx0 help                Show this help
echo.
echo Environment Variables:
echo   MIGRATION_WAIT_TIMEOUT       Timeout for database ^(default: 60s^)
echo   MIGRATION_FAIL_ON_ERROR      Fail on error ^(default: true^)
echo   MIGRATION_DRY_RUN            Dry run mode ^(default: false^)
echo.
goto :end

:end
endlocal
