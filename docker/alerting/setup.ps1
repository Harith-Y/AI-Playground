# ML Platform Alerting Setup Script for Windows (PowerShell)
# This script helps you set up the alerting system

$ErrorActionPreference = "Stop"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "ML Platform Alerting System Setup" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Check if .env exists
if (-not (Test-Path "docker\alerting\.env")) {
    Write-Host "Creating .env file from template..." -ForegroundColor Yellow
    Copy-Item "docker\alerting\.env.example" "docker\alerting\.env"
    Write-Host "✓ Created .env file" -ForegroundColor Green
    Write-Host ""
    Write-Host "⚠️  Please edit docker\alerting\.env with your configuration:" -ForegroundColor Yellow
    Write-Host "   - SMTP credentials for email alerts"
    Write-Host "   - Slack webhook URL (optional)"
    Write-Host "   - PagerDuty service key (optional)"
    Write-Host "   - Alert recipient emails"
    Write-Host ""
    Read-Host "Press Enter after configuring .env file"
} else {
    Write-Host "✓ .env file already exists" -ForegroundColor Green
}

Write-Host ""
Write-Host "Checking Docker..." -ForegroundColor Cyan
try {
    $dockerVersion = docker --version
    Write-Host "✓ Docker is installed: $dockerVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Docker is not installed" -ForegroundColor Red
    Write-Host "Please install Docker Desktop from https://www.docker.com/products/docker-desktop" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "Checking Docker Compose..." -ForegroundColor Cyan
try {
    $composeVersion = docker-compose --version
    Write-Host "✓ Docker Compose is installed: $composeVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Docker Compose is not installed" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "Creating ml-platform network if it doesn't exist..." -ForegroundColor Cyan
try {
    docker network create ml-platform 2>$null
    Write-Host "✓ Network created" -ForegroundColor Green
} catch {
    Write-Host "✓ Network already exists" -ForegroundColor Green
}

Write-Host ""
Write-Host "Starting alerting services..." -ForegroundColor Cyan
docker-compose -f docker\docker-compose.alerting.yml up -d

Write-Host ""
Write-Host "Waiting for services to be healthy..." -ForegroundColor Cyan
Start-Sleep -Seconds 10

Write-Host ""
Write-Host "Checking service health..." -ForegroundColor Cyan

# Check Prometheus
try {
    $response = Invoke-WebRequest -Uri "http://localhost:9090/-/healthy" -UseBasicParsing -TimeoutSec 5 -ErrorAction Stop
    Write-Host "✓ Prometheus is healthy" -ForegroundColor Green
} catch {
    Write-Host "⚠️  Prometheus health check failed" -ForegroundColor Yellow
}

# Check AlertManager
try {
    $response = Invoke-WebRequest -Uri "http://localhost:9093/-/healthy" -UseBasicParsing -TimeoutSec 5 -ErrorAction Stop
    Write-Host "✓ AlertManager is healthy" -ForegroundColor Green
} catch {
    Write-Host "⚠️  AlertManager health check failed" -ForegroundColor Yellow
}

# Check Grafana
try {
    $response = Invoke-WebRequest -Uri "http://localhost:3001/api/health" -UseBasicParsing -TimeoutSec 5 -ErrorAction Stop
    Write-Host "✓ Grafana is healthy" -ForegroundColor Green
} catch {
    Write-Host "⚠️  Grafana health check failed" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Access the services:" -ForegroundColor White
Write-Host "  • Prometheus:    http://localhost:9090" -ForegroundColor Cyan
Write-Host "  • AlertManager:  http://localhost:9093" -ForegroundColor Cyan
Write-Host "  • Grafana:       http://localhost:3001" -ForegroundColor Cyan
Write-Host ""
Write-Host "Default Grafana credentials:" -ForegroundColor White
Write-Host "  • Username: admin" -ForegroundColor Yellow
Write-Host "  • Password: admin (change in production!)" -ForegroundColor Yellow
Write-Host ""
Write-Host "To view logs:" -ForegroundColor White
Write-Host "  docker-compose -f docker\docker-compose.alerting.yml logs -f" -ForegroundColor Gray
Write-Host ""
Write-Host "To stop services:" -ForegroundColor White
Write-Host "  docker-compose -f docker\docker-compose.alerting.yml down" -ForegroundColor Gray
Write-Host ""
Write-Host "For more information, see docker\alerting\README.md" -ForegroundColor White
Write-Host ""

Read-Host "Press Enter to exit"
