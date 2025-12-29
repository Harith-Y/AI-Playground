#!/bin/bash

# ML Platform Alerting Setup Script
# This script helps you set up the alerting system

set -e

echo "=========================================="
echo "ML Platform Alerting System Setup"
echo "=========================================="
echo ""

# Check if .env exists
if [ ! -f "docker/alerting/.env" ]; then
    echo "Creating .env file from template..."
    cp docker/alerting/.env.example docker/alerting/.env
    echo "✓ Created .env file"
    echo ""
    echo "⚠️  Please edit docker/alerting/.env with your configuration:"
    echo "   - SMTP credentials for email alerts"
    echo "   - Slack webhook URL (optional)"
    echo "   - PagerDuty service key (optional)"
    echo "   - Alert recipient emails"
    echo ""
    read -p "Press Enter after configuring .env file..."
else
    echo "✓ .env file already exists"
fi

echo ""
echo "Checking Docker..."
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed"
    exit 1
fi
echo "✓ Docker is installed"

echo ""
echo "Checking Docker Compose..."
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed"
    exit 1
fi
echo "✓ Docker Compose is installed"

echo ""
echo "Creating ml-platform network if it doesn't exist..."
docker network create ml-platform 2>/dev/null || echo "✓ Network already exists"

echo ""
echo "Starting alerting services..."
docker-compose -f docker/docker-compose.alerting.yml up -d

echo ""
echo "Waiting for services to be healthy..."
sleep 10

echo ""
echo "Checking service health..."

# Check Prometheus
if curl -s http://localhost:9090/-/healthy > /dev/null; then
    echo "✓ Prometheus is healthy"
else
    echo "⚠️  Prometheus health check failed"
fi

# Check AlertManager
if curl -s http://localhost:9093/-/healthy > /dev/null; then
    echo "✓ AlertManager is healthy"
else
    echo "⚠️  AlertManager health check failed"
fi

# Check Grafana
if curl -s http://localhost:3001/api/health > /dev/null; then
    echo "✓ Grafana is healthy"
else
    echo "⚠️  Grafana health check failed"
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Access the services:"
echo "  • Prometheus:    http://localhost:9090"
echo "  • AlertManager:  http://localhost:9093"
echo "  • Grafana:       http://localhost:3001"
echo ""
echo "Default Grafana credentials:"
echo "  • Username: admin"
echo "  • Password: admin (change in production!)"
echo ""
echo "To view logs:"
echo "  docker-compose -f docker/docker-compose.alerting.yml logs -f"
echo ""
echo "To stop services:"
echo "  docker-compose -f docker/docker-compose.alerting.yml down"
echo ""
echo "For more information, see docker/alerting/README.md"
echo ""
