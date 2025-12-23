#!/bin/bash

# AI-Playground Docker Quick Start Script
# This script helps you quickly start the Docker environment

set -e

echo "🚀 AI-Playground Docker Quick Start"
echo "===================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker is not installed. Please install Docker first.${NC}"
    echo "   Download from: https://www.docker.com/products/docker-desktop"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}❌ Docker Compose is not installed. Please install Docker Compose first.${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Docker and Docker Compose are installed${NC}"
echo ""

# Check if Docker is running
if ! docker info &> /dev/null; then
    echo -e "${RED}❌ Docker is not running. Please start Docker Desktop.${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Docker is running${NC}"
echo ""

# Function to show menu
show_menu() {
    echo "What would you like to do?"
    echo ""
    echo "1) 🏗️  Build all services"
    echo "2) 🚀 Start all services"
    echo "3) 🔄 Build and start all services"
    echo "4) 🛑 Stop all services"
    echo "5) 🗑️  Stop and remove all data (clean slate)"
    echo "6) 📊 View service status"
    echo "7) 📝 View logs"
    echo "8) 🔧 Run database migrations"
    echo "9) 🧪 Test services"
    echo "0) ❌ Exit"
    echo ""
}

# Function to build services
build_services() {
    echo -e "${YELLOW}Building Docker images...${NC}"
    docker-compose build
    echo -e "${GREEN}✅ Build complete!${NC}"
}

# Function to start services
start_services() {
    echo -e "${YELLOW}Starting services...${NC}"
    docker-compose up -d
    echo ""
    echo -e "${GREEN}✅ Services started!${NC}"
    echo ""
    echo "Access points:"
    echo "  - Frontend:  http://localhost:3000"
    echo "  - Backend:   http://localhost:8000"
    echo "  - API Docs:  http://localhost:8000/docs"
    echo ""
}

# Function to stop services
stop_services() {
    echo -e "${YELLOW}Stopping services...${NC}"
    docker-compose down
    echo -e "${GREEN}✅ Services stopped!${NC}"
}

# Function to clean slate
clean_slate() {
    echo -e "${RED}⚠️  WARNING: This will delete all data (database, uploads, etc.)${NC}"
    read -p "Are you sure? (yes/no): " confirm
    if [ "$confirm" = "yes" ]; then
        echo -e "${YELLOW}Removing all services and volumes...${NC}"
        docker-compose down -v
        echo -e "${GREEN}✅ Clean slate complete!${NC}"
    else
        echo "Cancelled."
    fi
}

# Function to view status
view_status() {
    echo -e "${YELLOW}Service Status:${NC}"
    echo ""
    docker-compose ps
}

# Function to view logs
view_logs() {
    echo "Which service logs would you like to view?"
    echo "1) All services"
    echo "2) Backend"
    echo "3) Frontend"
    echo "4) Celery Worker"
    echo "5) PostgreSQL"
    echo "6) Redis"
    read -p "Enter choice: " log_choice
    
    case $log_choice in
        1) docker-compose logs -f ;;
        2) docker-compose logs -f backend ;;
        3) docker-compose logs -f frontend ;;
        4) docker-compose logs -f celery-worker ;;
        5) docker-compose logs -f postgres ;;
        6) docker-compose logs -f redis ;;
        *) echo "Invalid choice" ;;
    esac
}

# Function to run migrations
run_migrations() {
    echo -e "${YELLOW}Running database migrations...${NC}"
    docker-compose exec backend alembic upgrade head
    echo -e "${GREEN}✅ Migrations complete!${NC}"
}

# Function to test services
test_services() {
    echo -e "${YELLOW}Testing services...${NC}"
    echo ""
    
    # Test backend health
    echo -n "Backend health check: "
    if curl -f http://localhost:8000/health &> /dev/null; then
        echo -e "${GREEN}✅ OK${NC}"
    else
        echo -e "${RED}❌ Failed${NC}"
    fi
    
    # Test frontend
    echo -n "Frontend check: "
    if curl -f http://localhost:3000 &> /dev/null; then
        echo -e "${GREEN}✅ OK${NC}"
    else
        echo -e "${RED}❌ Failed${NC}"
    fi
    
    # Test PostgreSQL
    echo -n "PostgreSQL check: "
    if docker-compose exec -T postgres pg_isready -U aiplayground &> /dev/null; then
        echo -e "${GREEN}✅ OK${NC}"
    else
        echo -e "${RED}❌ Failed${NC}"
    fi
    
    # Test Redis
    echo -n "Redis check: "
    if docker-compose exec -T redis redis-cli ping &> /dev/null; then
        echo -e "${GREEN}✅ OK${NC}"
    else
        echo -e "${RED}❌ Failed${NC}"
    fi
    
    echo ""
}

# Main loop
while true; do
    show_menu
    read -p "Enter your choice: " choice
    echo ""
    
    case $choice in
        1) build_services ;;
        2) start_services ;;
        3) build_services && start_services ;;
        4) stop_services ;;
        5) clean_slate ;;
        6) view_status ;;
        7) view_logs ;;
        8) run_migrations ;;
        9) test_services ;;
        0) echo "Goodbye! 👋"; exit 0 ;;
        *) echo -e "${RED}Invalid choice. Please try again.${NC}" ;;
    esac
    
    echo ""
    read -p "Press Enter to continue..."
    clear
done
