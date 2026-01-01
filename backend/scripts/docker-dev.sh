#!/bin/bash
# Development Docker Helper Script for AI-Playground Backend

set -e

COMPOSE_FILE="docker-compose.dev.yml"
PROJECT_NAME="aiplayground"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper functions
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

# Main commands
case "${1}" in
    start)
        print_info "Starting development environment..."
        docker-compose -f "$COMPOSE_FILE" up -d
        print_success "Services started!"
        print_info "Backend: http://localhost:8000"
        print_info "API Docs: http://localhost:8000/docs"
        print_info "Flower: http://localhost:5555"
        print_info "pgAdmin: http://localhost:5050"
        ;;

    stop)
        print_info "Stopping development environment..."
        docker-compose -f "$COMPOSE_FILE" down
        print_success "Services stopped!"
        ;;

    restart)
        print_info "Restarting development environment..."
        docker-compose -f "$COMPOSE_FILE" restart
        print_success "Services restarted!"
        ;;

    rebuild)
        print_info "Rebuilding and restarting services..."
        docker-compose -f "$COMPOSE_FILE" up -d --build
        print_success "Services rebuilt and restarted!"
        ;;

    logs)
        SERVICE="${2:-backend}"
        print_info "Showing logs for $SERVICE (Ctrl+C to exit)..."
        docker-compose -f "$COMPOSE_FILE" logs -f "$SERVICE"
        ;;

    shell)
        SERVICE="${2:-backend}"
        print_info "Opening shell in $SERVICE..."
        docker-compose -f "$COMPOSE_FILE" exec "$SERVICE" bash
        ;;

    migrate)
        print_info "Running database migrations..."
        docker-compose -f "$COMPOSE_FILE" exec backend alembic upgrade head
        print_success "Migrations completed!"
        ;;

    migrate-create)
        if [ -z "$2" ]; then
            print_error "Please provide migration message"
            echo "Usage: $0 migrate-create 'migration message'"
            exit 1
        fi
        print_info "Creating new migration: $2"
        docker-compose -f "$COMPOSE_FILE" exec backend alembic revision --autogenerate -m "$2"
        print_success "Migration created!"
        ;;

    db-shell)
        print_info "Opening PostgreSQL shell..."
        docker-compose -f "$COMPOSE_FILE" exec postgres psql -U aiplayground -d aiplayground
        ;;

    db-backup)
        BACKUP_FILE="backup_$(date +%Y%m%d_%H%M%S).sql"
        print_info "Creating database backup: $BACKUP_FILE"
        docker-compose -f "$COMPOSE_FILE" exec -T postgres pg_dump -U aiplayground aiplayground > "$BACKUP_FILE"
        print_success "Backup created: $BACKUP_FILE"
        ;;

    db-restore)
        if [ -z "$2" ]; then
            print_error "Please provide backup file"
            echo "Usage: $0 db-restore backup.sql"
            exit 1
        fi
        print_info "Restoring database from: $2"
        docker-compose -f "$COMPOSE_FILE" exec -T postgres psql -U aiplayground aiplayground < "$2"
        print_success "Database restored!"
        ;;

    status)
        print_info "Service status:"
        docker-compose -f "$COMPOSE_FILE" ps
        ;;

    clean)
        print_info "Cleaning up containers and images..."
        docker-compose -f "$COMPOSE_FILE" down --rmi local
        print_success "Cleanup complete!"
        ;;

    clean-all)
        read -p "This will remove all containers, volumes, and data. Continue? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_info "Removing all containers, volumes, and images..."
            docker-compose -f "$COMPOSE_FILE" down -v --rmi all
            print_success "Full cleanup complete!"
        else
            print_info "Cancelled"
        fi
        ;;

    test)
        print_info "Running tests..."
        docker-compose -f "$COMPOSE_FILE" exec backend pytest "${@:2}"
        ;;

    lint)
        print_info "Running linters..."
        docker-compose -f "$COMPOSE_FILE" exec backend black app --check
        docker-compose -f "$COMPOSE_FILE" exec backend flake8 app
        docker-compose -f "$COMPOSE_FILE" exec backend mypy app
        print_success "Linting complete!"
        ;;

    format)
        print_info "Formatting code..."
        docker-compose -f "$COMPOSE_FILE" exec backend black app
        print_success "Code formatted!"
        ;;

    init)
        print_info "Initializing development environment..."

        # Check if .env exists
        if [ ! -f "../.env" ]; then
            print_info "Creating .env from template..."
            cp ../.env.docker ../.env
            print_success ".env created"
        fi

        # Start services
        print_info "Starting services..."
        docker-compose -f "$COMPOSE_FILE" up -d

        # Wait for database
        print_info "Waiting for database..."
        sleep 10

        # Run migrations
        print_info "Running migrations..."
        docker-compose -f "$COMPOSE_FILE" exec backend alembic upgrade head

        # Initialize database
        print_info "Initializing database..."
        docker-compose -f "$COMPOSE_FILE" exec backend python init_db.py || true

        print_success "Development environment initialized!"
        print_info "Backend: http://localhost:8000"
        print_info "API Docs: http://localhost:8000/docs"
        ;;

    help|*)
        echo "AI-Playground Backend - Development Docker Helper"
        echo ""
        echo "Usage: $0 <command> [options]"
        echo ""
        echo "Commands:"
        echo "  init              Initialize development environment"
        echo "  start             Start all services"
        echo "  stop              Stop all services"
        echo "  restart           Restart all services"
        echo "  rebuild           Rebuild and restart services"
        echo "  status            Show service status"
        echo ""
        echo "  logs [service]    Show logs (default: backend)"
        echo "  shell [service]   Open shell in service (default: backend)"
        echo ""
        echo "  migrate           Run database migrations"
        echo "  migrate-create 'msg'  Create new migration"
        echo ""
        echo "  db-shell          Open PostgreSQL shell"
        echo "  db-backup         Create database backup"
        echo "  db-restore file   Restore database from backup"
        echo ""
        echo "  test [args]       Run tests"
        echo "  lint              Run linters"
        echo "  format            Format code with black"
        echo ""
        echo "  clean             Clean up containers and images"
        echo "  clean-all         Remove everything including volumes"
        echo ""
        echo "  help              Show this help message"
        echo ""
        echo "Examples:"
        echo "  $0 init                          # First time setup"
        echo "  $0 start                         # Start services"
        echo "  $0 logs backend                  # View backend logs"
        echo "  $0 migrate-create 'add users'    # Create migration"
        echo "  $0 test tests/test_api.py        # Run specific test"
        ;;
esac
