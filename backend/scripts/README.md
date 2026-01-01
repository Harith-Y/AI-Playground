# Docker Helper Scripts

Quick access scripts for managing the AI-Playground backend Docker environment.

## Scripts

### `docker-dev.sh` (Linux/Mac)
Bash script for development Docker operations.

**Make executable:**
```bash
chmod +x scripts/docker-dev.sh
```

**Usage:**
```bash
./scripts/docker-dev.sh <command>
```

### `docker-dev.bat` (Windows)
Batch script for development Docker operations on Windows.

**Usage:**
```cmd
scripts\docker-dev.bat <command>
```

## Quick Commands

### First Time Setup
```bash
# Linux/Mac
./scripts/docker-dev.sh init

# Windows
scripts\docker-dev.bat init
```

### Daily Development
```bash
# Start services
./scripts/docker-dev.sh start

# View logs
./scripts/docker-dev.sh logs backend

# Run migrations
./scripts/docker-dev.sh migrate

# Stop services
./scripts/docker-dev.sh stop
```

### Database Operations
```bash
# Create backup
./scripts/docker-dev.sh db-backup

# Restore backup
./scripts/docker-dev.sh db-restore backup_20260101_120000.sql

# Access database shell
./scripts/docker-dev.sh db-shell
```

### Testing & Quality
```bash
# Run tests
./scripts/docker-dev.sh test

# Lint code
./scripts/docker-dev.sh lint

# Format code
./scripts/docker-dev.sh format
```

## Available Commands

| Command | Description |
|---------|-------------|
| `init` | Initialize development environment (first time) |
| `start` | Start all services |
| `stop` | Stop all services |
| `restart` | Restart all services |
| `rebuild` | Rebuild and restart services |
| `status` | Show service status |
| `logs [service]` | Show logs for service (default: backend) |
| `shell [service]` | Open shell in service (default: backend) |
| `migrate` | Run database migrations |
| `migrate-create "msg"` | Create new migration |
| `db-shell` | Open PostgreSQL shell |
| `db-backup` | Create database backup |
| `db-restore <file>` | Restore database from backup |
| `test [args]` | Run tests |
| `lint` | Run linters (black, flake8, mypy) |
| `format` | Format code with black |
| `clean` | Clean up containers and images |
| `clean-all` | Remove everything including volumes |
| `help` | Show help message |

## Tips

### Running Specific Services

View logs for a specific service:
```bash
./scripts/docker-dev.sh logs redis
./scripts/docker-dev.sh logs celery-worker
./scripts/docker-dev.sh logs postgres
```

Open shell in a service:
```bash
./scripts/docker-dev.sh shell backend
./scripts/docker-dev.sh shell postgres
```

### Testing

Run all tests:
```bash
./scripts/docker-dev.sh test
```

Run specific test file:
```bash
./scripts/docker-dev.sh test tests/test_api.py
```

Run with coverage:
```bash
./scripts/docker-dev.sh test --cov=app --cov-report=html
```

### Database Migrations

Create a new migration after model changes:
```bash
./scripts/docker-dev.sh migrate-create "add user profile table"
```

Apply migrations:
```bash
./scripts/docker-dev.sh migrate
```

### Cleanup

Remove containers and local images:
```bash
./scripts/docker-dev.sh clean
```

Full cleanup (WARNING: deletes all data!):
```bash
./scripts/docker-dev.sh clean-all
```

## Troubleshooting

### Permission Denied (Linux/Mac)

If you get "permission denied" errors:
```bash
chmod +x scripts/docker-dev.sh
```

### Docker Not Running

Make sure Docker Desktop is running:
```bash
docker --version
docker-compose --version
```

### Ports Already in Use

If ports 8000, 5432, or 6379 are in use, stop the script and:

1. Check what's using the port:
   ```bash
   # Linux/Mac
   lsof -i :8000

   # Windows
   netstat -ano | findstr :8000
   ```

2. Either stop that process or change ports in `docker-compose.dev.yml`

### Services Won't Start

View logs to diagnose:
```bash
./scripts/docker-dev.sh logs backend
```

Common issues:
- Database not ready: Wait a few seconds and try again
- Missing .env file: Run `./scripts/docker-dev.sh init`
- Port conflicts: See "Ports Already in Use" above

## Manual Docker Commands

If you prefer direct Docker commands:

```bash
# Start services
docker-compose -f docker-compose.dev.yml up -d

# View logs
docker-compose -f docker-compose.dev.yml logs -f backend

# Stop services
docker-compose -f docker-compose.dev.yml down

# Execute command
docker-compose -f docker-compose.dev.yml exec backend <command>
```

## See Also

- [DOCKER.md](../DOCKER.md) - Complete Docker documentation
- [README.md](../README.md) - Backend README
