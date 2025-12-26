# Docker Configuration for AI-Playground

This directory contains Docker configuration files for the AI-Playground project.

## 📁 Directory Structure

```
docker/
├── README.md           # This file
├── postgres/
│   └── init.sql       # PostgreSQL initialization script
└── redis/
    └── redis.conf     # Redis configuration
```

## 🐳 Docker Services

### PostgreSQL (postgres/)
- **Purpose**: Main application database
- **Image**: postgres:15-alpine
- **Port**: 5432
- **Configuration**: `init.sql` runs on first container creation
- **Features**:
  - UUID extension enabled
  - Text search optimization (pg_trgm)
  - Automatic privilege grants

### Redis (redis/)
- **Purpose**: Caching and Celery message broker
- **Image**: redis:7-alpine
- **Port**: 6379
- **Configuration**: Custom `redis.conf` with:
  - LRU eviction policy
  - 256MB memory limit
  - Persistence enabled (RDB snapshots)
  - Optimized for ML workload caching

## 🚀 Quick Start

### Using Quick Start Scripts

**Windows:**
```cmd
docker-start.bat
```

**Linux/Mac:**
```bash
chmod +x docker-start.sh
./docker-start.sh
```

### Manual Commands

```bash
# Build all services
docker-compose build

# Start all services
docker-compose up -d

# View status
docker-compose ps

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## ⚙️ Configuration Files

### postgres/init.sql

Initializes the PostgreSQL database with:
- Required extensions (uuid-ossp, pg_trgm)
- User privileges
- Database setup

**Customization:**
Edit `init.sql` to add:
- Additional extensions
- Custom schemas
- Initial data
- Database-level settings

### redis/redis.conf

Configures Redis with:
- Memory management (256MB limit, LRU eviction)
- Persistence (RDB snapshots)
- Network settings
- Performance optimizations

**Customization:**
Edit `redis.conf` to modify:
- Memory limits (`maxmemory`)
- Eviction policy (`maxmemory-policy`)
- Persistence settings (`save`, `appendonly`)
- Security (`requirepass`)

## 🔧 Common Tasks

### PostgreSQL

**Access database:**
```bash
docker-compose exec postgres psql -U aiplayground -d aiplayground
```

**Run SQL file:**
```bash
docker-compose exec -T postgres psql -U aiplayground -d aiplayground < your-script.sql
```

**Backup database:**
```bash
docker-compose exec postgres pg_dump -U aiplayground aiplayground > backup.sql
```

**Restore database:**
```bash
docker-compose exec -T postgres psql -U aiplayground aiplayground < backup.sql
```

**View database size:**
```bash
docker-compose exec postgres psql -U aiplayground -d aiplayground -c "SELECT pg_size_pretty(pg_database_size('aiplayground'));"
```

### Redis

**Access Redis CLI:**
```bash
docker-compose exec redis redis-cli
```

**Check memory usage:**
```bash
docker-compose exec redis redis-cli INFO memory
```

**View all keys:**
```bash
docker-compose exec redis redis-cli KEYS '*'
```

**Flush all data:**
```bash
docker-compose exec redis redis-cli FLUSHALL
```

**Monitor commands:**
```bash
docker-compose exec redis redis-cli MONITOR
```

## 📊 Monitoring

### Check Service Health

```bash
# PostgreSQL
docker-compose exec postgres pg_isready -U aiplayground

# Redis
docker-compose exec redis redis-cli ping
```

### View Resource Usage

```bash
# All containers
docker stats

# Specific container
docker stats aiplayground-postgres
```

### View Logs

```bash
# PostgreSQL logs
docker-compose logs -f postgres

# Redis logs
docker-compose logs -f redis

# Last 100 lines
docker-compose logs --tail=100 postgres
```

## 🔒 Security Considerations

### Development (Current Setup)
- Default passwords (change in production!)
- No authentication required for Redis
- Database accessible on localhost:5432

### Production Recommendations

1. **Change default passwords:**
```yaml
environment:
  POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}  # Use strong password
```

2. **Enable Redis authentication:**
```conf
# In redis.conf
requirepass your_strong_password_here
```

3. **Restrict network access:**
```yaml
# Don't expose ports publicly
ports:
  - "127.0.0.1:5432:5432"  # Only localhost
```

4. **Use secrets management:**
```yaml
secrets:
  postgres_password:
    file: ./secrets/postgres_password.txt
```

5. **Enable SSL/TLS:**
- Configure PostgreSQL SSL
- Use Redis TLS mode
- Add SSL certificates

## 🐛 Troubleshooting

### PostgreSQL Issues

**Container won't start:**
```bash
# Check logs
docker-compose logs postgres

# Remove volume and restart (WARNING: deletes data!)
docker-compose down -v
docker-compose up -d postgres
```

**Connection refused:**
```bash
# Check if container is running
docker-compose ps postgres

# Check if port is available
netstat -ano | findstr :5432  # Windows
lsof -i :5432                 # Mac/Linux
```

**Slow queries:**
```bash
# Enable query logging in init.sql
ALTER SYSTEM SET log_min_duration_statement = 1000;  # Log queries > 1s
```

### Redis Issues

**High memory usage:**
```bash
# Check memory info
docker-compose exec redis redis-cli INFO memory

# Flush unnecessary data
docker-compose exec redis redis-cli FLUSHDB

# Adjust maxmemory in redis.conf
maxmemory 512mb  # Increase limit
```

**Connection issues:**
```bash
# Test connection
docker-compose exec redis redis-cli ping

# Check Redis logs
docker-compose logs redis
```

## 📚 Additional Resources

- [PostgreSQL Docker Hub](https://hub.docker.com/_/postgres)
- [Redis Docker Hub](https://hub.docker.com/_/redis)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [Redis Documentation](https://redis.io/documentation)
- [Docker Compose Documentation](https://docs.docker.com/compose/)

## 🔄 Updates and Maintenance

### Updating PostgreSQL

```bash
# Backup data first!
docker-compose exec postgres pg_dump -U aiplayground aiplayground > backup.sql

# Update image version in docker-compose.yml
# postgres:15-alpine -> postgres:16-alpine

# Rebuild and restart
docker-compose down
docker-compose up -d postgres
```

### Updating Redis

```bash
# Update image version in docker-compose.yml
# redis:7-alpine -> redis:8-alpine

# Restart service
docker-compose down redis
docker-compose up -d redis
```

## 📝 Notes

- Configuration files are mounted as volumes for easy editing
- Changes to `init.sql` only apply on first container creation
- Changes to `redis.conf` require container restart
- Data is persisted in Docker volumes (survives container restarts)
- Use `docker-compose down -v` to completely reset (deletes all data!)

---

For more detailed Docker setup information, see [DOCKER_GUIDE.md](../DOCKER_GUIDE.md) in the project root.
