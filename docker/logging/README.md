# Logging Configuration Files

Quick reference for AI Playground logging configuration files.

## Files Overview

| File | Purpose |
|------|---------|
| [logrotate.conf](logrotate.conf) | Linux logrotate configuration for automatic log rotation |
| [fluent.conf](fluent.conf) | Fluentd configuration for log aggregation and forwarding |
| [loki-config.yml](loki-config.yml) | Grafana Loki configuration for log storage |
| [promtail-config.yml](promtail-config.yml) | Promtail configuration for collecting logs to Loki |
| [grafana-datasources.yml](grafana-datasources.yml) | Grafana datasource configuration |
| [grafana-dashboards.yml](grafana-dashboards.yml) | Grafana dashboard provisioning |

## Quick Start

### Option 1: ELK Stack (Elasticsearch + Kibana)

```bash
# Start ELK stack
docker-compose -f ../../docker-compose.logging.yml up -d elasticsearch kibana

# Access Kibana
open http://localhost:5601
```

### Option 2: Loki + Grafana

```bash
# Start Loki stack
docker-compose -f ../../docker-compose.logging.yml up -d loki promtail grafana

# Access Grafana
open http://localhost:3000
```

### Option 3: Both

```bash
# Start full logging stack
docker-compose -f ../../docker-compose.logging.yml up -d
```

## Configuration

### Environment Variables

Set in `.env.docker.production`:

```env
# Elasticsearch
ELASTICSEARCH_HOST=elasticsearch
ELASTICSEARCH_PORT=9200

# Loki
LOKI_URL=http://loki:3100

# Grafana
GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD=change-me

# S3 (optional long-term storage)
AWS_ACCESS_KEY_ID=your-key
AWS_SECRET_ACCESS_KEY=your-secret
S3_LOG_BUCKET=aiplayground-logs
```

## Services

### Elasticsearch
- **Port:** 9200
- **Purpose:** Log storage and search
- **Query:** http://localhost:9200/_search

### Kibana
- **Port:** 5601
- **Purpose:** Log visualization
- **URL:** http://localhost:5601

### Loki
- **Port:** 3100
- **Purpose:** Log aggregation
- **API:** http://localhost:3100/loki/api/v1/query

### Grafana
- **Port:** 3000
- **Purpose:** Dashboards and visualization
- **URL:** http://localhost:3000

### Fluentd
- **Port:** 24224
- **Purpose:** Log collection and forwarding

### Promtail
- **Purpose:** Loki log shipper

## Common Tasks

### View Logs in Kibana

1. Open http://localhost:5601
2. Create index pattern: `aiplayground-*`
3. Go to Discover
4. Search and filter logs

### Query Loki

```bash
# Via API
curl -G -s "http://localhost:3100/loki/api/v1/query" \
  --data-urlencode 'query={app="backend"}' | jq .
```

### Clear Old Logs

```bash
# Elasticsearch
curl -X DELETE "localhost:9200/aiplayground-2025.12.*"

# Loki (configure retention in loki-config.yml)
```

## Troubleshooting

### Elasticsearch Yellow/Red Status

```bash
# Check cluster health
curl http://localhost:9200/_cluster/health?pretty

# Check indices
curl http://localhost:9200/_cat/indices?v
```

### Kibana Can't Connect

```bash
# Check Elasticsearch is running
docker-compose -f ../../docker-compose.logging.yml ps elasticsearch

# Check logs
docker-compose -f ../../docker-compose.logging.yml logs kibana
```

### Loki Not Receiving Logs

```bash
# Check Promtail
docker-compose -f ../../docker-compose.logging.yml logs promtail

# Check Loki
curl http://localhost:3100/ready
```

## Full Documentation

See [../../LOGGING.md](../../LOGGING.md) for complete documentation.
