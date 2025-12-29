# ML Platform Alerting System

Comprehensive alerting system for monitoring training jobs and system health.

## Overview

This alerting system provides:
- **Real-time monitoring** of training jobs
- **Automatic alerts** for failures and warnings
- **Multiple notification channels** (Email, Slack, PagerDuty, Webhooks)
- **Customizable alert rules** for different scenarios
- **Rich alert templates** with actionable information

## Architecture

```
┌─────────────┐
│   Backend   │──► Exposes metrics
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Prometheus  │──► Collects metrics & evaluates rules
└──────┬──────┘
       │
       ▼
┌─────────────┐
│AlertManager │──► Routes alerts to receivers
└──────┬──────┘
       │
       ├──► Email
       ├──► Slack
       ├──► PagerDuty
       └──► Webhooks
```

## Components

### 1. Prometheus
- Collects metrics from backend and system
- Evaluates alert rules
- Sends alerts to AlertManager
- Web UI: http://localhost:9090

### 2. AlertManager
- Receives alerts from Prometheus
- Routes alerts based on labels
- Deduplicates and groups alerts
- Sends notifications to multiple channels
- Web UI: http://localhost:9093

### 3. Grafana
- Visualizes metrics and alerts
- Provides dashboards for monitoring
- Web UI: http://localhost:3001

### 4. Exporters
- **Node Exporter**: System metrics (CPU, memory, disk)
- **cAdvisor**: Container metrics
- **Pushgateway**: Batch job metrics

## Quick Start

### 1. Setup Environment

```bash
# Copy environment template
cp docker/alerting/.env.example docker/alerting/.env

# Edit configuration
nano docker/alerting/.env
```

### 2. Configure Email

Update in `.env`:
```env
SMTP_HOST=smtp.gmail.com:587
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-app-password
SMTP_FROM=ml-alerts@example.com
ML_TEAM_EMAIL=team@example.com
```

### 3. Configure Slack (Optional)

1. Create a Slack webhook: https://api.slack.com/messaging/webhooks
2. Update in `.env`:
```env
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL
SLACK_ALERTS_CHANNEL=#ml-alerts
```

### 4. Start Alerting Stack

```bash
# Start all alerting services
docker-compose -f docker/docker-compose.alerting.yml up -d

# Check status
docker-compose -f docker/docker-compose.alerting.yml ps

# View logs
docker-compose -f docker/docker-compose.alerting.yml logs -f
```

### 5. Verify Setup

```bash
# Check Prometheus
curl http://localhost:9090/-/healthy

# Check AlertManager
curl http://localhost:9093/-/healthy

# Check Grafana
curl http://localhost:3001/api/health
```

## Alert Types

### Critical Alerts

#### TrainingJobFailed
- **Trigger**: Training job status = failed
- **Duration**: 1 minute
- **Notification**: Email + Slack + PagerDuty
- **Actions**: View logs, retry job

#### TrainingJobTimeout
- **Trigger**: Job running longer than timeout
- **Duration**: 5 minutes
- **Notification**: Email + Slack + PagerDuty
- **Actions**: Check if stuck, increase timeout

#### TrainingOutOfMemory
- **Trigger**: Memory usage > 95%
- **Duration**: 2 minutes
- **Notification**: Email + Slack
- **Actions**: Reduce batch size, request more memory

#### TrainingJobCrashed
- **Trigger**: Job restarts detected
- **Duration**: 1 minute
- **Notification**: Email + Slack + PagerDuty
- **Actions**: Check logs for crashes

### Warning Alerts

#### TrainingProgressSlow
- **Trigger**: Progress rate < 1% per 10 minutes
- **Duration**: 15 minutes
- **Notification**: Email + Slack
- **Actions**: Check data loading, optimize

#### TrainingLossHigh
- **Trigger**: Loss value > 10
- **Duration**: 10 minutes
- **Notification**: Email + Slack
- **Actions**: Check learning rate, verify data

#### TrainingLossNotDecreasing
- **Trigger**: Loss not improving for 30 minutes
- **Duration**: 30 minutes
- **Notification**: Email + Slack
- **Actions**: Adjust learning rate, check gradients

#### ValidationAccuracyStagnant
- **Trigger**: Val accuracy not improving for 1 hour
- **Duration**: 1 hour
- **Notification**: Email + Slack
- **Actions**: Consider early stopping

### Data Quality Alerts

#### HighMissingValues
- **Trigger**: Missing values > 20%
- **Duration**: 5 minutes
- **Notification**: Email to data team
- **Actions**: Apply imputation, check pipeline

#### DataDistributionShift
- **Trigger**: Data drift score > 0.5
- **Duration**: 10 minutes
- **Notification**: Email to data team
- **Actions**: Verify data source, retrain

#### ImbalancedClasses
- **Trigger**: Class imbalance ratio > 10:1
- **Duration**: 5 minutes
- **Notification**: Email to data team
- **Actions**: Apply class weighting, sampling

## Notification Channels

### Email
- **Recipients**: Configurable per alert type
- **Format**: HTML with rich formatting
- **Includes**: Error details, suggestions, action buttons
- **Configuration**: `alertmanager.yml` email_configs

### Slack
- **Channels**: #ml-alerts, #ml-warnings, #ops-alerts
- **Format**: Markdown with links
- **Includes**: Job details, quick action links
- **Configuration**: `alertmanager.yml` slack_configs

### PagerDuty
- **Severity**: Critical alerts only
- **Integration**: Service key required
- **Configuration**: `alertmanager.yml` pagerduty_configs

### Webhooks
- **Endpoint**: Backend API webhook handler
- **Format**: JSON payload
- **Authentication**: Bearer token
- **Configuration**: `alertmanager.yml` webhook_configs

## Customization

### Adding New Alert Rules

Edit `docker/alerting/prometheus-rules.yml`:

```yaml
- alert: MyCustomAlert
  expr: my_metric > threshold
  for: 5m
  labels:
    severity: warning
    alert_type: custom
    team: ml
  annotations:
    summary: "Custom alert triggered"
    description: "Metric value: {{ $value }}"
    suggestions: "Do something to fix this"
```

### Modifying Email Templates

Edit `docker/alerting/templates/email.tmpl`:

```html
{{ define "email.custom.html" }}
<!DOCTYPE html>
<html>
<body>
  <h1>Custom Alert</h1>
  {{ range .Alerts }}
    <p>{{ .Annotations.description }}</p>
  {{ end }}
</body>
</html>
{{ end }}
```

### Changing Alert Routing

Edit `docker/alerting/alertmanager.yml`:

```yaml
routes:
  - match:
      severity: critical
      team: ml
    receiver: 'ml-team-critical'
    repeat_interval: 2h
```

### Adding New Receivers

Edit `docker/alerting/alertmanager.yml`:

```yaml
receivers:
  - name: 'my-team'
    email_configs:
      - to: 'my-team@example.com'
    slack_configs:
      - api_url: '${SLACK_WEBHOOK_URL}'
        channel: '#my-channel'
```

## Monitoring Dashboards

### Grafana Dashboards

Access Grafana at http://localhost:3001

Default credentials:
- Username: admin
- Password: admin (change in production)

Pre-configured dashboards:
1. **Training Jobs Overview**: All training jobs status
2. **Training Job Details**: Individual job metrics
3. **System Resources**: CPU, memory, disk, GPU
4. **Alert Status**: Active alerts and history
5. **Data Quality**: Data drift and quality metrics

### Prometheus UI

Access Prometheus at http://localhost:9090

Features:
- Query metrics with PromQL
- View active alerts
- Check target health
- Explore time series data

### AlertManager UI

Access AlertManager at http://localhost:9093

Features:
- View active alerts
- Silence alerts temporarily
- View alert history
- Test alert routing

## Troubleshooting

### Alerts Not Firing

1. Check Prometheus is scraping metrics:
```bash
curl http://localhost:9090/api/v1/targets
```

2. Check alert rules are loaded:
```bash
curl http://localhost:9090/api/v1/rules
```

3. Check AlertManager is receiving alerts:
```bash
curl http://localhost:9093/api/v2/alerts
```

### Email Not Sending

1. Verify SMTP configuration in `.env`
2. Check AlertManager logs:
```bash
docker logs ml-platform-alertmanager
```

3. Test SMTP connection:
```bash
docker exec ml-platform-alertmanager \
  wget --spider smtp://smtp.gmail.com:587
```

### Slack Not Working

1. Verify webhook URL in `.env`
2. Test webhook manually:
```bash
curl -X POST ${SLACK_WEBHOOK_URL} \
  -H 'Content-Type: application/json' \
  -d '{"text":"Test message"}'
```

### High Alert Volume

1. Adjust alert thresholds in `prometheus-rules.yml`
2. Increase `group_wait` and `group_interval` in `alertmanager.yml`
3. Use inhibition rules to suppress related alerts

## Best Practices

### Alert Fatigue Prevention

1. **Set appropriate thresholds**: Avoid too sensitive alerts
2. **Use proper durations**: Don't alert on transient issues
3. **Group related alerts**: Reduce notification spam
4. **Implement inhibition**: Suppress redundant alerts
5. **Regular review**: Tune alerts based on feedback

### Alert Quality

1. **Actionable**: Every alert should have clear actions
2. **Contextual**: Include relevant information
3. **Prioritized**: Use severity levels correctly
4. **Documented**: Link to runbooks
5. **Tested**: Verify alerts work as expected

### Security

1. **Secure credentials**: Use environment variables
2. **Rotate tokens**: Regular rotation of API keys
3. **Limit access**: Restrict AlertManager UI access
4. **Encrypt communications**: Use TLS for SMTP
5. **Audit logs**: Monitor alert configuration changes

## Maintenance

### Regular Tasks

1. **Review alert rules**: Monthly review and tuning
2. **Clean up old data**: Prometheus retention policy
3. **Update templates**: Improve notification content
4. **Test notifications**: Verify all channels work
5. **Monitor alerting system**: Alert on alerting failures

### Backup

```bash
# Backup Prometheus data
docker run --rm -v prometheus-data:/data \
  -v $(pwd)/backups:/backup \
  alpine tar czf /backup/prometheus-$(date +%Y%m%d).tar.gz /data

# Backup AlertManager data
docker run --rm -v alertmanager-data:/data \
  -v $(pwd)/backups:/backup \
  alpine tar czf /backup/alertmanager-$(date +%Y%m%d).tar.gz /data
```

### Upgrade

```bash
# Pull latest images
docker-compose -f docker/docker-compose.alerting.yml pull

# Restart services
docker-compose -f docker/docker-compose.alerting.yml up -d
```

## Support

For issues or questions:
1. Check logs: `docker-compose logs -f`
2. Review documentation: This README
3. Check Prometheus/AlertManager docs
4. Contact ops team

## References

- [Prometheus Documentation](https://prometheus.io/docs/)
- [AlertManager Documentation](https://prometheus.io/docs/alerting/latest/alertmanager/)
- [Grafana Documentation](https://grafana.com/docs/)
- [PromQL Guide](https://prometheus.io/docs/prometheus/latest/querying/basics/)
