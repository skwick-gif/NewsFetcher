# Tariff Radar Monitoring Runbook

## Table of Contents
1. [Overview](#overview)
2. [Monitoring Dashboard](#monitoring-dashboard)
3. [Alert Response Procedures](#alert-response-procedures)
4. [Performance Monitoring](#performance-monitoring)
5. [Log Analysis](#log-analysis)
6. [Maintenance Procedures](#maintenance-procedures)
7. [Emergency Response](#emergency-response)
8. [Escalation Procedures](#escalation-procedures)

## Overview

This runbook provides operational procedures for monitoring and maintaining the Tariff Radar system. It includes step-by-step instructions for common monitoring tasks, alert response, and troubleshooting.

### Key Monitoring Endpoints

- **System Status**: `http://localhost:8000/monitoring/system/status`
- **Health Checks**: `http://localhost:8000/monitoring/health`
- **Monitoring Dashboard**: `http://localhost:8000/monitoring/dashboard`
- **Prometheus Metrics**: `http://localhost:8000/monitoring/metrics`
- **Active Alerts**: `http://localhost:8000/monitoring/alerts`

### Monitoring Access

```bash
# Quick health check
curl http://localhost:8000/monitoring/health

# Detailed system status
curl http://localhost:8000/monitoring/system/status

# Check active alerts
curl -H "X-API-Key: your_api_key" http://localhost:8000/monitoring/alerts
```

## Monitoring Dashboard

### Accessing the Dashboard

1. **Web Interface**: Navigate to `http://localhost:8000/monitoring/dashboard`
2. **API Access**: Use API key authentication for programmatic access
3. **Mobile Access**: Dashboard is mobile-responsive

### Dashboard Sections

#### 1. System Health Overview
- Overall system status (Healthy/Degraded/Unhealthy)
- Component health status
- Last update timestamp
- Quick action buttons

#### 2. Performance Metrics
- CPU, Memory, Disk usage
- Response time trends
- Request rate and error rate
- Database performance

#### 3. Active Alerts
- Current active alerts
- Alert severity distribution
- Recent alert history
- Alert acknowledgment status

#### 4. Business Metrics
- Articles processed today
- Notification success rate
- RSS feed health
- Queue lengths

#### 5. System Information
- Service versions
- Uptime
- Configuration status
- Resource utilization

### Dashboard Interpretation

#### Health Status Colors
- 🟢 **Green (Healthy)**: All systems operational
- 🟡 **Yellow (Degraded)**: Some issues detected, service available
- 🔴 **Red (Unhealthy)**: Critical issues, service may be impacted
- ⚫ **Gray (Unknown)**: Status cannot be determined

#### Key Performance Indicators (KPIs)

| Metric | Good | Warning | Critical |
|--------|------|---------|----------|
| CPU Usage | < 70% | 70-85% | > 85% |
| Memory Usage | < 80% | 80-90% | > 90% |
| Disk Usage | < 70% | 70-85% | > 85% |
| Response Time | < 500ms | 500ms-2s | > 2s |
| Error Rate | < 1% | 1-5% | > 5% |
| Queue Length | < 50 | 50-100 | > 100 |

## Alert Response Procedures

### Alert Severity Levels

#### 🔴 Critical Alerts
**Response Time**: Immediate (< 5 minutes)
**Escalation**: Page on-call engineer

Common Critical Alerts:
- `database_unhealthy`: Database connection failed
- `high_memory_usage`: Memory usage > 95%
- System completely down

**Response Steps**:
1. Acknowledge alert immediately
2. Check system health dashboard
3. Review recent changes/deployments
4. Follow specific alert procedures below
5. Update incident status
6. Escalate if not resolved in 15 minutes

#### 🟡 High Alerts
**Response Time**: 15 minutes
**Escalation**: Notify team lead

Common High Alerts:
- `high_cpu_usage`: CPU usage > 90%
- `redis_unhealthy`: Redis connection issues
- `disk_space_low`: Disk usage > 85%

**Response Steps**:
1. Acknowledge alert
2. Investigate root cause
3. Apply immediate mitigation
4. Schedule permanent fix
5. Document resolution

#### 🔵 Medium Alerts
**Response Time**: 1 hour
**Escalation**: Create ticket

Common Medium Alerts:
- `high_error_rate`: HTTP error rate > 5%
- `slow_response_time`: Average response > 2s
- `celery_queue_backup`: Queue length > 100

**Response Steps**:
1. Acknowledge alert
2. Investigate during business hours
3. Plan resolution
4. Implement fix during maintenance window

#### ⚪ Low Alerts
**Response Time**: 4 hours
**Escalation**: Weekly review

**Response Steps**:
1. Log for trend analysis
2. Review during regular maintenance
3. Address in next release cycle

### Specific Alert Procedures

#### Database Connection Failure (`database_unhealthy`)

**Symptoms**:
- Database health check failing
- Application errors related to database
- Connection timeout errors

**Investigation Steps**:
```bash
# 1. Check database container status
docker-compose ps postgres

# 2. Check database logs
docker-compose logs postgres --tail=50

# 3. Check database connectivity
docker-compose exec postgres pg_isready -U tariff_user

# 4. Check database connections
docker-compose exec postgres psql -U tariff_user -d tariff_radar -c "
SELECT count(*) as total_connections,
       count(*) FILTER (WHERE state = 'active') as active_connections
FROM pg_stat_activity;
"
```

**Resolution Steps**:
1. **If container is down**:
   ```bash
   docker-compose restart postgres
   ```

2. **If connection pool exhausted**:
   ```bash
   # Kill long-running queries
   docker-compose exec postgres psql -U tariff_user -d tariff_radar -c "
   SELECT pg_terminate_backend(pid) 
   FROM pg_stat_activity 
   WHERE state = 'active' AND query_start < now() - interval '5 minutes';
   "
   ```

3. **If disk space full**:
   ```bash
   # Check disk space
   df -h
   
   # Clean old logs
   docker system prune -f
   
   # Vacuum database
   docker-compose exec postgres psql -U tariff_user -d tariff_radar -c "VACUUM;"
   ```

#### High Memory Usage (`high_memory_usage`)

**Investigation Steps**:
```bash
# 1. Check memory usage by container
docker stats --no-stream

# 2. Check system memory
free -h

# 3. Check for memory leaks
docker-compose exec app ps aux --sort=-%mem | head -10
```

**Resolution Steps**:
1. **Immediate mitigation**:
   ```bash
   # Restart memory-intensive services
   docker-compose restart worker
   docker-compose restart app
   ```

2. **Clear caches**:
   ```bash
   # Clear Redis cache
   docker-compose exec redis redis-cli FLUSHDB
   
   # Clear application cache
   curl -X POST -H "X-API-Key: your_api_key" \
     http://localhost:8000/admin/clear-cache
   ```

3. **Scale down if needed**:
   ```bash
   # Reduce worker count temporarily
   docker-compose up -d --scale worker=1
   ```

#### High CPU Usage (`high_cpu_usage`)

**Investigation Steps**:
```bash
# 1. Check CPU usage by container
docker stats --no-stream

# 2. Check system load
uptime

# 3. Check for runaway processes
docker-compose exec app top -o %CPU
```

**Resolution Steps**:
1. **Identify CPU-intensive tasks**:
   ```bash
   # Check Celery tasks
   docker-compose exec worker celery -A sched.tasks inspect active
   
   # Check database queries
   docker-compose exec postgres psql -U tariff_user -d tariff_radar -c "
   SELECT pid, query, state, query_start 
   FROM pg_stat_activity 
   WHERE state = 'active' 
   ORDER BY query_start;
   "
   ```

2. **Optimize or kill intensive tasks**:
   ```bash
   # Kill specific Celery task
   docker-compose exec worker celery -A sched.tasks control revoke <task_id>
   
   # Kill long-running database query
   docker-compose exec postgres psql -U tariff_user -d tariff_radar -c "
   SELECT pg_terminate_backend(<pid>);
   "
   ```

#### Queue Backup (`celery_queue_backup`)

**Investigation Steps**:
```bash
# 1. Check queue length
docker-compose exec worker celery -A sched.tasks inspect active

# 2. Check worker status
docker-compose exec worker celery -A sched.tasks status

# 3. Check failed tasks
docker-compose exec worker celery -A sched.tasks events
```

**Resolution Steps**:
1. **Scale up workers**:
   ```bash
   docker-compose up -d --scale worker=3
   ```

2. **Clear failed tasks**:
   ```bash
   docker-compose exec worker celery -A sched.tasks purge
   ```

3. **Restart queue if stuck**:
   ```bash
   docker-compose restart worker
   docker-compose restart redis
   ```

## Performance Monitoring

### Key Metrics to Monitor

#### Application Performance
```bash
# Response time metrics
curl http://localhost:8000/monitoring/metrics | grep http_request_duration

# Request rate
curl http://localhost:8000/monitoring/metrics | grep http_requests_total

# Error rate
curl http://localhost:8000/monitoring/metrics | grep http_requests_total | grep -E '4[0-9][0-9]|5[0-9][0-9]'
```

#### Database Performance
```bash
# Query performance
docker-compose exec postgres psql -U tariff_user -d tariff_radar -c "
SELECT query, mean_time, calls, total_time
FROM pg_stat_statements 
ORDER BY mean_time DESC 
LIMIT 10;
"

# Connection stats
docker-compose exec postgres psql -U tariff_user -d tariff_radar -c "
SELECT count(*) as total, state 
FROM pg_stat_activity 
GROUP BY state;
"
```

#### System Resources
```bash
# CPU and memory
docker stats --no-stream

# Disk usage
df -h

# Network I/O
docker-compose exec app cat /proc/net/dev
```

### Performance Optimization

#### Database Optimization
```bash
# Analyze query performance
docker-compose exec postgres psql -U tariff_user -d tariff_radar -c "
ANALYZE;
"

# Update statistics
docker-compose exec postgres psql -U tariff_user -d tariff_radar -c "
VACUUM ANALYZE;
"

# Check index usage
docker-compose exec postgres psql -U tariff_user -d tariff_radar -c "
SELECT schemaname, tablename, indexname, idx_scan, idx_tup_read, idx_tup_fetch
FROM pg_stat_user_indexes 
ORDER BY idx_scan DESC;
"
```

#### Cache Optimization
```bash
# Check Redis memory usage
docker-compose exec redis redis-cli info memory

# Check hit rate
docker-compose exec redis redis-cli info stats | grep keyspace

# Clear old cache entries
docker-compose exec redis redis-cli --scan --pattern "cache:*" | \
  xargs docker-compose exec redis redis-cli del
```

## Log Analysis

### Log Locations

```bash
# Application logs
tail -f logs/tariff-radar.json

# Error logs
tail -f logs/tariff-radar_errors.json

# Security logs
tail -f logs/tariff-radar_security.json

# Performance logs
tail -f logs/tariff-radar_performance.json
```

### Log Analysis Commands

#### Find Errors
```bash
# Recent errors
grep -i error logs/tariff-radar.json | tail -20

# Errors by component
jq -r 'select(.level=="ERROR") | .logger + ": " + .message' logs/tariff-radar.json

# Error rate over time
jq -r 'select(.level=="ERROR") | .timestamp' logs/tariff-radar.json | \
  cut -c1-13 | sort | uniq -c
```

#### Performance Analysis
```bash
# Slow requests
jq -r 'select(.performance_metric==true and .duration_seconds > 2) | 
       .timestamp + " " + .operation + " " + (.duration_seconds|tostring)' \
       logs/tariff-radar_performance.json

# Average response times
jq -r 'select(.method and .duration_seconds) | .duration_seconds' \
  logs/tariff-radar.json | awk '{sum+=$1; n++} END {print sum/n}'
```

#### Security Analysis
```bash
# Failed authentication attempts
jq -r 'select(.event_type=="authentication" and .success==false) | 
       .timestamp + " " + .username + " " + .ip_address' \
       logs/tariff-radar_security.json

# Rate limit violations
jq -r 'select(.event_type=="rate_limit_exceeded") | 
       .timestamp + " " + .identifier + " " + .ip_address' \
       logs/tariff-radar_security.json
```

### Log Rotation and Cleanup

```bash
# Check log sizes
du -h logs/

# Archive old logs
tar -czf logs/archive/logs_$(date +%Y%m%d).tar.gz logs/*.json
find logs/ -name "*.json" -mtime +7 -delete

# Clean up Docker logs
docker system prune -f
```

## Maintenance Procedures

### Daily Maintenance

#### Health Check Script
```bash
#!/bin/bash
# daily_health_check.sh

echo "=== Daily Health Check $(date) ==="

# System health
curl -s http://localhost:8000/monitoring/health | jq .

# Check disk space
echo "Disk Usage:"
df -h | grep -E '(Filesystem|/dev/)'

# Check memory
echo "Memory Usage:"
free -h

# Check active alerts
echo "Active Alerts:"
curl -s -H "X-API-Key: $API_KEY" http://localhost:8000/monitoring/alerts | \
  jq '.alerts | length'

# Database health
echo "Database Health:"
docker-compose exec postgres pg_isready -U tariff_user

echo "=== Health Check Complete ==="
```

#### Log Review Script
```bash
#!/bin/bash
# daily_log_review.sh

echo "=== Daily Log Review $(date) ==="

# Count errors in last 24 hours
echo "Errors in last 24 hours:"
find logs/ -name "*.json" -mtime -1 -exec \
  grep -c '"level":"ERROR"' {} \; | \
  awk '{sum+=$1} END {print sum}'

# Top error messages
echo "Top error messages:"
find logs/ -name "*.json" -mtime -1 -exec \
  grep '"level":"ERROR"' {} \; | \
  jq -r '.message' | sort | uniq -c | sort -nr | head -5

# Security events
echo "Security events:"
find logs/ -name "*security.json" -mtime -1 -exec \
  grep -c '"security_event":true' {} \; | \
  awk '{sum+=$1} END {print sum}'

echo "=== Log Review Complete ==="
```

### Weekly Maintenance

#### Performance Review
```bash
#!/bin/bash
# weekly_performance_review.sh

echo "=== Weekly Performance Review $(date) ==="

# Average response times
echo "Average response times (last 7 days):"
find logs/ -name "*.json" -mtime -7 -exec \
  grep '"duration_seconds"' {} \; | \
  jq -r '.duration_seconds' | \
  awk '{sum+=$1; n++} END {printf "%.3f seconds\n", sum/n}'

# Database performance
echo "Database query performance:"
docker-compose exec postgres psql -U tariff_user -d tariff_radar -c "
SELECT query, mean_time, calls 
FROM pg_stat_statements 
WHERE mean_time > 1000 
ORDER BY mean_time DESC 
LIMIT 5;
"

# Resource utilization trends
echo "Resource utilization trends:"
# Add Prometheus queries or custom metrics analysis

echo "=== Performance Review Complete ==="
```

### Monthly Maintenance

#### Database Maintenance
```bash
#!/bin/bash
# monthly_database_maintenance.sh

echo "=== Monthly Database Maintenance $(date) ==="

# Full vacuum and analyze
docker-compose exec postgres psql -U tariff_user -d tariff_radar -c "
VACUUM FULL;
ANALYZE;
"

# Update statistics
docker-compose exec postgres psql -U tariff_user -d tariff_radar -c "
SELECT pg_stat_reset();
"

# Check table sizes
docker-compose exec postgres psql -U tariff_user -d tariff_radar -c "
SELECT schemaname, tablename, 
       pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
FROM pg_tables 
WHERE schemaname = 'public' 
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
"

# Backup database
pg_dump -h localhost -U tariff_user tariff_radar | \
  gzip > backups/monthly_backup_$(date +%Y%m%d).sql.gz

echo "=== Database Maintenance Complete ==="
```

## Emergency Response

### Service Down Procedure

#### Complete System Down
1. **Immediate Response**:
   ```bash
   # Check all services
   docker-compose ps
   
   # Check system resources
   df -h
   free -h
   uptime
   ```

2. **Emergency Restart**:
   ```bash
   # Stop all services
   docker-compose down
   
   # Clean up resources
   docker system prune -f
   
   # Restart core services first
   docker-compose up -d postgres redis
   sleep 30
   
   # Start application
   docker-compose up -d app
   sleep 15
   
   # Start workers
   docker-compose up -d worker
   ```

3. **Verify Recovery**:
   ```bash
   # Check health
   curl http://localhost:8000/monitoring/health
   
   # Check logs
   docker-compose logs app --tail=20
   ```

#### Database Emergency Recovery
1. **If database is corrupted**:
   ```bash
   # Stop application
   docker-compose stop app worker
   
   # Backup current state
   docker-compose exec postgres pg_dump -U tariff_user tariff_radar > emergency_backup.sql
   
   # Restore from backup
   docker-compose exec -T postgres psql -U tariff_user tariff_radar < latest_backup.sql
   
   # Restart services
   docker-compose start app worker
   ```

2. **If disk full**:
   ```bash
   # Clear old logs
   find logs/ -name "*.json" -mtime +3 -delete
   
   # Clear Docker cache
   docker system prune -a -f
   
   # Clear old database logs
   docker-compose exec postgres find /var/lib/postgresql/data/log -mtime +1 -delete
   ```

### Data Recovery

#### Article Data Recovery
```bash
# Export articles from backup
docker-compose exec -T postgres psql -U tariff_user tariff_radar -c "
COPY (SELECT * FROM articles WHERE created_at >= '2024-01-01') 
TO STDOUT WITH CSV HEADER;" > articles_recovery.csv

# Import articles
docker-compose exec -T postgres psql -U tariff_user tariff_radar -c "
COPY articles FROM STDIN WITH CSV HEADER;" < articles_recovery.csv
```

#### Configuration Recovery
```bash
# Restore configuration from backup
cp config_backup/rss_sources.yaml config/
cp config_backup/keywords.yaml config/

# Restart services to reload config
docker-compose restart app worker
```

## Escalation Procedures

### Escalation Matrix

| Severity | Response Time | Initial Contact | Escalation 1 | Escalation 2 |
|----------|---------------|----------------|--------------|--------------|
| Critical | 5 minutes | On-call Engineer | Team Lead (15 min) | Manager (30 min) |
| High | 15 minutes | Team Member | Team Lead (1 hour) | Manager (4 hours) |
| Medium | 1 hour | Team Member | Team Lead (4 hours) | Manager (24 hours) |
| Low | 4 hours | Team Member | Team Lead (Weekly) | - |

### Escalation Triggers

#### Automatic Escalation
- Critical alert not acknowledged in 5 minutes
- High alert not resolved in 1 hour
- System downtime > 15 minutes
- Data loss detected
- Security breach suspected

#### Manual Escalation
- Complex technical issue
- Multiple system failures
- Customer impact
- Need for architectural changes

### Communication Templates

#### Critical Incident Notification
```
CRITICAL INCIDENT: Tariff Radar System Down

Time: [TIMESTAMP]
Impact: [DESCRIPTION]
Status: Investigating
ETA: TBD

Actions Taken:
- [ACTION 1]
- [ACTION 2]

Next Update: [TIME]
```

#### Resolution Notification
```
RESOLVED: Tariff Radar Issue

Time: [TIMESTAMP]
Duration: [DURATION]
Root Cause: [CAUSE]
Resolution: [RESOLUTION]

Post-Incident Actions:
- [ACTION 1]
- [ACTION 2]
```

### Contact Information

- **On-Call Engineer**: [phone/email]
- **Team Lead**: [phone/email]
- **Manager**: [phone/email]
- **Emergency**: [emergency contact]

---

## Monitoring Checklist

### Daily Tasks
- [ ] Review system health dashboard
- [ ] Check for active alerts
- [ ] Review error logs
- [ ] Verify backup completion
- [ ] Check disk space
- [ ] Monitor resource usage

### Weekly Tasks
- [ ] Performance trend analysis
- [ ] Security log review
- [ ] Database maintenance
- [ ] Alert rule review
- [ ] Capacity planning
- [ ] Documentation updates

### Monthly Tasks
- [ ] Full system backup
- [ ] Database optimization
- [ ] Performance baseline update
- [ ] Security audit
- [ ] Alert threshold review
- [ ] Disaster recovery test

This runbook should be kept up-to-date and reviewed regularly with the operations team.