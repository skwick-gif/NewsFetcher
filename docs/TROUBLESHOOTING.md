# Tariff Radar Troubleshooting Guide

## Table of Contents
1. [Common Issues](#common-issues)
2. [Performance Problems](#performance-problems)
3. [Database Issues](#database-issues)
4. [Network and Connectivity](#network-and-connectivity)
5. [Configuration Problems](#configuration-problems)
6. [Security Issues](#security-issues)
7. [Data Processing Issues](#data-processing-issues)
8. [Notification Problems](#notification-problems)
9. [Diagnostic Tools](#diagnostic-tools)
10. [Recovery Procedures](#recovery-procedures)

## Common Issues

### Application Won't Start

#### Symptoms
- Docker containers fail to start
- Application exits immediately
- Health check endpoints not responding

#### Diagnostic Steps
```bash
# Check container status
docker-compose ps

# Check logs for startup errors
docker-compose logs app

# Check port conflicts
netstat -tulpn | grep :8000

# Check Docker resources
docker system df
```

#### Solutions

**Port Already in Use:**
```bash
# Find process using port 8000
lsof -i :8000

# Kill the process or change port in docker-compose.yml
kill -9 <PID>
```

**Out of Disk Space:**
```bash
# Clean Docker resources
docker system prune -a -f

# Remove old images
docker images | grep '<none>' | awk '{print $3}' | xargs docker rmi
```

**Missing Environment Variables:**
```bash
# Check environment file
cat .env

# Verify required variables are set
docker-compose config
```

**Database Connection Failed:**
```bash
# Check if database is running
docker-compose ps postgres

# Test database connection
docker-compose exec postgres pg_isready -U tariff_user

# Check database logs
docker-compose logs postgres
```

### Slow Application Response

#### Symptoms
- Web dashboard loads slowly
- API responses take > 5 seconds
- Timeout errors in logs

#### Diagnostic Steps
```bash
# Check response times
curl -w "@curl-format.txt" -o /dev/null -s http://localhost:8000/health

# Check system resources
docker stats --no-stream

# Check database performance
docker-compose exec postgres psql -U tariff_user -d tariff_radar -c "
SELECT query, mean_time, calls FROM pg_stat_statements 
WHERE mean_time > 1000 ORDER BY mean_time DESC LIMIT 10;
"
```

#### Solutions

**High CPU Usage:**
```bash
# Identify CPU-intensive processes
docker-compose exec app top

# Scale down workers temporarily
docker-compose up -d --scale worker=1

# Check for infinite loops in logs
grep -i "loop\|infinite\|stuck" logs/tariff-radar.json
```

**Memory Issues:**
```bash
# Check memory usage by container
docker stats --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}"

# Restart memory-intensive services
docker-compose restart worker

# Clear caches
docker-compose exec redis redis-cli FLUSHDB
```

**Database Lock Issues:**
```bash
# Check for blocked queries
docker-compose exec postgres psql -U tariff_user -d tariff_radar -c "
SELECT blocked_locks.pid AS blocked_pid,
       blocked_activity.usename AS blocked_user,
       blocking_locks.pid AS blocking_pid,
       blocking_activity.usename AS blocking_user,
       blocked_activity.query AS blocked_statement,
       blocking_activity.query AS current_statement_in_blocking_process
FROM pg_catalog.pg_locks blocked_locks
JOIN pg_catalog.pg_stat_activity blocked_activity ON blocked_activity.pid = blocked_locks.pid
JOIN pg_catalog.pg_locks blocking_locks ON blocking_locks.locktype = blocked_locks.locktype
JOIN pg_catalog.pg_stat_activity blocking_activity ON blocking_activity.pid = blocking_locks.pid
WHERE NOT blocked_locks.granted;
"
```

### Service Crashes Randomly

#### Symptoms
- Containers restart frequently
- "Exited" status in docker-compose ps
- Memory or resource exhaustion errors

#### Diagnostic Steps
```bash
# Check container restart count
docker-compose ps

# Check system resources
free -h
df -h

# Check for OOM kills
dmesg | grep -i "killed process\|out of memory"

# Check application logs for crashes
grep -i "crash\|segfault\|exception" logs/tariff-radar_errors.json
```

#### Solutions

**Memory Limits:**
```bash
# Set memory limits in docker-compose.yml
services:
  app:
    mem_limit: 2g
    mem_reservation: 1g
  worker:
    mem_limit: 1g
    mem_reservation: 512m
```

**Resource Exhaustion:**
```bash
# Monitor resource usage over time
while true; do
  docker stats --no-stream >> resource_usage.log
  sleep 60
done
```

## Performance Problems

### High Memory Usage

#### Investigation
```bash
# Check memory usage by process
docker-compose exec app ps aux --sort=-%mem | head -10

# Check Python memory usage
docker-compose exec app python -c "
import psutil
process = psutil.Process()
print(f'Memory: {process.memory_info().rss / 1024 / 1024:.1f} MB')
print(f'Memory %: {process.memory_percent():.1f}%')
"

# Check for memory leaks
docker-compose exec app python -c "
import gc
print(f'Objects tracked by GC: {len(gc.get_objects())}')
"
```

#### Solutions
```bash
# Restart services to clear memory
docker-compose restart app worker

# Optimize database connections
# Edit config to reduce connection pool size
DATABASE_POOL_SIZE=10
DATABASE_MAX_OVERFLOW=20

# Clear Python caches
docker-compose exec app python -c "
import gc
gc.collect()
"
```

### Slow Database Queries

#### Investigation
```bash
# Enable query logging temporarily
docker-compose exec postgres psql -U tariff_user -d tariff_radar -c "
ALTER SYSTEM SET log_statement = 'all';
ALTER SYSTEM SET log_min_duration_statement = 1000;
SELECT pg_reload_conf();
"

# Check slow queries
docker-compose exec postgres psql -U tariff_user -d tariff_radar -c "
SELECT query, mean_time, calls, total_time, stddev_time
FROM pg_stat_statements 
WHERE mean_time > 100
ORDER BY mean_time DESC 
LIMIT 20;
"

# Check table sizes
docker-compose exec postgres psql -U tariff_user -d tariff_radar -c "
SELECT schemaname, tablename, 
       pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size,
       pg_total_relation_size(schemaname||'.'||tablename) as bytes
FROM pg_tables 
WHERE schemaname = 'public' 
ORDER BY bytes DESC;
"
```

#### Solutions
```bash
# Analyze and vacuum tables
docker-compose exec postgres psql -U tariff_user -d tariff_radar -c "
VACUUM ANALYZE;
"

# Reindex tables if needed
docker-compose exec postgres psql -U tariff_user -d tariff_radar -c "
REINDEX DATABASE tariff_radar;
"

# Check and create missing indexes
docker-compose exec postgres psql -U tariff_user -d tariff_radar -c "
SELECT schemaname, tablename, attname, n_distinct, correlation 
FROM pg_stats 
WHERE schemaname = 'public' 
ORDER BY n_distinct DESC;
"
```

### High CPU Usage

#### Investigation
```bash
# Check CPU usage by container
docker stats --format "table {{.Name}}\t{{.CPUPerc}}\t{{.PIDs}}"

# Check for CPU-intensive processes
docker-compose exec app top -o %CPU

# Check for busy loops
strace -p $(docker-compose exec app pgrep python) -c
```

#### Solutions
```bash
# Optimize worker configuration
# Reduce worker concurrency
CELERY_WORKER_CONCURRENCY=2

# Add CPU limits
services:
  app:
    cpus: '2.0'
  worker:
    cpus: '1.0'

# Check for inefficient algorithms
grep -i "loop\|while\|for" logs/tariff-radar_performance.json
```

## Database Issues

### Connection Pool Exhaustion

#### Symptoms
- "connection pool exhausted" errors
- Long wait times for database operations
- Application hangs on database operations

#### Investigation
```bash
# Check current connections
docker-compose exec postgres psql -U tariff_user -d tariff_radar -c "
SELECT count(*) as total_connections,
       count(*) FILTER (WHERE state = 'active') as active_connections,
       count(*) FILTER (WHERE state = 'idle') as idle_connections
FROM pg_stat_activity 
WHERE datname = 'tariff_radar';
"

# Check connection configuration
docker-compose exec postgres psql -U tariff_user -d tariff_radar -c "
SHOW max_connections;
SHOW shared_buffers;
"
```

#### Solutions
```bash
# Kill idle connections
docker-compose exec postgres psql -U tariff_user -d tariff_radar -c "
SELECT pg_terminate_backend(pid)
FROM pg_stat_activity 
WHERE datname = 'tariff_radar' 
  AND state = 'idle' 
  AND state_change < now() - interval '1 hour';
"

# Increase connection limit (restart required)
# Edit postgresql.conf
max_connections = 200
shared_buffers = 256MB

# Optimize application connection pooling
DATABASE_POOL_SIZE=20
DATABASE_POOL_RECYCLE=3600
```

### Database Corruption

#### Symptoms
- Checksum failures
- Data inconsistency errors
- Unexpected query results

#### Investigation
```bash
# Check database integrity
docker-compose exec postgres psql -U tariff_user -d tariff_radar -c "
SELECT datname, checksum_failures, checksum_last_failure 
FROM pg_stat_database 
WHERE datname = 'tariff_radar';
"

# Check for corruption
docker-compose exec postgres psql -U tariff_user -d tariff_radar -c "
SELECT schemaname, tablename, 
       pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
FROM pg_tables 
WHERE schemaname = 'public';
"
```

#### Solutions
```bash
# Repair corruption
docker-compose exec postgres psql -U tariff_user -d tariff_radar -c "
REINDEX DATABASE tariff_radar;
"

# If severe corruption, restore from backup
docker-compose exec -T postgres psql -U tariff_user tariff_radar < backup.sql
```

## Network and Connectivity

### External API Failures

#### Symptoms
- RSS feed fetch failures
- OpenAI API timeouts
- Notification sending failures

#### Investigation
```bash
# Test external connectivity
docker-compose exec app curl -I https://api.openai.com/v1/models

# Check DNS resolution
docker-compose exec app nslookup openai.com

# Test specific endpoints
docker-compose exec app curl -v https://feeds.reuters.com/reuters/businessNews
```

#### Solutions
```bash
# Configure proxy if needed
export HTTP_PROXY=http://proxy.company.com:8080
export HTTPS_PROXY=http://proxy.company.com:8080

# Add DNS servers
echo "nameserver 8.8.8.8" >> /etc/resolv.conf

# Configure timeout settings
EXTERNAL_API_TIMEOUT=30
RSS_FETCH_TIMEOUT=60
```

### Internal Service Communication

#### Symptoms
- Service discovery failures
- Connection refused errors between containers
- Intermittent connectivity issues

#### Investigation
```bash
# Check container network
docker network ls
docker network inspect tariff-radar_default

# Test internal connectivity
docker-compose exec app ping postgres
docker-compose exec app ping redis

# Check port binding
docker-compose exec app netstat -tulpn
```

#### Solutions
```bash
# Restart networking
docker-compose down
docker network prune
docker-compose up -d

# Use explicit container names
# In docker-compose.yml, ensure service names are consistent
```

## Configuration Problems

### Environment Variable Issues

#### Symptoms
- Services start with default values
- Configuration not applied
- Authentication failures

#### Investigation
```bash
# Check environment variables
docker-compose exec app env | grep -E "(DATABASE|REDIS|API)"

# Validate configuration loading
docker-compose exec app python -c "
from app.config.settings import get_settings
settings = get_settings()
print(f'Database URL: {settings.DATABASE_URL}')
print(f'Redis URL: {settings.REDIS_URL}')
"
```

#### Solutions
```bash
# Reload environment variables
docker-compose down
docker-compose up -d

# Check .env file syntax
cat .env | grep -v "^#" | grep -v "^$"

# Validate YAML configuration
python -c "
import yaml
with open('config/rss_sources.yaml') as f:
    yaml.safe_load(f)
print('YAML is valid')
"
```

### SSL/TLS Certificate Issues

#### Symptoms
- HTTPS connection failures
- Certificate validation errors
- SSL handshake failures

#### Investigation
```bash
# Check certificate validity
openssl x509 -in /path/to/cert.pem -text -noout

# Test SSL connection
openssl s_client -connect your-domain.com:443

# Check certificate chain
curl -vI https://your-domain.com
```

#### Solutions
```bash
# Renew certificate
certbot renew --nginx

# Update certificate in configuration
cp /etc/letsencrypt/live/your-domain.com/fullchain.pem ssl/
cp /etc/letsencrypt/live/your-domain.com/privkey.pem ssl/

# Restart services with new certificate
docker-compose restart nginx app
```

## Security Issues

### Authentication Failures

#### Symptoms
- Login failures with correct credentials
- API key validation errors
- JWT token expired errors

#### Investigation
```bash
# Check authentication logs
grep -i "authentication" logs/tariff-radar_security.json

# Test API key
curl -H "X-API-Key: your_api_key" http://localhost:8000/api/articles

# Check JWT token
python -c "
import jwt
token = 'your_jwt_token'
print(jwt.decode(token, options={'verify_signature': False}))
"
```

#### Solutions
```bash
# Reset API keys
API_KEY_ADMIN=new_secure_api_key_here

# Clear authentication cache
docker-compose exec redis redis-cli DEL auth:*

# Check system time (JWT issues)
date
docker-compose exec app date
```

### Rate Limiting Issues

#### Symptoms
- Requests blocked by rate limiter
- 429 Too Many Requests errors
- Legitimate users blocked

#### Investigation
```bash
# Check rate limit status
docker-compose exec redis redis-cli KEYS "rate_limit:*"

# Check rate limit counters
docker-compose exec redis redis-cli GET "rate_limit:api:192.168.1.100"

# Review rate limit logs
grep "rate_limit" logs/tariff-radar_security.json
```

#### Solutions
```bash
# Clear rate limit counters
docker-compose exec redis redis-cli DEL "rate_limit:*"

# Adjust rate limits
RATE_LIMIT_REQUESTS=2000
RATE_LIMIT_WINDOW=3600

# Whitelist specific IPs
RATE_LIMIT_WHITELIST=192.168.1.0/24,10.0.0.0/8
```

## Data Processing Issues

### RSS Feed Processing Failures

#### Symptoms
- No new articles being processed
- RSS fetch errors in logs
- Stale data in dashboard

#### Investigation
```bash
# Check RSS source status
curl -H "X-API-Key: your_api_key" http://localhost:8000/api/sources

# Test RSS feed manually
curl -I https://feeds.reuters.com/reuters/businessNews

# Check processing logs
grep -i "rss\|feed" logs/tariff-radar.json
```

#### Solutions
```bash
# Manually trigger RSS processing
curl -X POST -H "X-API-Key: your_api_key" \
  http://localhost:8000/admin/process-feeds

# Update RSS source configuration
# Edit config/rss_sources.yaml

# Clear processing cache
docker-compose exec redis redis-cli DEL "rss:*"
```

### AI Processing Failures

#### Symptoms
- Articles stuck in pending status
- OpenAI API errors
- Classification not working

#### Investigation
```bash
# Check AI service configuration
echo $OPENAI_API_KEY | cut -c1-10

# Test AI API connectivity
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
  https://api.openai.com/v1/models

# Check processing queue
docker-compose exec worker celery -A sched.tasks inspect active
```

#### Solutions
```bash
# Retry failed processing
curl -X POST -H "X-API-Key: your_api_key" \
  http://localhost:8000/admin/retry-failed

# Clear processing queue
docker-compose exec worker celery -A sched.tasks purge

# Update AI model configuration
AI_MODEL=gpt-3.5-turbo
AI_MAX_TOKENS=1000
```

## Notification Problems

### Email Notifications Not Sent

#### Symptoms
- No email notifications received
- SMTP connection errors
- Email bounce-backs

#### Investigation
```bash
# Test SMTP configuration
python -c "
import smtplib
server = smtplib.SMTP('$SMTP_SERVER', $SMTP_PORT)
server.starttls()
server.login('$SMTP_USERNAME', '$SMTP_PASSWORD')
print('SMTP connection successful')
server.quit()
"

# Check notification logs
grep -i "email\|smtp" logs/tariff-radar.json
```

#### Solutions
```bash
# Update SMTP configuration
SMTP_SERVER=smtp.company.com
SMTP_PORT=587
SMTP_USE_TLS=true

# Test email sending
curl -X POST -H "X-API-Key: your_api_key" \
  http://localhost:8000/monitoring/test/alert

# Check firewall rules for SMTP
telnet smtp.company.com 587
```

### WeCom/Telegram Notifications Failed

#### Symptoms
- Webhook timeouts
- Invalid bot token errors
- Messages not delivered

#### Investigation
```bash
# Test webhook URL
curl -X POST "$WECOM_WEBHOOK_URL" \
  -H "Content-Type: application/json" \
  -d '{"msgtype":"text","text":{"content":"Test message"}}'

# Test Telegram bot
curl "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/getMe"
```

#### Solutions
```bash
# Update webhook URLs
WECOM_WEBHOOK_URL=https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=new_key
TELEGRAM_BOT_TOKEN=new_bot_token

# Check network connectivity
curl -I https://api.telegram.org
curl -I https://qyapi.weixin.qq.com
```

## Diagnostic Tools

### Health Check Scripts

```bash
#!/bin/bash
# comprehensive_health_check.sh

echo "=== Comprehensive System Health Check ==="

# Basic connectivity
echo "1. Basic Health Check:"
curl -s http://localhost:8000/monitoring/health | jq .status

# Detailed health
echo "2. Detailed Health Check:"
curl -s http://localhost:8000/monitoring/health?detailed=true | \
  jq '.checks[] | {component: .component, status: .status, response_time_ms: .response_time_ms}'

# Resource usage
echo "3. Resource Usage:"
docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}"

# Database health
echo "4. Database Health:"
docker-compose exec postgres pg_isready -U tariff_user

# Redis health
echo "5. Redis Health:"
docker-compose exec redis redis-cli ping

# Queue status
echo "6. Queue Status:"
docker-compose exec worker celery -A sched.tasks inspect active | \
  python -c "import sys, json; data=json.load(sys.stdin); print(f'Active tasks: {sum(len(v) for v in data.values())}')"

echo "=== Health Check Complete ==="
```

### Performance Monitoring Script

```bash
#!/bin/bash
# performance_monitor.sh

LOGFILE="performance_$(date +%Y%m%d_%H%M%S).log"

echo "Starting performance monitoring - Press Ctrl+C to stop"
echo "Logging to: $LOGFILE"

while true; do
  echo "$(date): $(docker stats --no-stream --format '{{.Name}}: CPU={{.CPUPerc}} MEM={{.MemPerc}}')" >> $LOGFILE
  
  # Check response time
  RESPONSE_TIME=$(curl -w "%{time_total}" -o /dev/null -s http://localhost:8000/health)
  echo "$(date): API Response Time: ${RESPONSE_TIME}s" >> $LOGFILE
  
  # Check queue length
  QUEUE_LENGTH=$(docker-compose exec worker celery -A sched.tasks inspect active 2>/dev/null | \
    python -c "import sys, json; data=json.load(sys.stdin); print(sum(len(v) for v in data.values()))" 2>/dev/null || echo "0")
  echo "$(date): Queue Length: $QUEUE_LENGTH" >> $LOGFILE
  
  sleep 60
done
```

### Log Analysis Tools

```bash
#!/bin/bash
# analyze_logs.sh

echo "=== Log Analysis Report ==="

# Error summary
echo "1. Error Summary (last 24 hours):"
find logs/ -name "*.json" -mtime -1 -exec \
  jq -r 'select(.level=="ERROR") | .timestamp + " " + .logger + ": " + .message' {} \; | \
  sort | uniq -c | sort -nr | head -10

# Performance issues
echo "2. Slow Operations (>2 seconds):"
find logs/ -name "*.json" -mtime -1 -exec \
  jq -r 'select(.duration_seconds > 2) | .timestamp + " " + .operation + ": " + (.duration_seconds|tostring) + "s"' {} \; | \
  sort | head -10

# Security events
echo "3. Security Events:"
find logs/ -name "*security.json" -mtime -1 -exec \
  jq -r 'select(.security_event==true) | .timestamp + " " + .event_type + ": " + .description' {} \; | \
  sort | tail -10

# Resource usage trends
echo "4. Resource Usage Patterns:"
grep -h "cpu_usage\|memory_usage" logs/tariff-radar.json | \
  jq -r '.timestamp + " CPU:" + (.cpu_usage|tostring) + "% MEM:" + (.memory_usage|tostring) + "%' | \
  tail -20

echo "=== Analysis Complete ==="
```

## Recovery Procedures

### Data Recovery

```bash
#!/bin/bash
# data_recovery.sh

BACKUP_DATE=$1
if [ -z "$BACKUP_DATE" ]; then
  echo "Usage: $0 YYYY-MM-DD"
  exit 1
fi

echo "Recovering data from backup: $BACKUP_DATE"

# Stop services
docker-compose stop app worker

# Create current backup
echo "Creating current backup..."
docker-compose exec postgres pg_dump -U tariff_user tariff_radar > "current_backup_$(date +%Y%m%d_%H%M%S).sql"

# Restore from backup
echo "Restoring from backup..."
if [ -f "backup_${BACKUP_DATE}.sql.gz" ]; then
  gunzip -c "backup_${BACKUP_DATE}.sql.gz" | \
    docker-compose exec -T postgres psql -U tariff_user tariff_radar
else
  echo "Backup file not found: backup_${BACKUP_DATE}.sql.gz"
  exit 1
fi

# Restart services
echo "Restarting services..."
docker-compose start app worker

# Verify recovery
echo "Verifying recovery..."
curl -s http://localhost:8000/monitoring/health

echo "Data recovery complete"
```

### Configuration Recovery

```bash
#!/bin/bash
# config_recovery.sh

echo "Recovering configuration..."

# Backup current config
cp -r config/ "config_backup_$(date +%Y%m%d_%H%M%S)/"

# Restore from version control or backup
git checkout HEAD -- config/
# OR
# cp -r config_backup_YYYYMMDD/* config/

# Restart services to reload config
docker-compose restart app worker

echo "Configuration recovery complete"
```

---

## Emergency Contacts

- **On-Call Engineer**: [Contact Information]
- **Database Administrator**: [Contact Information]
- **Security Team**: [Contact Information]
- **Infrastructure Team**: [Contact Information]

## Additional Resources

- **System Documentation**: `docs/DEPLOYMENT.md`
- **API Documentation**: `docs/API.md`
- **Monitoring Runbook**: `docs/MONITORING.md`
- **Configuration Guide**: `config/README.md`

Remember to keep this troubleshooting guide updated as new issues are discovered and resolved.