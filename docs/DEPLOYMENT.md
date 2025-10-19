# Tariff Radar - Production Deployment Guide

## Table of Contents
1. [System Overview](#system-overview)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Deployment](#deployment)
6. [Monitoring & Maintenance](#monitoring--maintenance)
7. [Troubleshooting](#troubleshooting)
8. [Security Considerations](#security-considerations)

## System Overview

Tariff Radar is a comprehensive US-China trade monitoring system that provides:

- **Real-time news monitoring** from multiple RSS sources
- **AI-powered content analysis** with multilingual support
- **Smart filtering and classification** using ML and LLMs
- **Multi-channel notifications** (WeCom, Email, Telegram)
- **Production-grade monitoring** with Prometheus metrics
- **Enterprise security** with JWT authentication and rate limiting

### Architecture Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI Web   │    │   PostgreSQL    │    │     Redis       │
│   Dashboard     │    │   Database      │    │   Cache/Queue   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Qdrant Vector │    │  Celery Workers │    │   Monitoring    │
│   Database      │    │  (Background)   │    │   & Alerting    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Prerequisites

### System Requirements

**Minimum Requirements:**
- CPU: 4 cores
- RAM: 8GB
- Storage: 50GB SSD
- Network: Stable internet connection

**Recommended for Production:**
- CPU: 8 cores
- RAM: 16GB
- Storage: 100GB SSD
- Network: High-speed internet with redundancy

### Software Dependencies

- **Docker Engine**: 24.0+
- **Docker Compose**: 2.20+
- **Python**: 3.11+ (if running without Docker)
- **PostgreSQL**: 15+ (if external database)
- **Redis**: 7.0+ (if external cache)

### External Services

- **OpenAI API** (for LLM analysis)
- **WeCom Webhook** (for notifications)
- **SMTP Server** (for email notifications)
- **Telegram Bot** (for Telegram notifications)

## Installation

### 1. Clone Repository

```bash
git clone <repository-url>
cd tariff-radar
```

### 2. Environment Setup

Create environment files from templates:

```bash
# Main application environment
cp .env.template .env

# Monitoring configuration
cp monitoring.env.template monitoring.env

# Docker environment
cp docker-compose.env.template docker-compose.env
```

### 3. Configure Environment Variables

Edit `.env` with your settings:

```bash
# Database Configuration
DATABASE_URL=postgresql://tariff_user:secure_password@postgres:5432/tariff_radar
REDIS_URL=redis://redis:6379/0

# AI Service Configuration
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Notification Configuration
WECOM_WEBHOOK_URL=https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=your_key
SMTP_SERVER=smtp.company.com
SMTP_PORT=587
SMTP_USERNAME=notifications@company.com
SMTP_PASSWORD=your_smtp_password
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_chat_id

# Security Configuration
SECRET_KEY=your_super_secret_jwt_key_here
API_KEY_ADMIN=your_admin_api_key_here
ADMIN_USERS=admin@company.com,manager@company.com

# Application Configuration
RSS_SOURCES_CONFIG=config/rss_sources.yaml
KEYWORDS_CONFIG=config/keywords.yaml
```

### 4. Configure Monitoring

Edit `monitoring.env`:

```bash
# Environment
ENVIRONMENT=production

# Monitoring Settings
PROMETHEUS_ENABLED=true
ALERTS_ENABLED=true
METRICS_COLLECTION_INTERVAL=30
ALERT_CHECK_INTERVAL=60

# Alert Thresholds
CPU_ALERT_THRESHOLD=85.0
MEMORY_ALERT_THRESHOLD=90.0
DISK_ALERT_THRESHOLD=80.0

# Notification Channels
DEFAULT_NOTIFICATION_CHANNELS=wecom,email
CRITICAL_ALERT_CHANNELS=wecom,email,telegram
```

## Configuration

### 1. RSS Sources Configuration

Create `config/rss_sources.yaml`:

```yaml
sources:
  - name: "Reuters Trade"
    url: "https://feeds.reuters.com/reuters/businessNews"
    enabled: true
    fetch_interval: 300
    language: "en"
    
  - name: "Xinhua Economics"
    url: "http://www.xinhuanet.com/english/rss/economic.xml"
    enabled: true
    fetch_interval: 600
    language: "en"
    
  - name: "SCMP Business"
    url: "https://www.scmp.com/rss/91/feed"
    enabled: true
    fetch_interval: 300
    language: "en"
```

### 2. Keywords Configuration

Create `config/keywords.yaml`:

```yaml
keywords:
  trade_war:
    - "trade war"
    - "tariff"
    - "import duty"
    - "trade dispute"
    - "商贸战"
    - "关税"
    
  sanctions:
    - "sanctions"
    - "embargo"
    - "restrictions"
    - "制裁"
    
  companies:
    - "Huawei"
    - "TikTok"
    - "SMIC"
    - "ByteDance"
    
  economic_indicators:
    - "GDP growth"
    - "inflation"
    - "unemployment"
    - "manufacturing PMI"
```

### 3. Database Initialization

The system will automatically create database tables on startup. For manual initialization:

```bash
# Access database container
docker-compose exec postgres psql -U tariff_user -d tariff_radar

# Create extensions if needed
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
```

## Deployment

### 1. Docker Compose Deployment (Recommended)

```bash
# Build and start all services
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f app

# Scale Celery workers if needed
docker-compose up -d --scale worker=3
```

### 2. Production Docker Compose

For production, use the production override:

```bash
# Production deployment
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# With monitoring stack
docker-compose -f docker-compose.yml -f docker-compose.monitoring.yml up -d
```

### 3. Kubernetes Deployment

For Kubernetes deployment, see `k8s/` directory:

```bash
# Apply namespace
kubectl apply -f k8s/namespace.yaml

# Apply configurations
kubectl apply -f k8s/configmaps/
kubectl apply -f k8s/secrets/

# Deploy services
kubectl apply -f k8s/postgres/
kubectl apply -f k8s/redis/
kubectl apply -f k8s/qdrant/
kubectl apply -f k8s/app/

# Check deployment
kubectl get pods -n tariff-radar
kubectl get services -n tariff-radar
```

### 4. Health Check Verification

After deployment, verify system health:

```bash
# Basic health check
curl http://localhost:8000/health

# Detailed health check
curl http://localhost:8000/monitoring/health?detailed=true

# System status
curl http://localhost:8000/monitoring/system/status
```

## Monitoring & Maintenance

### 1. Monitoring Dashboard

Access the monitoring dashboard at:
- **Main Dashboard**: `http://localhost:8000/dashboard`
- **Monitoring API**: `http://localhost:8000/monitoring/dashboard`
- **Prometheus Metrics**: `http://localhost:8000/monitoring/metrics`

### 2. Log Management

Logs are stored in the `logs/` directory:

```bash
# View application logs
tail -f logs/tariff-radar.json

# View error logs
tail -f logs/tariff-radar_errors.json

# View security logs
tail -f logs/tariff-radar_security.json

# View performance logs
tail -f logs/tariff-radar_performance.json
```

### 3. Database Maintenance

```bash
# Database backup
docker-compose exec postgres pg_dump -U tariff_user tariff_radar > backup_$(date +%Y%m%d).sql

# Database restore
docker-compose exec -T postgres psql -U tariff_user tariff_radar < backup_20241017.sql

# Check database size
docker-compose exec postgres psql -U tariff_user -d tariff_radar -c "
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
FROM pg_tables 
WHERE schemaname = 'public' 
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
"
```

### 4. Performance Optimization

```bash
# Check Redis memory usage
docker-compose exec redis redis-cli info memory

# Monitor Celery queue
docker-compose exec worker celery -A sched.tasks inspect active

# Check system resources
docker stats

# Database connection monitoring
docker-compose exec postgres psql -U tariff_user -d tariff_radar -c "
SELECT count(*) as connection_count, state 
FROM pg_stat_activity 
GROUP BY state;
"
```

## Troubleshooting

### Common Issues

#### 1. Service Startup Failures

**Symptom**: Services fail to start or crash immediately

**Solutions**:
```bash
# Check logs for specific service
docker-compose logs service_name

# Check resource availability
docker system df
docker system prune

# Restart specific service
docker-compose restart service_name

# Full rebuild
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

#### 2. Database Connection Issues

**Symptom**: `ConnectionError` or database timeout errors

**Solutions**:
```bash
# Check database status
docker-compose exec postgres pg_isready -U tariff_user

# Check database connections
docker-compose exec postgres psql -U tariff_user -d tariff_radar -c "
SELECT count(*) FROM pg_stat_activity WHERE datname = 'tariff_radar';
"

# Restart database
docker-compose restart postgres

# Check database logs
docker-compose logs postgres
```

#### 3. High Memory Usage

**Symptom**: System becomes slow or services get killed

**Solutions**:
```bash
# Check memory usage by service
docker stats --no-stream

# Reduce Celery workers
docker-compose up -d --scale worker=1

# Clear Redis cache
docker-compose exec redis redis-cli FLUSHDB

# Restart services with memory limits
docker-compose down
docker-compose up -d
```

#### 4. SSL/TLS Certificate Issues

**Symptom**: HTTPS connection errors or certificate warnings

**Solutions**:
```bash
# Check certificate expiry
openssl x509 -in cert.pem -text -noout | grep "Not After"

# Renew Let's Encrypt certificate
certbot renew --nginx

# Test SSL configuration
curl -I https://your-domain.com
```

### 5. Performance Issues

**Symptom**: Slow response times or high CPU usage

**Solutions**:
```bash
# Check application metrics
curl http://localhost:8000/monitoring/metrics | grep http_request_duration

# Profile database queries
docker-compose exec postgres psql -U tariff_user -d tariff_radar -c "
SELECT query, mean_time, calls 
FROM pg_stat_statements 
ORDER BY mean_time DESC LIMIT 10;
"

# Check Celery task performance
docker-compose exec worker celery -A sched.tasks inspect stats

# Optimize database
docker-compose exec postgres psql -U tariff_user -d tariff_radar -c "VACUUM ANALYZE;"
```

### Emergency Procedures

#### 1. System Recovery

```bash
# Stop all services
docker-compose down

# Backup current state
cp -r logs/ logs_backup_$(date +%Y%m%d)
docker-compose exec postgres pg_dump -U tariff_user tariff_radar > emergency_backup.sql

# Restart with minimal services
docker-compose up -d postgres redis
sleep 30
docker-compose up -d app

# Check health
curl http://localhost:8000/health
```

#### 2. Data Recovery

```bash
# Restore from backup
docker-compose exec -T postgres psql -U tariff_user tariff_radar < backup.sql

# Rebuild search indices
curl -X POST http://localhost:8000/admin/rebuild-indices \
  -H "X-API-Key: your_api_key"

# Clear caches
docker-compose exec redis redis-cli FLUSHALL
```

## Security Considerations

### 1. Production Security Checklist

- [ ] Change all default passwords
- [ ] Configure strong JWT secret keys
- [ ] Set up proper SSL/TLS certificates
- [ ] Configure firewall rules
- [ ] Enable audit logging
- [ ] Set up intrusion detection
- [ ] Configure backup encryption
- [ ] Review API key access
- [ ] Enable rate limiting
- [ ] Configure security headers

### 2. Network Security

```bash
# Firewall configuration (Ubuntu/Debian)
ufw allow 22    # SSH
ufw allow 80    # HTTP
ufw allow 443   # HTTPS
ufw deny 5432   # PostgreSQL (internal only)
ufw deny 6379   # Redis (internal only)
ufw enable

# Check open ports
netstat -tulpn | grep LISTEN
```

### 3. Regular Security Updates

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Update Docker images
docker-compose pull
docker-compose up -d

# Check for vulnerabilities
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  aquasec/trivy image tariff-radar:latest
```

### 4. Backup Strategy

```bash
# Automated backup script
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups"

# Database backup
docker-compose exec postgres pg_dump -U tariff_user tariff_radar | \
  gzip > "$BACKUP_DIR/db_backup_$DATE.sql.gz"

# Application data backup
tar -czf "$BACKUP_DIR/app_data_$DATE.tar.gz" \
  logs/ config/ uploads/

# Upload to cloud storage
aws s3 cp "$BACKUP_DIR/" s3://your-backup-bucket/ --recursive --exclude "*" --include "*$DATE*"

# Cleanup old backups (keep 30 days)
find "$BACKUP_DIR" -name "*.gz" -mtime +30 -delete
```

---

## Support

For support and questions:
- **Documentation**: Check this guide and API documentation
- **Monitoring**: Use `/monitoring/dashboard` for system status
- **Logs**: Check application logs in `logs/` directory
- **Health Checks**: Monitor `/monitoring/health` endpoint

## Version Information

- **Application Version**: 1.0.0
- **Docker Compose Version**: 2.20+
- **Python Version**: 3.11+
- **Documentation Version**: 1.0 (October 2025)