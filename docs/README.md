# Tariff Radar - Complete Documentation

## 📚 Documentation Overview

Welcome to the comprehensive documentation for **Tariff Radar**, a production-ready US-China trade monitoring system with advanced AI analysis and enterprise-grade monitoring.

## 📋 Table of Contents

### 🚀 [Deployment Guide](DEPLOYMENT.md)
Complete production deployment instructions including:
- System requirements and prerequisites
- Docker deployment with monitoring stack
- Kubernetes deployment configurations
- Environment configuration and security setup
- Health verification and troubleshooting

### 🔧 [API Documentation](API.md)
Comprehensive REST API reference covering:
- Authentication and authorization
- Core endpoints (articles, sources, analytics)
- Monitoring and alerting APIs
- WebSocket real-time updates
- Error handling and rate limiting
- Python client examples

### 📊 [Monitoring Runbook](MONITORING.md)
Operational procedures for system monitoring:
- Monitoring dashboard guide
- Alert response procedures (Critical/High/Medium/Low)
- Performance monitoring and optimization
- Log analysis and security monitoring
- Emergency response and escalation procedures

### 🔍 [Troubleshooting Guide](TROUBLESHOOTING.md)
Step-by-step solutions for common issues:
- Application startup and performance problems
- Database and connectivity issues
- Configuration and security problems
- Data processing and notification failures
- Diagnostic tools and recovery procedures

## 🏗️ System Architecture

### Core Components

```mermaid
graph TB
    A[FastAPI Web App] --> B[PostgreSQL Database]
    A --> C[Redis Cache/Queue]
    A --> D[Qdrant Vector DB]
    E[Celery Workers] --> B
    E --> C
    E --> D
    F[RSS Sources] --> E
    G[AI Services] --> E
    E --> H[Notifications]
    I[Monitoring] --> A
    I --> B
    I --> C
    I --> D
```

### Key Features

✅ **Real-time Trade Monitoring**
- Multi-source RSS feed aggregation
- Advanced content deduplication
- Multilingual content processing

✅ **AI-Powered Analysis**
- Keyword-based filtering
- Semantic similarity search
- ML classification models
- LLM-powered content triage

✅ **Multi-Channel Notifications**
- WeCom enterprise messaging
- SMTP email notifications
- Telegram bot integration

✅ **Production Monitoring**
- Prometheus metrics collection
- Structured JSON logging
- Health checks and alerting
- Performance tracking

✅ **Enterprise Security**
- JWT authentication
- API key management
- Rate limiting
- Input validation and audit logging

## 🔧 Quick Start

### 1. Clone and Setup
```bash
git clone <repository-url>
cd tariff-radar
cp .env.template .env
cp monitoring.env.template monitoring.env
```

### 2. Configure Environment
Edit `.env` with your settings:
```bash
# Database
DATABASE_URL=postgresql://tariff_user:secure_password@postgres:5432/tariff_radar

# AI Services
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
PERPLEXITY_API_KEY=your_perplexity_api_key_here

# Notifications
WECOM_WEBHOOK_URL=https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=your_key
SMTP_SERVER=smtp.company.com
TELEGRAM_BOT_TOKEN=your_telegram_bot_token

# Security
SECRET_KEY=your_super_secret_jwt_key_here
API_KEY_ADMIN=your_admin_api_key_here
```

### 3. Deploy with Docker
```bash
# Start all services
docker-compose up -d

# Check health
curl http://localhost:8000/monitoring/health

# Access dashboard
open http://localhost:8000/dashboard
```

### 4. Configure Sources
Edit `config/rss_sources.yaml`:
```yaml
sources:
  - name: "Reuters Trade"
    url: "https://feeds.reuters.com/reuters/businessNews"
    enabled: true
    fetch_interval: 300
    language: "en"
```

## 📊 Monitoring Overview

### Key Endpoints
- **Health Check**: `GET /monitoring/health`
- **System Status**: `GET /monitoring/system/status`
- **Prometheus Metrics**: `GET /monitoring/metrics`
- **Active Alerts**: `GET /monitoring/alerts`
- **Dashboard**: `GET /monitoring/dashboard`

### Alert Severity Levels
- 🔴 **Critical**: Immediate response (< 5 min) - System down, data loss
- 🟡 **High**: 15 minute response - Performance degradation, service issues  
- 🔵 **Medium**: 1 hour response - Minor issues, optimization needed
- ⚪ **Low**: 4 hour response - Maintenance, trend analysis

### Performance Thresholds
| Metric | Good | Warning | Critical |
|--------|------|---------|----------|
| CPU Usage | < 70% | 70-85% | > 85% |
| Memory Usage | < 80% | 80-90% | > 90% |
| Response Time | < 500ms | 500ms-2s | > 2s |
| Error Rate | < 1% | 1-5% | > 5% |

## 🔒 Security Configuration

### Authentication Setup
```bash
# Generate secure API keys
API_KEY_ADMIN=$(openssl rand -hex 32)
API_KEY_USER=$(openssl rand -hex 32)

# Set admin users
ADMIN_USERS=admin@company.com,manager@company.com

# Configure JWT
SECRET_KEY=$(openssl rand -hex 32)
ACCESS_TOKEN_EXPIRE_MINUTES=30
```

### Rate Limiting
```bash
# API rate limits
RATE_LIMIT_REQUESTS=1000
RATE_LIMIT_WINDOW=3600

# Authentication limits
AUTH_RATE_LIMIT=10
AUTH_RATE_WINDOW=60
```

## 📈 Performance Optimization

### Database Optimization
```sql
-- Regular maintenance
VACUUM ANALYZE;

-- Check slow queries
SELECT query, mean_time, calls 
FROM pg_stat_statements 
WHERE mean_time > 1000 
ORDER BY mean_time DESC 
LIMIT 10;

-- Monitor connections
SELECT count(*) as connections, state 
FROM pg_stat_activity 
GROUP BY state;
```

### Application Tuning
```bash
# Scale workers based on load
docker-compose up -d --scale worker=3

# Monitor queue length
docker-compose exec worker celery -A sched.tasks inspect active

# Check resource usage
docker stats --no-stream
```

## 🔄 Backup and Recovery

### Automated Backups
```bash
#!/bin/bash
# Daily backup script
DATE=$(date +%Y%m%d_%H%M%S)

# Database backup
docker-compose exec postgres pg_dump -U tariff_user tariff_radar | \
  gzip > "backups/db_backup_$DATE.sql.gz"

# Configuration backup
tar -czf "backups/config_backup_$DATE.tar.gz" config/ .env

# Upload to cloud storage (optional)
aws s3 cp backups/ s3://your-backup-bucket/ --recursive
```

### Recovery Procedures
```bash
# Database recovery
gunzip -c backup_YYYYMMDD.sql.gz | \
  docker-compose exec -T postgres psql -U tariff_user tariff_radar

# Configuration recovery
tar -xzf config_backup_YYYYMMDD.tar.gz

# Restart services
docker-compose restart app worker
```

## 🛠️ Development Setup

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Setup development database
createdb tariff_radar_dev

# Run migrations
alembic upgrade head

# Start development server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Testing
```bash
# Run full test suite
python -m pytest tests/ -v

# Run specific test categories
python tests/run_tests.py --category unit
python tests/run_tests.py --category integration
python tests/run_tests.py --category security
python tests/run_tests.py --category performance

# Check test coverage
pytest --cov=app tests/
```

## 📞 Support and Escalation

### Contact Information
- **On-Call Engineer**: [Contact Details]
- **Team Lead**: [Contact Details]
- **System Administrator**: [Contact Details]
- **Emergency**: [Emergency Contact]

### Escalation Matrix
| Severity | Response Time | Contact |
|----------|---------------|---------|
| Critical | 5 minutes | On-call Engineer |
| High | 15 minutes | Team Member |
| Medium | 1 hour | Team Member |
| Low | 4 hours | Team Member |

### Support Channels
- **Documentation**: Check guides in `docs/` directory
- **Monitoring**: Use monitoring dashboard for real-time status
- **Logs**: Check application logs in `logs/` directory
- **Health Checks**: Monitor `/monitoring/health` endpoint

## 📝 Change Management

### Configuration Changes
1. Update configuration files in `config/`
2. Test changes in development environment
3. Create backup of current configuration
4. Apply changes during maintenance window
5. Verify system health after changes
6. Document changes in change log

### Deployment Updates
1. Review change requests and approvals
2. Schedule maintenance window
3. Create system backup
4. Deploy changes using blue-green strategy
5. Verify all monitoring checks pass
6. Monitor system for 30 minutes post-deployment
7. Document deployment and any issues

## 🏷️ Version Information

- **Application Version**: 1.0.0
- **Documentation Version**: 1.0
- **Last Updated**: October 2025
- **Supported Python**: 3.11+
- **Docker Compose**: 2.20+
- **Database**: PostgreSQL 15+

---

## 📖 Additional Resources

- **GitHub Repository**: [Repository URL]
- **Issue Tracker**: [Issue Tracker URL]
- **Change Log**: `CHANGELOG.md`
- **Security Policy**: `SECURITY.md`
- **Contributing Guidelines**: `CONTRIBUTING.md`

For questions not covered in this documentation, please check the monitoring dashboard or contact the support team using the escalation procedures above.