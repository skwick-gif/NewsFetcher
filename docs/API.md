# Tariff Radar API Documentation

## Table of Contents
1. [Overview](#overview)
2. [Authentication](#authentication)
3. [Core API Endpoints](#core-api-endpoints)
4. [Monitoring API](#monitoring-api)
5. [Administration API](#administration-api)
6. [WebSocket Endpoints](#websocket-endpoints)
7. [Error Handling](#error-handling)
8. [Rate Limiting](#rate-limiting)
9. [API Examples](#api-examples)

## Overview

Tariff Radar provides a comprehensive REST API for managing trade monitoring data, accessing analysis results, and system administration. All API endpoints follow RESTful conventions and return JSON responses.

**Base URL**: `https://your-domain.com` or `http://localhost:8000` (development)

**API Version**: v1

**Content-Type**: `application/json`

## Authentication

### API Key Authentication

Most endpoints require API key authentication. Include the API key in the request header:

```http
X-API-Key: your_api_key_here
```

### JWT Token Authentication

For user-based authentication, use JWT tokens:

```http
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

### Getting an Access Token

```http
POST /auth/token
Content-Type: application/x-www-form-urlencoded

username=your_username&password=your_password
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 1800
}
```

## Core API Endpoints

### Articles Management

#### Get Articles

```http
GET /api/articles
```

**Parameters:**
- `limit` (int, optional): Number of articles to return (default: 50, max: 500)
- `offset` (int, optional): Number of articles to skip (default: 0)
- `status` (string, optional): Filter by status (`pending`, `approved`, `rejected`)
- `source` (string, optional): Filter by source name
- `from_date` (string, optional): Filter articles from date (ISO format)
- `to_date` (string, optional): Filter articles to date (ISO format)
- `search` (string, optional): Search in title and content
- `language` (string, optional): Filter by language (`en`, `zh`, `zh-cn`)

**Example Request:**
```http
GET /api/articles?limit=20&status=approved&from_date=2024-01-01T00:00:00Z
X-API-Key: your_api_key_here
```

**Example Response:**
```json
{
  "articles": [
    {
      "id": "uuid-here",
      "title": "US Announces New Tariffs on Chinese Electronics",
      "url": "https://example.com/article/123",
      "content": "Article content here...",
      "summary": "AI-generated summary...",
      "source": "Reuters",
      "language": "en",
      "published_at": "2024-01-15T10:30:00Z",
      "created_at": "2024-01-15T10:35:00Z",
      "status": "approved",
      "scores": {
        "keyword_score": 0.85,
        "similarity_score": 0.72,
        "ml_score": 0.88,
        "final_score": 0.82
      },
      "tags": ["trade-war", "electronics", "tariffs"],
      "notification_sent": true
    }
  ],
  "total": 156,
  "limit": 20,
  "offset": 0,
  "has_more": true
}
```

#### Get Single Article

```http
GET /api/articles/{article_id}
```

**Example Response:**
```json
{
  "id": "uuid-here",
  "title": "Article Title",
  "url": "https://example.com/article/123",
  "content": "Full article content...",
  "summary": "AI-generated summary...",
  "source": "Reuters",
  "language": "en",
  "published_at": "2024-01-15T10:30:00Z",
  "created_at": "2024-01-15T10:35:00Z",
  "updated_at": "2024-01-15T11:00:00Z",
  "status": "approved",
  "scores": {
    "keyword_score": 0.85,
    "similarity_score": 0.72,
    "ml_score": 0.88,
    "final_score": 0.82
  },
  "tags": ["trade-war", "electronics", "tariffs"],
  "analysis": {
    "sentiment": "negative",
    "entities": ["United States", "China", "electronics"],
    "impact_assessment": "high"
  },
  "notification_sent": true,
  "similar_articles": ["uuid-1", "uuid-2"]
}
```

#### Update Article Status

```http
PUT /api/articles/{article_id}/status
X-API-Key: your_api_key_here
Content-Type: application/json

{
  "status": "approved",
  "reason": "Relevant trade war content"
}
```

**Response:**
```json
{
  "id": "uuid-here",
  "status": "approved",
  "updated_at": "2024-01-15T11:00:00Z"
}
```

#### Bulk Update Articles

```http
POST /api/articles/bulk-update
X-API-Key: your_api_key_here
Content-Type: application/json

{
  "article_ids": ["uuid-1", "uuid-2", "uuid-3"],
  "action": "approve",
  "reason": "Batch approval of relevant articles"
}
```

### Sources Management

#### Get Sources

```http
GET /api/sources
```

**Example Response:**
```json
{
  "sources": [
    {
      "id": "uuid-here",
      "name": "Reuters Trade",
      "url": "https://feeds.reuters.com/reuters/businessNews",
      "type": "rss",
      "enabled": true,
      "last_fetched": "2024-01-15T12:00:00Z",
      "fetch_interval": 300,
      "language": "en",
      "article_count": 1234,
      "success_rate": 0.98,
      "created_at": "2024-01-01T00:00:00Z"
    }
  ],
  "total": 5
}
```

#### Add New Source

```http
POST /api/sources
X-API-Key: your_api_key_here
Content-Type: application/json

{
  "name": "New Trade Source",
  "url": "https://example.com/rss/trade",
  "type": "rss",
  "enabled": true,
  "fetch_interval": 600,
  "language": "en"
}
```

#### Update Source

```http
PUT /api/sources/{source_id}
X-API-Key: your_api_key_here
Content-Type: application/json

{
  "enabled": false,
  "fetch_interval": 900
}
```

### Search and Analytics

#### Search Articles

```http
POST /api/search
X-API-Key: your_api_key_here
Content-Type: application/json

{
  "query": "semiconductor trade restrictions",
  "limit": 20,
  "filters": {
    "language": "en",
    "date_range": {
      "from": "2024-01-01T00:00:00Z",
      "to": "2024-01-31T23:59:59Z"
    },
    "sources": ["Reuters", "Bloomberg"],
    "min_score": 0.7
  }
}
```

**Response:**
```json
{
  "results": [
    {
      "article": {
        "id": "uuid-here",
        "title": "New Semiconductor Restrictions Announced",
        "summary": "...",
        "score": 0.92
      },
      "similarity_score": 0.89,
      "highlighted_text": "semiconductor <mark>trade</mark> <mark>restrictions</mark>"
    }
  ],
  "total": 45,
  "query_time_ms": 156
}
```

#### Get Analytics

```http
GET /api/analytics
X-API-Key: your_api_key_here
```

**Parameters:**
- `period` (string): Time period (`day`, `week`, `month`, `year`)
- `from_date` (string, optional): Start date
- `to_date` (string, optional): End date

**Example Response:**
```json
{
  "period": "week",
  "from_date": "2024-01-08T00:00:00Z",
  "to_date": "2024-01-15T00:00:00Z",
  "summary": {
    "total_articles": 234,
    "approved_articles": 89,
    "rejected_articles": 12,
    "pending_articles": 133,
    "notifications_sent": 76
  },
  "trends": {
    "daily_counts": [
      {"date": "2024-01-08", "count": 32},
      {"date": "2024-01-09", "count": 28},
      {"date": "2024-01-10", "count": 45}
    ],
    "top_sources": [
      {"source": "Reuters", "count": 67},
      {"source": "Bloomberg", "count": 45}
    ],
    "top_keywords": [
      {"keyword": "tariff", "count": 23},
      {"keyword": "semiconductor", "count": 18}
    ]
  },
  "performance": {
    "avg_processing_time": 2.34,
    "system_health": "healthy"
  }
}
```

## Monitoring API

### Health Checks

#### Basic Health Check

```http
GET /monitoring/health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T12:00:00Z"
}
```

#### Detailed Health Check

```http
GET /monitoring/health?detailed=true
```

**Response:**
```json
{
  "overall_status": "healthy",
  "timestamp": "2024-01-15T12:00:00Z",
  "checks": [
    {
      "component": "database",
      "status": "healthy",
      "message": "Database connection successful",
      "response_time_ms": 45.2,
      "details": {
        "active_connections": 5,
        "database_name": "tariff_radar",
        "postgres_version": "15.2"
      }
    },
    {
      "component": "redis",
      "status": "healthy",
      "message": "Redis connection and operations successful",
      "response_time_ms": 12.1,
      "details": {
        "redis_version": "7.0.5",
        "connected_clients": 3,
        "used_memory_human": "2.5M"
      }
    }
  ],
  "summary": {
    "total_checks": 5,
    "status_counts": {
      "healthy": 5,
      "degraded": 0,
      "unhealthy": 0,
      "unknown": 0
    },
    "average_response_time_ms": 28.4
  }
}
```

### Metrics

#### Get Prometheus Metrics

```http
GET /monitoring/metrics
```

**Response:** (Prometheus format)
```
# HELP http_requests_total Total HTTP requests
# TYPE http_requests_total counter
http_requests_total{method="GET",endpoint="/api/articles",status="200"} 1234
http_requests_total{method="POST",endpoint="/api/articles",status="201"} 45

# HELP http_request_duration_seconds HTTP request duration
# TYPE http_request_duration_seconds histogram
http_request_duration_seconds_bucket{method="GET",endpoint="/api/articles",le="0.1"} 234
http_request_duration_seconds_bucket{method="GET",endpoint="/api/articles",le="0.25"} 456
```

#### Get Metrics Summary

```http
GET /monitoring/metrics/summary
X-API-Key: your_api_key_here
```

### Alerts

#### Get Active Alerts

```http
GET /monitoring/alerts
X-API-Key: your_api_key_here
```

**Response:**
```json
{
  "alerts": [
    {
      "id": "high_cpu_usage_system_resources",
      "rule": {
        "name": "high_cpu_usage",
        "description": "CPU usage above 90% for extended period",
        "severity": "high",
        "component": "system_resources"
      },
      "triggered_at": "2024-01-15T11:45:00Z",
      "status": "triggered",
      "current_value": 92.5,
      "message": "🚨 HIGH ALERT: CPU usage above 90% for extended period"
    }
  ],
  "count": 1,
  "timestamp": "2024-01-15T12:00:00Z"
}
```

#### Acknowledge Alert

```http
POST /monitoring/alerts/{alert_id}/acknowledge
X-API-Key: your_api_key_here

{
  "acknowledged_by": "admin@company.com"
}
```

## Administration API

### System Control

#### Trigger Manual Processing

```http
POST /admin/process-feeds
X-API-Key: your_admin_api_key_here

{
  "sources": ["Reuters", "Bloomberg"],
  "force": false
}
```

#### Rebuild Search Index

```http
POST /admin/rebuild-index
X-API-Key: your_admin_api_key_here
```

#### System Statistics

```http
GET /admin/stats
X-API-Key: your_admin_api_key_here
```

**Response:**
```json
{
  "system": {
    "uptime_seconds": 86400,
    "version": "1.0.0",
    "environment": "production"
  },
  "database": {
    "total_articles": 15234,
    "total_sources": 12,
    "database_size_mb": 2048
  },
  "processing": {
    "articles_today": 234,
    "avg_processing_time": 2.1,
    "queue_length": 5
  },
  "performance": {
    "cpu_usage": 45.2,
    "memory_usage": 62.8,
    "disk_usage": 23.1
  }
}
```

## WebSocket Endpoints

### Real-time Updates

Connect to WebSocket for real-time updates:

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/updates');

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log('Update received:', data);
};
```

**Message Types:**
- `article_processed`: New article processed
- `alert_triggered`: New alert triggered
- `system_status`: System status change

**Example Message:**
```json
{
  "type": "article_processed",
  "timestamp": "2024-01-15T12:00:00Z",
  "data": {
    "article_id": "uuid-here",
    "title": "New Trade Agreement Announced",
    "status": "approved",
    "score": 0.89
  }
}
```

## Error Handling

### Error Response Format

All error responses follow this format:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid input parameters",
    "details": {
      "field": "date_range",
      "reason": "Invalid date format"
    }
  },
  "timestamp": "2024-01-15T12:00:00Z",
  "request_id": "req_123456"
}
```

### HTTP Status Codes

- `200 OK`: Successful request
- `201 Created`: Resource created successfully
- `400 Bad Request`: Invalid request parameters
- `401 Unauthorized`: Missing or invalid authentication
- `403 Forbidden`: Insufficient permissions
- `404 Not Found`: Resource not found
- `409 Conflict`: Resource conflict (e.g., duplicate)
- `422 Unprocessable Entity`: Validation error
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server error
- `503 Service Unavailable`: Service temporarily unavailable

### Common Error Codes

| Code | Description |
|------|-------------|
| `AUTHENTICATION_REQUIRED` | API key or token required |
| `INVALID_API_KEY` | Invalid or expired API key |
| `RATE_LIMIT_EXCEEDED` | Too many requests |
| `VALIDATION_ERROR` | Input validation failed |
| `RESOURCE_NOT_FOUND` | Requested resource not found |
| `PERMISSION_DENIED` | Insufficient permissions |
| `SYSTEM_ERROR` | Internal system error |
| `SERVICE_UNAVAILABLE` | Service temporarily down |

## Rate Limiting

### Rate Limit Headers

All responses include rate limiting headers:

```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1642248000
X-RateLimit-Window: 3600
```

### Rate Limits by Endpoint

| Endpoint Category | Limit | Window |
|------------------|-------|---------|
| Authentication | 10 requests | 1 minute |
| General API | 1000 requests | 1 hour |
| Search | 100 requests | 1 hour |
| Admin API | 100 requests | 1 hour |
| Monitoring | 500 requests | 1 hour |

## API Examples

### Complete Workflow Example

```bash
# 1. Get access token
curl -X POST http://localhost:8000/auth/token \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=admin&password=secret"

# 2. Get recent articles
curl -X GET "http://localhost:8000/api/articles?limit=10&status=approved" \
  -H "X-API-Key: your_api_key_here"

# 3. Search for specific content
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key_here" \
  -d '{
    "query": "semiconductor sanctions",
    "limit": 5
  }'

# 4. Check system health
curl -X GET http://localhost:8000/monitoring/health?detailed=true

# 5. Get system analytics
curl -X GET "http://localhost:8000/api/analytics?period=week" \
  -H "X-API-Key: your_api_key_here"
```

### Python Client Example

```python
import requests
import json

class TariffRadarClient:
    def __init__(self, base_url, api_key):
        self.base_url = base_url
        self.headers = {
            'X-API-Key': api_key,
            'Content-Type': 'application/json'
        }
    
    def get_articles(self, limit=50, status=None):
        params = {'limit': limit}
        if status:
            params['status'] = status
        
        response = requests.get(
            f"{self.base_url}/api/articles",
            headers=self.headers,
            params=params
        )
        return response.json()
    
    def search_articles(self, query, filters=None):
        data = {'query': query}
        if filters:
            data['filters'] = filters
        
        response = requests.post(
            f"{self.base_url}/api/search",
            headers=self.headers,
            json=data
        )
        return response.json()
    
    def get_health(self):
        response = requests.get(f"{self.base_url}/monitoring/health")
        return response.json()

# Usage
client = TariffRadarClient('http://localhost:8000', 'your_api_key')

# Get approved articles
articles = client.get_articles(limit=10, status='approved')
print(f"Found {len(articles['articles'])} articles")

# Search for specific content
results = client.search_articles(
    'trade war',
    filters={'language': 'en', 'min_score': 0.8}
)

# Check system health
health = client.get_health()
print(f"System status: {health['status']}")
```

---

## API Reference Summary

| Category | Endpoints | Authentication |
|----------|-----------|----------------|
| Articles | `/api/articles/*` | API Key |
| Sources | `/api/sources/*` | API Key |
| Search | `/api/search` | API Key |
| Analytics | `/api/analytics` | API Key |
| Health | `/monitoring/health` | None |
| Metrics | `/monitoring/metrics` | None (Prometheus) |
| Alerts | `/monitoring/alerts/*` | API Key |
| Admin | `/admin/*` | Admin API Key |
| WebSocket | `/ws/*` | Optional |

For additional help or questions about the API, please refer to the system documentation or contact support.