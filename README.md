# Tariff Radar - US-China Trade Monitoring System

A comprehensive automated monitoring system for US-China trade news, tariffs, and economic developments with AI-powered analysis and multi-channel notifications.

## 🎯 Overview

Tariff Radar combines deterministic data ingestion with smart AI analysis to monitor global trade news sources, identify relevant articles about US-China trade relationships, and deliver timely notifications through multiple channels (WeChat Work, Email, Telegram).

### Key Features

- **Multi-Source Ingestion**: Automated RSS feed monitoring from major financial news sources
- **Smart Content Analysis**: Multilingual keyword filtering + semantic embeddings + ML classification + LLM triage
- **Duplicate Detection**: Advanced similarity detection to avoid redundant alerts
- **Multi-Channel Notifications**: WeCom (Enterprise WeChat), Email, and Telegram integration
- **Web Dashboard**: Real-time monitoring interface with article management
- **Scalable Architecture**: Docker Compose deployment with Celery task queues
- **דשבורד**: ממשק ניהול וצפייה ב-FastAPI

## Architecture

```
[Sources] → [Ingestion] → ["Smart" Layer] → [Storage] → [Alerting]
```

### תהליך עבודה:
1. **איסוף נתונים**: RSSHub, scrapers, APIs
2. **נירמול**: ניקוי HTML, זיהוי שפה, דה-דופליקציה
3. **ניתוח**: מילות מפתח → embeddings → classifier → LLM triage
4. **התראות**: WeCom + Email לפי רמת חשיבות
5. **ארכיון**: PostgreSQL + Qdrant vectors

## Quick Start

```bash
# Clone and setup
cd tariff-radar
cp .env.example .env
# Edit .env with your credentials

# Start services
docker-compose up -d

# Access dashboard
http://localhost:8000
```

## Configuration

ערוך את `config.yaml` להגדרת מקורות, ספים, והתראות.

## Requirements

- Docker & Docker Compose
- Python 3.11+
- WeCom corporate account (optional)
- OpenAI/Anthropic/Perplexity API key (for LLM triage)

## Project Structure

```
tariff-radar/
  ├─ app/
  │  ├─ ingest/         # Data collection & normalization
  │  ├─ smart/          # AI analysis layer
  │  ├─ notify/         # Alert handlers
  │  ├─ storage/        # DB models & vector store
  │  └─ sched/          # Task scheduling
  ├─ docker-compose.yml
  ├─ config.yaml
  └─ .env
```

## License

MIT