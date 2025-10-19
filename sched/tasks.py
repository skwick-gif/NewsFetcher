"""
Celery tasks for the Tariff Radar system.
"""
import os
import sys
from pathlib import Path

# Add the app directory to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from celery import Celery
from celery.schedules import crontab

from tasks.news_collector import collect_news_data
from tasks.tariff_processor import process_tariff_data
from tasks.alert_sender import send_alerts

# Create Celery application
app = Celery('tariff-radar')

# Configuration
app.conf.update(
    broker_url=os.getenv('REDIS_URL', 'redis://redis:6379/0'),
    result_backend=os.getenv('REDIS_URL', 'redis://redis:6379/0'),
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    
    # Beat schedule
    beat_schedule={
        'collect-news-every-30-minutes': {
            'task': 'sched.tasks.collect_news',
            'schedule': crontab(minute='*/30'),
        },
        'process-tariffs-every-hour': {
            'task': 'sched.tasks.process_tariffs',
            'schedule': crontab(minute=0),
        },
        'send-alerts-daily': {
            'task': 'sched.tasks.send_daily_alerts',
            'schedule': crontab(hour=9, minute=0),
        },
    },
)

# Task definitions
@app.task(name='sched.tasks.collect_news')
def collect_news():
    """Collect news from various sources."""
    return collect_news_data()

@app.task(name='sched.tasks.process_tariffs')
def process_tariffs():
    """Process collected tariff data."""
    return process_tariff_data()

@app.task(name='sched.tasks.send_daily_alerts')
def send_daily_alerts():
    """Send daily alert summaries."""
    return send_alerts()

if __name__ == '__main__':
    app.start()