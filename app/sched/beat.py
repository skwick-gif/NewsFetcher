"""
Celery Beat Scheduler Configuration
Handles periodic task scheduling for the Tariff Radar system
"""

import os
from celery import Celery
from celery.schedules import crontab
from datetime import timedelta

# Import tasks
from tasks import (
    ingest_articles_task,
    send_notifications_task, 
    cleanup_old_articles_task,
    health_check_task
)

# Initialize Celery with beat scheduler
celery_app = Celery(
    'tariff_radar_beat',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/0'
)

# Beat scheduler configuration
celery_app.conf.beat_schedule = {
    # Article ingestion - every 30 minutes during business hours
    'ingest-articles-frequent': {
        'task': 'sched.tasks.ingest_articles_task',
        'schedule': crontab(minute='*/30', hour='6-22'),  # Every 30min, 6AM-10PM UTC
        'kwargs': {
            'source_configs': [
                {
                    "type": "rss",
                    "url": "https://rsshub.app/reuters/business",
                    "source": "Reuters Business",
                    "category": "business"
                },
                {
                    "type": "rss",
                    "url": "https://rsshub.app/reuters/world/china",
                    "source": "Reuters China", 
                    "category": "china"
                },
                {
                    "type": "rss",
                    "url": "https://rsshub.app/bloomberg/markets",
                    "source": "Bloomberg Markets",
                    "category": "markets"
                },
                {
                    "type": "rss",
                    "url": "https://rsshub.app/wsj/world",
                    "source": "WSJ World",
                    "category": "world"
                },
                {
                    "type": "rss",
                    "url": "https://rsshub.app/ft/world",
                    "source": "Financial Times World",
                    "category": "world"
                }
            ]
        },
        'options': {
            'expires': 1800,  # Task expires in 30 minutes
            'retry_policy': {
                'max_retries': 3,
                'interval_start': 60,
                'interval_step': 60,
                'interval_max': 300
            }
        }
    },
    
    # Extended article ingestion - every 2 hours for broader sources
    'ingest-articles-extended': {
        'task': 'sched.tasks.ingest_articles_task',
        'schedule': crontab(minute=15, hour='*/2'),  # Every 2 hours at 15min past
        'kwargs': {
            'source_configs': [
                {
                    "type": "rss",
                    "url": "https://rsshub.app/scmp/china",
                    "source": "SCMP China",
                    "category": "china"
                },
                {
                    "type": "rss",
                    "url": "https://rsshub.app/nikkei/china", 
                    "source": "Nikkei China",
                    "category": "asia"
                },
                {
                    "type": "rss",
                    "url": "https://rsshub.app/economist/china",
                    "source": "Economist China",
                    "category": "analysis"
                },
                {
                    "type": "rss",
                    "url": "https://rsshub.app/politico/china",
                    "source": "Politico China",
                    "category": "politics"
                },
                {
                    "type": "rss",
                    "url": "https://rsshub.app/cnbc/world",
                    "source": "CNBC World",
                    "category": "business"
                }
            ]
        }
    },
    
    # Night-time light ingestion - every 4 hours during off-hours  
    'ingest-articles-light': {
        'task': 'sched.tasks.ingest_articles_task', 
        'schedule': crontab(minute=30, hour='23-5/4'),  # 11:30PM, 3:30AM (every 4h off-hours)
        'kwargs': {
            'source_configs': [
                {
                    "type": "rss",
                    "url": "https://rsshub.app/reuters/business",
                    "source": "Reuters Business",
                    "category": "business"
                },
                {
                    "type": "rss", 
                    "url": "https://rsshub.app/bloomberg/markets",
                    "source": "Bloomberg Markets",
                    "category": "markets"
                }
            ]
        }
    },
    
    # Daily cleanup - 2:00 AM UTC
    'cleanup-old-articles': {
        'task': 'sched.tasks.cleanup_old_articles_task',
        'schedule': crontab(minute=0, hour=2),  # 2:00 AM UTC daily
        'kwargs': {'days_to_keep': 30},
        'options': {'expires': 3600}
    },
    
    # System health check - every 15 minutes
    'health-check': {
        'task': 'sched.tasks.health_check_task',
        'schedule': crontab(minute='*/15'),  # Every 15 minutes
        'options': {'expires': 300}  # 5 minute expiry
    },
    
    # 🇨🇳 Chinese Government Sites Scraping - CRITICAL FOR "WHO PUBLISHED FIRST"
    # Run every 15 minutes during Chinese business hours (Beijing time)
    # Beijing is UTC+8, so 9AM-6PM Beijing = 1AM-10AM UTC
    'scrape-chinese-gov-sites': {
        'task': 'sched.tasks.scrape_chinese_gov_sites_task',
        'schedule': crontab(minute='*/15', hour='1-10'),  # Every 15min during China business hours
        'options': {
            'expires': 900,  # 15 minute expiry
            'retry_policy': {
                'max_retries': 3,
                'interval_start': 120,
                'interval_step': 60,
                'interval_max': 300
            }
        }
    },
    
    # 🇨🇳 Chinese Government Sites - Off-hours check (less frequent)
    'scrape-chinese-gov-sites-offhours': {
        'task': 'sched.tasks.scrape_chinese_gov_sites_task',
        'schedule': crontab(minute=0, hour='11-23,0'),  # Every hour during off-hours
        'options': {'expires': 1800}
    },
    
    # 🐦 Twitter/X Monitoring - HIGHEST PRIORITY!
    # Trump often tweets tariff announcements BEFORE official statements
    # Run every 5 minutes during US waking hours (6AM-11PM EST = 10AM-3AM UTC)
    'scrape-twitter-frequent': {
        'task': 'sched.tasks.scrape_twitter_accounts_task',
        'schedule': crontab(minute='*/5', hour='10-23,0-3'),  # Every 5min during US hours
        'options': {
            'expires': 300,  # 5 minute expiry
            'retry_policy': {
                'max_retries': 3,
                'interval_start': 30,
                'interval_step': 30,
                'interval_max': 180
            }
        }
    },
    
    # 🐦 Twitter/X - Off-hours monitoring (less frequent)
    'scrape-twitter-offhours': {
        'task': 'sched.tasks.scrape_twitter_accounts_task',
        'schedule': crontab(minute='*/30', hour='4-9'),  # Every 30min during night
        'options': {'expires': 900}
    },
    
    # Weekly deep cleanup - Sunday 3:00 AM UTC
    'weekly-deep-cleanup': {
        'task': 'sched.tasks.cleanup_old_articles_task',
        'schedule': crontab(minute=0, hour=3, day_of_week=0),  # Sunday 3:00 AM
        'kwargs': {'days_to_keep': 7},  # More aggressive weekly cleanup
        'options': {'expires': 7200}  # 2 hour expiry
    }
}

# Additional configuration
celery_app.conf.update(
    timezone='UTC',
    enable_utc=True,
    
    # Task routing
    task_routes={
        'sched.tasks.ingest_articles_task': {'queue': 'ingestion'},
        'sched.tasks.scrape_chinese_gov_sites_task': {'queue': 'ingestion'},
        'sched.tasks.send_notifications_task': {'queue': 'notifications'}, 
        'sched.tasks.cleanup_old_articles_task': {'queue': 'maintenance'},
        'sched.tasks.health_check_task': {'queue': 'monitoring'}
    },
    
    # Worker configuration  
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    worker_max_tasks_per_child=50,
    
    # Beat configuration
    beat_schedule_filename='celerybeat-schedule',
    beat_sync_every=1,
    
    # Result backend settings
    result_expires=3600,
    result_compression='gzip'
)


def create_custom_schedules():
    """
    Create dynamic schedules based on configuration or conditions
    Can be used to adjust scheduling based on system load, time zones, etc.
    """
    
    # Example: Add more frequent monitoring during market hours
    market_hours_schedule = {
        'market-hours-intensive-monitoring': {
            'task': 'sched.tasks.ingest_articles_task',
            'schedule': crontab(minute='*/10', hour='13-21', day_of_week='1-5'),  # Market hours UTC
            'kwargs': {
                'source_configs': [
                    {
                        "type": "rss",
                        "url": "https://rsshub.app/reuters/business", 
                        "source": "Reuters Business",
                        "category": "business"
                    },
                    {
                        "type": "rss",
                        "url": "https://rsshub.app/bloomberg/markets",
                        "source": "Bloomberg Markets", 
                        "category": "markets"
                    }
                ]
            }
        }
    }
    
    # Example: Emergency mode - more frequent checks
    emergency_schedule = {
        'emergency-monitoring': {
            'task': 'sched.tasks.ingest_articles_task',
            'schedule': timedelta(minutes=5),  # Every 5 minutes
            'kwargs': {
                'source_configs': [
                    {
                        "type": "rss",
                        "url": "https://rsshub.app/reuters/business",
                        "source": "Reuters Business - Emergency",
                        "category": "emergency"
                    }
                ]
            }
        }
    }
    
    return market_hours_schedule  # Return the schedule to use


def update_schedule_dynamically():
    """
    Update beat schedule based on runtime conditions
    This could be called from a management command or API endpoint
    """
    
    # Check system load, time of day, or other conditions
    import datetime
    current_hour = datetime.datetime.utcnow().hour
    
    if 13 <= current_hour <= 21:  # Market hours
        # Add intensive monitoring
        additional_schedule = create_custom_schedules()
        celery_app.conf.beat_schedule.update(additional_schedule)
        print("Switched to market hours intensive monitoring")
    
    # Could also modify existing schedules
    # celery_app.conf.beat_schedule['ingest-articles-frequent']['schedule'] = crontab(minute='*/10')


if __name__ == "__main__":
    print("Celery Beat Scheduler Configuration")
    print("=" * 50)
    print("\nScheduled Tasks:")
    
    for task_name, config in celery_app.conf.beat_schedule.items():
        schedule_info = config['schedule']
        task_path = config['task']
        
        if isinstance(schedule_info, crontab):
            schedule_desc = f"Cron: {schedule_info}"
        else:
            schedule_desc = f"Interval: {schedule_info}"
            
        print(f"📅 {task_name}")
        print(f"   Task: {task_path}")
        print(f"   Schedule: {schedule_desc}")
        if 'kwargs' in config and config['kwargs']:
            print(f"   Args: {config['kwargs']}")
        print()
    
    print("\nTo start the beat scheduler, run:")
    print("celery -A sched.beat beat --loglevel=info")
    print("\nTo start workers, run:")
    print("celery -A sched.tasks worker --loglevel=info --queues=ingestion,notifications,maintenance,monitoring")