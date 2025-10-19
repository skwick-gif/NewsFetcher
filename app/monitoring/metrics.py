"""
Application Metrics Collection
Prometheus metrics for monitoring system performance and health
"""

import time
import logging
from typing import Dict, Any, Optional, List, Callable
from functools import wraps
from datetime import datetime, timedelta
import psutil
import os
from collections import defaultdict, deque
import threading
import asyncio

try:
    from prometheus_client import (
        Counter, Histogram, Gauge, Info, Enum,
        CollectorRegistry, generate_latest, 
        start_http_server, CONTENT_TYPE_LATEST
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    # Fallback implementations for when Prometheus is not available
    class Counter:
        def __init__(self, *args, **kwargs): pass
        def inc(self, amount=1): pass
        def labels(self, **kwargs): return self
    
    class Histogram:
        def __init__(self, *args, **kwargs): pass
        def observe(self, amount): pass
        def time(self): return self
        def labels(self, **kwargs): return self
        def __enter__(self): return self
        def __exit__(self, *args): pass
    
    class Gauge:
        def __init__(self, *args, **kwargs): pass
        def set(self, value): pass
        def inc(self, amount=1): pass
        def dec(self, amount=1): pass
        def labels(self, **kwargs): return self


class MetricsCollector:
    """Central metrics collection and management"""
    
    def __init__(self, enable_prometheus: bool = True):
        self.enable_prometheus = enable_prometheus and PROMETHEUS_AVAILABLE
        self.registry = CollectorRegistry() if self.enable_prometheus else None
        
        # Internal metrics storage for when Prometheus is not available
        self.internal_metrics = defaultdict(lambda: defaultdict(float))
        self.time_series = defaultdict(lambda: deque(maxlen=1000))
        
        # Initialize metrics
        self._init_system_metrics()
        self._init_application_metrics()
        self._init_business_metrics()
        
        # Background monitoring
        self.monitoring_active = False
        self.monitor_thread = None
        
        self.logger = logging.getLogger(__name__)
    
    def _init_system_metrics(self):
        """Initialize system-level metrics"""
        if self.enable_prometheus:
            # System resource metrics
            self.cpu_usage = Gauge(
                'system_cpu_usage_percent',
                'CPU usage percentage',
                registry=self.registry
            )
            
            self.memory_usage = Gauge(
                'system_memory_usage_bytes',
                'Memory usage in bytes',
                ['type'],  # rss, vms, shared
                registry=self.registry
            )
            
            self.disk_usage = Gauge(
                'system_disk_usage_bytes',
                'Disk usage in bytes',
                ['type'],  # used, free, total
                registry=self.registry
            )
            
            # Network metrics
            self.network_bytes = Counter(
                'system_network_bytes_total',
                'Network bytes transferred',
                ['direction'],  # sent, received
                registry=self.registry
            )
        else:
            # Fallback to internal storage
            self.cpu_usage = Gauge()
            self.memory_usage = Gauge()
            self.disk_usage = Gauge()
            self.network_bytes = Counter()
    
    def _init_application_metrics(self):
        """Initialize application-level metrics"""
        if self.enable_prometheus:
            # HTTP request metrics
            self.http_requests_total = Counter(
                'http_requests_total',
                'Total HTTP requests',
                ['method', 'endpoint', 'status'],
                registry=self.registry
            )
            
            self.http_request_duration = Histogram(
                'http_request_duration_seconds',
                'HTTP request duration',
                ['method', 'endpoint'],
                buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
                registry=self.registry
            )
            
            # Database metrics
            self.db_connections_active = Gauge(
                'database_connections_active',
                'Active database connections',
                registry=self.registry
            )
            
            self.db_query_duration = Histogram(
                'database_query_duration_seconds',
                'Database query duration',
                ['operation'],
                buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0),
                registry=self.registry
            )
            
            # Cache metrics
            self.cache_operations = Counter(
                'cache_operations_total',
                'Cache operations',
                ['type', 'result'],  # get/set/delete, hit/miss/error
                registry=self.registry
            )
            
            # Task queue metrics
            self.celery_tasks_total = Counter(
                'celery_tasks_total',
                'Total Celery tasks',
                ['task_name', 'status'],  # success, failure, retry
                registry=self.registry
            )
            
            self.celery_task_duration = Histogram(
                'celery_task_duration_seconds',
                'Celery task duration',
                ['task_name'],
                buckets=(1.0, 5.0, 10.0, 30.0, 60.0, 300.0, 600.0),
                registry=self.registry
            )
            
            self.celery_queue_length = Gauge(
                'celery_queue_length',
                'Celery queue length',
                ['queue_name'],
                registry=self.registry
            )
        else:
            # Fallback implementations
            self.http_requests_total = Counter()
            self.http_request_duration = Histogram()
            self.db_connections_active = Gauge()
            self.db_query_duration = Histogram()
            self.cache_operations = Counter()
            self.celery_tasks_total = Counter()
            self.celery_task_duration = Histogram()
            self.celery_queue_length = Gauge()
    
    def _init_business_metrics(self):
        """Initialize business-specific metrics"""
        if self.enable_prometheus:
            # Article processing metrics
            self.articles_processed_total = Counter(
                'articles_processed_total',
                'Total articles processed',
                ['source', 'status'],  # pending, approved, rejected
                registry=self.registry
            )
            
            self.article_processing_duration = Histogram(
                'article_processing_duration_seconds',
                'Article processing duration',
                ['stage'],  # ingestion, analysis, classification
                buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0),
                registry=self.registry
            )
            
            self.article_scores = Histogram(
                'article_scores',
                'Article scoring distribution',
                ['score_type'],  # keyword, similarity, ml, final
                buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
                registry=self.registry
            )
            
            # Notification metrics
            self.notifications_sent_total = Counter(
                'notifications_sent_total',
                'Total notifications sent',
                ['channel', 'status'],  # wecom/email/telegram, success/failure
                registry=self.registry
            )
            
            # RSS feed metrics
            self.rss_feeds_fetched = Counter(
                'rss_feeds_fetched_total',
                'RSS feeds fetched',
                ['source', 'status'],  # success, error
                registry=self.registry
            )
            
            self.rss_articles_found = Gauge(
                'rss_articles_found',
                'Articles found in RSS feeds',
                ['source'],
                registry=self.registry
            )
            
            # Duplicate detection metrics
            self.duplicates_detected = Counter(
                'duplicates_detected_total',
                'Duplicate articles detected',
                ['detection_method'],  # url, content_hash, similarity
                registry=self.registry
            )
        else:
            # Fallback implementations
            self.articles_processed_total = Counter()
            self.article_processing_duration = Histogram()
            self.article_scores = Histogram()
            self.notifications_sent_total = Counter()
            self.rss_feeds_fetched = Counter()
            self.rss_articles_found = Gauge()
            self.duplicates_detected = Counter()
    
    def start_monitoring(self, interval: int = 30):
        """Start background system monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_system_metrics,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        self.logger.info(f"Started system monitoring with {interval}s interval")
    
    def stop_monitoring(self):
        """Stop background monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        self.logger.info("Stopped system monitoring")
    
    def _monitor_system_metrics(self, interval: int):
        """Background thread for system monitoring"""
        while self.monitoring_active:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent()
                self.cpu_usage.set(cpu_percent)
                
                # Memory usage
                memory = psutil.virtual_memory()
                self.memory_usage.labels(type='rss').set(memory.used)
                self.memory_usage.labels(type='available').set(memory.available)
                self.memory_usage.labels(type='total').set(memory.total)
                
                # Disk usage
                disk = psutil.disk_usage('/')
                self.disk_usage.labels(type='used').set(disk.used)
                self.disk_usage.labels(type='free').set(disk.free)
                self.disk_usage.labels(type='total').set(disk.total)
                
                # Process-specific metrics
                process = psutil.Process()
                process_memory = process.memory_info()
                self.memory_usage.labels(type='process_rss').set(process_memory.rss)
                self.memory_usage.labels(type='process_vms').set(process_memory.vms)
                
                # Store internal metrics if Prometheus not available
                if not self.enable_prometheus:
                    timestamp = datetime.utcnow()
                    self.time_series['cpu_usage'].append((timestamp, cpu_percent))
                    self.time_series['memory_used'].append((timestamp, memory.used))
                
            except Exception as e:
                self.logger.error(f"Error collecting system metrics: {e}")
            
            time.sleep(interval)
    
    def record_http_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """Record HTTP request metrics"""
        self.http_requests_total.labels(
            method=method, 
            endpoint=endpoint, 
            status=str(status_code)
        ).inc()
        
        self.http_request_duration.labels(
            method=method, 
            endpoint=endpoint
        ).observe(duration)
    
    def record_db_query(self, operation: str, duration: float):
        """Record database query metrics"""
        self.db_query_duration.labels(operation=operation).observe(duration)
    
    def record_article_processing(self, source: str, status: str, stage: str, duration: float):
        """Record article processing metrics"""
        self.articles_processed_total.labels(source=source, status=status).inc()
        self.article_processing_duration.labels(stage=stage).observe(duration)
    
    def record_article_score(self, score_type: str, score: float):
        """Record article scoring metrics"""
        self.article_scores.labels(score_type=score_type).observe(score)
    
    def record_notification(self, channel: str, success: bool):
        """Record notification metrics"""
        status = 'success' if success else 'failure'
        self.notifications_sent_total.labels(channel=channel, status=status).inc()
    
    def record_rss_fetch(self, source: str, success: bool, articles_count: int = 0):
        """Record RSS feed fetch metrics"""
        status = 'success' if success else 'error'
        self.rss_feeds_fetched.labels(source=source, status=status).inc()
        
        if success:
            self.rss_articles_found.labels(source=source).set(articles_count)
    
    def record_duplicate(self, detection_method: str):
        """Record duplicate detection metrics"""
        self.duplicates_detected.labels(detection_method=detection_method).inc()
    
    def record_celery_task(self, task_name: str, status: str, duration: float = None):
        """Record Celery task metrics"""
        self.celery_tasks_total.labels(task_name=task_name, status=status).inc()
        
        if duration is not None:
            self.celery_task_duration.labels(task_name=task_name).observe(duration)
    
    def set_queue_length(self, queue_name: str, length: int):
        """Set queue length metric"""
        self.celery_queue_length.labels(queue_name=queue_name).set(length)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get current metrics summary"""
        if self.enable_prometheus:
            # Return Prometheus metrics in text format
            return {
                'prometheus_metrics': generate_latest(self.registry).decode('utf-8'),
                'timestamp': datetime.utcnow().isoformat()
            }
        else:
            # Return internal metrics
            summary = {}
            for metric_name, values in self.internal_metrics.items():
                summary[metric_name] = dict(values)
            
            # Add time series data
            summary['time_series'] = {}
            for series_name, data in self.time_series.items():
                if data:
                    latest = data[-1]
                    summary['time_series'][series_name] = {
                        'latest_value': latest[1],
                        'latest_timestamp': latest[0].isoformat(),
                        'data_points': len(data)
                    }
            
            return summary
    
    def export_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format"""
        if self.enable_prometheus:
            return generate_latest(self.registry).decode('utf-8')
        else:
            # Generate Prometheus-like format from internal metrics
            lines = []
            lines.append(f"# Generated at {datetime.utcnow().isoformat()}")
            
            for metric_name, labels_dict in self.internal_metrics.items():
                for labels, value in labels_dict.items():
                    if labels:
                        label_str = '{' + labels + '}'
                    else:
                        label_str = ''
                    lines.append(f"{metric_name}{label_str} {value}")
            
            return '\n'.join(lines)


# Decorators for automatic metrics collection
def track_time(metrics: MetricsCollector, metric_name: str, **labels):
    """Decorator to track execution time"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Record metrics based on metric type
                if hasattr(metrics, metric_name):
                    metric = getattr(metrics, metric_name)
                    if hasattr(metric, 'observe'):
                        if labels:
                            metric.labels(**labels).observe(duration)
                        else:
                            metric.observe(duration)
                
                return result
            except Exception as e:
                duration = time.time() - start_time
                # Record error metrics if available
                raise
        return wrapper
    return decorator


def count_calls(metrics: MetricsCollector, metric_name: str, **labels):
    """Decorator to count function calls"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                
                # Record successful call
                if hasattr(metrics, metric_name):
                    metric = getattr(metrics, metric_name)
                    if hasattr(metric, 'inc'):
                        success_labels = {**labels, 'status': 'success'}
                        metric.labels(**success_labels).inc()
                
                return result
            except Exception as e:
                # Record failed call
                if hasattr(metrics, metric_name):
                    metric = getattr(metrics, metric_name)
                    if hasattr(metric, 'inc'):
                        error_labels = {**labels, 'status': 'error'}
                        metric.labels(**error_labels).inc()
                raise
        return wrapper
    return decorator


# Global metrics instance
metrics = MetricsCollector()


# Context managers for timing
class TimeMetric:
    """Context manager for timing operations"""
    
    def __init__(self, metrics_collector: MetricsCollector, metric_name: str, **labels):
        self.metrics = metrics_collector
        self.metric_name = metric_name
        self.labels = labels
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            
            if hasattr(self.metrics, self.metric_name):
                metric = getattr(self.metrics, self.metric_name)
                if hasattr(metric, 'observe'):
                    if self.labels:
                        metric.labels(**self.labels).observe(duration)
                    else:
                        metric.observe(duration)


# Health check metrics aggregator
class HealthMetrics:
    """Aggregate health metrics for monitoring"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status"""
        try:
            # System health
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Application health indicators
            health_status = {
                'timestamp': datetime.utcnow().isoformat(),
                'system': {
                    'cpu_usage_percent': cpu_percent,
                    'memory_usage_percent': memory.percent,
                    'memory_available_bytes': memory.available,
                    'disk_usage_percent': (disk.used / disk.total) * 100,
                    'disk_free_bytes': disk.free
                },
                'application': {
                    'prometheus_enabled': self.metrics.enable_prometheus,
                    'monitoring_active': self.metrics.monitoring_active
                },
                'status': 'healthy'
            }
            
            # Determine overall health
            if cpu_percent > 90:
                health_status['status'] = 'degraded'
                health_status['warnings'] = health_status.get('warnings', [])
                health_status['warnings'].append('High CPU usage')
            
            if memory.percent > 90:
                health_status['status'] = 'degraded'
                health_status['warnings'] = health_status.get('warnings', [])
                health_status['warnings'].append('High memory usage')
            
            if (disk.used / disk.total) > 0.9:
                health_status['status'] = 'degraded'
                health_status['warnings'] = health_status.get('warnings', [])
                health_status['warnings'].append('Low disk space')
            
            return health_status
            
        except Exception as e:
            return {
                'timestamp': datetime.utcnow().isoformat(),
                'status': 'unhealthy',
                'error': str(e)
            }