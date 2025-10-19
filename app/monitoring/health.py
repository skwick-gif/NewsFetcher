"""
Health Check System
Comprehensive health monitoring for all system components
"""

import asyncio
import time
import psutil
import redis
import asyncpg
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import httpx
import json
import os

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from app.config.settings import get_settings
from app.monitoring.logging import get_logger


class HealthStatus(Enum):
    """Health check status enumeration"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Individual health check result"""
    component: str
    status: HealthStatus
    message: str
    timestamp: datetime
    response_time_ms: float
    details: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = asdict(self)
        result['status'] = self.status.value
        result['timestamp'] = self.timestamp.isoformat()
        return result


@dataclass
class SystemHealthStatus:
    """Overall system health status"""
    overall_status: HealthStatus
    checks: List[HealthCheckResult]
    timestamp: datetime
    summary: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'overall_status': self.overall_status.value,
            'checks': [check.to_dict() for check in self.checks],
            'timestamp': self.timestamp.isoformat(),
            'summary': self.summary
        }


class BaseHealthChecker:
    """Base class for health checkers"""
    
    def __init__(self, name: str, timeout: float = 5.0):
        self.name = name
        self.timeout = timeout
        self.logger = get_logger(f'app.health.{name}')
    
    async def check(self) -> HealthCheckResult:
        """Perform health check"""
        start_time = time.time()
        
        try:
            status, message, details = await asyncio.wait_for(
                self._perform_check(), 
                timeout=self.timeout
            )
            
            response_time = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                component=self.name,
                status=status,
                message=message,
                timestamp=datetime.utcnow(),
                response_time_ms=response_time,
                details=details
            )
            
        except asyncio.TimeoutError:
            response_time = (time.time() - start_time) * 1000
            return HealthCheckResult(
                component=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check timed out after {self.timeout}s",
                timestamp=datetime.utcnow(),
                response_time_ms=response_time,
                details={'timeout': self.timeout}
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            self.logger.error(f"Health check failed for {self.name}: {e}", exc_info=True)
            
            return HealthCheckResult(
                component=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {str(e)}",
                timestamp=datetime.utcnow(),
                response_time_ms=response_time,
                details={'error': str(e), 'error_type': type(e).__name__}
            )
    
    async def _perform_check(self) -> tuple[HealthStatus, str, Dict[str, Any]]:
        """Override this method in subclasses"""
        raise NotImplementedError


class DatabaseHealthChecker(BaseHealthChecker):
    """PostgreSQL database health checker"""
    
    def __init__(self, database_url: str, timeout: float = 5.0):
        super().__init__("database", timeout)
        self.database_url = database_url
    
    async def _perform_check(self) -> tuple[HealthStatus, str, Dict[str, Any]]:
        try:
            # Create engine and test connection
            engine = create_engine(self.database_url)
            
            with engine.connect() as conn:
                # Test basic connectivity
                result = conn.execute(text("SELECT 1"))
                result.fetchone()
                
                # Get database stats
                db_stats = conn.execute(text("""
                    SELECT 
                        count(*) as connection_count,
                        current_database() as database_name,
                        version() as postgres_version
                    FROM pg_stat_activity 
                    WHERE state = 'active'
                """)).fetchone()
                
                # Check table existence
                table_count = conn.execute(text("""
                    SELECT count(*) 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public'
                """)).scalar()
                
                details = {
                    'active_connections': db_stats[0],
                    'database_name': db_stats[1],
                    'postgres_version': db_stats[2].split(' ')[0],
                    'table_count': table_count
                }
                
                return HealthStatus.HEALTHY, "Database connection successful", details
                
        except SQLAlchemyError as e:
            return HealthStatus.UNHEALTHY, f"Database error: {str(e)}", {'error': str(e)}
        except Exception as e:
            return HealthStatus.UNHEALTHY, f"Unexpected database error: {str(e)}", {'error': str(e)}


class RedisHealthChecker(BaseHealthChecker):
    """Redis health checker"""
    
    def __init__(self, redis_url: str, timeout: float = 5.0):
        super().__init__("redis", timeout)
        self.redis_url = redis_url
    
    async def _perform_check(self) -> tuple[HealthStatus, str, Dict[str, Any]]:
        try:
            # Connect to Redis
            r = redis.from_url(self.redis_url, decode_responses=True)
            
            # Test basic operations
            test_key = f"health_check_{int(time.time())}"
            r.set(test_key, "test_value", ex=10)
            value = r.get(test_key)
            r.delete(test_key)
            
            if value != "test_value":
                return HealthStatus.UNHEALTHY, "Redis read/write test failed", {}
            
            # Get Redis info
            info = r.info()
            
            details = {
                'redis_version': info.get('redis_version'),
                'connected_clients': info.get('connected_clients'),
                'used_memory_human': info.get('used_memory_human'),
                'total_commands_processed': info.get('total_commands_processed'),
                'keyspace_hits': info.get('keyspace_hits'),
                'keyspace_misses': info.get('keyspace_misses')
            }
            
            return HealthStatus.HEALTHY, "Redis connection and operations successful", details
            
        except redis.RedisError as e:
            return HealthStatus.UNHEALTHY, f"Redis error: {str(e)}", {'error': str(e)}
        except Exception as e:
            return HealthStatus.UNHEALTHY, f"Unexpected Redis error: {str(e)}", {'error': str(e)}


class QdrantHealthChecker(BaseHealthChecker):
    """Qdrant vector database health checker"""
    
    def __init__(self, qdrant_url: str, timeout: float = 5.0):
        super().__init__("qdrant", timeout)
        self.qdrant_url = qdrant_url.rstrip('/')
    
    async def _perform_check(self) -> tuple[HealthStatus, str, Dict[str, Any]]:
        try:
            async with httpx.AsyncClient() as client:
                # Check cluster info
                response = await client.get(f"{self.qdrant_url}/cluster", timeout=self.timeout)
                
                if response.status_code != 200:
                    return HealthStatus.UNHEALTHY, f"Qdrant returned status {response.status_code}", {}
                
                cluster_info = response.json()
                
                # Check collections
                collections_response = await client.get(f"{self.qdrant_url}/collections", timeout=self.timeout)
                collections_info = collections_response.json() if collections_response.status_code == 200 else {}
                
                details = {
                    'status': cluster_info.get('result', {}).get('status'),
                    'peer_id': cluster_info.get('result', {}).get('peer_id'),
                    'collections_count': len(collections_info.get('result', {}).get('collections', []))
                }
                
                return HealthStatus.HEALTHY, "Qdrant connection successful", details
                
        except httpx.TimeoutException:
            return HealthStatus.UNHEALTHY, "Qdrant connection timeout", {}
        except httpx.RequestError as e:
            return HealthStatus.UNHEALTHY, f"Qdrant connection error: {str(e)}", {'error': str(e)}
        except Exception as e:
            return HealthStatus.UNHEALTHY, f"Unexpected Qdrant error: {str(e)}", {'error': str(e)}


class CeleryHealthChecker(BaseHealthChecker):
    """Celery task queue health checker"""
    
    def __init__(self, broker_url: str, timeout: float = 5.0):
        super().__init__("celery", timeout)
        self.broker_url = broker_url
    
    async def _perform_check(self) -> tuple[HealthStatus, str, Dict[str, Any]]:
        try:
            from celery import Celery
            
            # Create Celery app for inspection
            app = Celery('health_check', broker=self.broker_url)
            inspect = app.control.inspect()
            
            # Check active workers
            stats = inspect.stats()
            active_tasks = inspect.active()
            scheduled_tasks = inspect.scheduled()
            
            if not stats:
                return HealthStatus.UNHEALTHY, "No Celery workers available", {}
            
            worker_count = len(stats)
            total_active_tasks = sum(len(tasks) for tasks in (active_tasks or {}).values())
            total_scheduled_tasks = sum(len(tasks) for tasks in (scheduled_tasks or {}).values())
            
            details = {
                'worker_count': worker_count,
                'workers': list(stats.keys()) if stats else [],
                'active_tasks': total_active_tasks,
                'scheduled_tasks': total_scheduled_tasks
            }
            
            # Check broker connectivity (Redis)
            if self.broker_url.startswith('redis://'):
                redis_checker = RedisHealthChecker(self.broker_url, timeout=2.0)
                redis_result = await redis_checker.check()
                
                if redis_result.status != HealthStatus.HEALTHY:
                    return HealthStatus.DEGRADED, "Celery workers active but broker unhealthy", details
            
            return HealthStatus.HEALTHY, f"Celery operational with {worker_count} workers", details
            
        except ImportError:
            return HealthStatus.UNKNOWN, "Celery not available (import error)", {}
        except Exception as e:
            return HealthStatus.UNHEALTHY, f"Celery check failed: {str(e)}", {'error': str(e)}


class SystemResourcesHealthChecker(BaseHealthChecker):
    """System resources health checker"""
    
    def __init__(self, 
                 cpu_threshold: float = 90.0,
                 memory_threshold: float = 90.0, 
                 disk_threshold: float = 90.0,
                 timeout: float = 5.0):
        super().__init__("system_resources", timeout)
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
        self.disk_threshold = disk_threshold
    
    async def _perform_check(self) -> tuple[HealthStatus, str, Dict[str, Any]]:
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            
            # Network stats
            network = psutil.net_io_counters()
            
            # Process info
            process = psutil.Process()
            process_memory = process.memory_info()
            
            details = {
                'cpu': {
                    'usage_percent': cpu_percent,
                    'threshold': self.cpu_threshold,
                    'status': 'ok' if cpu_percent < self.cpu_threshold else 'high'
                },
                'memory': {
                    'usage_percent': memory.percent,
                    'used_bytes': memory.used,
                    'available_bytes': memory.available,
                    'total_bytes': memory.total,
                    'threshold': self.memory_threshold,
                    'status': 'ok' if memory.percent < self.memory_threshold else 'high'
                },
                'disk': {
                    'usage_percent': disk_percent,
                    'used_bytes': disk.used,
                    'free_bytes': disk.free,
                    'total_bytes': disk.total,
                    'threshold': self.disk_threshold,
                    'status': 'ok' if disk_percent < self.disk_threshold else 'high'
                },
                'network': {
                    'bytes_sent': network.bytes_sent,
                    'bytes_recv': network.bytes_recv,
                    'packets_sent': network.packets_sent,
                    'packets_recv': network.packets_recv
                },
                'process': {
                    'pid': process.pid,
                    'memory_rss': process_memory.rss,
                    'memory_vms': process_memory.vms,
                    'num_threads': process.num_threads(),
                    'num_fds': process.num_fds() if hasattr(process, 'num_fds') else None
                }
            }
            
            # Determine status
            issues = []
            
            if cpu_percent >= self.cpu_threshold:
                issues.append(f"High CPU usage: {cpu_percent:.1f}%")
            
            if memory.percent >= self.memory_threshold:
                issues.append(f"High memory usage: {memory.percent:.1f}%")
            
            if disk_percent >= self.disk_threshold:
                issues.append(f"High disk usage: {disk_percent:.1f}%")
            
            if issues:
                status = HealthStatus.DEGRADED
                message = "; ".join(issues)
            else:
                status = HealthStatus.HEALTHY
                message = "System resources within normal limits"
            
            return status, message, details
            
        except Exception as e:
            return HealthStatus.UNHEALTHY, f"System resources check failed: {str(e)}", {'error': str(e)}


class ExternalServicesHealthChecker(BaseHealthChecker):
    """External services health checker"""
    
    def __init__(self, services: Dict[str, str], timeout: float = 5.0):
        super().__init__("external_services", timeout)
        self.services = services  # {'service_name': 'url'}
    
    async def _perform_check(self) -> tuple[HealthStatus, str, Dict[str, Any]]:
        service_results = {}
        overall_healthy = True
        
        async with httpx.AsyncClient() as client:
            for service_name, url in self.services.items():
                try:
                    start_time = time.time()
                    response = await client.get(url, timeout=self.timeout)
                    response_time = (time.time() - start_time) * 1000
                    
                    service_results[service_name] = {
                        'status': 'healthy' if response.status_code == 200 else 'unhealthy',
                        'status_code': response.status_code,
                        'response_time_ms': response_time,
                        'url': url
                    }
                    
                    if response.status_code != 200:
                        overall_healthy = False
                        
                except Exception as e:
                    service_results[service_name] = {
                        'status': 'unhealthy',
                        'error': str(e),
                        'url': url
                    }
                    overall_healthy = False
        
        details = {'services': service_results}
        
        if overall_healthy:
            return HealthStatus.HEALTHY, "All external services accessible", details
        else:
            unhealthy_services = [name for name, result in service_results.items() 
                                if result['status'] == 'unhealthy']
            return HealthStatus.DEGRADED, f"Some external services unavailable: {', '.join(unhealthy_services)}", details


class HealthMonitor:
    """Main health monitoring system"""
    
    def __init__(self):
        self.logger = get_logger('app.health')
        self.checkers: List[BaseHealthChecker] = []
        self.last_check_time: Optional[datetime] = None
        self.last_results: List[HealthCheckResult] = []
        
        # Initialize checkers based on configuration
        self._initialize_checkers()
    
    def _initialize_checkers(self):
        """Initialize health checkers based on configuration"""
        settings = get_settings()
        
        # Database checker
        if settings.DATABASE_URL:
            self.checkers.append(DatabaseHealthChecker(settings.DATABASE_URL))
        
        # Redis checker
        if settings.REDIS_URL:
            self.checkers.append(RedisHealthChecker(settings.REDIS_URL))
        
        # Qdrant checker
        if hasattr(settings, 'QDRANT_URL') and settings.QDRANT_URL:
            self.checkers.append(QdrantHealthChecker(settings.QDRANT_URL))
        
        # Celery checker
        if settings.REDIS_URL:  # Assuming Redis is used as Celery broker
            self.checkers.append(CeleryHealthChecker(settings.REDIS_URL))
        
        # System resources checker
        self.checkers.append(SystemResourcesHealthChecker())
        
        # External services checker
        external_services = {}
        if hasattr(settings, 'EXTERNAL_SERVICES') and settings.EXTERNAL_SERVICES:
            external_services = settings.EXTERNAL_SERVICES
        
        if external_services:
            self.checkers.append(ExternalServicesHealthChecker(external_services))
    
    async def check_health(self, component: str = None) -> SystemHealthStatus:
        """Perform health checks"""
        start_time = datetime.utcnow()
        
        # Filter checkers if specific component requested
        checkers_to_run = self.checkers
        if component:
            checkers_to_run = [c for c in self.checkers if c.name == component]
            
            if not checkers_to_run:
                # Return unknown status for non-existent component
                unknown_result = HealthCheckResult(
                    component=component,
                    status=HealthStatus.UNKNOWN,
                    message=f"Component '{component}' not found",
                    timestamp=start_time,
                    response_time_ms=0.0
                )
                
                return SystemHealthStatus(
                    overall_status=HealthStatus.UNKNOWN,
                    checks=[unknown_result],
                    timestamp=start_time,
                    summary={'message': f"Component '{component}' not found"}
                )
        
        # Run health checks concurrently
        try:
            results = await asyncio.gather(
                *[checker.check() for checker in checkers_to_run],
                return_exceptions=True
            )
            
            # Process results
            health_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    # Handle checker that raised an exception
                    checker = checkers_to_run[i]
                    health_results.append(HealthCheckResult(
                        component=checker.name,
                        status=HealthStatus.UNHEALTHY,
                        message=f"Health check failed: {str(result)}",
                        timestamp=start_time,
                        response_time_ms=0.0,
                        details={'error': str(result)}
                    ))
                else:
                    health_results.append(result)
            
            # Determine overall status
            overall_status = self._determine_overall_status(health_results)
            
            # Create summary
            summary = self._create_summary(health_results, start_time)
            
            # Store results
            self.last_check_time = start_time
            self.last_results = health_results
            
            return SystemHealthStatus(
                overall_status=overall_status,
                checks=health_results,
                timestamp=start_time,
                summary=summary
            )
            
        except Exception as e:
            self.logger.error(f"Health check system failed: {e}", exc_info=True)
            
            error_result = HealthCheckResult(
                component="health_system",
                status=HealthStatus.UNHEALTHY,
                message=f"Health check system error: {str(e)}",
                timestamp=start_time,
                response_time_ms=0.0,
                details={'error': str(e)}
            )
            
            return SystemHealthStatus(
                overall_status=HealthStatus.UNHEALTHY,
                checks=[error_result],
                timestamp=start_time,
                summary={'error': 'Health check system failure'}
            )
    
    def _determine_overall_status(self, results: List[HealthCheckResult]) -> HealthStatus:
        """Determine overall system health status"""
        if not results:
            return HealthStatus.UNKNOWN
        
        statuses = [result.status for result in results]
        
        # If any component is unhealthy, system is unhealthy
        if HealthStatus.UNHEALTHY in statuses:
            return HealthStatus.UNHEALTHY
        
        # If any component is degraded, system is degraded
        if HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED
        
        # If any component is unknown, system status is unknown
        if HealthStatus.UNKNOWN in statuses:
            return HealthStatus.UNKNOWN
        
        # All components healthy
        return HealthStatus.HEALTHY
    
    def _create_summary(self, results: List[HealthCheckResult], check_time: datetime) -> Dict[str, Any]:
        """Create health check summary"""
        status_counts = {status.value: 0 for status in HealthStatus}
        total_response_time = 0.0
        
        for result in results:
            status_counts[result.status.value] += 1
            total_response_time += result.response_time_ms
        
        avg_response_time = total_response_time / len(results) if results else 0.0
        
        return {
            'total_checks': len(results),
            'status_counts': status_counts,
            'average_response_time_ms': round(avg_response_time, 2),
            'check_duration_seconds': (datetime.utcnow() - check_time).total_seconds(),
            'timestamp': check_time.isoformat()
        }
    
    def get_component_names(self) -> List[str]:
        """Get list of available component names"""
        return [checker.name for checker in self.checkers]


# Global health monitor instance
health_monitor = HealthMonitor()