"""
Structured Logging Configuration
Advanced logging with JSON formatting, correlation IDs, and centralized collection
"""

import logging
import logging.config
import json
import sys
import os
import uuid
import threading
from datetime import datetime
from typing import Dict, Any, Optional, List
from contextlib import contextmanager
from pathlib import Path
import traceback

try:
    import colorama
    from colorama import Fore, Back, Style
    colorama.init()
    COLORS_AVAILABLE = True
except ImportError:
    COLORS_AVAILABLE = False


class CorrelationIdFilter(logging.Filter):
    """Add correlation ID to log records"""
    
    def __init__(self):
        super().__init__()
        self.local = threading.local()
    
    def filter(self, record):
        record.correlation_id = getattr(self.local, 'correlation_id', None) or 'no-correlation'
        return True
    
    def set_correlation_id(self, correlation_id: str):
        """Set correlation ID for current thread"""
        self.local.correlation_id = correlation_id
    
    def clear_correlation_id(self):
        """Clear correlation ID for current thread"""
        if hasattr(self.local, 'correlation_id'):
            delattr(self.local, 'correlation_id')


class JSONFormatter(logging.Formatter):
    """JSON log formatter with structured fields"""
    
    def __init__(self, include_trace: bool = True):
        super().__init__()
        self.include_trace = include_trace
    
    def format(self, record):
        # Basic log data
        log_data = {
            'timestamp': datetime.utcfromtimestamp(record.created).isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'correlation_id': getattr(record, 'correlation_id', None),
            'thread_id': record.thread,
            'thread_name': record.threadName,
            'process_id': record.process,
        }
        
        # Add source location
        if record.pathname:
            log_data['source'] = {
                'file': record.pathname,
                'line': record.lineno,
                'function': record.funcName,
                'module': record.module
            }
        
        # Add exception info
        if record.exc_info and self.include_trace:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        # Add custom fields from extra
        for key, value in record.__dict__.items():
            if key not in [
                'name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                'filename', 'module', 'lineno', 'funcName', 'created',
                'msecs', 'relativeCreated', 'thread', 'threadName',
                'processName', 'process', 'getMessage', 'exc_info',
                'exc_text', 'stack_info', 'correlation_id'
            ]:
                log_data['extra'] = log_data.get('extra', {})
                log_data['extra'][key] = value
        
        return json.dumps(log_data, default=str, ensure_ascii=False)


class ColoredFormatter(logging.Formatter):
    """Colored console formatter for development"""
    
    COLORS = {
        'DEBUG': Fore.BLUE,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Back.WHITE + Style.BRIGHT,
    } if COLORS_AVAILABLE else {}
    
    def __init__(self):
        super().__init__()
        self.format_string = (
            '{color}[{timestamp}] {level:8} {correlation_id:12} '
            '{logger:20} {message}{reset}'
        )
    
    def format(self, record):
        color = self.COLORS.get(record.levelname, '') if COLORS_AVAILABLE else ''
        reset = Style.RESET_ALL if COLORS_AVAILABLE else ''
        
        correlation_id = getattr(record, 'correlation_id', 'no-correlation')[:12]
        
        formatted = self.format_string.format(
            color=color,
            timestamp=datetime.utcfromtimestamp(record.created).strftime('%H:%M:%S'),
            level=record.levelname,
            correlation_id=correlation_id,
            logger=record.name[-20:],  # Truncate long logger names
            message=record.getMessage(),
            reset=reset
        )
        
        # Add exception info
        if record.exc_info:
            formatted += '\n' + ''.join(traceback.format_exception(*record.exc_info))
        
        return formatted


class SecurityLogFilter(logging.Filter):
    """Filter to identify and mark security-related logs"""
    
    SECURITY_KEYWORDS = [
        'authentication', 'authorization', 'login', 'logout', 'token',
        'password', 'security', 'breach', 'attack', 'unauthorized',
        'forbidden', 'access denied', 'rate limit', 'suspicious'
    ]
    
    def filter(self, record):
        message = record.getMessage().lower()
        
        # Mark as security log if contains security keywords
        record.is_security = any(keyword in message for keyword in self.SECURITY_KEYWORDS)
        
        # Add security context
        if record.is_security:
            record.log_category = 'security'
        
        return True


class PerformanceLogFilter(logging.Filter):
    """Filter to identify and mark performance-related logs"""
    
    PERFORMANCE_KEYWORDS = [
        'slow', 'timeout', 'performance', 'latency', 'duration',
        'memory', 'cpu', 'database', 'query', 'cache', 'response time'
    ]
    
    def filter(self, record):
        message = record.getMessage().lower()
        
        # Mark as performance log if contains performance keywords
        record.is_performance = any(keyword in message for keyword in self.PERFORMANCE_KEYWORDS)
        
        # Add performance context
        if record.is_performance:
            record.log_category = getattr(record, 'log_category', 'performance')
        
        return True


class LoggerManager:
    """Centralized logger management"""
    
    def __init__(self, log_dir: str = "logs", app_name: str = "tariff-radar"):
        self.log_dir = Path(log_dir)
        self.app_name = app_name
        self.correlation_filter = CorrelationIdFilter()
        
        # Create log directory
        self.log_dir.mkdir(exist_ok=True)
        
        # Setup logging configuration
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup comprehensive logging configuration"""
        
        config = {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'json': {
                    '()': JSONFormatter,
                    'include_trace': True
                },
                'colored': {
                    '()': ColoredFormatter
                },
                'simple': {
                    'format': '[%(asctime)s] %(levelname)-8s %(name)-20s %(message)s',
                    'datefmt': '%Y-%m-%d %H:%M:%S'
                }
            },
            'filters': {
                'correlation': {
                    '()': lambda: self.correlation_filter
                },
                'security': {
                    '()': SecurityLogFilter
                },
                'performance': {
                    '()': PerformanceLogFilter
                }
            },
            'handlers': {
                'console': {
                    'class': 'logging.StreamHandler',
                    'level': 'INFO',
                    'formatter': 'colored' if COLORS_AVAILABLE else 'simple',
                    'stream': 'ext://sys.stdout',
                    'filters': ['correlation', 'security', 'performance']
                },
                'file_json': {
                    'class': 'logging.handlers.RotatingFileHandler',
                    'level': 'DEBUG',
                    'formatter': 'json',
                    'filename': str(self.log_dir / f'{self.app_name}.json'),
                    'maxBytes': 50 * 1024 * 1024,  # 50MB
                    'backupCount': 10,
                    'encoding': 'utf-8',
                    'filters': ['correlation', 'security', 'performance']
                },
                'file_error': {
                    'class': 'logging.handlers.RotatingFileHandler',
                    'level': 'ERROR',
                    'formatter': 'json',
                    'filename': str(self.log_dir / f'{self.app_name}_errors.json'),
                    'maxBytes': 10 * 1024 * 1024,  # 10MB
                    'backupCount': 5,
                    'encoding': 'utf-8',
                    'filters': ['correlation', 'security', 'performance']
                },
                'file_security': {
                    'class': 'logging.handlers.RotatingFileHandler',
                    'level': 'INFO',
                    'formatter': 'json',
                    'filename': str(self.log_dir / f'{self.app_name}_security.json'),
                    'maxBytes': 10 * 1024 * 1024,  # 10MB
                    'backupCount': 20,  # Keep more security logs
                    'encoding': 'utf-8',
                    'filters': ['correlation', 'security']
                },
                'file_performance': {
                    'class': 'logging.handlers.RotatingFileHandler',
                    'level': 'INFO',
                    'formatter': 'json',
                    'filename': str(self.log_dir / f'{self.app_name}_performance.json'),
                    'maxBytes': 10 * 1024 * 1024,  # 10MB
                    'backupCount': 5,
                    'encoding': 'utf-8',
                    'filters': ['correlation', 'performance']
                }
            },
            'loggers': {
                # Application loggers
                'app': {
                    'level': 'DEBUG',
                    'handlers': ['console', 'file_json', 'file_error'],
                    'propagate': False
                },
                'app.security': {
                    'level': 'INFO',
                    'handlers': ['console', 'file_json', 'file_security'],
                    'propagate': False
                },
                'app.performance': {
                    'level': 'INFO',
                    'handlers': ['console', 'file_json', 'file_performance'],
                    'propagate': False
                },
                # Third-party library loggers
                'sqlalchemy.engine': {
                    'level': 'WARNING',
                    'handlers': ['file_json'],
                    'propagate': False
                },
                'celery': {
                    'level': 'INFO',
                    'handlers': ['file_json'],
                    'propagate': False
                },
                'uvicorn': {
                    'level': 'INFO',
                    'handlers': ['console', 'file_json'],
                    'propagate': False
                },
                'fastapi': {
                    'level': 'INFO',
                    'handlers': ['file_json'],
                    'propagate': False
                }
            },
            'root': {
                'level': 'WARNING',
                'handlers': ['console', 'file_json']
            }
        }
        
        logging.config.dictConfig(config)
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get logger with proper configuration"""
        return logging.getLogger(name)
    
    @contextmanager
    def correlation_context(self, correlation_id: str = None):
        """Context manager for correlation ID"""
        if correlation_id is None:
            correlation_id = str(uuid.uuid4())[:8]
        
        self.correlation_filter.set_correlation_id(correlation_id)
        try:
            yield correlation_id
        finally:
            self.correlation_filter.clear_correlation_id()
    
    def get_log_stats(self) -> Dict[str, Any]:
        """Get logging statistics"""
        stats = {
            'log_directory': str(self.log_dir),
            'log_files': [],
            'total_size_bytes': 0
        }
        
        for log_file in self.log_dir.glob('*.json'):
            if log_file.is_file():
                size = log_file.stat().st_size
                stats['log_files'].append({
                    'name': log_file.name,
                    'size_bytes': size,
                    'size_mb': round(size / (1024 * 1024), 2),
                    'modified': datetime.fromtimestamp(log_file.stat().st_mtime).isoformat()
                })
                stats['total_size_bytes'] += size
        
        stats['total_size_mb'] = round(stats['total_size_bytes'] / (1024 * 1024), 2)
        return stats


# Performance timing decorator
def log_performance(logger: logging.Logger, operation: str = None):
    """Decorator to log performance metrics"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            op_name = operation or f"{func.__module__}.{func.__name__}"
            start_time = datetime.utcnow()
            
            try:
                result = func(*args, **kwargs)
                duration = (datetime.utcnow() - start_time).total_seconds()
                
                logger.info(
                    f"Operation completed: {op_name}",
                    extra={
                        'operation': op_name,
                        'duration_seconds': duration,
                        'status': 'success',
                        'performance_metric': True
                    }
                )
                
                return result
                
            except Exception as e:
                duration = (datetime.utcnow() - start_time).total_seconds()
                
                logger.error(
                    f"Operation failed: {op_name}",
                    extra={
                        'operation': op_name,
                        'duration_seconds': duration,
                        'status': 'error',
                        'error_type': type(e).__name__,
                        'error_message': str(e),
                        'performance_metric': True
                    },
                    exc_info=True
                )
                
                raise
        return wrapper
    return decorator


# Security event logger
class SecurityLogger:
    """Specialized logger for security events"""
    
    def __init__(self, logger_manager: LoggerManager):
        self.logger = logger_manager.get_logger('app.security')
    
    def log_authentication_attempt(self, username: str, success: bool, ip_address: str = None, user_agent: str = None):
        """Log authentication attempts"""
        self.logger.info(
            f"Authentication attempt: {username}",
            extra={
                'event_type': 'authentication',
                'username': username,
                'success': success,
                'ip_address': ip_address,
                'user_agent': user_agent,
                'security_event': True
            }
        )
    
    def log_authorization_failure(self, username: str, resource: str, action: str, ip_address: str = None):
        """Log authorization failures"""
        self.logger.warning(
            f"Authorization denied: {username} tried to {action} {resource}",
            extra={
                'event_type': 'authorization_failure',
                'username': username,
                'resource': resource,
                'action': action,
                'ip_address': ip_address,
                'security_event': True
            }
        )
    
    def log_rate_limit_exceeded(self, identifier: str, limit_type: str, ip_address: str = None):
        """Log rate limit violations"""
        self.logger.warning(
            f"Rate limit exceeded: {identifier}",
            extra={
                'event_type': 'rate_limit_exceeded',
                'identifier': identifier,
                'limit_type': limit_type,
                'ip_address': ip_address,
                'security_event': True
            }
        )
    
    def log_suspicious_activity(self, description: str, details: Dict[str, Any] = None):
        """Log suspicious activities"""
        self.logger.error(
            f"Suspicious activity detected: {description}",
            extra={
                'event_type': 'suspicious_activity',
                'description': description,
                'details': details or {},
                'security_event': True
            }
        )


# Global logger manager instance
logger_manager = LoggerManager()

# Convenience functions
def get_logger(name: str) -> logging.Logger:
    """Get a configured logger"""
    return logger_manager.get_logger(name)

def with_correlation(correlation_id: str = None):
    """Context manager for correlation ID"""
    return logger_manager.correlation_context(correlation_id)

# Pre-configured loggers
app_logger = get_logger('app')
security_logger = SecurityLogger(logger_manager)
performance_logger = get_logger('app.performance')