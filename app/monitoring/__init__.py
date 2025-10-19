"""
Monitoring System Initialization
Initialize and configure all monitoring components
"""

import asyncio
import os
from pathlib import Path
from typing import Optional

from app.monitoring.metrics import metrics
from app.monitoring.alerts import alert_manager
from app.monitoring.health import health_monitor
from app.monitoring.logging import logger_manager, get_logger
from app.config.settings import get_settings


logger = get_logger('app.monitoring.init')


async def initialize_monitoring_system(
    enable_metrics: bool = True,
    enable_alerts: bool = True,
    metrics_interval: int = 30,
    alerts_interval: int = 60
) -> dict:
    """
    Initialize complete monitoring system
    
    Args:
        enable_metrics: Enable Prometheus metrics collection
        enable_alerts: Enable alert monitoring  
        metrics_interval: Metrics collection interval in seconds
        alerts_interval: Alert checking interval in seconds
    
    Returns:
        dict: Initialization status
    """
    logger.info("Initializing monitoring system...")
    
    try:
        settings = get_settings()
        initialization_status = {
            'timestamp': '2024-01-01T00:00:00Z',
            'components': {},
            'errors': []
        }
        
        # Create monitoring directories
        await _create_monitoring_directories()
        initialization_status['components']['directories'] = 'success'
        
        # Initialize logging system (already done in logger_manager)
        logger.info("Logging system initialized")
        initialization_status['components']['logging'] = 'success'
        
        # Initialize metrics collection
        if enable_metrics:
            try:
                metrics.start_monitoring(interval=metrics_interval)
                logger.info(f"Metrics collection started with {metrics_interval}s interval")
                initialization_status['components']['metrics'] = 'success'
            except Exception as e:
                logger.error(f"Failed to start metrics collection: {e}")
                initialization_status['components']['metrics'] = 'failed'
                initialization_status['errors'].append(f"Metrics: {str(e)}")
        else:
            initialization_status['components']['metrics'] = 'disabled'
        
        # Initialize alert monitoring
        if enable_alerts:
            try:
                alert_manager.start_monitoring(interval=alerts_interval)
                logger.info(f"Alert monitoring started with {alerts_interval}s interval")
                initialization_status['components']['alerts'] = 'success'
            except Exception as e:
                logger.error(f"Failed to start alert monitoring: {e}")
                initialization_status['components']['alerts'] = 'failed'
                initialization_status['errors'].append(f"Alerts: {str(e)}")
        else:
            initialization_status['components']['alerts'] = 'disabled'
        
        # Test health checks
        try:
            health_status = await health_monitor.check_health()
            logger.info(f"Health check system operational - Status: {health_status.overall_status.value}")
            initialization_status['components']['health_checks'] = 'success'
            initialization_status['initial_health_status'] = health_status.overall_status.value
        except Exception as e:
            logger.error(f"Health check system failed: {e}")
            initialization_status['components']['health_checks'] = 'failed'
            initialization_status['errors'].append(f"Health checks: {str(e)}")
        
        # Log initialization summary
        success_count = len([status for status in initialization_status['components'].values() 
                           if status == 'success'])
        total_components = len(initialization_status['components'])
        
        if initialization_status['errors']:
            logger.warning(
                f"Monitoring system initialized with issues: {success_count}/{total_components} components successful",
                extra={'initialization_status': initialization_status}
            )
        else:
            logger.info(
                f"Monitoring system fully initialized: {success_count}/{total_components} components successful",
                extra={'initialization_status': initialization_status}
            )
        
        return initialization_status
        
    except Exception as e:
        logger.error(f"Critical error initializing monitoring system: {e}", exc_info=True)
        return {
            'timestamp': '2024-01-01T00:00:00Z',
            'components': {'initialization': 'critical_failure'},
            'errors': [f"Critical initialization error: {str(e)}"]
        }


async def _create_monitoring_directories():
    """Create required monitoring directories"""
    directories = [
        'logs',
        'logs/archive',
        'monitoring/grafana',
        'monitoring/prometheus',
        'monitoring/config'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created monitoring directory: {directory}")


async def shutdown_monitoring_system():
    """Gracefully shutdown monitoring system"""
    logger.info("Shutting down monitoring system...")
    
    try:
        # Stop metrics collection
        if metrics.monitoring_active:
            metrics.stop_monitoring()
            logger.info("Metrics collection stopped")
        
        # Stop alert monitoring  
        if alert_manager.monitoring_active:
            alert_manager.stop_monitoring()
            logger.info("Alert monitoring stopped")
        
        logger.info("Monitoring system shutdown complete")
        
    except Exception as e:
        logger.error(f"Error during monitoring system shutdown: {e}", exc_info=True)


def get_monitoring_status() -> dict:
    """Get current monitoring system status"""
    return {
        'timestamp': '2024-01-01T00:00:00Z',
        'components': {
            'metrics_collection': {
                'active': metrics.monitoring_active,
                'prometheus_enabled': metrics.enable_prometheus
            },
            'alert_monitoring': {
                'active': alert_manager.monitoring_active,
                'active_alerts_count': len(alert_manager.get_active_alerts()),
                'total_rules': len(alert_manager.rules)
            },
            'health_checks': {
                'available_components': len(health_monitor.get_component_names()),
                'last_check_time': health_monitor.last_check_time.isoformat() if health_monitor.last_check_time else None
            },
            'logging': {
                'log_directory': str(logger_manager.log_dir),
                'correlation_tracking': True
            }
        },
        'system_info': {
            'monitoring_enabled': True,
            'version': '1.0.0'
        }
    }


# Monitoring configuration for different environments
MONITORING_CONFIGS = {
    'development': {
        'enable_metrics': True,
        'enable_alerts': False,  # Disable alerts in development
        'metrics_interval': 60,
        'alerts_interval': 300
    },
    'staging': {
        'enable_metrics': True,
        'enable_alerts': True,
        'metrics_interval': 30,
        'alerts_interval': 120
    },
    'production': {
        'enable_metrics': True,
        'enable_alerts': True,
        'metrics_interval': 15,
        'alerts_interval': 60
    }
}


async def initialize_for_environment(environment: str = None) -> dict:
    """Initialize monitoring system for specific environment"""
    if environment is None:
        settings = get_settings()
        environment = getattr(settings, 'ENVIRONMENT', 'production')
    
    config = MONITORING_CONFIGS.get(environment, MONITORING_CONFIGS['production'])
    
    logger.info(f"Initializing monitoring for {environment} environment")
    
    return await initialize_monitoring_system(
        enable_metrics=config['enable_metrics'],
        enable_alerts=config['enable_alerts'],
        metrics_interval=config['metrics_interval'],
        alerts_interval=config['alerts_interval']
    )


# FastAPI startup/shutdown events
async def startup_monitoring():
    """FastAPI startup event for monitoring"""
    try:
        # Initialize monitoring system for current environment
        status = await initialize_for_environment()
        
        # Log startup completion
        if status['errors']:
            logger.warning("Monitoring system started with issues", extra={'status': status})
        else:
            logger.info("Monitoring system started successfully", extra={'status': status})
            
        return status
        
    except Exception as e:
        logger.error(f"Failed to start monitoring system: {e}", exc_info=True)
        raise


async def shutdown_monitoring_event():
    """FastAPI shutdown event for monitoring"""
    try:
        await shutdown_monitoring_system()
    except Exception as e:
        logger.error(f"Error during monitoring shutdown: {e}", exc_info=True)