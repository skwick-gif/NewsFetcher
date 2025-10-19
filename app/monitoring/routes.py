"""
Monitoring API Routes
FastAPI routes for monitoring, health checks, metrics, and alerts
"""

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from fastapi.responses import PlainTextResponse, JSONResponse
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import asyncio

from app.monitoring.health import health_monitor, HealthStatus
from app.monitoring.metrics import metrics
from app.monitoring.alerts import alert_manager, AlertRule, AlertSeverity
from app.monitoring.logging import get_logger, with_correlation
from app.auth.security import require_api_key

router = APIRouter(prefix="/monitoring", tags=["monitoring"])
logger = get_logger('app.monitoring.api')


@router.get("/health")
async def get_health_status(
    component: Optional[str] = Query(None, description="Specific component to check"),
    detailed: bool = Query(False, description="Include detailed health information")
):
    """
    Get system health status
    
    Returns overall system health or specific component health.
    Use this endpoint for load balancer health checks and monitoring.
    """
    try:
        with with_correlation() as correlation_id:
            health_status = await health_monitor.check_health(component)
            
            if not detailed:
                # Simple health check response for load balancers
                if health_status.overall_status == HealthStatus.HEALTHY:
                    return {"status": "healthy", "timestamp": health_status.timestamp.isoformat()}
                else:
                    return JSONResponse(
                        status_code=503,
                        content={
                            "status": health_status.overall_status.value,
                            "timestamp": health_status.timestamp.isoformat()
                        }
                    )
            
            # Detailed health information
            return health_status.to_dict()
            
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )


@router.get("/health/components")
async def list_health_components():
    """List available health check components"""
    try:
        components = health_monitor.get_component_names()
        return {
            "components": components,
            "total_count": len(components),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to list health components: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics", response_class=PlainTextResponse)
async def get_prometheus_metrics():
    """
    Get Prometheus-formatted metrics
    
    Returns metrics in Prometheus exposition format for scraping.
    """
    try:
        metrics_data = metrics.export_prometheus_metrics()
        return PlainTextResponse(
            content=metrics_data,
            media_type="text/plain; version=0.0.4; charset=utf-8"
        )
    except Exception as e:
        logger.error(f"Failed to export metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to export metrics")


@router.get("/metrics/summary")
async def get_metrics_summary(_: str = Depends(require_api_key)):
    """
    Get metrics summary (requires authentication)
    
    Returns a summary of current metrics and system statistics.
    """
    try:
        summary = metrics.get_metrics_summary()
        return summary
    except Exception as e:
        logger.error(f"Failed to get metrics summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/metrics/start")
async def start_metrics_collection(
    interval: int = Query(30, ge=10, le=300, description="Collection interval in seconds"),
    _: str = Depends(require_api_key)
):
    """Start background metrics collection"""
    try:
        metrics.start_monitoring(interval)
        return {
            "status": "started",
            "interval_seconds": interval,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to start metrics collection: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/metrics/stop")
async def stop_metrics_collection(_: str = Depends(require_api_key)):
    """Stop background metrics collection"""
    try:
        metrics.stop_monitoring()
        return {
            "status": "stopped",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to stop metrics collection: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/alerts")
async def get_active_alerts(_: str = Depends(require_api_key)):
    """Get all active alerts"""
    try:
        active_alerts = alert_manager.get_active_alerts()
        return {
            "alerts": [alert.to_dict() for alert in active_alerts],
            "count": len(active_alerts),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get active alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/alerts/summary")
async def get_alerts_summary(_: str = Depends(require_api_key)):
    """Get alerts system summary"""
    try:
        summary = alert_manager.get_alert_summary()
        return summary
    except Exception as e:
        logger.error(f"Failed to get alerts summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/alerts/history")
async def get_alerts_history(
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of alerts to return"),
    _: str = Depends(require_api_key)
):
    """Get alert history"""
    try:
        history = alert_manager.get_alert_history(limit)
        return {
            "alerts": history,
            "count": len(history),
            "limit": limit,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get alert history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/alerts/rules")
async def list_alert_rules(_: str = Depends(require_api_key)):
    """List all alert rules"""
    try:
        rules = alert_manager.list_rules()
        return {
            "rules": [
                {
                    "name": rule.name,
                    "description": rule.description,
                    "severity": rule.severity.value,
                    "component": rule.component,
                    "condition": rule.condition,
                    "threshold": rule.threshold,
                    "duration_seconds": rule.duration_seconds,
                    "cooldown_seconds": rule.cooldown_seconds,
                    "enabled": rule.enabled,
                    "notification_channels": rule.notification_channels
                } for rule in rules
            ],
            "count": len(rules),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to list alert rules: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/alerts/rules")
async def create_alert_rule(
    rule_data: Dict[str, Any],
    _: str = Depends(require_api_key)
):
    """Create new alert rule"""
    try:
        # Validate rule data
        required_fields = ['name', 'description', 'severity', 'component', 'condition', 'threshold']
        for field in required_fields:
            if field not in rule_data:
                raise HTTPException(status_code=400, detail=f"Missing required field: {field}")
        
        # Create rule
        rule = AlertRule(
            name=rule_data['name'],
            description=rule_data['description'],
            severity=AlertSeverity(rule_data['severity']),
            component=rule_data['component'],
            condition=rule_data['condition'],
            threshold=float(rule_data['threshold']),
            duration_seconds=rule_data.get('duration_seconds', 300),
            cooldown_seconds=rule_data.get('cooldown_seconds', 1800),
            enabled=rule_data.get('enabled', True),
            notification_channels=rule_data.get('notification_channels', ['wecom', 'email'])
        )
        
        alert_manager.add_rule(rule)
        
        return {
            "status": "created",
            "rule_name": rule.name,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid rule data: {str(e)}")
    except Exception as e:
        logger.error(f"Failed to create alert rule: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/alerts/rules/{rule_name}")
async def delete_alert_rule(
    rule_name: str,
    _: str = Depends(require_api_key)
):
    """Delete alert rule"""
    try:
        rule = alert_manager.get_rule(rule_name)
        if not rule:
            raise HTTPException(status_code=404, detail=f"Rule '{rule_name}' not found")
        
        alert_manager.remove_rule(rule_name)
        
        return {
            "status": "deleted",
            "rule_name": rule_name,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete alert rule: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(
    alert_id: str,
    acknowledged_by: str = Query(..., description="Who acknowledged the alert"),
    _: str = Depends(require_api_key)
):
    """Acknowledge an active alert"""
    try:
        success = alert_manager.acknowledge_alert(alert_id, acknowledged_by)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Alert '{alert_id}' not found")
        
        return {
            "status": "acknowledged",
            "alert_id": alert_id,
            "acknowledged_by": acknowledged_by,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to acknowledge alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/alerts/{alert_id}/silence")
async def silence_alert(
    alert_id: str,
    duration_hours: int = Query(24, ge=1, le=168, description="Silence duration in hours"),
    _: str = Depends(require_api_key)
):
    """Silence an active alert"""
    try:
        success = alert_manager.silence_alert(alert_id, duration_hours)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Alert '{alert_id}' not found")
        
        return {
            "status": "silenced",
            "alert_id": alert_id,
            "duration_hours": duration_hours,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to silence alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/alerts/start")
async def start_alert_monitoring(
    interval: int = Query(60, ge=30, le=600, description="Check interval in seconds"),
    _: str = Depends(require_api_key)
):
    """Start alert monitoring"""
    try:
        alert_manager.start_monitoring(interval)
        return {
            "status": "started",
            "interval_seconds": interval,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to start alert monitoring: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/alerts/stop")
async def stop_alert_monitoring(_: str = Depends(require_api_key)):
    """Stop alert monitoring"""
    try:
        alert_manager.stop_monitoring()
        return {
            "status": "stopped",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to stop alert monitoring: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/logs/stats")
async def get_log_statistics(_: str = Depends(require_api_key)):
    """Get logging statistics"""
    try:
        from app.monitoring.logging import logger_manager
        stats = logger_manager.get_log_stats()
        return stats
    except Exception as e:
        logger.error(f"Failed to get log statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/system/status")
async def get_system_status():
    """
    Get comprehensive system status
    
    Public endpoint that provides overall system health without sensitive details.
    """
    try:
        # Get health status
        health_status = await health_monitor.check_health()
        
        # Get basic metrics
        metrics_summary = {
            'prometheus_enabled': metrics.enable_prometheus,
            'monitoring_active': metrics.monitoring_active
        }
        
        # Get alerts summary (without sensitive details)
        alerts_summary = {
            'active_alerts_count': len(alert_manager.get_active_alerts()),
            'monitoring_active': alert_manager.monitoring_active
        }
        
        return {
            "system_status": health_status.overall_status.value,
            "timestamp": datetime.utcnow().isoformat(),
            "components": {
                check.component: check.status.value 
                for check in health_status.checks
            },
            "metrics": metrics_summary,
            "alerts": alerts_summary,
            "uptime_check": True
        }
        
    except Exception as e:
        logger.error(f"System status check failed: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "system_status": "error",
                "timestamp": datetime.utcnow().isoformat(),
                "error": "System status check failed",
                "uptime_check": False
            }
        )


@router.post("/test/alert")
async def trigger_test_alert(
    severity: str = Query("low", description="Alert severity (low, medium, high, critical)"),
    _: str = Depends(require_api_key)
):
    """Trigger test alert for notification testing"""
    try:
        # Validate severity
        if severity not in ['low', 'medium', 'high', 'critical']:
            raise HTTPException(status_code=400, detail="Invalid severity")
        
        # Create test alert rule
        test_rule = AlertRule(
            name="test_alert",
            description=f"Test alert with {severity} severity",
            severity=AlertSeverity(severity),
            component="test",
            condition="greater_than",
            threshold=0.0,
            duration_seconds=0,  # Immediate
            notification_channels=['wecom', 'email']
        )
        
        # Trigger test alert
        current_value = 100.0  # Always triggers condition
        await alert_manager._trigger_alert(test_rule, current_value, {"test": True})
        
        return {
            "status": "test_alert_triggered",
            "severity": severity,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to trigger test alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dashboard")
async def get_monitoring_dashboard(_: str = Depends(require_api_key)):
    """
    Get comprehensive monitoring dashboard data
    
    Returns all monitoring information for dashboard display.
    """
    try:
        # Get all monitoring data
        health_status = await health_monitor.check_health()
        active_alerts = alert_manager.get_active_alerts()
        alert_summary = alert_manager.get_alert_summary()
        metrics_summary = metrics.get_metrics_summary()
        
        # Get recent alert history
        recent_alerts = alert_manager.get_alert_history(limit=20)
        
        dashboard_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "system_health": health_status.to_dict(),
            "alerts": {
                "active": [alert.to_dict() for alert in active_alerts],
                "summary": alert_summary,
                "recent_history": recent_alerts
            },
            "metrics": metrics_summary,
            "system_info": {
                "monitoring_services": {
                    "health_checks": True,
                    "metrics_collection": metrics.monitoring_active,
                    "alert_monitoring": alert_manager.monitoring_active
                }
            }
        }
        
        return dashboard_data
        
    except Exception as e:
        logger.error(f"Failed to get dashboard data: {e}")
        raise HTTPException(status_code=500, detail=str(e))