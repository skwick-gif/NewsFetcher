"""
System Alerts and Monitoring Dashboard
Real-time monitoring with configurable alerts and notification triggers
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable, Set
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from collections import defaultdict, deque

from app.monitoring.health import health_monitor, HealthStatus, SystemHealthStatus
from app.monitoring.metrics import metrics
from app.monitoring.logging import get_logger, with_correlation
from app.notify.wecom import WeCom
from app.notify.emailer import Emailer
from app.notify.telegram import TelegramBot


class AlertSeverity(Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """Alert lifecycle status"""
    TRIGGERED = "triggered"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SILENCED = "silenced"


@dataclass
class AlertRule:
    """Alert rule configuration"""
    name: str
    description: str
    severity: AlertSeverity
    component: str
    condition: str
    threshold: float
    duration_seconds: int = 300  # Alert after 5 minutes
    cooldown_seconds: int = 1800  # Don't re-alert for 30 minutes
    enabled: bool = True
    notification_channels: List[str] = None  # ['wecom', 'email', 'telegram']
    
    def __post_init__(self):
        if self.notification_channels is None:
            self.notification_channels = ['wecom', 'email']


@dataclass
class Alert:
    """Active alert instance"""
    id: str
    rule: AlertRule
    triggered_at: datetime
    status: AlertStatus
    current_value: float
    message: str
    context: Dict[str, Any] = None
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    resolved_at: Optional[datetime] = None
    last_notification_sent: Optional[datetime] = None
    notification_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = asdict(self)
        result['rule'] = asdict(self.rule)
        result['severity'] = self.rule.severity.value
        result['status'] = self.status.value
        result['triggered_at'] = self.triggered_at.isoformat()
        
        if self.acknowledged_at:
            result['acknowledged_at'] = self.acknowledged_at.isoformat()
        if self.resolved_at:
            result['resolved_at'] = self.resolved_at.isoformat()
        if self.last_notification_sent:
            result['last_notification_sent'] = self.last_notification_sent.isoformat()
            
        return result


class MetricCondition:
    """Evaluates metric-based conditions"""
    
    @staticmethod
    def evaluate_condition(condition: str, current_value: float, threshold: float) -> bool:
        """Evaluate alert condition"""
        conditions = {
            'greater_than': lambda v, t: v > t,
            'less_than': lambda v, t: v < t,
            'equals': lambda v, t: abs(v - t) < 0.001,
            'not_equals': lambda v, t: abs(v - t) >= 0.001,
            'greater_equal': lambda v, t: v >= t,
            'less_equal': lambda v, t: v <= t
        }
        
        return conditions.get(condition, lambda v, t: False)(current_value, threshold)


class AlertManager:
    """Alert management system"""
    
    def __init__(self):
        self.logger = get_logger('app.alerts')
        self.rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=10000)
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # Notification clients
        self.wecom = None
        self.emailer = None
        self.telegram = None
        
        # Metric tracking for alert conditions
        self.metric_buffer: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Initialize default rules
        self._initialize_default_rules()
        
        # Initialize notification clients
        self._initialize_notifications()
    
    def _initialize_default_rules(self):
        """Initialize default alert rules"""
        default_rules = [
            AlertRule(
                name="high_cpu_usage",
                description="CPU usage above 90% for extended period",
                severity=AlertSeverity.HIGH,
                component="system_resources",
                condition="greater_than",
                threshold=90.0,
                duration_seconds=300,
                notification_channels=['wecom', 'email']
            ),
            AlertRule(
                name="high_memory_usage", 
                description="Memory usage above 95% for extended period",
                severity=AlertSeverity.CRITICAL,
                component="system_resources",
                condition="greater_than",
                threshold=95.0,
                duration_seconds=180,
                notification_channels=['wecom', 'email', 'telegram']
            ),
            AlertRule(
                name="database_unhealthy",
                description="Database health check failing",
                severity=AlertSeverity.CRITICAL,
                component="database",
                condition="equals",
                threshold=0.0,  # 0 = unhealthy, 1 = healthy
                duration_seconds=60,
                notification_channels=['wecom', 'email', 'telegram']
            ),
            AlertRule(
                name="redis_unhealthy",
                description="Redis health check failing", 
                severity=AlertSeverity.HIGH,
                component="redis",
                condition="equals",
                threshold=0.0,
                duration_seconds=120,
                notification_channels=['wecom', 'email']
            ),
            AlertRule(
                name="high_error_rate",
                description="HTTP error rate above 5%",
                severity=AlertSeverity.MEDIUM,
                component="http_errors",
                condition="greater_than",
                threshold=5.0,
                duration_seconds=600,
                notification_channels=['wecom']
            ),
            AlertRule(
                name="slow_response_time",
                description="Average response time above 2 seconds",
                severity=AlertSeverity.MEDIUM,
                component="response_time",
                condition="greater_than",
                threshold=2000.0,  # milliseconds
                duration_seconds=600,
                notification_channels=['wecom']
            ),
            AlertRule(
                name="celery_queue_backup",
                description="Celery queue length above 100 tasks",
                severity=AlertSeverity.MEDIUM,
                component="celery_queue",
                condition="greater_than",
                threshold=100.0,
                duration_seconds=300,
                notification_channels=['wecom', 'email']
            ),
            AlertRule(
                name="disk_space_low",
                description="Disk space usage above 85%",
                severity=AlertSeverity.HIGH,
                component="disk_usage",
                condition="greater_than",
                threshold=85.0,
                duration_seconds=600,
                notification_channels=['wecom', 'email']
            )
        ]
        
        for rule in default_rules:
            self.rules[rule.name] = rule
    
    def _initialize_notifications(self):
        """Initialize notification clients"""
        try:
            self.wecom = WeCom()
        except Exception as e:
            self.logger.warning(f"Failed to initialize WeCom notifications: {e}")
        
        try:
            self.emailer = Emailer()
        except Exception as e:
            self.logger.warning(f"Failed to initialize email notifications: {e}")
        
        try:
            self.telegram = TelegramBot()
        except Exception as e:
            self.logger.warning(f"Failed to initialize Telegram notifications: {e}")
    
    def add_rule(self, rule: AlertRule):
        """Add or update alert rule"""
        self.rules[rule.name] = rule
        self.logger.info(f"Added alert rule: {rule.name}")
    
    def remove_rule(self, rule_name: str):
        """Remove alert rule"""
        if rule_name in self.rules:
            del self.rules[rule_name]
            self.logger.info(f"Removed alert rule: {rule_name}")
    
    def get_rule(self, rule_name: str) -> Optional[AlertRule]:
        """Get alert rule by name"""
        return self.rules.get(rule_name)
    
    def list_rules(self) -> List[AlertRule]:
        """List all alert rules"""
        return list(self.rules.values())
    
    def start_monitoring(self, interval: int = 60):
        """Start alert monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_alerts,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        self.logger.info(f"Started alert monitoring with {interval}s interval")
    
    def stop_monitoring(self):
        """Stop alert monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=10)
        self.logger.info("Stopped alert monitoring")
    
    def _monitor_alerts(self, interval: int):
        """Background alert monitoring thread"""
        while self.monitoring_active:
            try:
                asyncio.run(self._check_all_alerts())
            except Exception as e:
                self.logger.error(f"Error in alert monitoring: {e}", exc_info=True)
            
            time.sleep(interval)
    
    async def _check_all_alerts(self):
        """Check all alert conditions"""
        with with_correlation(f"alert_check_{int(time.time())}"):
            # Get current system health
            health_status = await health_monitor.check_health()
            
            # Get current metrics
            current_metrics = self._extract_metrics(health_status)
            
            # Check each rule
            for rule_name, rule in self.rules.items():
                if not rule.enabled:
                    continue
                
                try:
                    await self._check_rule(rule, current_metrics, health_status)
                except Exception as e:
                    self.logger.error(f"Error checking rule {rule_name}: {e}", exc_info=True)
    
    def _extract_metrics(self, health_status: SystemHealthStatus) -> Dict[str, float]:
        """Extract metrics from health status for alert evaluation"""
        metrics_data = {}
        
        for check in health_status.checks:
            component = check.component
            
            # Health status as numeric (1=healthy, 0.5=degraded, 0=unhealthy)
            health_value = {
                HealthStatus.HEALTHY: 1.0,
                HealthStatus.DEGRADED: 0.5,
                HealthStatus.UNHEALTHY: 0.0,
                HealthStatus.UNKNOWN: 0.0
            }.get(check.status, 0.0)
            
            metrics_data[component] = health_value
            
            # Extract specific metrics from details
            if check.details:
                if component == "system_resources":
                    details = check.details
                    if 'cpu' in details:
                        metrics_data['cpu_usage'] = details['cpu']['usage_percent']
                    if 'memory' in details:
                        metrics_data['memory_usage'] = details['memory']['usage_percent']
                    if 'disk' in details:
                        metrics_data['disk_usage'] = details['disk']['usage_percent']
                
                elif component == "celery":
                    if 'active_tasks' in check.details:
                        metrics_data['celery_queue'] = check.details['active_tasks']
            
            # Response time metric
            metrics_data[f'{component}_response_time'] = check.response_time_ms
        
        return metrics_data
    
    async def _check_rule(self, rule: AlertRule, current_metrics: Dict[str, float], health_status: SystemHealthStatus):
        """Check individual alert rule"""
        # Get current value for the metric
        metric_key = self._get_metric_key(rule.component, rule.condition)
        current_value = current_metrics.get(metric_key, 0.0)
        
        # Store metric in buffer for duration-based checking
        self.metric_buffer[rule.name].append({
            'timestamp': datetime.utcnow(),
            'value': current_value
        })
        
        # Check if condition is met
        condition_met = MetricCondition.evaluate_condition(
            rule.condition, current_value, rule.threshold
        )
        
        alert_id = f"{rule.name}_{rule.component}"
        
        if condition_met:
            # Check if condition has been met for required duration
            if self._condition_duration_exceeded(rule):
                if alert_id not in self.active_alerts:
                    # Trigger new alert
                    await self._trigger_alert(rule, current_value, current_metrics)
                else:
                    # Update existing alert
                    alert = self.active_alerts[alert_id]
                    alert.current_value = current_value
                    
                    # Check if we need to send notification reminder
                    await self._check_notification_reminder(alert)
        else:
            # Condition not met, resolve any active alert
            if alert_id in self.active_alerts:
                await self._resolve_alert(alert_id)
    
    def _get_metric_key(self, component: str, condition: str) -> str:
        """Get metric key based on component and condition"""
        # Map component names to metric keys
        component_mapping = {
            'system_resources': 'cpu_usage',  # Default to CPU for system resources
            'database': 'database',
            'redis': 'redis',
            'celery_queue': 'celery_queue',
            'disk_usage': 'disk_usage'
        }
        
        return component_mapping.get(component, component)
    
    def _condition_duration_exceeded(self, rule: AlertRule) -> bool:
        """Check if alert condition has been met for required duration"""
        buffer = self.metric_buffer[rule.name]
        
        if len(buffer) < 2:
            return False
        
        # Check if condition has been consistently met for duration
        now = datetime.utcnow()
        duration_threshold = now - timedelta(seconds=rule.duration_seconds)
        
        relevant_metrics = [
            m for m in buffer 
            if m['timestamp'] >= duration_threshold
        ]
        
        if not relevant_metrics:
            return False
        
        # Check if condition was met for all relevant metrics
        for metric in relevant_metrics:
            if not MetricCondition.evaluate_condition(
                rule.condition, metric['value'], rule.threshold
            ):
                return False
        
        return True
    
    async def _trigger_alert(self, rule: AlertRule, current_value: float, context: Dict[str, Any]):
        """Trigger a new alert"""
        alert_id = f"{rule.name}_{rule.component}"
        
        # Check cooldown period
        if self._in_cooldown(rule):
            return
        
        alert = Alert(
            id=alert_id,
            rule=rule,
            triggered_at=datetime.utcnow(),
            status=AlertStatus.TRIGGERED,
            current_value=current_value,
            message=self._format_alert_message(rule, current_value),
            context=context
        )
        
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert.to_dict())
        
        self.logger.warning(
            f"Alert triggered: {rule.name}",
            extra={
                'alert_id': alert_id,
                'rule': rule.name,
                'severity': rule.severity.value,
                'component': rule.component,
                'current_value': current_value,
                'threshold': rule.threshold,
                'alert_event': True
            }
        )
        
        # Send notifications
        await self._send_alert_notifications(alert)
    
    async def _resolve_alert(self, alert_id: str):
        """Resolve an active alert"""
        if alert_id not in self.active_alerts:
            return
        
        alert = self.active_alerts[alert_id]
        alert.status = AlertStatus.RESOLVED
        alert.resolved_at = datetime.utcnow()
        
        self.logger.info(
            f"Alert resolved: {alert.rule.name}",
            extra={
                'alert_id': alert_id,
                'rule': alert.rule.name,
                'duration_seconds': (alert.resolved_at - alert.triggered_at).total_seconds(),
                'alert_event': True
            }
        )
        
        # Send resolution notification
        await self._send_resolution_notifications(alert)
        
        # Move to history and remove from active
        self.alert_history.append(alert.to_dict())
        del self.active_alerts[alert_id]
    
    def _in_cooldown(self, rule: AlertRule) -> bool:
        """Check if rule is in cooldown period"""
        # Look for recent alerts in history
        cooldown_threshold = datetime.utcnow() - timedelta(seconds=rule.cooldown_seconds)
        
        for alert_data in reversed(self.alert_history):
            if alert_data['rule']['name'] == rule.name:
                triggered_at = datetime.fromisoformat(alert_data['triggered_at'])
                if triggered_at >= cooldown_threshold:
                    return True
                break  # Only check most recent
        
        return False
    
    async def _check_notification_reminder(self, alert: Alert):
        """Check if we should send notification reminder"""
        if not alert.last_notification_sent:
            return
        
        # Send reminder every hour for critical alerts
        reminder_interval = timedelta(hours=1) if alert.rule.severity == AlertSeverity.CRITICAL else timedelta(hours=4)
        
        if datetime.utcnow() - alert.last_notification_sent >= reminder_interval:
            alert.message = f"REMINDER: {alert.message}"
            await self._send_alert_notifications(alert, is_reminder=True)
    
    def _format_alert_message(self, rule: AlertRule, current_value: float) -> str:
        """Format alert message"""
        return (
            f"🚨 {rule.severity.value.upper()} ALERT: {rule.description}\n"
            f"Component: {rule.component}\n"
            f"Current value: {current_value:.2f}\n"
            f"Threshold: {rule.threshold:.2f}\n"
            f"Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC"
        )
    
    async def _send_alert_notifications(self, alert: Alert, is_reminder: bool = False):
        """Send alert notifications to configured channels"""
        message = alert.message
        if is_reminder:
            message = f"⏰ REMINDER: {message}"
        
        for channel in alert.rule.notification_channels:
            try:
                if channel == 'wecom' and self.wecom:
                    await self.wecom.send_alert(
                        message=message,
                        severity=alert.rule.severity.value,
                        component=alert.rule.component
                    )
                
                elif channel == 'email' and self.emailer:
                    await self.emailer.send_alert_email(
                        subject=f"[{alert.rule.severity.value.upper()}] {alert.rule.name}",
                        message=message,
                        alert_data=alert.to_dict()
                    )
                
                elif channel == 'telegram' and self.telegram:
                    await self.telegram.send_alert(
                        message=message,
                        severity=alert.rule.severity.value
                    )
                
                self.logger.info(f"Sent alert notification via {channel}: {alert.rule.name}")
                
            except Exception as e:
                self.logger.error(f"Failed to send alert via {channel}: {e}", exc_info=True)
        
        alert.last_notification_sent = datetime.utcnow()
        alert.notification_count += 1
    
    async def _send_resolution_notifications(self, alert: Alert):
        """Send alert resolution notifications"""
        duration = alert.resolved_at - alert.triggered_at
        message = (
            f"✅ RESOLVED: {alert.rule.description}\n"
            f"Component: {alert.rule.component}\n"
            f"Duration: {duration}\n"
            f"Resolved at: {alert.resolved_at.strftime('%Y-%m-%d %H:%M:%S')} UTC"
        )
        
        for channel in alert.rule.notification_channels:
            try:
                if channel == 'wecom' and self.wecom:
                    await self.wecom.send_message(message)
                
                elif channel == 'email' and self.emailer:
                    await self.emailer.send_email(
                        subject=f"[RESOLVED] {alert.rule.name}",
                        message=message
                    )
                
                elif channel == 'telegram' and self.telegram:
                    await self.telegram.send_message(message)
                    
            except Exception as e:
                self.logger.error(f"Failed to send resolution notification via {channel}: {e}")
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str = "system") -> bool:
        """Acknowledge an active alert"""
        if alert_id not in self.active_alerts:
            return False
        
        alert = self.active_alerts[alert_id]
        alert.status = AlertStatus.ACKNOWLEDGED
        alert.acknowledged_at = datetime.utcnow()
        alert.acknowledged_by = acknowledged_by
        
        self.logger.info(
            f"Alert acknowledged: {alert.rule.name}",
            extra={
                'alert_id': alert_id,
                'acknowledged_by': acknowledged_by,
                'alert_event': True
            }
        )
        
        return True
    
    def silence_alert(self, alert_id: str, duration_hours: int = 24) -> bool:
        """Silence an active alert"""
        if alert_id not in self.active_alerts:
            return False
        
        alert = self.active_alerts[alert_id]
        alert.status = AlertStatus.SILENCED
        
        # Add silence duration to context
        alert.context = alert.context or {}
        alert.context['silenced_until'] = (
            datetime.utcnow() + timedelta(hours=duration_hours)
        ).isoformat()
        
        self.logger.info(
            f"Alert silenced: {alert.rule.name} for {duration_hours} hours",
            extra={
                'alert_id': alert_id,
                'silence_duration_hours': duration_hours,
                'alert_event': True
            }
        )
        
        return True
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts"""
        return list(self.active_alerts.values())
    
    def get_alert_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get alert history"""
        return list(self.alert_history)[-limit:]
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get alert system summary"""
        active_by_severity = defaultdict(int)
        for alert in self.active_alerts.values():
            active_by_severity[alert.rule.severity.value] += 1
        
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'active_alerts_count': len(self.active_alerts),
            'active_by_severity': dict(active_by_severity),
            'total_rules': len(self.rules),
            'enabled_rules': len([r for r in self.rules.values() if r.enabled]),
            'monitoring_active': self.monitoring_active,
            'total_alerts_triggered': len(self.alert_history)
        }


# Global alert manager instance
alert_manager = AlertManager()