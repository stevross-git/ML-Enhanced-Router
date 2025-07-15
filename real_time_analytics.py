#!/usr/bin/env python3
"""
Real-time Analytics and Monitoring System
Provides comprehensive performance monitoring and predictive analytics
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import threading
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    RATE = "rate"


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class MetricPoint:
    """A single metric data point"""
    timestamp: datetime
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AlertRule:
    """Alert rule configuration"""
    name: str
    metric_name: str
    condition: str  # "gt", "lt", "eq", "ne", "gte", "lte"
    threshold: float
    duration: int  # seconds
    severity: AlertSeverity
    enabled: bool = True
    cooldown: int = 300  # seconds
    labels: Dict[str, str] = field(default_factory=dict)
    description: str = ""


@dataclass
class Alert:
    """Alert instance"""
    rule: AlertRule
    triggered_at: datetime
    value: float
    status: str = "active"  # active, resolved, suppressed
    message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class MetricStore:
    """In-memory metric storage with time-series capabilities"""
    
    def __init__(self, retention_hours: int = 24):
        self.retention_hours = retention_hours
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.aggregated_metrics: Dict[str, Dict] = defaultdict(dict)
        self.lock = threading.RLock()
        
    def add_metric(self, name: str, value: float, labels: Dict[str, str] = None, 
                  metadata: Dict[str, Any] = None):
        """Add a metric data point"""
        with self.lock:
            point = MetricPoint(
                timestamp=datetime.now(),
                value=value,
                labels=labels or {},
                metadata=metadata or {}
            )
            
            self.metrics[name].append(point)
            self._update_aggregations(name, point)
            
    def _update_aggregations(self, name: str, point: MetricPoint):
        """Update aggregated metrics"""
        now = datetime.now()
        
        # Initialize aggregations if not exists
        if name not in self.aggregated_metrics:
            self.aggregated_metrics[name] = {
                "1m": {"sum": 0, "count": 0, "min": float('inf'), "max": float('-inf')},
                "5m": {"sum": 0, "count": 0, "min": float('inf'), "max": float('-inf')},
                "15m": {"sum": 0, "count": 0, "min": float('inf'), "max": float('-inf')},
                "1h": {"sum": 0, "count": 0, "min": float('inf'), "max": float('-inf')}
            }
            
        # Update aggregations
        agg = self.aggregated_metrics[name]
        for period in ["1m", "5m", "15m", "1h"]:
            period_minutes = {"1m": 1, "5m": 5, "15m": 15, "1h": 60}[period]
            cutoff = now - timedelta(minutes=period_minutes)
            
            # Get relevant points
            relevant_points = [p for p in self.metrics[name] if p.timestamp >= cutoff]
            
            if relevant_points:
                values = [p.value for p in relevant_points]
                agg[period] = {
                    "sum": sum(values),
                    "count": len(values),
                    "min": min(values),
                    "max": max(values),
                    "avg": sum(values) / len(values)
                }
                
    def get_metric_history(self, name: str, duration: timedelta = None) -> List[MetricPoint]:
        """Get metric history"""
        with self.lock:
            if name not in self.metrics:
                return []
                
            if duration is None:
                return list(self.metrics[name])
                
            cutoff = datetime.now() - duration
            return [p for p in self.metrics[name] if p.timestamp >= cutoff]
            
    def get_aggregated_metrics(self, name: str) -> Dict[str, Dict]:
        """Get aggregated metrics"""
        with self.lock:
            return self.aggregated_metrics.get(name, {})
            
    def cleanup_old_metrics(self):
        """Clean up old metrics based on retention policy"""
        cutoff = datetime.now() - timedelta(hours=self.retention_hours)
        
        with self.lock:
            for name in self.metrics:
                # Remove old points
                while self.metrics[name] and self.metrics[name][0].timestamp < cutoff:
                    self.metrics[name].popleft()


class AlertManager:
    """Alert management system"""
    
    def __init__(self, metric_store: MetricStore):
        self.metric_store = metric_store
        self.rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=1000)
        self.notification_handlers: List[Callable] = []
        self.lock = threading.RLock()
        
    def add_rule(self, rule: AlertRule):
        """Add an alert rule"""
        with self.lock:
            self.rules[rule.name] = rule
            
    def remove_rule(self, rule_name: str):
        """Remove an alert rule"""
        with self.lock:
            if rule_name in self.rules:
                del self.rules[rule_name]
                
    def add_notification_handler(self, handler: Callable):
        """Add a notification handler"""
        self.notification_handlers.append(handler)
        
    def evaluate_rules(self):
        """Evaluate all alert rules"""
        with self.lock:
            for rule_name, rule in self.rules.items():
                if not rule.enabled:
                    continue
                    
                self._evaluate_rule(rule)
                
    def _evaluate_rule(self, rule: AlertRule):
        """Evaluate a single alert rule"""
        # Get recent metrics
        recent_metrics = self.metric_store.get_metric_history(
            rule.metric_name, 
            timedelta(seconds=rule.duration)
        )
        
        if not recent_metrics:
            return
            
        # Calculate current value
        current_value = recent_metrics[-1].value
        
        # Check condition
        triggered = self._check_condition(rule.condition, current_value, rule.threshold)
        
        alert_key = f"{rule.name}_{rule.metric_name}"
        
        if triggered:
            if alert_key not in self.active_alerts:
                # Create new alert
                alert = Alert(
                    rule=rule,
                    triggered_at=datetime.now(),
                    value=current_value,
                    message=f"Alert {rule.name}: {rule.metric_name} {rule.condition} {rule.threshold} (current: {current_value})"
                )
                
                self.active_alerts[alert_key] = alert
                self.alert_history.append(alert)
                
                # Notify handlers
                for handler in self.notification_handlers:
                    try:
                        handler(alert)
                    except Exception as e:
                        logger.error(f"Error in notification handler: {e}")
                        
        else:
            # Resolve alert if active
            if alert_key in self.active_alerts:
                alert = self.active_alerts[alert_key]
                alert.status = "resolved"
                del self.active_alerts[alert_key]
                
                # Notify handlers
                for handler in self.notification_handlers:
                    try:
                        handler(alert)
                    except Exception as e:
                        logger.error(f"Error in notification handler: {e}")
                        
    def _check_condition(self, condition: str, value: float, threshold: float) -> bool:
        """Check if condition is met"""
        conditions = {
            "gt": lambda v, t: v > t,
            "lt": lambda v, t: v < t,
            "eq": lambda v, t: v == t,
            "ne": lambda v, t: v != t,
            "gte": lambda v, t: v >= t,
            "lte": lambda v, t: v <= t
        }
        
        return conditions.get(condition, lambda v, t: False)(value, threshold)
        
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts"""
        with self.lock:
            return list(self.active_alerts.values())
            
    def get_alert_history(self) -> List[Alert]:
        """Get alert history"""
        with self.lock:
            return list(self.alert_history)


class PerformanceAnalyzer:
    """Advanced performance analysis and prediction"""
    
    def __init__(self, metric_store: MetricStore):
        self.metric_store = metric_store
        self.models: Dict[str, Any] = {}
        
    def analyze_trends(self, metric_name: str, window_hours: int = 2) -> Dict[str, Any]:
        """Analyze metric trends"""
        history = self.metric_store.get_metric_history(
            metric_name, 
            timedelta(hours=window_hours)
        )
        
        if len(history) < 2:
            return {"trend": "insufficient_data"}
            
        values = [p.value for p in history]
        timestamps = [p.timestamp.timestamp() for p in history]
        
        # Calculate trend
        if len(values) >= 2:
            slope = np.polyfit(timestamps, values, 1)[0]
            
            if slope > 0.1:
                trend = "increasing"
            elif slope < -0.1:
                trend = "decreasing"
            else:
                trend = "stable"
        else:
            trend = "unknown"
            
        # Calculate statistics
        stats = {
            "trend": trend,
            "slope": slope if len(values) >= 2 else 0,
            "current_value": values[-1],
            "min_value": min(values),
            "max_value": max(values),
            "avg_value": sum(values) / len(values),
            "std_dev": np.std(values) if len(values) > 1 else 0,
            "data_points": len(values)
        }
        
        return stats
        
    def predict_future_values(self, metric_name: str, hours_ahead: int = 1) -> Dict[str, Any]:
        """Predict future values using simple linear regression"""
        history = self.metric_store.get_metric_history(
            metric_name, 
            timedelta(hours=6)  # Use 6 hours of history
        )
        
        if len(history) < 10:
            return {"prediction": None, "error": "insufficient_data"}
            
        values = [p.value for p in history]
        timestamps = [p.timestamp.timestamp() for p in history]
        
        try:
            # Simple linear regression
            coeffs = np.polyfit(timestamps, values, 1)
            slope, intercept = coeffs
            
            # Predict future timestamp
            future_timestamp = datetime.now().timestamp() + (hours_ahead * 3600)
            predicted_value = slope * future_timestamp + intercept
            
            # Calculate confidence interval
            residuals = values - np.polyval(coeffs, timestamps)
            mse = np.mean(residuals**2)
            std_error = np.sqrt(mse)
            
            return {
                "prediction": predicted_value,
                "confidence_interval": {
                    "lower": predicted_value - 1.96 * std_error,
                    "upper": predicted_value + 1.96 * std_error
                },
                "r_squared": 1 - (np.sum(residuals**2) / np.sum((values - np.mean(values))**2)),
                "standard_error": std_error
            }
            
        except Exception as e:
            return {"prediction": None, "error": str(e)}
            
    def detect_anomalies(self, metric_name: str, sensitivity: float = 2.0) -> List[Dict[str, Any]]:
        """Detect anomalies in metric data"""
        history = self.metric_store.get_metric_history(
            metric_name, 
            timedelta(hours=4)
        )
        
        if len(history) < 10:
            return []
            
        values = [p.value for p in history]
        timestamps = [p.timestamp for p in history]
        
        # Calculate rolling statistics
        window_size = min(20, len(values) // 4)
        anomalies = []
        
        for i in range(window_size, len(values)):
            window = values[i-window_size:i]
            window_mean = np.mean(window)
            window_std = np.std(window)
            
            current_value = values[i]
            z_score = abs(current_value - window_mean) / window_std if window_std > 0 else 0
            
            if z_score > sensitivity:
                anomalies.append({
                    "timestamp": timestamps[i],
                    "value": current_value,
                    "z_score": z_score,
                    "expected_range": {
                        "min": window_mean - sensitivity * window_std,
                        "max": window_mean + sensitivity * window_std
                    }
                })
                
        return anomalies
        
    def generate_performance_report(self, metrics: List[str]) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        report = {
            "generated_at": datetime.now().isoformat(),
            "metrics_analyzed": len(metrics),
            "summary": {},
            "trends": {},
            "predictions": {},
            "anomalies": {},
            "recommendations": []
        }
        
        for metric in metrics:
            # Analyze trends
            trend_analysis = self.analyze_trends(metric)
            report["trends"][metric] = trend_analysis
            
            # Predict future values
            predictions = self.predict_future_values(metric)
            report["predictions"][metric] = predictions
            
            # Detect anomalies
            anomalies = self.detect_anomalies(metric)
            report["anomalies"][metric] = anomalies
            
            # Generate recommendations
            recommendations = self._generate_metric_recommendations(metric, trend_analysis, predictions, anomalies)
            report["recommendations"].extend(recommendations)
            
        # Generate summary
        report["summary"] = self._generate_summary(report)
        
        return report
        
    def _generate_metric_recommendations(self, metric: str, trend: Dict, prediction: Dict, 
                                       anomalies: List[Dict]) -> List[Dict[str, Any]]:
        """Generate recommendations for a metric"""
        recommendations = []
        
        # Trend-based recommendations
        if trend["trend"] == "increasing":
            if metric in ["response_time", "error_rate", "cpu_usage"]:
                recommendations.append({
                    "type": "warning",
                    "metric": metric,
                    "title": f"Increasing {metric}",
                    "description": f"{metric} is trending upward, consider investigating",
                    "priority": "medium"
                })
                
        elif trend["trend"] == "decreasing":
            if metric in ["throughput", "success_rate", "cache_hit_rate"]:
                recommendations.append({
                    "type": "warning",
                    "metric": metric,
                    "title": f"Decreasing {metric}",
                    "description": f"{metric} is trending downward, may need attention",
                    "priority": "medium"
                })
                
        # Prediction-based recommendations
        if prediction.get("prediction"):
            predicted_value = prediction["prediction"]
            
            if metric == "response_time" and predicted_value > 1000:  # 1 second
                recommendations.append({
                    "type": "alert",
                    "metric": metric,
                    "title": "High Response Time Predicted",
                    "description": f"Response time predicted to reach {predicted_value:.2f}ms",
                    "priority": "high"
                })
                
        # Anomaly-based recommendations
        if anomalies:
            recent_anomalies = [a for a in anomalies if (datetime.now() - a["timestamp"]).seconds < 3600]
            if recent_anomalies:
                recommendations.append({
                    "type": "alert",
                    "metric": metric,
                    "title": f"Anomalies Detected in {metric}",
                    "description": f"{len(recent_anomalies)} anomalies detected in the last hour",
                    "priority": "high"
                })
                
        return recommendations
        
    def _generate_summary(self, report: Dict) -> Dict[str, Any]:
        """Generate report summary"""
        total_anomalies = sum(len(anomalies) for anomalies in report["anomalies"].values())
        
        trending_up = sum(1 for trend in report["trends"].values() if trend["trend"] == "increasing")
        trending_down = sum(1 for trend in report["trends"].values() if trend["trend"] == "decreasing")
        
        high_priority_recommendations = sum(1 for rec in report["recommendations"] if rec["priority"] == "high")
        
        return {
            "total_anomalies": total_anomalies,
            "trending_up_count": trending_up,
            "trending_down_count": trending_down,
            "high_priority_recommendations": high_priority_recommendations,
            "overall_health": self._calculate_overall_health(report)
        }
        
    def _calculate_overall_health(self, report: Dict) -> str:
        """Calculate overall system health"""
        score = 100
        
        # Deduct points for issues
        score -= len(report["recommendations"]) * 2
        total_anomalies = sum(len(anomalies) for anomalies in report["anomalies"].values())
        score -= total_anomalies * 5
        
        # Determine health status
        if score >= 90:
            return "excellent"
        elif score >= 70:
            return "good"
        elif score >= 50:
            return "fair"
        else:
            return "poor"


class RealTimeAnalytics:
    """Main real-time analytics system"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.metric_store = MetricStore(retention_hours=self.config.get("retention_hours", 24))
        self.alert_manager = AlertManager(self.metric_store)
        self.performance_analyzer = PerformanceAnalyzer(self.metric_store)
        self.running = False
        self.background_tasks = []
        
        # Initialize default alert rules
        self._initialize_default_rules()
        
    def _initialize_default_rules(self):
        """Initialize default alert rules"""
        default_rules = [
            AlertRule(
                name="high_response_time",
                metric_name="response_time",
                condition="gt",
                threshold=1000,  # 1 second
                duration=60,
                severity=AlertSeverity.WARNING,
                description="Response time is too high"
            ),
            AlertRule(
                name="low_success_rate",
                metric_name="success_rate",
                condition="lt",
                threshold=0.95,
                duration=120,
                severity=AlertSeverity.ERROR,
                description="Success rate is below threshold"
            ),
            AlertRule(
                name="high_error_rate",
                metric_name="error_rate",
                condition="gt",
                threshold=0.05,
                duration=60,
                severity=AlertSeverity.ERROR,
                description="Error rate is too high"
            ),
            AlertRule(
                name="low_cache_hit_rate",
                metric_name="cache_hit_rate",
                condition="lt",
                threshold=0.8,
                duration=300,
                severity=AlertSeverity.WARNING,
                description="Cache hit rate is below optimal"
            )
        ]
        
        for rule in default_rules:
            self.alert_manager.add_rule(rule)
            
    async def start(self):
        """Start the analytics system"""
        self.running = True
        
        # Start background tasks
        self.background_tasks = [
            asyncio.create_task(self._metric_cleanup_loop()),
            asyncio.create_task(self._alert_evaluation_loop()),
            asyncio.create_task(self._performance_analysis_loop())
        ]
        
        logger.info("Real-time analytics system started")
        
    async def stop(self):
        """Stop the analytics system"""
        self.running = False
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
            
        logger.info("Real-time analytics system stopped")
        
    async def _metric_cleanup_loop(self):
        """Background task for metric cleanup"""
        while self.running:
            try:
                self.metric_store.cleanup_old_metrics()
                await asyncio.sleep(3600)  # Run every hour
            except Exception as e:
                logger.error(f"Error in metric cleanup: {e}")
                await asyncio.sleep(60)
                
    async def _alert_evaluation_loop(self):
        """Background task for alert evaluation"""
        while self.running:
            try:
                self.alert_manager.evaluate_rules()
                await asyncio.sleep(30)  # Run every 30 seconds
            except Exception as e:
                logger.error(f"Error in alert evaluation: {e}")
                await asyncio.sleep(30)
                
    async def _performance_analysis_loop(self):
        """Background task for performance analysis"""
        while self.running:
            try:
                # Generate periodic reports
                metrics = ["response_time", "throughput", "error_rate", "cache_hit_rate"]
                report = self.performance_analyzer.generate_performance_report(metrics)
                
                # Log significant findings
                if report["summary"]["high_priority_recommendations"] > 0:
                    logger.warning(f"Performance report: {report['summary']['high_priority_recommendations']} high priority recommendations")
                    
                await asyncio.sleep(1800)  # Run every 30 minutes
            except Exception as e:
                logger.error(f"Error in performance analysis: {e}")
                await asyncio.sleep(300)
                
    def record_metric(self, name: str, value: float, labels: Dict[str, str] = None):
        """Record a metric"""
        self.metric_store.add_metric(name, value, labels)
        
    def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get current real-time metrics"""
        metrics = {}
        
        # Get key metrics
        key_metrics = ["response_time", "throughput", "error_rate", "cache_hit_rate", 
                      "active_connections", "cpu_usage", "memory_usage"]
        
        for metric in key_metrics:
            aggregated = self.metric_store.get_aggregated_metrics(metric)
            if aggregated:
                metrics[metric] = aggregated.get("1m", {})
                
        return metrics
        
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data"""
        return {
            "real_time_metrics": self.get_real_time_metrics(),
            "active_alerts": [
                {
                    "name": alert.rule.name,
                    "severity": alert.rule.severity.value,
                    "message": alert.message,
                    "triggered_at": alert.triggered_at.isoformat(),
                    "value": alert.value
                }
                for alert in self.alert_manager.get_active_alerts()
            ],
            "system_health": self._get_system_health(),
            "performance_insights": self._get_performance_insights()
        }
        
    def _get_system_health(self) -> Dict[str, Any]:
        """Get overall system health"""
        active_alerts = self.alert_manager.get_active_alerts()
        critical_alerts = [a for a in active_alerts if a.rule.severity == AlertSeverity.CRITICAL]
        error_alerts = [a for a in active_alerts if a.rule.severity == AlertSeverity.ERROR]
        
        if critical_alerts:
            status = "critical"
        elif error_alerts:
            status = "degraded"
        elif active_alerts:
            status = "warning"
        else:
            status = "healthy"
            
        return {
            "status": status,
            "active_alerts": len(active_alerts),
            "critical_alerts": len(critical_alerts),
            "error_alerts": len(error_alerts)
        }
        
    def _get_performance_insights(self) -> List[Dict[str, Any]]:
        """Get performance insights"""
        insights = []
        
        # Analyze key metrics
        key_metrics = ["response_time", "throughput", "error_rate"]
        
        for metric in key_metrics:
            trend = self.performance_analyzer.analyze_trends(metric, window_hours=1)
            if trend["trend"] != "insufficient_data":
                insights.append({
                    "metric": metric,
                    "trend": trend["trend"],
                    "current_value": trend["current_value"],
                    "change": trend["slope"]
                })
                
        return insights
        
    def add_custom_alert_rule(self, rule: AlertRule):
        """Add a custom alert rule"""
        self.alert_manager.add_rule(rule)
        
    def get_analytics_stats(self) -> Dict[str, Any]:
        """Get analytics system statistics"""
        return {
            "metrics_stored": sum(len(metric_data) for metric_data in self.metric_store.metrics.values()),
            "active_alert_rules": len(self.alert_manager.rules),
            "active_alerts": len(self.alert_manager.active_alerts),
            "total_alerts_history": len(self.alert_manager.alert_history),
            "system_running": self.running
        }