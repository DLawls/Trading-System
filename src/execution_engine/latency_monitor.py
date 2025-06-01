"""
Latency Monitor - Real-time performance tracking for execution engine

Tracks critical timing metrics:
- Signal to order creation time
- Order submission latency
- Broker response time  
- Fill notification latency
- End-to-end execution time

Designed for ultra-low latency optimization and C++ portability.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
import threading
from collections import defaultdict, deque
import statistics


@dataclass
class LatencyMetrics:
    """Container for latency measurements"""
    
    # Core timing metrics (all in milliseconds)
    signal_to_order: float = 0.0          # Signal processing to order creation
    order_creation: float = 0.0           # Order object creation time
    order_validation: float = 0.0         # Order validation time
    order_submission: float = 0.0         # Submission to broker
    broker_response: float = 0.0          # Broker acknowledgment time
    fill_notification: float = 0.0        # Fill event processing
    end_to_end: float = 0.0               # Total signal to fill time
    
    # Network latency
    network_rtt: float = 0.0              # Round-trip time to broker
    api_call_latency: float = 0.0         # API call overhead
    
    # System performance
    cpu_time: float = 0.0                 # CPU processing time
    memory_allocations: int = 0           # Memory allocation count
    
    # Queue metrics
    queue_wait_time: float = 0.0          # Time waiting in queues
    queue_depth: int = 0                  # Queue depth at submission
    
    # Market data latency
    market_data_lag: float = 0.0          # Market data freshness
    price_staleness: float = 0.0          # Price quote age
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.utcnow)
    order_id: str = ""
    symbol: str = ""
    venue: str = ""
    
    def total_latency(self) -> float:
        """Calculate total latency excluding fill notification"""
        return (self.signal_to_order + self.order_creation + 
                self.order_validation + self.order_submission + 
                self.broker_response)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization"""
        return {
            'signal_to_order': self.signal_to_order,
            'order_creation': self.order_creation,
            'order_validation': self.order_validation,
            'order_submission': self.order_submission,
            'broker_response': self.broker_response,
            'fill_notification': self.fill_notification,
            'end_to_end': self.end_to_end,
            'network_rtt': self.network_rtt,
            'api_call_latency': self.api_call_latency,
            'cpu_time': self.cpu_time,
            'memory_allocations': self.memory_allocations,
            'queue_wait_time': self.queue_wait_time,
            'queue_depth': self.queue_depth,
            'market_data_lag': self.market_data_lag,
            'price_staleness': self.price_staleness,
            'total_latency': self.total_latency(),
            'timestamp': self.timestamp.isoformat(),
            'order_id': self.order_id,
            'symbol': self.symbol,
            'venue': self.venue
        }


@dataclass
class LatencyStats:
    """Statistical summary of latency metrics"""
    
    metric_name: str
    count: int = 0
    mean: float = 0.0
    median: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    min_value: float = float('inf')
    max_value: float = 0.0
    std_dev: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'metric_name': self.metric_name,
            'count': self.count,
            'mean': self.mean,
            'median': self.median,
            'p95': self.p95,
            'p99': self.p99,
            'min': self.min_value,
            'max': self.max_value,
            'std_dev': self.std_dev
        }


class LatencyTimer:
    """
    High-precision timer for measuring latency
    
    Uses time.perf_counter() for maximum precision.
    Designed to be lightweight and fast.
    """
    
    def __init__(self, name: str = ""):
        self.name = name
        self.start_time = 0.0
        self.end_time = 0.0
        self.is_running = False
    
    def start(self) -> 'LatencyTimer':
        """Start the timer"""
        self.start_time = time.perf_counter()
        self.is_running = True
        return self
    
    def stop(self) -> float:
        """Stop the timer and return elapsed milliseconds"""
        if not self.is_running:
            return 0.0
            
        self.end_time = time.perf_counter()
        self.is_running = False
        
        # Return elapsed time in milliseconds
        return (self.end_time - self.start_time) * 1000.0
    
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds (timer can still be running)"""
        if not self.is_running:
            return (self.end_time - self.start_time) * 1000.0
        else:
            return (time.perf_counter() - self.start_time) * 1000.0
    
    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()


class LatencyMonitor:
    """
    Real-time latency monitoring system
    
    Thread-safe, high-performance latency tracking.
    Maintains rolling statistics and percentiles.
    """
    
    def __init__(self, history_size: int = 10000):
        """
        Initialize latency monitor
        
        Args:
            history_size: Number of recent measurements to keep
        """
        self.history_size = history_size
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Active timers for tracking ongoing operations
        self._active_timers: Dict[str, Dict[str, LatencyTimer]] = defaultdict(dict)
        
        # Raw measurements storage
        self._measurements: Dict[str, deque] = defaultdict(lambda: deque(maxlen=history_size))
        
        # Order-specific metrics
        self._order_metrics: Dict[str, LatencyMetrics] = {}
        
        # Running statistics
        self._stats: Dict[str, LatencyStats] = {}
        
        # System-wide metrics
        self._total_orders = 0
        self._total_latency = 0.0
        self._start_time = datetime.utcnow()
        
        # Performance alerts
        self._alert_thresholds = {
            'signal_to_order': 5.0,      # 5ms alert threshold
            'order_submission': 10.0,    # 10ms alert threshold
            'broker_response': 50.0,     # 50ms alert threshold
            'end_to_end': 100.0         # 100ms alert threshold
        }
        
        self._alerts_triggered = []
    
    def start_timer(self, order_id: str, metric_name: str) -> LatencyTimer:
        """
        Start a latency timer for specific order and metric
        
        Args:
            order_id: Unique order identifier
            metric_name: Name of the metric being measured
            
        Returns:
            LatencyTimer instance
        """
        with self._lock:
            timer = LatencyTimer(f"{order_id}_{metric_name}")
            timer.start()
            self._active_timers[order_id][metric_name] = timer
            return timer
    
    def stop_timer(self, order_id: str, metric_name: str) -> float:
        """
        Stop a latency timer and record the measurement
        
        Args:
            order_id: Unique order identifier
            metric_name: Name of the metric being measured
            
        Returns:
            Elapsed time in milliseconds
        """
        with self._lock:
            if order_id in self._active_timers and metric_name in self._active_timers[order_id]:
                timer = self._active_timers[order_id][metric_name]
                elapsed_ms = timer.stop()
                
                # Record measurement
                self._record_measurement(order_id, metric_name, elapsed_ms)
                
                # Clean up timer
                del self._active_timers[order_id][metric_name]
                if not self._active_timers[order_id]:
                    del self._active_timers[order_id]
                
                return elapsed_ms
            
            return 0.0
    
    def record_latency(self, order_id: str, metric_name: str, latency_ms: float) -> None:
        """
        Directly record a latency measurement
        
        Args:
            order_id: Unique order identifier
            metric_name: Name of the metric
            latency_ms: Latency in milliseconds
        """
        with self._lock:
            self._record_measurement(order_id, metric_name, latency_ms)
    
    def _record_measurement(self, order_id: str, metric_name: str, latency_ms: float) -> None:
        """Internal method to record a measurement"""
        
        # Store raw measurement
        self._measurements[metric_name].append(latency_ms)
        
        # Update order-specific metrics
        if order_id not in self._order_metrics:
            self._order_metrics[order_id] = LatencyMetrics(order_id=order_id)
        
        # Update the specific metric
        if hasattr(self._order_metrics[order_id], metric_name):
            setattr(self._order_metrics[order_id], metric_name, latency_ms)
        
        # Update running statistics
        self._update_stats(metric_name)
        
        # Check for alerts
        self._check_alerts(metric_name, latency_ms)
        
        # Update totals
        self._total_latency += latency_ms
        if metric_name == 'end_to_end':
            self._total_orders += 1
    
    def _update_stats(self, metric_name: str) -> None:
        """Update running statistics for a metric"""
        
        measurements = list(self._measurements[metric_name])
        if not measurements:
            return
        
        # Calculate statistics
        count = len(measurements)
        mean_val = statistics.mean(measurements)
        median_val = statistics.median(measurements)
        min_val = min(measurements)
        max_val = max(measurements)
        
        # Calculate percentiles
        sorted_measurements = sorted(measurements)
        p95_idx = int(0.95 * count)
        p99_idx = int(0.99 * count)
        p95_val = sorted_measurements[min(p95_idx, count - 1)]
        p99_val = sorted_measurements[min(p99_idx, count - 1)]
        
        # Calculate standard deviation
        std_dev = statistics.stdev(measurements) if count > 1 else 0.0
        
        # Update stats
        self._stats[metric_name] = LatencyStats(
            metric_name=metric_name,
            count=count,
            mean=mean_val,
            median=median_val,
            p95=p95_val,
            p99=p99_val,
            min_value=min_val,
            max_value=max_val,
            std_dev=std_dev
        )
    
    def _check_alerts(self, metric_name: str, latency_ms: float) -> None:
        """Check if latency exceeds alert thresholds"""
        
        threshold = self._alert_thresholds.get(metric_name)
        if threshold and latency_ms > threshold:
            alert = {
                'timestamp': datetime.utcnow(),
                'metric': metric_name,
                'value': latency_ms,
                'threshold': threshold,
                'severity': 'HIGH' if latency_ms > threshold * 2 else 'MEDIUM'
            }
            self._alerts_triggered.append(alert)
            
            # Keep only recent alerts (last 100)
            if len(self._alerts_triggered) > 100:
                self._alerts_triggered = self._alerts_triggered[-100:]
    
    def get_order_metrics(self, order_id: str) -> Optional[LatencyMetrics]:
        """Get latency metrics for a specific order"""
        with self._lock:
            return self._order_metrics.get(order_id)
    
    def get_stats(self, metric_name: str = None) -> Dict[str, LatencyStats]:
        """
        Get latency statistics
        
        Args:
            metric_name: Specific metric name, or None for all metrics
            
        Returns:
            Dictionary of statistics
        """
        with self._lock:
            if metric_name:
                return {metric_name: self._stats.get(metric_name)}
            return self._stats.copy()
    
    def get_recent_alerts(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent latency alerts"""
        with self._lock:
            return self._alerts_triggered[-count:] if self._alerts_triggered else []
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive latency summary"""
        with self._lock:
            uptime = (datetime.utcnow() - self._start_time).total_seconds()
            
            summary = {
                'uptime_seconds': uptime,
                'total_orders': self._total_orders,
                'average_latency': self._total_latency / max(self._total_orders, 1),
                'orders_per_second': self._total_orders / max(uptime, 1),
                'active_timers': len(self._active_timers),
                'metrics_tracked': len(self._measurements),
                'recent_alerts': len([a for a in self._alerts_triggered 
                                    if (datetime.utcnow() - a['timestamp']).total_seconds() < 300])
            }
            
            # Add per-metric summaries
            summary['metrics'] = {}
            for metric_name, stats in self._stats.items():
                summary['metrics'][metric_name] = stats.to_dict()
            
            return summary
    
    def get_percentile_breakdown(self, metric_name: str) -> Dict[str, float]:
        """Get detailed percentile breakdown for a metric"""
        with self._lock:
            measurements = list(self._measurements.get(metric_name, []))
            if not measurements:
                return {}
            
            sorted_measurements = sorted(measurements)
            count = len(sorted_measurements)
            
            percentiles = [50, 75, 90, 95, 99, 99.9]
            breakdown = {}
            
            for p in percentiles:
                idx = int((p / 100.0) * count)
                idx = min(idx, count - 1)
                breakdown[f'p{p}'] = sorted_measurements[idx]
            
            return breakdown
    
    def reset_stats(self) -> None:
        """Reset all statistics and measurements"""
        with self._lock:
            self._measurements.clear()
            self._order_metrics.clear()
            self._stats.clear()
            self._alerts_triggered.clear()
            self._total_orders = 0
            self._total_latency = 0.0
            self._start_time = datetime.utcnow()
    
    def create_timer(self, name: str = "") -> LatencyTimer:
        """Create a standalone timer (not tied to an order)"""
        return LatencyTimer(name)
    
    def set_alert_threshold(self, metric_name: str, threshold_ms: float) -> None:
        """Set custom alert threshold for a metric"""
        with self._lock:
            self._alert_thresholds[metric_name] = threshold_ms
    
    def export_measurements(self, metric_name: str = None) -> Dict[str, List[float]]:
        """Export raw measurements for analysis"""
        with self._lock:
            if metric_name:
                return {metric_name: list(self._measurements.get(metric_name, []))}
            
            return {name: list(measurements) for name, measurements in self._measurements.items()}


# Global latency monitor instance
_global_monitor = None
_monitor_lock = threading.Lock()


def get_latency_monitor() -> LatencyMonitor:
    """Get the global latency monitor instance (singleton)"""
    global _global_monitor
    
    if _global_monitor is None:
        with _monitor_lock:
            if _global_monitor is None:
                _global_monitor = LatencyMonitor()
    
    return _global_monitor


# Convenience functions for common usage patterns
def start_timer(order_id: str, metric_name: str) -> LatencyTimer:
    """Start a latency timer"""
    return get_latency_monitor().start_timer(order_id, metric_name)


def stop_timer(order_id: str, metric_name: str) -> float:
    """Stop a latency timer"""
    return get_latency_monitor().stop_timer(order_id, metric_name)


def record_latency(order_id: str, metric_name: str, latency_ms: float) -> None:
    """Record a latency measurement"""
    get_latency_monitor().record_latency(order_id, metric_name, latency_ms)


def create_timer(name: str = "") -> LatencyTimer:
    """Create a standalone timer"""
    return get_latency_monitor().create_timer(name) 