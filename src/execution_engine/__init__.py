"""
Execution Engine Module

High-performance execution engine for converting trading signals into market orders.
Designed for ultra-low latency with clean architecture for easy C++ porting.

Core Components:
- BrokerAdapter: Interface to broker APIs (Alpaca)
- OrderManager: Order lifecycle management  
- Router: Order routing and execution logic
- LatencyMonitor: Real-time latency tracking
- ExecutionEngine: Main orchestration engine
"""

from .order_schemas import Order, OrderStatus, OrderType, TimeInForce, OrderSide
from .broker_adapter import BrokerAdapter, AlpacaBrokerAdapter
from .order_manager import OrderManager, OrderManagerConfig
from .execution_router import ExecutionRouter, RoutingConfig
from .latency_monitor import LatencyMonitor, LatencyMetrics
from .execution_engine import ExecutionEngine, ExecutionConfig

__all__ = [
    # Order schemas
    'Order',
    'OrderStatus', 
    'OrderType',
    'TimeInForce',
    'OrderSide',
    
    # Core components
    'BrokerAdapter',
    'AlpacaBrokerAdapter',
    'OrderManager',
    'OrderManagerConfig',
    'ExecutionRouter',
    'RoutingConfig',
    'LatencyMonitor',
    'LatencyMetrics',
    'ExecutionEngine',
    'ExecutionConfig'
] 