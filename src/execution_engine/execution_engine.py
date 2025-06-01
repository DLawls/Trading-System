"""
Execution Engine - Main orchestration engine for order execution

The central hub that coordinates:
- Signal processing and order creation
- Order management and lifecycle tracking
- Broker communication and execution
- Smart routing and algorithmic execution
- Real-time performance monitoring
- Risk management and controls

This is the primary interface for converting trading signals into market executions.
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime
import json

from .order_schemas import Order, OrderStatus
from .broker_adapter import BrokerAdapter, AlpacaBrokerAdapter, BrokerConfig
from .order_manager import OrderManager, OrderManagerConfig
from .execution_router import ExecutionRouter, RoutingConfig
from .latency_monitor import get_latency_monitor, LatencyMonitor
from ..signal_generation.signal_schema import TradingSignal


@dataclass
class ExecutionConfig:
    """Configuration for the Execution Engine"""
    
    # Component configurations
    broker_config: BrokerConfig
    order_manager_config: OrderManagerConfig
    routing_config: RoutingConfig
    
    # Engine settings
    enable_execution: bool = True           # Master switch for execution
    dry_run_mode: bool = True              # Paper trading mode
    max_concurrent_signals: int = 10       # Max signals processing simultaneously
    
    # Performance monitoring
    enable_latency_monitoring: bool = True
    latency_alert_threshold_ms: float = 100.0
    
    # Logging and analytics
    log_level: str = "INFO"
    enable_trade_logging: bool = True
    enable_performance_analytics: bool = True
    
    # Safety controls
    daily_loss_limit: float = 10000.0      # Daily loss limit in USD
    max_portfolio_value: float = 100000.0  # Maximum portfolio value
    circuit_breaker_enabled: bool = True   # Emergency stop on errors


class ExecutionEngine:
    """
    Main execution engine orchestrating all trading operations
    
    This is the primary interface between trading signals and market execution.
    Coordinates all execution components with comprehensive monitoring and controls.
    """
    
    def __init__(self, config: ExecutionConfig):
        self.config = config
        self.logger = logging.getLogger("ExecutionEngine")
        
        # Configure logging
        logging.basicConfig(level=getattr(logging, config.log_level.upper()))
        
        # Initialize components
        self.broker: Optional[BrokerAdapter] = None
        self.order_manager: Optional[OrderManager] = None
        self.router: Optional[ExecutionRouter] = None
        self.latency_monitor: LatencyMonitor = get_latency_monitor()
        
        # Engine state
        self.running = False
        self.initialized = False
        
        # Signal processing
        self.pending_signals: Dict[str, TradingSignal] = {}
        self.signal_callbacks: List[Callable[[str, List[Order]], None]] = []
        
        # Performance tracking
        self.start_time: Optional[datetime] = None
        self.total_signals_processed = 0
        self.total_orders_created = 0
        self.total_orders_executed = 0
        self.total_pnl = 0.0
        
        # Real-time metrics
        self.current_metrics = {
            'signals_per_minute': 0.0,
            'orders_per_minute': 0.0,
            'execution_success_rate': 0.0,
            'average_latency_ms': 0.0,
            'active_positions': 0,
            'portfolio_value': 0.0,
            'daily_pnl': 0.0
        }
        
        # Circuit breaker state
        self.circuit_breaker_active = False
        self.last_error_time: Optional[datetime] = None
        self.consecutive_errors = 0
    
    async def initialize(self) -> bool:
        """Initialize the execution engine and all components"""
        try:
            self.logger.info("Initializing Execution Engine...")
            
            # Initialize broker adapter
            if self.config.dry_run_mode:
                # Use paper trading configuration
                broker_config = self.config.broker_config
                broker_config.paper_trading = True
                self.broker = AlpacaBrokerAdapter(broker_config)
            else:
                self.broker = AlpacaBrokerAdapter(self.config.broker_config)
            
            # Connect to broker
            if not await self.broker.connect():
                self.logger.error("Failed to connect to broker")
                return False
            
            # Initialize order manager
            self.order_manager = OrderManager(
                config=self.config.order_manager_config,
                broker=self.broker
            )
            
            # Set up order callbacks
            self.order_manager.add_order_callback(self._on_order_update)
            self.order_manager.add_signal_completion_callback(self._on_signal_completion)
            
            # Start order manager
            if not await self.order_manager.start():
                self.logger.error("Failed to start Order Manager")
                return False
            
            # Initialize execution router
            self.router = ExecutionRouter(self.config.routing_config)
            await self.router.start()
            
            # Configure latency monitoring
            if self.config.enable_latency_monitoring:
                self.latency_monitor.set_alert_threshold(
                    'end_to_end', 
                    self.config.latency_alert_threshold_ms
                )
            
            self.initialized = True
            self.logger.info("Execution Engine initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Execution Engine: {e}")
            return False
    
    async def start(self) -> bool:
        """Start the execution engine"""
        if not self.initialized:
            if not await self.initialize():
                return False
        
        try:
            self.running = True
            self.start_time = datetime.utcnow()
            
            # Start background monitoring
            asyncio.create_task(self._performance_monitor())
            asyncio.create_task(self._health_monitor())
            
            # Log startup
            mode = "DRY RUN" if self.config.dry_run_mode else "LIVE TRADING"
            self.logger.info(f"🚀 Execution Engine started in {mode} mode")
            
            # Print account info
            if self.broker.account:
                account = self.broker.account
                self.logger.info(f"💰 Account: ${account.equity:,.2f} equity, "
                               f"${account.buying_power:,.2f} buying power")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start Execution Engine: {e}")
            return False
    
    async def stop(self) -> None:
        """Stop the execution engine gracefully"""
        self.logger.info("Stopping Execution Engine...")
        
        self.running = False
        
        # Stop components in reverse order
        if self.router:
            await self.router.stop()
        
        if self.order_manager:
            await self.order_manager.stop()
        
        if self.broker:
            await self.broker.disconnect()
        
        # Log final statistics
        if self.start_time:
            uptime = (datetime.utcnow() - self.start_time).total_seconds()
            self.logger.info(f"📊 Final Stats: {self.total_signals_processed} signals, "
                           f"{self.total_orders_executed} orders, "
                           f"{uptime/60:.1f}min uptime")
        
        self.logger.info("Execution Engine stopped")
    
    async def process_signal(self, signal: TradingSignal) -> bool:
        """
        Process a trading signal and execute the corresponding order
        
        Args:
            signal: Trading signal to process
            
        Returns:
            True if signal was processed successfully
        """
        if not self.running:
            self.logger.warning("Engine not running - signal ignored")
            return False
        
        if self.circuit_breaker_active:
            self.logger.warning("Circuit breaker active - signal ignored")
            return False
        
        # Check concurrency limits
        if len(self.pending_signals) >= self.config.max_concurrent_signals:
            self.logger.warning(f"Too many concurrent signals ({len(self.pending_signals)}) - signal queued")
            # In production, would implement proper queuing
            return False
        
        timer = self.latency_monitor.start_timer(signal.signal_id, "end_to_end")
        
        try:
            self.logger.info(f"🎯 Processing signal: {signal.symbol} {signal.signal_direction.value} "
                           f"${signal.position_size:,.2f} @ ${signal.target_price:.2f}")
            
            # Store signal for tracking
            self.pending_signals[signal.signal_id] = signal
            
            # Create order from signal
            order = await self.order_manager.create_order_from_signal(signal)
            if not order:
                self.logger.error(f"Failed to create order from signal {signal.signal_id}")
                self.pending_signals.pop(signal.signal_id, None)
                return False
            
            # Route the order
            execution_plan = await self.router.route_order(order)
            
            # Submit the order
            success = await self.order_manager.submit_order(order)
            
            if success:
                self.total_signals_processed += 1
                self.total_orders_created += 1
                
                self.logger.info(f"✅ Signal processed successfully: {signal.signal_id} -> {order.order_id}")
                return True
            else:
                self.logger.error(f"❌ Failed to submit order for signal {signal.signal_id}")
                self.pending_signals.pop(signal.signal_id, None)
                return False
        
        except Exception as e:
            self.latency_monitor.stop_timer(signal.signal_id, "end_to_end")
            self.logger.error(f"Error processing signal {signal.signal_id}: {e}")
            self._handle_error(e)
            self.pending_signals.pop(signal.signal_id, None)
            return False
    
    async def cancel_signal(self, signal_id: str) -> bool:
        """Cancel all orders associated with a signal"""
        try:
            if signal_id not in self.pending_signals:
                self.logger.warning(f"Signal {signal_id} not found")
                return False
            
            # Get orders for signal
            orders = self.order_manager.get_orders_for_signal(signal_id)
            
            canceled_count = 0
            for order in orders:
                if await self.order_manager.cancel_order(order.order_id, "Signal cancellation"):
                    canceled_count += 1
            
            self.logger.info(f"Canceled {canceled_count} orders for signal {signal_id}")
            self.pending_signals.pop(signal_id, None)
            
            return canceled_count > 0
            
        except Exception as e:
            self.logger.error(f"Error canceling signal {signal_id}: {e}")
            return False
    
    async def emergency_stop(self) -> None:
        """Emergency stop - cancel all orders and halt trading"""
        self.logger.warning("🚨 EMERGENCY STOP ACTIVATED")
        
        self.circuit_breaker_active = True
        
        # Cancel all active orders
        if self.order_manager:
            canceled_count = await self.order_manager.cancel_all_orders()
            self.logger.info(f"Emergency canceled {canceled_count} orders")
        
        # Clear pending signals
        self.pending_signals.clear()
        
        self.logger.warning("🛑 All trading halted")
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive engine status"""
        
        status = {
            'running': self.running,
            'initialized': self.initialized,
            'circuit_breaker_active': self.circuit_breaker_active,
            'dry_run_mode': self.config.dry_run_mode,
            'uptime_seconds': 0,
            'components': {
                'broker_connected': self.broker.connected if self.broker else False,
                'order_manager_running': self.order_manager._running if self.order_manager else False,
                'router_running': self.router._running if self.router else False
            },
            'metrics': self.current_metrics.copy(),
            'counters': {
                'total_signals_processed': self.total_signals_processed,
                'total_orders_created': self.total_orders_created,
                'total_orders_executed': self.total_orders_executed,
                'pending_signals': len(self.pending_signals),
                'consecutive_errors': self.consecutive_errors
            }
        }
        
        if self.start_time:
            status['uptime_seconds'] = (datetime.utcnow() - self.start_time).total_seconds()
        
        return status
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get detailed performance summary"""
        
        summary = {
            'engine': self.get_status(),
            'latency': self.latency_monitor.get_summary(),
            'positions': self.order_manager.get_positions() if self.order_manager else {},
            'account': None
        }
        
        # Add component performance
        if self.order_manager:
            summary['order_manager'] = self.order_manager.get_performance_summary()
        
        if self.router:
            summary['router'] = self.router.get_performance_summary()
        
        if self.broker and self.broker.account:
            summary['account'] = {
                'equity': self.broker.account.equity,
                'buying_power': self.broker.account.buying_power,
                'cash': self.broker.account.cash,
                'day_trade_count': self.broker.account.day_trade_count
            }
        
        return summary
    
    def add_signal_completion_callback(self, callback: Callable[[str, List[Order]], None]) -> None:
        """Add callback for signal completion events"""
        self.signal_callbacks.append(callback)
    
    def _on_order_update(self, order: Order) -> None:
        """Handle order updates"""
        try:
            # Update execution count
            if order.status == OrderStatus.FILLED:
                self.total_orders_executed += 1
                
                # Calculate PnL (simplified)
                if order.side.value == "buy":
                    pnl = 0  # Will be calculated when position is closed
                else:
                    # Estimate PnL for sell orders
                    pnl = order.filled_quantity * order.average_fill_price
                
                self.total_pnl += pnl
            
            # Log significant order events
            if order.status in [OrderStatus.FILLED, OrderStatus.REJECTED, OrderStatus.CANCELED]:
                self.logger.info(f"📝 Order {order.order_id}: {order.status.value} - "
                               f"{order.symbol} {order.side.value} {order.filled_quantity:.2f}")
        
        except Exception as e:
            self.logger.error(f"Error in order update handler: {e}")
    
    def _on_signal_completion(self, signal_id: str, orders: List[Order]) -> None:
        """Handle signal completion"""
        try:
            # Calculate signal performance
            total_filled = sum(order.filled_quantity for order in orders)
            avg_price = 0.0
            
            if total_filled > 0:
                total_value = sum(order.filled_quantity * order.average_fill_price 
                                for order in orders if order.filled_quantity > 0)
                avg_price = total_value / total_filled
            
            self.logger.info(f"🏁 Signal {signal_id} completed: {total_filled:.2f} filled @ ${avg_price:.2f}")
            
            # Remove from pending signals
            self.pending_signals.pop(signal_id, None)
            
            # Record end-to-end latency
            for order in orders:
                if order.signal_id:
                    self.latency_monitor.stop_timer(order.signal_id, "end_to_end")
            
            # Trigger callbacks
            for callback in self.signal_callbacks:
                try:
                    callback(signal_id, orders)
                except Exception as e:
                    self.logger.error(f"Error in signal completion callback: {e}")
        
        except Exception as e:
            self.logger.error(f"Error in signal completion handler: {e}")
    
    def _handle_error(self, error: Exception) -> None:
        """Handle execution errors with circuit breaker logic"""
        self.consecutive_errors += 1
        self.last_error_time = datetime.utcnow()
        
        self.logger.error(f"Execution error #{self.consecutive_errors}: {error}")
        
        # Circuit breaker logic
        if (self.config.circuit_breaker_enabled and 
            self.consecutive_errors >= 5):
            
            self.logger.error("Too many consecutive errors - activating circuit breaker")
            asyncio.create_task(self.emergency_stop())
    
    async def _performance_monitor(self) -> None:
        """Background task to monitor and update performance metrics"""
        
        while self.running:
            try:
                await asyncio.sleep(60)  # Update every minute
                
                if not self.start_time:
                    continue
                
                # Calculate time-based metrics
                uptime_minutes = (datetime.utcnow() - self.start_time).total_seconds() / 60
                
                if uptime_minutes > 0:
                    self.current_metrics['signals_per_minute'] = self.total_signals_processed / uptime_minutes
                    self.current_metrics['orders_per_minute'] = self.total_orders_created / uptime_minutes
                
                # Calculate success rate
                if self.total_orders_created > 0:
                    self.current_metrics['execution_success_rate'] = (
                        self.total_orders_executed / self.total_orders_created) * 100
                
                # Get latency metrics
                latency_stats = self.latency_monitor.get_stats()
                if 'end_to_end' in latency_stats:
                    self.current_metrics['average_latency_ms'] = latency_stats['end_to_end'].mean
                
                # Update portfolio metrics
                if self.order_manager:
                    positions = self.order_manager.get_positions()
                    self.current_metrics['active_positions'] = len(positions)
                
                if self.broker and self.broker.account:
                    self.current_metrics['portfolio_value'] = self.broker.account.equity
                
                # Reset consecutive errors if no recent errors
                if (self.last_error_time and 
                    (datetime.utcnow() - self.last_error_time).total_seconds() > 300):
                    self.consecutive_errors = 0
                
            except Exception as e:
                self.logger.error(f"Error in performance monitor: {e}")
    
    async def _health_monitor(self) -> None:
        """Background task to monitor system health"""
        
        while self.running:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                # Check component health
                if self.broker and not self.broker.connected:
                    self.logger.warning("Broker connection lost - attempting reconnect")
                    await self.broker.connect()
                
                # Check latency alerts
                alerts = self.latency_monitor.get_recent_alerts(5)
                for alert in alerts:
                    if alert['severity'] == 'HIGH':
                        self.logger.warning(f"🐌 High latency alert: {alert['metric']} = {alert['value']:.1f}ms")
                
                # Check circuit breaker conditions
                if self.circuit_breaker_active:
                    # Auto-reset circuit breaker after cooldown period
                    if (self.last_error_time and 
                        (datetime.utcnow() - self.last_error_time).total_seconds() > 600):  # 10 min cooldown
                        
                        self.logger.info("Circuit breaker cooldown complete - resetting")
                        self.circuit_breaker_active = False
                        self.consecutive_errors = 0
                
            except Exception as e:
                self.logger.error(f"Error in health monitor: {e}")


# Factory function for easy engine creation
def create_execution_engine(
    alpaca_api_key: str,
    alpaca_api_secret: str,
    dry_run: bool = True,
    max_order_size: float = 10000.0
) -> ExecutionEngine:
    """
    Factory function to create a configured execution engine
    
    Args:
        alpaca_api_key: Alpaca API key
        alpaca_api_secret: Alpaca API secret
        dry_run: Whether to use paper trading
        max_order_size: Maximum order size in USD
        
    Returns:
        Configured ExecutionEngine instance
    """
    
    # Create configurations
    broker_config = BrokerConfig(
        api_key=alpaca_api_key,
        api_secret=alpaca_api_secret,
        base_url="",  # Will be set based on paper_trading flag
        paper_trading=dry_run,
        timeout_seconds=10.0
    )
    
    order_manager_config = OrderManagerConfig(
        max_order_size=max_order_size,
        max_daily_loss=5000.0,
        max_open_orders=20,
        emergency_stop=False
    )
    
    routing_config = RoutingConfig(
        max_order_size=max_order_size,
        enable_smart_routing=True,
        enable_impact_modeling=True
    )
    
    execution_config = ExecutionConfig(
        broker_config=broker_config,
        order_manager_config=order_manager_config,
        routing_config=routing_config,
        dry_run_mode=dry_run,
        enable_execution=True
    )
    
    return ExecutionEngine(execution_config) 