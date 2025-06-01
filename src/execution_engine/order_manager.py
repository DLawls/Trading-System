"""
Order Manager - Central order lifecycle management

Handles:
- Order creation from trading signals
- Order validation and risk checks
- Order state management and tracking
- Error handling and retry logic
- Performance monitoring

Designed for ultra-low latency with thread-safe operations.
"""

import asyncio
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable, Any, Set
from datetime import datetime, timedelta
from enum import Enum
import logging
from collections import defaultdict

from .order_schemas import Order, OrderStatus, OrderSide, OrderType, TimeInForce, BracketOrder
from .broker_adapter import BrokerAdapter, BrokerError, OrderRejectedError, InsufficientFundsError
from .latency_monitor import get_latency_monitor
from ..signal_generation.signal_schema import TradingSignal


@dataclass
class OrderManagerConfig:
    """Configuration for the Order Manager"""
    
    # Order validation
    max_order_size: float = 10000.0         # Maximum order size in USD
    min_order_size: float = 1.0             # Minimum order size in USD
    max_position_size: float = 50000.0      # Maximum position size in USD
    max_orders_per_minute: int = 100        # Rate limit for order submission
    
    # Risk management
    max_daily_loss: float = 5000.0          # Maximum daily loss limit
    max_drawdown_percent: float = 10.0      # Maximum portfolio drawdown
    enable_day_trading_limits: bool = True   # Enforce PDT limits
    
    # Order management
    order_timeout_minutes: int = 60         # Order timeout (cancel if not filled)
    max_retries: int = 3                    # Maximum retry attempts
    retry_delay_seconds: float = 1.0        # Delay between retries
    
    # Portfolio limits
    max_open_orders: int = 50               # Maximum concurrent open orders
    max_symbols: int = 20                   # Maximum number of symbols
    concentration_limit: float = 0.25       # Maximum allocation per symbol
    
    # Performance
    enable_performance_monitoring: bool = True
    log_all_operations: bool = True
    
    # Emergency controls
    emergency_stop: bool = False            # Emergency stop all trading
    halt_on_errors: bool = True             # Halt trading on consecutive errors


class OrderValidationError(Exception):
    """Order validation failed"""
    pass


class RiskLimitError(Exception):
    """Risk limit exceeded"""
    pass


class OrderManager:
    """
    Central order management system
    
    Thread-safe order lifecycle management with comprehensive risk controls.
    Coordinates between trading signals and broker execution.
    """
    
    def __init__(self, config: OrderManagerConfig, broker: BrokerAdapter):
        self.config = config
        self.broker = broker
        self.logger = logging.getLogger("OrderManager")
        self.latency_monitor = get_latency_monitor()
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Order tracking
        self.orders: Dict[str, Order] = {}                    # All orders
        self.active_orders: Dict[str, Order] = {}             # Active orders
        self.pending_orders: Set[str] = set()                 # Orders awaiting submission
        self.completed_orders: List[Order] = []               # Completed orders
        
        # Signal to order mapping
        self.signal_to_orders: Dict[str, List[str]] = defaultdict(list)
        
        # Portfolio state
        self.positions: Dict[str, float] = defaultdict(float)  # Symbol -> position size
        self.daily_pnl: float = 0.0
        self.total_pnl: float = 0.0
        
        # Risk tracking
        self.daily_order_count = 0
        self.error_count = 0
        self.consecutive_errors = 0
        self.last_error_time: Optional[datetime] = None
        
        # Performance metrics
        self.orders_submitted = 0
        self.orders_filled = 0
        self.orders_rejected = 0
        self.orders_canceled = 0
        self.total_slippage = 0.0
        
        # Background tasks
        self._monitor_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Callbacks
        self.order_callbacks: List[Callable[[Order], None]] = []
        self.signal_completion_callbacks: List[Callable[[str, List[Order]], None]] = []
    
    async def start(self) -> bool:
        """Start the order manager"""
        try:
            self._running = True
            
            # Start background monitoring tasks
            self._monitor_task = asyncio.create_task(self._monitor_orders())
            self._cleanup_task = asyncio.create_task(self._cleanup_completed_orders())
            
            self.logger.info("Order Manager started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start Order Manager: {e}")
            return False
    
    async def stop(self) -> None:
        """Stop the order manager"""
        self._running = False
        
        # Cancel background tasks
        if self._monitor_task:
            self._monitor_task.cancel()
        if self._cleanup_task:
            self._cleanup_task.cancel()
        
        # Cancel all pending orders
        await self.cancel_all_orders()
        
        self.logger.info("Order Manager stopped")
    
    async def create_order_from_signal(self, signal: TradingSignal) -> Optional[Order]:
        """
        Create an order from a trading signal
        
        Args:
            signal: Trading signal to convert to order
            
        Returns:
            Created order or None if validation fails
        """
        if self.config.emergency_stop:
            self.logger.warning("Trading halted - emergency stop active")
            return None
        
        # Start latency timer
        timer = self.latency_monitor.start_timer(signal.signal_id, "signal_to_order")
        
        try:
            # Validate signal
            if not self._validate_signal(signal):
                return None
            
            # Create order from signal
            order = self._convert_signal_to_order(signal)
            
            # Validate order
            if not await self._validate_order(order):
                return None
            
            # Record order
            with self._lock:
                self.orders[order.order_id] = order
                self.signal_to_orders[signal.signal_id].append(order.order_id)
                self.pending_orders.add(order.order_id)
            
            self.latency_monitor.stop_timer(signal.signal_id, "signal_to_order")
            
            self.logger.info(f"Created order {order.order_id} from signal {signal.signal_id}")
            return order
            
        except Exception as e:
            self.latency_monitor.stop_timer(signal.signal_id, "signal_to_order")
            self.logger.error(f"Failed to create order from signal: {e}")
            self._handle_error(e)
            return None
    
    async def submit_order(self, order: Order) -> bool:
        """
        Submit an order to the broker
        
        Args:
            order: Order to submit
            
        Returns:
            True if submitted successfully
        """
        if self.config.emergency_stop:
            self.logger.warning(f"Order submission blocked - emergency stop active: {order.order_id}")
            return False
        
        # Start latency timer
        timer = self.latency_monitor.start_timer(order.order_id, "order_creation")
        
        try:
            # Final validation before submission
            if not await self._pre_submission_validation(order):
                return False
            
            # Add order callback for tracking
            self.broker.add_order_callback(order.order_id, self._on_order_update)
            
            # Submit to broker
            success = await self.broker.submit_order(order)
            
            if success:
                with self._lock:
                    self.active_orders[order.order_id] = order
                    self.pending_orders.discard(order.order_id)
                    self.orders_submitted += 1
                    self.daily_order_count += 1
                
                self.latency_monitor.stop_timer(order.order_id, "order_creation")
                self.logger.info(f"Order submitted successfully: {order.order_id}")
                
                # Trigger callbacks
                self._trigger_order_callbacks(order)
                
                return True
            else:
                with self._lock:
                    self.pending_orders.discard(order.order_id)
                    self.orders_rejected += 1
                
                self.latency_monitor.stop_timer(order.order_id, "order_creation")
                self.logger.error(f"Order submission failed: {order.order_id}")
                return False
        
        except Exception as e:
            self.latency_monitor.stop_timer(order.order_id, "order_creation")
            self.logger.error(f"Error submitting order {order.order_id}: {e}")
            self._handle_error(e)
            
            with self._lock:
                self.pending_orders.discard(order.order_id)
                self.orders_rejected += 1
            
            return False
    
    async def cancel_order(self, order_id: str, reason: str = "User request") -> bool:
        """Cancel an order"""
        try:
            with self._lock:
                order = self.orders.get(order_id)
                if not order:
                    self.logger.warning(f"Order {order_id} not found for cancellation")
                    return False
                
                if not order.can_be_canceled():
                    self.logger.warning(f"Order {order_id} cannot be canceled (status: {order.status})")
                    return False
            
            # Cancel with broker
            success = await self.broker.cancel_order(order_id)
            
            if success:
                with self._lock:
                    self.orders_canceled += 1
                
                self.logger.info(f"Order canceled: {order_id} - {reason}")
                return True
            else:
                self.logger.error(f"Failed to cancel order: {order_id}")
                return False
        
        except Exception as e:
            self.logger.error(f"Error canceling order {order_id}: {e}")
            self._handle_error(e)
            return False
    
    async def cancel_all_orders(self) -> int:
        """Cancel all active orders"""
        canceled_count = 0
        
        with self._lock:
            orders_to_cancel = [order for order in self.active_orders.values() 
                              if order.can_be_canceled()]
        
        for order in orders_to_cancel:
            if await self.cancel_order(order.order_id, "Cancel all request"):
                canceled_count += 1
        
        self.logger.info(f"Canceled {canceled_count} orders")
        return canceled_count
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID"""
        with self._lock:
            return self.orders.get(order_id)
    
    def get_orders_for_signal(self, signal_id: str) -> List[Order]:
        """Get all orders for a signal"""
        with self._lock:
            order_ids = self.signal_to_orders.get(signal_id, [])
            return [self.orders[order_id] for order_id in order_ids if order_id in self.orders]
    
    def get_active_orders(self) -> List[Order]:
        """Get all active orders"""
        with self._lock:
            return list(self.active_orders.values())
    
    def get_positions(self) -> Dict[str, float]:
        """Get current positions"""
        with self._lock:
            return self.positions.copy()
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        with self._lock:
            fill_rate = (self.orders_filled / max(self.orders_submitted, 1)) * 100
            avg_slippage = self.total_slippage / max(self.orders_filled, 1)
            
            return {
                'orders_submitted': self.orders_submitted,
                'orders_filled': self.orders_filled,
                'orders_rejected': self.orders_rejected,
                'orders_canceled': self.orders_canceled,
                'fill_rate_percent': fill_rate,
                'daily_order_count': self.daily_order_count,
                'active_orders': len(self.active_orders),
                'pending_orders': len(self.pending_orders),
                'total_pnl': self.total_pnl,
                'daily_pnl': self.daily_pnl,
                'error_count': self.error_count,
                'consecutive_errors': self.consecutive_errors,
                'average_slippage': avg_slippage,
                'positions': len(self.positions)
            }
    
    def add_order_callback(self, callback: Callable[[Order], None]) -> None:
        """Add order update callback"""
        self.order_callbacks.append(callback)
    
    def add_signal_completion_callback(self, callback: Callable[[str, List[Order]], None]) -> None:
        """Add signal completion callback"""
        self.signal_completion_callbacks.append(callback)
    
    def _validate_signal(self, signal: TradingSignal) -> bool:
        """Validate trading signal"""
        try:
            # Check emergency stop
            if self.config.emergency_stop:
                return False
            
            # Check order rate limits
            if self.daily_order_count >= self.config.max_orders_per_minute * 24 * 60:
                raise RiskLimitError("Daily order limit exceeded")
            
            # Check symbol limits
            with self._lock:
                unique_symbols = len(set(order.symbol for order in self.active_orders.values()))
                if unique_symbols >= self.config.max_symbols:
                    if signal.symbol not in [order.symbol for order in self.active_orders.values()]:
                        raise RiskLimitError("Maximum symbols limit exceeded")
            
            # Check position size limits
            position_value = abs(self.positions.get(signal.symbol, 0)) * signal.target_price
            if position_value > self.config.max_position_size:
                raise RiskLimitError(f"Position size limit exceeded for {signal.symbol}")
            
            # Check concentration limits
            target_allocation = signal.position_size / signal.portfolio_value
            if target_allocation > self.config.concentration_limit:
                raise RiskLimitError(f"Concentration limit exceeded: {target_allocation:.1%}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Signal validation failed: {e}")
            return False
    
    def _convert_signal_to_order(self, signal: TradingSignal) -> Order:
        """Convert trading signal to order"""
        
        # Determine order side
        if signal.signal_direction.value > 0:
            side = OrderSide.BUY
        else:
            side = OrderSide.SELL
        
        # Calculate quantity
        quantity = abs(signal.position_size / signal.target_price)
        
        # Determine order type and pricing
        if signal.signal_type.value == "market":
            order_type = OrderType.MARKET
            limit_price = None
        else:
            order_type = OrderType.LIMIT
            limit_price = signal.target_price
        
        # Create order
        order = Order(
            order_id=Order.generate_order_id(),
            client_order_id=Order.generate_client_order_id(),
            signal_id=signal.signal_id,
            symbol=signal.symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            limit_price=limit_price,
            time_in_force=TimeInForce.DAY,
            max_slippage=0.005,  # 0.5% slippage tolerance
            metadata={
                'signal_strength': signal.signal_strength.value,
                'signal_confidence': signal.confidence,
                'risk_level': signal.risk_level.value
            }
        )
        
        return order
    
    async def _validate_order(self, order: Order) -> bool:
        """Validate order before submission"""
        try:
            # Check order size limits
            order_value = order.quantity * (order.limit_price or 100)  # Estimate for market orders
            
            if order_value < self.config.min_order_size:
                raise OrderValidationError(f"Order size too small: ${order_value:.2f}")
            
            if order_value > self.config.max_order_size:
                raise OrderValidationError(f"Order size too large: ${order_value:.2f}")
            
            # Check open order limits
            with self._lock:
                if len(self.active_orders) >= self.config.max_open_orders:
                    raise RiskLimitError("Maximum open orders limit exceeded")
            
            # Check buying power (if we have account info)
            if self.broker.account:
                if order.side == OrderSide.BUY:
                    if order_value > self.broker.account.buying_power:
                        raise InsufficientFundsError("Insufficient buying power")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Order validation failed: {e}")
            return False
    
    async def _pre_submission_validation(self, order: Order) -> bool:
        """Final validation before broker submission"""
        try:
            # Check for duplicate orders
            with self._lock:
                if order.order_id in self.active_orders:
                    raise OrderValidationError("Duplicate order ID")
            
            # Check market hours (basic check)
            current_time = datetime.utcnow()
            if current_time.weekday() >= 5:  # Weekend
                if order.order_type == OrderType.MARKET:
                    raise OrderValidationError("Market orders not allowed on weekends")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Pre-submission validation failed: {e}")
            return False
    
    def _on_order_update(self, order: Order) -> None:
        """Handle order updates from broker"""
        try:
            with self._lock:
                # Update order in our records
                self.orders[order.order_id] = order
                
                # Update active orders
                if order.is_active():
                    self.active_orders[order.order_id] = order
                elif order.order_id in self.active_orders:
                    del self.active_orders[order.order_id]
                    self.completed_orders.append(order)
                
                # Update statistics
                if order.status == OrderStatus.FILLED:
                    self.orders_filled += 1
                    self._update_position(order)
                    
                    # Calculate slippage if we have reference price
                    if hasattr(order, 'reference_price') and order.reference_price:
                        slippage = abs(order.average_fill_price - order.reference_price) / order.reference_price
                        self.total_slippage += slippage
                
                elif order.status == OrderStatus.REJECTED:
                    self.orders_rejected += 1
                    self._handle_order_rejection(order)
                
                elif order.status == OrderStatus.CANCELED:
                    self.orders_canceled += 1
            
            # Trigger callbacks
            self._trigger_order_callbacks(order)
            
            # Check if signal is complete
            self._check_signal_completion(order)
            
        except Exception as e:
            self.logger.error(f"Error handling order update: {e}")
    
    def _update_position(self, order: Order) -> None:
        """Update position tracking"""
        position_change = order.filled_quantity
        if order.side == OrderSide.SELL:
            position_change = -position_change
        
        self.positions[order.symbol] += position_change
        
        # Remove zero positions
        if abs(self.positions[order.symbol]) < 0.001:
            del self.positions[order.symbol]
    
    def _handle_order_rejection(self, order: Order) -> None:
        """Handle order rejection"""
        self.logger.warning(f"Order rejected: {order.order_id}")
        
        # Check if we should retry
        retry_count = order.metadata.get('retry_count', 0)
        if retry_count < self.config.max_retries:
            # Schedule retry (simplified - in production would use proper task scheduling)
            order.metadata['retry_count'] = retry_count + 1
            self.logger.info(f"Scheduling retry {retry_count + 1} for order {order.order_id}")
    
    def _check_signal_completion(self, order: Order) -> None:
        """Check if all orders for a signal are complete"""
        if not order.signal_id:
            return
        
        with self._lock:
            signal_orders = self.get_orders_for_signal(order.signal_id)
            
            # Check if all orders are in terminal state
            all_complete = all(order.is_terminal() for order in signal_orders)
            
            if all_complete:
                # Signal execution complete
                for callback in self.signal_completion_callbacks:
                    try:
                        callback(order.signal_id, signal_orders)
                    except Exception as e:
                        self.logger.error(f"Error in signal completion callback: {e}")
    
    def _trigger_order_callbacks(self, order: Order) -> None:
        """Trigger order update callbacks"""
        for callback in self.order_callbacks:
            try:
                callback(order)
            except Exception as e:
                self.logger.error(f"Error in order callback: {e}")
    
    def _handle_error(self, error: Exception) -> None:
        """Handle and track errors"""
        self.error_count += 1
        self.consecutive_errors += 1
        self.last_error_time = datetime.utcnow()
        
        # Check if we should halt trading
        if (self.config.halt_on_errors and 
            self.consecutive_errors >= 3 and
            isinstance(error, (BrokerError, RiskLimitError))):
            
            self.logger.error("Multiple consecutive errors - activating emergency stop")
            self.config.emergency_stop = True
    
    async def _monitor_orders(self) -> None:
        """Background task to monitor order status"""
        while self._running:
            try:
                await asyncio.sleep(5)  # Check every 5 seconds
                
                current_time = datetime.utcnow()
                
                with self._lock:
                    orders_to_check = list(self.active_orders.values())
                
                for order in orders_to_check:
                    # Check for timeout
                    if (order.created_at and 
                        current_time - order.created_at > timedelta(minutes=self.config.order_timeout_minutes)):
                        
                        self.logger.warning(f"Order timeout - canceling: {order.order_id}")
                        await self.cancel_order(order.order_id, "Timeout")
                
                # Reset consecutive errors if no recent errors
                if (self.last_error_time and 
                    current_time - self.last_error_time > timedelta(minutes=5)):
                    self.consecutive_errors = 0
            
            except Exception as e:
                self.logger.error(f"Error in order monitoring: {e}")
    
    async def _cleanup_completed_orders(self) -> None:
        """Background task to clean up old completed orders"""
        while self._running:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                cutoff_time = datetime.utcnow() - timedelta(hours=24)
                
                with self._lock:
                    # Keep only recent completed orders
                    self.completed_orders = [
                        order for order in self.completed_orders
                        if order.last_updated > cutoff_time
                    ]
                
            except Exception as e:
                self.logger.error(f"Error in cleanup task: {e}") 