"""
Execution Router - Smart order routing and execution optimization

Features:
- Smart order routing based on market conditions
- Order slicing for large orders
- Time-weighted average price (TWAP) execution
- Volume-weighted average price (VWAP) execution
- Algorithmic execution strategies
- Real-time market analysis for optimal routing

Designed for C++ portability with minimal latency.
"""

import asyncio
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from enum import Enum
import logging

from .order_schemas import Order, OrderStatus, OrderSide, OrderType, TimeInForce
from .latency_monitor import get_latency_monitor


class ExecutionStrategy(Enum):
    """Execution strategy types"""
    AGGRESSIVE = "aggressive"       # Market orders, immediate execution
    PASSIVE = "passive"             # Limit orders, patient execution
    TWAP = "twap"                  # Time-weighted average price
    VWAP = "vwap"                  # Volume-weighted average price
    ICEBERG = "iceberg"            # Hidden quantity strategy
    SMART = "smart"                # Dynamic strategy selection


@dataclass
class RoutingConfig:
    """Configuration for execution routing"""
    
    # Strategy selection
    default_strategy: ExecutionStrategy = ExecutionStrategy.SMART
    enable_smart_routing: bool = True
    
    # Order slicing
    max_order_size: float = 10000.0        # Maximum single order size
    slice_size_percent: float = 0.1        # Percentage of volume for slicing
    min_slice_size: float = 100.0          # Minimum slice size
    
    # TWAP configuration
    twap_duration_minutes: int = 30        # TWAP execution duration
    twap_interval_seconds: int = 60        # Time between TWAP slices
    
    # VWAP configuration
    vwap_participation_rate: float = 0.15  # Percentage of volume to participate
    vwap_lookback_minutes: int = 60        # Volume lookback period
    
    # Market impact estimation
    enable_impact_modeling: bool = True
    impact_threshold: float = 0.002        # 0.2% impact threshold
    
    # Venue selection
    preferred_venues: List[str] = None     # Preferred execution venues
    avoid_venues: List[str] = None         # Venues to avoid
    
    # Risk controls
    max_market_impact: float = 0.005       # Maximum allowed market impact
    enable_liquidity_checks: bool = True
    min_liquidity_ratio: float = 5.0      # Order size vs available liquidity
    
    def __post_init__(self):
        if self.preferred_venues is None:
            self.preferred_venues = []
        if self.avoid_venues is None:
            self.avoid_venues = []


@dataclass
class MarketData:
    """Real-time market data for routing decisions"""
    
    symbol: str
    bid_price: float
    ask_price: float
    bid_size: float
    ask_size: float
    last_price: float
    volume: float
    
    # Additional market metrics
    spread: float = 0.0
    spread_percent: float = 0.0
    liquidity_score: float = 0.0
    volatility: float = 0.0
    
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        
        # Calculate derived metrics
        if self.bid_price > 0 and self.ask_price > 0:
            self.spread = self.ask_price - self.bid_price
            if self.ask_price > 0:
                self.spread_percent = self.spread / self.ask_price
        
        # Estimate liquidity score
        if self.bid_size > 0 and self.ask_size > 0:
            self.liquidity_score = math.sqrt(self.bid_size * self.ask_size)


@dataclass
class ExecutionPlan:
    """Execution plan for an order"""
    
    original_order: Order
    strategy: ExecutionStrategy
    child_orders: List[Order]
    
    # Timing parameters
    start_time: datetime
    end_time: datetime
    interval_seconds: float
    
    # Execution parameters
    participation_rate: float = 0.0
    total_slices: int = 0
    completed_slices: int = 0
    
    # Performance tracking
    total_filled: float = 0.0
    average_price: float = 0.0
    total_commission: float = 0.0
    estimated_impact: float = 0.0
    actual_impact: float = 0.0
    
    # Status
    status: str = "pending"  # pending, active, completed, canceled
    
    def get_progress_percent(self) -> float:
        """Get execution progress percentage"""
        if self.total_slices == 0:
            return 0.0
        return (self.completed_slices / self.total_slices) * 100.0
    
    def is_complete(self) -> bool:
        """Check if execution is complete"""
        return self.status in ["completed", "canceled"]


class ExecutionRouter:
    """
    Smart execution router with algorithmic strategies
    
    Routes orders optimally based on market conditions, order size,
    and execution objectives. Implements various execution algorithms.
    """
    
    def __init__(self, config: RoutingConfig):
        self.config = config
        self.logger = logging.getLogger("ExecutionRouter")
        self.latency_monitor = get_latency_monitor()
        
        # Active execution plans
        self.active_plans: Dict[str, ExecutionPlan] = {}
        
        # Market data cache
        self.market_data: Dict[str, MarketData] = {}
        
        # Performance tracking
        self.total_orders_routed = 0
        self.successful_executions = 0
        self.total_market_impact = 0.0
        
        # Background tasks
        self._execution_task: Optional[asyncio.Task] = None
        self._running = False
    
    async def start(self) -> None:
        """Start the execution router"""
        self._running = True
        self._execution_task = asyncio.create_task(self._execution_engine())
        self.logger.info("Execution Router started")
    
    async def stop(self) -> None:
        """Stop the execution router"""
        self._running = False
        
        if self._execution_task:
            self._execution_task.cancel()
        
        # Cancel all active plans
        for plan in self.active_plans.values():
            plan.status = "canceled"
        
        self.logger.info("Execution Router stopped")
    
    async def route_order(self, order: Order) -> ExecutionPlan:
        """
        Route an order using optimal execution strategy
        
        Args:
            order: Order to route
            
        Returns:
            Execution plan for the order
        """
        timer = self.latency_monitor.start_timer(order.order_id, "routing")
        
        try:
            # Get market data for the symbol
            market_data = await self._get_market_data(order.symbol)
            
            # Select execution strategy
            strategy = self._select_strategy(order, market_data)
            
            # Create execution plan
            plan = await self._create_execution_plan(order, strategy, market_data)
            
            # Store the plan
            self.active_plans[order.order_id] = plan
            
            self.latency_monitor.stop_timer(order.order_id, "routing")
            self.total_orders_routed += 1
            
            self.logger.info(f"Routed order {order.order_id} using {strategy.value} strategy")
            return plan
            
        except Exception as e:
            self.latency_monitor.stop_timer(order.order_id, "routing")
            self.logger.error(f"Error routing order {order.order_id}: {e}")
            raise e
    
    def _select_strategy(self, order: Order, market_data: MarketData) -> ExecutionStrategy:
        """Select optimal execution strategy"""
        
        # If strategy is explicitly set, use it
        if hasattr(order, 'execution_strategy'):
            return order.execution_strategy
        
        # Use configured default if not smart routing
        if not self.config.enable_smart_routing:
            return self.config.default_strategy
        
        # Smart strategy selection based on order characteristics
        order_value = order.quantity * (order.limit_price or market_data.last_price)
        
        # For small orders, use aggressive execution
        if order_value < self.config.min_slice_size:
            return ExecutionStrategy.AGGRESSIVE
        
        # For very large orders, use TWAP or VWAP
        if order_value > self.config.max_order_size * 5:
            if market_data.volume > 0:
                return ExecutionStrategy.VWAP
            else:
                return ExecutionStrategy.TWAP
        
        # Check market conditions
        if market_data.spread_percent > 0.01:  # Wide spread
            return ExecutionStrategy.PASSIVE
        
        # Check liquidity
        available_liquidity = market_data.bid_size + market_data.ask_size
        if order.quantity > available_liquidity * 0.5:
            return ExecutionStrategy.ICEBERG
        
        # Default to passive for limit orders, aggressive for market orders
        if order.order_type == OrderType.MARKET:
            return ExecutionStrategy.AGGRESSIVE
        else:
            return ExecutionStrategy.PASSIVE
    
    async def _create_execution_plan(self, order: Order, strategy: ExecutionStrategy, 
                                   market_data: MarketData) -> ExecutionPlan:
        """Create detailed execution plan"""
        
        start_time = datetime.utcnow()
        child_orders = []
        
        if strategy == ExecutionStrategy.AGGRESSIVE:
            # Single market order
            child_orders = [order]
            end_time = start_time + timedelta(seconds=30)
            interval_seconds = 0
            
        elif strategy == ExecutionStrategy.PASSIVE:
            # Single limit order
            child_orders = [order]
            end_time = start_time + timedelta(hours=1)
            interval_seconds = 0
            
        elif strategy == ExecutionStrategy.TWAP:
            # Time-weighted average price execution
            child_orders, interval_seconds = self._create_twap_slices(order)
            end_time = start_time + timedelta(minutes=self.config.twap_duration_minutes)
            
        elif strategy == ExecutionStrategy.VWAP:
            # Volume-weighted average price execution
            child_orders, interval_seconds = await self._create_vwap_slices(order, market_data)
            end_time = start_time + timedelta(minutes=self.config.twap_duration_minutes)
            
        elif strategy == ExecutionStrategy.ICEBERG:
            # Iceberg order execution
            child_orders, interval_seconds = self._create_iceberg_slices(order, market_data)
            end_time = start_time + timedelta(hours=2)
            
        else:  # SMART - dynamic strategy
            # Start with passive and adapt
            child_orders = [order]
            end_time = start_time + timedelta(hours=1)
            interval_seconds = 60
        
        # Estimate market impact
        estimated_impact = self._estimate_market_impact(order, market_data)
        
        plan = ExecutionPlan(
            original_order=order,
            strategy=strategy,
            child_orders=child_orders,
            start_time=start_time,
            end_time=end_time,
            interval_seconds=interval_seconds,
            total_slices=len(child_orders),
            estimated_impact=estimated_impact,
            status="pending"
        )
        
        return plan
    
    def _create_twap_slices(self, order: Order) -> Tuple[List[Order], float]:
        """Create TWAP execution slices"""
        
        duration_seconds = self.config.twap_duration_minutes * 60
        interval_seconds = self.config.twap_interval_seconds
        num_slices = max(1, int(duration_seconds / interval_seconds))
        
        slice_size = order.quantity / num_slices
        child_orders = []
        
        for i in range(num_slices):
            # Adjust last slice for rounding
            if i == num_slices - 1:
                slice_size = order.quantity - (slice_size * i)
            
            child_order = Order(
                order_id=Order.generate_order_id(),
                client_order_id=Order.generate_client_order_id(),
                signal_id=order.signal_id,
                symbol=order.symbol,
                side=order.side,
                order_type=OrderType.LIMIT if order.limit_price else OrderType.MARKET,
                quantity=slice_size,
                limit_price=order.limit_price,
                time_in_force=TimeInForce.IOC,  # Immediate or cancel for slices
                parent_order_id=order.order_id,
                metadata={'slice_index': i, 'strategy': 'twap'}
            )
            
            child_orders.append(child_order)
        
        return child_orders, interval_seconds
    
    async def _create_vwap_slices(self, order: Order, market_data: MarketData) -> Tuple[List[Order], float]:
        """Create VWAP execution slices"""
        
        # Use volume profile to determine slice timing and sizing
        # This is simplified - in production would use historical volume patterns
        
        participation_rate = self.config.vwap_participation_rate
        expected_volume = market_data.volume * participation_rate
        
        if expected_volume <= 0:
            # Fall back to TWAP if no volume data
            return self._create_twap_slices(order)
        
        # Calculate number of slices based on volume participation
        duration_seconds = self.config.twap_duration_minutes * 60
        interval_seconds = min(120, max(30, duration_seconds / 20))  # 30s to 2min intervals
        num_slices = max(1, int(duration_seconds / interval_seconds))
        
        child_orders = []
        remaining_quantity = order.quantity
        
        for i in range(num_slices):
            # Volume-weighted slice sizing (simplified)
            if i == num_slices - 1:
                slice_size = remaining_quantity
            else:
                volume_weight = 1.0 + (0.2 * math.sin(2 * math.pi * i / num_slices))  # Simplified volume pattern
                slice_size = min(remaining_quantity, order.quantity * volume_weight / num_slices)
            
            if slice_size > 0:
                child_order = Order(
                    order_id=Order.generate_order_id(),
                    client_order_id=Order.generate_client_order_id(),
                    signal_id=order.signal_id,
                    symbol=order.symbol,
                    side=order.side,
                    order_type=OrderType.LIMIT if order.limit_price else OrderType.MARKET,
                    quantity=slice_size,
                    limit_price=order.limit_price,
                    time_in_force=TimeInForce.IOC,
                    parent_order_id=order.order_id,
                    metadata={'slice_index': i, 'strategy': 'vwap'}
                )
                
                child_orders.append(child_order)
                remaining_quantity -= slice_size
        
        return child_orders, interval_seconds
    
    def _create_iceberg_slices(self, order: Order, market_data: MarketData) -> Tuple[List[Order], float]:
        """Create iceberg order slices"""
        
        # Size each slice to be a fraction of visible liquidity
        available_liquidity = market_data.bid_size + market_data.ask_size
        max_slice_size = min(
            order.quantity * 0.1,  # Max 10% per slice
            available_liquidity * 0.2  # Max 20% of visible liquidity
        )
        max_slice_size = max(max_slice_size, self.config.min_slice_size)
        
        num_slices = max(1, math.ceil(order.quantity / max_slice_size))
        slice_size = order.quantity / num_slices
        
        child_orders = []
        
        for i in range(num_slices):
            # Adjust last slice for rounding
            if i == num_slices - 1:
                slice_size = order.quantity - (slice_size * i)
            
            child_order = Order(
                order_id=Order.generate_order_id(),
                client_order_id=Order.generate_client_order_id(),
                signal_id=order.signal_id,
                symbol=order.symbol,
                side=order.side,
                order_type=order.order_type,
                quantity=slice_size,
                limit_price=order.limit_price,
                time_in_force=order.time_in_force,
                parent_order_id=order.order_id,
                hidden=True,  # Hidden quantity
                metadata={'slice_index': i, 'strategy': 'iceberg'}
            )
            
            child_orders.append(child_order)
        
        interval_seconds = 30  # 30 seconds between iceberg slices
        return child_orders, interval_seconds
    
    def _estimate_market_impact(self, order: Order, market_data: MarketData) -> float:
        """Estimate market impact of order"""
        
        if not self.config.enable_impact_modeling:
            return 0.0
        
        # Simplified market impact model
        # In production, would use more sophisticated models
        
        order_value = order.quantity * market_data.last_price
        available_liquidity = (market_data.bid_size + market_data.ask_size) * market_data.last_price
        
        if available_liquidity <= 0:
            return 0.01  # 1% default impact if no liquidity data
        
        liquidity_ratio = order_value / available_liquidity
        
        # Square root impact model (simplified)
        base_impact = math.sqrt(liquidity_ratio) * 0.01
        
        # Adjust for spread
        spread_impact = market_data.spread_percent * 0.5
        
        # Adjust for volatility (if available)
        volatility_impact = market_data.volatility * 0.1 if market_data.volatility > 0 else 0
        
        total_impact = base_impact + spread_impact + volatility_impact
        
        return min(total_impact, 0.05)  # Cap at 5% impact
    
    async def _get_market_data(self, symbol: str) -> MarketData:
        """Get current market data for symbol"""
        
        # Check cache first
        if symbol in self.market_data:
            data = self.market_data[symbol]
            # Use cached data if recent (within 5 seconds)
            if (datetime.utcnow() - data.timestamp).total_seconds() < 5:
                return data
        
        # In production, would fetch from real market data feed
        # For now, create mock data
        mock_data = MarketData(
            symbol=symbol,
            bid_price=100.0,
            ask_price=100.05,
            bid_size=1000,
            ask_size=1000,
            last_price=100.02,
            volume=50000,
            volatility=0.02
        )
        
        self.market_data[symbol] = mock_data
        return mock_data
    
    async def _execution_engine(self) -> None:
        """Background execution engine for managing active plans"""
        
        while self._running:
            try:
                await asyncio.sleep(1)  # Check every second
                
                current_time = datetime.utcnow()
                
                # Process all active plans
                for plan_id, plan in list(self.active_plans.items()):
                    if plan.is_complete():
                        continue
                    
                    # Check if plan should start
                    if plan.status == "pending" and current_time >= plan.start_time:
                        plan.status = "active"
                        self.logger.info(f"Started execution plan for order {plan_id}")
                    
                    # Check if plan should end
                    if current_time >= plan.end_time:
                        plan.status = "completed"
                        self.logger.info(f"Completed execution plan for order {plan_id}")
                        continue
                    
                    # Execute next slice if it's time
                    if (plan.status == "active" and 
                        plan.interval_seconds > 0 and
                        plan.completed_slices < len(plan.child_orders)):
                        
                        # Check if it's time for next slice
                        time_since_start = (current_time - plan.start_time).total_seconds()
                        expected_slices = int(time_since_start / plan.interval_seconds) + 1
                        
                        if expected_slices > plan.completed_slices:
                            await self._execute_next_slice(plan)
                
                # Clean up completed plans
                self.active_plans = {
                    plan_id: plan for plan_id, plan in self.active_plans.items()
                    if not plan.is_complete() or 
                    (current_time - plan.end_time).total_seconds() < 3600  # Keep for 1 hour
                }
                
            except Exception as e:
                self.logger.error(f"Error in execution engine: {e}")
    
    async def _execute_next_slice(self, plan: ExecutionPlan) -> None:
        """Execute the next slice in a plan"""
        
        if plan.completed_slices >= len(plan.child_orders):
            return
        
        next_order = plan.child_orders[plan.completed_slices]
        
        # Submit the slice order
        # In production, would submit to OrderManager
        self.logger.info(f"Executing slice {plan.completed_slices + 1}/{len(plan.child_orders)} "
                        f"for plan {plan.original_order.order_id}")
        
        plan.completed_slices += 1
        
        # Update progress
        if plan.completed_slices >= len(plan.child_orders):
            plan.status = "completed"
            self.successful_executions += 1
    
    def get_plan_status(self, order_id: str) -> Optional[ExecutionPlan]:
        """Get execution plan status"""
        return self.active_plans.get(order_id)
    
    def get_active_plans(self) -> List[ExecutionPlan]:
        """Get all active execution plans"""
        return [plan for plan in self.active_plans.values() if not plan.is_complete()]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get router performance summary"""
        
        avg_impact = self.total_market_impact / max(self.successful_executions, 1)
        success_rate = (self.successful_executions / max(self.total_orders_routed, 1)) * 100
        
        return {
            'total_orders_routed': self.total_orders_routed,
            'successful_executions': self.successful_executions,
            'success_rate_percent': success_rate,
            'average_market_impact': avg_impact,
            'active_plans': len(self.get_active_plans()),
            'strategies_used': {
                strategy.value: sum(1 for plan in self.active_plans.values() 
                                  if plan.strategy == strategy)
                for strategy in ExecutionStrategy
            }
        } 