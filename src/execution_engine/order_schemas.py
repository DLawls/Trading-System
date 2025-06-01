"""
Order Schemas - Data structures for order management and execution

Designed for ultra-low latency with minimal allocations.
Clean, simple schemas that can be easily ported to C++ structs.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Any
from datetime import datetime
import uuid


class OrderSide(Enum):
    """Order side"""
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Order types supported"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


class TimeInForce(Enum):
    """Time in force options"""
    DAY = "day"           # Good for day
    GTC = "gtc"           # Good till canceled
    IOC = "ioc"           # Immediate or cancel
    FOK = "fok"           # Fill or kill
    OPG = "opg"           # At the opening
    CLS = "cls"           # At the close


class OrderStatus(Enum):
    """Order lifecycle status"""
    PENDING = "pending"           # Order created, not yet sent
    SUBMITTED = "submitted"       # Sent to broker
    ACCEPTED = "accepted"         # Accepted by broker
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"            # Completely filled
    CANCELED = "canceled"         # Canceled by user/system
    REJECTED = "rejected"         # Rejected by broker
    EXPIRED = "expired"           # Expired (time-based)
    REPLACED = "replaced"         # Order was replaced
    PENDING_CANCEL = "pending_cancel"  # Cancel request pending
    PENDING_REPLACE = "pending_replace" # Replace request pending


@dataclass
class OrderExecution:
    """Individual execution/fill details"""
    
    execution_id: str           # Unique execution ID
    order_id: str              # Parent order ID
    timestamp: datetime        # Execution timestamp
    
    # Execution details
    quantity: float            # Quantity filled
    price: float              # Execution price
    commission: float         # Commission paid
    
    # Market data
    market_price: float       # Market price at execution
    spread: float            # Bid-ask spread
    
    # Venue info
    venue: str               # Execution venue
    venue_order_id: str      # Venue-specific order ID
    
    # Metadata
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class Order:
    """
    Core order structure - designed for ultra-low latency
    
    Keep this lean and fast for high-frequency operations.
    All timestamps in UTC for consistency.
    """
    
    # Core identification
    order_id: str               # Unique order ID
    client_order_id: str        # Client-side order ID
    signal_id: Optional[str]    # Originating signal ID
    
    # Order specifications
    symbol: str                 # Asset symbol (e.g., 'AAPL', 'BTC/USD')
    side: OrderSide            # Buy or sell
    order_type: OrderType      # Market, limit, etc.
    quantity: float            # Order quantity
    
    # Pricing (optional based on order type)
    limit_price: Optional[float] = None     # Limit price
    stop_price: Optional[float] = None      # Stop price
    trail_amount: Optional[float] = None    # Trailing stop amount
    trail_percent: Optional[float] = None   # Trailing stop percentage
    
    # Time controls
    time_in_force: TimeInForce = TimeInForce.DAY
    good_till_date: Optional[datetime] = None
    
    # Status and lifecycle
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = None
    submitted_at: Optional[datetime] = None
    last_updated: datetime = None
    
    # Execution tracking
    filled_quantity: float = 0.0
    remaining_quantity: float = 0.0
    average_fill_price: float = 0.0
    total_commission: float = 0.0
    
    # Risk and routing
    max_slippage: float = 0.01          # Maximum acceptable slippage
    min_quantity: Optional[float] = None # Minimum fill quantity
    all_or_none: bool = False           # All-or-none flag
    hidden: bool = False                # Hidden/iceberg order
    
    # Broker details
    broker_order_id: Optional[str] = None
    venue: Optional[str] = None
    venue_order_id: Optional[str] = None
    
    # Performance tracking
    latency_metrics: Dict[str, float] = None
    
    # Related orders
    parent_order_id: Optional[str] = None    # For bracket orders
    child_order_ids: List[str] = None        # Child orders (stop loss, take profit)
    
    # Raw broker response
    broker_response: Dict[str, Any] = None
    
    # Additional metadata
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        """Post-initialization setup"""
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        
        if self.last_updated is None:
            self.last_updated = self.created_at
            
        if self.remaining_quantity == 0.0:
            self.remaining_quantity = self.quantity
            
        if self.latency_metrics is None:
            self.latency_metrics = {}
            
        if self.child_order_ids is None:
            self.child_order_ids = []
            
        if self.broker_response is None:
            self.broker_response = {}
            
        if self.metadata is None:
            self.metadata = {}
    
    @classmethod
    def generate_order_id(cls) -> str:
        """Generate unique order ID"""
        return f"ord_{uuid.uuid4().hex[:12]}"
    
    @classmethod
    def generate_client_order_id(cls) -> str:
        """Generate unique client order ID"""
        timestamp = int(datetime.utcnow().timestamp() * 1000000)  # Microseconds
        return f"cli_{timestamp}"
    
    def update_status(self, new_status: OrderStatus, timestamp: datetime = None) -> None:
        """Update order status with timestamp"""
        if timestamp is None:
            timestamp = datetime.utcnow()
            
        self.status = new_status
        self.last_updated = timestamp
        
        # Track submission time
        if new_status == OrderStatus.SUBMITTED and self.submitted_at is None:
            self.submitted_at = timestamp
    
    def add_execution(self, execution: OrderExecution) -> None:
        """Add an execution to this order"""
        
        # Update fill quantities
        self.filled_quantity += execution.quantity
        self.remaining_quantity = self.quantity - self.filled_quantity
        self.total_commission += execution.commission
        
        # Update average fill price
        if self.filled_quantity > 0:
            total_value = (self.average_fill_price * (self.filled_quantity - execution.quantity) + 
                          execution.price * execution.quantity)
            self.average_fill_price = total_value / self.filled_quantity
        
        # Update status based on fill
        if self.remaining_quantity <= 0:
            self.update_status(OrderStatus.FILLED)
        elif self.filled_quantity > 0:
            self.update_status(OrderStatus.PARTIALLY_FILLED)
    
    def get_fill_percentage(self) -> float:
        """Get percentage of order filled"""
        if self.quantity <= 0:
            return 0.0
        return (self.filled_quantity / self.quantity) * 100.0
    
    def is_active(self) -> bool:
        """Check if order is still active (can be filled)"""
        active_statuses = {
            OrderStatus.PENDING,
            OrderStatus.SUBMITTED, 
            OrderStatus.ACCEPTED,
            OrderStatus.PARTIALLY_FILLED,
            OrderStatus.PENDING_CANCEL,
            OrderStatus.PENDING_REPLACE
        }
        return self.status in active_statuses
    
    def is_terminal(self) -> bool:
        """Check if order is in terminal state (final)"""
        terminal_statuses = {
            OrderStatus.FILLED,
            OrderStatus.CANCELED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
            OrderStatus.REPLACED
        }
        return self.status in terminal_statuses
    
    def can_be_canceled(self) -> bool:
        """Check if order can be canceled"""
        cancelable_statuses = {
            OrderStatus.PENDING,
            OrderStatus.SUBMITTED,
            OrderStatus.ACCEPTED,
            OrderStatus.PARTIALLY_FILLED
        }
        return self.status in cancelable_statuses
    
    def get_market_value(self, current_price: float) -> float:
        """Get current market value of the order"""
        return self.quantity * current_price
    
    def get_slippage(self, execution_price: float, reference_price: float) -> float:
        """Calculate slippage for an execution"""
        if reference_price <= 0:
            return 0.0
            
        if self.side == OrderSide.BUY:
            return (execution_price - reference_price) / reference_price
        else:
            return (reference_price - execution_price) / reference_price
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert order to dictionary for serialization"""
        return {
            'order_id': self.order_id,
            'client_order_id': self.client_order_id,
            'signal_id': self.signal_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'order_type': self.order_type.value,
            'quantity': self.quantity,
            'limit_price': self.limit_price,
            'stop_price': self.stop_price,
            'trail_amount': self.trail_amount,
            'trail_percent': self.trail_percent,
            'time_in_force': self.time_in_force.value,
            'good_till_date': self.good_till_date.isoformat() if self.good_till_date else None,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'submitted_at': self.submitted_at.isoformat() if self.submitted_at else None,
            'last_updated': self.last_updated.isoformat(),
            'filled_quantity': self.filled_quantity,
            'remaining_quantity': self.remaining_quantity,
            'average_fill_price': self.average_fill_price,
            'total_commission': self.total_commission,
            'max_slippage': self.max_slippage,
            'min_quantity': self.min_quantity,
            'all_or_none': self.all_or_none,
            'hidden': self.hidden,
            'broker_order_id': self.broker_order_id,
            'venue': self.venue,
            'venue_order_id': self.venue_order_id,
            'latency_metrics': self.latency_metrics,
            'parent_order_id': self.parent_order_id,
            'child_order_ids': self.child_order_ids,
            'broker_response': self.broker_response,
            'metadata': self.metadata
        }
    
    def __str__(self) -> str:
        """String representation for logging"""
        return (f"Order({self.order_id}: {self.side.value} {self.quantity} {self.symbol} "
                f"@ {self.order_type.value} - {self.status.value})")


@dataclass
class BracketOrder:
    """
    Bracket order containing main order + stop loss + take profit
    
    Designed for risk management - every trade has defined exit points.
    """
    
    bracket_id: str             # Unique bracket ID
    main_order: Order          # Primary order
    stop_loss_order: Optional[Order] = None    # Stop loss order
    take_profit_order: Optional[Order] = None  # Take profit order
    
    created_at: datetime = None
    status: str = "pending"     # pending, active, filled, canceled
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
            
        # Link child orders to parent
        if self.stop_loss_order:
            self.stop_loss_order.parent_order_id = self.main_order.order_id
            self.main_order.child_order_ids.append(self.stop_loss_order.order_id)
            
        if self.take_profit_order:
            self.take_profit_order.parent_order_id = self.main_order.order_id
            self.main_order.child_order_ids.append(self.take_profit_order.order_id)
    
    def get_all_orders(self) -> List[Order]:
        """Get all orders in the bracket"""
        orders = [self.main_order]
        if self.stop_loss_order:
            orders.append(self.stop_loss_order)
        if self.take_profit_order:
            orders.append(self.take_profit_order)
        return orders
    
    def is_complete(self) -> bool:
        """Check if bracket is complete (main order filled)"""
        return self.main_order.status == OrderStatus.FILLED
    
    def cancel_all(self) -> None:
        """Mark all orders for cancellation"""
        for order in self.get_all_orders():
            if order.can_be_canceled():
                order.update_status(OrderStatus.PENDING_CANCEL) 