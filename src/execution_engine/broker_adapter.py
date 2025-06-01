"""
Broker Adapter - Interface to broker APIs for order execution

Provides a clean abstraction layer for different brokers.
Current implementation: Alpaca Markets API (REST + WebSocket)

Designed for ultra-low latency with proper error handling and retry logic.
"""

import json
import time
import asyncio
import aiohttp
import websockets
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable, Any, Tuple
from datetime import datetime
import logging
from enum import Enum

from .order_schemas import Order, OrderStatus, OrderSide, OrderType, TimeInForce, OrderExecution
from .latency_monitor import get_latency_monitor, create_timer


class BrokerError(Exception):
    """Base exception for broker-related errors"""
    pass


class OrderRejectedError(BrokerError):
    """Order was rejected by the broker"""
    pass


class InsufficientFundsError(BrokerError):
    """Insufficient funds for the order"""
    pass


class MarketClosedError(BrokerError):
    """Market is closed"""
    pass


class ConnectionError(BrokerError):
    """Connection to broker failed"""
    pass


@dataclass
class BrokerConfig:
    """Configuration for broker connection"""
    
    api_key: str
    api_secret: str
    base_url: str
    paper_trading: bool = True
    timeout_seconds: float = 5.0
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # WebSocket configuration  
    ws_url: str = ""
    enable_websocket: bool = True
    heartbeat_interval: float = 30.0
    
    # Rate limiting
    orders_per_second: float = 10.0
    api_calls_per_minute: float = 200.0


@dataclass 
class BrokerAccount:
    """Broker account information"""
    
    account_id: str
    buying_power: float
    cash: float
    equity: float
    day_trading_buying_power: float
    maintenance_margin: float
    
    # Position information
    long_market_value: float = 0.0
    short_market_value: float = 0.0
    
    # Trading restrictions
    day_trade_count: int = 0
    pattern_day_trader: bool = False
    
    # Status
    account_blocked: bool = False
    trade_suspended: bool = False
    
    last_updated: datetime = None
    
    def __post_init__(self):
        if self.last_updated is None:
            self.last_updated = datetime.utcnow()


class BrokerAdapter(ABC):
    """
    Abstract base class for broker adapters
    
    Defines the interface that all broker implementations must follow.
    Designed for easy extension to other brokers (Interactive Brokers, etc.)
    """
    
    def __init__(self, config: BrokerConfig):
        self.config = config
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.latency_monitor = get_latency_monitor()
        
        # Connection state
        self.connected = False
        self.authenticated = False
        
        # Order tracking
        self.pending_orders: Dict[str, Order] = {}
        self.order_callbacks: Dict[str, List[Callable]] = {}
        
        # Account info
        self.account: Optional[BrokerAccount] = None
        
        # Rate limiting
        self._last_request_time = 0.0
        self._request_count = 0
        
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to the broker"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the broker"""
        pass
    
    @abstractmethod
    async def submit_order(self, order: Order) -> bool:
        """Submit an order to the broker"""
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        pass
    
    @abstractmethod
    async def modify_order(self, order_id: str, new_quantity: float = None, 
                          new_limit_price: float = None) -> bool:
        """Modify an existing order"""
        pass
    
    @abstractmethod
    async def get_order_status(self, order_id: str) -> Optional[Order]:
        """Get current status of an order"""
        pass
    
    @abstractmethod
    async def get_account_info(self) -> Optional[BrokerAccount]:
        """Get account information"""
        pass
    
    @abstractmethod
    async def get_positions(self) -> List[Dict[str, Any]]:
        """Get current positions"""
        pass
    
    def add_order_callback(self, order_id: str, callback: Callable[[Order], None]) -> None:
        """Add callback for order updates"""
        if order_id not in self.order_callbacks:
            self.order_callbacks[order_id] = []
        self.order_callbacks[order_id].append(callback)
    
    def _trigger_order_callbacks(self, order: Order) -> None:
        """Trigger callbacks for order updates"""
        callbacks = self.order_callbacks.get(order.order_id, [])
        for callback in callbacks:
            try:
                callback(order)
            except Exception as e:
                self.logger.error(f"Error in order callback: {e}")
    
    def _check_rate_limit(self) -> None:
        """Check and enforce rate limits"""
        current_time = time.time()
        time_since_last = current_time - self._last_request_time
        
        # Enforce orders per second limit
        min_interval = 1.0 / self.config.orders_per_second
        if time_since_last < min_interval:
            sleep_time = min_interval - time_since_last
            time.sleep(sleep_time)
        
        self._last_request_time = current_time
        self._request_count += 1


class AlpacaBrokerAdapter(BrokerAdapter):
    """
    Alpaca Markets broker adapter implementation
    
    Supports both paper and live trading via Alpaca's REST API.
    Includes WebSocket streaming for real-time order updates.
    """
    
    def __init__(self, config: BrokerConfig):
        super().__init__(config)
        
        # Alpaca-specific URLs
        if config.paper_trading:
            self.rest_base_url = "https://paper-api.alpaca.markets"
            self.ws_url = "wss://paper-api.alpaca.markets/stream"
        else:
            self.rest_base_url = "https://api.alpaca.markets"
            self.ws_url = "wss://api.alpaca.markets/stream"
        
        # HTTP session for REST API
        self.session: Optional[aiohttp.ClientSession] = None
        
        # WebSocket connection
        self.ws_connection = None
        self.ws_task = None
        
        # Headers for API requests
        self.headers = {
            'APCA-API-KEY-ID': config.api_key,
            'APCA-API-SECRET-KEY': config.api_secret,
            'Content-Type': 'application/json'
        }
    
    async def connect(self) -> bool:
        """Connect to Alpaca API"""
        try:
            # Create HTTP session
            timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
            self.session = aiohttp.ClientSession(
                headers=self.headers,
                timeout=timeout
            )
            
            # Test connection by getting account info
            account = await self.get_account_info()
            if account is None:
                self.logger.error("Failed to authenticate with Alpaca")
                return False
            
            self.account = account
            self.connected = True
            self.authenticated = True
            
            # Start WebSocket connection for real-time updates
            if self.config.enable_websocket:
                await self._start_websocket()
            
            self.logger.info(f"Successfully connected to Alpaca ({'Paper' if self.config.paper_trading else 'Live'})")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to Alpaca: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from Alpaca API"""
        self.connected = False
        self.authenticated = False
        
        # Close WebSocket
        if self.ws_task:
            self.ws_task.cancel()
        if self.ws_connection:
            await self.ws_connection.close()
        
        # Close HTTP session
        if self.session:
            await self.session.close()
        
        self.logger.info("Disconnected from Alpaca")
    
    async def submit_order(self, order: Order) -> bool:
        """Submit order to Alpaca"""
        if not self.connected:
            raise ConnectionError("Not connected to Alpaca")
        
        # Start latency timer
        timer = self.latency_monitor.start_timer(order.order_id, "order_submission")
        
        try:
            self._check_rate_limit()
            
            # Convert our order to Alpaca format
            alpaca_order = self._convert_to_alpaca_order(order)
            
            # Submit to Alpaca
            url = f"{self.rest_base_url}/v2/orders"
            
            async with self.session.post(url, json=alpaca_order) as response:
                self.latency_monitor.stop_timer(order.order_id, "order_submission")
                
                if response.status == 201:
                    # Order accepted
                    response_data = await response.json()
                    
                    # Update order with broker response
                    order.broker_order_id = response_data.get('id')
                    order.venue_order_id = response_data.get('id')  
                    order.venue = 'alpaca'
                    order.broker_response = response_data
                    order.update_status(OrderStatus.SUBMITTED)
                    
                    # Track the order
                    self.pending_orders[order.order_id] = order
                    
                    self.logger.info(f"Order submitted successfully: {order.order_id}")
                    self._trigger_order_callbacks(order)
                    
                    return True
                
                elif response.status == 422:
                    # Order validation error
                    error_data = await response.json()
                    error_msg = error_data.get('message', 'Order validation failed')
                    
                    order.update_status(OrderStatus.REJECTED)
                    order.broker_response = error_data
                    
                    self.logger.error(f"Order rejected: {error_msg}")
                    self._trigger_order_callbacks(order)
                    
                    raise OrderRejectedError(error_msg)
                
                elif response.status == 403:
                    # Insufficient buying power
                    error_data = await response.json()
                    order.update_status(OrderStatus.REJECTED)
                    order.broker_response = error_data
                    
                    self._trigger_order_callbacks(order)
                    raise InsufficientFundsError("Insufficient buying power")
                
                else:
                    # Other error
                    error_data = await response.text()
                    self.logger.error(f"Order submission failed: {response.status} - {error_data}")
                    
                    order.update_status(OrderStatus.REJECTED)
                    self._trigger_order_callbacks(order)
                    
                    return False
        
        except Exception as e:
            self.latency_monitor.stop_timer(order.order_id, "order_submission")
            self.logger.error(f"Error submitting order: {e}")
            
            order.update_status(OrderStatus.REJECTED)
            self._trigger_order_callbacks(order)
            
            raise e
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order in Alpaca"""
        if not self.connected:
            raise ConnectionError("Not connected to Alpaca")
        
        try:
            order = self.pending_orders.get(order_id)
            if not order:
                self.logger.error(f"Order {order_id} not found")
                return False
            
            if not order.can_be_canceled():
                self.logger.warning(f"Order {order_id} cannot be canceled (status: {order.status})")
                return False
            
            # Mark as pending cancel
            order.update_status(OrderStatus.PENDING_CANCEL)
            
            # Send cancel request to Alpaca
            url = f"{self.rest_base_url}/v2/orders/{order.broker_order_id}"
            
            async with self.session.delete(url) as response:
                if response.status == 204:
                    # Cancel successful
                    order.update_status(OrderStatus.CANCELED)
                    self.logger.info(f"Order canceled successfully: {order_id}")
                    self._trigger_order_callbacks(order)
                    return True
                else:
                    # Cancel failed
                    error_data = await response.text()
                    self.logger.error(f"Cancel failed: {response.status} - {error_data}")
                    
                    # Revert status
                    order.update_status(OrderStatus.SUBMITTED)
                    return False
        
        except Exception as e:
            self.logger.error(f"Error canceling order: {e}")
            return False
    
    async def modify_order(self, order_id: str, new_quantity: float = None, 
                          new_limit_price: float = None) -> bool:
        """Modify order in Alpaca"""
        if not self.connected:
            raise ConnectionError("Not connected to Alpaca")
        
        try:
            order = self.pending_orders.get(order_id)
            if not order:
                self.logger.error(f"Order {order_id} not found")
                return False
            
            # Build modification request
            modify_data = {}
            if new_quantity is not None:
                modify_data['qty'] = str(new_quantity)
            if new_limit_price is not None:
                modify_data['limit_price'] = str(new_limit_price)
            
            if not modify_data:
                self.logger.warning("No modifications specified")
                return False
            
            # Send modify request
            url = f"{self.rest_base_url}/v2/orders/{order.broker_order_id}"
            
            async with self.session.patch(url, json=modify_data) as response:
                if response.status == 200:
                    response_data = await response.json()
                    
                    # Update order details
                    if new_quantity is not None:
                        order.quantity = new_quantity
                        order.remaining_quantity = new_quantity - order.filled_quantity
                    if new_limit_price is not None:
                        order.limit_price = new_limit_price
                    
                    order.broker_response = response_data
                    order.update_status(OrderStatus.REPLACED)
                    
                    self.logger.info(f"Order modified successfully: {order_id}")
                    self._trigger_order_callbacks(order)
                    return True
                else:
                    error_data = await response.text()
                    self.logger.error(f"Modify failed: {response.status} - {error_data}")
                    return False
        
        except Exception as e:
            self.logger.error(f"Error modifying order: {e}")
            return False
    
    async def get_order_status(self, order_id: str) -> Optional[Order]:
        """Get order status from Alpaca"""
        if not self.connected:
            return None
        
        try:
            order = self.pending_orders.get(order_id)
            if not order or not order.broker_order_id:
                return None
            
            url = f"{self.rest_base_url}/v2/orders/{order.broker_order_id}"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    response_data = await response.json()
                    
                    # Update order with latest info
                    self._update_order_from_alpaca(order, response_data)
                    
                    return order
                else:
                    self.logger.error(f"Failed to get order status: {response.status}")
                    return None
        
        except Exception as e:
            self.logger.error(f"Error getting order status: {e}")
            return None
    
    async def get_account_info(self) -> Optional[BrokerAccount]:
        """Get account information from Alpaca"""
        if not self.session:
            return None
        
        try:
            url = f"{self.rest_base_url}/v2/account"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    account = BrokerAccount(
                        account_id=data.get('id', ''),
                        buying_power=float(data.get('buying_power', 0)),
                        cash=float(data.get('cash', 0)),
                        equity=float(data.get('equity', 0)),
                        day_trading_buying_power=float(data.get('daytrading_buying_power', 0)),
                        maintenance_margin=float(data.get('maintenance_margin', 0)),
                        long_market_value=float(data.get('long_market_value', 0)),
                        short_market_value=float(data.get('short_market_value', 0)),
                        day_trade_count=int(data.get('daytrade_count', 0)),
                        pattern_day_trader=data.get('pattern_day_trader', False),
                        account_blocked=data.get('account_blocked', False),
                        trade_suspended=data.get('trade_suspended_by_user', False)
                    )
                    
                    return account
                else:
                    self.logger.error(f"Failed to get account info: {response.status}")
                    return None
        
        except Exception as e:
            self.logger.error(f"Error getting account info: {e}")
            return None
    
    async def get_positions(self) -> List[Dict[str, Any]]:
        """Get current positions from Alpaca"""
        if not self.connected:
            return []
        
        try:
            url = f"{self.rest_base_url}/v2/positions"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    self.logger.error(f"Failed to get positions: {response.status}")
                    return []
        
        except Exception as e:
            self.logger.error(f"Error getting positions: {e}")
            return []
    
    def _convert_to_alpaca_order(self, order: Order) -> Dict[str, Any]:
        """Convert our order format to Alpaca API format"""
        
        alpaca_order = {
            'symbol': order.symbol,
            'qty': str(order.quantity),
            'side': order.side.value,
            'type': self._convert_order_type(order.order_type),
            'time_in_force': self._convert_time_in_force(order.time_in_force),
            'client_order_id': order.client_order_id
        }
        
        # Add price fields based on order type
        if order.order_type == OrderType.LIMIT and order.limit_price:
            alpaca_order['limit_price'] = str(order.limit_price)
        elif order.order_type == OrderType.STOP and order.stop_price:
            alpaca_order['stop_price'] = str(order.stop_price)
        elif order.order_type == OrderType.STOP_LIMIT:
            if order.limit_price:
                alpaca_order['limit_price'] = str(order.limit_price)
            if order.stop_price:
                alpaca_order['stop_price'] = str(order.stop_price)
        elif order.order_type == OrderType.TRAILING_STOP:
            if order.trail_amount:
                alpaca_order['trail_amount'] = str(order.trail_amount)
            elif order.trail_percent:
                alpaca_order['trail_percent'] = str(order.trail_percent)
        
        # Add extended hours flag if needed
        if order.time_in_force == TimeInForce.DAY:
            alpaca_order['extended_hours'] = False
        
        return alpaca_order
    
    def _convert_order_type(self, order_type: OrderType) -> str:
        """Convert order type to Alpaca format"""
        mapping = {
            OrderType.MARKET: 'market',
            OrderType.LIMIT: 'limit',
            OrderType.STOP: 'stop',
            OrderType.STOP_LIMIT: 'stop_limit',
            OrderType.TRAILING_STOP: 'trailing_stop'
        }
        return mapping.get(order_type, 'market')
    
    def _convert_time_in_force(self, tif: TimeInForce) -> str:
        """Convert time in force to Alpaca format"""
        mapping = {
            TimeInForce.DAY: 'day',
            TimeInForce.GTC: 'gtc',
            TimeInForce.IOC: 'ioc',
            TimeInForce.FOK: 'fok',
            TimeInForce.OPG: 'opg',
            TimeInForce.CLS: 'cls'
        }
        return mapping.get(tif, 'day')
    
    def _update_order_from_alpaca(self, order: Order, alpaca_data: Dict[str, Any]) -> None:
        """Update our order object with data from Alpaca"""
        
        # Update status
        alpaca_status = alpaca_data.get('status', '').lower()
        status_mapping = {
            'new': OrderStatus.SUBMITTED,
            'accepted': OrderStatus.ACCEPTED,
            'pending_new': OrderStatus.PENDING,
            'partially_filled': OrderStatus.PARTIALLY_FILLED,
            'filled': OrderStatus.FILLED,
            'canceled': OrderStatus.CANCELED,
            'rejected': OrderStatus.REJECTED,
            'expired': OrderStatus.EXPIRED,
            'replaced': OrderStatus.REPLACED,
            'pending_cancel': OrderStatus.PENDING_CANCEL,
            'pending_replace': OrderStatus.PENDING_REPLACE
        }
        
        new_status = status_mapping.get(alpaca_status, OrderStatus.PENDING)
        order.update_status(new_status)
        
        # Update fill information
        filled_qty = float(alpaca_data.get('filled_qty', 0))
        if filled_qty > order.filled_quantity:
            # New fill occurred
            fill_qty = filled_qty - order.filled_quantity
            fill_price = float(alpaca_data.get('filled_avg_price', 0))
            
            # Create execution record
            execution = OrderExecution(
                execution_id=f"exec_{int(time.time() * 1000000)}",
                order_id=order.order_id,
                timestamp=datetime.utcnow(),
                quantity=fill_qty,
                price=fill_price,
                commission=0.0,  # Alpaca is commission-free
                market_price=fill_price,
                spread=0.0,
                venue='alpaca',
                venue_order_id=order.broker_order_id or ''
            )
            
            order.add_execution(execution)
        
        # Update broker response
        order.broker_response = alpaca_data
        
        # Trigger callbacks
        self._trigger_order_callbacks(order)
    
    async def _start_websocket(self) -> None:
        """Start WebSocket connection for real-time updates"""
        try:
            self.ws_task = asyncio.create_task(self._websocket_handler())
        except Exception as e:
            self.logger.error(f"Failed to start WebSocket: {e}")
    
    async def _websocket_handler(self) -> None:
        """Handle WebSocket messages"""
        try:
            auth_data = {
                'action': 'auth',
                'key': self.config.api_key,
                'secret': self.config.api_secret
            }
            
            async with websockets.connect(self.ws_url) as websocket:
                self.ws_connection = websocket
                
                # Authenticate
                await websocket.send(json.dumps(auth_data))
                
                # Subscribe to trade updates
                subscribe_data = {
                    'action': 'listen',
                    'data': {
                        'streams': ['trade_updates']
                    }
                }
                await websocket.send(json.dumps(subscribe_data))
                
                self.logger.info("WebSocket connected and subscribed to trade updates")
                
                # Listen for messages
                async for message in websocket:
                    try:
                        data = json.loads(message)
                        await self._handle_websocket_message(data)
                    except Exception as e:
                        self.logger.error(f"Error processing WebSocket message: {e}")
        
        except Exception as e:
            self.logger.error(f"WebSocket error: {e}")
            # Attempt to reconnect after delay
            await asyncio.sleep(5)
            if self.connected:
                await self._start_websocket()
    
    async def _handle_websocket_message(self, data: Dict[str, Any]) -> None:
        """Handle incoming WebSocket messages"""
        
        stream = data.get('stream')
        if stream == 'trade_updates':
            # Trade update message
            trade_data = data.get('data', {})
            order_id = trade_data.get('order', {}).get('client_order_id')
            
            if order_id:
                # Find our order
                order = None
                for pending_order in self.pending_orders.values():
                    if pending_order.client_order_id == order_id:
                        order = pending_order
                        break
                
                if order:
                    # Update order with new information
                    alpaca_order_data = trade_data.get('order', {})
                    self._update_order_from_alpaca(order, alpaca_order_data)
                    
                    # Record latency for fill notification
                    if trade_data.get('event') == 'fill':
                        self.latency_monitor.record_latency(
                            order.order_id, 
                            'fill_notification', 
                            (datetime.utcnow() - order.last_updated).total_seconds() * 1000
                        ) 