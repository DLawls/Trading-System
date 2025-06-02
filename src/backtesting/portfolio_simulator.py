"""
Portfolio Simulator

Models realistic trading execution including fills, slippage, transaction costs,
and portfolio management for backtesting purposes.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger

from ..signal_generation.schemas import TradingSignal, SignalAction
from ..execution_engine.order_schemas import Order, OrderType, OrderStatus, OrderSide


class FillModel(Enum):
    """Different fill models for order execution"""
    IMMEDIATE = "immediate"  # Fill immediately at market price
    REALISTIC = "realistic"  # Model bid-ask spread and partial fills
    CONSERVATIVE = "conservative"  # More conservative fill assumptions


@dataclass
class Position:
    """Represents a position in a single asset"""
    symbol: str
    quantity: float
    avg_price: float
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    last_update: datetime = field(default_factory=datetime.now)


@dataclass
class Transaction:
    """Represents a completed transaction"""
    timestamp: datetime
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    commission: float
    total_cost: float
    order_id: str


@dataclass
class PortfolioState:
    """Current state of the portfolio"""
    timestamp: datetime
    cash: float
    total_value: float
    positions: Dict[str, Position]
    day_pnl: float
    total_pnl: float
    drawdown: float
    max_drawdown: float
    transactions_today: int


class PortfolioSimulator:
    """
    Simulates portfolio management and order execution with realistic
    transaction costs, slippage, and market impact modeling.
    """
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission_rate: float = 0.001,  # 0.1% per trade
        min_commission: float = 1.0,     # Minimum $1 commission
        max_position_size: float = 0.1,   # Max 10% of portfolio per position
        fill_model: FillModel = FillModel.REALISTIC,
        slippage_model: str = "linear",   # "linear", "sqrt", "fixed"
        slippage_rate: float = 0.0005,    # 0.05% default slippage
        bid_ask_spread: float = 0.001,    # 0.1% bid-ask spread
        max_daily_trades: int = 100,
        risk_free_rate: float = 0.02      # 2% risk-free rate for Sharpe
    ):
        
        # Portfolio configuration
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.min_commission = min_commission
        self.max_position_size = max_position_size
        self.fill_model = fill_model
        self.slippage_model = slippage_model
        self.slippage_rate = slippage_rate
        self.bid_ask_spread = bid_ask_spread
        self.max_daily_trades = max_daily_trades
        self.risk_free_rate = risk_free_rate
        
        # Portfolio state
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.pending_orders: List[Order] = []
        self.completed_transactions: List[Transaction] = []
        
        # Performance tracking
        self.portfolio_history: List[PortfolioState] = []
        self.daily_returns: List[float] = []
        self.peak_value = initial_capital
        self.max_drawdown = 0.0
        self.current_drawdown = 0.0
        
        # Market data for pricing
        self.current_market_data: Dict[str, Dict[str, float]] = {}
        
        logger.info(f"Initialized PortfolioSimulator with ${initial_capital:,.2f} capital")
    
    def update_market_data(self, symbol: str, market_data: Dict[str, float]) -> None:
        """Update current market data for a symbol"""
        
        # Expected fields: open, high, low, close, volume
        self.current_market_data[symbol] = market_data.copy()
        
        # Update position market values
        if symbol in self.positions:
            position = self.positions[symbol]
            current_price = market_data.get('close', position.avg_price)
            
            position.market_value = position.quantity * current_price
            position.unrealized_pnl = (current_price - position.avg_price) * position.quantity
            position.last_update = datetime.now()
    
    def process_signal(self, signal: TradingSignal, current_time: datetime) -> Optional[Order]:
        """
        Convert a trading signal into an order and attempt execution
        
        Returns:
            Order object if order was created and processed
        """
        
        if signal.action == SignalAction.HOLD:
            return None
        
        # Check daily trade limit
        today_trades = len([t for t in self.completed_transactions 
                           if t.timestamp.date() == current_time.date()])
        
        if today_trades >= self.max_daily_trades:
            logger.warning(f"Daily trade limit reached ({self.max_daily_trades})")
            return None
        
        # Calculate position size
        position_size = self._calculate_position_size(signal)
        if position_size <= 0:
            logger.warning(f"Invalid position size calculated for {signal.symbol}")
            return None
        
        # Create order
        order = self._create_order_from_signal(signal, position_size, current_time)
        
        # Attempt to execute immediately
        if self._execute_order(order, current_time):
            return order
        else:
            # Add to pending orders if not immediately executed
            self.pending_orders.append(order)
            return order
    
    def _calculate_position_size(self, signal: TradingSignal) -> float:
        """Calculate appropriate position size based on signal and risk management"""
        
        total_portfolio_value = self.get_total_portfolio_value()
        
        if signal.action == SignalAction.BUY:
            # Calculate maximum position value allowed
            max_position_value = total_portfolio_value * self.max_position_size
            
            # Use signal sizing or default to max position
            if hasattr(signal, 'position_size') and signal.position_size > 0:
                target_value = total_portfolio_value * signal.position_size
                position_value = min(target_value, max_position_value)
            else:
                position_value = max_position_value
            
            # Check available cash
            available_cash = self.cash * 0.95  # Reserve 5% cash buffer
            position_value = min(position_value, available_cash)
            
            # Convert to shares
            current_price = self._get_current_price(signal.symbol)
            if current_price > 0:
                shares = position_value / current_price
                return max(0, int(shares))  # Round down to whole shares
            
        elif signal.action == SignalAction.SELL:
            # Sell existing position
            if signal.symbol in self.positions:
                current_quantity = self.positions[signal.symbol].quantity
                
                # Use signal sizing or sell entire position
                if hasattr(signal, 'position_size') and signal.position_size > 0:
                    sell_quantity = current_quantity * signal.position_size
                else:
                    sell_quantity = current_quantity
                
                return max(0, int(sell_quantity))
        
        return 0
    
    def _create_order_from_signal(self, signal: TradingSignal, quantity: float, current_time: datetime) -> Order:
        """Create an Order object from a TradingSignal"""
        
        order_side = OrderSide.BUY if signal.action == SignalAction.BUY else OrderSide.SELL
        
        order = Order(
            symbol=signal.symbol,
            side=order_side,
            order_type=OrderType.MARKET,  # Use market orders for backtesting
            quantity=quantity,
            timestamp=current_time,
            signal_id=getattr(signal, 'signal_id', None)
        )
        
        return order
    
    def _execute_order(self, order: Order, current_time: datetime) -> bool:
        """
        Execute an order with realistic fill modeling
        
        Returns:
            True if order was filled, False otherwise
        """
        
        if order.symbol not in self.current_market_data:
            logger.warning(f"No market data available for {order.symbol}")
            return False
        
        market_data = self.current_market_data[order.symbol]
        base_price = market_data.get('close', 0)
        
        if base_price <= 0:
            logger.warning(f"Invalid price for {order.symbol}: {base_price}")
            return False
        
        # Apply fill model
        fill_price, fill_quantity = self._apply_fill_model(order, base_price, market_data)
        
        if fill_quantity <= 0:
            return False
        
        # Calculate transaction costs
        commission = max(self.min_commission, fill_quantity * fill_price * self.commission_rate)
        
        # Execute the fill
        if order.side == OrderSide.BUY:
            return self._execute_buy(order, fill_price, fill_quantity, commission, current_time)
        else:
            return self._execute_sell(order, fill_price, fill_quantity, commission, current_time)
    
    def _apply_fill_model(self, order: Order, base_price: float, market_data: Dict[str, float]) -> Tuple[float, float]:
        """Apply fill model to determine execution price and quantity"""
        
        if self.fill_model == FillModel.IMMEDIATE:
            # Fill immediately at market price
            return base_price, order.quantity
        
        elif self.fill_model == FillModel.REALISTIC:
            # Model bid-ask spread and slippage
            spread_adjustment = self.bid_ask_spread / 2
            
            if order.side == OrderSide.BUY:
                # Buy at ask price + slippage
                slippage = self._calculate_slippage(order.quantity, base_price, market_data)
                fill_price = base_price * (1 + spread_adjustment + slippage)
            else:
                # Sell at bid price - slippage
                slippage = self._calculate_slippage(order.quantity, base_price, market_data)
                fill_price = base_price * (1 - spread_adjustment - slippage)
            
            # Model partial fills for large orders
            fill_quantity = self._model_partial_fill(order.quantity, market_data)
            
            return fill_price, fill_quantity
        
        elif self.fill_model == FillModel.CONSERVATIVE:
            # More conservative assumptions
            spread_adjustment = self.bid_ask_spread
            slippage = self.slippage_rate * 2  # Double the slippage
            
            if order.side == OrderSide.BUY:
                fill_price = base_price * (1 + spread_adjustment + slippage)
            else:
                fill_price = base_price * (1 - spread_adjustment - slippage)
            
            # Assume only 80% fill on large orders
            if order.quantity * base_price > self.get_total_portfolio_value() * 0.05:
                fill_quantity = order.quantity * 0.8
            else:
                fill_quantity = order.quantity
            
            return fill_price, fill_quantity
        
        return base_price, order.quantity
    
    def _calculate_slippage(self, quantity: float, price: float, market_data: Dict[str, float]) -> float:
        """Calculate slippage based on order size and market conditions"""
        
        order_value = quantity * price
        portfolio_value = self.get_total_portfolio_value()
        
        # Base slippage
        base_slippage = self.slippage_rate
        
        if self.slippage_model == "linear":
            # Linear slippage based on order size relative to portfolio
            size_factor = min(order_value / portfolio_value, 0.1)  # Cap at 10%
            return base_slippage * (1 + size_factor * 10)
        
        elif self.slippage_model == "sqrt":
            # Square root slippage model
            size_factor = min(order_value / portfolio_value, 0.1)
            return base_slippage * (1 + np.sqrt(size_factor * 100))
        
        elif self.slippage_model == "fixed":
            # Fixed slippage
            return base_slippage
        
        return base_slippage
    
    def _model_partial_fill(self, quantity: float, market_data: Dict[str, float]) -> float:
        """Model partial fills based on market volume"""
        
        # If we have volume data, use it
        volume = market_data.get('volume', 1000000)  # Default to 1M volume
        
        # Assume we can fill up to 1% of daily volume instantly
        max_instant_fill = volume * 0.01
        
        if quantity <= max_instant_fill:
            return quantity
        else:
            # Partial fill - can only execute portion immediately
            return min(quantity, max_instant_fill)
    
    def _execute_buy(self, order: Order, price: float, quantity: float, commission: float, timestamp: datetime) -> bool:
        """Execute a buy order"""
        
        total_cost = quantity * price + commission
        
        if total_cost > self.cash:
            logger.warning(f"Insufficient cash for buy order: need ${total_cost:.2f}, have ${self.cash:.2f}")
            return False
        
        # Update cash
        self.cash -= total_cost
        
        # Update position
        if order.symbol in self.positions:
            # Add to existing position
            position = self.positions[order.symbol]
            total_quantity = position.quantity + quantity
            total_cost_basis = (position.quantity * position.avg_price) + (quantity * price)
            position.avg_price = total_cost_basis / total_quantity
            position.quantity = total_quantity
        else:
            # Create new position
            self.positions[order.symbol] = Position(
                symbol=order.symbol,
                quantity=quantity,
                avg_price=price,
                last_update=timestamp
            )
        
        # Record transaction
        transaction = Transaction(
            timestamp=timestamp,
            symbol=order.symbol,
            side=OrderSide.BUY,
            quantity=quantity,
            price=price,
            commission=commission,
            total_cost=total_cost,
            order_id=order.order_id
        )
        self.completed_transactions.append(transaction)
        
        # Update order status
        order.status = OrderStatus.FILLED
        order.filled_quantity = quantity
        order.filled_price = price
        
        logger.debug(f"Executed BUY: {quantity} {order.symbol} @ ${price:.2f}")
        return True
    
    def _execute_sell(self, order: Order, price: float, quantity: float, commission: float, timestamp: datetime) -> bool:
        """Execute a sell order"""
        
        if order.symbol not in self.positions:
            logger.warning(f"Cannot sell {order.symbol}: no position exists")
            return False
        
        position = self.positions[order.symbol]
        
        if quantity > position.quantity:
            logger.warning(f"Cannot sell {quantity} {order.symbol}: only have {position.quantity}")
            return False
        
        # Calculate proceeds
        gross_proceeds = quantity * price
        net_proceeds = gross_proceeds - commission
        
        # Calculate realized P&L
        cost_basis = quantity * position.avg_price
        realized_pnl = gross_proceeds - cost_basis - commission
        
        # Update cash
        self.cash += net_proceeds
        
        # Update position
        position.quantity -= quantity
        position.realized_pnl += realized_pnl
        
        # Remove position if fully sold
        if position.quantity <= 0.001:  # Account for floating point precision
            del self.positions[order.symbol]
        
        # Record transaction
        transaction = Transaction(
            timestamp=timestamp,
            symbol=order.symbol,
            side=OrderSide.SELL,
            quantity=quantity,
            price=price,
            commission=commission,
            total_cost=-net_proceeds,  # Negative for proceeds
            order_id=order.order_id
        )
        self.completed_transactions.append(transaction)
        
        # Update order status
        order.status = OrderStatus.FILLED
        order.filled_quantity = quantity
        order.filled_price = price
        
        logger.debug(f"Executed SELL: {quantity} {order.symbol} @ ${price:.2f} (P&L: ${realized_pnl:.2f})")
        return True
    
    def _get_current_price(self, symbol: str) -> float:
        """Get current market price for a symbol"""
        
        if symbol in self.current_market_data:
            return self.current_market_data[symbol].get('close', 0)
        return 0
    
    def get_total_portfolio_value(self) -> float:
        """Calculate total portfolio value (cash + positions)"""
        
        total_value = self.cash
        
        for symbol, position in self.positions.items():
            current_price = self._get_current_price(symbol)
            if current_price > 0:
                position.market_value = position.quantity * current_price
                total_value += position.market_value
        
        return total_value
    
    def update_portfolio_state(self, current_time: datetime) -> PortfolioState:
        """Update and record current portfolio state"""
        
        total_value = self.get_total_portfolio_value()
        
        # Calculate daily P&L
        if self.portfolio_history:
            previous_value = self.portfolio_history[-1].total_value
            day_pnl = total_value - previous_value
        else:
            day_pnl = total_value - self.initial_capital
        
        # Calculate total P&L
        total_pnl = total_value - self.initial_capital
        
        # Update drawdown
        if total_value > self.peak_value:
            self.peak_value = total_value
            self.current_drawdown = 0.0
        else:
            self.current_drawdown = (self.peak_value - total_value) / self.peak_value
            self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
        
        # Count today's transactions
        transactions_today = len([t for t in self.completed_transactions 
                                if t.timestamp.date() == current_time.date()])
        
        # Create portfolio state
        state = PortfolioState(
            timestamp=current_time,
            cash=self.cash,
            total_value=total_value,
            positions=self.positions.copy(),
            day_pnl=day_pnl,
            total_pnl=total_pnl,
            drawdown=self.current_drawdown,
            max_drawdown=self.max_drawdown,
            transactions_today=transactions_today
        )
        
        self.portfolio_history.append(state)
        
        # Update daily returns for Sharpe calculation
        if len(self.portfolio_history) > 1:
            daily_return = day_pnl / previous_value
            self.daily_returns.append(daily_return)
        
        return state
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        
        if not self.portfolio_history:
            return {}
        
        current_value = self.get_total_portfolio_value()
        total_return = (current_value - self.initial_capital) / self.initial_capital
        
        # Calculate Sharpe ratio
        if len(self.daily_returns) > 1:
            mean_return = np.mean(self.daily_returns)
            std_return = np.std(self.daily_returns)
            
            if std_return > 0:
                # Annualized Sharpe ratio
                risk_free_daily = self.risk_free_rate / 252
                sharpe_ratio = (mean_return - risk_free_daily) / std_return * np.sqrt(252)
            else:
                sharpe_ratio = 0.0
        else:
            sharpe_ratio = 0.0
        
        # Calculate win rate
        profitable_trades = len([t for t in self.completed_transactions 
                               if t.side == OrderSide.SELL and 
                               (t.quantity * t.price - t.commission) > 0])
        total_sell_trades = len([t for t in self.completed_transactions if t.side == OrderSide.SELL])
        win_rate = profitable_trades / total_sell_trades if total_sell_trades > 0 else 0.0
        
        # Calculate turnover
        total_traded_value = sum(t.quantity * t.price for t in self.completed_transactions)
        avg_portfolio_value = np.mean([s.total_value for s in self.portfolio_history])
        turnover = total_traded_value / avg_portfolio_value if avg_portfolio_value > 0 else 0.0
        
        return {
            'total_return': total_return,
            'total_pnl': current_value - self.initial_capital,
            'max_drawdown': self.max_drawdown,
            'current_drawdown': self.current_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'win_rate': win_rate,
            'total_trades': len(self.completed_transactions),
            'turnover': turnover,
            'current_value': current_value,
            'cash_remaining': self.cash
        }
    
    def get_positions_summary(self) -> pd.DataFrame:
        """Get summary of current positions"""
        
        if not self.positions:
            return pd.DataFrame()
        
        positions_data = []
        for symbol, position in self.positions.items():
            current_price = self._get_current_price(symbol)
            market_value = position.quantity * current_price if current_price > 0 else 0
            unrealized_pnl = (current_price - position.avg_price) * position.quantity if current_price > 0 else 0
            
            positions_data.append({
                'symbol': symbol,
                'quantity': position.quantity,
                'avg_price': position.avg_price,
                'current_price': current_price,
                'market_value': market_value,
                'unrealized_pnl': unrealized_pnl,
                'realized_pnl': position.realized_pnl
            })
        
        return pd.DataFrame(positions_data)
    
    def get_transaction_history(self) -> pd.DataFrame:
        """Get complete transaction history"""
        
        if not self.completed_transactions:
            return pd.DataFrame()
        
        transactions_data = []
        for transaction in self.completed_transactions:
            transactions_data.append({
                'timestamp': transaction.timestamp,
                'symbol': transaction.symbol,
                'side': transaction.side.value,
                'quantity': transaction.quantity,
                'price': transaction.price,
                'commission': transaction.commission,
                'total_cost': transaction.total_cost,
                'order_id': transaction.order_id
            })
        
        return pd.DataFrame(transactions_data)
    
    def reset(self) -> None:
        """Reset portfolio to initial state"""
        
        self.cash = self.initial_capital
        self.positions = {}
        self.pending_orders = []
        self.completed_transactions = []
        self.portfolio_history = []
        self.daily_returns = []
        self.peak_value = self.initial_capital
        self.max_drawdown = 0.0
        self.current_drawdown = 0.0
        self.current_market_data = {}
        
        logger.info("Portfolio simulator reset to initial state") 