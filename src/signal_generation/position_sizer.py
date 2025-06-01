"""
Position Sizer - Calculates exact position sizes with volatility-based risk management
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from loguru import logger
from dataclasses import dataclass
import math

from .signal_schema import TradingSignal, SignalDirection


@dataclass
class PositionSizingConfig:
    """Configuration for position sizing"""
    
    # Portfolio parameters
    total_portfolio_value: float = 100000.0    # Total portfolio value
    max_portfolio_risk: float = 0.02           # Max risk per trade (2% of portfolio)
    max_position_size: float = 0.20            # Max position size (20% of portfolio)
    min_position_size: float = 0.01            # Min position size (1% of portfolio)
    
    # Risk management
    default_stop_loss_pct: float = 0.03        # Default stop loss (3%)
    max_stop_loss_pct: float = 0.08            # Maximum stop loss (8%)
    min_stop_loss_pct: float = 0.01            # Minimum stop loss (1%)
    
    # Take profit settings
    default_risk_reward_ratio: float = 2.0     # Default risk:reward ratio
    max_risk_reward_ratio: float = 5.0         # Maximum risk:reward ratio
    min_risk_reward_ratio: float = 1.2         # Minimum risk:reward ratio
    
    # Volatility adjustments
    volatility_lookback_days: int = 20         # Days to look back for volatility
    volatility_target: float = 0.02           # Target daily volatility (2%)
    volatility_adjustment_factor: float = 1.5  # How much to adjust for volatility
    
    # Transaction costs
    commission_rate: float = 0.001             # Commission rate (0.1%)
    slippage_factor: float = 0.0005           # Expected slippage (0.05%)
    
    # Cash management
    min_cash_reserve: float = 0.05             # Minimum cash reserve (5%)
    max_leverage: float = 1.0                  # Maximum leverage (1.0 = no leverage)


class PositionSizer:
    """
    Calculates optimal position sizes with risk management and volatility adjustment
    """
    
    def __init__(self, config: PositionSizingConfig = None):
        """
        Initialize the PositionSizer
        
        Args:
            config: Position sizing configuration
        """
        self.config = config or PositionSizingConfig()
        self.volatility_cache = {}  # Cache for volatility calculations
        self.position_history = []
        
        logger.info(f"PositionSizer initialized with portfolio value: ${self.config.total_portfolio_value:,.2f}")
    
    def size_position(
        self,
        signal: TradingSignal,
        current_price: float,
        market_data: pd.DataFrame = None,
        existing_positions: Dict[str, float] = None,
        available_cash: float = None
    ) -> TradingSignal:
        """
        Calculate optimal position size and risk parameters for a signal
        
        Args:
            signal: Trading signal to size
            current_price: Current market price of the asset
            market_data: Historical price data for volatility calculation
            existing_positions: Current portfolio positions
            available_cash: Available cash for trading
            
        Returns:
            Updated TradingSignal with position sizing information
        """
        
        try:
            # Calculate available cash
            if available_cash is None:
                available_cash = self._calculate_available_cash(existing_positions)
            
            # Calculate asset volatility
            volatility = self._calculate_volatility(signal.asset_id, market_data)
            
            # Adjust base position size for volatility
            volatility_adjusted_size = self._adjust_position_for_volatility(
                signal.position_size, volatility
            )
            
            # Calculate stop loss price and percentage
            stop_loss_pct, stop_loss_price = self._calculate_stop_loss(
                signal, current_price, volatility
            )
            
            # Calculate position size based on risk
            risk_adjusted_size = self._calculate_risk_adjusted_size(
                volatility_adjusted_size, stop_loss_pct, signal.confidence
            )
            
            # Apply portfolio constraints
            final_position_size = self._apply_portfolio_constraints(
                risk_adjusted_size, existing_positions, available_cash
            )
            
            # Calculate dollar amount
            dollar_amount = final_position_size * self.config.total_portfolio_value
            
            # Adjust for available cash
            if dollar_amount > available_cash:
                dollar_amount = available_cash * 0.95  # Leave 5% buffer
                final_position_size = dollar_amount / self.config.total_portfolio_value
            
            # Calculate number of shares (if equity)
            shares = math.floor(dollar_amount / current_price)
            actual_dollar_amount = shares * current_price
            
            # Recalculate final position size based on actual shares
            final_position_size = actual_dollar_amount / self.config.total_portfolio_value
            
            # Calculate take profit
            take_profit_price = self._calculate_take_profit(
                signal, current_price, stop_loss_price, stop_loss_pct
            )
            
            # Calculate expected returns and costs
            expected_return = self._calculate_expected_return(
                signal, current_price, take_profit_price, final_position_size
            )
            
            # Calculate transaction costs
            transaction_costs = self._calculate_transaction_costs(actual_dollar_amount)
            
            # Update signal with position sizing information
            updated_signal = TradingSignal(
                signal_id=signal.signal_id,
                timestamp=signal.timestamp,
                asset_id=signal.asset_id,
                signal_type=signal.signal_type,
                direction=signal.direction,
                strength=signal.strength,
                confidence=signal.confidence,
                position_size=final_position_size,
                dollar_amount=actual_dollar_amount,
                stop_loss=stop_loss_price,
                take_profit=take_profit_price,
                max_risk_pct=signal.max_risk_pct,
                volatility_adj=signal.volatility_adj,
                portfolio_weight=final_position_size,
                model_id=signal.model_id,
                prediction_value=signal.prediction_value,
                features_used=signal.features_used,
                triggering_event=signal.triggering_event,
                event_impact_score=signal.event_impact_score,
                time_to_event=signal.time_to_event,
                market_regime=signal.market_regime,
                sector=signal.sector,
                correlation_risk=signal.correlation_risk,
                urgency=signal.urgency,
                valid_until=signal.valid_until,
                max_slippage=signal.max_slippage,
                metadata={
                    **signal.metadata,
                    'position_sizing': {
                        'current_price': current_price,
                        'shares': shares,
                        'volatility': volatility,
                        'stop_loss_pct': stop_loss_pct,
                        'expected_return': expected_return,
                        'transaction_costs': transaction_costs,
                        'risk_reward_ratio': (take_profit_price - current_price) / (current_price - stop_loss_price) if signal.direction == SignalDirection.LONG else (current_price - take_profit_price) / (stop_loss_price - current_price),
                        'sizing_config': self.config.__dict__
                    }
                }
            )
            
            # Store for analysis
            self.position_history.append({
                'signal_id': signal.signal_id,
                'asset_id': signal.asset_id,
                'timestamp': signal.timestamp,
                'position_size': final_position_size,
                'dollar_amount': actual_dollar_amount,
                'volatility': volatility,
                'stop_loss_pct': stop_loss_pct
            })
            
            logger.info(f"Sized position for {signal.asset_id}: "
                       f"${actual_dollar_amount:,.2f} ({final_position_size:.1%}) "
                       f"SL: ${stop_loss_price:.2f} TP: ${take_profit_price:.2f}")
            
            return updated_signal
            
        except Exception as e:
            logger.error(f"Error sizing position for {signal.asset_id}: {e}")
            return signal  # Return original signal if sizing fails
    
    def size_batch_positions(
        self,
        signals: List[TradingSignal],
        current_prices: Dict[str, float],
        market_data: Dict[str, pd.DataFrame] = None,
        existing_positions: Dict[str, float] = None
    ) -> List[TradingSignal]:
        """
        Size multiple positions in batch with portfolio-level constraints
        
        Args:
            signals: List of trading signals
            current_prices: Current prices for each asset
            market_data: Historical data for each asset
            existing_positions: Current portfolio positions
            
        Returns:
            List of sized trading signals
        """
        
        sized_signals = []
        available_cash = self._calculate_available_cash(existing_positions)
        total_requested_exposure = 0.0
        
        # First pass: calculate individual sizes
        for signal in signals:
            if signal.asset_id not in current_prices:
                logger.warning(f"No current price for {signal.asset_id}, skipping")
                continue
            
            current_price = current_prices[signal.asset_id]
            asset_market_data = market_data.get(signal.asset_id) if market_data else None
            
            # Size the position
            sized_signal = self.size_position(
                signal, current_price, asset_market_data, existing_positions, available_cash
            )
            
            sized_signals.append(sized_signal)
            total_requested_exposure += sized_signal.position_size
            
            # Update available cash for next signal
            if sized_signal.dollar_amount:
                available_cash -= sized_signal.dollar_amount
        
        # Second pass: apply portfolio-level scaling if needed
        if total_requested_exposure > self.config.max_leverage:
            scale_factor = self.config.max_leverage / total_requested_exposure
            logger.info(f"Scaling positions by {scale_factor:.3f} due to leverage constraints")
            
            # Scale all positions
            for i, signal in enumerate(sized_signals):
                original_size = signal.position_size
                scaled_size = original_size * scale_factor
                scaled_dollar = scaled_size * self.config.total_portfolio_value
                
                # Update the signal
                sized_signals[i] = TradingSignal(
                    **{k: v for k, v in signal.__dict__.items() if k not in ['position_size', 'dollar_amount', 'portfolio_weight']},
                    position_size=scaled_size,
                    dollar_amount=scaled_dollar,
                    portfolio_weight=scaled_size
                )
        
        logger.info(f"Sized {len(sized_signals)} positions with total exposure: "
                   f"{sum(s.position_size for s in sized_signals):.1%}")
        
        return sized_signals
    
    def _calculate_available_cash(self, existing_positions: Dict[str, float] = None) -> float:
        """Calculate available cash for new positions"""
        
        if not existing_positions:
            return self.config.total_portfolio_value * (1 - self.config.min_cash_reserve)
        
        # Calculate current exposure
        total_exposure = sum(abs(weight) for weight in existing_positions.values())
        available_exposure = self.config.max_leverage - total_exposure
        
        # Reserve minimum cash
        max_available = self.config.total_portfolio_value * (1 - self.config.min_cash_reserve)
        available_cash = min(max_available, available_exposure * self.config.total_portfolio_value)
        
        return max(available_cash, 0)
    
    def _calculate_volatility(self, asset_id: str, market_data: pd.DataFrame = None) -> float:
        """Calculate asset volatility"""
        
        # Check cache first
        cache_key = f"{asset_id}_{datetime.now().date()}"
        if cache_key in self.volatility_cache:
            return self.volatility_cache[cache_key]
        
        if market_data is not None and len(market_data) >= self.config.volatility_lookback_days:
            # Calculate from actual data
            returns = market_data['close'].pct_change().dropna()
            recent_returns = returns.tail(self.config.volatility_lookback_days)
            volatility = recent_returns.std()
        else:
            # Use default based on asset type
            if asset_id.endswith('-USD'):  # Crypto
                volatility = 0.05  # 5% daily volatility
            elif asset_id in ['SPY', 'QQQ', 'IWM']:  # ETFs
                volatility = 0.015  # 1.5% daily volatility
            else:  # Individual stocks
                volatility = 0.025  # 2.5% daily volatility
        
        # Cache the result
        self.volatility_cache[cache_key] = volatility
        return volatility
    
    def _adjust_position_for_volatility(self, base_size: float, volatility: float) -> float:
        """Adjust position size based on asset volatility"""
        
        # Calculate volatility adjustment factor
        vol_ratio = volatility / self.config.volatility_target
        adjustment = 1 / (1 + (vol_ratio - 1) * self.config.volatility_adjustment_factor)
        
        # Apply bounds
        adjustment = max(0.3, min(adjustment, 2.0))  # Limit adjustment to 30%-200%
        
        return base_size * adjustment
    
    def _calculate_stop_loss(
        self,
        signal: TradingSignal,
        current_price: float,
        volatility: float
    ) -> Tuple[float, float]:
        """Calculate stop loss percentage and price"""
        
        # Base stop loss on volatility
        volatility_stop = volatility * 2.0  # 2x daily volatility
        
        # Adjust based on signal confidence (higher confidence = tighter stops)
        confidence_adjustment = 1.5 - signal.confidence  # 0.5 to 1.5 multiplier
        adjusted_stop = volatility_stop * confidence_adjustment
        
        # Apply bounds
        stop_loss_pct = max(
            self.config.min_stop_loss_pct,
            min(adjusted_stop, self.config.max_stop_loss_pct)
        )
        
        # Calculate stop loss price
        if signal.direction == SignalDirection.LONG:
            stop_loss_price = current_price * (1 - stop_loss_pct)
        else:  # SHORT
            stop_loss_price = current_price * (1 + stop_loss_pct)
        
        return stop_loss_pct, stop_loss_price
    
    def _calculate_risk_adjusted_size(
        self,
        base_size: float,
        stop_loss_pct: float,
        confidence: float
    ) -> float:
        """Calculate position size based on risk tolerance"""
        
        # Calculate risk per dollar invested
        risk_per_dollar = stop_loss_pct
        
        # Target risk amount
        target_risk = self.config.max_portfolio_risk * confidence  # Scale with confidence
        
        # Calculate maximum position size based on risk
        max_risk_size = target_risk / risk_per_dollar
        
        # Take minimum of base size and risk-adjusted size
        return min(base_size, max_risk_size)
    
    def _apply_portfolio_constraints(
        self,
        position_size: float,
        existing_positions: Dict[str, float],
        available_cash: float
    ) -> float:
        """Apply portfolio-level constraints"""
        
        # Apply absolute limits
        constrained_size = max(
            self.config.min_position_size,
            min(position_size, self.config.max_position_size)
        )
        
        # Check available cash constraint
        max_cash_size = available_cash / self.config.total_portfolio_value
        constrained_size = min(constrained_size, max_cash_size)
        
        # Check total exposure constraint
        if existing_positions:
            current_exposure = sum(abs(weight) for weight in existing_positions.values())
            max_additional_exposure = self.config.max_leverage - current_exposure
            constrained_size = min(constrained_size, max_additional_exposure)
        
        return max(constrained_size, 0)
    
    def _calculate_take_profit(
        self,
        signal: TradingSignal,
        current_price: float,
        stop_loss_price: float,
        stop_loss_pct: float
    ) -> float:
        """Calculate take profit price"""
        
        # Base risk:reward ratio on signal confidence
        base_ratio = self.config.default_risk_reward_ratio
        confidence_bonus = (signal.confidence - 0.5) * 2  # 0 to 1 multiplier
        risk_reward_ratio = base_ratio + confidence_bonus
        
        # Apply bounds
        risk_reward_ratio = max(
            self.config.min_risk_reward_ratio,
            min(risk_reward_ratio, self.config.max_risk_reward_ratio)
        )
        
        # Calculate take profit based on stop loss distance
        stop_distance = abs(current_price - stop_loss_price)
        profit_distance = stop_distance * risk_reward_ratio
        
        if signal.direction == SignalDirection.LONG:
            take_profit_price = current_price + profit_distance
        else:  # SHORT
            take_profit_price = current_price - profit_distance
        
        return take_profit_price
    
    def _calculate_expected_return(
        self,
        signal: TradingSignal,
        current_price: float,
        take_profit_price: float,
        position_size: float
    ) -> float:
        """Calculate expected return for the position"""
        
        # Simple expected return based on confidence
        if signal.direction == SignalDirection.LONG:
            max_return = (take_profit_price - current_price) / current_price
        else:
            max_return = (current_price - take_profit_price) / current_price
        
        # Weight by confidence
        expected_return = max_return * signal.confidence
        
        return expected_return * position_size * self.config.total_portfolio_value
    
    def _calculate_transaction_costs(self, dollar_amount: float) -> float:
        """Calculate transaction costs"""
        
        commission = dollar_amount * self.config.commission_rate
        slippage = dollar_amount * self.config.slippage_factor
        
        return commission + slippage
    
    def update_portfolio_value(self, new_value: float) -> None:
        """Update the total portfolio value"""
        
        self.config.total_portfolio_value = new_value
        logger.info(f"Updated portfolio value to ${new_value:,.2f}")
    
    def get_sizing_summary(self) -> Dict[str, Any]:
        """Get summary of position sizing activity"""
        
        if not self.position_history:
            return {'total_positions': 0}
        
        df = pd.DataFrame(self.position_history)
        
        return {
            'total_positions': len(self.position_history),
            'avg_position_size': df['position_size'].mean(),
            'avg_dollar_amount': df['dollar_amount'].mean(),
            'avg_volatility': df['volatility'].mean(),
            'avg_stop_loss_pct': df['stop_loss_pct'].mean(),
            'total_exposure': df['position_size'].sum(),
            'config': self.config.__dict__
        } 