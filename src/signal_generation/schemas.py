"""
Signal Schemas - Compatibility layer for trading signals
"""

from enum import Enum
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any

# Import existing classes from signal_schema
from .signal_schema import (
    TradingSignal as BaseTradingSignal,
    PortfolioSignal,
    SignalType,
    SignalDirection, 
    SignalStrength
)


class SignalAction(Enum):
    """Signal actions for compatibility with backtesting system"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE = "close"


@dataclass
class TradingSignal(BaseTradingSignal):
    """
    Extended TradingSignal with action compatibility
    """
    action: Optional[SignalAction] = None
    
    def __post_init__(self):
        """Post-initialization with action mapping"""
        super().__post_init__()
        
        # Map direction to action for backwards compatibility
        if self.action is None:
            direction_to_action = {
                SignalDirection.LONG: SignalAction.BUY,
                SignalDirection.SHORT: SignalAction.SELL,
                SignalDirection.HOLD: SignalAction.HOLD,
                SignalDirection.CLOSE: SignalAction.CLOSE
            }
            self.action = direction_to_action.get(self.direction, SignalAction.HOLD)
    
    @classmethod
    def from_base_signal(cls, base_signal: BaseTradingSignal) -> 'TradingSignal':
        """Create TradingSignal from BaseTradingSignal"""
        signal_dict = base_signal.to_dict()
        signal_dict['timestamp'] = base_signal.timestamp
        signal_dict['valid_until'] = base_signal.valid_until
        signal_dict['signal_type'] = base_signal.signal_type
        signal_dict['direction'] = base_signal.direction
        signal_dict['strength'] = base_signal.strength
        
        return cls(**signal_dict)


# Export all classes that backtesting expects
__all__ = [
    'TradingSignal',
    'PortfolioSignal', 
    'SignalAction',
    'SignalType',
    'SignalDirection',
    'SignalStrength'
] 