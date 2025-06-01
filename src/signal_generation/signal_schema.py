"""
Signal Schema - Data structures for trading signals
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Any
from datetime import datetime


class SignalType(Enum):
    """Types of trading signals"""
    EVENT_DRIVEN = "event_driven"     # Based on news events
    MOMENTUM = "momentum"             # Based on price momentum
    REVERSAL = "reversal"             # Based on mean reversion
    BREAKOUT = "breakout"             # Based on price breakouts
    VOLATILITY = "volatility"         # Based on volatility patterns
    ENSEMBLE = "ensemble"             # Combined signal from multiple models


class SignalDirection(Enum):
    """Signal direction"""
    LONG = "long"      # Buy signal
    SHORT = "short"    # Sell signal  
    HOLD = "hold"      # Hold current position
    CLOSE = "close"    # Close position


class SignalStrength(Enum):
    """Signal strength levels"""
    WEAK = "weak"          # Low confidence
    MODERATE = "moderate"  # Medium confidence
    STRONG = "strong"      # High confidence
    EXTREME = "extreme"    # Very high confidence


@dataclass
class TradingSignal:
    """
    Comprehensive trading signal with all necessary information
    """
    
    # Basic signal information
    signal_id: str                    # Unique signal identifier
    timestamp: datetime               # When signal was generated
    asset_id: str                     # Asset symbol (e.g., 'AAPL', 'BTC-USD')
    
    # Signal details
    signal_type: SignalType           # Type of signal
    direction: SignalDirection        # Buy/sell/hold/close
    strength: SignalStrength          # Signal strength level
    confidence: float                 # Confidence score (0.0 to 1.0)
    
    # Position sizing
    position_size: float              # Suggested position size (0.0 to 1.0)
    dollar_amount: Optional[float]    # Dollar amount to trade
    stop_loss: Optional[float]        # Stop loss price
    take_profit: Optional[float]      # Take profit price
    
    # Risk management
    max_risk_pct: float               # Maximum risk as % of portfolio
    volatility_adj: float             # Volatility adjustment factor
    portfolio_weight: float           # Suggested portfolio weight
    
    # Model information
    model_id: str                     # ID of the model that generated signal
    prediction_value: float           # Raw model prediction
    features_used: Dict[str, float]   # Key features that drove the signal
    
    # Event context (if applicable)
    triggering_event: Optional[str]   # Event that triggered the signal
    event_impact_score: Optional[float] # Predicted event impact
    time_to_event: Optional[float]    # Hours until event
    
    # Market context
    market_regime: Optional[str]      # Current market regime
    sector: Optional[str]             # Asset sector
    correlation_risk: float           # Correlation with existing positions
    
    # Execution details
    urgency: str = "normal"           # Execution urgency (low/normal/high)
    valid_until: Optional[datetime] = None  # Signal expiration time
    max_slippage: float = 0.01        # Maximum acceptable slippage
    
    # Metadata
    metadata: Dict[str, Any] = None   # Additional metadata
    
    def __post_init__(self):
        """Post-initialization validation"""
        if self.metadata is None:
            self.metadata = {}
            
        # Validate confidence
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")
        
        # Validate position size
        if not 0.0 <= self.position_size <= 1.0:
            raise ValueError(f"Position size must be between 0.0 and 1.0, got {self.position_size}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert signal to dictionary for serialization"""
        return {
            'signal_id': self.signal_id,
            'timestamp': self.timestamp.isoformat(),
            'asset_id': self.asset_id,
            'signal_type': self.signal_type.value,
            'direction': self.direction.value,
            'strength': self.strength.value,
            'confidence': self.confidence,
            'position_size': self.position_size,
            'dollar_amount': self.dollar_amount,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'max_risk_pct': self.max_risk_pct,
            'volatility_adj': self.volatility_adj,
            'portfolio_weight': self.portfolio_weight,
            'model_id': self.model_id,
            'prediction_value': self.prediction_value,
            'features_used': self.features_used,
            'triggering_event': self.triggering_event,
            'event_impact_score': self.event_impact_score,
            'time_to_event': self.time_to_event,
            'market_regime': self.market_regime,
            'sector': self.sector,
            'correlation_risk': self.correlation_risk,
            'urgency': self.urgency,
            'valid_until': self.valid_until.isoformat() if self.valid_until else None,
            'max_slippage': self.max_slippage,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TradingSignal':
        """Create signal from dictionary"""
        # Convert string enums back to enums
        data['signal_type'] = SignalType(data['signal_type'])
        data['direction'] = SignalDirection(data['direction'])
        data['strength'] = SignalStrength(data['strength'])
        
        # Convert timestamp strings back to datetime
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        if data['valid_until']:
            data['valid_until'] = datetime.fromisoformat(data['valid_until'])
            
        return cls(**data)
    
    def get_risk_metrics(self) -> Dict[str, float]:
        """Get risk metrics for the signal"""
        return {
            'max_risk_pct': self.max_risk_pct,
            'volatility_adj': self.volatility_adj,
            'correlation_risk': self.correlation_risk,
            'position_size': self.position_size,
            'portfolio_weight': self.portfolio_weight,
            'confidence': self.confidence
        }
    
    def is_valid(self, current_time: datetime = None) -> bool:
        """Check if signal is still valid"""
        if current_time is None:
            current_time = datetime.utcnow()
            
        if self.valid_until and current_time > self.valid_until:
            return False
            
        return True
    
    def get_execution_priority(self) -> int:
        """Get execution priority (higher = more urgent)"""
        priority_map = {
            "low": 1,
            "normal": 2, 
            "high": 3
        }
        
        base_priority = priority_map.get(self.urgency, 2)
        
        # Boost priority for high confidence signals
        if self.confidence > 0.8:
            base_priority += 1
            
        # Boost priority for strong signals
        if self.strength in [SignalStrength.STRONG, SignalStrength.EXTREME]:
            base_priority += 1
            
        return min(base_priority, 5)  # Cap at 5


@dataclass 
class PortfolioSignal:
    """
    Portfolio-level signal containing multiple asset signals
    """
    
    portfolio_id: str
    timestamp: datetime
    signals: List[TradingSignal]
    
    # Portfolio-level metrics
    total_exposure: float             # Total portfolio exposure
    sector_exposure: Dict[str, float] # Exposure by sector
    correlation_matrix: Dict[str, Dict[str, float]] # Asset correlations
    
    # Risk metrics
    portfolio_var: float              # Value at Risk
    max_drawdown_risk: float          # Maximum drawdown risk
    sharpe_estimate: float            # Estimated Sharpe ratio
    
    # Rebalancing info
    rebalance_required: bool          # Whether rebalancing is needed
    cash_required: float              # Cash needed for trades
    
    def get_total_dollar_amount(self) -> float:
        """Get total dollar amount across all signals"""
        return sum(s.dollar_amount for s in self.signals if s.dollar_amount)
    
    def get_signals_by_direction(self) -> Dict[SignalDirection, List[TradingSignal]]:
        """Group signals by direction"""
        groups = {}
        for signal in self.signals:
            if signal.direction not in groups:
                groups[signal.direction] = []
            groups[signal.direction].append(signal)
        return groups
    
    def get_high_priority_signals(self) -> List[TradingSignal]:
        """Get signals with high execution priority"""
        return [s for s in self.signals if s.get_execution_priority() >= 3] 