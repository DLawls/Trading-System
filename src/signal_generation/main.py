"""
Signal Generation Main Module

Provides a unified interface for generating trading signals by orchestrating
the signal evaluator, position sizer, and portfolio allocator components.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any
from loguru import logger

from .signal_evaluator import SignalEvaluator, EvaluationConfig
from .position_sizer import PositionSizer, PositionSizingConfig
from .portfolio_allocator import PortfolioAllocator, AllocationConfig
from .schemas import TradingSignal, SignalAction, SignalType, SignalDirection, SignalStrength


class SignalGenerator:
    """
    Main signal generation orchestrator that combines evaluation, sizing, and allocation
    """
    
    def __init__(
        self,
        evaluation_config: Optional[EvaluationConfig] = None,
        sizing_config: Optional[PositionSizingConfig] = None,
        allocation_config: Optional[AllocationConfig] = None
    ):
        
        # Initialize components with default configs if not provided
        self.signal_evaluator = SignalEvaluator(evaluation_config or EvaluationConfig())
        self.position_sizer = PositionSizer(sizing_config or PositionSizingConfig())
        self.portfolio_allocator = PortfolioAllocator(allocation_config or AllocationConfig())
        
        logger.info("Initialized SignalGenerator with all components")
    
    async def generate_signals(
        self,
        symbols: List[str],
        market_data: Dict[str, pd.DataFrame],
        news_data: Optional[pd.DataFrame] = None,
        events: Optional[List] = None,
        features: Optional[Dict[str, Any]] = None,
        current_time: Optional[datetime] = None
    ) -> List[TradingSignal]:
        """
        Generate trading signals for the given symbols and market conditions
        
        Args:
            symbols: List of symbols to generate signals for
            market_data: Dictionary of symbol -> OHLCV DataFrame
            news_data: Optional news data
            events: Optional detected events
            features: Optional feature data
            current_time: Current simulation time
            
        Returns:
            List of TradingSignal objects
        """
        
        if current_time is None:
            current_time = datetime.now()
        
        all_signals = []
        
        try:
            for symbol in symbols:
                if symbol not in market_data or market_data[symbol].empty:
                    continue
                
                df = market_data[symbol]
                if len(df) < 20:  # Need sufficient data
                    continue
                
                # Generate signals for this symbol
                symbol_signals = await self._generate_symbol_signals(
                    symbol=symbol,
                    market_data=df,
                    news_data=news_data,
                    events=events,
                    features=features,
                    current_time=current_time
                )
                
                all_signals.extend(symbol_signals)
            
            logger.debug(f"Generated {len(all_signals)} signals for {len(symbols)} symbols")
            
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
        
        return all_signals
    
    async def _generate_symbol_signals(
        self,
        symbol: str,
        market_data: pd.DataFrame,
        news_data: Optional[pd.DataFrame],
        events: Optional[List],
        features: Optional[Dict[str, Any]],
        current_time: datetime
    ) -> List[TradingSignal]:
        """Generate signals for a specific symbol"""
        
        signals = []
        
        try:
            # Generate basic signals using simple strategies
            momentum_signals = self._generate_momentum_signals(symbol, market_data, current_time)
            signals.extend(momentum_signals)
            
            # Generate mean reversion signals
            reversion_signals = self._generate_mean_reversion_signals(symbol, market_data, current_time)
            signals.extend(reversion_signals)
            
            # Generate event-driven signals if events are available
            if events:
                event_signals = self._generate_event_signals(symbol, market_data, events, current_time)
                signals.extend(event_signals)
            
            # Evaluate and filter signals
            evaluated_signals = []
            for signal in signals:
                if self.signal_evaluator.should_execute_signal(signal):
                    evaluated_signals.append(signal)
            
            return evaluated_signals
            
        except Exception as e:
            logger.warning(f"Error generating signals for {symbol}: {e}")
            return []
    
    def _generate_momentum_signals(
        self, 
        symbol: str, 
        df: pd.DataFrame, 
        current_time: datetime
    ) -> List[TradingSignal]:
        """Generate momentum-based signals"""
        
        signals = []
        
        if len(df) >= 20:
            # Calculate moving averages
            short_ma = df['close'].rolling(5).mean().iloc[-1]
            medium_ma = df['close'].rolling(10).mean().iloc[-1]
            long_ma = df['close'].rolling(20).mean().iloc[-1]
            current_price = df['close'].iloc[-1]
            
            # Strong momentum up
            if short_ma > medium_ma > long_ma and current_price > short_ma * 1.01:
                signals.append(TradingSignal(
                    signal_id=f"momentum_buy_{symbol}_{int(current_time.timestamp())}",
                    timestamp=current_time,
                    asset_id=symbol,
                    signal_type=SignalType.MOMENTUM,
                    direction=SignalDirection.LONG,
                    strength=SignalStrength.MODERATE,
                    confidence=0.7,
                    position_size=0.1,
                    dollar_amount=None,
                    stop_loss=current_price * 0.95,
                    take_profit=current_price * 1.10,
                    max_risk_pct=0.02,
                    volatility_adj=1.0,
                    portfolio_weight=0.1,
                    model_id="momentum_strategy",
                    prediction_value=0.7,
                    features_used={"short_ma": short_ma, "long_ma": long_ma},
                    triggering_event="momentum_breakout",
                    event_impact_score=0.6,
                    time_to_event=None,
                    market_regime="trending",
                    sector="unknown",
                    correlation_risk=0.3
                ))
            
            # Strong momentum down
            elif short_ma < medium_ma < long_ma and current_price < short_ma * 0.99:
                signals.append(TradingSignal(
                    signal_id=f"momentum_sell_{symbol}_{int(current_time.timestamp())}",
                    timestamp=current_time,
                    asset_id=symbol,
                    signal_type=SignalType.MOMENTUM,
                    direction=SignalDirection.SHORT,
                    strength=SignalStrength.MODERATE,
                    confidence=0.6,
                    position_size=1.0,
                    dollar_amount=None,
                    stop_loss=current_price * 1.05,
                    take_profit=current_price * 0.90,
                    max_risk_pct=0.02,
                    volatility_adj=1.0,
                    portfolio_weight=0.1,
                    model_id="momentum_strategy",
                    prediction_value=0.6,
                    features_used={"short_ma": short_ma, "long_ma": long_ma},
                    triggering_event="momentum_breakdown",
                    event_impact_score=0.5,
                    time_to_event=None,
                    market_regime="trending",
                    sector="unknown",
                    correlation_risk=0.3
                ))
        
        return signals
    
    def _generate_mean_reversion_signals(
        self, 
        symbol: str, 
        df: pd.DataFrame, 
        current_time: datetime
    ) -> List[TradingSignal]:
        """Generate mean reversion signals"""
        
        signals = []
        
        if len(df) >= 20:
            price = df['close'].iloc[-1]
            mean_price = df['close'].rolling(20).mean().iloc[-1]
            std_price = df['close'].rolling(20).std().iloc[-1]
            
            # Oversold condition
            if price < mean_price - 2.0 * std_price:
                signals.append(TradingSignal(
                    signal_id=f"reversion_buy_{symbol}_{int(current_time.timestamp())}",
                    timestamp=current_time,
                    asset_id=symbol,
                    signal_type=SignalType.REVERSAL,
                    direction=SignalDirection.LONG,
                    strength=SignalStrength.STRONG,
                    confidence=0.75,
                    position_size=0.05,
                    dollar_amount=None,
                    stop_loss=price * 0.98,
                    take_profit=mean_price,
                    max_risk_pct=0.01,
                    volatility_adj=1.0,
                    portfolio_weight=0.05,
                    model_id="mean_reversion_strategy",
                    prediction_value=0.75,
                    features_used={"price": price, "mean_price": mean_price, "std_price": std_price},
                    triggering_event="oversold_condition",
                    event_impact_score=0.7,
                    time_to_event=None,
                    market_regime="mean_reverting",
                    sector="unknown",
                    correlation_risk=0.2
                ))
            
            # Overbought condition
            elif price > mean_price + 2.0 * std_price:
                signals.append(TradingSignal(
                    signal_id=f"reversion_sell_{symbol}_{int(current_time.timestamp())}",
                    timestamp=current_time,
                    asset_id=symbol,
                    signal_type=SignalType.REVERSAL,
                    direction=SignalDirection.SHORT,
                    strength=SignalStrength.STRONG,
                    confidence=0.75,
                    position_size=0.5,
                    dollar_amount=None,
                    stop_loss=price * 1.02,
                    take_profit=mean_price,
                    max_risk_pct=0.01,
                    volatility_adj=1.0,
                    portfolio_weight=0.05,
                    model_id="mean_reversion_strategy",
                    prediction_value=0.75,
                    features_used={"price": price, "mean_price": mean_price, "std_price": std_price},
                    triggering_event="overbought_condition",
                    event_impact_score=0.7,
                    time_to_event=None,
                    market_regime="mean_reverting",
                    sector="unknown",
                    correlation_risk=0.2
                ))
        
        return signals
    
    def _generate_event_signals(
        self, 
        symbol: str, 
        df: pd.DataFrame, 
        events: List, 
        current_time: datetime
    ) -> List[TradingSignal]:
        """Generate event-driven signals"""
        
        signals = []
        
        # Filter events for this symbol
        symbol_events = [e for e in events if hasattr(e, 'symbol') and e.symbol == symbol]
        
        for event in symbol_events:
            # Check if event is recent (within last hour)
            if hasattr(event, 'timestamp'):
                time_diff = abs((current_time - event.timestamp).total_seconds())
                if time_diff < 3600:  # 1 hour
                    
                    # Generate signal based on event type and confidence
                    if hasattr(event, 'confidence') and event.confidence > 0.7:
                        
                        # Positive event
                        if hasattr(event, 'sentiment') and event.sentiment > 0.5:
                            signals.append(TradingSignal(
                                signal_id=f"event_buy_{symbol}_{int(current_time.timestamp())}",
                                timestamp=current_time,
                                asset_id=symbol,
                                signal_type=SignalType.EVENT_DRIVEN,
                                direction=SignalDirection.LONG,
                                strength=SignalStrength.STRONG,
                                confidence=min(0.9, event.confidence + 0.1),
                                position_size=0.08,
                                dollar_amount=None,
                                stop_loss=None,
                                take_profit=None,
                                max_risk_pct=0.02,
                                volatility_adj=1.0,
                                portfolio_weight=0.08,
                                model_id="event_driven_strategy",
                                prediction_value=event.confidence,
                                features_used={"event_confidence": event.confidence},
                                triggering_event=str(event.event_type),
                                event_impact_score=event.confidence,
                                time_to_event=0.0,
                                market_regime="event_driven",
                                sector="unknown",
                                correlation_risk=0.3
                            ))
                        
                        # Negative event
                        elif hasattr(event, 'sentiment') and event.sentiment < -0.5:
                            signals.append(TradingSignal(
                                signal_id=f"event_sell_{symbol}_{int(current_time.timestamp())}",
                                timestamp=current_time,
                                asset_id=symbol,
                                signal_type=SignalType.EVENT_DRIVEN,
                                direction=SignalDirection.SHORT,
                                strength=SignalStrength.STRONG,
                                confidence=min(0.9, event.confidence),
                                position_size=0.5,
                                dollar_amount=None,
                                stop_loss=None,
                                take_profit=None,
                                max_risk_pct=0.02,
                                volatility_adj=1.0,
                                portfolio_weight=0.05,
                                model_id="event_driven_strategy",
                                prediction_value=event.confidence,
                                features_used={"event_confidence": event.confidence},
                                triggering_event=str(event.event_type),
                                event_impact_score=event.confidence,
                                time_to_event=0.0,
                                market_regime="event_driven",
                                sector="unknown",
                                correlation_risk=0.3
                            ))
        
        return signals 