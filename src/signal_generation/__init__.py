"""
Signal Generation Module

This module contains components for converting ML predictions into 
actionable trading signals with proper risk management.
"""

from .signal_evaluator import SignalEvaluator
from .position_sizer import PositionSizer  
from .portfolio_allocator import PortfolioAllocator
from .signal_schema import TradingSignal, SignalType, SignalDirection

__all__ = [
    'SignalEvaluator',
    'PositionSizer', 
    'PortfolioAllocator',
    'TradingSignal',
    'SignalType',
    'SignalDirection'
] 