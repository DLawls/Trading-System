"""
Signal Generation Module

This module contains components for converting ML predictions into 
actionable trading signals with proper risk management.
"""

from .signal_schema import TradingSignal, SignalType, SignalDirection, SignalStrength, PortfolioSignal
from .signal_evaluator import SignalEvaluator, EvaluationConfig
from .position_sizer import PositionSizer, PositionSizingConfig
from .portfolio_allocator import PortfolioAllocator, AllocationConfig

__all__ = [
    'TradingSignal',
    'SignalType',
    'SignalDirection',
    'SignalStrength',
    'PortfolioSignal',
    'SignalEvaluator',
    'EvaluationConfig',
    'PositionSizer',
    'PositionSizingConfig',
    'PortfolioAllocator',
    'AllocationConfig'
] 