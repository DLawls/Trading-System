"""
Backtesting & Evaluation Module

This module provides comprehensive backtesting capabilities for the trading system,
including historical event simulation, portfolio modeling, and performance analytics.
"""

from .historical_simulator import HistoricalEventSimulator
from .portfolio_simulator import PortfolioSimulator
from .metrics_logger import MetricsLogger
from .backtest_engine import BacktestEngine

__all__ = [
    'HistoricalEventSimulator',
    'PortfolioSimulator', 
    'MetricsLogger',
    'BacktestEngine'
] 