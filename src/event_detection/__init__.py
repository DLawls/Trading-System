"""
Event Detection Module

This module contains components for detecting and classifying
trading-relevant events from news and market data.
"""

from .event_classifier import EventClassifier
from .impact_scorer import ImpactScorer
from .entity_linker import EntityLinker
from .event_store import EventStore
from .historical_analyzer import HistoricalAnalyzer
# from .entity_linker import EntityLinker  # Coming next 