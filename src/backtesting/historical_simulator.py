"""
Historical Event Simulator

Replays historical market data, news events, and trading scenarios
with realistic timing and data availability constraints.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Iterator, Tuple, Any
from dataclasses import dataclass
from loguru import logger
import asyncio

from ..data_ingestion.market_data import MarketDataIngestor
from ..data_ingestion.news import NewsIngestor
from ..event_detection.event_classifier import EventClassifier, DetectedEvent
from ..feature_engineering.main import FeatureEngineer


@dataclass
class SimulationEvent:
    """Represents a single event in the historical simulation"""
    timestamp: datetime
    event_type: str  # 'market_data', 'news', 'event', 'signal'
    data: Any
    symbol: Optional[str] = None


@dataclass
class SimulationState:
    """Current state of the simulation"""
    current_time: datetime
    available_market_data: Dict[str, pd.DataFrame]
    available_news: List[Dict]
    detected_events: List[DetectedEvent]
    features: Dict[str, Any]


class HistoricalEventSimulator:
    """
    Simulates historical trading scenarios by replaying market data,
    news events, and trading signals in chronological order.
    """
    
    def __init__(
        self,
        start_date: datetime,
        end_date: datetime,
        symbols: List[str],
        lookback_days: int = 30,
        time_resolution: str = '1min'  # '1min', '5min', '1h', '1d'
    ):
        self.start_date = start_date
        self.end_date = end_date
        self.symbols = symbols
        self.lookback_days = lookback_days
        self.time_resolution = time_resolution
        
        # Simulation state
        self.current_time = start_date
        self.simulation_events: List[SimulationEvent] = []
        self.state = SimulationState(
            current_time=start_date,
            available_market_data={},
            available_news=[],
            detected_events=[],
            features={}
        )
        
        # Components
        self.event_classifier = EventClassifier()
        self.feature_engineer = FeatureEngineer()
        
        # Historical data storage
        self.historical_market_data: Dict[str, pd.DataFrame] = {}
        self.historical_news: List[Dict] = []
        self.preprocessed_events: List[SimulationEvent] = []
        
        logger.info(f"Initialized HistoricalEventSimulator: {start_date} to {end_date}")
    
    async def load_historical_data(
        self,
        market_data_source: Optional[MarketDataIngestor] = None,
        news_data_source: Optional[NewsIngestor] = None,
        data_directory: Optional[str] = None
    ) -> None:
        """Load historical market data and news for the simulation period"""
        
        logger.info("Loading historical data...")
        
        # Load market data
        if market_data_source:
            await self._load_market_data_from_api(market_data_source)
        elif data_directory:
            self._load_market_data_from_files(data_directory)
        else:
            logger.warning("No market data source provided")
        
        # Load news data
        if news_data_source:
            await self._load_news_data_from_api(news_data_source)
        elif data_directory:
            self._load_news_data_from_files(data_directory)
        else:
            logger.warning("No news data source provided")
        
        # Preprocess and sort all events chronologically
        self._preprocess_events()
        
        logger.info(f"Loaded {len(self.preprocessed_events)} simulation events")
    
    async def _load_market_data_from_api(self, market_data_source: MarketDataIngestor) -> None:
        """Load historical market data from API"""
        
        from alpaca.data.timeframe import TimeFrame
        
        # Map time resolution to Alpaca TimeFrame
        timeframe_map = {
            '1min': TimeFrame.Minute,
            '5min': TimeFrame(5, TimeFrame.Unit.Minute),
            '1h': TimeFrame.Hour,
            '1d': TimeFrame.Day
        }
        
        timeframe = timeframe_map.get(self.time_resolution, TimeFrame.Minute)
        
        for symbol in self.symbols:
            try:
                # Calculate total days needed (including lookback)
                total_days = (self.end_date - self.start_date).days + self.lookback_days
                
                data = await market_data_source.get_latest_bars(
                    symbols=[symbol],
                    timeframe=timeframe,
                    limit=min(total_days * 390, 10000)  # Market hours approximation
                )
                
                if symbol in data and not data[symbol].empty:
                    # Filter to our date range (keeping some lookback)
                    lookback_start = self.start_date - timedelta(days=self.lookback_days)
                    mask = (data[symbol].index >= lookback_start) & (data[symbol].index <= self.end_date)
                    self.historical_market_data[symbol] = data[symbol][mask].copy()
                    
                    logger.info(f"Loaded {len(self.historical_market_data[symbol])} bars for {symbol}")
                
            except Exception as e:
                logger.error(f"Failed to load market data for {symbol}: {e}")
    
    def _load_market_data_from_files(self, data_directory: str) -> None:
        """Load historical market data from CSV files"""
        
        import os
        
        for symbol in self.symbols:
            file_path = os.path.join(data_directory, f"{symbol}_market_data.csv")
            
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                    
                    # Filter to our date range
                    lookback_start = self.start_date - timedelta(days=self.lookback_days)
                    mask = (df.index >= lookback_start) & (df.index <= self.end_date)
                    self.historical_market_data[symbol] = df[mask].copy()
                    
                    logger.info(f"Loaded {len(self.historical_market_data[symbol])} bars for {symbol} from file")
                    
                except Exception as e:
                    logger.error(f"Failed to load market data file for {symbol}: {e}")
    
    async def _load_news_data_from_api(self, news_data_source: NewsIngestor) -> None:
        """Load historical news data from API"""
        
        # Calculate date range for news collection
        news_start = self.start_date - timedelta(days=self.lookback_days)
        days_total = (self.end_date - news_start).days
        
        try:
            # Get market news
            market_news = news_data_source.get_market_news(days_back=days_total)
            if not market_news.empty:
                self.historical_news.extend(market_news.to_dict('records'))
            
            # Get company-specific news
            for symbol in self.symbols:
                try:
                    company_news = news_data_source.get_company_news(symbol, days_back=days_total)
                    if not company_news.empty:
                        # Add symbol information
                        company_news['target_symbol'] = symbol
                        self.historical_news.extend(company_news.to_dict('records'))
                
                except Exception as e:
                    logger.warning(f"Failed to load news for {symbol}: {e}")
            
            logger.info(f"Loaded {len(self.historical_news)} news articles")
            
        except Exception as e:
            logger.error(f"Failed to load news data: {e}")
    
    def _load_news_data_from_files(self, data_directory: str) -> None:
        """Load historical news data from CSV files"""
        
        import os
        
        news_file = os.path.join(data_directory, "historical_news.csv")
        
        if os.path.exists(news_file):
            try:
                df = pd.read_csv(news_file, parse_dates=['published_at'])
                
                # Filter to our date range
                lookback_start = self.start_date - timedelta(days=self.lookback_days)
                mask = (df['published_at'] >= lookback_start) & (df['published_at'] <= self.end_date)
                filtered_news = df[mask]
                
                self.historical_news = filtered_news.to_dict('records')
                logger.info(f"Loaded {len(self.historical_news)} news articles from file")
                
            except Exception as e:
                logger.error(f"Failed to load news data file: {e}")
    
    def _preprocess_events(self) -> None:
        """Preprocess and chronologically sort all historical events"""
        
        events = []
        
        # Add market data events
        for symbol, data in self.historical_market_data.items():
            for timestamp, row in data.iterrows():
                events.append(SimulationEvent(
                    timestamp=timestamp,
                    event_type='market_data',
                    data=row.to_dict(),
                    symbol=symbol
                ))
        
        # Add news events and classify them
        for news_item in self.historical_news:
            timestamp = pd.to_datetime(news_item['published_at'])
            
            # Create news event
            events.append(SimulationEvent(
                timestamp=timestamp,
                event_type='news',
                data=news_item,
                symbol=news_item.get('target_symbol')
            ))
            
            # Classify news and create event classification
            try:
                detected_events = self.event_classifier.classify_text(
                    title=news_item.get('title', ''),
                    content=news_item.get('description', '')
                )
                
                for detected_event in detected_events:
                    events.append(SimulationEvent(
                        timestamp=timestamp + timedelta(seconds=1),  # Slight delay for classification
                        event_type='event_detection',
                        data=detected_event,
                        symbol=news_item.get('target_symbol')
                    ))
                    
            except Exception as e:
                logger.warning(f"Failed to classify news: {e}")
        
        # Sort events chronologically
        self.preprocessed_events = sorted(events, key=lambda x: x.timestamp)
        
        logger.info(f"Preprocessed {len(self.preprocessed_events)} total simulation events")
    
    def simulate(self) -> Iterator[Tuple[datetime, SimulationState]]:
        """
        Run the historical simulation, yielding state at each time step
        
        Yields:
            Tuple of (current_time, simulation_state)
        """
        
        if not self.preprocessed_events:
            logger.error("No historical data loaded. Call load_historical_data() first.")
            return
        
        logger.info(f"Starting simulation from {self.start_date} to {self.end_date}")
        
        event_index = 0
        self.current_time = self.start_date
        
        # Time step based on resolution
        time_step_map = {
            '1min': timedelta(minutes=1),
            '5min': timedelta(minutes=5),
            '1h': timedelta(hours=1),
            '1d': timedelta(days=1)
        }
        time_step = time_step_map.get(self.time_resolution, timedelta(minutes=1))
        
        while self.current_time <= self.end_date:
            # Process all events that occur at or before current time
            while (event_index < len(self.preprocessed_events) and 
                   self.preprocessed_events[event_index].timestamp <= self.current_time):
                
                event = self.preprocessed_events[event_index]
                self._process_simulation_event(event)
                event_index += 1
            
            # Update simulation state
            self.state.current_time = self.current_time
            
            # Generate features if we have enough data
            if self._has_sufficient_data():
                self._update_features()
            
            # Yield current state
            yield self.current_time, self.state
            
            # Advance time
            self.current_time += time_step
        
        logger.info("Simulation completed")
    
    def _process_simulation_event(self, event: SimulationEvent) -> None:
        """Process a single simulation event and update state"""
        
        if event.event_type == 'market_data':
            # Update available market data
            symbol = event.symbol
            if symbol not in self.state.available_market_data:
                self.state.available_market_data[symbol] = pd.DataFrame()
            
            # Add new market data row
            new_row = pd.DataFrame([event.data], index=[event.timestamp])
            self.state.available_market_data[symbol] = pd.concat([
                self.state.available_market_data[symbol], 
                new_row
            ])
            
            # Keep only recent data (sliding window)
            cutoff_time = event.timestamp - timedelta(days=self.lookback_days)
            self.state.available_market_data[symbol] = self.state.available_market_data[symbol][
                self.state.available_market_data[symbol].index >= cutoff_time
            ]
        
        elif event.event_type == 'news':
            # Add news to available news
            self.state.available_news.append(event.data)
            
            # Keep only recent news
            cutoff_time = event.timestamp - timedelta(days=self.lookback_days)
            self.state.available_news = [
                news for news in self.state.available_news
                if pd.to_datetime(news['published_at']) >= cutoff_time
            ]
        
        elif event.event_type == 'event_detection':
            # Add detected event
            self.state.detected_events.append(event.data)
            
            # Keep only recent events
            cutoff_time = event.timestamp - timedelta(days=self.lookback_days)
            self.state.detected_events = [
                evt for evt in self.state.detected_events
                if evt.timestamp >= cutoff_time
            ]
    
    def _has_sufficient_data(self) -> bool:
        """Check if we have enough data to generate meaningful features"""
        
        min_data_points = 20  # Minimum data points per symbol
        
        for symbol in self.symbols:
            if (symbol not in self.state.available_market_data or 
                len(self.state.available_market_data[symbol]) < min_data_points):
                return False
        
        return True
    
    def _update_features(self) -> None:
        """Update feature state based on currently available data"""
        
        try:
            # Convert current state to format expected by feature engineer
            market_data = self.state.available_market_data
            news_data = pd.DataFrame(self.state.available_news) if self.state.available_news else pd.DataFrame()
            
            # Generate features
            features = self.feature_engineer.generate_features(
                market_data=market_data,
                news_data=news_data,
                events=self.state.detected_events,
                current_time=self.current_time
            )
            
            self.state.features = features
            
        except Exception as e:
            logger.warning(f"Failed to update features: {e}")
    
    def get_available_data_at_time(self, timestamp: datetime, symbol: str) -> Optional[pd.DataFrame]:
        """Get market data available at a specific timestamp (look-ahead bias prevention)"""
        
        if symbol not in self.historical_market_data:
            return None
        
        # Only return data that would have been available at that time
        available_data = self.historical_market_data[symbol][
            self.historical_market_data[symbol].index <= timestamp
        ]
        
        if available_data.empty:
            return None
        
        return available_data.tail(1000)  # Return last 1000 data points
    
    def get_news_at_time(self, timestamp: datetime, hours_lookback: int = 24) -> List[Dict]:
        """Get news available at a specific timestamp"""
        
        cutoff_time = timestamp - timedelta(hours=hours_lookback)
        
        available_news = [
            news for news in self.historical_news
            if cutoff_time <= pd.to_datetime(news['published_at']) <= timestamp
        ]
        
        return available_news
    
    def reset(self) -> None:
        """Reset simulation to initial state"""
        
        self.current_time = self.start_date
        self.state = SimulationState(
            current_time=self.start_date,
            available_market_data={},
            available_news=[],
            detected_events=[],
            features={}
        )
        
        logger.info("Simulation reset to initial state")
    
    def export_state_history(self, filepath: str) -> None:
        """Export complete simulation state history for analysis"""
        
        # This would be implemented to save detailed state history
        # for post-simulation analysis
        pass 