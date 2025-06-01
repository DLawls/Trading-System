"""
Historical Event Analyzer for backtesting preparation
"""

import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from loguru import logger

from .event_classifier import EventClassifier, EventType
from .impact_scorer import ImpactScorer
from .entity_linker import EntityLinker
from .event_store import EventStore


class HistoricalAnalyzer:
    """
    Analyzes historical news data to detect events for backtesting
    """
    
    def __init__(self, event_store: Optional[EventStore] = None):
        """Initialize the historical analyzer"""
        
        self.classifier = EventClassifier()
        self.impact_scorer = ImpactScorer()
        self.entity_linker = EntityLinker()
        self.event_store = event_store or EventStore()
        
        # Processing statistics
        self.stats = {
            'total_articles': 0,
            'events_detected': 0,
            'high_impact_events': 0,
            'processing_errors': 0,
            'start_time': None,
            'end_time': None
        }
    
    def analyze_historical_data(
        self,
        news_data: pd.DataFrame,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        min_impact_threshold: float = 0.3,
        store_events: bool = True,
        batch_size: int = 100
    ) -> Dict[str, Any]:
        """
        Analyze historical news data to detect and store events
        
        Args:
            news_data: DataFrame with columns ['title', 'content', 'published_date', 'source', 'url']
            start_date: Start date for analysis (optional)
            end_date: End date for analysis (optional)
            min_impact_threshold: Minimum impact score to store events
            store_events: Whether to store detected events
            batch_size: Number of articles to process in each batch
            
        Returns:
            Analysis results and statistics
        """
        
        logger.info("Starting historical event analysis...")
        self.stats['start_time'] = datetime.utcnow()
        
        # Filter data by date range if specified
        filtered_data = self._filter_by_date_range(news_data, start_date, end_date)
        self.stats['total_articles'] = len(filtered_data)
        
        logger.info(f"Processing {len(filtered_data)} historical articles...")
        
        # Process articles in batches
        all_events = []
        for i in range(0, len(filtered_data), batch_size):
            batch = filtered_data.iloc[i:i+batch_size]
            batch_events = self._process_batch(batch, min_impact_threshold, store_events)
            all_events.extend(batch_events)
            
            # Log progress
            progress = min(i + batch_size, len(filtered_data))
            logger.info(f"Processed {progress}/{len(filtered_data)} articles "
                       f"({progress/len(filtered_data)*100:.1f}%)")
        
        self.stats['end_time'] = datetime.utcnow()
        processing_time = self.stats['end_time'] - self.stats['start_time']
        
        # Compile results
        results = {
            'events_detected': all_events,
            'statistics': self._compile_statistics(all_events),
            'processing_stats': {
                **self.stats,
                'processing_time_seconds': processing_time.total_seconds(),
                'articles_per_second': len(filtered_data) / processing_time.total_seconds() if processing_time.total_seconds() > 0 else 0
            }
        }
        
        logger.info(f"Historical analysis complete. Detected {len(all_events)} events "
                   f"in {processing_time.total_seconds():.1f} seconds")
        
        return results
    
    def _filter_by_date_range(
        self,
        data: pd.DataFrame,
        start_date: Optional[datetime],
        end_date: Optional[datetime]
    ) -> pd.DataFrame:
        """Filter DataFrame by date range"""
        
        if 'published_date' not in data.columns:
            logger.warning("No 'published_date' column found. Skipping date filtering.")
            return data
        
        filtered = data.copy()
        
        if start_date:
            filtered = filtered[pd.to_datetime(filtered['published_date']) >= start_date]
        
        if end_date:
            filtered = filtered[pd.to_datetime(filtered['published_date']) <= end_date]
        
        return filtered
    
    def _process_batch(
        self,
        batch: pd.DataFrame,
        min_impact_threshold: float,
        store_events: bool
    ) -> List[Dict[str, Any]]:
        """Process a batch of articles"""
        
        batch_events = []
        
        for idx, row in batch.iterrows():
            try:
                # Extract text content
                title = str(row.get('title', ''))
                content = str(row.get('content', ''))
                
                if not title and not content:
                    continue
                
                # Classify events
                detected_events = self.classifier.classify_text(title, content)
                
                if not detected_events:
                    continue
                
                # Process each detected event
                for event in detected_events:
                    # Extract entities
                    full_text = f"{title} {content}"
                    entities = self.entity_linker.link_entities(full_text)
                    
                    # Score impact
                    impact = self.impact_scorer.score_event_impact(event)
                    
                    # Filter by impact threshold
                    if impact.impact_score >= min_impact_threshold:
                        # Prepare source info
                        source_info = {
                            'url': row.get('url'),
                            'source': row.get('source'),
                            'published_date': pd.to_datetime(row.get('published_date')) if 'published_date' in row else None,
                            'is_backfill': True
                        }
                        
                        # Store event if requested
                        event_id = None
                        if store_events:
                            event_id = self.event_store.store_event(
                                event, impact, entities, source_info
                            )
                        
                        # Compile event data
                        event_data = {
                            'id': event_id,
                            'event': event,
                            'impact': impact,
                            'entities': entities,
                            'source_info': source_info,
                            'article_index': idx
                        }
                        
                        batch_events.append(event_data)
                        self.stats['events_detected'] += 1
                        
                        if impact.impact_score >= 0.6:
                            self.stats['high_impact_events'] += 1
                
            except Exception as e:
                logger.error(f"Error processing article {idx}: {e}")
                self.stats['processing_errors'] += 1
        
        return batch_events
    
    def _compile_statistics(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compile analysis statistics"""
        
        if not events:
            return {
                'total_events': 0,
                'avg_impact_score': 0,
                'event_type_distribution': {},
                'impact_level_distribution': {},
                'ticker_distribution': {},
                'sector_distribution': {}
            }
        
        # Basic stats
        impact_scores = [e['impact'].impact_score for e in events]
        avg_impact = sum(impact_scores) / len(impact_scores)
        
        # Event type distribution
        event_types = {}
        for event in events:
            event_type = event['event'].event_type.value
            event_types[event_type] = event_types.get(event_type, 0) + 1
        
        # Impact level distribution
        impact_levels = {}
        for event in events:
            impact_level = event['impact'].impact_level.value
            impact_levels[impact_level] = impact_levels.get(impact_level, 0) + 1
        
        # Ticker distribution
        tickers = {}
        for event in events:
            primary_ticker = self._extract_primary_ticker(event['entities'])
            if primary_ticker:
                tickers[primary_ticker] = tickers.get(primary_ticker, 0) + 1
        
        # Sector distribution
        sectors = {}
        for event in events:
            primary_sector = self._extract_primary_sector(event['entities'])
            if primary_sector:
                sectors[primary_sector] = sectors.get(primary_sector, 0) + 1
        
        return {
            'total_events': len(events),
            'avg_impact_score': avg_impact,
            'max_impact_score': max(impact_scores),
            'min_impact_score': min(impact_scores),
            'high_impact_events': len([e for e in events if e['impact'].impact_score >= 0.6]),
            'event_type_distribution': dict(sorted(event_types.items(), key=lambda x: x[1], reverse=True)),
            'impact_level_distribution': dict(sorted(impact_levels.items(), key=lambda x: x[1], reverse=True)),
            'ticker_distribution': dict(sorted(tickers.items(), key=lambda x: x[1], reverse=True)[:20]),  # Top 20
            'sector_distribution': dict(sorted(sectors.items(), key=lambda x: x[1], reverse=True))
        }
    
    def analyze_event_timeline(
        self,
        ticker: str,
        days: int = 30,
        min_impact: float = 0.4
    ) -> pd.DataFrame:
        """
        Analyze event timeline for a specific ticker
        
        Args:
            ticker: Stock ticker to analyze
            days: Number of days to look back
            min_impact: Minimum impact score to include
            
        Returns:
            DataFrame with event timeline
        """
        
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        events = self.event_store.get_events(
            ticker=ticker,
            impact_threshold=min_impact,
            start_date=start_date,
            end_date=end_date,
            limit=1000
        )
        
        if not events:
            return pd.DataFrame()
        
        # Convert to DataFrame and add time-based features
        df = pd.DataFrame(events)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date'] = df['timestamp'].dt.date
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.day_name()
        
        return df.sort_values('timestamp')
    
    def get_market_moving_events(
        self,
        days: int = 7,
        min_impact: float = 0.7,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get recent high-impact market-moving events"""
        
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        events = self.event_store.get_events(
            impact_threshold=min_impact,
            start_date=start_date,
            end_date=end_date,
            limit=limit
        )
        
        # Sort by impact score descending
        return sorted(events, key=lambda x: x['impact_score'], reverse=True)
    
    def generate_backtest_dataset(
        self,
        start_date: datetime,
        end_date: datetime,
        tickers: List[str] = None,
        min_impact: float = 0.4
    ) -> pd.DataFrame:
        """
        Generate a clean dataset for backtesting
        
        Args:
            start_date: Start date for dataset
            end_date: End date for dataset
            tickers: List of tickers to include (None for all)
            min_impact: Minimum impact score to include
            
        Returns:
            DataFrame ready for backtesting
        """
        
        events = self.event_store.get_events(
            impact_threshold=min_impact,
            start_date=start_date,
            end_date=end_date,
            limit=10000
        )
        
        if not events:
            return pd.DataFrame()
        
        df = pd.DataFrame(events)
        
        # Filter by tickers if specified
        if tickers:
            df = df[df['ticker'].isin(tickers)]
        
        # Add features for backtesting
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date'] = df['timestamp'].dt.date
        df['time'] = df['timestamp'].dt.time
        df['is_market_hours'] = df['timestamp'].dt.hour.between(9, 16)
        df['is_weekend'] = df['timestamp'].dt.dayofweek.isin([5, 6])
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df
    
    def _extract_primary_ticker(self, entities) -> Optional[str]:
        """Extract primary ticker from entities"""
        if not entities:
            return None
        
        for entity in entities:
            if hasattr(entity, 'ticker') and entity.ticker:
                return entity.ticker
        
        return None
    
    def _extract_primary_sector(self, entities) -> Optional[str]:
        """Extract primary sector from entities"""
        if not entities:
            return None
        
        for entity in entities:
            if hasattr(entity, 'aliases') and entity.aliases:
                for alias in entity.aliases:
                    if alias.startswith('sector:'):
                        return alias.replace('sector:', '')
        
        return None 