"""
Event Store for storing and retrieving detected events from database
"""

import uuid
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import asdict
import pandas as pd
from loguru import logger

try:
    from sqlalchemy import create_engine, Column, String, Float, DateTime, Text, Integer, Boolean
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import sessionmaker, Session
    from sqlalchemy.dialects.postgresql import UUID
    import sqlalchemy as sa
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    logger.warning("SQLAlchemy not available. EventStore will use in-memory storage.")

from .event_classifier import DetectedEvent, EventType
from .impact_scorer import ImpactScore, ImpactLevel
from .entity_linker import LinkedEntity, EntityType

# Database Models
if SQLALCHEMY_AVAILABLE:
    Base = declarative_base()
    
    class StoredEvent(Base):
        """Database model for stored events"""
        __tablename__ = 'events'
        
        id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
        timestamp = Column(DateTime, default=datetime.utcnow)
        
        # Event Classification
        title = Column(Text, nullable=False)
        description = Column(Text)
        entity = Column(String)
        event_type = Column(String, nullable=False)  # EventType enum value
        classification_confidence = Column(Float)
        
        # Impact Assessment
        impact_level = Column(String)  # ImpactLevel enum value
        impact_score = Column(Float)
        price_target_change = Column(Float)
        volatility_increase = Column(Float)
        time_horizon = Column(String)
        impact_confidence = Column(Float)
        
        # Source Information
        source_url = Column(Text)
        source_name = Column(String)
        published_date = Column(DateTime)
        
        # Processing Metadata
        processed_date = Column(DateTime, default=datetime.utcnow)
        is_backfill = Column(Boolean, default=False)
        
        # Financial Context
        ticker = Column(String)
        sector = Column(String)
        market_cap_category = Column(String)
        
        # Raw Data
        raw_event_data = Column(Text)  # JSON string of original event
        raw_impact_data = Column(Text)  # JSON string of impact assessment
        
    class StoredEntity(Base):
        """Database model for stored entities"""
        __tablename__ = 'entities'
        
        id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
        event_id = Column(String, nullable=False)  # Foreign key to events
        
        text = Column(String, nullable=False)
        entity_type = Column(String, nullable=False)  # EntityType enum value
        confidence = Column(Float)
        ticker = Column(String)
        canonical_name = Column(String)
        start_char = Column(Integer)
        end_char = Column(Integer)
        context = Column(Text)
        aliases = Column(Text)  # JSON string of aliases list


class EventStore:
    """
    Storage and retrieval system for detected events and their associated data
    """
    
    def __init__(self, database_url: Optional[str] = None):
        """
        Initialize EventStore
        
        Args:
            database_url: Database connection URL. If None, uses in-memory storage.
        """
        
        self.use_database = SQLALCHEMY_AVAILABLE and database_url is not None
        
        if self.use_database:
            try:
                self.engine = create_engine(database_url)
                Base.metadata.create_all(self.engine)
                Session = sessionmaker(bind=self.engine)
                self.session_factory = Session
                logger.info(f"EventStore initialized with database: {database_url}")
            except Exception as e:
                logger.error(f"Failed to initialize database: {e}")
                self.use_database = False
        
        if not self.use_database:
            # Fallback to in-memory storage
            self.events = []
            self.entities = []
            logger.info("EventStore initialized with in-memory storage")
    
    def store_event(
        self, 
        event: DetectedEvent, 
        impact: ImpactScore,
        entities: List[LinkedEntity] = None,
        source_info: Dict[str, Any] = None
    ) -> str:
        """
        Store a detected event with its impact assessment and entities
        
        Args:
            event: The detected event
            impact: Impact assessment for the event
            entities: List of entities found in the event
            source_info: Additional source information
            
        Returns:
            Event ID
        """
        
        event_id = str(uuid.uuid4())
        
        if self.use_database:
            return self._store_event_db(event_id, event, impact, entities, source_info)
        else:
            return self._store_event_memory(event_id, event, impact, entities, source_info)
    
    def _store_event_db(
        self, 
        event_id: str,
        event: DetectedEvent, 
        impact: ImpactScore,
        entities: List[LinkedEntity] = None,
        source_info: Dict[str, Any] = None
    ) -> str:
        """Store event in database"""
        
        session = self.session_factory()
        
        try:
            # Prepare source info
            source_info = source_info or {}
            
            # Create stored event
            stored_event = StoredEvent(
                id=event_id,
                title=event.title,
                description=event.description,
                entity=event.entity,
                event_type=event.event_type.value,
                classification_confidence=event.confidence,
                
                impact_level=impact.impact_level.value,
                impact_score=impact.impact_score,
                price_target_change=impact.price_target_change,
                volatility_increase=impact.volatility_increase,
                time_horizon=impact.time_horizon,
                impact_confidence=impact.confidence,
                
                source_url=source_info.get('url'),
                source_name=source_info.get('source'),
                published_date=source_info.get('published_date'),
                
                is_backfill=source_info.get('is_backfill', False),
                
                # Extract ticker and sector from entities
                ticker=self._extract_primary_ticker(entities),
                sector=self._extract_primary_sector(entities),
                
                raw_event_data=str(asdict(event)),
                raw_impact_data=str(asdict(impact))
            )
            
            session.add(stored_event)
            
            # Store associated entities
            if entities:
                for entity in entities:
                    stored_entity = StoredEntity(
                        event_id=event_id,
                        text=entity.text,
                        entity_type=entity.entity_type.value,
                        confidence=entity.confidence,
                        ticker=entity.ticker,
                        canonical_name=entity.canonical_name,
                        start_char=entity.start_char,
                        end_char=entity.end_char,
                        context=entity.context,
                        aliases=str(entity.aliases) if entity.aliases else None
                    )
                    session.add(stored_entity)
            
            session.commit()
            logger.info(f"Stored event {event_id} in database")
            
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to store event in database: {e}")
            raise
        finally:
            session.close()
        
        return event_id
    
    def _store_event_memory(
        self, 
        event_id: str,
        event: DetectedEvent, 
        impact: ImpactScore,
        entities: List[LinkedEntity] = None,
        source_info: Dict[str, Any] = None
    ) -> str:
        """Store event in memory"""
        
        source_info = source_info or {}
        
        event_record = {
            'id': event_id,
            'timestamp': datetime.utcnow(),
            'event': event,
            'impact': impact,
            'entities': entities or [],
            'source_info': source_info,
            'ticker': self._extract_primary_ticker(entities),
            'sector': self._extract_primary_sector(entities)
        }
        
        self.events.append(event_record)
        
        if entities:
            for entity in entities:
                entity_record = {
                    'id': str(uuid.uuid4()),
                    'event_id': event_id,
                    'entity': entity
                }
                self.entities.append(entity_record)
        
        logger.info(f"Stored event {event_id} in memory")
        return event_id
    
    def get_events(
        self,
        event_type: Optional[EventType] = None,
        ticker: Optional[str] = None,
        impact_threshold: Optional[float] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Retrieve events based on filters
        
        Args:
            event_type: Filter by event type
            ticker: Filter by ticker symbol
            impact_threshold: Minimum impact score
            start_date: Filter events after this date
            end_date: Filter events before this date
            limit: Maximum number of events to return
            
        Returns:
            List of event dictionaries
        """
        
        if self.use_database:
            return self._get_events_db(event_type, ticker, impact_threshold, start_date, end_date, limit)
        else:
            return self._get_events_memory(event_type, ticker, impact_threshold, start_date, end_date, limit)
    
    def _get_events_db(
        self,
        event_type: Optional[EventType] = None,
        ticker: Optional[str] = None,
        impact_threshold: Optional[float] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get events from database"""
        
        session = self.session_factory()
        
        try:
            query = session.query(StoredEvent)
            
            # Apply filters
            if event_type:
                query = query.filter(StoredEvent.event_type == event_type.value)
            if ticker:
                query = query.filter(StoredEvent.ticker == ticker)
            if impact_threshold:
                query = query.filter(StoredEvent.impact_score >= impact_threshold)
            if start_date:
                query = query.filter(StoredEvent.timestamp >= start_date)
            if end_date:
                query = query.filter(StoredEvent.timestamp <= end_date)
            
            # Order by timestamp descending and limit
            events = query.order_by(StoredEvent.timestamp.desc()).limit(limit).all()
            
            # Convert to dictionaries
            result = []
            for event in events:
                event_dict = {
                    'id': event.id,
                    'timestamp': event.timestamp,
                    'title': event.title,
                    'description': event.description,
                    'entity': event.entity,
                    'event_type': event.event_type,
                    'classification_confidence': event.classification_confidence,
                    'impact_level': event.impact_level,
                    'impact_score': event.impact_score,
                    'price_target_change': event.price_target_change,
                    'volatility_increase': event.volatility_increase,
                    'ticker': event.ticker,
                    'sector': event.sector,
                    'source_url': event.source_url,
                    'published_date': event.published_date
                }
                result.append(event_dict)
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to retrieve events from database: {e}")
            return []
        finally:
            session.close()
    
    def _get_events_memory(
        self,
        event_type: Optional[EventType] = None,
        ticker: Optional[str] = None,
        impact_threshold: Optional[float] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get events from memory"""
        
        filtered_events = []
        
        for record in self.events:
            # Apply filters
            if event_type and record['event'].event_type != event_type:
                continue
            if ticker and record['ticker'] != ticker:
                continue
            if impact_threshold and record['impact'].impact_score < impact_threshold:
                continue
            if start_date and record['timestamp'] < start_date:
                continue
            if end_date and record['timestamp'] > end_date:
                continue
            
            # Convert to dictionary format
            event_dict = {
                'id': record['id'],
                'timestamp': record['timestamp'],
                'title': record['event'].title,
                'description': record['event'].description,
                'entity': record['event'].entity,
                'event_type': record['event'].event_type.value,
                'classification_confidence': record['event'].confidence,
                'impact_level': record['impact'].impact_level.value,
                'impact_score': record['impact'].impact_score,
                'price_target_change': record['impact'].price_target_change,
                'volatility_increase': record['impact'].volatility_increase,
                'ticker': record['ticker'],
                'sector': record['sector'],
                'source_url': record['source_info'].get('url'),
                'published_date': record['source_info'].get('published_date')
            }
            
            filtered_events.append(event_dict)
        
        # Sort by timestamp descending and limit
        filtered_events.sort(key=lambda x: x['timestamp'], reverse=True)
        return filtered_events[:limit]
    
    def get_event_statistics(self, days: int = 30) -> Dict[str, Any]:
        """Get statistics about stored events"""
        
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        recent_events = self.get_events(start_date=start_date, end_date=end_date, limit=10000)
        
        if not recent_events:
            return {'total_events': 0, 'period_days': days}
        
        # Calculate statistics
        stats = {
            'total_events': len(recent_events),
            'period_days': days,
            'avg_impact_score': sum(e['impact_score'] for e in recent_events) / len(recent_events),
            'event_type_distribution': {},
            'impact_level_distribution': {},
            'top_tickers': {},
            'top_sectors': {},
            'high_impact_events': len([e for e in recent_events if e['impact_score'] >= 0.6])
        }
        
        # Event type distribution
        for event in recent_events:
            event_type = event['event_type']
            stats['event_type_distribution'][event_type] = stats['event_type_distribution'].get(event_type, 0) + 1
        
        # Impact level distribution
        for event in recent_events:
            impact_level = event['impact_level']
            stats['impact_level_distribution'][impact_level] = stats['impact_level_distribution'].get(impact_level, 0) + 1
        
        # Top tickers
        for event in recent_events:
            if event['ticker']:
                stats['top_tickers'][event['ticker']] = stats['top_tickers'].get(event['ticker'], 0) + 1
        
        # Top sectors
        for event in recent_events:
            if event['sector']:
                stats['top_sectors'][event['sector']] = stats['top_sectors'].get(event['sector'], 0) + 1
        
        return stats
    
    def to_dataframe(self, **filters) -> pd.DataFrame:
        """Convert stored events to DataFrame"""
        
        events = self.get_events(**filters)
        if not events:
            return pd.DataFrame()
        
        return pd.DataFrame(events)
    
    def _extract_primary_ticker(self, entities: List[LinkedEntity]) -> Optional[str]:
        """Extract the primary ticker from entities"""
        
        if not entities:
            return None
        
        # Look for ticker entities first
        for entity in entities:
            if entity.entity_type == EntityType.TICKER and entity.ticker:
                return entity.ticker
        
        # Look for companies with tickers
        for entity in entities:
            if entity.ticker:
                return entity.ticker
        
        return None
    
    def _extract_primary_sector(self, entities: List[LinkedEntity]) -> Optional[str]:
        """Extract the primary sector from entities"""
        
        if not entities:
            return None
        
        for entity in entities:
            if entity.aliases:
                for alias in entity.aliases:
                    if alias.startswith('sector:'):
                        return alias.replace('sector:', '')
        
        return None 