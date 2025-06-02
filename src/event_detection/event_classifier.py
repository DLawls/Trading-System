"""
Event Classifier for detecting trading-relevant events from news
"""

import re
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import pandas as pd
from loguru import logger
from datetime import datetime


class EventType(Enum):
    """Types of trading events we can detect"""
    EARNINGS = "earnings"
    MERGER_ACQUISITION = "merger_acquisition"
    PRODUCT_LAUNCH = "product_launch" 
    REGULATORY = "regulatory"
    LEADERSHIP_CHANGE = "leadership_change"
    PARTNERSHIP = "partnership"
    LEGAL_ISSUE = "legal_issue"
    MARKET_MOVEMENT = "market_movement"
    ECONOMIC_DATA = "economic_data"
    UNKNOWN = "unknown"


@dataclass
class DetectedEvent:
    """Represents a detected event from news"""
    event_type: EventType
    confidence: float  # 0.0 to 1.0
    entity: str  # Company/symbol
    title: str
    description: str
    keywords_matched: List[str]
    sentiment: Optional[str] = None
    timestamp: Optional[datetime] = None  # When the event was detected/published
    metadata: Dict[str, Any] = None


class EventClassifier:
    """
    Rule-based classifier to detect trading events from news
    """
    
    def __init__(self):
        self.event_patterns = self._initialize_patterns()
    
    def _initialize_patterns(self) -> Dict[EventType, Dict[str, Any]]:
        """Initialize keyword patterns for each event type"""
        
        patterns = {
            EventType.EARNINGS: {
                'keywords': [
                    'earnings', 'quarterly results', 'financial results', 
                    'revenue', 'profit', 'eps', 'earnings per share',
                    'quarterly report', 'fiscal quarter', 'beats estimates',
                    'misses estimates', 'guidance', 'outlook'
                ],
                'weight': 1.0,
                'required_terms': ['earnings', 'quarter', 'revenue']
            },
            
            EventType.MERGER_ACQUISITION: {
                'keywords': [
                    'merger', 'acquisition', 'buyout', 'takeover',
                    'acquires', 'merges with', 'deal', 'purchase',
                    'acquisition agreement', 'merger agreement',
                    'bought by', 'sells to'
                ],
                'weight': 1.0,
                'required_terms': ['acquisition', 'merger', 'deal']
            },
            
            EventType.PRODUCT_LAUNCH: {
                'keywords': [
                    'launches', 'introduces', 'announces new', 'unveils',
                    'product release', 'new product', 'innovation',
                    'breakthrough', 'patent', 'technology'
                ],
                'weight': 0.8,
                'required_terms': ['new', 'product', 'launch']
            },
            
            EventType.REGULATORY: {
                'keywords': [
                    'fda approval', 'regulatory approval', 'sec', 'investigation',
                    'lawsuit', 'legal action', 'court', 'ruling',
                    'compliance', 'violation', 'fine', 'penalty',
                    'regulatory filing', 'antitrust'
                ],
                'weight': 0.9,
                'required_terms': ['regulatory', 'fda', 'sec', 'legal']
            },
            
            EventType.LEADERSHIP_CHANGE: {
                'keywords': [
                    'ceo', 'chief executive', 'president', 'chairman',
                    'resigns', 'steps down', 'appointed', 'hired',
                    'executive', 'management change', 'leadership',
                    'board of directors'
                ],
                'weight': 0.7,
                'required_terms': ['ceo', 'executive', 'management']
            },
            
            EventType.PARTNERSHIP: {
                'keywords': [
                    'partnership', 'collaboration', 'joint venture',
                    'alliance', 'agreement', 'contract', 'deal',
                    'partners with', 'teams up', 'cooperation'
                ],
                'weight': 0.6,
                'required_terms': ['partnership', 'agreement', 'collaboration']
            },
            
            EventType.LEGAL_ISSUE: {
                'keywords': [
                    'lawsuit', 'litigation', 'court case', 'legal battle',
                    'settlement', 'damages', 'trial', 'judge',
                    'jury', 'verdict', 'appeal', 'injunction'
                ],
                'weight': 0.8,
                'required_terms': ['lawsuit', 'court', 'legal']
            },
            
            EventType.MARKET_MOVEMENT: {
                'keywords': [
                    'stock price', 'shares', 'market cap', 'valuation',
                    'trading', 'volume', 'price target', 'upgrade',
                    'downgrade', 'analyst', 'recommendation'
                ],
                'weight': 0.5,
                'required_terms': ['stock', 'price', 'trading']
            },
            
            EventType.ECONOMIC_DATA: {
                'keywords': [
                    'gdp', 'inflation', 'unemployment', 'interest rate',
                    'federal reserve', 'fed', 'economic data',
                    'consumer price index', 'cpi', 'jobs report'
                ],
                'weight': 0.9,
                'required_terms': ['economic', 'gdp', 'inflation', 'fed']
            }
        }
        
        return patterns
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract potential company/stock entities from text"""
        # Simple entity extraction - look for uppercase words that could be tickers
        # In production, you'd use a proper NER model
        
        entities = []
        
        # Look for stock ticker patterns (2-5 uppercase letters)
        ticker_pattern = r'\b[A-Z]{2,5}\b'
        tickers = re.findall(ticker_pattern, text)
        entities.extend(tickers)
        
        # Look for common company name patterns
        company_pattern = r'\b([A-Z][a-z]+ (?:Inc|Corp|LLC|Ltd|Company|Technologies|Systems|Group))\b'
        companies = re.findall(company_pattern, text)
        entities.extend(companies)
        
        return list(set(entities))  # Remove duplicates
    
    def _calculate_confidence(
        self, 
        matched_keywords: List[str], 
        event_type: EventType,
        text: str
    ) -> float:
        """Calculate confidence score for event detection"""
        
        pattern_info = self.event_patterns[event_type]
        total_keywords = len(pattern_info['keywords'])
        matched_count = len(matched_keywords)
        
        # Base confidence from keyword matching
        keyword_confidence = (matched_count / total_keywords) * pattern_info['weight']
        
        # Bonus for required terms
        required_terms = pattern_info.get('required_terms', [])
        text_lower = text.lower()
        required_found = sum(1 for term in required_terms if term in text_lower)
        required_bonus = (required_found / len(required_terms)) * 0.3 if required_terms else 0
        
        # Length penalty - longer texts might have coincidental matches
        length_penalty = min(0.1, len(text) / 10000)
        
        confidence = min(1.0, keyword_confidence + required_bonus - length_penalty)
        return confidence
    
    def classify_text(self, title: str, content: str = "") -> List[DetectedEvent]:
        """
        Classify a news article and detect events
        
        Args:
            title: News article title
            content: News article content (optional)
            
        Returns:
            List of detected events with confidence scores
        """
        
        full_text = f"{title} {content}".lower()
        detected_events = []
        
        # Extract entities first
        entities = self._extract_entities(title + " " + content)
        primary_entity = entities[0] if entities else "Unknown"
        
        # Check each event type
        for event_type, pattern_info in self.event_patterns.items():
            matched_keywords = []
            
            # Find matching keywords
            for keyword in pattern_info['keywords']:
                if keyword.lower() in full_text:
                    matched_keywords.append(keyword)
            
            # If we have matches, create an event
            if matched_keywords:
                confidence = self._calculate_confidence(
                    matched_keywords, event_type, full_text
                )
                
                # Only include events with reasonable confidence
                if confidence > 0.3:
                    event = DetectedEvent(
                        event_type=event_type,
                        confidence=confidence,
                        entity=primary_entity,
                        title=title,
                        description=content[:200] + "..." if len(content) > 200 else content,
                        keywords_matched=matched_keywords,
                        timestamp=datetime.now(),
                        metadata={
                            'all_entities': entities,
                            'text_length': len(full_text)
                        }
                    )
                    detected_events.append(event)
        
        # Sort by confidence (highest first)
        detected_events.sort(key=lambda x: x.confidence, reverse=True)
        
        return detected_events
    
    def classify_news_df(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify events for a DataFrame of news articles
        
        Args:
            news_df: DataFrame with 'title' and optionally 'content' columns
            
        Returns:
            DataFrame with additional event classification columns
        """
        
        if news_df.empty:
            return news_df
        
        results = []
        
        for idx, row in news_df.iterrows():
            title = row.get('title', '')
            content = row.get('content', '') or row.get('description', '')
            
            events = self.classify_text(title, content)
            
            if events:
                # Take the highest confidence event
                primary_event = events[0]
                results.append({
                    'event_type': primary_event.event_type.value,
                    'event_confidence': primary_event.confidence,
                    'event_entity': primary_event.entity,
                    'keywords_matched': ', '.join(primary_event.keywords_matched),
                    'num_events_detected': len(events)
                })
            else:
                results.append({
                    'event_type': EventType.UNKNOWN.value,
                    'event_confidence': 0.0,
                    'event_entity': '',
                    'keywords_matched': '',
                    'num_events_detected': 0
                })
        
        # Add results to original DataFrame
        result_df = news_df.copy()
        for key in results[0].keys():
            result_df[key] = [r[key] for r in results]
        
        logger.info(f"Classified {len(news_df)} news articles, found events in {sum(1 for r in results if r['event_confidence'] > 0)} articles")
        
        return result_df 