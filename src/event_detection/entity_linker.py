"""
Entity Linker for identifying and linking entities in news text to financial instruments
"""

import re
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import pandas as pd
from loguru import logger

try:
    import spacy
    from spacy import displacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logger.warning("spaCy not available. Using basic regex-based entity extraction.")


class EntityType(Enum):
    """Types of entities we can identify"""
    COMPANY = "company"
    PERSON = "person"
    LOCATION = "location"
    ORGANIZATION = "organization"
    TICKER = "ticker"
    SECTOR = "sector"
    FINANCIAL_INSTRUMENT = "financial_instrument"


@dataclass
class LinkedEntity:
    """Represents an entity found and linked in text"""
    text: str                    # Original text span
    entity_type: EntityType      # Type of entity
    confidence: float            # Confidence in entity identification
    ticker: Optional[str] = None # Associated stock ticker if found
    canonical_name: Optional[str] = None  # Standardized name
    start_char: int = 0         # Character position in original text
    end_char: int = 0           # End character position
    context: str = ""           # Surrounding context
    aliases: List[str] = None   # Known aliases/alternative names


class EntityLinker:
    """
    Advanced entity linking using NER to identify financial entities in text
    """
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        """Initialize the EntityLinker with spaCy model"""
        
        self.nlp = None
        self.model_name = model_name
        
        # Initialize spaCy if available
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load(model_name)
                logger.info(f"Loaded spaCy model: {model_name}")
            except OSError:
                logger.warning(f"spaCy model {model_name} not found. Using basic extraction.")
                self.nlp = None
        
        # Initialize entity knowledge bases
        self.ticker_to_company = self._initialize_ticker_mapping()
        self.company_to_ticker = {v: k for k, v in self.ticker_to_company.items()}
        self.company_aliases = self._initialize_company_aliases()
        self.sector_keywords = self._initialize_sector_keywords()
        self.financial_instruments = self._initialize_financial_instruments()
    
    def _initialize_ticker_mapping(self) -> Dict[str, str]:
        """Initialize mapping of tickers to company names"""
        
        # This would typically come from a market data provider
        # For now, using a representative sample
        return {
            'AAPL': 'Apple Inc.',
            'MSFT': 'Microsoft Corporation', 
            'GOOGL': 'Alphabet Inc.',
            'GOOG': 'Alphabet Inc.',
            'AMZN': 'Amazon.com Inc.',
            'TSLA': 'Tesla Inc.',
            'META': 'Meta Platforms Inc.',
            'NVDA': 'NVIDIA Corporation',
            'NFLX': 'Netflix Inc.',
            'AMD': 'Advanced Micro Devices Inc.',
            'CRM': 'Salesforce Inc.',
            'ADBE': 'Adobe Inc.',
            'PYPL': 'PayPal Holdings Inc.',
            'INTC': 'Intel Corporation',
            'ORCL': 'Oracle Corporation',
            'JPM': 'JPMorgan Chase & Co.',
            'BAC': 'Bank of America Corporation',
            'WMT': 'Walmart Inc.',
            'JNJ': 'Johnson & Johnson',
            'PG': 'Procter & Gamble Company',
            'V': 'Visa Inc.',
            'MA': 'Mastercard Incorporated',
            'HD': 'Home Depot Inc.',
            'DIS': 'Walt Disney Company',
            'COIN': 'Coinbase Global Inc.',
            'SQ': 'Block Inc.',
            'SHOP': 'Shopify Inc.',
            'SPOT': 'Spotify Technology S.A.',
            'UBER': 'Uber Technologies Inc.',
            'LYFT': 'Lyft Inc.',
            'ABNB': 'Airbnb Inc.',
            'ROKU': 'Roku Inc.',
            'ZM': 'Zoom Video Communications Inc.',
            'SNOW': 'Snowflake Inc.',
            'PLTR': 'Palantir Technologies Inc.',
            'BTC-USD': 'Bitcoin',
            'ETH-USD': 'Ethereum'
        }
    
    def _initialize_company_aliases(self) -> Dict[str, str]:
        """Initialize common company aliases and variations"""
        
        return {
            'Apple': 'Apple Inc.',
            'Microsoft': 'Microsoft Corporation',
            'Google': 'Alphabet Inc.',
            'Amazon': 'Amazon.com Inc.',
            'Tesla': 'Tesla Inc.',
            'Facebook': 'Meta Platforms Inc.',
            'Meta': 'Meta Platforms Inc.',
            'Nvidia': 'NVIDIA Corporation',
            'Netflix': 'Netflix Inc.',
            'Intel': 'Intel Corporation',
            'Oracle': 'Oracle Corporation',
            'JPMorgan': 'JPMorgan Chase & Co.',
            'Bank of America': 'Bank of America Corporation',
            'BofA': 'Bank of America Corporation',
            'Walmart': 'Walmart Inc.',
            'Johnson & Johnson': 'Johnson & Johnson',
            'J&J': 'Johnson & Johnson',
            'Visa': 'Visa Inc.',
            'Mastercard': 'Mastercard Incorporated',
            'Home Depot': 'Home Depot Inc.',
            'Disney': 'Walt Disney Company',
            'Coinbase': 'Coinbase Global Inc.',
            'Square': 'Block Inc.',
            'Shopify': 'Shopify Inc.',
            'Spotify': 'Spotify Technology S.A.',
            'Uber': 'Uber Technologies Inc.',
            'Lyft': 'Lyft Inc.',
            'Airbnb': 'Airbnb Inc.',
            'Roku': 'Roku Inc.',
            'Zoom': 'Zoom Video Communications Inc.',
            'Snowflake': 'Snowflake Inc.',
            'Palantir': 'Palantir Technologies Inc.',
            'Bitcoin': 'Bitcoin',
            'BTC': 'Bitcoin',
            'Ethereum': 'Ethereum',
            'ETH': 'Ethereum'
        }
    
    def _initialize_sector_keywords(self) -> Dict[str, List[str]]:
        """Initialize sector classification keywords"""
        
        return {
            'Technology': ['tech', 'software', 'cloud', 'AI', 'artificial intelligence', 
                          'machine learning', 'digital', 'internet', 'semiconductor'],
            'Financial': ['bank', 'finance', 'credit', 'loan', 'investment', 'insurance',
                         'payment', 'fintech', 'trading', 'broker'],
            'Healthcare': ['pharma', 'medical', 'drug', 'biotech', 'hospital', 'FDA',
                          'clinical', 'therapy', 'treatment'],
            'Consumer': ['retail', 'consumer', 'brand', 'product', 'store', 'e-commerce',
                        'shopping', 'subscription'],
            'Energy': ['oil', 'gas', 'energy', 'renewable', 'solar', 'wind', 'electric'],
            'Automotive': ['car', 'auto', 'vehicle', 'electric vehicle', 'EV', 'automotive'],
            'Real Estate': ['real estate', 'property', 'housing', 'REIT'],
            'Crypto': ['crypto', 'blockchain', 'bitcoin', 'ethereum', 'NFT', 'DeFi']
        }
    
    def _initialize_financial_instruments(self) -> Set[str]:
        """Initialize financial instrument keywords"""
        
        return {
            'stock', 'share', 'equity', 'bond', 'option', 'future', 'derivative',
            'ETF', 'mutual fund', 'index', 'commodity', 'forex', 'currency',
            'cryptocurrency', 'token', 'coin'
        }
    
    def _extract_tickers_regex(self, text: str) -> List[Tuple[str, int, int]]:
        """Extract ticker symbols using regex patterns"""
        
        tickers = []
        
        # Common ticker patterns
        patterns = [
            r'\b[A-Z]{1,5}\b',  # 1-5 uppercase letters
            r'\$[A-Z]{1,5}\b',  # $ prefix
            r'\b[A-Z]{1,5}-USD\b',  # Crypto format
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                ticker = match.group().replace('$', '')
                # Filter out common false positives
                if (ticker in self.ticker_to_company and 
                    ticker not in ['CEO', 'CFO', 'COO', 'CTO', 'SEC', 'FDA', 'AI', 'ML']):
                    tickers.append((ticker, match.start(), match.end()))
        
        return tickers
    
    def _extract_companies_regex(self, text: str) -> List[Tuple[str, int, int]]:
        """Extract company names using regex patterns"""
        
        companies = []
        
        # Look for known company aliases
        for alias, canonical in self.company_aliases.items():
            pattern = r'\b' + re.escape(alias) + r'\b'
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                companies.append((canonical, match.start(), match.end()))
        
        # Look for Inc., Corp., etc.
        company_suffix_pattern = r'\b[A-Z][a-zA-Z\s&\.]{2,30}\s+(Inc\.|Corp\.|Corporation|Company|Co\.|Ltd\.)'
        matches = re.finditer(company_suffix_pattern, text)
        for match in matches:
            companies.append((match.group(), match.start(), match.end()))
        
        return companies
    
    def _classify_sector(self, entity_text: str, context: str) -> Optional[str]:
        """Classify entity into business sector based on context"""
        
        full_text = f"{entity_text} {context}".lower()
        
        sector_scores = {}
        for sector, keywords in self.sector_keywords.items():
            score = sum(1 for keyword in keywords if keyword in full_text)
            if score > 0:
                sector_scores[sector] = score
        
        if sector_scores:
            return max(sector_scores, key=sector_scores.get)
        
        return None
    
    def _get_context(self, text: str, start: int, end: int, window: int = 100) -> str:
        """Extract context around an entity mention"""
        
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        return text[context_start:context_end]
    
    def extract_entities_spacy(self, text: str) -> List[LinkedEntity]:
        """Extract entities using spaCy NER"""
        
        if not self.nlp:
            return []
        
        entities = []
        doc = self.nlp(text)
        
        for ent in doc.ents:
            entity_type = None
            confidence = 0.8  # spaCy doesn't provide confidence scores by default
            ticker = None
            canonical_name = None
            
            # Map spaCy entity types to our types
            if ent.label_ in ['ORG', 'PERSON', 'GPE', 'LOC']:
                if ent.label_ == 'ORG':
                    entity_type = EntityType.COMPANY
                    # Try to link to known company
                    if ent.text in self.company_aliases:
                        canonical_name = self.company_aliases[ent.text]
                        ticker = self.company_to_ticker.get(canonical_name)
                elif ent.label_ == 'PERSON':
                    entity_type = EntityType.PERSON
                elif ent.label_ in ['GPE', 'LOC']:
                    entity_type = EntityType.LOCATION
                
                context = self._get_context(text, ent.start_char, ent.end_char)
                
                entities.append(LinkedEntity(
                    text=ent.text,
                    entity_type=entity_type,
                    confidence=confidence,
                    ticker=ticker,
                    canonical_name=canonical_name,
                    start_char=ent.start_char,
                    end_char=ent.end_char,
                    context=context
                ))
        
        return entities
    
    def extract_entities_regex(self, text: str) -> List[LinkedEntity]:
        """Extract entities using regex patterns as fallback"""
        
        entities = []
        
        # Extract tickers
        tickers = self._extract_tickers_regex(text)
        for ticker, start, end in tickers:
            context = self._get_context(text, start, end)
            canonical_name = self.ticker_to_company.get(ticker)
            
            entities.append(LinkedEntity(
                text=ticker,
                entity_type=EntityType.TICKER,
                confidence=0.9,
                ticker=ticker,
                canonical_name=canonical_name,
                start_char=start,
                end_char=end,
                context=context
            ))
        
        # Extract companies
        companies = self._extract_companies_regex(text)
        for company, start, end in companies:
            context = self._get_context(text, start, end)
            ticker = self.company_to_ticker.get(company)
            
            entities.append(LinkedEntity(
                text=company,
                entity_type=EntityType.COMPANY,
                confidence=0.7,
                ticker=ticker,
                canonical_name=company,
                start_char=start,
                end_char=end,
                context=context
            ))
        
        return entities
    
    def link_entities(self, text: str) -> List[LinkedEntity]:
        """
        Main method to extract and link entities in text
        
        Args:
            text: Input text to process
            
        Returns:
            List of LinkedEntity objects found in text
        """
        
        entities = []
        
        # Try spaCy first if available
        if self.nlp:
            spacy_entities = self.extract_entities_spacy(text)
            entities.extend(spacy_entities)
        
        # Always run regex extraction for financial entities
        regex_entities = self.extract_entities_regex(text)
        entities.extend(regex_entities)
        
        # Remove duplicates and rank by confidence
        unique_entities = self._deduplicate_entities(entities)
        
        # Add sector classification
        for entity in unique_entities:
            if entity.entity_type == EntityType.COMPANY:
                sector = self._classify_sector(entity.text, entity.context)
                if sector:
                    entity.aliases = entity.aliases or []
                    entity.aliases.append(f"sector:{sector}")
        
        return sorted(unique_entities, key=lambda x: x.confidence, reverse=True)
    
    def _deduplicate_entities(self, entities: List[LinkedEntity]) -> List[LinkedEntity]:
        """Remove duplicate entities, keeping highest confidence"""
        
        # Group by text span position
        position_groups = {}
        for entity in entities:
            key = (entity.start_char, entity.end_char)
            if key not in position_groups:
                position_groups[key] = []
            position_groups[key].append(entity)
        
        # Keep highest confidence entity for each position
        deduplicated = []
        for group in position_groups.values():
            best_entity = max(group, key=lambda x: x.confidence)
            deduplicated.append(best_entity)
        
        return deduplicated
    
    def get_financial_entities(self, entities: List[LinkedEntity]) -> List[LinkedEntity]:
        """Filter for entities relevant to financial trading"""
        
        financial_types = {EntityType.COMPANY, EntityType.TICKER, EntityType.FINANCIAL_INSTRUMENT}
        return [e for e in entities if e.entity_type in financial_types or e.ticker]
    
    def to_dataframe(self, entities: List[LinkedEntity]) -> pd.DataFrame:
        """Convert entities to DataFrame for analysis"""
        
        if not entities:
            return pd.DataFrame()
        
        data = []
        for entity in entities:
            data.append({
                'text': entity.text,
                'entity_type': entity.entity_type.value,
                'confidence': entity.confidence,
                'ticker': entity.ticker,
                'canonical_name': entity.canonical_name,
                'start_char': entity.start_char,
                'end_char': entity.end_char,
                'context': entity.context[:100] + "..." if len(entity.context) > 100 else entity.context,
                'aliases': ', '.join(entity.aliases) if entity.aliases else None
            })
        
        return pd.DataFrame(data) 