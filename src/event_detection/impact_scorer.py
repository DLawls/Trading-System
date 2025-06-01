"""
Impact Scorer for assessing potential market impact of detected events
"""

import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import pandas as pd
from loguru import logger

from .event_classifier import DetectedEvent, EventType


class ImpactLevel(Enum):
    """Impact levels for market events"""
    MINIMAL = "minimal"      # < 1% expected move
    LOW = "low"             # 1-3% expected move  
    MEDIUM = "medium"       # 3-7% expected move
    HIGH = "high"           # 7-15% expected move
    EXTREME = "extreme"     # > 15% expected move


@dataclass
class ImpactScore:
    """Represents the assessed impact of an event"""
    event: DetectedEvent
    impact_level: ImpactLevel
    impact_score: float  # 0.0 to 1.0
    price_target_change: Optional[float] = None  # Expected % price change
    volatility_increase: Optional[float] = None  # Expected volatility increase
    time_horizon: str = "1-5 days"  # Expected timeframe for impact
    confidence: float = 0.0  # Confidence in impact assessment
    reasoning: List[str] = None  # Explanation of impact factors


class ImpactScorer:
    """
    Scores the potential market impact of detected events using heuristics
    """
    
    def __init__(self):
        self.impact_weights = self._initialize_impact_weights()
        self.company_size_multipliers = self._initialize_company_multipliers()
        self.market_condition_multipliers = self._initialize_market_multipliers()
    
    def _initialize_impact_weights(self) -> Dict[EventType, Dict[str, float]]:
        """Initialize base impact weights for each event type"""
        
        return {
            EventType.EARNINGS: {
                'base_impact': 0.7,
                'beat_estimates': 1.5,
                'miss_estimates': 1.8,
                'guidance_change': 1.3,
                'surprise_factor': 2.0
            },
            
            EventType.MERGER_ACQUISITION: {
                'base_impact': 0.9,
                'acquisition_premium': 2.0,
                'deal_size': 1.5,
                'strategic_fit': 1.2,
                'regulatory_risk': 0.8
            },
            
            EventType.REGULATORY: {
                'base_impact': 0.8,
                'fda_approval': 1.8,
                'sec_investigation': 1.6,
                'antitrust': 1.4,
                'compliance_violation': 1.2
            },
            
            EventType.LEADERSHIP_CHANGE: {
                'base_impact': 0.4,
                'ceo_change': 1.8,
                'founder_departure': 2.0,
                'scandal_related': 1.6,
                'succession_planned': 0.8
            },
            
            EventType.PRODUCT_LAUNCH: {
                'base_impact': 0.5,
                'breakthrough_technology': 1.6,
                'market_disruption': 1.8,
                'patent_significance': 1.3,
                'competitive_advantage': 1.4
            },
            
            EventType.LEGAL_ISSUE: {
                'base_impact': 0.6,
                'lawsuit_size': 1.5,
                'class_action': 1.7,
                'settlement_amount': 1.4,
                'criminal_charges': 2.0
            },
            
            EventType.PARTNERSHIP: {
                'base_impact': 0.3,
                'strategic_value': 1.4,
                'market_expansion': 1.3,
                'technology_access': 1.2,
                'revenue_potential': 1.5
            },
            
            EventType.MARKET_MOVEMENT: {
                'base_impact': 0.4,
                'analyst_upgrade': 1.2,
                'price_target_change': 1.3,
                'rating_change': 1.1,
                'coverage_initiation': 1.0
            },
            
            EventType.ECONOMIC_DATA: {
                'base_impact': 0.6,
                'fed_decision': 1.8,
                'inflation_surprise': 1.6,
                'gdp_miss': 1.4,
                'employment_data': 1.3
            }
        }
    
    def _initialize_company_multipliers(self) -> Dict[str, float]:
        """Initialize multipliers based on company characteristics"""
        
        return {
            'mega_cap': 0.7,     # Large companies move less on news
            'large_cap': 0.8,
            'mid_cap': 1.0,
            'small_cap': 1.3,    # Small companies more volatile
            'micro_cap': 1.6,
            
            'high_profile': 1.2,  # High-profile companies get more attention
            'meme_stock': 1.8,    # Meme stocks more sensitive to news
            'etf_holding': 0.9,   # ETF holdings can dampen moves
            'low_volume': 1.4,    # Low volume stocks more volatile
        }
    
    def _initialize_market_multipliers(self) -> Dict[str, float]:
        """Initialize multipliers based on market conditions"""
        
        return {
            'high_volatility': 1.3,
            'low_volatility': 0.8,
            'bull_market': 1.1,
            'bear_market': 1.4,
            'earnings_season': 1.2,
            'holiday_period': 0.7,
            'market_hours': 1.0,
            'after_hours': 1.3,
        }
    
    def _extract_financial_metrics(self, event: DetectedEvent) -> Dict[str, Any]:
        """Extract financial metrics from event text"""
        
        metrics = {}
        text = f"{event.title} {event.description}".lower()
        
        # Extract percentage changes
        pct_pattern = r'(\d+(?:\.\d+)?)\s*%'
        percentages = re.findall(pct_pattern, text)
        if percentages:
            metrics['percentages'] = [float(p) for p in percentages]
        
        # Extract dollar amounts (millions/billions)
        dollar_pattern = r'\$(\d+(?:\.\d+)?)\s*(million|billion|m|b)'
        dollar_matches = re.findall(dollar_pattern, text)
        if dollar_matches:
            amounts = []
            for amount, unit in dollar_matches:
                multiplier = 1000000 if unit.lower() in ['million', 'm'] else 1000000000
                amounts.append(float(amount) * multiplier)
            metrics['dollar_amounts'] = amounts
        
        # Detect earnings-specific terms
        if 'beat' in text and 'estimate' in text:
            metrics['earnings_beat'] = True
        elif 'miss' in text and 'estimate' in text:
            metrics['earnings_miss'] = True
        
        # Detect guidance changes
        if any(word in text for word in ['guidance', 'outlook', 'forecast']):
            if any(word in text for word in ['raise', 'increase', 'upgrade']):
                metrics['guidance_raised'] = True
            elif any(word in text for word in ['lower', 'cut', 'reduce']):
                metrics['guidance_lowered'] = True
        
        return metrics
    
    def _assess_company_characteristics(self, entity: str) -> Dict[str, float]:
        """Assess company characteristics that affect impact"""
        
        # This is a simplified heuristic - in production you'd use market cap data
        characteristics = {}
        
        # Common large-cap tickers (simplified)
        mega_caps = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA']
        large_caps = ['NFLX', 'AMD', 'CRM', 'ADBE', 'PYPL', 'INTC', 'ORCL']
        
        if entity.upper() in mega_caps:
            characteristics['size_multiplier'] = self.company_size_multipliers['mega_cap']
        elif entity.upper() in large_caps:
            characteristics['size_multiplier'] = self.company_size_multipliers['large_cap']
        else:
            characteristics['size_multiplier'] = self.company_size_multipliers['mid_cap']
        
        # Check for high-profile status
        high_profile = ['TSLA', 'AAPL', 'AMZN', 'GOOGL', 'META']
        if entity.upper() in high_profile:
            characteristics['profile_multiplier'] = self.company_size_multipliers['high_profile']
        else:
            characteristics['profile_multiplier'] = 1.0
        
        return characteristics
    
    def _calculate_base_impact(self, event: DetectedEvent) -> Tuple[float, List[str]]:
        """Calculate base impact score before adjustments"""
        
        event_weights = self.impact_weights.get(event.event_type, {})
        base_impact = event_weights.get('base_impact', 0.5)
        reasoning = [f"Base impact for {event.event_type.value}: {base_impact}"]
        
        # Extract financial metrics
        metrics = self._extract_financial_metrics(event)
        
        # Apply event-specific adjustments
        if event.event_type == EventType.EARNINGS:
            if metrics.get('earnings_beat'):
                base_impact *= event_weights.get('beat_estimates', 1.0)
                reasoning.append("Earnings beat estimates - increased impact")
            elif metrics.get('earnings_miss'):
                base_impact *= event_weights.get('miss_estimates', 1.0)
                reasoning.append("Earnings miss estimates - increased impact")
            
            if metrics.get('guidance_raised'):
                base_impact *= event_weights.get('guidance_change', 1.0)
                reasoning.append("Guidance raised - increased impact")
            elif metrics.get('guidance_lowered'):
                base_impact *= event_weights.get('guidance_change', 1.0)
                reasoning.append("Guidance lowered - increased impact")
        
        elif event.event_type == EventType.REGULATORY:
            text = f"{event.title} {event.description}".lower()
            if 'fda' in text and 'approval' in text:
                base_impact *= event_weights.get('fda_approval', 1.0)
                reasoning.append("FDA approval - high impact event")
            elif 'sec' in text and 'investigation' in text:
                base_impact *= event_weights.get('sec_investigation', 1.0)
                reasoning.append("SEC investigation - high impact event")
        
        elif event.event_type == EventType.LEADERSHIP_CHANGE:
            text = f"{event.title} {event.description}".lower()
            if 'ceo' in text:
                base_impact *= event_weights.get('ceo_change', 1.0)
                reasoning.append("CEO change - significant impact")
        
        # Factor in confidence from event detection
        confidence_adjustment = event.confidence * 0.5 + 0.5  # Scale 0.5-1.0
        base_impact *= confidence_adjustment
        reasoning.append(f"Event confidence adjustment: {confidence_adjustment:.2f}")
        
        return min(base_impact, 1.0), reasoning
    
    def score_event_impact(
        self, 
        event: DetectedEvent,
        market_conditions: Optional[Dict[str, Any]] = None
    ) -> ImpactScore:
        """
        Score the market impact of a detected event
        
        Args:
            event: The detected event to score
            market_conditions: Optional market condition context
            
        Returns:
            ImpactScore with impact assessment
        """
        
        # Calculate base impact
        base_impact, reasoning = self._calculate_base_impact(event)
        
        # Assess company characteristics
        company_chars = self._assess_company_characteristics(event.entity)
        
        # Apply company size adjustment
        size_adjusted_impact = base_impact * company_chars.get('size_multiplier', 1.0)
        reasoning.append(f"Company size adjustment: {company_chars.get('size_multiplier', 1.0):.2f}")
        
        # Apply profile adjustment
        profile_adjusted_impact = size_adjusted_impact * company_chars.get('profile_multiplier', 1.0)
        
        # Market condition adjustments (if provided)
        final_impact = profile_adjusted_impact
        if market_conditions:
            market_mult = 1.0
            for condition, value in market_conditions.items():
                if condition in self.market_condition_multipliers:
                    market_mult *= self.market_condition_multipliers[condition]
            final_impact *= market_mult
            reasoning.append(f"Market conditions adjustment: {market_mult:.2f}")
        
        # Determine impact level
        if final_impact < 0.2:
            impact_level = ImpactLevel.MINIMAL
            price_target = 0.5  # < 1%
        elif final_impact < 0.4:
            impact_level = ImpactLevel.LOW
            price_target = 2.0  # 1-3%
        elif final_impact < 0.6:
            impact_level = ImpactLevel.MEDIUM
            price_target = 5.0  # 3-7%
        elif final_impact < 0.8:
            impact_level = ImpactLevel.HIGH
            price_target = 10.0  # 7-15%
        else:
            impact_level = ImpactLevel.EXTREME
            price_target = 20.0  # > 15%
        
        # Estimate volatility increase
        volatility_increase = final_impact * 50  # 0-50% vol increase
        
        # Calculate confidence in the impact assessment
        assessment_confidence = min(event.confidence * 0.8, 0.9)  # Cap at 90%
        
        return ImpactScore(
            event=event,
            impact_level=impact_level,
            impact_score=min(final_impact, 1.0),
            price_target_change=price_target,
            volatility_increase=volatility_increase,
            time_horizon="1-5 days",
            confidence=assessment_confidence,
            reasoning=reasoning
        )
    
    def score_events_batch(
        self, 
        events: List[DetectedEvent],
        market_conditions: Optional[Dict[str, Any]] = None
    ) -> List[ImpactScore]:
        """Score multiple events for impact"""
        
        impact_scores = []
        for event in events:
            try:
                impact = self.score_event_impact(event, market_conditions)
                impact_scores.append(impact)
            except Exception as e:
                logger.error(f"Error scoring event impact: {e}")
                # Create minimal impact score as fallback
                impact_scores.append(ImpactScore(
                    event=event,
                    impact_level=ImpactLevel.MINIMAL,
                    impact_score=0.1,
                    confidence=0.3,
                    reasoning=[f"Error in impact assessment: {str(e)}"]
                ))
        
        return impact_scores
    
    def get_high_impact_events(
        self, 
        impact_scores: List[ImpactScore],
        threshold: float = 0.6
    ) -> List[ImpactScore]:
        """Filter for high-impact events above threshold"""
        
        return [score for score in impact_scores if score.impact_score >= threshold]
    
    def to_dataframe(self, impact_scores: List[ImpactScore]) -> pd.DataFrame:
        """Convert impact scores to DataFrame for analysis"""
        
        if not impact_scores:
            return pd.DataFrame()
        
        data = []
        for score in impact_scores:
            data.append({
                'entity': score.event.entity,
                'event_type': score.event.event_type.value,
                'event_confidence': score.event.confidence,
                'impact_level': score.impact_level.value,
                'impact_score': score.impact_score,
                'price_target_change': score.price_target_change,
                'volatility_increase': score.volatility_increase,
                'assessment_confidence': score.confidence,
                'title': score.event.title[:100] + "..." if len(score.event.title) > 100 else score.event.title
            })
        
        return pd.DataFrame(data) 