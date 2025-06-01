"""
Signal Evaluator - Converts ML predictions into trading signals with thresholding
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from loguru import logger
from dataclasses import dataclass

from .signal_schema import (
    TradingSignal, SignalType, SignalDirection, SignalStrength
)


@dataclass
class EvaluationConfig:
    """Configuration for signal evaluation"""
    
    # Confidence thresholds
    min_confidence: float = 0.6          # Minimum confidence to generate signal
    strong_confidence: float = 0.8       # Threshold for strong signals
    extreme_confidence: float = 0.9      # Threshold for extreme signals
    
    # Prediction thresholds  
    long_threshold: float = 0.55         # Threshold for long signals
    short_threshold: float = 0.45        # Threshold for short signals
    
    # Risk management
    max_position_size: float = 0.20      # Maximum position size (20% of portfolio)
    min_position_size: float = 0.01      # Minimum position size (1% of portfolio)
    max_risk_per_trade: float = 0.02     # Maximum risk per trade (2%)
    
    # Signal filtering
    min_event_impact: float = 0.3        # Minimum event impact score
    max_correlation: float = 0.7         # Maximum correlation with existing positions
    
    # Time constraints
    signal_validity_hours: float = 24.0  # How long signals remain valid
    event_window_hours: float = 48.0     # Window around events to consider
    
    # Market regime adjustments
    regime_adjustments: Dict[str, float] = None  # Adjustment factors by market regime
    
    def __post_init__(self):
        if self.regime_adjustments is None:
            self.regime_adjustments = {
                'bull': 1.0,      # No adjustment in bull market
                'bear': 0.7,      # Reduce confidence in bear market
                'sideways': 0.8,  # Slightly reduce in sideways market
                'volatile': 0.6   # Significantly reduce in volatile market
            }


class SignalEvaluator:
    """
    Evaluates ML predictions and converts them to actionable trading signals
    """
    
    def __init__(self, config: EvaluationConfig = None):
        """
        Initialize the SignalEvaluator
        
        Args:
            config: Evaluation configuration
        """
        self.config = config or EvaluationConfig()
        self.signal_history = []
        self.performance_metrics = {}
        
        logger.info("SignalEvaluator initialized")
    
    def evaluate_prediction(
        self,
        prediction_value: float,
        confidence: float,
        asset_id: str,
        model_id: str,
        features_used: Dict[str, float],
        event_context: Dict[str, Any] = None,
        market_context: Dict[str, Any] = None,
        existing_positions: Dict[str, float] = None
    ) -> Optional[TradingSignal]:
        """
        Evaluate a single ML prediction and generate trading signal
        
        Args:
            prediction_value: Raw model prediction (0.0 to 1.0)
            confidence: Model confidence score (0.0 to 1.0) 
            asset_id: Asset identifier
            model_id: Model that generated prediction
            features_used: Key features that drove prediction
            event_context: Context about triggering events
            market_context: Current market conditions
            existing_positions: Current portfolio positions
            
        Returns:
            TradingSignal if criteria met, None otherwise
        """
        
        try:
            # Apply initial filters
            if not self._passes_initial_filters(
                prediction_value, confidence, asset_id, event_context, market_context
            ):
                return None
            
            # Determine signal direction
            direction = self._determine_direction(prediction_value)
            if direction == SignalDirection.HOLD:
                return None
            
            # Adjust confidence based on market regime
            adjusted_confidence = self._adjust_confidence_for_regime(
                confidence, market_context
            )
            
            # Check correlation risk
            correlation_risk = self._calculate_correlation_risk(
                asset_id, existing_positions, market_context
            )
            
            if correlation_risk > self.config.max_correlation:
                logger.debug(f"Signal filtered due to high correlation risk: {correlation_risk}")
                return None
            
            # Determine signal strength
            strength = self._determine_signal_strength(adjusted_confidence)
            
            # Calculate position sizing inputs
            base_position_size = self._calculate_base_position_size(
                adjusted_confidence, prediction_value
            )
            
            # Calculate risk metrics
            volatility_adj = self._calculate_volatility_adjustment(
                asset_id, market_context
            )
            
            max_risk_pct = min(
                self.config.max_risk_per_trade,
                adjusted_confidence * self.config.max_risk_per_trade
            )
            
            # Generate signal ID
            timestamp = datetime.utcnow()
            signal_id = f"{asset_id}_{model_id}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
            
            # Determine signal type
            signal_type = self._determine_signal_type(event_context, features_used)
            
            # Set signal validity
            valid_until = timestamp + timedelta(hours=self.config.signal_validity_hours)
            
            # Extract event information
            triggering_event = None
            event_impact_score = None
            time_to_event = None
            
            if event_context:
                triggering_event = event_context.get('event_description')
                event_impact_score = event_context.get('impact_score')
                time_to_event = event_context.get('time_to_event_hours')
            
            # Extract market information
            market_regime = market_context.get('regime') if market_context else None
            sector = market_context.get('sector') if market_context else None
            
            # Determine urgency based on event timing and confidence
            urgency = self._determine_urgency(
                adjusted_confidence, time_to_event, event_impact_score
            )
            
            # Create trading signal
            signal = TradingSignal(
                signal_id=signal_id,
                timestamp=timestamp,
                asset_id=asset_id,
                signal_type=signal_type,
                direction=direction,
                strength=strength,
                confidence=adjusted_confidence,
                position_size=base_position_size,
                dollar_amount=None,  # Will be set by PositionSizer
                stop_loss=None,      # Will be calculated by PositionSizer
                take_profit=None,    # Will be calculated by PositionSizer
                max_risk_pct=max_risk_pct,
                volatility_adj=volatility_adj,
                portfolio_weight=base_position_size,
                model_id=model_id,
                prediction_value=prediction_value,
                features_used=features_used,
                triggering_event=triggering_event,
                event_impact_score=event_impact_score,
                time_to_event=time_to_event,
                market_regime=market_regime,
                sector=sector,
                correlation_risk=correlation_risk,
                urgency=urgency,
                valid_until=valid_until,
                metadata={
                    'original_confidence': confidence,
                    'adjusted_confidence': adjusted_confidence,
                    'evaluation_config': self.config.__dict__
                }
            )
            
            # Store signal for analysis
            self.signal_history.append(signal)
            
            logger.info(f"Generated signal {signal_id}: {direction.value} {asset_id} "
                       f"(confidence: {adjusted_confidence:.3f}, strength: {strength.value})")
            
            return signal
            
        except Exception as e:
            logger.error(f"Error evaluating prediction for {asset_id}: {e}")
            return None
    
    def evaluate_batch_predictions(
        self,
        predictions_df: pd.DataFrame,
        existing_positions: Dict[str, float] = None,
        market_context: Dict[str, Any] = None
    ) -> List[TradingSignal]:
        """
        Evaluate multiple predictions in batch
        
        Args:
            predictions_df: DataFrame with prediction results
            existing_positions: Current portfolio positions
            market_context: Current market conditions
            
        Returns:
            List of generated trading signals
        """
        
        signals = []
        
        for idx, row in predictions_df.iterrows():
            try:
                # Extract required fields
                asset_id = row.get('asset_id') or row.get('ticker')
                prediction_value = row.get('prediction_value') or row.get('prediction')
                confidence = row.get('confidence', 0.5)
                model_id = row.get('model_id', 'unknown')
                
                # Extract features (assume columns starting with 'feature_' or in features dict)
                features_used = {}
                for col in row.index:
                    if col.startswith('feature_') or col in ['momentum', 'volatility', 'sentiment']:
                        features_used[col] = row[col]
                
                # Extract event context if available
                event_context = None
                if any(col.startswith('event_') for col in row.index):
                    event_context = {
                        'event_description': row.get('event_description'),
                        'impact_score': row.get('event_impact_score'),
                        'time_to_event_hours': row.get('time_to_event_hours')
                    }
                
                # Extract asset-specific market context
                asset_market_context = market_context.copy() if market_context else {}
                asset_market_context.update({
                    'sector': row.get('sector'),
                    'volatility': row.get('volatility'),
                })
                
                # Generate signal
                signal = self.evaluate_prediction(
                    prediction_value=prediction_value,
                    confidence=confidence,
                    asset_id=asset_id,
                    model_id=model_id,
                    features_used=features_used,
                    event_context=event_context,
                    market_context=asset_market_context,
                    existing_positions=existing_positions
                )
                
                if signal:
                    signals.append(signal)
                    
            except Exception as e:
                logger.warning(f"Error processing prediction for row {idx}: {e}")
                continue
        
        logger.info(f"Generated {len(signals)} signals from {len(predictions_df)} predictions")
        return signals
    
    def _passes_initial_filters(
        self,
        prediction_value: float,
        confidence: float,
        asset_id: str,
        event_context: Dict[str, Any],
        market_context: Dict[str, Any]
    ) -> bool:
        """Apply initial filters to prediction"""
        
        # Confidence filter
        if confidence < self.config.min_confidence:
            return False
        
        # Event impact filter (if event-driven)
        if event_context and event_context.get('impact_score'):
            if event_context['impact_score'] < self.config.min_event_impact:
                return False
        
        # Prediction strength filter
        if self.config.short_threshold < prediction_value < self.config.long_threshold:
            return False  # Prediction too weak
        
        return True
    
    def _determine_direction(self, prediction_value: float) -> SignalDirection:
        """Determine signal direction from prediction"""
        
        if prediction_value >= self.config.long_threshold:
            return SignalDirection.LONG
        elif prediction_value <= self.config.short_threshold:
            return SignalDirection.SHORT
        else:
            return SignalDirection.HOLD
    
    def _adjust_confidence_for_regime(
        self,
        confidence: float,
        market_context: Dict[str, Any]
    ) -> float:
        """Adjust confidence based on market regime"""
        
        if not market_context or 'regime' not in market_context:
            return confidence
        
        regime = market_context['regime']
        adjustment = self.config.regime_adjustments.get(regime, 1.0)
        
        return min(confidence * adjustment, 1.0)
    
    def _calculate_correlation_risk(
        self,
        asset_id: str,
        existing_positions: Dict[str, float],
        market_context: Dict[str, Any]
    ) -> float:
        """Calculate correlation risk with existing positions"""
        
        if not existing_positions:
            return 0.0
        
        # Simple heuristic: if same sector, assume 0.6 correlation
        # In production, use actual correlation matrix
        asset_sector = market_context.get('sector') if market_context else None
        
        if not asset_sector:
            return 0.3  # Default moderate correlation
        
        # Check if we have positions in same sector
        for position_asset, weight in existing_positions.items():
            if weight > 0.05:  # Only consider significant positions
                # Simple sector-based correlation (would use real data in production)
                if asset_sector == 'Technology':
                    return 0.6
                else:
                    return 0.3
        
        return 0.1  # Low correlation if no similar positions
    
    def _determine_signal_strength(self, confidence: float) -> SignalStrength:
        """Determine signal strength from confidence"""
        
        if confidence >= self.config.extreme_confidence:
            return SignalStrength.EXTREME
        elif confidence >= self.config.strong_confidence:
            return SignalStrength.STRONG
        elif confidence >= 0.7:
            return SignalStrength.MODERATE
        else:
            return SignalStrength.WEAK
    
    def _calculate_base_position_size(
        self,
        confidence: float,
        prediction_value: float
    ) -> float:
        """Calculate base position size"""
        
        # Scale position size with confidence
        confidence_factor = min(confidence / 0.8, 1.0)  # Cap at 80% confidence
        
        # Scale with prediction strength
        if prediction_value >= 0.5:
            strength_factor = (prediction_value - 0.5) * 2  # 0.5->1.0 maps to 0.0->1.0
        else:
            strength_factor = (0.5 - prediction_value) * 2  # 0.0->0.5 maps to 1.0->0.0
        
        base_size = confidence_factor * strength_factor * self.config.max_position_size
        
        return max(min(base_size, self.config.max_position_size), self.config.min_position_size)
    
    def _calculate_volatility_adjustment(
        self,
        asset_id: str,
        market_context: Dict[str, Any]
    ) -> float:
        """Calculate volatility adjustment factor"""
        
        if not market_context:
            return 1.0
        
        volatility = market_context.get('volatility', 0.02)  # Default 2% daily vol
        
        # Normalize to typical volatility range
        # Higher volatility = higher adjustment factor
        if volatility < 0.01:
            return 0.8  # Low vol
        elif volatility < 0.03:
            return 1.0  # Normal vol
        elif volatility < 0.05:
            return 1.3  # High vol
        else:
            return 1.6  # Very high vol
    
    def _determine_signal_type(
        self,
        event_context: Dict[str, Any],
        features_used: Dict[str, float]
    ) -> SignalType:
        """Determine the type of signal"""
        
        if event_context and event_context.get('impact_score', 0) > 0.3:
            return SignalType.EVENT_DRIVEN
        
        # Analyze features to determine type
        feature_names = list(features_used.keys())
        
        if any('momentum' in name for name in feature_names):
            return SignalType.MOMENTUM
        elif any('volatility' in name or 'vol' in name for name in feature_names):
            return SignalType.VOLATILITY
        elif any('breakout' in name or 'resistance' in name for name in feature_names):
            return SignalType.BREAKOUT
        elif any('mean' in name or 'reversion' in name for name in feature_names):
            return SignalType.REVERSAL
        else:
            return SignalType.ENSEMBLE
    
    def _determine_urgency(
        self,
        confidence: float,
        time_to_event: Optional[float],
        event_impact_score: Optional[float]
    ) -> str:
        """Determine execution urgency"""
        
        # High confidence = higher urgency
        if confidence >= 0.9:
            base_urgency = "high"
        elif confidence >= 0.8:
            base_urgency = "normal"
        else:
            base_urgency = "low"
        
        # Event timing affects urgency
        if time_to_event is not None:
            if time_to_event < 2:  # Less than 2 hours
                return "high"
            elif time_to_event < 6:  # Less than 6 hours
                return max(base_urgency, "normal")
        
        # High impact events are more urgent
        if event_impact_score and event_impact_score > 0.8:
            if base_urgency == "low":
                return "normal"
            else:
                return "high"
        
        return base_urgency
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Get summary of signal evaluation performance"""
        
        if not self.signal_history:
            return {'total_signals': 0}
        
        # Count by direction
        direction_counts = {}
        strength_counts = {}
        
        for signal in self.signal_history:
            direction = signal.direction.value
            strength = signal.strength.value
            
            direction_counts[direction] = direction_counts.get(direction, 0) + 1
            strength_counts[strength] = strength_counts.get(strength, 0) + 1
        
        # Calculate average metrics
        avg_confidence = np.mean([s.confidence for s in self.signal_history])
        avg_position_size = np.mean([s.position_size for s in self.signal_history])
        
        return {
            'total_signals': len(self.signal_history),
            'direction_distribution': direction_counts,
            'strength_distribution': strength_counts,
            'avg_confidence': avg_confidence,
            'avg_position_size': avg_position_size,
            'config': self.config.__dict__
        } 