"""
Portfolio Allocator - Manages portfolio-level diversification and risk allocation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from loguru import logger
from dataclasses import dataclass
from collections import defaultdict

from .signal_schema import TradingSignal, PortfolioSignal, SignalDirection, SignalType


@dataclass
class AllocationConfig:
    """Configuration for portfolio allocation"""
    
    # Portfolio constraints
    max_total_exposure: float = 0.95          # Maximum total portfolio exposure
    min_cash_reserve: float = 0.05            # Minimum cash reserve
    max_single_position: float = 0.15         # Maximum single position size
    
    # Diversification limits
    max_sector_exposure: float = 0.30         # Maximum exposure per sector
    max_signal_type_exposure: float = 0.40    # Maximum exposure per signal type
    max_correlation_cluster: float = 0.35     # Maximum exposure to correlated assets
    
    # Risk management
    max_portfolio_var: float = 0.03           # Maximum portfolio VaR (3%)
    max_drawdown_risk: float = 0.20           # Maximum expected drawdown
    target_sharpe_ratio: float = 1.5          # Target Sharpe ratio
    
    # Concentration limits
    max_positions: int = 20                   # Maximum number of positions
    min_position_size: float = 0.01          # Minimum position size
    concentration_penalty: float = 0.1       # Penalty for concentration
    
    # Rebalancing thresholds
    rebalance_threshold: float = 0.05         # When to trigger rebalancing
    drift_tolerance: float = 0.02             # Tolerance for position drift
    
    # Dynamic allocation
    volatility_scaling: bool = True           # Scale allocations by volatility
    momentum_bias: float = 0.1                # Bias towards momentum signals
    regime_adjustment: bool = True            # Adjust for market regime


class PortfolioAllocator:
    """
    Manages portfolio-level allocation with diversification and risk constraints
    """
    
    def __init__(self, config: AllocationConfig = None):
        """
        Initialize the PortfolioAllocator
        
        Args:
            config: Allocation configuration
        """
        self.config = config or AllocationConfig()
        self.allocation_history = []
        self.current_allocations = {}
        self.risk_metrics = {}
        
        logger.info("PortfolioAllocator initialized")
    
    def allocate_portfolio(
        self,
        signals: List[TradingSignal],
        existing_positions: Dict[str, float] = None,
        market_context: Dict[str, Any] = None,
        correlation_matrix: Dict[str, Dict[str, float]] = None
    ) -> PortfolioSignal:
        """
        Allocate portfolio across multiple signals with diversification constraints
        
        Args:
            signals: List of trading signals to allocate
            existing_positions: Current portfolio positions
            market_context: Current market conditions
            correlation_matrix: Asset correlation matrix
            
        Returns:
            PortfolioSignal with optimized allocations
        """
        
        try:
            # Filter and validate signals
            valid_signals = self._filter_valid_signals(signals)
            if not valid_signals:
                logger.warning("No valid signals for allocation")
                return self._create_empty_portfolio_signal()
            
            # Group signals by characteristics
            signal_groups = self._group_signals(valid_signals)
            
            # Calculate base allocations
            base_allocations = self._calculate_base_allocations(valid_signals)
            
            # Apply diversification constraints
            diversified_allocations = self._apply_diversification_constraints(
                base_allocations, signal_groups, correlation_matrix
            )
            
            # Apply risk constraints
            risk_adjusted_allocations = self._apply_risk_constraints(
                diversified_allocations, existing_positions, market_context
            )
            
            # Optimize allocations
            optimized_allocations = self._optimize_allocations(
                risk_adjusted_allocations, signal_groups, market_context
            )
            
            # Create final signals with allocations
            final_signals = self._create_final_signals(
                valid_signals, optimized_allocations
            )
            
            # Calculate portfolio metrics
            portfolio_metrics = self._calculate_portfolio_metrics(
                final_signals, existing_positions, correlation_matrix
            )
            
            # Create portfolio signal
            portfolio_signal = PortfolioSignal(
                portfolio_id=f"portfolio_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                timestamp=datetime.utcnow(),
                signals=final_signals,
                total_exposure=sum(s.position_size for s in final_signals),
                sector_exposure=self._calculate_sector_exposure(final_signals),
                correlation_matrix=correlation_matrix or {},
                portfolio_var=portfolio_metrics['var'],
                max_drawdown_risk=portfolio_metrics['max_drawdown_risk'],
                sharpe_estimate=portfolio_metrics['sharpe_estimate'],
                rebalance_required=self._should_rebalance(final_signals, existing_positions),
                cash_required=sum(s.dollar_amount for s in final_signals if s.dollar_amount)
            )
            
            # Store allocation for analysis
            self._store_allocation(portfolio_signal)
            
            logger.info(f"Allocated portfolio with {len(final_signals)} signals, "
                       f"total exposure: {portfolio_signal.total_exposure:.1%}")
            
            return portfolio_signal
            
        except Exception as e:
            logger.error(f"Error allocating portfolio: {e}")
            return self._create_empty_portfolio_signal()
    
    def rebalance_portfolio(
        self,
        new_signals: List[TradingSignal],
        current_positions: Dict[str, float],
        market_context: Dict[str, Any] = None
    ) -> PortfolioSignal:
        """
        Rebalance existing portfolio with new signals
        
        Args:
            new_signals: New trading signals
            current_positions: Current portfolio positions
            market_context: Current market conditions
            
        Returns:
            Rebalanced portfolio signal
        """
        
        # Calculate position drifts
        drift_analysis = self._analyze_position_drift(current_positions)
        
        # Combine new signals with rebalancing needs
        combined_signals = self._combine_signals_with_rebalancing(
            new_signals, drift_analysis
        )
        
        # Allocate the combined portfolio
        return self.allocate_portfolio(
            combined_signals, current_positions, market_context
        )
    
    def _filter_valid_signals(self, signals: List[TradingSignal]) -> List[TradingSignal]:
        """Filter signals that meet basic criteria"""
        
        valid_signals = []
        
        for signal in signals:
            # Check if signal is still valid
            if not signal.is_valid():
                continue
            
            # Check minimum confidence
            if signal.confidence < 0.5:
                continue
            
            # Check minimum position size
            if signal.position_size < self.config.min_position_size:
                continue
            
            valid_signals.append(signal)
        
        return valid_signals
    
    def _group_signals(self, signals: List[TradingSignal]) -> Dict[str, List[TradingSignal]]:
        """Group signals by various characteristics"""
        
        groups = {
            'by_sector': defaultdict(list),
            'by_signal_type': defaultdict(list),
            'by_direction': defaultdict(list),
            'by_urgency': defaultdict(list)
        }
        
        for signal in signals:
            # Group by sector
            sector = signal.sector or 'Unknown'
            groups['by_sector'][sector].append(signal)
            
            # Group by signal type
            groups['by_signal_type'][signal.signal_type.value].append(signal)
            
            # Group by direction
            groups['by_direction'][signal.direction.value].append(signal)
            
            # Group by urgency
            groups['by_urgency'][signal.urgency].append(signal)
        
        return groups
    
    def _calculate_base_allocations(self, signals: List[TradingSignal]) -> Dict[str, float]:
        """Calculate base allocations based on signal strength"""
        
        allocations = {}
        total_score = 0.0
        
        # Calculate weighted scores
        for signal in signals:
            # Base score from confidence and strength
            confidence_score = signal.confidence
            strength_multiplier = {
                'weak': 0.5,
                'moderate': 1.0,
                'strong': 1.5,
                'extreme': 2.0
            }.get(signal.strength.value, 1.0)
            
            # Adjust for prediction strength
            prediction_adjustment = abs(signal.prediction_value - 0.5) * 2  # 0 to 1
            
            # Calculate total score
            score = confidence_score * strength_multiplier * (1 + prediction_adjustment)
            
            allocations[signal.signal_id] = score
            total_score += score
        
        # Normalize to sum to max exposure
        if total_score > 0:
            target_exposure = min(self.config.max_total_exposure, 1.0)
            for signal_id in allocations:
                allocations[signal_id] = (allocations[signal_id] / total_score) * target_exposure
        
        return allocations
    
    def _apply_diversification_constraints(
        self,
        base_allocations: Dict[str, float],
        signal_groups: Dict[str, Dict],
        correlation_matrix: Dict[str, Dict[str, float]] = None
    ) -> Dict[str, float]:
        """Apply diversification constraints"""
        
        adjusted_allocations = base_allocations.copy()
        
        # Apply sector limits
        for sector, sector_signals in signal_groups['by_sector'].items():
            sector_allocation = sum(base_allocations.get(s.signal_id, 0) for s in sector_signals)
            
            if sector_allocation > self.config.max_sector_exposure:
                # Scale down sector positions proportionally
                scale_factor = self.config.max_sector_exposure / sector_allocation
                for signal in sector_signals:
                    if signal.signal_id in adjusted_allocations:
                        adjusted_allocations[signal.signal_id] *= scale_factor
        
        # Apply signal type limits
        for signal_type, type_signals in signal_groups['by_signal_type'].items():
            type_allocation = sum(adjusted_allocations.get(s.signal_id, 0) for s in type_signals)
            
            if type_allocation > self.config.max_signal_type_exposure:
                # Scale down signal type positions proportionally
                scale_factor = self.config.max_signal_type_exposure / type_allocation
                for signal in type_signals:
                    if signal.signal_id in adjusted_allocations:
                        adjusted_allocations[signal.signal_id] *= scale_factor
        
        # Apply correlation constraints
        if correlation_matrix:
            adjusted_allocations = self._apply_correlation_constraints(
                adjusted_allocations, correlation_matrix
            )
        
        return adjusted_allocations
    
    def _apply_correlation_constraints(
        self,
        allocations: Dict[str, float],
        correlation_matrix: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """Apply correlation-based diversification constraints"""
        
        # Simple correlation clustering approach
        adjusted_allocations = allocations.copy()
        
        # Group highly correlated assets
        correlation_clusters = []
        processed_assets = set()
        
        for asset1, correlations in correlation_matrix.items():
            if asset1 in processed_assets:
                continue
            
            cluster = [asset1]
            for asset2, correlation in correlations.items():
                if asset2 != asset1 and correlation > 0.7 and asset2 not in processed_assets:
                    cluster.append(asset2)
            
            if len(cluster) > 1:
                correlation_clusters.append(cluster)
                processed_assets.update(cluster)
        
        # Apply cluster limits
        for cluster in correlation_clusters:
            cluster_allocation = sum(allocations.get(asset, 0) for asset in cluster)
            
            if cluster_allocation > self.config.max_correlation_cluster:
                # Scale down cluster positions proportionally
                scale_factor = self.config.max_correlation_cluster / cluster_allocation
                for asset in cluster:
                    if asset in adjusted_allocations:
                        adjusted_allocations[asset] *= scale_factor
        
        return adjusted_allocations
    
    def _apply_risk_constraints(
        self,
        allocations: Dict[str, float],
        existing_positions: Dict[str, float],
        market_context: Dict[str, Any]
    ) -> Dict[str, float]:
        """Apply portfolio-level risk constraints"""
        
        adjusted_allocations = allocations.copy()
        
        # Calculate portfolio VaR estimate
        estimated_var = self._estimate_portfolio_var(allocations)
        
        if estimated_var > self.config.max_portfolio_var:
            # Scale down all positions to meet VaR constraint
            scale_factor = self.config.max_portfolio_var / estimated_var
            for signal_id in adjusted_allocations:
                adjusted_allocations[signal_id] *= scale_factor
        
        # Apply market regime adjustments
        if self.config.regime_adjustment and market_context:
            regime = market_context.get('regime')
            if regime == 'bear':
                # Reduce allocations in bear market
                for signal_id in adjusted_allocations:
                    adjusted_allocations[signal_id] *= 0.7
            elif regime == 'volatile':
                # Reduce allocations in volatile market
                for signal_id in adjusted_allocations:
                    adjusted_allocations[signal_id] *= 0.8
        
        return adjusted_allocations
    
    def _optimize_allocations(
        self,
        allocations: Dict[str, float],
        signal_groups: Dict[str, Dict],
        market_context: Dict[str, Any]
    ) -> Dict[str, float]:
        """Optimize allocations using advanced techniques"""
        
        optimized_allocations = allocations.copy()
        
        # Apply momentum bias
        if self.config.momentum_bias > 0:
            momentum_signals = signal_groups['by_signal_type'].get('momentum', [])
            for signal in momentum_signals:
                if signal.signal_id in optimized_allocations:
                    boost = optimized_allocations[signal.signal_id] * self.config.momentum_bias
                    optimized_allocations[signal.signal_id] += boost
        
        # Apply volatility scaling
        if self.config.volatility_scaling:
            # Reduce allocation for high volatility assets
            for signal_id, allocation in optimized_allocations.items():
                # Simple volatility scaling (would use real volatility data in production)
                vol_adjustment = 0.9  # Assume 10% reduction for high vol assets
                optimized_allocations[signal_id] = allocation * vol_adjustment
        
        # Ensure allocations don't exceed limits
        total_allocation = sum(optimized_allocations.values())
        if total_allocation > self.config.max_total_exposure:
            scale_factor = self.config.max_total_exposure / total_allocation
            for signal_id in optimized_allocations:
                optimized_allocations[signal_id] *= scale_factor
        
        return optimized_allocations
    
    def _create_final_signals(
        self,
        original_signals: List[TradingSignal],
        allocations: Dict[str, float]
    ) -> List[TradingSignal]:
        """Create final signals with optimized allocations"""
        
        final_signals = []
        
        for signal in original_signals:
            if signal.signal_id in allocations:
                new_allocation = allocations[signal.signal_id]
                
                # Skip signals with very small allocations
                if new_allocation < self.config.min_position_size:
                    continue
                
                # Update signal with new allocation
                updated_signal = TradingSignal(
                    **{k: v for k, v in signal.__dict__.items() 
                       if k not in ['position_size', 'portfolio_weight', 'dollar_amount']},
                    position_size=new_allocation,
                    portfolio_weight=new_allocation,
                    dollar_amount=signal.dollar_amount * (new_allocation / signal.position_size) if signal.dollar_amount else None
                )
                
                final_signals.append(updated_signal)
        
        return final_signals
    
    def _calculate_portfolio_metrics(
        self,
        signals: List[TradingSignal],
        existing_positions: Dict[str, float],
        correlation_matrix: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """Calculate portfolio-level risk metrics"""
        
        # Simple risk metric calculations
        total_exposure = sum(s.position_size for s in signals)
        
        # Estimate portfolio VaR (simplified)
        avg_confidence = np.mean([s.confidence for s in signals]) if signals else 0.5
        portfolio_var = total_exposure * 0.03 * (2 - avg_confidence)  # Simplified VaR
        
        # Estimate max drawdown risk
        max_drawdown_risk = min(0.3, total_exposure * 0.15)
        
        # Estimate Sharpe ratio
        expected_return = sum(s.confidence * s.position_size * 0.1 for s in signals)  # Simplified
        estimated_volatility = portfolio_var * 16  # Convert daily to annual
        sharpe_estimate = expected_return / estimated_volatility if estimated_volatility > 0 else 0
        
        return {
            'var': portfolio_var,
            'max_drawdown_risk': max_drawdown_risk,
            'sharpe_estimate': sharpe_estimate
        }
    
    def _calculate_sector_exposure(self, signals: List[TradingSignal]) -> Dict[str, float]:
        """Calculate exposure by sector"""
        
        sector_exposure = defaultdict(float)
        
        for signal in signals:
            sector = signal.sector or 'Unknown'
            sector_exposure[sector] += signal.position_size
        
        return dict(sector_exposure)
    
    def _should_rebalance(
        self,
        new_signals: List[TradingSignal],
        existing_positions: Dict[str, float]
    ) -> bool:
        """Determine if rebalancing is required"""
        
        if not existing_positions:
            return bool(new_signals)
        
        # Check if total exposure change exceeds threshold
        new_exposure = sum(s.position_size for s in new_signals)
        current_exposure = sum(abs(pos) for pos in existing_positions.values())
        
        exposure_change = abs(new_exposure - current_exposure)
        return exposure_change > self.config.rebalance_threshold
    
    def _analyze_position_drift(self, current_positions: Dict[str, float]) -> Dict[str, float]:
        """Analyze how much positions have drifted from targets"""
        
        # Simple drift analysis (would be more sophisticated in production)
        drift_analysis = {}
        
        for asset, position in current_positions.items():
            target_position = self.current_allocations.get(asset, 0)
            drift = abs(position - target_position)
            drift_analysis[asset] = drift
        
        return drift_analysis
    
    def _combine_signals_with_rebalancing(
        self,
        new_signals: List[TradingSignal],
        drift_analysis: Dict[str, float]
    ) -> List[TradingSignal]:
        """Combine new signals with rebalancing needs"""
        
        # For now, just return new signals
        # In production, would create rebalancing signals for drifted positions
        return new_signals
    
    def _estimate_portfolio_var(self, allocations: Dict[str, float]) -> float:
        """Estimate portfolio Value at Risk"""
        
        # Simplified VaR calculation
        # In production, would use actual volatility and correlation data
        total_allocation = sum(allocations.values())
        estimated_volatility = 0.02  # 2% daily volatility assumption
        
        return total_allocation * estimated_volatility * 2.33  # 99% VaR
    
    def _create_empty_portfolio_signal(self) -> PortfolioSignal:
        """Create empty portfolio signal when no valid signals"""
        
        return PortfolioSignal(
            portfolio_id=f"empty_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.utcnow(),
            signals=[],
            total_exposure=0.0,
            sector_exposure={},
            correlation_matrix={},
            portfolio_var=0.0,
            max_drawdown_risk=0.0,
            sharpe_estimate=0.0,
            rebalance_required=False,
            cash_required=0.0
        )
    
    def _store_allocation(self, portfolio_signal: PortfolioSignal) -> None:
        """Store allocation for analysis and tracking"""
        
        allocation_record = {
            'timestamp': portfolio_signal.timestamp,
            'portfolio_id': portfolio_signal.portfolio_id,
            'num_signals': len(portfolio_signal.signals),
            'total_exposure': portfolio_signal.total_exposure,
            'sector_exposure': portfolio_signal.sector_exposure,
            'portfolio_var': portfolio_signal.portfolio_var,
            'sharpe_estimate': portfolio_signal.sharpe_estimate
        }
        
        self.allocation_history.append(allocation_record)
        
        # Update current allocations
        self.current_allocations = {
            s.asset_id: s.position_size for s in portfolio_signal.signals
        }
    
    def get_allocation_summary(self) -> Dict[str, Any]:
        """Get summary of allocation activity"""
        
        if not self.allocation_history:
            return {'total_allocations': 0}
        
        df = pd.DataFrame(self.allocation_history)
        
        return {
            'total_allocations': len(self.allocation_history),
            'avg_num_signals': df['num_signals'].mean(),
            'avg_exposure': df['total_exposure'].mean(),
            'avg_var': df['portfolio_var'].mean(),
            'avg_sharpe': df['sharpe_estimate'].mean(),
            'current_allocations': self.current_allocations,
            'config': self.config.__dict__
        } 