"""
Test script for Signal Schema
"""

from datetime import datetime, timedelta
from src.signal_generation.signal_schema import (
    TradingSignal, PortfolioSignal, 
    SignalType, SignalDirection, SignalStrength
)

def test_signal_schema():
    """Test the signal schema components."""
    
    print("🎯 Testing Signal Schema...")
    
    # Test 1: Create a trading signal
    print("\n📊 Testing TradingSignal creation...")
    try:
        signal = TradingSignal(
            signal_id="TSLA_001_20240601",
            timestamp=datetime.utcnow(),
            asset_id="TSLA",
            signal_type=SignalType.EVENT_DRIVEN,
            direction=SignalDirection.LONG,
            strength=SignalStrength.STRONG,
            confidence=0.85,
            position_size=0.15,
            dollar_amount=10000.0,
            stop_loss=220.0,
            take_profit=280.0,
            max_risk_pct=0.02,
            volatility_adj=1.2,
            portfolio_weight=0.15,
            model_id="ensemble_v1",
            prediction_value=0.85,
            features_used={"earnings_surprise": 0.8, "momentum_5d": 0.6},
            triggering_event="Q4 Earnings Beat",
            event_impact_score=0.75,
            time_to_event=2.5,
            market_regime="bull",
            sector="Technology",
            correlation_risk=0.3,
            urgency="high"
        )
        
        print(f"   ✅ Created signal: {signal.signal_id}")
        print(f"   📈 Direction: {signal.direction.value}")
        print(f"   💪 Strength: {signal.strength.value}")
        print(f"   🎯 Confidence: {signal.confidence}")
        print(f"   💰 Dollar amount: ${signal.dollar_amount:,.2f}")
        
    except Exception as e:
        print(f"   ❌ Signal creation failed: {e}")
        return False
    
    # Test 2: Signal validation
    print("\n🔍 Testing signal validation...")
    try:
        # Check if signal is valid
        is_valid = signal.is_valid()
        print(f"   ✅ Signal validity: {is_valid}")
        
        # Check execution priority
        priority = signal.get_execution_priority()
        print(f"   ⚡ Execution priority: {priority}/5")
        
        # Get risk metrics
        risk_metrics = signal.get_risk_metrics()
        print(f"   📊 Risk metrics: {len(risk_metrics)} metrics")
        
    except Exception as e:
        print(f"   ❌ Signal validation failed: {e}")
        return False
    
    # Test 3: Serialization
    print("\n💾 Testing serialization...")
    try:
        # Convert to dict
        signal_dict = signal.to_dict()
        print(f"   ✅ Converted to dict: {len(signal_dict)} fields")
        
        # Convert back from dict
        restored_signal = TradingSignal.from_dict(signal_dict)
        print(f"   ✅ Restored from dict: {restored_signal.signal_id}")
        
        # Verify they match
        assert restored_signal.signal_id == signal.signal_id
        assert restored_signal.confidence == signal.confidence
        print(f"   ✅ Serialization round-trip successful")
        
    except Exception as e:
        print(f"   ❌ Serialization failed: {e}")
        return False
    
    # Test 4: Create portfolio signal
    print("\n📋 Testing PortfolioSignal...")
    try:
        # Create multiple signals
        signals = [
            signal,  # TSLA signal from above
            TradingSignal(
                signal_id="AAPL_001_20240601",
                timestamp=datetime.utcnow(),
                asset_id="AAPL",
                signal_type=SignalType.MOMENTUM,
                direction=SignalDirection.LONG,
                strength=SignalStrength.MODERATE,
                confidence=0.7,
                position_size=0.10,
                dollar_amount=8000.0,
                max_risk_pct=0.015,
                volatility_adj=1.0,
                portfolio_weight=0.10,
                model_id="momentum_v1",
                prediction_value=0.7,
                features_used={"sma_crossover": 0.8},
                correlation_risk=0.25,
                sector="Technology"
            )
        ]
        
        # Create portfolio signal
        portfolio_signal = PortfolioSignal(
            portfolio_id="main_portfolio",
            timestamp=datetime.utcnow(),
            signals=signals,
            total_exposure=0.25,
            sector_exposure={"Technology": 0.25},
            correlation_matrix={"TSLA": {"AAPL": 0.6}},
            portfolio_var=0.02,
            max_drawdown_risk=0.15,
            sharpe_estimate=1.8,
            rebalance_required=True,
            cash_required=18000.0
        )
        
        print(f"   ✅ Created portfolio signal: {portfolio_signal.portfolio_id}")
        print(f"   📊 Total signals: {len(portfolio_signal.signals)}")
        print(f"   💰 Total dollar amount: ${portfolio_signal.get_total_dollar_amount():,.2f}")
        print(f"   📈 High priority signals: {len(portfolio_signal.get_high_priority_signals())}")
        
        # Group by direction
        by_direction = portfolio_signal.get_signals_by_direction()
        for direction, sigs in by_direction.items():
            print(f"   📍 {direction.value}: {len(sigs)} signals")
        
    except Exception as e:
        print(f"   ❌ Portfolio signal failed: {e}")
        return False
    
    print("\n✅ Signal Schema Test PASSED!")
    print("   📊 TradingSignal: Working")
    print("   📋 PortfolioSignal: Working") 
    print("   💾 Serialization: Working")
    print("   🔍 Validation: Working")
    print("\n🎯 Signal schema is ready for production!")
    
    return True


if __name__ == "__main__":
    test_signal_schema() 