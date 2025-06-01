"""
Test script for Phase 5: Signal Generation Pipeline
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List

def test_signal_generation_pipeline():
    """Test the complete signal generation pipeline."""
    
    print("🎯 Testing Phase 5: Signal Generation Pipeline...")
    
    # Test 1: Import all components
    print("\n📦 Testing imports...")
    try:
        from src.signal_generation import (
            SignalEvaluator, EvaluationConfig,
            PositionSizer, PositionSizingConfig,
            PortfolioAllocator, AllocationConfig,
            TradingSignal, SignalType, SignalDirection, SignalStrength
        )
        print("   ✅ All signal generation components imported")
        
    except Exception as e:
        print(f"   ❌ Import failed: {e}")
        return False
    
    # Test 2: Create mock ML predictions
    print("\n🤖 Creating mock ML predictions...")
    try:
        # Create sample prediction data
        predictions_data = [
            {
                'asset_id': 'TSLA',
                'prediction_value': 0.85,
                'confidence': 0.9,
                'model_id': 'ensemble_v1',
                'features_used': {'earnings_surprise': 0.8, 'momentum_5d': 0.7},
                'event_description': 'Q4 Earnings Beat',
                'event_impact_score': 0.8,
                'time_to_event_hours': 2.0,
                'sector': 'Technology',
                'volatility': 0.04
            },
            {
                'asset_id': 'AAPL',
                'prediction_value': 0.75,
                'confidence': 0.8,
                'model_id': 'momentum_v1',
                'features_used': {'momentum': 0.9, 'volume': 0.6},
                'sector': 'Technology',
                'volatility': 0.025
            },
            {
                'asset_id': 'SPY',
                'prediction_value': 0.65,
                'confidence': 0.7,
                'model_id': 'regime_v1',
                'features_used': {'regime_strength': 0.8},
                'sector': 'ETF',
                'volatility': 0.015
            },
            {
                'asset_id': 'BTC-USD',
                'prediction_value': 0.9,
                'confidence': 0.85,
                'model_id': 'crypto_v1',
                'features_used': {'momentum': 0.95, 'sentiment': 0.8},
                'sector': 'Cryptocurrency',
                'volatility': 0.08
            }
        ]
        
        print(f"   📊 Created {len(predictions_data)} mock predictions")
        
    except Exception as e:
        print(f"   ❌ Mock data creation failed: {e}")
        return False
    
    # Test 3: Signal Evaluation
    print("\n🔍 Testing SignalEvaluator...")
    try:
        # Initialize signal evaluator
        eval_config = EvaluationConfig(
            min_confidence=0.6,
            strong_confidence=0.8,
            long_threshold=0.6,
            max_position_size=0.2
        )
        signal_evaluator = SignalEvaluator(eval_config)
        
        # Evaluate signals
        signals = []
        for pred in predictions_data:
            event_context = None
            if 'event_description' in pred:
                event_context = {
                    'event_description': pred['event_description'],
                    'impact_score': pred['event_impact_score'],
                    'time_to_event_hours': pred['time_to_event_hours']
                }
            
            market_context = {
                'sector': pred['sector'],
                'volatility': pred['volatility'],
                'regime': 'bull'
            }
            
            signal = signal_evaluator.evaluate_prediction(
                prediction_value=pred['prediction_value'],
                confidence=pred['confidence'],
                asset_id=pred['asset_id'],
                model_id=pred['model_id'],
                features_used=pred['features_used'],
                event_context=event_context,
                market_context=market_context
            )
            
            if signal:
                signals.append(signal)
        
        print(f"   ✅ Generated {len(signals)} trading signals")
        
        # Print signal details
        for signal in signals:
            print(f"      📈 {signal.asset_id}: {signal.direction.value} "
                  f"(confidence: {signal.confidence:.3f}, strength: {signal.strength.value})")
        
    except Exception as e:
        print(f"   ❌ Signal evaluation failed: {e}")
        return False
    
    # Test 4: Position Sizing
    print("\n💰 Testing PositionSizer...")
    try:
        # Initialize position sizer
        sizing_config = PositionSizingConfig(
            total_portfolio_value=100000.0,
            max_position_size=0.15,
            default_stop_loss_pct=0.03
        )
        position_sizer = PositionSizer(sizing_config)
        
        # Create mock current prices
        current_prices = {
            'TSLA': 250.0,
            'AAPL': 190.0,
            'SPY': 450.0,
            'BTC-USD': 45000.0
        }
        
        # Size positions
        sized_signals = []
        for signal in signals:
            if signal.asset_id in current_prices:
                sized_signal = position_sizer.size_position(
                    signal=signal,
                    current_price=current_prices[signal.asset_id]
                )
                sized_signals.append(sized_signal)
        
        print(f"   ✅ Sized {len(sized_signals)} positions")
        
        # Print sizing details
        for signal in sized_signals:
            if signal.dollar_amount:
                print(f"      💵 {signal.asset_id}: ${signal.dollar_amount:,.0f} "
                      f"({signal.position_size:.1%}) SL: ${signal.stop_loss:.2f} "
                      f"TP: ${signal.take_profit:.2f}")
        
    except Exception as e:
        print(f"   ❌ Position sizing failed: {e}")
        return False
    
    # Test 5: Portfolio Allocation
    print("\n📋 Testing PortfolioAllocator...")
    try:
        # Initialize portfolio allocator
        allocation_config = AllocationConfig(
            max_total_exposure=0.85,
            max_sector_exposure=0.3,
            max_single_position=0.15
        )
        portfolio_allocator = PortfolioAllocator(allocation_config)
        
        # Create mock existing positions
        existing_positions = {
            'MSFT': 0.05,  # 5% existing position in Microsoft
            'GOOGL': 0.03  # 3% existing position in Google
        }
        
        # Create mock correlation matrix
        correlation_matrix = {
            'TSLA': {'AAPL': 0.6, 'TSLA': 1.0},
            'AAPL': {'TSLA': 0.6, 'AAPL': 1.0},
            'SPY': {'SPY': 1.0},
            'BTC-USD': {'BTC-USD': 1.0}
        }
        
        # Allocate portfolio
        portfolio_signal = portfolio_allocator.allocate_portfolio(
            signals=sized_signals,
            existing_positions=existing_positions,
            market_context={'regime': 'bull', 'volatility': 0.02},
            correlation_matrix=correlation_matrix
        )
        
        print(f"   ✅ Created portfolio with {len(portfolio_signal.signals)} final signals")
        print(f"      📊 Total exposure: {portfolio_signal.total_exposure:.1%}")
        print(f"      💰 Cash required: ${portfolio_signal.cash_required:,.0f}")
        print(f"      📈 Portfolio VaR: {portfolio_signal.portfolio_var:.3f}")
        print(f"      🔄 Rebalance required: {portfolio_signal.rebalance_required}")
        
        # Print sector breakdown
        print(f"      🏢 Sector exposure:")
        for sector, exposure in portfolio_signal.sector_exposure.items():
            print(f"         {sector}: {exposure:.1%}")
        
    except Exception as e:
        print(f"   ❌ Portfolio allocation failed: {e}")
        return False
    
    # Test 6: End-to-End Pipeline
    print("\n🔄 Testing end-to-end pipeline...")
    try:
        # Test the complete pipeline flow
        pipeline_success = True
        
        # Check that we have signals at each stage
        if not predictions_data:
            print("      ❌ No predictions generated")
            pipeline_success = False
        elif not signals:
            print("      ❌ No signals from evaluator")
            pipeline_success = False
        elif not sized_signals:
            print("      ❌ No sized signals")
            pipeline_success = False
        elif not portfolio_signal.signals:
            print("      ❌ No final portfolio signals")
            pipeline_success = False
        else:
            print("      ✅ Complete pipeline flow successful")
            
            # Print final summary
            total_dollar_amount = sum(s.dollar_amount for s in portfolio_signal.signals if s.dollar_amount)
            avg_confidence = np.mean([s.confidence for s in portfolio_signal.signals])
            
            print(f"      📊 Pipeline Summary:")
            print(f"         🎯 {len(predictions_data)} predictions → {len(portfolio_signal.signals)} final signals")
            print(f"         💰 Total investment: ${total_dollar_amount:,.0f}")
            print(f"         🎯 Average confidence: {avg_confidence:.3f}")
            print(f"         📈 Portfolio exposure: {portfolio_signal.total_exposure:.1%}")
        
        if not pipeline_success:
            return False
        
    except Exception as e:
        print(f"   ❌ End-to-end pipeline failed: {e}")
        return False
    
    # Test 7: Component summaries
    print("\n📊 Testing component summaries...")
    try:
        # Signal evaluator summary
        eval_summary = signal_evaluator.get_evaluation_summary()
        print(f"   📊 SignalEvaluator: {eval_summary['total_signals']} signals generated")
        
        # Position sizer summary
        sizing_summary = position_sizer.get_sizing_summary()
        print(f"   💰 PositionSizer: {sizing_summary['total_positions']} positions sized")
        
        # Portfolio allocator summary
        allocation_summary = portfolio_allocator.get_allocation_summary()
        print(f"   📋 PortfolioAllocator: {allocation_summary['total_allocations']} allocations made")
        
        print("   ✅ All component summaries working")
        
    except Exception as e:
        print(f"   ❌ Component summaries failed: {e}")
        return False
    
    print("\n✅ Phase 5: Signal Generation Test PASSED!")
    print("   🎯 SignalEvaluator: Working")
    print("   💰 PositionSizer: Working")
    print("   📋 PortfolioAllocator: Working")
    print("   🔄 End-to-End Pipeline: Working")
    print("\n🚀 Phase 5: Signal Generation is complete and ready for production!")
    
    return True


if __name__ == "__main__":
    test_signal_generation_pipeline() 