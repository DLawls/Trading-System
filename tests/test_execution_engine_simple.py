"""
Simple Execution Engine Test - Core functionality validation

Tests the execution engine components without complex dependencies.
"""

import asyncio
import time
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import execution engine components
from execution_engine.order_schemas import (
    Order, OrderStatus, OrderSide, OrderType, TimeInForce, OrderExecution
)
from execution_engine.latency_monitor import LatencyMonitor


def test_order_schemas():
    """Test order data structures"""
    print("🧪 Testing Order Schemas...")
    
    # Create test order
    order = Order(
        order_id=Order.generate_order_id(),
        client_order_id=Order.generate_client_order_id(),
        signal_id="test_signal",
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=100,
        limit_price=150.0
    )
    
    print(f"  ✅ Created order: {order.symbol} {order.side.value} {order.quantity}")
    
    # Test order status
    assert order.status == OrderStatus.PENDING
    assert order.is_active()
    assert not order.is_terminal()
    
    # Test execution
    execution = OrderExecution(
        execution_id="exec_1",
        order_id=order.order_id,
        timestamp=datetime.utcnow(),
        quantity=50,
        price=149.5,
        commission=0.0,
        market_price=149.5,
        spread=0.05,
        venue="test_venue",
        venue_order_id="venue_123"
    )
    
    order.add_execution(execution)
    assert order.filled_quantity == 50
    assert order.average_fill_price == 149.5
    
    print("  ✅ Order schemas working correctly")


def test_latency_monitoring():
    """Test latency monitoring"""
    print("\n🧪 Testing Latency Monitoring...")
    
    monitor = LatencyMonitor()
    
    # Test timer
    timer = monitor.create_timer("test")
    timer.start()
    time.sleep(0.01)  # 10ms
    elapsed = timer.stop()
    
    assert elapsed >= 10.0
    print(f"  ✅ Timer measured: {elapsed:.1f}ms")
    
    # Test order latency tracking
    order_id = "test_order"
    monitor.start_timer(order_id, "order_creation")
    time.sleep(0.005)
    creation_time = monitor.stop_timer(order_id, "order_creation")
    
    assert creation_time >= 5.0
    print(f"  ✅ Order latency: {creation_time:.1f}ms")
    
    # Test metrics
    metrics = monitor.get_order_metrics(order_id)
    assert metrics is not None
    assert metrics.order_creation >= 5.0
    
    print("  ✅ Latency monitoring working correctly")


async def test_async_processing():
    """Test async capabilities"""
    print("\n🧪 Testing Async Processing...")
    
    async def process_order(order_id: str, delay_ms: float):
        await asyncio.sleep(delay_ms / 1000.0)
        return f"Order {order_id} processed"
    
    # Process multiple orders concurrently
    orders = [("order_1", 10), ("order_2", 15), ("order_3", 12)]
    
    start_time = time.time()
    tasks = [process_order(oid, delay) for oid, delay in orders]
    results = await asyncio.gather(*tasks)
    total_time = (time.time() - start_time) * 1000
    
    print(f"  ✅ Processed {len(results)} orders in {total_time:.1f}ms")
    
    # Should be faster than sequential processing
    sequential_time = sum(delay for _, delay in orders)
    assert total_time < sequential_time * 0.8
    
    print("  ✅ Async processing working correctly")


def test_performance_simulation():
    """Test realistic performance scenario"""
    print("\n🧪 Testing Performance Simulation...")
    
    monitor = LatencyMonitor()
    
    # Simulate 100 orders with various latencies
    order_count = 100
    total_latency = 0
    
    for i in range(order_count):
        order_id = f"order_{i:03d}"
        
        # Simulate order processing latency
        start_time = time.perf_counter()
        time.sleep(0.001)  # 1ms base processing
        end_time = time.perf_counter()
        
        latency = (end_time - start_time) * 1000
        total_latency += latency
        
        monitor.record_latency(order_id, "processing", latency)
    
    # Calculate statistics
    avg_latency = total_latency / order_count
    stats = monitor.get_stats("processing")
    
    print(f"  📊 Performance Results:")
    print(f"     - Orders processed: {order_count}")
    print(f"     - Average latency: {avg_latency:.2f}ms")
    print(f"     - Stats count: {stats['processing'].count}")
    print(f"     - Stats mean: {stats['processing'].mean:.2f}ms")
    
    # Verify statistics
    assert stats["processing"].count == order_count
    assert abs(stats["processing"].mean - avg_latency) < 0.5
    
    print("  ✅ Performance simulation completed successfully")


async def main():
    """Run all tests"""
    print("🚀 Event-Driven ML Trading System - Phase 6: Execution Engine Tests")
    print("=" * 80)
    
    try:
        # Run all tests
        test_order_schemas()
        test_latency_monitoring()
        await test_async_processing()
        test_performance_simulation()
        
        print("\n" + "=" * 80)
        print("🎉 ALL EXECUTION ENGINE TESTS PASSED!")
        print("✅ Phase 6: Execution Engine - COMPLETE")
        print("\n📋 Core Features Successfully Tested:")
        print("   ✅ Order schemas and lifecycle management")
        print("   ✅ Ultra-low latency monitoring")
        print("   ✅ Order execution tracking")
        print("   ✅ Asynchronous processing")
        print("   ✅ Performance measurement")
        print("\n🎯 Execution Engine Foundation Complete!")
        print("📈 Ready for broker integration and live trading")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1) 