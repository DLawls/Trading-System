"""
Test Runner for Event-Driven ML Trading System

Usage:
    python tests/run_tests.py                    # Run all tests
    python tests/run_tests.py data_ingestion    # Run data ingestion tests
    python tests/run_tests.py events            # Run event detection tests
    python tests/run_tests.py features          # Run feature engineering tests
    python tests/run_tests.py ml                # Run ML modeling tests
    python tests/run_tests.py signals           # Run signal generation tests
    python tests/run_tests.py smoke             # Run quick smoke tests only
"""

import sys
import subprocess
import os
from pathlib import Path

def run_test_file(test_file: str) -> bool:
    """Run a single test file and return success status"""
    
    print(f"\n{'='*60}")
    print(f"🧪 Running {test_file}")
    print(f"{'='*60}")
    
    try:
        # Change to project root directory
        project_root = Path(__file__).parent.parent
        os.chdir(project_root)
        
        # Run the test
        result = subprocess.run([sys.executable, f"tests/{test_file}"], 
                              capture_output=False, text=True)
        
        if result.returncode == 0:
            print(f"✅ {test_file} PASSED")
            return True
        else:
            print(f"❌ {test_file} FAILED")
            return False
            
    except Exception as e:
        print(f"❌ Error running {test_file}: {e}")
        return False

def run_test_category(category: str) -> dict:
    """Run tests for a specific category"""
    
    test_mapping = {
        'data_ingestion': ['test_data_ingestion.py', 'test_events.py'],
        'events': ['test_entity_linker.py', 'test_impact_scorer.py', 'test_event_pipeline.py'],
        'features': ['test_feature_smoke.py', 'test_feature_engineering_simple.py'],
        'ml': ['test_ml_modeling.py'],
        'signals': ['test_signal_schema.py', 'test_signal_generation.py'],
        'smoke': ['test_feature_smoke.py', 'test_signal_schema.py'],
        'all': [
            'test_data_ingestion.py',
            'test_events.py', 
            'test_entity_linker.py',
            'test_impact_scorer.py',
            'test_event_pipeline.py',
            'test_feature_smoke.py',
            'test_ml_modeling.py',
            'test_signal_schema.py',
            'test_signal_generation.py'
        ]
    }
    
    if category not in test_mapping:
        print(f"❌ Unknown test category: {category}")
        print(f"Available categories: {list(test_mapping.keys())}")
        return {'total': 0, 'passed': 0, 'failed': 0}
    
    test_files = test_mapping[category]
    results = {'total': len(test_files), 'passed': 0, 'failed': 0}
    
    print(f"\n🎯 Running {category.upper()} tests ({len(test_files)} files)")
    
    for test_file in test_files:
        if run_test_file(test_file):
            results['passed'] += 1
        else:
            results['failed'] += 1
    
    return results

def print_summary(results: dict):
    """Print test summary"""
    
    print(f"\n{'='*60}")
    print("📊 TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Total tests: {results['total']}")
    print(f"✅ Passed: {results['passed']}")
    print(f"❌ Failed: {results['failed']}")
    
    if results['failed'] == 0:
        print("\n🎉 ALL TESTS PASSED! 🎉")
        success_rate = 100.0
    else:
        success_rate = (results['passed'] / results['total']) * 100
        print(f"\n📈 Success rate: {success_rate:.1f}%")
    
    return results['failed'] == 0

def main():
    """Main test runner"""
    
    print("🧪 Event-Driven ML Trading System - Test Runner")
    print("=" * 60)
    
    # Determine what to run
    if len(sys.argv) == 1:
        category = 'all'
    else:
        category = sys.argv[1].lower()
    
    # Run tests
    results = run_test_category(category)
    
    # Print summary
    success = print_summary(results)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 