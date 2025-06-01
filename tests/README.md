# 🧪 Tests - Event-Driven ML Trading System

This directory contains all test files for the trading system, organized by component.

## 📁 Test Organization

### 📊 Data Ingestion Tests
- `test_data_ingestion.py` - Tests for data ingestion components
- `test_events.py` - Tests for event scheduling and calendar

### 🧠 Event Detection Tests  
- `test_entity_linker.py` - Tests for entity linking and NER
- `test_impact_scorer.py` - Tests for event impact scoring
- `test_event_pipeline.py` - Tests for complete event detection pipeline

### ⚙️ Feature Engineering Tests
- `test_feature_smoke.py` - Quick smoke tests for feature components
- `test_feature_engineering_simple.py` - Fast feature engineering tests
- `test_feature_engineering.py` - Comprehensive feature engineering tests

### 🤖 ML Modeling Tests
- `test_ml_modeling.py` - Tests for ML modeling components

### 🎯 Signal Generation Tests
- `test_signal_schema.py` - Tests for signal data structures
- `test_signal_generation.py` - Tests for complete signal generation pipeline

## 🚀 Running Tests

### Test Runner Usage

```bash
# Run all tests
python tests/run_tests.py

# Run specific test categories
python tests/run_tests.py data_ingestion    # Data ingestion tests
python tests/run_tests.py events            # Event detection tests  
python tests/run_tests.py features          # Feature engineering tests
python tests/run_tests.py ml                # ML modeling tests
python tests/run_tests.py signals           # Signal generation tests
python tests/run_tests.py smoke             # Quick smoke tests only
```

### Individual Test Files

```bash
# Run individual test files
python tests/test_signal_generation.py
python tests/test_feature_smoke.py
python tests/test_ml_modeling.py
```

## 📊 Test Categories

### 🔥 Smoke Tests
Quick tests that verify basic functionality without heavy computation:
- Feature engineering imports and instantiation
- Signal schema validation
- Component connectivity

### 🏃‍♂️ Fast Tests  
Tests that run quickly but provide more coverage:
- Simple feature engineering
- ML modeling components
- Signal evaluation

### 🔍 Comprehensive Tests
Full integration tests (may take longer):
- Complete feature engineering pipeline
- End-to-end event detection
- Full signal generation pipeline

## ✅ Test Status

All tests are designed to:
- ✅ Run independently 
- ✅ Use mock/sample data
- ✅ Provide clear pass/fail feedback
- ✅ Test both happy path and edge cases
- ✅ Validate component integration

## 🛠️ Development Workflow

1. **Before commits**: Run smoke tests
   ```bash
   python tests/run_tests.py smoke
   ```

2. **Before releases**: Run all tests
   ```bash
   python tests/run_tests.py
   ```

3. **Component development**: Run relevant category
   ```bash
   python tests/run_tests.py signals  # When working on signal generation
   ```

## 📈 Test Coverage

The tests cover:
- 📊 **Data Ingestion**: Market data, news, events, sentiment
- 🧠 **Event Detection**: Classification, impact scoring, entity linking
- ⚙️ **Feature Engineering**: Time series, events, sentiment, market context
- 🤖 **ML Modeling**: Target building, training, prediction, ensembles
- 🎯 **Signal Generation**: Evaluation, position sizing, portfolio allocation

---

**Note**: Tests use mock data and don't require external API keys for basic functionality. 