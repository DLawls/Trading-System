# Module 05 – Signal Generation

## Purpose
Transform ML predictions into executable trading signals while managing confidence, position sizing, and portfolio constraints.

---

## Responsibilities

- Filter and validate predictions for actionability
- Size positions based on volatility and risk models
- Allocate capital across multiple assets under constraints

---

## Submodules

### 1. `SignalEvaluator` (Python)
**Input:** ML predictions  
**Output:** Filtered trade signals

- Threshold-based filtering (e.g., P(up) > 0.7)
- Confidence-based suppression
- Optional: ensemble disagreement veto

---

### 2. `PositionSizer` (Python)
**Input:** Filtered signals + asset stats  
**Output:** Suggested trade size

- Risk-based sizing (e.g., Kelly Criterion, volatility targeting)
- Capital exposure limits per asset
- Optional: drawdown-aware scaling

---

### 3. `PortfolioAllocator` (Python)
**Input:** Sized trade candidates  
**Output:** Portfolio-level allocation plan

- Ensures diversification across assets/sectors
- Applies max/min asset weights
- Avoids overexposure and ensures margin compliance

---

## Output

Each signal includes:
- `asset_id`
- `action` (buy/sell/hold)
- `size`
- `confidence_score`
- `risk_tag` (e.g., "aggressive", "hedge", "neutral")

---

## Next Steps

- Define risk model and constraints
- Implement position sizing logic with real volatility data
- Backtest allocation strategies under portfolio constraints
