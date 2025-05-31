# Module 06 – Execution Engine (C++)

## Purpose
Execute trades with low latency using Alpaca’s trading API. Prioritizes speed, reliability, and fault tolerance.

---

## Responsibilities

- Convert trade signals into actionable orders
- Manage order states and handle retries
- Interface with Alpaca via REST and WebSocket
- Optimize for latency and throughput

---

## Implementation Strategy

- Core components written in **C++**
- Exposed to Python via `pybind11`
- Allows Python orchestration + C++ performance

---

## Submodules

### 1. `LatencySensitiveRouter` (C++)
**Input:** Portfolio-level order plan  
**Output:** Fast order submission

- Parallel order dispatch
- Optimized for REST call efficiency and retry logic
- Tracks time-to-fill and slippage

---

### 2. `OrderManager` (C++)
**Input:** Execution confirmations  
**Output:** Order state tracking

- Tracks pending, filled, partial, and failed orders
- Handles re-submissions if needed
- Stores audit trail

---

### 3. `BrokerAdapter` (C++)
**Interface:** Alpaca REST and WebSocket APIs

- Authenticated requests
- Order placement, cancellation, status queries
- Real-time order fill and trade updates

---

### 4. Python Bindings
- Use `pybind11` to create Python-callable interfaces
- Enables orchestration from main Python strategy loop

---

## Output

Executed trades are:
- Logged with timestamps and order status
- Matched back to their originating signals
- Stored for backtest and performance auditing

---

## Next Steps

- Design C++ class headers and API spec
- Build test stubs for Alpaca execution
- Benchmark latency and integrate with live system
