# Module 04 – ML Prediction

## Purpose
Train and deploy machine learning models to predict asset behavior based on engineered features and detected events.

---

## Responsibilities

- Train models on historical event-feature data
- Perform real-time inference for live predictions
- Version and manage models
- Combine predictions across multiple model types

---

## Submodules

### 1. `ModelTrainer` (Python)
**Input:** Historical features + target labels  
**Output:** Trained model artifacts

- Supports classification (e.g. P(up|event)) and regression (e.g. return prediction)
- Cross-validation, hyperparameter tuning
- Supported frameworks: scikit-learn, XGBoost, PyTorch

---

### 2. `ModelPredictor` (Python)
**Input:** Live features  
**Output:** Real-time predictions

- Loads pre-trained model from `ModelStore`
- Scores assets at each time step
- Can be wrapped as a REST/gRPC service in future

---

### 3. `ModelStore` (Python)
**Purpose:** Version and manage trained models

- Stores metadata (date, asset, metrics)
- Loads best-performing models dynamically
- Can store models in filesystem or cloud (e.g. S3)

---

### 4. `EnsembleManager` (Python)
**Purpose:** Combine predictions from multiple models

- Weighted voting, averaging, stacking
- Handles model disagreement and confidence scoring
- Useful for combining sentiment model + price model

---

## Output

Each prediction includes:
- `asset_id`
- `timestamp`
- `prediction_value`
- `model_id`
- `confidence_score`

---

## Next Steps

- Define target variables (e.g. +1% in next hour)
- Choose baseline and advanced model types
- Set up experiment tracking (e.g. MLflow, Weights & Biases)
