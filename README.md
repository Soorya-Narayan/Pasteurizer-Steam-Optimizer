# HTST Pasteurizer Thermal Digital Twin 

> **High-frequency predictive thermal modeling and early warning system for continuous milk pasteurization.**

This repository implements a production-grade machine learning pipeline to forecast holding-tube outlet temperatures (`TT-06`) 30 seconds ahead across 200,000+ SCADA telemetry samples (~7 days of continuous production), predicting thermal deviations before they cause automated flow diversions.

---

## Pipeline Highlights

* **Thermal Lag Modeling**: Models temperature change ($\Delta T_{t+h} = T_{t+h} - T_t$) rather than static levels, capturing thermodynamic delay and steam valve dynamics.
* **Leakage-Free Production Segmentation**: Only valid production runs (flow $\ge 1000\text{ L/h}$) are modeled, with lag features bounded within continuous production runs.
* **Models Trained & Benchmarked**:
  * **Persistence Baseline** ($T_{t+h} = T_t$)
  * **Regularized Ridge Regression**
  * **LightGBM** (Hist gradient boosting)
  * **XGBoost** (Hist gradient boosting)
  * **Residual MLP** (PyTorch deep neural net with LayerNorm & GELU)
  * **Ensemble** (Averaged predictions)
* **Real-Time Visual Monitoring**: Displays console progress bars and auto-refreshes `outputs/live_training_progress.png` during training.
* **Diagnostic Visualizations & HTML Report**: Full suite of dark-themed plots (learning curves, parity hexbins, feature importances, error breakdown) and a standalone HTML dashboard.

---

## Benchmark Performance (Unseen Test Set)

| Model | MAE (°C) | RMSE (°C) | 95th %-ile Error (°C) | $R^2$ (Level) | $R^2$ ($\Delta$) | Skill vs Persistence | Divert Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Persistence (Baseline)** | 0.557 | 0.822 | 1.830 | 0.965 | -0.002 | 0.0% | 100% |
| **Ridge Regression** | 0.450 | 1.081 | 1.048 | 0.939 | -0.735 | +19.3% | 100% |
| **Residual MLP (PyTorch)** | 0.281 | 0.437 | 0.774 | 0.990 | 0.717 | +49.6% | 100% |
| **XGBoost** | 0.266 | 0.444 | 0.681 | 0.990 | 0.708 | +52.3% | 100% |
| **LightGBM** | 0.257 | 0.412 | 0.688 | 0.991 | 0.749 | +54.0% | 100% |
| **Ensemble (Best)** | **0.260** | **0.418** | **0.683** | **0.991** | **0.741** | **+53.3%** | **100%** |

---

## Visual Diagnostics & Results

### 1. 30-Second Ahead Temperature Forecasts vs Ground Truth
*Upper panel shows the entire unseen test week; lower panel zooms into the most dynamic 45-minute operational transient:*
![Test Forecasts](figures/test_forecasts.png)

### 2. Model Accuracy & Benchmark Metrics
*Comparison across MAE, RMSE, Skill Score vs. persistence, and Divert Early-Warning safety recall:*
![Model Comparison](figures/model_comparison.png)

### 3. Training & Validation Learning Curves
*Loss trajectories with early-stopping convergence markers:*
![Learning Curves](figures/learning_curves.png)

### 4. Parity Density Hexbins
*Forecast vs actual alignment along the identity line (diagonal):*
![Parity Plot](figures/parity.png)

### 5. Error Distribution & Operational Regimes
*Left: residual error distribution; Right: MAE grouped by time elapsed since production start (start-ups vs steady state):*
![Error Analysis](figures/error_analysis.png)

### 6. Feature Importance
*Top telemetry signals driving predictions in LightGBM and XGBoost:*
![Feature Importance](figures/feature_importance.png)

### 7. 7-Day SCADA Telemetry Overview
*Holding tube temperature, steam control valve position, and feed flow across the continuous 7-day dataset with train/val/test splits:*
![Data Overview](figures/data_overview.png)

---

## Quickstart

### 1. Installation
Ensure your environment has the required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Train Models
Run the full training pipeline:
```bash
python train.py
```
* Options:
  * `--horizon 30`: Set forecast horizon in seconds (default: 30s).
  * `--models lightgbm xgboost mlp`: Select models to train.
  * `--quick`: Fast smoke run with fewer rounds/epochs.

While training is running, open `outputs/live_training_progress.png` or inspect `outputs/latest/report.html` once complete.

### 3. Generate Predictions on New CSV Logs
Run inference on any raw SCADA CSV export:
```bash
python predict.py --csv csv_files/pasteurizer_dataset41.csv
```
This produces:
* `<run>/predictions/pasteurizer_dataset41_predictions.csv` with predictions and boolean divert flags.
* `<run>/predictions/pasteurizer_dataset41_forecast.png` visualizing the forecast vs actual readings.

### 4. Launch Interactive Streamlit Dashboard
Launch the web interface for operators and engineers:
```bash
streamlit run app.py
```
Open `http://localhost:8501` to access.

### 5. Launch Modern React Web Dashboard (shadcn/ui)
For a high-performance modern web application inspired by **shadcn/ui** (Tailwind CSS, Radix UI, Lucide icons, and Recharts):

1. **Start the FastAPI Backend Server** (port 8000):
   ```bash
   uvicorn api:app --host 127.0.0.1 --port 8000
   ```

2. **Start the Vite React Frontend** (port 5173):
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

Open `http://localhost:5173` to access:
* **⚡ Closed-Loop Steam Optimizer**: Real-time Model Predictive Control actuator setpoints, slew rate adjustments, and thermodynamic energy audit with multi-series Recharts trajectories.
* **📈 SCADA Telemetry & Lookahead**: Continuous dual-axis time series and food safety trip hazard early warning system.
* **🧪 What-If Simulator**: Real-time slider perturbations testing thermal sensitivity to valve biases and milk throughput surges.
* **📊 Model Diagnostics**: Dynamic benchmark tables across LightGBM, XGBoost, Residual MLP, and Ensemble models.

---

## ⚡ Closed-Loop Steam Optimizer & Energy Audit

In pasteurization operations, operators frequently over-steam by $4^\circ\text{C} - 8^\circ\text{C}$ to avoid the catastrophic plant trip caused by under-pasteurizing ($< 82.0^\circ\text{C}$). This thermal safety buffer burns hundreds of kilograms of excess steam every shift.

The built-in **Closed-Loop Steam Optimizer** (`src/optimizer/steam_mpc.py`) pairs the ML surrogate digital twin with a constrained Model Predictive Control (MPC) solver:

### 1. Cost Objective Function
$$\min_{\Delta u_t} \quad w_{\text{track}} \left(T_{t+h}(u_t + \Delta u_t) - T_{\text{target}}\right)^2 + w_{\text{move}} (\Delta u_t)^2 + P_{\text{barrier}}(T_{t+h})$$

* **Tracking Term**: Minimizes holding tube temperature deviation from target setpoint (default $85.0^\circ\text{C}$).
* **Actuator Slew Rate**: Restricts sudden valve chatter with physical rate-of-change limits ($|\Delta u| \le 15\%$).
* **Barrier Penalty**: Asymmetric penalty imposing steep costs ($10^4$) if the forecast approaches within $0.5^\circ\text{C}$ of the legal divert threshold ($82.0^\circ\text{C}$).

### 2. Thermodynamic Energy Savings Formulation
The thermodynamic audit computes the enthalpy required to raise the milk stream above the required target:

$$\dot{Q}_{\text{excess}} = \dot{m}_{\text{milk}} \cdot C_{p,\text{milk}} \cdot \max\left(0, T_{\text{actual}} - T_{\text{target}}\right)$$

$$\dot{m}_{\text{steam\_waste}} = \frac{\dot{Q}_{\text{excess}}}{h_{fg,\text{steam}} \cdot \eta_{\text{boiler}}}$$

Where:
* $\dot{m}_{\text{milk}} = \text{Feed Flow (L/h)} \times 1.03\text{ kg/L}$
* $C_{p,\text{milk}} = 3.93\text{ kJ/kg}\cdot^\circ\text{C}$ (Specific heat capacity of whole milk)
* $h_{fg,\text{steam}} = 2100\text{ kJ/kg}$ (Latent heat of condensation at ~3 bar saturated steam)
* $\eta_{\text{boiler}} = 82\%$ (Steam heat exchanger efficiency)

**Shift Audit Benchmark**: On Shift 41, the optimizer calculated **369.5 kg** of steam saved (177 kWh excess heat eliminated), cutting fuel costs by ~\$14.80/shift while maintaining zero under-pasteurization trips.

---

## Repository Structure

```
├── configs/
│   └── default.yaml             # System configuration, features & hyperparameters
├── figures/                     # Publication-quality diagnostic graphs for README
├── src/
│   ├── data/
│   │   └── loader.py            # CSV deduplication, sanitization, and parquet caching
│   ├── features/
│   │   └── engineering.py       # Production segmentation, lag features & split logic
│   ├── models/
│   │   ├── tree_models.py       # Ridge, LightGBM, XGBoost with progress tracking
│   │   └── neural_models.py     # PyTorch Residual MLP architecture
│   ├── optimizer/
│   │   ├── __init__.py          # Optimizer module exports
│   │   └── steam_mpc.py         # Constrained MPC engine & thermodynamic energy calculator
│   ├── utils/
│   │   ├── metrics.py           # Regression metrics, skill scores & divert recall
│   │   └── report.py            # Standalone HTML report generator
│   └── visualization/
│       ├── plots.py             # Publication-quality diagnostic figures
│       └── style.py             # Cohesive dark theme styling
├── train.py                     # Training entry point
├── predict.py                   # Inference entry point
├── app.py                       # Streamlit 4-tab interactive web dashboard
├── requirements.txt             # Project dependencies
└── outputs/                     # Generated models, plots, metrics and reports
```

