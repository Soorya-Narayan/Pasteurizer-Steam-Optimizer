"""HTST Pasteurizer · Thermal Digital Twin & Early Warning Dashboard.

Run with:
    streamlit run app.py
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from src.data.loader import clean_frame, read_csvs
from src.features.engineering import make_supervised
from src.models import MODEL_REGISTRY
from src.optimizer.steam_mpc import SteamOptimizer, ThermodynamicProfile, optimize_shift_trajectory
from src.utils.metrics import divert_warning_metrics, regression_metrics, skill_score

st.set_page_config(
    page_title="HTST Pasteurizer Digital Twin",
    page_icon="🥛",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for dark industrial aesthetic
st.markdown("""
<style>
    .metric-card {
        background-color: #161b26;
        border: 1px solid #2a3142;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .status-badge-ok {
        background-color: #06d6a020;
        color: #06d6a0;
        border: 1px solid #06d6a0;
        padding: 4px 12px;
        border-radius: 16px;
        font-weight: 600;
        display: inline-block;
    }
    .status-badge-warn {
        background-color: #ef476f20;
        color: #ef476f;
        border: 1px solid #ef476f;
        padding: 4px 12px;
        border-radius: 16px;
        font-weight: 600;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_model_bundle(run_dir_str: str = "outputs/latest"):
    run_dir = Path(run_dir_str).resolve()
    meta_path = run_dir / "metadata.json"
    if not meta_path.exists():
        st.error(f"Missing run metadata at {meta_path}. Please train models first using `python train.py`.")
        st.stop()
    meta = json.loads(meta_path.read_text())
    models = {n: MODEL_REGISTRY[n].load(run_dir / "models") for n in meta["models"]}
    return meta, models


@st.cache_data
def process_scada_csv(csv_path_or_buffer, timestamp_col: str):
    if isinstance(csv_path_or_buffer, (str, Path)):
        df = read_csvs([Path(csv_path_or_buffer)], timestamp_col, progress=False)
    else:
        raw_df = pd.read_csv(csv_path_or_buffer, low_memory=False)
        df = clean_frame(raw_df, timestamp_col)
    return df


def predict_trajectory(df, meta, models, selected_model):
    cfg = meta["config"]
    data, _ = make_supervised(df, cfg, with_target=False)
    if len(data) == 0:
        return None, None
    X = data.X[meta["features"]]
    if selected_model == "Ensemble":
        delta = np.mean([models[m].predict(X) for m in meta["ensemble_members"]], axis=0)
    else:
        delta = models[selected_model].predict(X)
    pred_temp = data.current.values + delta
    
    labeled, _ = make_supervised(df, cfg, with_target=True)
    actual_future = None
    if len(labeled):
        actual_future = labeled.y_future.reindex(data.X.index).values
        
    return data, pred_temp, actual_future


# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------
st.sidebar.image("https://raw.githubusercontent.com/Soorya-Narayan/Pasteurizer-Steam-Optimizer/main/figures/test_forecasts.png", use_container_width=True)
st.sidebar.title("🥛 HTST Digital Twin")
st.sidebar.markdown("**Milk Pasteurizer Thermal Predictor & Early Warning System**")

meta, models = get_model_bundle()
cfg = meta["config"]
horizon_s = meta["horizon_s"]
divert_thr = cfg["target"]["divert_threshold_c"]
margin = cfg["target"]["alert_margin_c"]

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Configuration")
model_options = ["Ensemble"] + [m for m in meta["models"] if m != "Ridge"] + ["Ridge"]
selected_model = st.sidebar.selectbox("Predictive Model", model_options, index=0)

data_source = st.sidebar.radio("Data Source", ["Preloaded Shift Logs", "Upload New Shift CSV"])

csv_file = None
if data_source == "Preloaded Shift Logs":
    available_csvs = sorted(list(Path("csv_files").glob("*.csv")), key=lambda p: p.name)
    csv_names = [p.name for p in available_csvs]
    default_idx = csv_names.index("pasteurizer_dataset41.csv") if "pasteurizer_dataset41.csv" in csv_names else 0
    selected_csv_name = st.sidebar.selectbox("Select Shift Log", csv_names, index=default_idx)
    csv_file = Path("csv_files") / selected_csv_name
else:
    uploaded = st.sidebar.file_uploader("Upload SCADA CSV File", type=["csv"])
    if uploaded is not None:
        csv_file = uploaded

st.sidebar.markdown(f"""
- **Forecast Horizon**: `+{horizon_s} seconds`
- **Divert Threshold**: `{divert_thr} °C`
- **Alert Margin**: `+{margin} °C`
""")

# -----------------------------------------------------------------------------
# MAIN CONTENT
# -----------------------------------------------------------------------------
st.title("HTST Pasteurizer · Thermal Digital Twin & Early Warning")
st.markdown("Real-time predictive forecasting of holding tube temperature (`TT-06`) to anticipate under-pasteurization diversions before they trigger safety valves.")

if csv_file is None:
    st.info("👈 Please select or upload a shift CSV file from the sidebar to inspect telemetry and forecasts.")
    st.stop()

with st.spinner("Processing SCADA telemetry and generating predictive horizon..."):
    df = process_scada_csv(csv_file, cfg["data"]["timestamp_col"])
    data_tuple = predict_trajectory(df, meta, models, selected_model)

if data_tuple[0] is None:
    st.warning("No continuous active milk production segments found in this file (minimum feed flow ≥ 1,000 L/H required for ≥ 60s).")
    st.stop()

data, pred_temp, actual_future = data_tuple

divert_alerts = pred_temp < (divert_thr + margin)
n_alerts = int(divert_alerts.sum())
pct_alerts = (n_alerts / len(pred_temp)) * 100

# Quick status banner
col_s1, col_s2, col_s3, col_s4, col_s5 = st.columns(5)
curr_temp = data.current.values[-1]
latest_pred = pred_temp[-1]
temp_delta = latest_pred - curr_temp

with col_s1:
    st.metric("Current Temp (TT-06)", f"{curr_temp:.2f} °C")
with col_s2:
    st.metric(f"Forecast (+{horizon_s}s)", f"{latest_pred:.2f} °C", delta=f"{temp_delta:+.2f} °C")
with col_s3:
    st.metric("Divert Threshold", f"{divert_thr:.1f} °C")
with col_s4:
    st.metric("Divert Alerts Count", f"{n_alerts} ({pct_alerts:.1f}%)")
with col_s5:
    status_text = "NORMAL PRODUCTION" if latest_pred >= divert_thr + margin else "DIVERT RISK IMMINENT"
    badge_class = "status-badge-ok" if latest_pred >= divert_thr + margin else "status-badge-warn"
    st.markdown(f"<div style='margin-top: 15px;'><span class='{badge_class}'>{status_text}</span></div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# TABS
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Shift Telemetry & Forecast",
    "⚡ Closed-Loop Steam Optimizer",
    "🧪 What-If Steam Simulator",
    "📊 Model Benchmark & Diagnostics",
])

with tab1:
    st.subheader(f"Shift Telemetry & {horizon_s}s Lookahead ({selected_model})")

    # Interactive Plotly Subplots
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Temperature Profiles & Lookahead (°C)", "Actuator & Process Dynamics (Steam CV % & Feed Flow kL/H)"),
        row_heights=[0.65, 0.35]
    )

    t_actual = pd.to_datetime(data.timestamp.values)
    t_forecast = t_actual + pd.Timedelta(seconds=horizon_s)

    # Upper panel: Temperatures
    fig.add_trace(go.Scatter(
        x=t_actual, y=data.current.values,
        mode="lines", name="Actual Holding Out (TT-06)",
        line=dict(color="#e6e9f0", width=1.5)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=t_forecast, y=pred_temp,
        mode="lines", name=f"Forecast +{horizon_s}s ({selected_model})",
        line=dict(color="#4cc9f0", width=1.8)
    ), row=1, col=1)

    if actual_future is not None and not np.isnan(actual_future).all():
        ok = ~np.isnan(actual_future)
        fig.add_trace(go.Scatter(
            x=t_forecast[ok], y=actual_future[ok],
            mode="lines", name="Ground Truth at Target Time",
            line=dict(color="#a0aec0", width=1, dash="dot"),
            opacity=0.7
        ), row=1, col=1)

    # Divert line
    fig.add_hline(
        y=divert_thr, line_dash="dash", line_color="#ef476f",
        annotation_text=f"FDV Divert Threshold ({divert_thr}°C)",
        annotation_position="bottom right",
        row=1, col=1
    )

    # Lower panel: Steam CV and Flow
    steam_col = "steam_cv"
    flow_col = cfg["production"]["flow_col"]

    if steam_col in data.X.columns:
        fig.add_trace(go.Scatter(
            x=t_actual, y=data.X[steam_col],
            mode="lines", name="Steam Control Valve (%)",
            line=dict(color="#ffb703", width=1.2)
        ), row=2, col=1)

    if flow_col in data.X.columns:
        fig.add_trace(go.Scatter(
            x=t_actual, y=data.X[flow_col] / 1000.0,
            mode="lines", name="Milk Feed Flow (kL/H)",
            line=dict(color="#06d6a0", width=1.2)
        ), row=2, col=1)

    fig.update_layout(
        template="plotly_dark",
        height=650,
        hovermode="x unified",
        margin=dict(l=40, r=40, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True)

    # Evaluation on this file if ground truth exists
    if actual_future is not None and not np.isnan(actual_future).all():
        ok = ~np.isnan(actual_future)
        m = regression_metrics(actual_future[ok], pred_temp[ok], data.current.values[ok])
        pm = regression_metrics(actual_future[ok], data.current.values[ok])
        sk = skill_score(m["MAE_C"], pm["MAE_C"])
        dv = divert_warning_metrics(actual_future[ok], pred_temp[ok], divert_thr, margin)

        st.markdown("### 🎯 Shift Accuracy Metrics")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Model MAE", f"{m['MAE_C']:.3f} °C", delta=f"{m['MAE_C'] - pm['MAE_C']:.3f} °C vs naive", delta_color="inverse")
        c2.metric("Skill vs Persistence", f"{100*sk:+.1f} %")
        c3.metric("R² Score", f"{m['R2']:.3f}")
        c4.metric("Divert Event Recall", f"{dv['recall']*100:.1f} %" if not np.isnan(dv['recall']) else "N/A")

    # Export CSV
    export_df = pd.DataFrame({
        "timestamp": t_actual,
        "forecast_timestamp": t_forecast,
        "actual_holding_temp_c": data.current.values,
        "predicted_holding_temp_c": np.round(pred_temp, 3),
        "divert_warning": divert_alerts,
    })
    csv_bytes = export_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Predictions CSV",
        data=csv_bytes,
        file_name="pasteurizer_predictions.csv",
        mime="text/csv",
    )

with tab2:
    st.subheader("⚡ Closed-Loop Steam Optimizer (MPC Engine)")
    st.markdown("""
    **Inverse Thermal Optimization**: Finds the exact Steam Control Valve opening ($CV^*$) required to keep the pasteurizer at the target setpoint
    without wasting boiler steam or risking an under-pasteurization divert trip ($< 82.0^\\circ\\text{C}$).
    """)

    opt_engine = SteamOptimizer(meta, models, model_name=selected_model)

    c_opt1, c_opt2 = st.columns([1, 2])
    with c_opt1:
        st.markdown("#### 🎯 Optimization Settings")
        target_sp = st.slider("Target Holding Temp Setpoint (°C)", min_value=83.0, max_value=89.0, value=85.0, step=0.5)
        max_slew = st.slider("Max Valve Slew Limit (Δ CV % per step)", min_value=5.0, max_value=25.0, value=15.0, step=1.0)
        
        st.info(f"""
        - **Legal Safety Limit**: `{divert_thr} °C`
        - **Safe Operating Floor**: `{divert_thr + margin} °C`
        - **Surrogate Model**: `{selected_model}`
        """)

        # Optimize current real-time state
        latest_row = data.X.iloc[-1]
        step_opt = opt_engine.optimize_step(latest_row, curr_temp, target_temp=target_sp, max_slew_cv=max_slew)

    with c_opt2:
        st.markdown("#### 💡 Live Operator Recommendation")
        
        act_col1, act_col2, act_col3 = st.columns(3)
        with act_col1:
            st.metric("Operator Steam CV", f"{step_opt['current_cv']:.1f} %")
        with act_col2:
            st.metric("AI Optimal Steam CV", f"{step_opt['optimal_cv']:.1f} %", delta=f"{step_opt['delta_cv']:+.1f} %", delta_color="inverse")
        with act_col3:
            st.metric("Resulting Temp (+30s)", f"{step_opt['predicted_temp_optimal']:.2f} °C", delta=f"{step_opt['predicted_temp_optimal'] - target_sp:+.2f} °C vs target")

        # Action banner
        badge_style = "status-badge-ok" if "THROTTLE" in step_opt["action"] else ("status-badge-warn" if "INCREASE" in step_opt["action"] else "metric-card")
        st.markdown(f"**Recommended Control Action**: <span class='{badge_style}' style='font-size: 1.1em;'>{step_opt['action']}</span>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### ♨️ Real-Time Waste Rate (Overheating Penalty)")
        w_c1, w_c2, w_c3 = st.columns(3)
        w_c1.metric("Steam Waste Rate", f"{step_opt['steam_waste_kg_h']:.1f} kg/h")
        w_c2.metric("Excess Thermal Power", f"{step_opt['power_waste_kw']:.1f} kW")
        w_c3.metric("Financial Loss Rate", f"${step_opt['cost_waste_usd_h']:.2f} / hr")

    st.markdown("---")
    st.subheader("📊 Full-Shift Closed-Loop Simulation & Energy Audit")
    st.markdown("Simulate what would happen if the AI Steam Optimizer operated the steam control valve autonomously throughout this entire production shift:")

    if st.button("🚀 Run Shift Energy & Steam Audit", type="primary"):
        with st.spinner("Simulating closed-loop trajectory across all shift timesteps..."):
            sim_df = optimize_shift_trajectory(data, meta, models, target_temp=target_sp, model_name=selected_model, step_subsample=4)

        tot_steam_kg = float(sim_df["cumulative_steam_saved_kg"].iloc[-1])
        tot_energy_kwh = float(sim_df["cumulative_energy_saved_kwh"].iloc[-1])
        tot_cost_usd = float(sim_df["cumulative_cost_saved_usd"].iloc[-1])
        annual_projection_usd = tot_cost_usd * 3.0 * 365.0  # 3 shifts/day * 365 days

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Total Steam Saved (Shift)", f"{tot_steam_kg:,.1f} kg")
        kpi2.metric("Thermal Energy Saved", f"{tot_energy_kwh:,.1f} kWh")
        kpi3.metric("Shift Cost Savings", f"${tot_cost_usd:,.2f}")
        kpi4.metric("Projected Annual Savings", f"${annual_projection_usd:,.0f} / yr", delta="+ ROI Impact")

        # Optimization Trajectory Plots
        t_sim = pd.to_datetime(sim_df["timestamp"])
        fig_opt = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.06,
            subplot_titles=(
                "Steam Control Valve: Actual Operator vs AI Optimal (%)",
                "Holding Tube Temperature: Actual vs AI Controlled (°C)",
                "Cumulative Steam Mass Conserved (kg)"
            ),
            row_heights=[0.38, 0.38, 0.24]
        )

        # 1. Steam CV
        fig_opt.add_trace(go.Scatter(
            x=t_sim, y=sim_df["current_cv"],
            mode="lines", name="Actual Operator Steam CV (%)",
            line=dict(color="#ffb703", width=1.5)
        ), row=1, col=1)

        fig_opt.add_trace(go.Scatter(
            x=t_sim, y=sim_df["optimal_cv"],
            mode="lines", name="AI Optimal Recommended CV (%)",
            line=dict(color="#06d6a0", width=1.8)
        ), row=1, col=1)

        # 2. Temperature
        fig_opt.add_trace(go.Scatter(
            x=t_sim, y=sim_df["current_temp"],
            mode="lines", name="Actual Holding Temp (°C)",
            line=dict(color="#e6e9f0", width=1.5)
        ), row=2, col=1)

        fig_opt.add_trace(go.Scatter(
            x=t_sim, y=sim_df["predicted_temp_optimal"],
            mode="lines", name="AI Optimized Temp (°C)",
            line=dict(color="#4cc9f0", width=1.8)
        ), row=2, col=1)

        fig_opt.add_hline(
            y=target_sp, line_dash="dash", line_color="#06d6a0",
            annotation_text=f"Target Setpoint ({target_sp}°C)",
            row=2, col=1
        )
        fig_opt.add_hline(
            y=divert_thr, line_dash="dot", line_color="#ef476f",
            annotation_text=f"FDV Divert Limit ({divert_thr}°C)",
            row=2, col=1
        )

        # 3. Cumulative Steam Saved
        fig_opt.add_trace(go.Scatter(
            x=t_sim, y=sim_df["cumulative_steam_saved_kg"],
            mode="lines", name="Cumulative Steam Saved (kg)",
            line=dict(color="#06d6a0", width=2.2),
            fill="tozeroy",
            fillcolor="rgba(6, 214, 160, 0.15)"
        ), row=3, col=1)

        fig_opt.update_layout(
            template="plotly_dark",
            height=750,
            hovermode="x unified",
            margin=dict(l=40, r=40, t=40, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_opt, use_container_width=True)

        # Export audit dataframe
        audit_csv = sim_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Shift Energy Audit Report (CSV)",
            data=audit_csv,
            file_name="steam_optimizer_shift_audit.csv",
            mime="text/csv",
        )

with tab3:
    st.subheader("🧪 Steam Control What-If Simulation")
    st.markdown("""
    Test hypothetical operator interventions on the latest production state:
    Simulate adjusting the **Steam Control Valve (CV %)** or experiencing a sudden **Milk Feed Flow Surge/Drop** to forecast resulting thermal changes.
    """)

    sim_col1, sim_col2 = st.columns([1, 2])
    with sim_col1:
        st.markdown("#### Simulation Controls")
        cv_adjust = st.slider("Steam Valve Adjustment (Δ %)", min_value=-20.0, max_value=+20.0, value=0.0, step=1.0)
        flow_adjust = st.slider("Milk Flow Rate Adjustment (Δ %)", min_value=-30.0, max_value=+30.0, value=0.0, step=5.0)

        # Apply perturbation to the last feature vector
        X_last = data.X.iloc[[-1]].copy()
        if "steam_cv" in X_last.columns:
            X_last["steam_cv"] = np.clip(X_last["steam_cv"] + cv_adjust, 0, 100)
        if cfg["production"]["flow_col"] in X_last.columns:
            X_last[cfg["production"]["flow_col"]] = np.clip(X_last[cfg["production"]["flow_col"]] * (1 + flow_adjust / 100.0), 0, 40000)

        # Rerun prediction on perturbed sample
        if selected_model == "Ensemble":
            sim_delta = np.mean([models[m].predict(X_last) for m in meta["ensemble_members"]], axis=0)[0]
        else:
            sim_delta = models[selected_model].predict(X_last)[0]
        sim_pred = curr_temp + sim_delta

    with sim_col2:
        st.markdown("#### Projected Thermal Impact (+30s)")
        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Base Forecast (+30s)", f"{latest_pred:.2f} °C")
        sc2.metric("Simulated Forecast (+30s)", f"{sim_pred:.2f} °C", delta=f"{sim_pred - latest_pred:+.2f} °C")
        
        sim_status = "SAFE" if sim_pred >= divert_thr + margin else "UNDERPASTEURIZATION HAZARD"
        sim_badge = "status-badge-ok" if sim_pred >= divert_thr + margin else "status-badge-warn"
        sc3.markdown(f"<div style='margin-top: 15px;'><span class='{sim_badge}'>{sim_status}</span></div>", unsafe_allow_html=True)

        # Gauge Chart
        gauge_fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=sim_pred,
            domain={'x': [0, 1], 'y': [0, 1]},
            delta={'reference': latest_pred, 'increasing': {'color': "#4cc9f0"}},
            gauge={
                'axis': {'range': [78, 96], 'tickwidth': 1, 'tickcolor': "white"},
                'bar': {'color': "#4cc9f0" if sim_pred >= divert_thr + margin else "#ef476f"},
                'bgcolor': "#161b26",
                'threshold': {
                    'line': {'color': "#ef476f", 'width': 4},
                    'thickness': 0.75,
                    'value': divert_thr
                },
                'steps': [
                    {'range': [78, divert_thr], 'color': 'rgba(239, 71, 111, 0.25)'},
                    {'range': [divert_thr, divert_thr + margin], 'color': 'rgba(255, 183, 3, 0.25)'},
                    {'range': [divert_thr + margin, 96], 'color': 'rgba(6, 214, 160, 0.25)'}
                ]
            }
        ))
        gauge_fig.update_layout(template="plotly_dark", height=280, margin=dict(l=30, r=30, t=30, b=30))
        st.plotly_chart(gauge_fig, use_container_width=True)

with tab3:
    st.subheader("📊 Offline Benchmark Metrics & Visual Diagnostics")

    # Load metrics from latest run
    metrics_csv = Path("outputs/latest/metrics.csv")
    if metrics_csv.exists():
        m_df = pd.read_csv(metrics_csv, index_col=0)
        st.markdown("#### Test-Set Benchmark (Unseen Operational Week)")
        st.dataframe(m_df.style.format("{:.4f}").highlight_min(subset=["MAE_C", "RMSE_C"], color="#06d6a040"), use_container_width=True)

    fig_select = st.selectbox("Select Diagnostic Plot to View", [
        "30s-Ahead Test Forecasts (test_forecasts.png)",
        "Model Comparison Benchmark (model_comparison.png)",
        "Training & Validation Learning Curves (learning_curves.png)",
        "Parity Density Hexbins (parity.png)",
        "Residual Error & Production Regimes (error_analysis.png)",
        "Telemetry Feature Importance (feature_importance.png)",
        "7-Day SCADA Overview (data_overview.png)"
    ])

    fig_map = {
        "30s-Ahead Test Forecasts (test_forecasts.png)": "figures/test_forecasts.png",
        "Model Comparison Benchmark (model_comparison.png)": "figures/model_comparison.png",
        "Training & Validation Learning Curves (learning_curves.png)": "figures/learning_curves.png",
        "Parity Density Hexbins (parity.png)": "figures/parity.png",
        "Residual Error & Production Regimes (error_analysis.png)": "figures/error_analysis.png",
        "Telemetry Feature Importance (feature_importance.png)": "figures/feature_importance.png",
        "7-Day SCADA Overview (data_overview.png)": "figures/data_overview.png"
    }

    target_fig = Path(fig_map[fig_select])
    if target_fig.exists():
        st.image(str(target_fig), use_container_width=True)
    else:
        st.warning(f"Figure {target_fig} not found.")
