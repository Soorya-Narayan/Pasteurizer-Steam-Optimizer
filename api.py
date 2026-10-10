"""FastAPI Backend for HTST Pasteurizer Digital Twin & Steam Optimizer.

Provides high-speed REST endpoints for:
- Live SCADA telemetry and lookahead forecasting
- Closed-Loop Model Predictive Control (MPC) optimization
- Full-shift thermodynamic energy audit simulation
- Interactive what-if perturbations
- Model benchmarking diagnostics
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
torch.set_num_threads(1)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.data.loader import read_csvs
from src.features.engineering import make_supervised
from src.models import MODEL_REGISTRY
from src.optimizer.steam_mpc import SteamOptimizer, ThermodynamicProfile, optimize_shift_trajectory

app = FastAPI(
    title="Pasteurizer Steam Optimizer API",
    description="Digital Twin and Model Predictive Control Engine",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global State / Cache
MODELS: Dict[str, Any] = {}
META: Dict[str, Any] = {}
OPTIMIZER: Optional[SteamOptimizer] = None
CACHED_SHIFTS: Dict[str, Any] = {}


def init_engine():
    global MODELS, META, OPTIMIZER, CACHED_SHIFTS
    meta_path = Path("outputs/latest/metadata.json")
    if not meta_path.exists():
        raise RuntimeError("No trained model metadata found in outputs/latest")
    
    META = json.loads(meta_path.read_text())
    models_dir = Path("outputs/latest/models")
    
    MODELS = {}
    for name in META["models"]:
        if name in MODEL_REGISTRY:
            MODELS[name] = MODEL_REGISTRY[name].load(models_dir)
            
    OPTIMIZER = SteamOptimizer(META, MODELS, model_name="Ensemble")
    
    # Preload Shift 41
    cfg = META["config"]
    csv_path = Path("csv_files/pasteurizer_dataset41.csv")
    if csv_path.exists():
        df = read_csvs([csv_path], cfg["data"]["timestamp_col"], progress=False)
        data, _ = make_supervised(df, cfg, with_target=False)
        
        # Precompute ensemble 30s lookahead predictions
        X_eval = data.X[META["features"]]
        members = META["ensemble_members"]
        preds = [MODELS[m].predict(X_eval) for m in members]
        ensemble_delta = np.mean(preds, axis=0)
        pred_temp = (data.current.values + ensemble_delta).astype(np.float32)
        divert_risk = pred_temp < 82.0
        
        CACHED_SHIFTS["shift41"] = {
            "name": "Shift 41 (Continuous Production)",
            "df": df,
            "data": data,
            "pred_temp": pred_temp,
            "divert_risk": divert_risk,
            "total_points": len(data.current),
            "start_time": str(data.timestamp.iloc[0]),
            "end_time": str(data.timestamp.iloc[-1]),
            "avg_temp": float(data.current.mean()),
            "avg_cv": float(data.X["steam_cv"].mean()),
            "avg_flow": float(data.X[cfg["production"]["flow_col"]].mean()),
        }


@app.on_event("startup")
def startup_event():
    init_engine()


@app.get("/api/status")
def get_status():
    return {
        "status": "online",
        "models_loaded": list(MODELS.keys()),
        "ensemble_members": META.get("ensemble_members", []),
        "forecast_horizon_seconds": META.get("config", {}).get("data", {}).get("horizon_seconds", 30),
        "target_col": META.get("config", {}).get("data", {}).get("target_col", "TT-06_PV"),
        "divert_threshold_c": 82.0,
        "default_target_setpoint_c": 85.0,
    }


@app.get("/api/shifts")
def list_shifts():
    shifts = []
    for sid, sinfo in CACHED_SHIFTS.items():
        shifts.append({
            "id": sid,
            "name": sinfo["name"],
            "total_points": sinfo["total_points"],
            "start_time": sinfo["start_time"],
            "end_time": sinfo["end_time"],
            "avg_temp": round(sinfo["avg_temp"], 2),
            "avg_cv": round(sinfo["avg_cv"], 1),
            "avg_flow": round(sinfo["avg_flow"], 0),
        })
    return {"shifts": shifts}


@app.get("/api/shifts/{shift_id}/telemetry")
def get_shift_telemetry(shift_id: str, limit: int = 400):
    if shift_id not in CACHED_SHIFTS:
        raise HTTPException(status_code=404, detail="Shift not found")
    
    sinfo = CACHED_SHIFTS[shift_id]
    data = sinfo["data"]
    pred_temp = sinfo["pred_temp"]
    divert_risk = sinfo["divert_risk"]
    cfg = META["config"]
    flow_col = cfg["production"]["flow_col"]
    
    total = len(data.current)
    step = max(1, total // limit)
    indices = np.arange(0, total, step)
    
    telemetry = []
    for idx in indices:
        telemetry.append({
            "timestamp": str(data.timestamp.iloc[idx]),
            "holding_temp": round(float(data.current.iloc[idx]), 2),
            "lookahead_temp": round(float(pred_temp[idx]), 2),
            "steam_cv": round(float(data.X["steam_cv"].iloc[idx]), 1),
            "feed_flow": round(float(data.X[flow_col].iloc[idx]), 0),
            "divert_risk": bool(divert_risk[idx]),
        })
        
    return {
        "shift_id": shift_id,
        "points": len(telemetry),
        "telemetry": telemetry,
    }


class OptimizeStepRequest(BaseModel):
    current_temp: float
    current_cv: float
    flow_rate: float = 12000.0
    target_setpoint: float = 85.0
    max_slew_cv: float = 15.0


@app.post("/api/optimizer/step")
def optimize_step_endpoint(req: OptimizeStepRequest):
    if OPTIMIZER is None:
        raise HTTPException(status_code=500, detail="Optimizer not initialized")
    
    data = CACHED_SHIFTS["shift41"]["data"]
    sample_row = data.X.iloc[len(data) // 2].copy()
    sample_row["steam_cv"] = req.current_cv
    flow_col = META["config"]["production"]["flow_col"]
    sample_row[flow_col] = req.flow_rate

    result = OPTIMIZER.optimize_step(
        sample_row,
        current_temp=req.current_temp,
        target_temp=req.target_setpoint,
        max_slew_cv=req.max_slew_cv,
    )
    return result


class ShiftSimulationRequest(BaseModel):
    shift_id: str = "shift41"
    target_setpoint: float = 85.0
    max_slew_cv: float = 15.0


@app.post("/api/optimizer/simulate")
def simulate_shift_endpoint(req: ShiftSimulationRequest):
    if req.shift_id not in CACHED_SHIFTS:
        raise HTTPException(status_code=404, detail="Shift not found")
    
    data = CACHED_SHIFTS[req.shift_id]["data"]
    df_sim, kpis = optimize_shift_trajectory(
        OPTIMIZER,
        data,
        target_temp=req.target_setpoint,
        max_slew_cv=req.max_slew_cv,
    )
    
    # Downsample trajectory for fast frontend chart rendering
    total = len(df_sim)
    step = max(1, total // 350)
    sub = df_sim.iloc[::step]
    
    trajectory = []
    for _, r in sub.iterrows():
        trajectory.append({
            "timestamp": str(r["timestamp"]),
            "actual_temp": round(float(r["actual_temp"]), 2),
            "target_temp": round(float(r["target_temp"]), 2),
            "actual_cv": round(float(r["actual_cv"]), 1),
            "optimized_cv": round(float(r["optimized_cv"]), 1),
            "steam_waste_kg_h": round(float(r["steam_waste_kg_h"]), 1),
            "divert_under_actual": bool(r["divert_under_actual"]),
            "divert_under_opt": bool(r["divert_under_opt"]),
        })
        
    return {
        "kpis": kpis,
        "trajectory": trajectory,
    }


class WhatIfRequest(BaseModel):
    delta_steam_cv: float = 0.0
    delta_flow_percent: float = 0.0


@app.post("/api/simulator/what-if")
def what_if_endpoint(req: WhatIfRequest):
    data = CACHED_SHIFTS["shift41"]["data"]
    mid_idx = len(data) // 2
    x_orig = data.X.iloc[[mid_idx]].copy()
    cur_temp = float(data.current.iloc[mid_idx])
    cur_cv = float(x_orig["steam_cv"].values[0])
    flow_col = META["config"]["production"]["flow_col"]
    orig_flow = float(x_orig[flow_col].values[0])

    new_cv = float(np.clip(cur_cv + req.delta_steam_cv, 0.0, 100.0))
    new_flow = float(max(1000.0, orig_flow * (1.0 + req.delta_flow_percent / 100.0)))

    x_pert = x_orig.copy()
    x_pert["steam_cv"] = new_cv
    x_pert[flow_col] = new_flow
    if "steam_per_kflow" in x_pert.columns:
        x_pert["steam_per_kflow"] = new_cv / (new_flow / 1000.0)

    # Re-evaluate models
    x_eval = x_pert[META["features"]]
    members = META["ensemble_members"]
    preds = [MODELS[m].predict(x_eval) for m in members]
    ensemble_delta = float(np.mean(preds, axis=0)[0])
    predicted_temp = float(cur_temp + ensemble_delta)

    return {
        "baseline_temp": round(cur_temp, 2),
        "baseline_cv": round(cur_cv, 1),
        "baseline_flow": round(orig_flow, 0),
        "simulated_cv": round(new_cv, 1),
        "simulated_flow": round(new_flow, 0),
        "predicted_temp": round(predicted_temp, 2),
        "predicted_delta": round(predicted_temp - cur_temp, 2),
        "divert_warning": predicted_temp < 82.0,
    }


@app.get("/api/benchmarks")
def get_benchmarks():
    metrics_path = Path("outputs/latest/metrics.json")
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
    else:
        metrics = {}
        
    return {
        "metrics": metrics,
        "models": [
            {
                "model": "Persistence",
                "mae": 0.557,
                "rmse": 0.822,
                "r2_level": 0.965,
                "skill_score": "0.0%",
                "divert_recall": "100%",
                "is_ensemble": False,
            },
            {
                "model": "Ridge Regression",
                "mae": 0.450,
                "rmse": 1.081,
                "r2_level": 0.939,
                "skill_score": "+19.3%",
                "divert_recall": "100%",
                "is_ensemble": False,
            },
            {
                "model": "Residual MLP",
                "mae": 0.281,
                "rmse": 0.437,
                "r2_level": 0.990,
                "skill_score": "+49.6%",
                "divert_recall": "100%",
                "is_ensemble": True,
            },
            {
                "model": "XGBoost",
                "mae": 0.266,
                "rmse": 0.444,
                "r2_level": 0.990,
                "skill_score": "+52.3%",
                "divert_recall": "100%",
                "is_ensemble": True,
            },
            {
                "model": "LightGBM",
                "mae": 0.257,
                "rmse": 0.412,
                "r2_level": 0.991,
                "skill_score": "+54.0%",
                "divert_recall": "100%",
                "is_ensemble": True,
            },
            {
                "model": "Ensemble (Best)",
                "mae": 0.260,
                "rmse": 0.418,
                "r2_level": 0.991,
                "skill_score": "+53.3%",
                "divert_recall": "100%",
                "is_ensemble": True,
            },
        ],
    }
