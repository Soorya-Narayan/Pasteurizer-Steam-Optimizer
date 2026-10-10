"""Closed-loop Steam Optimization and Model Predictive Control (MPC).

Solves the inverse problem:
Finds the optimal Steam Control Valve opening (CV* %) that maintains the milk
holding-tube temperature at the desired target setpoint (e.g. 85.0 °C) while
minimizing steam consumption and preventing under-pasteurization trips.
"""
from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "1"

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class ThermodynamicProfile:
    """Industrial pasteurization thermodynamic constants."""
    milk_density_kg_l: float = 1.03           # kg per liter of whole/toned milk
    specific_heat_milk_kj_kg_c: float = 3.93  # kJ / (kg * °C)
    latent_heat_steam_kj_kg: float = 2100.0   # kJ/kg saturated steam at 4-6 bar
    boiler_efficiency: float = 0.82           # typical industrial boiler efficiency
    steam_cost_per_kg_usd: float = 0.04       # approximate industrial steam cost ($/kg)
    divert_threshold_c: float = 82.0          # legal food safety divert temperature
    safety_margin_c: float = 0.5              # buffer above divert limit (82.5 °C minimum)


class SteamOptimizer:
    """Inverse thermal controller using trained surrogate models."""

    def __init__(
        self,
        meta: Dict[str, Any],
        models: Dict[str, Any],
        model_name: str = "Ensemble",
        profile: ThermodynamicProfile | None = None,
    ):
        self.meta = meta
        self.models = models
        self.model_name = model_name
        self.profile = profile or ThermodynamicProfile()
        self.features = meta["features"]
        self.cfg = meta["config"]

    def _predict_batch_delta(self, X_batch: pd.DataFrame) -> np.ndarray:
        """Evaluate forward surrogate model on a batch of candidate states."""
        X_eval = X_batch[self.features]
        if self.model_name == "Ensemble":
            members = self.meta["ensemble_members"]
            preds = [self.models[m].predict(X_eval) for m in members]
            return np.mean(preds, axis=0)
        else:
            return self.models[self.model_name].predict(X_eval)

    def optimize_step(
        self,
        row_features: pd.Series,
        current_temp: float,
        target_temp: float = 85.0,
        max_slew_cv: float = 15.0,
        cv_grid_points: int = 51,
        weight_tracking: float = 10.0,
        weight_slew: float = 0.08,
        weight_energy: float = 0.05,
    ) -> Dict[str, Any]:
        """Find the optimal Steam CV % for a single operating sample."""
        current_cv = float(row_features["steam_cv"])
        flow_col = self.cfg["production"]["flow_col"]
        flow = float(row_features.get(flow_col, 10000.0))

        # Search grid bounded by physical actuator limits and slew constraint
        min_cv = max(0.0, current_cv - max_slew_cv)
        max_cv = min(100.0, current_cv + max_slew_cv)
        cv_candidates = np.linspace(min_cv, max_cv, cv_grid_points, dtype=np.float32)

        # Vectorized candidate feature table construction
        r_dict = row_features.to_dict()
        batch_dict = {
            col: np.full(len(cv_candidates), r_dict[col], dtype=np.float32)
            for col in row_features.index
        }
        batch_dict["steam_cv"] = cv_candidates
        if "steam_per_kflow" in batch_dict:
            batch_dict["steam_per_kflow"] = (cv_candidates / max(0.5, flow / 1000.0)).astype(np.float32)
        for k in [1, 3, 10]:
            col_diff = f"steam_cv_diff{k}"
            col_lag = f"steam_cv_lag{k}"
            if col_diff in batch_dict and col_lag in batch_dict:
                batch_dict[col_diff] = (cv_candidates - float(r_dict[col_lag])).astype(np.float32)

        X_candidates = pd.DataFrame(batch_dict)
        pred_deltas = self._predict_batch_delta(X_candidates)
        pred_temps = current_temp + pred_deltas

        # Objective function with hard barrier on food safety divert limit
        safety_floor = self.profile.divert_threshold_c + self.profile.safety_margin_c
        
        # Penalties: tracking error, valve slew change, energy cost, and safety barrier
        tracking_costs = weight_tracking * ((pred_temps - target_temp) ** 2)
        slew_costs = weight_slew * ((cv_candidates - current_cv) ** 2)
        energy_costs = weight_energy * cv_candidates
        
        under_safe_mask = pred_temps < safety_floor
        barrier_costs = np.zeros_like(pred_temps)
        barrier_costs[under_safe_mask] = 10000.0 * ((safety_floor - pred_temps[under_safe_mask]) ** 2)

        total_costs = tracking_costs + slew_costs + energy_costs + barrier_costs

        best_idx = int(np.argmin(total_costs))
        optimal_cv = float(cv_candidates[best_idx])
        optimal_temp_pred = float(pred_temps[best_idx])

        # Energy & Steam savings vs current operation
        # If current temp > target temp, excess heating is being wasted
        excess_temp = max(0.0, current_temp - target_temp)
        mass_flow_milk_kg_h = flow * self.profile.milk_density_kg_l
        excess_heat_kj_h = mass_flow_milk_kg_h * self.profile.specific_heat_milk_kj_kg_c * excess_temp
        steam_waste_kg_h = excess_heat_kj_h / (self.profile.boiler_efficiency * self.profile.latent_heat_steam_kj_kg)
        cost_waste_usd_h = steam_waste_kg_h * self.profile.steam_cost_per_kg_usd
        power_waste_kw = excess_heat_kj_h / 3600.0

        action = "MAINTAIN"
        if optimal_cv < current_cv - 1.0:
            action = "THROTTLE DOWN (SAVE STEAM)"
        elif optimal_cv > current_cv + 1.0:
            action = "INCREASE STEAM (PREVENT DIVERT)"

        return {
            "current_cv": current_cv,
            "optimal_cv": optimal_cv,
            "delta_cv": optimal_cv - current_cv,
            "current_temp": current_temp,
            "target_temp": target_temp,
            "predicted_temp_optimal": optimal_temp_pred,
            "action": action,
            "steam_waste_kg_h": steam_waste_kg_h,
            "power_waste_kw": power_waste_kw,
            "cost_waste_usd_h": cost_waste_usd_h,
            "is_safe": bool(optimal_temp_pred >= safety_floor),
        }


def optimize_shift_trajectory(
    data,
    meta: Dict[str, Any],
    models: Dict[str, Any],
    target_temp: float = 85.0,
    model_name: str = "Ensemble",
    step_subsample: int = 1,
) -> pd.DataFrame:
    """Run closed-loop optimization across an entire production shift."""
    opt = SteamOptimizer(meta, models, model_name=model_name)
    rows = []
    indices = range(0, len(data), step_subsample)

    for i in indices:
        x_row = data.X.iloc[i]
        curr_temp = float(data.current.iloc[i])
        ts = data.timestamp.iloc[i]

        res = opt.optimize_step(x_row, curr_temp, target_temp=target_temp)
        res["timestamp"] = ts
        rows.append(res)

    df_out = pd.DataFrame(rows)
    # Compute cumulative metrics over time
    dt_hours = (df_out["timestamp"].diff().dt.total_seconds().fillna(3.0 * step_subsample)) / 3600.0
    df_out["cumulative_steam_saved_kg"] = (df_out["steam_waste_kg_h"] * dt_hours).cumsum()
    df_out["cumulative_cost_saved_usd"] = (df_out["cost_waste_usd_h"] * dt_hours).cumsum()
    df_out["cumulative_energy_saved_kwh"] = (df_out["power_waste_kw"] * dt_hours).cumsum()

    return df_out
