export interface SystemStatus {
  status: string
  models_loaded: string[]
  ensemble_members: string[]
  forecast_horizon_seconds: number
  target_col: string
  divert_threshold_c: number
  default_target_setpoint_c: number
}

export interface ShiftInfo {
  id: string
  name: string
  total_points: number
  start_time: string
  end_time: string
  avg_temp: number
  avg_cv: number
  avg_flow: number
}

export interface TelemetryPoint {
  timestamp: string
  holding_temp: number
  lookahead_temp: number
  steam_cv: number
  feed_flow: number
  divert_risk: boolean
}

export interface OptimizeStepResult {
  current_cv: number
  optimal_cv: number
  delta_cv: number
  current_temp: number
  target_temp: number
  predicted_temp_optimal: number
  action: string
  steam_waste_kg_h: number
  cost_waste_usd_h: number
  power_waste_kw: number
}

export interface ShiftSimulationKPIs {
  total_steam_waste_actual_kg: number
  total_steam_waste_opt_kg: number
  steam_saved_kg: number
  excess_heat_eliminated_kwh: number
  cost_saved_usd: number
  divert_trips_actual: number
  divert_trips_opt: number
  hours_simulated: number
}

export interface TrajectoryPoint {
  timestamp: string
  actual_temp: number
  target_temp: number
  actual_cv: number
  optimized_cv: number
  steam_waste_kg_h: number
  divert_under_actual: boolean
  divert_under_opt: boolean
}

export interface WhatIfResult {
  baseline_temp: number
  baseline_cv: number
  baseline_flow: number
  simulated_cv: number
  simulated_flow: number
  predicted_temp: number
  predicted_delta: number
  divert_warning: boolean
}

export interface ModelBenchmark {
  model: string
  mae: number
  rmse: number
  r2_level: number
  skill_score: string
  divert_recall: string
  is_ensemble: boolean
}
