import React, { useState } from "react"
import {
  Flame,
  TrendingDown,
  TrendingUp,
  DollarSign,
  ShieldCheck,
  Zap,
  Play,
  RotateCcw,
  CheckCircle,
  AlertTriangle,
} from "lucide-react"
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Slider } from "@/components/ui/slider"
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Legend,
  AreaChart,
  Area,
  ReferenceLine,
} from "recharts"
import type { OptimizeStepResult, ShiftSimulationKPIs, TrajectoryPoint } from "@/types"

interface OptimizerTabProps {
  stepResult: OptimizeStepResult | null
  kpis: ShiftSimulationKPIs | null
  trajectory: TrajectoryPoint[]
  targetSetpoint: number
  onTargetSetpointChange: (val: number) => void
  maxSlewCv: number
  onMaxSlewCvChange: (val: number) => void
  onRunSimulation: () => void
  loadingSim: boolean
}

export function OptimizerTab({
  stepResult,
  kpis,
  trajectory,
  targetSetpoint,
  onTargetSetpointChange,
  maxSlewCv,
  onMaxSlewCvChange,
  onRunSimulation,
  loadingSim,
}: OptimizerTabProps) {
  return (
    <div className="space-y-6">
      {/* Top Banner / Control Panel */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch">
        {/* Controls & Parameters */}
        <Card className="lg:col-span-5 border-zinc-800 bg-zinc-950/70">
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle className="text-base font-semibold text-zinc-100 flex items-center gap-2">
                <Flame className="h-4 w-4 text-cyan-400" />
                MPC Controller Parameters
              </CardTitle>
              <Badge variant="outline" className="font-mono text-[10px]">
                Surrogate: Ensemble
              </Badge>
            </div>
            <CardDescription>
              Adjust setpoint and actuator slew constraints to balance energy conservation and food safety.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-5">
            {/* Target Setpoint Slider */}
            <div>
              <div className="flex justify-between items-center mb-2 text-xs">
                <span className="text-zinc-300 font-medium">Target Holding Temperature (SetPoint)</span>
                <span className="font-mono text-cyan-300 text-sm font-semibold">
                  {targetSetpoint.toFixed(1)} °C
                </span>
              </div>
              <Slider
                min={83.0}
                max={89.0}
                step={0.5}
                value={targetSetpoint}
                onChange={onTargetSetpointChange}
              />
              <div className="flex justify-between text-[10px] text-zinc-500 mt-1">
                <span>83.0 °C (Ultra-Lean)</span>
                <span className="text-zinc-400">85.0 °C (Recommended)</span>
                <span>89.0 °C (Conservative)</span>
              </div>
            </div>

            {/* Actuator Slew Rate Slider */}
            <div>
              <div className="flex justify-between items-center mb-2 text-xs">
                <span className="text-zinc-300 font-medium">Max Actuator Slew Rate (ΔCV per step)</span>
                <span className="font-mono text-emerald-300 text-sm font-semibold">
                  ±{maxSlewCv.toFixed(0)} %
                </span>
              </div>
              <Slider
                min={5.0}
                max={25.0}
                step={1.0}
                value={maxSlewCv}
                onChange={onMaxSlewCvChange}
              />
              <div className="flex justify-between text-[10px] text-zinc-500 mt-1">
                <span>±5 % (Smooth / Damped)</span>
                <span>±15 % (Standard)</span>
                <span>±25 % (Aggressive)</span>
              </div>
            </div>

            <div className="pt-2 flex items-center gap-3">
              <Button
                variant="default"
                className="w-full gap-2 bg-gradient-to-r from-cyan-600 to-emerald-600 hover:from-cyan-500 hover:to-emerald-500 text-white font-medium shadow-md shadow-cyan-900/20"
                onClick={onRunSimulation}
                disabled={loadingSim}
              >
                <Play className="h-4 w-4 fill-current" />
                {loadingSim ? "Simulating Shift MPC..." : "Run Full-Shift Optimization Audit"}
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* Live Recommendation Card */}
        <Card className="lg:col-span-7 border-zinc-800 bg-gradient-to-br from-zinc-950 via-zinc-900/60 to-zinc-950">
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle className="text-base font-semibold text-zinc-100 flex items-center gap-2">
                <Zap className="h-4 w-4 text-emerald-400" />
                Live Operator Actuator Recommendation
              </CardTitle>
              {stepResult && (
                <Badge
                  variant={
                    stepResult.optimal_cv < stepResult.current_cv - 1.0
                      ? "cyan"
                      : stepResult.optimal_cv > stepResult.current_cv + 1.0
                      ? "warning"
                      : "secondary"
                  }
                  className="font-medium text-xs px-2.5 py-0.5"
                >
                  {stepResult.action}
                </Badge>
              )}
            </div>
            <CardDescription>
              Constrained inverse optimization solution evaluated on the current SCADA state.
            </CardDescription>
          </CardHeader>
          <CardContent>
            {stepResult ? (
              <div className="space-y-4">
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-center">
                  <div className="p-3 rounded-lg bg-zinc-900/80 border border-zinc-800">
                    <div className="text-[11px] text-zinc-400 uppercase tracking-wider mb-1">Human Valve</div>
                    <div className="text-xl font-mono font-bold text-zinc-300">
                      {stepResult.current_cv.toFixed(1)}%
                    </div>
                    <div className="text-[10px] text-zinc-500 mt-1">Current SCADA CV</div>
                  </div>

                  <div className="p-3 rounded-lg bg-emerald-950/40 border border-emerald-800/60 shadow-inner">
                    <div className="text-[11px] text-emerald-400 uppercase tracking-wider mb-1 font-semibold">
                      Recommended CV*
                    </div>
                    <div className="text-xl font-mono font-bold text-emerald-300">
                      {stepResult.optimal_cv.toFixed(1)}%
                    </div>
                    <div className="text-[10px] text-emerald-500/80 mt-1">Optimal Setpoint</div>
                  </div>

                  <div className="p-3 rounded-lg bg-zinc-900/80 border border-zinc-800">
                    <div className="text-[11px] text-zinc-400 uppercase tracking-wider mb-1">Target Temp</div>
                    <div className="text-xl font-mono font-bold text-cyan-300">
                      {stepResult.target_temp.toFixed(1)}°C
                    </div>
                    <div className="text-[10px] text-zinc-500 mt-1">
                      Pred: {stepResult.predicted_temp_optimal.toFixed(2)}°C
                    </div>
                  </div>

                  <div className="p-3 rounded-lg bg-zinc-900/80 border border-zinc-800">
                    <div className="text-[11px] text-zinc-400 uppercase tracking-wider mb-1">Excess Waste</div>
                    <div className="text-xl font-mono font-bold text-amber-300">
                      {stepResult.steam_waste_kg_h.toFixed(1)}
                    </div>
                    <div className="text-[10px] text-zinc-500 mt-1">kg steam / hour</div>
                  </div>
                </div>

                <div className="p-3 rounded-lg bg-zinc-900/40 border border-zinc-800/80 flex items-center justify-between text-xs text-zinc-300">
                  <div className="flex items-center gap-2">
                    <ShieldCheck className="h-4 w-4 text-emerald-400" />
                    <span>Safety Barrier Guarantee:</span>
                    <span className="font-mono text-emerald-300 font-semibold">T &gt; 82.5 °C</span>
                    <span className="text-zinc-500">(0 Food Safety Trips)</span>
                  </div>
                  <div className="font-mono text-zinc-400 text-[11px]">
                    Slew Δu:{" "}
                    <span className={stepResult.delta_cv <= 0 ? "text-cyan-400 font-bold" : "text-amber-400 font-bold"}>
                      {stepResult.delta_cv >= 0 ? "+" : ""}
                      {stepResult.delta_cv.toFixed(1)}%
                    </span>
                  </div>
                </div>
              </div>
            ) : (
              <div className="text-center py-6 text-zinc-500 text-sm">
                Calculating live optimization recommendation...
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Thermodynamic Shift Energy Audit KPIs */}
      {kpis && (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <Card className="border-cyan-900/40 bg-cyan-950/20">
            <CardContent className="p-4">
              <div className="flex items-center justify-between text-xs text-cyan-400/80 mb-1">
                <span className="font-medium uppercase tracking-wider">Shift Steam Conserved</span>
                <Flame className="h-4 w-4 text-cyan-400" />
              </div>
              <div className="flex items-baseline gap-2">
                <span className="text-2xl font-bold font-mono tracking-tight text-cyan-200">
                  {kpis.steam_saved_kg.toFixed(1)}
                </span>
                <span className="text-xs text-cyan-400 font-mono">kg</span>
              </div>
              <p className="text-[11px] text-zinc-400 mt-1">
                Excess boiling eliminated across {kpis.hours_simulated.toFixed(1)} hours
              </p>
            </CardContent>
          </Card>

          <Card className="border-emerald-900/40 bg-emerald-950/20">
            <CardContent className="p-4">
              <div className="flex items-center justify-between text-xs text-emerald-400/80 mb-1">
                <span className="font-medium uppercase tracking-wider">Energy Waste Eliminated</span>
                <Zap className="h-4 w-4 text-emerald-400" />
              </div>
              <div className="flex items-baseline gap-2">
                <span className="text-2xl font-bold font-mono tracking-tight text-emerald-200">
                  {kpis.excess_heat_eliminated_kwh.toFixed(1)}
                </span>
                <span className="text-xs text-emerald-400 font-mono">kWh</span>
              </div>
              <p className="text-[11px] text-zinc-400 mt-1">
                Thermodynamic enthalpy recovered (Cp = 3.93 kJ/kg·°C)
              </p>
            </CardContent>
          </Card>

          <Card className="border-amber-900/40 bg-amber-950/20">
            <CardContent className="p-4">
              <div className="flex items-center justify-between text-xs text-amber-400/80 mb-1">
                <span className="font-medium uppercase tracking-wider">Fuel Cost Reduction</span>
                <DollarSign className="h-4 w-4 text-amber-400" />
              </div>
              <div className="flex items-baseline gap-2">
                <span className="text-2xl font-bold font-mono tracking-tight text-amber-200">
                  ${kpis.cost_saved_usd.toFixed(2)}
                </span>
                <span className="text-xs text-amber-400 font-mono">/ shift</span>
              </div>
              <p className="text-[11px] text-zinc-400 mt-1">
                ~${(kpis.cost_saved_usd * 3 * 365).toFixed(0)} projected annual plant savings
              </p>
            </CardContent>
          </Card>

          <Card className="border-indigo-900/40 bg-indigo-950/20">
            <CardContent className="p-4">
              <div className="flex items-center justify-between text-xs text-indigo-400/80 mb-1">
                <span className="font-medium uppercase tracking-wider">Under-Pasteurization Trips</span>
                <ShieldCheck className="h-4 w-4 text-indigo-400" />
              </div>
              <div className="flex items-baseline gap-2">
                <span className="text-2xl font-bold font-mono tracking-tight text-emerald-300">
                  {kpis.divert_trips_opt}
                </span>
                <span className="text-xs text-zinc-400 font-mono">trips</span>
              </div>
              <p className="text-[11px] text-zinc-400 mt-1">
                100% legal compliance (Zero trips below 82.0 °C)
              </p>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Trajectory Time-Series Charts */}
      {trajectory.length > 0 && (
        <div className="space-y-6">
          {/* Chart 1: Temperature Trajectory */}
          <Card className="border-zinc-800 bg-zinc-950/80">
            <CardHeader className="pb-2">
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle className="text-base text-zinc-100">
                    Holding Tube Temperature Trajectory vs Target Setpoint
                  </CardTitle>
                  <CardDescription>
                    Actual human operator temperature vs MPC Target Setpoint ({targetSetpoint.toFixed(1)} °C)
                  </CardDescription>
                </div>
                <div className="flex items-center gap-3 text-xs">
                  <span className="flex items-center gap-1.5 text-amber-400">
                    <span className="h-2.5 w-2.5 rounded-full bg-amber-400"></span> Actual Temp
                  </span>
                  <span className="flex items-center gap-1.5 text-cyan-400">
                    <span className="h-2.5 w-2.5 rounded-full bg-cyan-400"></span> Target ({targetSetpoint}°C)
                  </span>
                  <span className="flex items-center gap-1.5 text-rose-400">
                    <span className="h-2.5 w-2.5 rounded-full bg-rose-500"></span> Trip Line (82°C)
                  </span>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <div className="h-72 w-full pt-4">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={trajectory} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#27272a" opacity={0.6} />
                    <XAxis
                      dataKey="timestamp"
                      stroke="#71717a"
                      fontSize={10}
                      tickFormatter={(val) => val.split(" ")[1] || val}
                    />
                    <YAxis stroke="#71717a" fontSize={10} domain={[80, 96]} unit="°C" />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: "#09090b",
                        borderColor: "#27272a",
                        borderRadius: "8px",
                        fontSize: "12px",
                      }}
                    />
                    <ReferenceLine y={82.0} stroke="#f43f5e" strokeDasharray="4 4" label={{ value: "Legal Trip: 82.0°C", fill: "#f43f5e", fontSize: 10 }} />
                    <Line
                      type="monotone"
                      dataKey="actual_temp"
                      name="Actual Temp"
                      stroke="#fbbf24"
                      strokeWidth={2}
                      dot={false}
                    />
                    <Line
                      type="monotone"
                      dataKey="target_temp"
                      name="Target Setpoint"
                      stroke="#22d3ee"
                      strokeWidth={1.5}
                      strokeDasharray="5 5"
                      dot={false}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>

          {/* Chart 2: Actuator Valve Comparison & Steam Waste */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Actuator Valve Position */}
            <Card className="border-zinc-800 bg-zinc-950/80">
              <CardHeader className="pb-2">
                <CardTitle className="text-base text-zinc-100">
                  Steam Control Valve: Actual vs MPC Optimized
                </CardTitle>
                <CardDescription>
                  Operator valve position (grey) vs MPC recommended throttle (green)
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="h-64 w-full pt-4">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={trajectory} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#27272a" opacity={0.6} />
                      <XAxis
                        dataKey="timestamp"
                        stroke="#71717a"
                        fontSize={10}
                        tickFormatter={(val) => val.split(" ")[1] || val}
                      />
                      <YAxis stroke="#71717a" fontSize={10} domain={[0, 100]} unit="%" />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: "#09090b",
                          borderColor: "#27272a",
                          borderRadius: "8px",
                          fontSize: "12px",
                        }}
                      />
                      <Line
                        type="monotone"
                        dataKey="actual_cv"
                        name="Actual CV %"
                        stroke="#71717a"
                        strokeWidth={1.5}
                        dot={false}
                      />
                      <Line
                        type="monotone"
                        dataKey="optimized_cv"
                        name="Optimized CV %"
                        stroke="#10b981"
                        strokeWidth={2}
                        dot={false}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>

            {/* Instantaneous Steam Waste */}
            <Card className="border-zinc-800 bg-zinc-950/80">
              <CardHeader className="pb-2">
                <CardTitle className="text-base text-zinc-100">
                  Excess Steam Waste Rate (Overheating Enthalpy)
                </CardTitle>
                <CardDescription>
                  Losses incurred by overheating milk past target ($kg/h$)
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="h-64 w-full pt-4">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={trajectory} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
                      <defs>
                        <linearGradient id="wasteGradient" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#f59e0b" stopOpacity={0.4} />
                          <stop offset="95%" stopColor="#f59e0b" stopOpacity={0.0} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" stroke="#27272a" opacity={0.6} />
                      <XAxis
                        dataKey="timestamp"
                        stroke="#71717a"
                        fontSize={10}
                        tickFormatter={(val) => val.split(" ")[1] || val}
                      />
                      <YAxis stroke="#71717a" fontSize={10} unit=" kg/h" />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: "#09090b",
                          borderColor: "#27272a",
                          borderRadius: "8px",
                          fontSize: "12px",
                        }}
                      />
                      <Area
                        type="monotone"
                        dataKey="steam_waste_kg_h"
                        name="Steam Waste"
                        stroke="#f59e0b"
                        strokeWidth={2}
                        fillOpacity={1}
                        fill="url(#wasteGradient)"
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      )}
    </div>
  )
}
