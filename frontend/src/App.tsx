import React, { useState, useEffect } from "react"
import { Navbar } from "@/components/Navbar"
import { MetricCards } from "@/components/MetricCards"
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs"
import { OptimizerTab } from "@/components/OptimizerTab"
import { TelemetryTab } from "@/components/TelemetryTab"
import { WhatIfTab } from "@/components/WhatIfTab"
import { BenchmarkTab } from "@/components/BenchmarkTab"
import { Flame, Activity, Sliders, Award, RefreshCw } from "lucide-react"
import { Button } from "@/components/ui/button"
import type {
  SystemStatus,
  ShiftInfo,
  TelemetryPoint,
  OptimizeStepResult,
  ShiftSimulationKPIs,
  TrajectoryPoint,
  WhatIfResult,
  ModelBenchmark,
} from "@/types"

export function App() {
  const [status, setStatus] = useState<SystemStatus | null>(null)
  const [shifts, setShifts] = useState<ShiftInfo[]>([])
  const [activeShift, setActiveShift] = useState<string>("shift41")
  const [telemetry, setTelemetry] = useState<TelemetryPoint[]>([])
  
  // Optimizer state
  const [targetSetpoint, setTargetSetpoint] = useState<number>(85.0)
  const [maxSlewCv, setMaxSlewCv] = useState<number>(15.0)
  const [stepResult, setStepResult] = useState<OptimizeStepResult | null>(null)
  const [kpis, setKpis] = useState<ShiftSimulationKPIs | null>(null)
  const [trajectory, setTrajectory] = useState<TrajectoryPoint[]>([])
  const [loadingSim, setLoadingSim] = useState<boolean>(false)

  // What-if state
  const [deltaCv, setDeltaCv] = useState<number>(0.0)
  const [deltaFlow, setDeltaFlow] = useState<number>(0.0)
  const [whatIfResult, setWhatIfResult] = useState<WhatIfResult | null>(null)

  // Benchmarks state
  const [benchmarks, setBenchmarks] = useState<ModelBenchmark[]>([])
  const [loading, setLoading] = useState<boolean>(true)

  // 1. Initial Data Fetch
  useEffect(() => {
    async function initData() {
      try {
        setLoading(true)
        const [statusRes, shiftsRes, benchmarksRes] = await Promise.all([
          fetch("/api/status").then((r) => r.json()),
          fetch("/api/shifts").then((r) => r.json()),
          fetch("/api/benchmarks").then((r) => r.json()),
        ])
        setStatus(statusRes)
        setShifts(shiftsRes.shifts || [])
        setBenchmarks(benchmarksRes.models || [])

        // Fetch telemetry
        const telRes = await fetch(`/api/shifts/shift41/telemetry?limit=300`).then((r) => r.json())
        setTelemetry(telRes.telemetry || [])

        // Initial optimization run
        runShiftSimulation(85.0, 15.0)
        fetchWhatIf(0.0, 0.0)
      } catch (err) {
        console.error("Failed to load initial data", err)
      } finally {
        setLoading(false)
      }
    }
    initData()
  }, [])

  // 2. Fetch Single Step Recommendation when parameters change
  useEffect(() => {
    if (telemetry.length === 0) return
    const midSample = telemetry[Math.floor(telemetry.length / 2)]
    
    fetch("/api/optimizer/step", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        current_temp: midSample.holding_temp,
        current_cv: midSample.steam_cv,
        flow_rate: midSample.feed_flow,
        target_setpoint: targetSetpoint,
        max_slew_cv: maxSlewCv,
      }),
    })
      .then((r) => r.json())
      .then((data) => setStepResult(data))
      .catch((err) => console.error("Error optimizing step", err))
  }, [targetSetpoint, maxSlewCv, telemetry])

  // 3. Run Shift Simulation
  async function runShiftSimulation(setpoint: number, slew: number) {
    try {
      setLoadingSim(true)
      const res = await fetch("/api/optimizer/simulate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          shift_id: activeShift,
          target_setpoint: setpoint,
          max_slew_cv: slew,
        }),
      }).then((r) => r.json())

      setKpis(res.kpis)
      setTrajectory(res.trajectory || [])
    } catch (err) {
      console.error("Failed to simulate shift trajectory", err)
    } finally {
      setLoadingSim(false)
    }
  }

  // 4. What-If Calculation
  async function fetchWhatIf(dCv: number, dFlow: number) {
    try {
      const res = await fetch("/api/simulator/what-if", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          delta_steam_cv: dCv,
          delta_flow_percent: dFlow,
        }),
      }).then((r) => r.json())
      setWhatIfResult(res)
    } catch (err) {
      console.error("Failed to compute what-if", err)
    }
  }

  function handleDeltaCvChange(val: number) {
    setDeltaCv(val)
    fetchWhatIf(val, deltaFlow)
  }

  function handleDeltaFlowChange(val: number) {
    setDeltaFlow(val)
    fetchWhatIf(deltaCv, val)
  }

  function handleResetWhatIf() {
    setDeltaCv(0.0)
    setDeltaFlow(0.0)
    fetchWhatIf(0.0, 0.0)
  }

  // Current SCADA sample for top metrics
  const latestSample = telemetry.length > 0 ? telemetry[telemetry.length - 1] : null
  const divertCount = telemetry.filter((t) => t.divert_risk).length

  return (
    <div className="min-h-screen bg-[#09090b] text-zinc-100 selection:bg-cyan-500 selection:text-black">
      {/* Background radial gradient highlights */}
      <div className="fixed inset-0 pointer-events-none opacity-20">
        <div className="absolute top-0 left-1/4 w-96 h-96 bg-cyan-600 rounded-full blur-[140px]" />
        <div className="absolute top-1/3 right-1/4 w-96 h-96 bg-emerald-600 rounded-full blur-[160px]" />
      </div>

      <Navbar
        status={status?.status || "online"}
        modelsCount={status?.ensemble_members.length || 3}
        horizonSeconds={status?.forecast_horizon_seconds || 30}
      />

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8 relative z-10">
        {/* Metric Cards Banner */}
        <MetricCards
          currentTemp={latestSample?.holding_temp || 0}
          lookaheadTemp={latestSample?.lookahead_temp || 0}
          currentCv={latestSample?.steam_cv || 0}
          feedFlow={latestSample?.feed_flow || 0}
          divertRisk={latestSample?.divert_risk || false}
        />

        {/* Main Tabs Navigation */}
        <Tabs defaultValue="optimizer" className="space-y-6">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 border-b border-zinc-800 pb-4">
            <TabsList className="bg-zinc-950/80 border-zinc-800 p-1">
              <TabsTrigger value="optimizer" className="gap-2 text-xs">
                <Flame className="h-4 w-4 text-cyan-400" />
                <span>Closed-Loop Steam Optimizer</span>
              </TabsTrigger>
              <TabsTrigger value="telemetry" className="gap-2 text-xs">
                <Activity className="h-4 w-4 text-amber-400" />
                <span>SCADA Telemetry & Lookahead</span>
              </TabsTrigger>
              <TabsTrigger value="what-if" className="gap-2 text-xs">
                <Sliders className="h-4 w-4 text-purple-400" />
                <span>What-If Simulator</span>
              </TabsTrigger>
              <TabsTrigger value="benchmarks" className="gap-2 text-xs">
                <Award className="h-4 w-4 text-emerald-400" />
                <span>Model Diagnostics</span>
              </TabsTrigger>
            </TabsList>

            <div className="flex items-center gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => runShiftSimulation(targetSetpoint, maxSlewCv)}
                disabled={loadingSim}
                className="h-8 text-xs text-zinc-300 gap-1.5"
              >
                <RefreshCw className={`h-3.5 w-3.5 ${loadingSim ? "animate-spin" : ""}`} />
                <span>Refresh Shift Audit</span>
              </Button>
            </div>
          </div>

          <TabsContent value="optimizer">
            <OptimizerTab
              stepResult={stepResult}
              kpis={kpis}
              trajectory={trajectory}
              targetSetpoint={targetSetpoint}
              onTargetSetpointChange={(val) => setTargetSetpoint(val)}
              maxSlewCv={maxSlewCv}
              onMaxSlewCvChange={(val) => setMaxSlewCv(val)}
              onRunSimulation={() => runShiftSimulation(targetSetpoint, maxSlewCv)}
              loadingSim={loadingSim}
            />
          </TabsContent>

          <TabsContent value="telemetry">
            <TelemetryTab telemetry={telemetry} divertCount={divertCount} />
          </TabsContent>

          <TabsContent value="what-if">
            <WhatIfTab
              whatIfResult={whatIfResult}
              deltaCv={deltaCv}
              onDeltaCvChange={handleDeltaCvChange}
              deltaFlow={deltaFlow}
              onDeltaFlowChange={handleDeltaFlowChange}
              onReset={handleResetWhatIf}
            />
          </TabsContent>

          <TabsContent value="benchmarks">
            <BenchmarkTab benchmarks={benchmarks} />
          </TabsContent>
        </Tabs>
      </main>

      {/* Footer */}
      <footer className="border-t border-zinc-800/80 bg-zinc-950/60 py-6 mt-16 text-center text-xs text-zinc-500">
        <div className="max-w-7xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-2">
          <span>Continuous HTST Pasteurizer Thermal Digital Twin &bull; 30-Second MPC Optimizer</span>
          <span>LightGBM &bull; XGBoost &bull; Residual PyTorch MLP &bull; Saturated Steam Enthalpy Model</span>
        </div>
      </footer>
    </div>
  )
}

export default App
