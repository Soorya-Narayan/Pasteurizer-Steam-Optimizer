import React, { useState } from "react"
import { Sliders, RotateCcw, AlertTriangle, ShieldCheck, Thermometer, Gauge, Droplets } from "lucide-react"
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Slider } from "@/components/ui/slider"
import type { WhatIfResult } from "@/types"

interface WhatIfTabProps {
  whatIfResult: WhatIfResult | null
  deltaCv: number
  onDeltaCvChange: (val: number) => void
  deltaFlow: number
  onDeltaFlowChange: (val: number) => void
  onReset: () => void
}

export function WhatIfTab({
  whatIfResult,
  deltaCv,
  onDeltaCvChange,
  deltaFlow,
  onDeltaFlowChange,
  onReset,
}: WhatIfTabProps) {
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch">
        {/* Perturbation Sliders */}
        <Card className="lg:col-span-5 border-zinc-800 bg-zinc-950/70">
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle className="text-base text-zinc-100 flex items-center gap-2">
                <Sliders className="h-4 w-4 text-cyan-400" />
                Perturbation Inputs
              </CardTitle>
              <Button
                variant="ghost"
                size="sm"
                onClick={onReset}
                className="h-7 text-xs text-zinc-400 hover:text-white gap-1"
              >
                <RotateCcw className="h-3.5 w-3.5" /> Reset
              </Button>
            </div>
            <CardDescription>
              Test plant thermal sensitivity to sudden actuator adjustments or throughput surges.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* Delta Steam CV */}
            <div>
              <div className="flex justify-between items-center mb-2 text-xs">
                <span className="text-zinc-300 font-medium">Steam Valve Bias (ΔSteam CV)</span>
                <span
                  className={`font-mono text-sm font-semibold ${
                    deltaCv > 0 ? "text-emerald-400" : deltaCv < 0 ? "text-cyan-400" : "text-zinc-400"
                  }`}
                >
                  {deltaCv > 0 ? "+" : ""}
                  {deltaCv.toFixed(1)} %
                </span>
              </div>
              <Slider min={-25.0} max={25.0} step={1.0} value={deltaCv} onChange={onDeltaCvChange} />
              <div className="flex justify-between text-[10px] text-zinc-500 mt-1">
                <span>-25% (Cut Steam)</span>
                <span>0% (Baseline)</span>
                <span>+25% (Boil Up)</span>
              </div>
            </div>

            {/* Delta Feed Flow */}
            <div>
              <div className="flex justify-between items-center mb-2 text-xs">
                <span className="text-zinc-300 font-medium">Milk Throughput Perturbation (ΔFlow %)</span>
                <span
                  className={`font-mono text-sm font-semibold ${
                    deltaFlow > 0 ? "text-amber-400" : deltaFlow < 0 ? "text-blue-400" : "text-zinc-400"
                  }`}
                >
                  {deltaFlow > 0 ? "+" : ""}
                  {deltaFlow.toFixed(0)} %
                </span>
              </div>
              <Slider min={-30.0} max={30.0} step={5.0} value={deltaFlow} onChange={onDeltaFlowChange} />
              <div className="flex justify-between text-[10px] text-zinc-500 mt-1">
                <span>-30% (Throttled)</span>
                <span>0% (Baseline)</span>
                <span>+30% (Surge)</span>
              </div>
            </div>

            {whatIfResult && (
              <div className="p-3 rounded-lg bg-zinc-900/60 border border-zinc-800 text-xs text-zinc-400 space-y-1.5">
                <div className="flex justify-between">
                  <span>Simulated Steam CV:</span>
                  <span className="font-mono text-zinc-200">{whatIfResult.simulated_cv.toFixed(1)}%</span>
                </div>
                <div className="flex justify-between">
                  <span>Simulated Feed Flow:</span>
                  <span className="font-mono text-zinc-200">
                    {Math.round(whatIfResult.simulated_flow).toLocaleString()} L/H
                  </span>
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Dynamic Response Gauge & Outcome */}
        <Card className="lg:col-span-7 border-zinc-800 bg-gradient-to-br from-zinc-950 via-zinc-900/40 to-zinc-950">
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle className="text-base text-zinc-100 flex items-center gap-2">
                <Thermometer className="h-4 w-4 text-amber-400" />
                Predicted Holding Tube Thermal Response (+30s)
              </CardTitle>
              {whatIfResult && (
                <Badge
                  variant={whatIfResult.divert_warning ? "destructive" : "success"}
                  className="font-medium text-xs px-2.5 py-0.5"
                >
                  {whatIfResult.divert_warning ? "Divert Warning (<82°C)" : "Safe Operating Regime"}
                </Badge>
              )}
            </div>
            <CardDescription>
              Inference computed across LightGBM, XGBoost, and Residual MLP surrogate models.
            </CardDescription>
          </CardHeader>
          <CardContent>
            {whatIfResult ? (
              <div className="space-y-6">
                <div className="flex flex-col sm:flex-row items-center justify-around gap-6 p-6 rounded-xl bg-zinc-900/40 border border-zinc-800/80">
                  <div className="text-center">
                    <div className="text-xs text-zinc-400 uppercase tracking-wider mb-1">Baseline Temp</div>
                    <div className="text-3xl font-mono font-bold text-zinc-300">
                      {whatIfResult.baseline_temp.toFixed(2)} °C
                    </div>
                    <div className="text-[11px] text-zinc-500 mt-1">
                      CV: {whatIfResult.baseline_cv.toFixed(1)}%
                    </div>
                  </div>

                  <div className="flex flex-col items-center">
                    <div className="h-8 w-8 rounded-full bg-zinc-800 border border-zinc-700 flex items-center justify-center font-bold text-zinc-400 text-xs">
                      ➔
                    </div>
                    <div
                      className={`font-mono text-xs font-semibold mt-1 ${
                        whatIfResult.predicted_delta >= 0 ? "text-emerald-400" : "text-rose-400"
                      }`}
                    >
                      {whatIfResult.predicted_delta >= 0 ? "+" : ""}
                      {whatIfResult.predicted_delta.toFixed(2)} °C
                    </div>
                  </div>

                  <div className="text-center">
                    <div className="text-xs text-cyan-400 uppercase tracking-wider mb-1 font-semibold">
                      Predicted Temp (+30s)
                    </div>
                    <div className="text-3xl font-mono font-bold text-cyan-300">
                      {whatIfResult.predicted_temp.toFixed(2)} °C
                    </div>
                    <div className="text-[11px] text-zinc-400 mt-1">
                      Legal Floor: 82.0 °C
                    </div>
                  </div>
                </div>

                {whatIfResult.divert_warning ? (
                  <div className="p-3.5 rounded-lg border border-rose-900/60 bg-rose-950/40 flex items-center gap-3 text-rose-300 text-xs">
                    <AlertTriangle className="h-4 w-4 text-rose-400 shrink-0" />
                    <span>
                      <strong>Warning:</strong> This perturbation causes milk holding temperature to drop below 82.0 °C, which would force the diversion valve to divert under-pasteurized milk back to the balance tank.
                    </span>
                  </div>
                ) : (
                  <div className="p-3.5 rounded-lg border border-emerald-900/60 bg-emerald-950/30 flex items-center gap-3 text-emerald-300 text-xs">
                    <ShieldCheck className="h-4 w-4 text-emerald-400 shrink-0" />
                    <span>
                      <strong>Thermal Stability Confirmed:</strong> Predicted temperature remains safely above statutory pasteurization thresholds with adequate buffer.
                    </span>
                  </div>
                )}
              </div>
            ) : (
              <div className="text-center py-8 text-zinc-500 text-sm">Evaluating surrogate response...</div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
