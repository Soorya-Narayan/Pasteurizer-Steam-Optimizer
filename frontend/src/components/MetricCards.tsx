import React from "react"
import { Thermometer, Gauge, Droplets, Zap, ShieldAlert, CheckCircle2 } from "lucide-react"
import { Card, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"

interface MetricCardsProps {
  currentTemp: number
  lookaheadTemp: number
  currentCv: number
  feedFlow: number
  divertRisk: boolean
}

export function MetricCards({
  currentTemp,
  lookaheadTemp,
  currentCv,
  feedFlow,
  divertRisk,
}: MetricCardsProps) {
  const tempDiff = lookaheadTemp - currentTemp

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
      {/* Current Temperature */}
      <Card className="relative overflow-hidden border-zinc-800/80 bg-zinc-950/60 hover:border-zinc-700/80">
        <div className="absolute top-0 left-0 h-1 w-full bg-gradient-to-r from-amber-500 to-rose-500" />
        <CardContent className="p-4">
          <div className="flex items-center justify-between text-xs text-zinc-400 mb-1">
            <span className="font-medium uppercase tracking-wider">Holding Tube (TT-06)</span>
            <Thermometer className="h-4 w-4 text-amber-400" />
          </div>
          <div className="flex items-baseline gap-2">
            <span className="text-2xl font-bold font-mono tracking-tight text-zinc-100">
              {currentTemp > 0 ? currentTemp.toFixed(2) : "--"}
            </span>
            <span className="text-xs text-zinc-400 font-mono">°C</span>
          </div>
          <div className="mt-2 flex items-center justify-between text-[11px]">
            <span className="text-zinc-500">Legal Min: 82.0 °C</span>
            {currentTemp >= 82.5 ? (
              <span className="text-emerald-400 flex items-center gap-1 font-medium">
                <CheckCircle2 className="h-3 w-3" /> Safe Zone
              </span>
            ) : (
              <span className="text-rose-400 flex items-center gap-1 font-medium">
                <ShieldAlert className="h-3 w-3" /> Trip Risk
              </span>
            )}
          </div>
        </CardContent>
      </Card>

      {/* 30s Lookahead Forecast */}
      <Card className="relative overflow-hidden border-zinc-800/80 bg-zinc-950/60 hover:border-zinc-700/80">
        <div className="absolute top-0 left-0 h-1 w-full bg-gradient-to-r from-cyan-500 to-blue-500" />
        <CardContent className="p-4">
          <div className="flex items-center justify-between text-xs text-zinc-400 mb-1">
            <span className="font-medium uppercase tracking-wider">30s AI Lookahead</span>
            <Zap className="h-4 w-4 text-cyan-400" />
          </div>
          <div className="flex items-baseline gap-2">
            <span className="text-2xl font-bold font-mono tracking-tight text-cyan-300">
              {lookaheadTemp > 0 ? lookaheadTemp.toFixed(2) : "--"}
            </span>
            <span className="text-xs text-zinc-400 font-mono">°C</span>
            <span className={`text-xs font-mono font-medium ${tempDiff >= 0 ? "text-emerald-400" : "text-rose-400"}`}>
              ({tempDiff >= 0 ? "+" : ""}{tempDiff.toFixed(2)}°)
            </span>
          </div>
          <div className="mt-2 flex items-center justify-between text-[11px]">
            <span className="text-zinc-500">Early Warning:</span>
            {divertRisk ? (
              <Badge variant="destructive" className="py-0 px-1.5 text-[10px] animate-pulse">
                Divert Hazard (&lt;82°C)
              </Badge>
            ) : (
              <Badge variant="success" className="py-0 px-1.5 text-[10px]">
                Safe Lookahead
              </Badge>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Steam CV */}
      <Card className="relative overflow-hidden border-zinc-800/80 bg-zinc-950/60 hover:border-zinc-700/80">
        <div className="absolute top-0 left-0 h-1 w-full bg-gradient-to-r from-purple-500 to-indigo-500" />
        <CardContent className="p-4">
          <div className="flex items-center justify-between text-xs text-zinc-400 mb-1">
            <span className="font-medium uppercase tracking-wider">Steam Valve (CV)</span>
            <Gauge className="h-4 w-4 text-indigo-400" />
          </div>
          <div className="flex items-baseline gap-2">
            <span className="text-2xl font-bold font-mono tracking-tight text-zinc-100">
              {currentCv > 0 ? currentCv.toFixed(1) : "--"}
            </span>
            <span className="text-xs text-zinc-400 font-mono">%</span>
          </div>
          <div className="mt-2 flex items-center justify-between text-[11px] text-zinc-400">
            <span>Actuator Range</span>
            <span className="font-mono text-zinc-300">0.0% – 100.0%</span>
          </div>
        </CardContent>
      </Card>

      {/* Feed Flow Rate */}
      <Card className="relative overflow-hidden border-zinc-800/80 bg-zinc-950/60 hover:border-zinc-700/80">
        <div className="absolute top-0 left-0 h-1 w-full bg-gradient-to-r from-emerald-500 to-teal-500" />
        <CardContent className="p-4">
          <div className="flex items-center justify-between text-xs text-zinc-400 mb-1">
            <span className="font-medium uppercase tracking-wider">Milk Flow Rate</span>
            <Droplets className="h-4 w-4 text-emerald-400" />
          </div>
          <div className="flex items-baseline gap-2">
            <span className="text-2xl font-bold font-mono tracking-tight text-zinc-100">
              {feedFlow > 0 ? Math.round(feedFlow).toLocaleString() : "--"}
            </span>
            <span className="text-xs text-zinc-400 font-mono">L/H</span>
          </div>
          <div className="mt-2 flex items-center justify-between text-[11px] text-zinc-400">
            <span>Production Mode</span>
            <Badge variant="secondary" className="py-0 px-1.5 text-[10px] text-emerald-400 border-emerald-900/50">
              Continuous
            </Badge>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
