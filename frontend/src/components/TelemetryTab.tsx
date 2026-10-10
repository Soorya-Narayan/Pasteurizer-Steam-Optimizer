import React from "react"
import { Activity, ShieldAlert, CheckCircle2, Droplets } from "lucide-react"
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Legend,
  ReferenceLine,
} from "recharts"
import type { TelemetryPoint } from "@/types"

interface TelemetryTabProps {
  telemetry: TelemetryPoint[]
  divertCount: number
}

export function TelemetryTab({ telemetry, divertCount }: TelemetryTabProps) {
  return (
    <div className="space-y-6">
      {/* Divert Safety Alert Banner */}
      {divertCount > 0 ? (
        <div className="p-4 rounded-xl border border-rose-900/60 bg-rose-950/40 flex items-center justify-between text-rose-200">
          <div className="flex items-center gap-3">
            <ShieldAlert className="h-5 w-5 text-rose-400 shrink-0" />
            <div>
              <div className="font-semibold text-sm text-rose-300">
                Early Warning Alert: {divertCount} Under-Pasteurization Risk Instances Detected
              </div>
              <div className="text-xs text-rose-400/80 mt-0.5">
                The 30-second digital twin predicts holding temperature dipping below 82.0 °C. Automated flow diversion valve will trigger unless steam is augmented.
              </div>
            </div>
          </div>
          <Badge variant="destructive" className="font-mono text-xs uppercase tracking-wider">
            Critical Divert Risk
          </Badge>
        </div>
      ) : (
        <div className="p-4 rounded-xl border border-emerald-900/40 bg-emerald-950/20 flex items-center justify-between text-emerald-200">
          <div className="flex items-center gap-3">
            <CheckCircle2 className="h-5 w-5 text-emerald-400 shrink-0" />
            <div>
              <div className="font-semibold text-sm text-emerald-300">
                Thermal Safety Compliant: Zero Divert Risks Detected
              </div>
              <div className="text-xs text-emerald-400/80 mt-0.5">
                Continuous production run is operating well above the 82.0 °C pasteurization statutory floor.
              </div>
            </div>
          </div>
          <Badge variant="success" className="font-mono text-xs uppercase tracking-wider">
            Safe Operation
          </Badge>
        </div>
      )}

      {/* Main Dual-Axis Chart */}
      <Card className="border-zinc-800 bg-zinc-950/80">
        <CardHeader className="pb-2">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
            <div>
              <CardTitle className="text-base text-zinc-100 flex items-center gap-2">
                <Activity className="h-4 w-4 text-cyan-400" />
                Continuous SCADA Telemetry & 30s Lookahead Digital Twin
              </CardTitle>
              <CardDescription>
                Synchronized telemetry stream: Holding tube temperature, AI ensemble lookahead forecast, and actuator valve position.
              </CardDescription>
            </div>
            <div className="flex items-center gap-3 text-xs">
              <span className="flex items-center gap-1.5 text-amber-400">
                <span className="h-2 w-2 rounded-full bg-amber-400"></span> Current Temp
              </span>
              <span className="flex items-center gap-1.5 text-cyan-400">
                <span className="h-2 w-2 rounded-full bg-cyan-400"></span> 30s AI Lookahead
              </span>
              <span className="flex items-center gap-1.5 text-indigo-400">
                <span className="h-2 w-2 rounded-full bg-indigo-400"></span> Steam CV %
              </span>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="h-80 w-full pt-4">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={telemetry} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#27272a" opacity={0.6} />
                <XAxis
                  dataKey="timestamp"
                  stroke="#71717a"
                  fontSize={10}
                  tickFormatter={(val) => val.split(" ")[1] || val}
                />
                <YAxis
                  yAxisId="temp"
                  stroke="#fbbf24"
                  fontSize={10}
                  domain={[78, 96]}
                  unit="°C"
                />
                <YAxis
                  yAxisId="cv"
                  orientation="right"
                  stroke="#818cf8"
                  fontSize={10}
                  domain={[0, 100]}
                  unit="%"
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "#09090b",
                    borderColor: "#27272a",
                    borderRadius: "8px",
                    fontSize: "12px",
                  }}
                />
                <ReferenceLine
                  yAxisId="temp"
                  y={82.0}
                  stroke="#f43f5e"
                  strokeDasharray="4 4"
                  label={{ value: "Divert Floor: 82°C", fill: "#f43f5e", fontSize: 10 }}
                />
                <Line
                  yAxisId="temp"
                  type="monotone"
                  dataKey="holding_temp"
                  name="Holding Temp"
                  stroke="#fbbf24"
                  strokeWidth={2}
                  dot={false}
                />
                <Line
                  yAxisId="temp"
                  type="monotone"
                  dataKey="lookahead_temp"
                  name="30s Lookahead"
                  stroke="#22d3ee"
                  strokeWidth={1.5}
                  strokeDasharray="3 3"
                  dot={false}
                />
                <Line
                  yAxisId="cv"
                  type="monotone"
                  dataKey="steam_cv"
                  name="Steam CV %"
                  stroke="#818cf8"
                  strokeWidth={1.5}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
