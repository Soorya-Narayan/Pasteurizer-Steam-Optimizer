import React from "react"
import { Award, BarChart3, Layers, CheckCircle2 } from "lucide-react"
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import type { ModelBenchmark } from "@/types"

interface BenchmarkTabProps {
  benchmarks: ModelBenchmark[]
}

export function BenchmarkTab({ benchmarks }: BenchmarkTabProps) {
  return (
    <div className="space-y-6">
      {/* Benchmark Summary Card */}
      <Card className="border-zinc-800 bg-zinc-950/80">
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="text-base text-zinc-100 flex items-center gap-2">
              <Award className="h-4 w-4 text-cyan-400" />
              Unseen Test Set Model Benchmarks (30s Horizon)
            </CardTitle>
            <Badge variant="cyan" className="font-mono text-xs">
              Continuous Split (200k+ SCADA points)
            </Badge>
          </div>
          <CardDescription>
            Rigorous evaluation on the chronologically holdout week of continuous production.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs border-collapse">
              <thead>
                <tr className="border-b border-zinc-800 text-zinc-400">
                  <th className="py-3 px-4 font-semibold uppercase tracking-wider">Model Architecture</th>
                  <th className="py-3 px-4 font-semibold uppercase tracking-wider text-right">MAE (°C)</th>
                  <th className="py-3 px-4 font-semibold uppercase tracking-wider text-right">RMSE (°C)</th>
                  <th className="py-3 px-4 font-semibold uppercase tracking-wider text-right">R² Score</th>
                  <th className="py-3 px-4 font-semibold uppercase tracking-wider text-right">Skill vs Baseline</th>
                  <th className="py-3 px-4 font-semibold uppercase tracking-wider text-center">Divert Recall</th>
                  <th className="py-3 px-4 font-semibold uppercase tracking-wider text-center">Ensemble</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-zinc-800/60 font-mono">
                {benchmarks.map((m, idx) => (
                  <tr
                    key={m.model}
                    className={`transition-colors hover:bg-zinc-900/40 ${
                      m.model.includes("Ensemble") ? "bg-cyan-950/20 font-medium" : ""
                    }`}
                  >
                    <td className="py-3.5 px-4 font-sans flex items-center gap-2 text-zinc-200">
                      {m.model.includes("Ensemble") && (
                        <span className="h-2 w-2 rounded-full bg-cyan-400 animate-pulse"></span>
                      )}
                      <span>{m.model}</span>
                    </td>
                    <td className="py-3.5 px-4 text-right text-zinc-300">{m.mae.toFixed(3)}</td>
                    <td className="py-3.5 px-4 text-right text-zinc-300">{m.rmse.toFixed(3)}</td>
                    <td className="py-3.5 px-4 text-right text-zinc-300">{m.r2_level.toFixed(3)}</td>
                    <td className="py-3.5 px-4 text-right text-emerald-400 font-semibold">{m.skill_score}</td>
                    <td className="py-3.5 px-4 text-center">
                      <Badge variant="success" className="py-0 px-2 text-[10px]">
                        {m.divert_recall}
                      </Badge>
                    </td>
                    <td className="py-3.5 px-4 text-center">
                      {m.is_ensemble ? (
                        <Badge variant="cyan" className="py-0 px-2 text-[10px]">
                          Active Member
                        </Badge>
                      ) : (
                        <span className="text-zinc-600 font-sans text-xs">—</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>

      {/* Architecture Highlights */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card className="border-zinc-800 bg-zinc-950/60">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-zinc-200 flex items-center gap-2">
              <Layers className="h-4 w-4 text-emerald-400" />
              LightGBM & XGBoost
            </CardTitle>
          </CardHeader>
          <CardContent className="text-xs text-zinc-400 space-y-1.5 leading-relaxed">
            <p>Histogram gradient boosted decision trees capturing high-order non-linear thermal interactions between milk feed flow rate, steam valve saturation, and thermal inertia.</p>
            <p className="text-emerald-400 font-mono text-[11px] pt-1">MAE: 0.257 °C | +54.0% Skill Score</p>
          </CardContent>
        </Card>

        <Card className="border-zinc-800 bg-zinc-950/60">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-zinc-200 flex items-center gap-2">
              <Layers className="h-4 w-4 text-indigo-400" />
              Residual PyTorch MLP
            </CardTitle>
          </CardHeader>
          <CardContent className="text-xs text-zinc-400 space-y-1.5 leading-relaxed">
            <p>Deep residual neural network featuring LayerNorm, GELU activations, and skip connections trained with Huber loss to resist SCADA measurement outliers and high-frequency noise.</p>
            <p className="text-indigo-400 font-mono text-[11px] pt-1">MAE: 0.281 °C | +49.6% Skill Score</p>
          </CardContent>
        </Card>

        <Card className="border-zinc-800 bg-zinc-950/60">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-zinc-200 flex items-center gap-2">
              <Layers className="h-4 w-4 text-cyan-400" />
              Tri-Model Ensemble
            </CardTitle>
          </CardHeader>
          <CardContent className="text-xs text-zinc-400 space-y-1.5 leading-relaxed">
            <p>Combines tree-based partition boundaries with smooth neural network gradients to produce variance-reduced, highly reliable 30-second thermal forecasts.</p>
            <p className="text-cyan-400 font-mono text-[11px] pt-1">R²: 0.991 | 100% Divert Recall</p>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
