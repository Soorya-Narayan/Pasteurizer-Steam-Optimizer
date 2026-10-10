import * as React from "react"
import { cn } from "@/lib/utils"

interface SliderProps {
  min: number
  max: number
  step?: number
  value: number
  onChange: (val: number) => void
  className?: string
  disabled?: boolean
}

export function Slider({
  min,
  max,
  step = 1,
  value,
  onChange,
  className,
  disabled = false,
}: SliderProps) {
  const percentage = Math.min(100, Math.max(0, ((value - min) / (max - min)) * 100))

  return (
    <div className={cn("relative flex w-full touch-none select-none items-center", className)}>
      <div className="relative h-2 w-full grow overflow-hidden rounded-full bg-zinc-800">
        <div
          className="h-full bg-gradient-to-r from-cyan-500 to-emerald-400 transition-all rounded-full"
          style={{ width: `${percentage}%` }}
        />
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="absolute inset-0 h-full w-full opacity-0 cursor-pointer disabled:cursor-not-allowed"
      />
      <div
        className="pointer-events-none absolute h-4 w-4 rounded-full border-2 border-zinc-950 bg-zinc-100 shadow transition-transform -translate-x-1/2"
        style={{ left: `${percentage}%` }}
      />
    </div>
  )
}
