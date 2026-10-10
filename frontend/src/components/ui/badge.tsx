import * as React from "react"
import { cva, type VariantProps } from "class-variance-authority"
import { cn } from "@/lib/utils"

const badgeVariants = cva(
  "inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-medium transition-colors focus:outline-none focus:ring-2 focus:ring-zinc-400 focus:ring-offset-2",
  {
    variants: {
      variant: {
        default:
          "border-transparent bg-zinc-100 text-zinc-900 shadow hover:bg-zinc-200",
        secondary:
          "border-zinc-800 bg-zinc-900 text-zinc-300 hover:bg-zinc-800",
        destructive:
          "border-rose-900/50 bg-rose-950/60 text-rose-300 shadow-sm",
        success:
          "border-emerald-900/50 bg-emerald-950/60 text-emerald-300 shadow-sm",
        warning:
          "border-amber-900/50 bg-amber-950/60 text-amber-300 shadow-sm",
        cyan:
          "border-cyan-900/50 bg-cyan-950/60 text-cyan-300 shadow-sm",
        outline: "border-zinc-750 text-zinc-300",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  }
)

export interface BadgeProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof badgeVariants> {}

function Badge({ className, variant, ...props }: BadgeProps) {
  return (
    <div className={cn(badgeVariants({ variant }), className)} {...props} />
  )
}

export { Badge, badgeVariants }
