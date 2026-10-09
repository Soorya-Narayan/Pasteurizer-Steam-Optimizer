"""Shared dark plot theme for all figures."""
from __future__ import annotations

import matplotlib as mpl

PALETTE = {
    "bg": "#0f1117",
    "panel": "#161b26",
    "grid": "#2a3142",
    "text": "#e6e9f0",
    "muted": "#7d869c",
    "accent": "#ffb703",
    "good": "#06d6a0",
    "bad": "#ef476f",
    "threshold": "#ef476f",
    "actual": "#e6e9f0",
    "series": ["#4cc9f0", "#f72585", "#06d6a0", "#ffb703", "#9b5de5", "#ff7f51", "#4361ee"],
}

MODEL_COLORS = {
    "Persistence": "#7d869c",
    "Ridge": "#9b5de5",
    "LightGBM": "#4cc9f0",
    "XGBoost": "#ff7f51",
    "MLP": "#f72585",
    "Ensemble": "#06d6a0",
}


def model_color(name: str, i: int = 0) -> str:
    return MODEL_COLORS.get(name, PALETTE["series"][i % len(PALETTE["series"])])


def apply_style() -> None:
    p = PALETTE
    mpl.rcParams.update({
        "figure.facecolor": p["bg"], "savefig.facecolor": p["bg"],
        "axes.facecolor": p["panel"], "axes.edgecolor": p["grid"], "axes.labelcolor": p["text"],
        "axes.titlecolor": p["text"], "axes.grid": True, "axes.titlesize": 12, "axes.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "grid.color": p["grid"], "grid.linewidth": 0.6, "grid.alpha": 0.8,
        "text.color": p["text"], "xtick.color": p["muted"], "ytick.color": p["muted"],
        "legend.facecolor": p["panel"], "legend.edgecolor": p["grid"], "legend.labelcolor": p["text"],
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "axes.prop_cycle": mpl.cycler(color=p["series"]),
    })
