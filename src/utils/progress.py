"""Live training monitor.

* Collects per-iteration / per-epoch train & validation curves from every model.
* Periodically re-draws a dashboard PNG (atomic write, so an open image viewer /
  VS Code preview just refreshes) and writes ``progress.json``.
* Tracks pipeline stages with timings for the console and the dashboard.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.visualization.style import PALETTE, apply_style  # noqa: E402


class TrainingMonitor:
    def __init__(self, stages: list[str], live_paths: list[Path], refresh_s: float = 2.0, enabled: bool = True):
        self.stages = {s: {"status": "pending", "seconds": None, "note": ""} for s in stages}
        self.history: dict[str, dict] = {}
        self.live_paths = [Path(p) for p in live_paths]
        self.refresh_s = refresh_s
        self.enabled = enabled
        self._t0 = time.time()
        self._stage_t0: dict[str, float] = {}
        self._last_render = 0.0
        apply_style()

    # ----------------------------------------------------------------- stages
    def start_stage(self, name: str) -> None:
        idx = list(self.stages).index(name) + 1
        self.stages[name]["status"] = "running"
        self._stage_t0[name] = time.time()
        print(f"\n\033[1;36m[{idx}/{len(self.stages)}] {name}\033[0m")
        self.render(force=True)

    def end_stage(self, name: str, note: str = "") -> None:
        st = self.stages[name]
        st.update(status="done", seconds=time.time() - self._stage_t0.get(name, time.time()), note=note)
        print(f"  \033[32m✓\033[0m {name} done in {st['seconds']:.1f}s" + (f"  –  {note}" if note else ""))
        self.render(force=True)

    def skip_stage(self, name: str) -> None:
        self.stages[name]["status"] = "skipped"

    # ---------------------------------------------------------------- curves
    def log(self, model: str, step: int, train: float, val: float, x_label: str = "iteration") -> None:
        h = self.history.setdefault(model, {"step": [], "train": [], "val": [], "x_label": x_label})
        h["step"].append(int(step)); h["train"].append(float(train)); h["val"].append(float(val))
        self.render()

    def best(self, model: str) -> tuple[int, float] | None:
        h = self.history.get(model)
        if not h or not h["val"]:
            return None
        i = min(range(len(h["val"])), key=h["val"].__getitem__)
        return h["step"][i], h["val"][i]

    # --------------------------------------------------------------- render
    def render(self, force: bool = False) -> None:
        if not self.enabled or (not force and time.time() - self._last_render < self.refresh_s):
            return
        self._last_render = time.time()
        models = list(self.history)
        n_plots = max(1, len(models))
        fig = plt.figure(figsize=(5.2 * n_plots + 4.2, 4.8))
        gs = fig.add_gridspec(1, n_plots + 1, width_ratios=[0.8] + [1] * n_plots, wspace=0.28)

        # stage panel
        ax = fig.add_subplot(gs[0, 0]); ax.axis("off")
        icons = {"done": ("●", PALETTE["good"]), "running": ("◉", PALETTE["accent"]),
                 "pending": ("○", PALETTE["muted"]), "skipped": ("–", PALETTE["muted"])}
        ax.text(0, 1.0, "Pipeline", fontsize=14, fontweight="bold", transform=ax.transAxes, va="top")
        for i, (name, st) in enumerate(self.stages.items()):
            sym, col = icons[st["status"]]
            secs = f"{st['seconds']:.0f}s" if st["seconds"] is not None else ("…" if st["status"] == "running" else "")
            y = 0.88 - i * 0.085
            ax.text(0.0, y, sym, color=col, fontsize=12, transform=ax.transAxes, va="center")
            ax.text(0.09, y, name, color=col if st["status"] != "pending" else PALETTE["muted"],
                    fontsize=10, transform=ax.transAxes, va="center")
            ax.text(1.0, y, secs, color=PALETTE["muted"], fontsize=9, transform=ax.transAxes, va="center", ha="right")
        ax.text(0, 0.0, f"elapsed {time.time() - self._t0:,.0f}s", color=PALETTE["muted"], fontsize=9, transform=ax.transAxes)

        # learning curves
        for j, m in enumerate(models):
            h = self.history[m]
            a = fig.add_subplot(gs[0, j + 1])
            a.plot(h["step"], h["train"], color=PALETTE["series"][0], lw=1.6, label="train")
            a.plot(h["step"], h["val"], color=PALETTE["series"][1], lw=1.6, label="validation")
            b = self.best(m)
            if b:
                a.scatter([b[0]], [b[1]], color=PALETTE["accent"], s=40, zorder=5, label=f"best {b[1]:.3f} °C")
            a.set_title(m, fontweight="bold"); a.set_xlabel(h["x_label"]); a.set_ylabel("MAE (°C)")
            a.legend(loc="upper right", fontsize=8)
        if not models:
            a = fig.add_subplot(gs[0, 1]); a.axis("off")
            a.text(0.5, 0.5, "waiting for first model…", ha="center", va="center", color=PALETTE["muted"], transform=a.transAxes)

        fig.suptitle("HTST Thermal Model · Live Training Progress", fontsize=15, fontweight="bold", x=0.01, ha="left")
        for p in self.live_paths:
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".tmp.png")
            fig.savefig(tmp, dpi=110, bbox_inches="tight")
            os.replace(tmp, p)
        plt.close(fig)
        self._dump_json()

    def _dump_json(self) -> None:
        payload = {"elapsed_s": time.time() - self._t0, "stages": self.stages, "history": self.history}
        for p in self.live_paths:
            tmp = p.with_name("progress.tmp.json")
            tmp.write_text(json.dumps(payload))
            os.replace(tmp, p.with_name("progress.json"))
