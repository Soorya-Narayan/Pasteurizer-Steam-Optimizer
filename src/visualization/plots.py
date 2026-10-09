"""Static result figures saved at the end of each training run."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.visualization.style import PALETTE, apply_style, model_color  # noqa: E402

apply_style()


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def _break_gaps(ts: pd.Series, values: np.ndarray, max_gap_s: float = 10.0):
    """Insert NaNs where consecutive timestamps are far apart so lines don't bridge idle periods."""
    ts = pd.Series(pd.to_datetime(ts)).reset_index(drop=True)
    v = np.asarray(values, float)
    gaps = np.where(ts.diff().dt.total_seconds().fillna(0).values > max_gap_s)[0]
    if len(gaps) == 0:
        return ts.values, v
    t_out = np.insert(ts.values, gaps, ts.values[gaps])
    v_out = np.insert(v, gaps, np.nan)
    return t_out, v_out


# ---------------------------------------------------------------- 1. data
def plot_data_overview(seg_df: pd.DataFrame, cfg: dict, split_bounds: dict, path: Path) -> Path:
    ts = seg_df[cfg["data"]["timestamp_col"]]
    target, thr = cfg["target"]["column"], cfg["target"]["divert_threshold_c"]
    prod = seg_df["is_production"].values

    fig, axes = plt.subplots(3, 1, figsize=(16, 9), sharex=True, gridspec_kw={"height_ratios": [2.2, 1, 1]})
    ax = axes[0]
    ax.plot(ts, seg_df[target].where(~prod), color=PALETTE["muted"], lw=0.6, label="idle / CIP / circulation")
    ax.plot(ts, seg_df[target].where(prod), color=PALETTE["series"][0], lw=0.8, label="production (modelled)")
    ax.axhline(thr, color=PALETTE["threshold"], ls="--", lw=1, label=f"divert threshold {thr:.0f} °C")
    ax.set_ylabel("Holding out TT-06 (°C)"); ax.legend(loc="lower left", fontsize=8, ncol=3)
    axes[1].plot(ts, seg_df["steam_cv"], color=PALETTE["series"][3], lw=0.6); axes[1].set_ylabel("Steam CV (%)")
    axes[2].plot(ts, seg_df[cfg["production"]["flow_col"]] / 1000, color=PALETTE["series"][2], lw=0.6)
    axes[2].set_ylabel("Feed flow (kL/h)")

    colors = {"train": "#4cc9f0", "val": "#ffb703", "test": "#f72585"}
    for name, (start, end) in split_bounds.items():
        for a in axes:
            a.axvspan(start, end, color=colors[name], alpha=0.07)
        axes[0].text(start, axes[0].get_ylim()[1], f" {name.upper()}", color=colors[name], va="top", fontsize=9, fontweight="bold")
    fig.suptitle("Dataset overview · 7 days of 3-second SCADA telemetry", fontsize=14, fontweight="bold", x=0.01, ha="left")
    return _save(fig, path)


# ------------------------------------------------------ 2. learning curves
def plot_learning_curves(history: dict, path: Path) -> Path:
    models = list(history)
    fig, axes = plt.subplots(1, len(models), figsize=(5.5 * len(models), 4.2), squeeze=False)
    for ax, m in zip(axes[0], models):
        h = history[m]
        ax.plot(h["step"], h["train"], color=PALETTE["series"][0], lw=1.6, label="train")
        ax.plot(h["step"], h["val"], color=PALETTE["series"][1], lw=1.6, label="validation")
        i = int(np.argmin(h["val"]))
        ax.scatter(h["step"][i], h["val"][i], color=PALETTE["accent"], zorder=5, s=45,
                   label=f"best {h['val'][i]:.3f} °C @ {h['step'][i]}")
        ax.set_title(m, fontweight="bold"); ax.set_xlabel(h["x_label"]); ax.set_ylabel("MAE (°C)")
        ax.legend(fontsize=8)
    fig.suptitle("Learning curves (train vs validation MAE)", fontsize=14, fontweight="bold", x=0.01, ha="left")
    return _save(fig, path)


# ------------------------------------------------------ 3. forecasts
def _busiest_window(ts: pd.Series, actual: np.ndarray, minutes: int = 45) -> tuple:
    s = pd.Series(actual, index=pd.to_datetime(ts.values))
    roll = s.rolling(f"{minutes}min").std()
    end = roll.idxmax()
    return end - pd.Timedelta(minutes=minutes), end


def plot_test_forecasts(ts, actual, preds: dict, thr: float, horizon_s: int, best: str, path: Path) -> Path:
    ts = pd.Series(pd.to_datetime(np.asarray(ts)))
    t_future = ts + pd.Timedelta(seconds=horizon_s)  # plot each forecast at the time it refers to
    actual = np.asarray(actual)
    fig, axes = plt.subplots(2, 1, figsize=(16, 9), gridspec_kw={"height_ratios": [1, 1.25]})

    ax = axes[0]
    ax.plot(*_break_gaps(t_future, actual), color=PALETTE["actual"], lw=0.9, label="actual")
    ax.plot(*_break_gaps(t_future, preds[best]), color=model_color(best), lw=0.8, alpha=0.85, label=f"{best} forecast")
    ax.axhline(thr, color=PALETTE["threshold"], ls="--", lw=1, label="divert threshold")
    ax.set_title(f"Full test period · {horizon_s}s-ahead holding-tube temperature", fontweight="bold")
    ax.set_ylabel("°C"); ax.legend(fontsize=8, loc="lower left", ncol=3)

    start, end = _busiest_window(t_future, actual)
    m = ((t_future >= start) & (t_future <= end)).values
    ax = axes[1]
    ax.plot(*_break_gaps(t_future[m], actual[m]), color=PALETTE["actual"], lw=2.2, label="actual", zorder=10)
    for i, (name, p) in enumerate(preds.items()):
        style = dict(ls=":", lw=1.2) if name == "Persistence" else dict(lw=1.3)
        ax.plot(*_break_gaps(t_future[m], np.asarray(p)[m]), color=model_color(name, i), label=name, alpha=0.9, **style)
    ax.axhline(thr, color=PALETTE["threshold"], ls="--", lw=1)
    ax.set_title(f"Zoom: most dynamic 45 min of the test set ({start:%b %d %H:%M} – {end:%H:%M})", fontweight="bold")
    ax.set_ylabel("°C"); ax.legend(fontsize=8, ncol=len(preds) + 1, loc="lower left")
    return _save(fig, path)


# ------------------------------------------------------ 4. parity
def plot_parity(actual, preds: dict, path: Path) -> Path:
    names = list(preds); n = len(names); cols = min(3, n); rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.6 * rows), squeeze=False)
    actual = np.asarray(actual)
    lo, hi = np.percentile(actual, 0.2), np.percentile(actual, 99.9)
    for ax, name in zip(axes.flat, names):
        p = np.asarray(preds[name])
        ax.hexbin(actual, p, gridsize=70, bins="log", cmap="magma", extent=(lo, hi, lo, hi), mincnt=1)
        ax.plot([lo, hi], [lo, hi], color=PALETTE["good"], lw=1, ls="--")
        ax.set_title(f"{name} · MAE {np.mean(np.abs(p - actual)):.3f} °C", fontweight="bold")
        ax.set_xlabel("actual (°C)"); ax.set_ylabel("forecast (°C)")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.suptitle("Parity plots (test set)", fontsize=14, fontweight="bold", x=0.01, ha="left")
    return _save(fig, path)


# ------------------------------------------------------ 5. errors
REGIME_BINS = [0, 120, 300, 900, 3600, np.inf]
REGIME_LABELS = ["0-2 min", "2-5 min", "5-15 min", "15-60 min", ">1 h"]


def regime_mae(actual, preds: dict, time_in_run) -> pd.DataFrame:
    regime = pd.cut(np.asarray(time_in_run), REGIME_BINS, labels=REGIME_LABELS, right=False)
    actual = np.asarray(actual)
    return pd.DataFrame({n: pd.Series(np.abs(np.asarray(p) - actual)).groupby(regime, observed=False).mean()
                         for n, p in preds.items()})


def plot_error_analysis(actual, preds: dict, time_in_run, path: Path) -> Path:
    actual = np.asarray(actual)
    fig, axes = plt.subplots(1, 2, figsize=(16, 4.8))
    ax = axes[0]
    bins = np.linspace(-4, 4, 81)
    for i, (name, p) in enumerate(preds.items()):
        ax.hist(np.clip(np.asarray(p) - actual, -4, 4), bins=bins, histtype="step", lw=1.5, color=model_color(name, i), label=name)
    ax.set_yscale("log"); ax.set_xlabel("forecast error (°C, clipped ±4)"); ax.set_ylabel("count (log)")
    ax.set_title("Error distribution", fontweight="bold"); ax.legend(fontsize=8)

    ax = axes[1]
    rm = regime_mae(actual, preds, time_in_run)
    x = np.arange(len(rm)); w = 0.8 / rm.shape[1]
    for i, name in enumerate(rm.columns):
        ax.bar(x + i * w - 0.4 + w / 2, rm[name].values, w, color=model_color(name, i), label=name)
    counts = pd.cut(np.asarray(time_in_run), REGIME_BINS, labels=REGIME_LABELS, right=False).value_counts().reindex(REGIME_LABELS)
    ax.set_xticks(x, [f"{l}\n(n={c:,})" for l, c in zip(rm.index, counts.values)])
    ax.set_ylabel("MAE (°C)"); ax.set_title("MAE by time since production start", fontweight="bold"); ax.legend(fontsize=8)
    return _save(fig, path)


# ------------------------------------------------------ 6. importance
def plot_feature_importance(importances: dict, path: Path, top: int = 20) -> Path:
    fig, axes = plt.subplots(1, len(importances), figsize=(7.5 * len(importances), 7), squeeze=False)
    for i, (ax, (name, imp)) in enumerate(zip(axes[0], importances.items())):
        imp = (imp / imp.sum() * 100).head(top)[::-1]
        ax.barh(imp.index, imp.values, color=model_color(name, i))
        ax.set_title(f"{name} · top {top} features (gain %)", fontweight="bold"); ax.tick_params(axis="y", labelsize=8)
    return _save(fig, path)


# ------------------------------------------------------ 7. comparison
def plot_model_comparison(metrics: pd.DataFrame, path: Path) -> Path:
    names = list(metrics.index)
    colors = [model_color(n, i) for i, n in enumerate(names)]
    panels = [("MAE_C", "MAE (°C) ↓"), ("RMSE_C", "RMSE (°C) ↓"), ("Skill_vs_Persistence_pct", "Skill vs persistence (%) ↑"),
              ("Divert_recall", "Divert early-warning recall ↑")]
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4.5))
    for ax, (col, title) in zip(axes, panels):
        vals = metrics[col].astype(float).values
        bars = ax.bar(names, vals, color=colors)
        for b, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.3f}" if abs(v) < 10 else f"{v:.1f}",
                        ha="center", va="bottom", fontsize=8, color=PALETTE["text"])
        ax.set_title(title, fontweight="bold"); ax.tick_params(axis="x", rotation=30, labelsize=8)
    fig.suptitle("Test-set model comparison", fontsize=14, fontweight="bold", x=0.01, ha="left")
    return _save(fig, path)
