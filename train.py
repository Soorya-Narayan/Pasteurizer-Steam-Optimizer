#!/usr/bin/env python
"""Train holding-tube temperature forecasters for the HTST pasteurizer.

Example
-------
    python train.py                          # all models, 30 s horizon
    python train.py --horizon 60 --models lightgbm mlp
    python train.py --quick                  # fast smoke run

While training, open ``outputs/live_training_progress.png`` – it refreshes
every couple of seconds with the learning curves and pipeline status.
"""
from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import json
import shutil
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data.loader import load_dataset
from src.features.engineering import chronological_split, horizon_steps, make_supervised
from src.models import CLI_NAMES, LightGBMModel, MLPModel, RidgeModel, XGBoostModel
from src.utils.metrics import divert_warning_metrics, regression_metrics, skill_score
from src.utils.progress import TrainingMonitor
from src.utils.report import write_report
from src.visualization import plots

warnings.filterwarnings("ignore", category=UserWarning)

STAGES = ["Load data", "Feature engineering", "Baselines", "LightGBM", "XGBoost", "MLP",
          "Evaluate on test set", "Graphs & report"]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--horizon", type=int, help="forecast horizon in seconds (overrides config)")
    ap.add_argument("--models", nargs="+", choices=["lightgbm", "xgboost", "mlp"], default=["lightgbm", "xgboost", "mlp"])
    ap.add_argument("--epochs", type=int, help="override MLP epochs")
    ap.add_argument("--quick", action="store_true", help="few rounds/epochs – for a fast end-to-end check")
    ap.add_argument("--rebuild-cache", action="store_true", help="re-read the raw CSVs")
    ap.add_argument("--no-live", action="store_true", help="disable the live dashboard PNG")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.horizon:
        cfg["target"]["horizon_s"] = args.horizon
    if args.epochs:
        cfg["models"]["mlp"]["epochs"] = args.epochs
    if args.quick:
        for m in ("lightgbm", "xgboost"):
            cfg["models"][m].update(num_boost_round=300, early_stopping_rounds=50)
        cfg["models"]["mlp"].update(epochs=5, patience=3)
    seed, h_s = cfg["seed"], cfg["target"]["horizon_s"]
    thr, margin = cfg["target"]["divert_threshold_c"], cfg["target"]["alert_margin_c"]
    selected = [CLI_NAMES[m] for m in args.models]

    root = Path(cfg["output"]["root"])
    run_id = f"run_{datetime.now():%Y%m%d_%H%M%S}_h{h_s}s"
    run_dir = root / run_id
    fig_dir, model_dir = run_dir / "figures", run_dir / "models"
    for d in (fig_dir, model_dir):
        d.mkdir(parents=True, exist_ok=True)

    live = [root / "live_training_progress.png", run_dir / "live_training_progress.png"]
    mon = TrainingMonitor(STAGES, live, refresh_s=cfg["output"]["live_plot_refresh_s"], enabled=not args.no_live)
    print("=" * 78)
    print(f"🥛  HTST PASTEURIZER · THERMAL PREDICTIVE MODEL   ({h_s}s-ahead holding-tube temperature)")
    print("=" * 78)
    print(f"Run directory : {run_dir}")
    if not args.no_live:
        print(f"Live dashboard: {live[0]}   (open it – it refreshes while training)")
    t_start = time.time()

    # ------------------------------------------------------------- data
    mon.start_stage("Load data")
    df = load_dataset(cfg, rebuild=args.rebuild_cache)
    mon.end_stage("Load data", f"{len(df):,} rows")

    mon.start_stage("Feature engineering")
    data, seg_df = make_supervised(df, cfg)
    splits = chronological_split(data, cfg)
    tr, va, te = splits["train"], splits["val"], splits["test"]
    n_segments = int(seg_df.loc[seg_df.is_production, "segment_id"].nunique())
    print(f"  • {n_segments} production segments · {len(data):,} supervised samples · {data.X.shape[1]} features")
    print(f"  • horizon {h_s}s = {horizon_steps(cfg)} samples · target = Δ {cfg['target']['column']}")
    for name, s in splits.items():
        print(f"  • {name:<5}: {len(s):>6,} rows  {s.timestamp.min():%m-%d %H:%M} → {s.timestamp.max():%m-%d %H:%M}"
              f"   (|Δ| mean {s.y_delta.abs().mean():.3f} °C)")
    mon.end_stage("Feature engineering", f"{data.X.shape[1]} features")

    # -------------------------------------------------------- baselines
    trained: dict = {}
    mon.start_stage("Baselines")
    persist_val_mae = float(np.mean(np.abs(va.y_delta)))
    print(f"  • Persistence (T[t+h] = T[t])  val MAE {persist_val_mae:.4f} °C")
    ridge = RidgeModel.fit(tr.X, tr.y_delta, cfg["models"]["ridge"]["alpha"])
    trained["Ridge"] = ridge
    print(f"  • Ridge regression             val MAE {np.mean(np.abs(ridge.predict(va.X) - va.y_delta)):.4f} °C")
    mon.end_stage("Baselines")

    # ------------------------------------------------------- ML models
    fitters = {
        "LightGBM": lambda: LightGBMModel.fit(tr.X, tr.y_delta, va.X, va.y_delta, cfg["models"]["lightgbm"], mon, seed),
        "XGBoost": lambda: XGBoostModel.fit(tr.X, tr.y_delta, va.X, va.y_delta, cfg["models"]["xgboost"], mon, seed),
        "MLP": lambda: MLPModel.fit(tr.X, tr.y_delta, va.X, va.y_delta, cfg["models"]["mlp"], mon, seed),
    }
    for name, fit in fitters.items():
        if name not in selected:
            mon.skip_stage(name)
            continue
        mon.start_stage(name)
        trained[name] = fit()
        best = mon.best(name)
        mon.end_stage(name, f"best val MAE {best[1]:.4f} °C" if best else "")

    # -------------------------------------------------------- evaluate
    mon.start_stage("Evaluate on test set")
    val_preds = {"Persistence": np.zeros(len(va))} | {n: m.predict(va.X) for n, m in trained.items()}
    test_delta = {"Persistence": np.zeros(len(te))} | {n: m.predict(te.X) for n, m in trained.items()}
    ml = [n for n in ("LightGBM", "XGBoost", "MLP") if n in trained]
    if len(ml) >= 2:
        val_preds["Ensemble"] = np.mean([val_preds[n] for n in ml], axis=0)
        test_delta["Ensemble"] = np.mean([test_delta[n] for n in ml], axis=0)

    y_true, cur = te.y_future.values, te.current.values
    test_abs = {n: cur + d for n, d in test_delta.items()}
    persist_mae = regression_metrics(y_true, cur)["MAE_C"]
    rows, drows = {}, {}
    for n, p in test_abs.items():
        m = regression_metrics(y_true, p, cur)
        m["Skill_vs_Persistence_pct"] = 100 * skill_score(m["MAE_C"], persist_mae)
        m["Val_MAE_C"] = float(np.mean(np.abs(val_preds[n] - va.y_delta.values)))
        dv = divert_warning_metrics(y_true, p, thr, margin)
        m["Divert_recall"] = dv["recall"]
        rows[n], drows[n] = m, dv
    metrics = pd.DataFrame(rows).T.rename_axis("model")
    divert = pd.DataFrame(drows).T.rename_axis("model")
    best_name = metrics.drop(index="Persistence")["Val_MAE_C"].idxmin()  # chosen on validation, reported on test
    regime = plots.regime_mae(y_true, test_abs, te.time_in_segment_s.values).rename_axis("time in run")

    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 160):
        print(metrics[["MAE_C", "RMSE_C", "P95_AbsErr_C", "R2", "R2_delta", "Skill_vs_Persistence_pct", "Val_MAE_C"]].to_string())
        print(f"\n  Divert early warning (< {thr} °C, margin {margin} °C):")
        print(divert[["events", "TP", "FP", "FN_missed", "precision", "recall", "F1"]].to_string())
    print(f"\n  ★ Best model (lowest validation MAE): {best_name}")
    mon.end_stage("Evaluate on test set", f"best = {best_name}")

    # ------------------------------------------------- persist outputs
    mon.start_stage("Graphs & report")
    for m in trained.values():
        m.save(model_dir)
    metrics.to_csv(run_dir / "metrics.csv"); divert.to_csv(run_dir / "divert_warning.csv"); regime.to_csv(run_dir / "regime_mae.csv")
    pd.DataFrame({"timestamp": te.timestamp.values, "current_c": cur, "actual_future_c": y_true}
                 | {f"pred_{n}": p for n, p in test_abs.items()}).to_csv(run_dir / "test_predictions.csv", index=False)
    (run_dir / "history.json").write_text(json.dumps(mon.history))
    meta = {"run_id": run_id, "horizon_s": h_s, "horizon_steps": horizon_steps(cfg), "best_model": best_name,
            "models": list(trained), "ensemble_members": ml if "Ensemble" in test_abs else [],
            "features": data.feature_names, "config": cfg}
    (run_dir / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))

    bounds = {k: (s.timestamp.min(), s.timestamp.max()) for k, s in splits.items()}
    pbar_figs = [
        ("data_overview.png", "Holding-tube temperature, steam valve and feed flow over the full week, with train/val/test periods.",
         lambda p: plots.plot_data_overview(seg_df, cfg, bounds, p)),
        ("learning_curves.png", "Training vs validation MAE per boosting round / epoch (dot = early-stopping point).",
         lambda p: plots.plot_learning_curves(mon.history, p)),
        ("model_comparison.png", "Test-set accuracy, skill over the persistence baseline and divert-warning recall.",
         lambda p: plots.plot_model_comparison(metrics, p)),
        ("test_forecasts.png", f"Actual vs forecast {h_s}s ahead: whole test period and the most dynamic 45 minutes.",
         lambda p: plots.plot_test_forecasts(te.timestamp, y_true, test_abs, thr, h_s, best_name, p)),
        ("parity.png", "Forecast vs actual density (log colour scale).",
         lambda p: plots.plot_parity(y_true, {k: v for k, v in test_abs.items()}, p)),
        ("error_analysis.png", "Error distribution and MAE by time since production start (start-ups are hardest).",
         lambda p: plots.plot_error_analysis(y_true, test_abs, te.time_in_segment_s.values, p)),
    ]
    imps = {n: trained[n].importance() for n in ("LightGBM", "XGBoost") if n in trained}
    if imps:
        pbar_figs.append(("feature_importance.png", "Which signals drive the forecast (normalised gain).",
                          lambda p: plots.plot_feature_importance(imps, p)))
    figures = []
    for fname, caption, fn in pbar_figs:
        fn(fig_dir / fname)
        figures.append((f"figures/{fname}", caption))
        print(f"  • figures/{fname}")

    mon.render(force=True)
    summary = {"run_id": run_id, "horizon_s": h_s, "target": cfg["target"]["column"], "threshold": thr, "margin": margin,
               "n_train": len(tr), "data": {
                   "raw rows (deduplicated)": f"{len(df):,}", "production segments": n_segments,
                   "supervised samples": f"{len(data):,}", "features": data.X.shape[1],
                   "train / val / test": f"{len(tr):,} / {len(va):,} / {len(te):,}",
                   "period": f"{df[cfg['data']['timestamp_col']].min():%Y-%m-%d %H:%M} → {df[cfg['data']['timestamp_col']].max():%Y-%m-%d %H:%M}"}}
    report = write_report(run_dir, summary, metrics, divert, regime, figures, best_name)
    mon.end_stage("Graphs & report")

    latest = root / "latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink() if latest.is_symlink() else shutil.rmtree(latest)
    os.symlink(run_id, latest)

    print("\n" + "=" * 78)
    print(f"✅ Done in {time.time() - t_start:.0f}s · best = {best_name} · "
          f"test MAE {metrics.loc[best_name, 'MAE_C']:.3f} °C vs persistence {persist_mae:.3f} °C "
          f"({metrics.loc[best_name, 'Skill_vs_Persistence_pct']:+.1f}% skill)")
    print(f"   Report : {report}")
    print(f"   Models : {model_dir}")
    print("=" * 78)


if __name__ == "__main__":
    main()
