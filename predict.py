#!/usr/bin/env python
"""Forecast holding-tube temperature with a trained run.

Example
-------
    python predict.py --csv csv_files/pasteurizer_dataset41.csv
    python predict.py --run outputs/run_... --csv new_shift.csv --model LightGBM

Writes ``<run>/predictions/<name>_predictions.csv`` and a forecast plot.  If the
input contains enough future data, accuracy versus persistence is printed too.
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
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.loader import read_csvs
from src.features.engineering import make_supervised
from src.models import MODEL_REGISTRY  # loads LightGBM/XGBoost before torch (macOS OpenMP)
from src.utils.metrics import divert_warning_metrics, regression_metrics, skill_score
from src.visualization import plots


def load_run(run_dir: Path):
    meta = json.loads((run_dir / "metadata.json").read_text())
    return meta, {n: MODEL_REGISTRY[n].load(run_dir / "models") for n in meta["models"]}


def predict_delta(name: str, models: dict, meta: dict, X: pd.DataFrame) -> np.ndarray:
    if name == "Ensemble":
        return np.mean([models[m].predict(X) for m in meta["ensemble_members"]], axis=0)
    return models[name].predict(X)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="outputs/latest", help="training run directory")
    ap.add_argument("--csv", nargs="+", required=True, help="one or more raw SCADA CSV exports")
    ap.add_argument("--model", help="model to use (default: best model of the run)")
    args = ap.parse_args()

    run_dir = Path(args.run).resolve()
    meta, models = load_run(run_dir)
    cfg, name = meta["config"], args.model or meta["best_model"]
    h_s, thr, margin = meta["horizon_s"], cfg["target"]["divert_threshold_c"], cfg["target"]["alert_margin_c"]
    print(f"Run {meta['run_id']} · model {name} · horizon {h_s}s")

    df = read_csvs([Path(p) for p in args.csv], cfg["data"]["timestamp_col"])
    data, _ = make_supervised(df, cfg, with_target=False)
    if len(data) == 0:
        raise SystemExit("No production rows with enough history (need feed flow ≥ "
                         f"{cfg['production']['min_flow_lph']} L/h for ≥ 60 s).")
    X = data.X[meta["features"]]
    pred = data.current.values + predict_delta(name, models, meta, X)

    out = pd.DataFrame({
        "timestamp": data.timestamp.values,
        "forecast_for": data.timestamp.values + pd.Timedelta(seconds=h_s),
        "current_temp_c": data.current.values,
        "forecast_temp_c": np.round(pred, 3),
        "divert_alert": pred < thr + margin,
    })

    # Evaluate when the file contains the actual future values
    labelled, _ = make_supervised(df, cfg, with_target=True)
    if len(labelled):
        actual = labelled.y_future.reindex(data.X.index)
        out["actual_temp_c"] = actual.values
        ok = actual.notna().values
        m = regression_metrics(actual[ok], pred[ok], data.current.values[ok])
        pm = regression_metrics(actual[ok], data.current.values[ok])
        dv = divert_warning_metrics(actual[ok], pred[ok], thr, margin)
        print(f"  MAE {m['MAE_C']:.3f} °C · persistence {pm['MAE_C']:.3f} °C · "
              f"skill {100 * skill_score(m['MAE_C'], pm['MAE_C']):+.1f}% · divert events {dv['events']} (recall {dv['recall']:.2f})")
        p_dir = run_dir / "predictions"; p_dir.mkdir(exist_ok=True)
        stem = Path(args.csv[0]).stem + ("_etc" if len(args.csv) > 1 else "")
        fig = plots.plot_test_forecasts(data.timestamp[ok], actual[ok].values,
                                        {"Persistence": data.current.values[ok], name: pred[ok]},
                                        thr, h_s, name, p_dir / f"{stem}_forecast.png")
        print(f"  plot → {fig}")

    p_dir = run_dir / "predictions"; p_dir.mkdir(exist_ok=True)
    stem = Path(args.csv[0]).stem + ("_etc" if len(args.csv) > 1 else "")
    path = p_dir / f"{stem}_predictions.csv"
    out.to_csv(path, index=False)
    print(f"  {len(out):,} forecasts ({int(out.divert_alert.sum())} divert alerts) → {path}")


if __name__ == "__main__":
    main()
