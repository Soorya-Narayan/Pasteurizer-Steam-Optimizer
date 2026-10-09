"""Evaluation metrics: regression accuracy, skill vs persistence, divert early-warning."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import r2_score


def regression_metrics(y_true, y_pred, current=None) -> dict:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    err = y_pred - y_true
    out = {
        "MAE_C": float(np.mean(np.abs(err))),
        "RMSE_C": float(np.sqrt(np.mean(err ** 2))),
        "P95_AbsErr_C": float(np.percentile(np.abs(err), 95)),
        "MaxAbsErr_C": float(np.max(np.abs(err))),
        "R2": float(r2_score(y_true, y_pred)),
    }
    if current is not None:
        # R² on the temperature *change* – the honest score (persistence gets 0 here)
        current = np.asarray(current, float)
        out["R2_delta"] = float(r2_score(y_true - current, y_pred - current))
    return out


def skill_score(model_mae: float, baseline_mae: float) -> float:
    """1 - MAE_model / MAE_persistence  (>0 means better than 'no change' forecast)."""
    return float(1.0 - model_mae / baseline_mae) if baseline_mae > 0 else float("nan")


def divert_warning_metrics(y_true, y_pred, threshold: float, margin: float = 0.0) -> dict:
    """Treat the forecast as an early warning that T[t+h] will drop below the divert threshold."""
    actual = np.asarray(y_true) < threshold
    alert = np.asarray(y_pred) < threshold + margin
    tp = int(np.sum(actual & alert)); fp = int(np.sum(~actual & alert))
    fn = int(np.sum(actual & ~alert)); tn = int(np.sum(~actual & ~alert))
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if tp and (precision + recall) else 0.0
    return {"events": int(actual.sum()), "TP": tp, "FP": fp, "FN_missed": fn, "TN": tn,
            "precision": precision, "recall": recall, "F1": f1}
