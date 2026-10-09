"""Segment-aware feature engineering for holding-tube temperature forecasting.

The plant only produces intermittently, so the time series is split into
*production segments* (continuous feed flow, no logging gaps).  Every lag,
difference, rolling statistic and the forecast target are computed strictly
inside a segment, so nothing leaks across start-ups, shutdowns or CIP.

Target: the *change* in holding-tube temperature over the horizon
(``T[t+h] - T[t]``).  Predicting the delta keeps the models focused on the
thermal dynamics instead of re-learning the current temperature; the absolute
forecast is simply ``T[t] + delta``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SupervisedData:
    X: pd.DataFrame
    current: pd.Series            # T[t]
    timestamp: pd.Series
    time_in_segment_s: pd.Series
    segment_id: pd.Series
    y_delta: pd.Series | None = None   # T[t+h] - T[t]
    y_future: pd.Series | None = None  # T[t+h]

    @property
    def feature_names(self) -> list[str]:
        return list(self.X.columns)

    def subset(self, mask) -> "SupervisedData":
        mask = np.asarray(mask)
        pick = lambda s: None if s is None else s[mask]
        return SupervisedData(
            X=self.X[mask], current=self.current[mask], timestamp=self.timestamp[mask],
            time_in_segment_s=self.time_in_segment_s[mask], segment_id=self.segment_id[mask],
            y_delta=pick(self.y_delta), y_future=pick(self.y_future),
        )

    def __len__(self) -> int:
        return len(self.X)


def horizon_steps(cfg: dict) -> int:
    return max(1, int(round(cfg["target"]["horizon_s"] / cfg["data"]["sample_period_s"])))


def mark_segments(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Add is_production, segment_id and time_in_segment_s columns."""
    ts_col = cfg["data"]["timestamp_col"]
    ts = df[ts_col]
    prod = df[cfg["production"]["flow_col"]].fillna(0) >= cfg["production"]["min_flow_lph"]
    gap = ts.diff().dt.total_seconds().fillna(0) > cfg["data"]["max_gap_s"]
    new_seg = (prod != prod.shift(fill_value=False)) | gap
    seg = new_seg.cumsum().astype("int64")
    out = df.assign(is_production=prod, segment_id=seg)
    out["time_in_segment_s"] = (ts - ts.groupby(seg).transform("first")).dt.total_seconds()
    return out


def build_features(seg_df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    f = cfg["features"]
    g = seg_df.groupby("segment_id", sort=False)
    feats: dict[str, pd.Series] = {c: seg_df[c].astype(float) for c in f["sensors"]}

    for c in f["lag_columns"]:
        for k in f["lags"]:
            feats[f"{c}_lag{k}"] = g[c].shift(k)
    for c in f["diff_columns"]:
        for k in f["diff_steps"]:
            feats[f"{c}_diff{k}"] = seg_df[c] - g[c].shift(k)
    for c in f["rolling_columns"]:
        for w in f["rolling_windows"]:
            r = g[c].rolling(w, min_periods=w)
            feats[f"{c}_rmean{w}"] = r.mean().reset_index(level=0, drop=True)
            feats[f"{c}_rstd{w}"] = r.std().reset_index(level=0, drop=True)

    # Physics-informed features
    t05, t06 = seg_df["tt_05_holding_in_c"], seg_df["tt_06_holding_out1_c"]
    flow = seg_df[cfg["production"]["flow_col"]]
    feats["holding_tube_loss_c"] = t05 - t06
    feats["margin_to_divert_c"] = t06 - cfg["target"]["divert_threshold_c"]
    feats["steam_per_kflow"] = seg_df["steam_cv"] / (flow / 1000.0).clip(lower=0.5)
    feats["heating_rise_c"] = t05 - seg_df["tt_03_regen_2_in_c"]
    feats["log_time_in_run_s"] = np.log1p(seg_df["time_in_segment_s"])

    return pd.DataFrame(feats, index=seg_df.index)


def make_supervised(df: pd.DataFrame, cfg: dict, with_target: bool = True) -> tuple[SupervisedData, pd.DataFrame]:
    """Return (supervised dataset, segmented frame).  Only valid production rows are kept."""
    seg_df = mark_segments(df, cfg)
    X = build_features(seg_df, cfg)
    target = cfg["target"]["column"]
    mask = seg_df["is_production"] & X.notna().all(axis=1)

    y_future = y_delta = None
    if with_target:
        future = seg_df.groupby("segment_id", sort=False)[target].shift(-horizon_steps(cfg))
        mask &= future.notna()
        y_future = future[mask].astype(float)
        y_delta = (future - seg_df[target])[mask].astype(float)

    data = SupervisedData(
        X=X[mask].astype("float32"),
        current=seg_df.loc[mask, target].astype(float),
        timestamp=seg_df.loc[mask, cfg["data"]["timestamp_col"]],
        time_in_segment_s=seg_df.loc[mask, "time_in_segment_s"],
        segment_id=seg_df.loc[mask, "segment_id"],
        y_delta=y_delta,
        y_future=y_future,
    )
    return data, seg_df


def chronological_split(data: SupervisedData, cfg: dict) -> dict[str, SupervisedData]:
    n = len(data)
    n_tr = int(n * cfg["split"]["train_frac"])
    n_va = int(n * cfg["split"]["val_frac"])
    idx = np.arange(n)
    return {
        "train": data.subset(idx < n_tr),
        "val": data.subset((idx >= n_tr) & (idx < n_tr + n_va)),
        "test": data.subset(idx >= n_tr + n_va),
    }
