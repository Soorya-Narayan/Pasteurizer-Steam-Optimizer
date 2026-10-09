"""Gradient-boosted tree models (LightGBM, XGBoost) + linear baseline, with live progress."""
from __future__ import annotations

from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

BAR_FMT = "{l_bar}{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}"


# ----------------------------------------------------------------- Ridge
class RidgeModel:
    name = "Ridge"
    file = "ridge.joblib"

    def __init__(self, pipe=None):
        self.pipe = pipe

    @classmethod
    def fit(cls, X: pd.DataFrame, y: pd.Series, alpha: float = 1.0) -> "RidgeModel":
        return cls(make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(X.values, y.values))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipe.predict(X.values)

    def save(self, d: Path) -> None:
        joblib.dump(self.pipe, d / self.file)

    @classmethod
    def load(cls, d: Path) -> "RidgeModel":
        return cls(joblib.load(d / cls.file))


# -------------------------------------------------------------- LightGBM
class LightGBMModel:
    name = "LightGBM"
    file = "lightgbm.txt"

    def __init__(self, booster: lgb.Booster):
        self.booster = booster

    @classmethod
    def fit(cls, Xtr, ytr, Xva, yva, params: dict, monitor=None, seed: int = 42) -> "LightGBMModel":
        params = dict(params)
        rounds = params.pop("num_boost_round")
        patience = params.pop("early_stopping_rounds")
        params.update(metric=["l2", "l1"], verbosity=-1, seed=seed)

        dtr = lgb.Dataset(Xtr, ytr, free_raw_data=False)
        dva = lgb.Dataset(Xva, yva, reference=dtr, free_raw_data=False)
        bar = tqdm(total=rounds, desc="  LightGBM", unit="tree", bar_format=BAR_FMT, ncols=90)
        best = {"val": float("inf"), "it": 0}

        def _progress(env):
            res = {(d, m): v for d, m, v, *_ in env.evaluation_result_list}
            tr, va = res[("train", "l1")], res[("val", "l1")]
            if va < best["val"]:
                best.update(val=va, it=env.iteration + 1)
            bar.update(1)
            bar.set_postfix_str(f"train_mae={tr:.4f} val_mae={va:.4f} best={best['val']:.4f}@{best['it']}")
            if monitor:
                monitor.log(cls.name, env.iteration + 1, tr, va, "boosting round")

        try:
            booster = lgb.train(
                params, dtr, num_boost_round=rounds, valid_sets=[dtr, dva], valid_names=["train", "val"],
                callbacks=[_progress, lgb.early_stopping(patience, first_metric_only=False, verbose=False)],
            )
        finally:
            bar.close()
        print(f"    early-stopped at round {booster.best_iteration} (val MAE {best['val']:.4f} °C)")
        return cls(booster)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.booster.predict(X, num_iteration=self.booster.best_iteration or None)

    def importance(self) -> pd.Series:
        return pd.Series(self.booster.feature_importance("gain"), index=self.booster.feature_name()).sort_values(ascending=False)

    def save(self, d: Path) -> None:
        self.booster.save_model(str(d / self.file), num_iteration=self.booster.best_iteration or None)

    @classmethod
    def load(cls, d: Path) -> "LightGBMModel":
        return cls(lgb.Booster(model_file=str(d / cls.file)))


# --------------------------------------------------------------- XGBoost
class _XGBProgress(xgb.callback.TrainingCallback):
    def __init__(self, total: int, monitor, name: str):
        self.bar = tqdm(total=total, desc="  XGBoost ", unit="tree", bar_format=BAR_FMT, ncols=90)
        self.monitor, self.name = monitor, name
        self.best = (float("inf"), 0)

    def after_iteration(self, model, epoch, evals_log) -> bool:
        tr, va = evals_log["train"]["mae"][-1], evals_log["val"]["mae"][-1]
        if va < self.best[0]:
            self.best = (va, epoch + 1)
        self.bar.update(1)
        self.bar.set_postfix_str(f"train_mae={tr:.4f} val_mae={va:.4f} best={self.best[0]:.4f}@{self.best[1]}")
        if self.monitor:
            self.monitor.log(self.name, epoch + 1, tr, va, "boosting round")
        return False

    def after_training(self, model):
        self.bar.close()
        return model


class XGBoostModel:
    name = "XGBoost"
    file = "xgboost.json"

    def __init__(self, booster: xgb.Booster):
        self.booster = booster

    @classmethod
    def fit(cls, Xtr, ytr, Xva, yva, params: dict, monitor=None, seed: int = 42) -> "XGBoostModel":
        params = dict(params)
        rounds = params.pop("num_boost_round")
        patience = params.pop("early_stopping_rounds")
        params.update(eval_metric=["rmse", "mae"], seed=seed)  # last metric (mae) drives early stopping

        dtr = xgb.DMatrix(Xtr, ytr, feature_names=list(Xtr.columns))
        dva = xgb.DMatrix(Xva, yva, feature_names=list(Xtr.columns))
        cb = _XGBProgress(rounds, monitor, cls.name)
        booster = xgb.train(params, dtr, num_boost_round=rounds, evals=[(dtr, "train"), (dva, "val")],
                            early_stopping_rounds=patience, verbose_eval=False, callbacks=[cb])
        best_it = booster.best_iteration
        print(f"    early-stopped at round {best_it + 1} (val MAE {cb.best[0]:.4f} °C)")
        return cls(booster[: best_it + 1])  # keep only the best trees

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.booster.predict(xgb.DMatrix(X, feature_names=list(X.columns)))

    def importance(self) -> pd.Series:
        return pd.Series(self.booster.get_score(importance_type="gain")).sort_values(ascending=False)

    def save(self, d: Path) -> None:
        self.booster.save_model(str(d / self.file))

    @classmethod
    def load(cls, d: Path) -> "XGBoostModel":
        b = xgb.Booster(); b.load_model(str(d / cls.file))
        return cls(b)
