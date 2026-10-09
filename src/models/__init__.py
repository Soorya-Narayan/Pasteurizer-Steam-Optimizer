"""Model registry: every model exposes fit / predict(X) -> delta / save(dir) / load(dir)."""
# NOTE: import order matters on macOS – LightGBM/XGBoost must be loaded *before*
# torch, otherwise the two bundled OpenMP runtimes clash and LightGBM segfaults.
from src.models.tree_models import LightGBMModel, RidgeModel, XGBoostModel  # isort: skip
from src.models.neural_models import MLPModel  # isort: skip

MODEL_REGISTRY = {cls.name: cls for cls in (RidgeModel, LightGBMModel, XGBoostModel, MLPModel)}
CLI_NAMES = {"ridge": "Ridge", "lightgbm": "LightGBM", "xgboost": "XGBoost", "mlp": "MLP"}

__all__ = ["MODEL_REGISTRY", "CLI_NAMES", "RidgeModel", "LightGBMModel", "XGBoostModel", "MLPModel"]
