"""Residual MLP (PyTorch) for temperature-delta regression, with epoch/batch progress."""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm.auto import tqdm


class _Block(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim * 2), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(dim * 2, dim))

    def forward(self, x):
        return x + self.net(x)


class ResidualMLP(nn.Module):
    def __init__(self, n_in: int, hidden: int = 256, n_blocks: int = 3, dropout: float = 0.1):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(n_in, hidden), nn.GELU())
        self.blocks = nn.Sequential(*[_Block(hidden, dropout) for _ in range(n_blocks)])
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.head(self.blocks(self.inp(x))).squeeze(-1)


class MLPModel:
    name = "MLP"
    file = "mlp.pt"

    def __init__(self, net: ResidualMLP, arch: dict, x_mean, x_std, y_mean: float, y_std: float, features: list[str]):
        self.net, self.arch = net.eval(), arch
        self.x_mean, self.x_std = np.asarray(x_mean, np.float32), np.asarray(x_std, np.float32)
        self.y_mean, self.y_std, self.features = float(y_mean), float(y_std), features

    # ------------------------------------------------------------- training
    @classmethod
    def fit(cls, Xtr: pd.DataFrame, ytr: pd.Series, Xva: pd.DataFrame, yva: pd.Series,
            params: dict, monitor=None, seed: int = 42) -> "MLPModel":
        torch.manual_seed(seed); np.random.seed(seed)
        torch.set_num_threads(1)
        device = torch.device("cpu")  # small tabular net: CPU is faster than MPS here

        x_mean = Xtr.values.mean(0); x_std = Xtr.values.std(0); x_std[x_std < 1e-6] = 1.0
        y_mean, y_std = float(ytr.mean()), float(ytr.std() or 1.0)
        norm = lambda X: torch.tensor((X.values - x_mean) / x_std, dtype=torch.float32)
        xt, xv = norm(Xtr), norm(Xva)
        yt = torch.tensor((ytr.values - y_mean) / y_std, dtype=torch.float32)
        yv_c = yva.values.astype(np.float32)

        arch = dict(n_in=Xtr.shape[1], hidden=params["hidden"], n_blocks=params["n_blocks"], dropout=params["dropout"])
        net = ResidualMLP(**arch).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=4)
        loss_fn = nn.HuberLoss(delta=params["huber_delta"])
        bs, n = params["batch_size"], len(xt)
        n_batches = (n + bs - 1) // bs

        best = {"val": float("inf"), "epoch": 0, "state": None}
        wait = 0
        epochs = tqdm(range(1, params["epochs"] + 1), desc="  MLP     ", unit="epoch", ncols=90,
                      bar_format="{l_bar}{bar:25}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}")
        for ep in epochs:
            net.train()
            perm = torch.randperm(n)
            abs_err_sum = 0.0
            for b in range(n_batches):
                idx = perm[b * bs:(b + 1) * bs]
                pred = net(xt[idx])
                loss = loss_fn(pred, yt[idx])
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                abs_err_sum += float((pred.detach() - yt[idx]).abs().sum()) * y_std
            train_mae = abs_err_sum / n

            net.eval()
            with torch.no_grad():
                val_pred = net(xv).numpy() * y_std + y_mean
            val_mae = float(np.mean(np.abs(val_pred - yv_c)))
            sched.step(val_mae)

            if val_mae < best["val"] - 1e-5:
                best.update(val=val_mae, epoch=ep, state=copy.deepcopy(net.state_dict()))
                wait = 0
            else:
                wait += 1
            lr = opt.param_groups[0]["lr"]
            epochs.set_postfix_str(f"tr_mae={train_mae:.4f} va_mae={val_mae:.4f} best={best['val']:.4f}@{best['epoch']}")
            if monitor:
                monitor.log(cls.name, ep, train_mae, val_mae, "epoch")
            if wait >= params["patience"]:
                epochs.close()
                print(f"    early-stopped at epoch {ep}; restoring best epoch {best['epoch']} (val MAE {best['val']:.4f} °C)")
                break
        else:
            epochs.close()

        if best["state"] is not None:
            net.load_state_dict(best["state"])
        return cls(net, arch, x_mean, x_std, y_mean, y_std, list(Xtr.columns))

    # ------------------------------------------------------------ inference
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        x = torch.tensor((X[self.features].values - self.x_mean) / self.x_std, dtype=torch.float32)
        with torch.no_grad():
            return self.net(x).numpy() * self.y_std + self.y_mean

    def save(self, d: Path) -> None:
        torch.save({"state_dict": self.net.state_dict(), "arch": self.arch, "x_mean": self.x_mean,
                    "x_std": self.x_std, "y_mean": self.y_mean, "y_std": self.y_std,
                    "features": self.features}, d / self.file)

    @classmethod
    def load(cls, d: Path) -> "MLPModel":
        ck = torch.load(d / cls.file, map_location="cpu", weights_only=False)
        net = ResidualMLP(**ck["arch"]); net.load_state_dict(ck["state_dict"])
        return cls(net, ck["arch"], ck["x_mean"], ck["x_std"], ck["y_mean"], ck["y_std"], ck["features"])
