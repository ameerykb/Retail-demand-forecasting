"""PyTorch model for demand forecasting: an MLP with embeddings.

Idea: categorical inputs (which store, weekday, month, store type, assortment)
become small learned vectors ("embeddings"); numeric inputs are log-scaled and
standardized; everything is concatenated and passed through a small
feed-forward network that predicts log(1 + sales).

The training loop is written by hand (no Trainer / Lightning) on purpose, so
every step is visible: batch -> forward -> loss -> backward -> optimizer step.
"""
import copy
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

MLP_DEFAULTS = dict(val_days=42, epochs=15, batch_size=2048, lr=2e-3,
                    weight_decay=1e-5, patience=3, hidden=(256, 128),
                    dropout=0.1, seed=42)


class Preprocessor:
    """Turns the feature table into (categorical ids, numeric matrix).
    Everything it learns (medians, means, stds) comes from TRAINING rows only."""
    CAT_COLS = ["store_cat", "dow", "month", "store_type", "assortment"]
    BIN_COLS = ["promo", "state_holiday", "school_holiday", "promo2"]

    def __init__(self, feature_cols):
        self.cat_cols = [c for c in self.CAT_COLS if c in feature_cols]
        self.bin_cols = [c for c in self.BIN_COLS if c in feature_cols]
        self.log_cols = [c for c in feature_cols if c.startswith(("lag_", "roll_"))]
        if "competition_distance" in feature_cols:
            self.log_cols.append("competition_distance")
        self.lag_col = next(c for c in feature_cols if c.startswith("lag_"))
        self.cardinality = {}

    def _numeric(self, df):
        out = pd.DataFrame(index=df.index)
        for c in self.bin_cols:
            out[c] = df[c].astype(float)
        for c in self.log_cols:                       # sales are skewed -> log scale
            out[c] = np.log1p(df[c].clip(lower=0))
        out["doy_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365.25)  # day 365 is next to day 1
        out["doy_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365.25)
        out["day_scaled"] = df["day"] / 31.0
        out["lag_missing"] = df[self.lag_col].isna().astype(float)
        return out

    def _cats(self, df):
        cols = []
        for c in self.cat_cols:
            if c == "store_cat":
                v = df[c].cat.codes.to_numpy()
            elif c == "month":
                v = df[c].to_numpy() - 1
            else:
                v = df[c].to_numpy()
            cols.append(np.clip(v.astype(np.int64), 0, self.cardinality[c] - 1))
        return np.stack(cols, axis=1)

    def fit(self, train):
        for c in self.cat_cols:
            if c == "store_cat":
                self.cardinality[c] = len(train[c].cat.categories)
            elif c == "dow":
                self.cardinality[c] = 7
            elif c == "month":
                self.cardinality[c] = 12
            else:
                self.cardinality[c] = int(train[c].max()) + 1
        num = self._numeric(train)
        self.median = num.median().fillna(0)
        num = num.fillna(self.median)
        self.mean = num.mean()
        self.std = num.std().replace(0, 1).fillna(1) + 1e-6
        return self

    def transform(self, df):
        num = (self._numeric(df).fillna(self.median) - self.mean) / self.std
        cats = torch.from_numpy(self._cats(df))
        return cats, torch.tensor(num.to_numpy(dtype=np.float32))


class TabularMLP(nn.Module):
    def __init__(self, cardinality: dict, n_num: int, hidden=(256, 128), dropout=0.1):
        super().__init__()
        self.embs = nn.ModuleList()
        emb_total = 0
        for card in cardinality.values():
            dim = min(16, max(2, card // 2))          # bigger vocab -> bigger vector
            self.embs.append(nn.Embedding(card, dim))
            emb_total += dim
        layers, size = [], emb_total + n_num
        for h in hidden:
            layers += [nn.Linear(size, h), nn.ReLU(), nn.Dropout(dropout)]
            size = h
        layers.append(nn.Linear(size, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, cats, nums):
        parts = [emb(cats[:, i]) for i, emb in enumerate(self.embs)]
        return self.mlp(torch.cat(parts + [nums], dim=1)).squeeze(1)


class MLPForecaster:
    def __init__(self, model, pre, y_mean, y_std, device, history):
        self.model, self.pre, self.y_mean, self.y_std = model, pre, y_mean, y_std
        self.device, self.history = device, history

    @torch.no_grad()
    def predict(self, df, batch_size=8192):
        self.model.eval()
        cats, nums = self.pre.transform(df)
        out = []
        for i in range(0, len(df), batch_size):
            out.append(self.model(cats[i:i + batch_size].to(self.device),
                                  nums[i:i + batch_size].to(self.device)).cpu())
        z = torch.cat(out).numpy()
        pred = np.expm1(z * self.y_std + self.y_mean).clip(min=0)
        return np.where(df["open"].to_numpy() == 1, pred, 0.0)   # closed days -> 0


def fit_mlp(train, feature_cols, **kwargs) -> MLPForecaster:
    cfg = {**MLP_DEFAULTS, **kwargs}
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    t = train[train["open"] == 1]
    # Time-based validation: the most recent days before the cutoff decide when to stop.
    split = t["date"].max() - pd.Timedelta(days=cfg["val_days"])
    tr, va = t[t["date"] <= split], t[t["date"] > split]

    pre = Preprocessor(feature_cols).fit(tr)
    Xc, Xn = (x.to(device) for x in pre.transform(tr))
    Vc, Vn = (x.to(device) for x in pre.transform(va))
    y = np.log1p(tr["sales"].to_numpy())
    y_mean, y_std = float(y.mean()), float(y.std())
    Y = torch.tensor((y - y_mean) / y_std, dtype=torch.float32, device=device)
    VY = torch.tensor((np.log1p(va["sales"].to_numpy()) - y_mean) / y_std,
                      dtype=torch.float32, device=device)

    model = TabularMLP(pre.cardinality, Xn.shape[1], cfg["hidden"], cfg["dropout"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    gen = torch.Generator().manual_seed(cfg["seed"])

    n, history = len(Y), {"train_loss": [], "val_loss": []}
    best_val, best_state, bad = float("inf"), None, 0
    for _epoch in range(cfg["epochs"]):
        model.train()
        perm = torch.randperm(n, generator=gen).to(device)     # shuffle every epoch
        total = torch.zeros((), device=device)
        for i in range(0, n, cfg["batch_size"]):
            idx = perm[i:i + cfg["batch_size"]]
            opt.zero_grad()                                    # 1. clear old gradients
            loss = F.mse_loss(model(Xc[idx], Xn[idx]), Y[idx]) # 2. forward pass + loss
            loss.backward()                                    # 3. backpropagation
            opt.step()                                         # 4. update weights
            total += loss.detach() * len(idx)
        model.eval()
        with torch.no_grad():
            val_loss = F.mse_loss(model(Vc, Vn), VY).item()
        history["train_loss"].append(total.item() / n)
        history["val_loss"].append(val_loss)
        if val_loss < best_val - 1e-5:                         # early stopping + checkpoint
            best_val, best_state, bad = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
    model.load_state_dict(best_state)
    return MLPForecaster(model, pre, y_mean, y_std, device, history)


def make_mlp_learner(histories: list, **kwargs):
    """Wraps fit + predict so the backtest can treat it like any other model."""
    def learner(train, test, feature_cols):
        fc = fit_mlp(train, feature_cols, **kwargs)
        histories.append(fc.history)
        return fc.predict(test)
    return learner
